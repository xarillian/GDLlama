# Chorus

[![CI](https://img.shields.io/github/actions/workflow/status/xarillian/chorus-llm/ci.yml?branch=master&label=CI)](https://github.com/xarillian/chorus-llm/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Godot 4.4+](https://img.shields.io/badge/Godot-4.4%2B-478CBF?logo=godotengine&logoColor=white)](https://godotengine.org/)

**Chorus** is a high-performance embeddable LLM runtime for native applications, simulations, and games.

Chorus is built for interactive workloads where one model serves many independently scheduled agents. On an RTX 4070, the benchmark configuration below produces 114 tok/s for one active response and 1,421 aggregate tok/s across 40 concurrent responses, using 5.3 GiB of VRAM at 40-way concurrency.

It runs local inference directly inside your process with continuous batching, priority scheduling, persistent conversation state, and a native C++ API and stable C ABI. No CUDA, Python runtime, cloud service, or separate inference server is required.

The current local provider is built on `llama.cpp`, with Vulkan on Linux and Windows and Metal on Apple Silicon. Chorus owns the application-level scheduler, request preparation, sampling, conversation lifecycle, and batch layout around that provider. It is embeddable by default, but its C ABI can also be hosted out-of-process when crash isolation or fault containment matters. Godot 4.4+ is the first supported engine integration.

Consumer GPUs are and will always be the first-class target. This is partly a technical choice and partly a philosophical one: a spirit for every machine, and local inference for all.

## Features

Run text generation and embeddings directly inside your application using a model you supply. Chorus manages model loading, concurrent requests, and conversation histories.

- **Local inference.** Run GGUF models directly through the local provider, with Vulkan acceleration on Linux and Windows and Metal on Apple Silicon. No cloud API, account, Python runtime, or inference daemon is required.
- **Concurrent agents.** Serve multiple active requests from one loaded model and one shared multi-sequence context, advancing active agents together in shared GPU steps.
- **Priority scheduling.** Higher-priority work is admitted first, and each inference step runs the highest runnable priority tier. Submission order is preserved among equal-priority requests.
- **Continuous batching.** Admit and retire sequences as capacity changes, and batch active work within a shared token budget. Long prompts are processed in bounded chunks alongside generation, so existing replies keep streaming while newly admitted requests continue making progress.
- **Conversation control.** Keep independent histories for your characters or agents. Import, export, edit, and regenerate messages, or inject context for one turn.
- **Structured output.** Constrain replies with JSON Schema or a GBNF grammar for results your application can parse.
- **Per-request generation settings.** Choose response length, sampling settings, and stop sequences for each request, or inherit shared defaults.
- **Explicit request outcomes.** Cancel queued or active requests and receive one terminal success or error for every accepted request. Failed or cancelled chat turns roll back their pending history changes.
- **Asynchronous model loading.** Show loading progress and request cancellation while a model loads. Replacing a model releases the old one before loading its successor.
- **Text embeddings.** Generate normalized vectors for your own similarity search or retrieval logic.
- **Embeddable by default.** Link Chorus directly into a native application through its C++ API or a stable C ABI, without running a separate inference service. The same ABI can be hosted out-of-process when fault isolation is preferable.
- **Godot and native integration.** Use typed request resources, signals, and Project Settings in Godot, or integrate through the host-independent C++ runtime and C ABI.

Supported features depend on the model; practical concurrency depends on the model and available hardware. See the [full feature list and planned work](docs/FEATURES.md).

## Performance

These tests were performed using `google/gemma-4-E4B-it-qat-q4_0-gguf` on an RTX 4070 (12 GB) through `llama.cpp`'s Vulkan backend, on a desktop with a browser and chat apps open. Each request is scoped to 64 output tokens (`max_tokens = 64`, `ignore_eos = true`). These are end-to-end Chorus measurements.

Performance results differ by hardware, model used, and settings.

- **First turn:** the whole history is processed, as after importing it or loading the model.
- **Next turn:** the previous turn's KV cache is reused, so only the new message is processed. This holds while each conversation keeps a slot (up to `max_concurrent_requests` conversations) and its earlier history is unedited.

| Context | Concurrent requests | All replies complete | First token | Aggregate output (tok/s) | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short prompt | 1 | 0.56 s | 22 ms | 114 | 2.9 GiB |
| | 4 | 0.70 s | 47 ms | 363 | 3.2 GiB |
| | 16 | 1.40 s | 113 ms | 734 | 3.9 GiB |
| | 40 | 1.80 s | 242 ms | 1,421 | 5.3 GiB |
| ~3,900-token history, first turn | 1 | 1.33 s | 722 ms | 48 | 3.3 GiB |
| | 4 | 3.72 s | 1.90 s | 69 | 3.6 GiB |
| | 16 | 13.33 s | 6.55 s | 77 | 4.8 GiB |
| | 40 | 32.82 s | 16.28 s | 78 | 7.3 GiB |
| ~3,900-token history, next turn | 1 | 0.73 s | 143 ms | 88 | 3.3 GiB |
| | 4 | 0.96 s | 197 ms | 267 | 3.6 GiB |
| | 16 | 2.30 s | 508 ms | 446 | 4.8 GiB |
| | 40 | 3.68 s | 789 ms | 695 | 7.3 GiB |

These figures will be updated occasionally and will always use a SOTA local model. With many agents at once, a smaller model will answer faster.

## Setup

Chorus builds from source with SCons, which also configures and compiles its pinned `llama.cpp` through CMake. There are three applicable build targets:
- Vulkan on Linux/Windows
- Metal on Apple Silicon
- CPU anywhere

### Requirements

- Git
- Python 3 with SCons
- CMake 3.24 or newer
- A C++20 toolchain: GCC or Clang on Linux, MSVC on Windows, or Xcode on macOS
- [Just](https://github.com/casey/just), for the test and Godot commands below

| Target | Platforms | Also requires |
| --- | --- | --- |
| Vulkan | Linux, Windows | The Vulkan SDK, including `glslc` and SPIR-V headers, and a Vulkan GPU driver |
| Metal | macOS on Apple silicon | Xcode |
| CPU | Linux, Windows, macOS | Nothing further |

On Ubuntu 24.04, `sudo apt install libvulkan-dev spirv-headers glslc` provides what Vulkan needs. Ubuntu 22.04 does not package `glslc`, so install the [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home) there.

On Windows, install the LunarG SDK and make sure `VULKAN_SDK` points at it; the build links `vulkan-1.lib` from that directory.

Chorus does not build for Intel Macs.

### Clone

```sh
git clone --recursive https://github.com/xarillian/chorus-llm.git
cd chorus-llm
```

If you cloned without `--recursive`, fetch the dependencies with `git submodule update --init --recursive`.

### Building with Vulkan

For Linux and Windows:

```sh
just build
```

This is the default `just` build and runs `scons use_vulkan=yes`.

### Building with Metal

For macOS on Apple silicon:

```sh
scons use_metal=yes
```

### Building for CPU

```sh
just build --cpu
```

### Build output

Builds are written to `bin/`:

- `libgodot_chorus` is the Godot extension.
- `libchorus_c` is the C ABI for native applications. Its header is `include/chorus_c/chorus_c.h`.

The default target is a debug build. For release, run 
- `just release` for Vulkan,
- `just release --cpu` for CPU, or
- `scons target=template_release use_metal=yes` for Metal.

### Testing

Run the native suite without models:

```sh
just check --quick
```

This builds a CPU test binary whatever your backend, so the first run compiles a CPU llama.cpp build.

Model tests use about 1.5 GB of small GGUF fixtures. Review the [fixture manifest and model terms](tests/model-fixtures.json), then run:

```sh
just download-fixtures
just check-model
```

On a Vulkan machine, `just check-gpu` builds a Vulkan test binary and runs the GPU model tests. Metal has no separate test build; on macOS, the CPU suite covers the runtime. I currently do not recommend running the full test suite on macOS.

### Godot

Chorus is compatible with all stable Godot 4.4+ releases. To build and stage the addon for Godot on Linux or Windows, run:

```sh
just godot
```

For Metal or CPU, build first as above, then stage the addon without rebuilding:

```sh
python tools/stage_godot.py
```

The staged addon is written to `plugin/addons/chorus`. Copy it into your project's `addons` directory, then enable **Chorus LLM** under _Project > Project Settings > Plugins_.

You can then get generation running quickly with:

```gdscript
func _ready() -> void:
    chorus.model_loaded.connect(func(_id, _model):
        chorus.generate(ChorusRequest.chat(&"guard", "Greet a traveler.")))

    chorus.generation_complete.connect(func(_id, _session, _message, content, _reasoning):
        print(content))

    chorus.load_model()
```

See the [Godot guide](docs/GODOT.md) for installation and getting to your first response.

## Architecture

See [Architecture](docs/ARCHITECTURE.md) for the dependency model, service contracts, provider boundaries, and threading invariants.

## License

Chorus is licensed under the [MIT License](LICENSE).

Third-party components retain their own licenses. See [Third-party notices](THIRD_PARTY_NOTICES.md). Model weights are not included and have separate terms.

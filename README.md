# Chorus
**Chorus** is a local LLM inference runtime for games initially delivered as a Godot 4.4+ GDExtension backed by `llama.cpp`.

The project’s thesis is that one consumer GPU should support real-time inference for many concurrent game agents. Its core is independent of any host or inference provider, so `Godot` and `llama.cpp` are the first integrations rather than permanent boundaries.      

Chorus is under active development; it is currently the hobby project of one dev. The native runtime is available through a C ABI and a Godot adapter, with `llama.cpp` as the inference backend. 

## Features
Run text generation and embeddings inside your game or application, using a model you supply. Chorus manages model loading, concurrent requests, and conversation histories; your application decides what to do with the results.

- **Local inference.** Run GGUF models on CPU or GPU through `llama.cpp`, without a cloud API, account, or separate inference server.
- **Concurrent agents.** Serve multiple active requests from one loaded model through shared inference batches. Set request priorities, and process long prompts in chunks alongside ongoing generation.
- **Conversation control.** Keep independent histories for your characters or agents. Import, export, edit, and regenerate messages, or inject context for one turn. Prompt fitting reports omitted messages without deleting stored history.
- **Structured output.** Constrain replies with JSON Schema or a GBNF grammar for results your application can parse. Invalid or unsupported constraints produce errors rather than unconstrained replies.
- **Per-request generation settings.** Choose response length, sampling settings, and stop sequences for each request, or inherit shared defaults. Stream visible text and reasoning through separate channels where the model supports them.
- **Explicit request outcomes.** Cancel queued or active requests and receive one terminal success or error for every accepted request. Failed or cancelled chat turns roll back their pending history changes.
- **Asynchronous model loading.** Show loading progress and request cancellation while a model loads. Replacing a model releases the old one before loading its successor.
- **Text embeddings.** Generate normalized vectors for your own similarity search or retrieval logic.
- **Godot and native integration.** Use typed request resources, signals, and Project Settings in Godot, or integrate through the host-independent C++ runtime and C ABI.

Supported features depend on the model; practical concurrency depends on the model and available hardware. See the [full feature list and planned work](docs/FEATURES.md).

## Performance



Results vary with the model, hardware and prompts.

### Demo
<brief clip - possibly doesn't require a header?>

## Setup
The Chorus build requires Git, Python 3 with SCons, CMake 3.24 or newer, a C++20 compiler toolchain, and optionally [Just](https://github.com/casey/just). The default Linux and Windows build also requires the Vulkan SDK, including `glslc` and SPIR-V headers.

From a fresh clone, run:

```sh
git submodule update --init --recursive
just build
just check --quick
```

On macOS, install Xcode and use the Metal build in place of `just build`:

```sh
scons use_metal=yes
```

Build artifacts are written to `bin/`.

For model tests, review the [fixture manifest and model terms](tests/model-fixtures.json), then run `just download-fixtures` and `just check-model`.

### Godot

Chorus is compatible with all stable Godot 4.4+ releases. To build and stage the addon for Godot on Linux or Windows, run:

```sh
just godot
```

For macOS, after a Metal build, stage the addon without rebuilding:

```sh
python tools/stage_godot.py
```

The staged addon is written to `plugin/addons/chorus`.

See the [Godot guide](docs/GODOT.md) for installation and your first response.

## Architecture
See [Architecture](docs/ARCHITECTURE.md) for the dependency model, service contracts, provider boundaries, and threading invariants.

## License
Chorus is licensed under the [MIT License](LICENSE).

Third-party components retain their own licenses. See [Third-party notices](THIRD_PARTY_NOTICES.md). Model weights are not included and have separate terms.

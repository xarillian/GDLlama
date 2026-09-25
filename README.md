# Chorus
**Chorus** is a local LLM inference runtime for games initially delivered as a Godot 4.4+ GDExtension backed by `llama.cpp`.

The project’s thesis is that one consumer GPU should support real-time inference for many concurrent game agents. Its core is independent of any host or inference provider, so `Godot` and `llama.cpp` are the first integrations rather than permanent boundaries.      

Chorus is under active development; it is currently the hobby project of one dev. The native runtime, `llama.cpp` provider and a Godot adapter are implemented. 

## Features

## Setup
Chorus requires Git, Python 3 with SCons, CMake 3.14 or newer, a C++20 compiler toolchain, and [Just](https://github.com/casey/just). The default Linux and Windows build also requires the Vulkan SDK.

After cloning Chorus, run:

```sh
git submodule update --init --recursive
just build
just check --quick
```

On macOS, install Xcode and use the Metal build in place of `just build`:

```sh
scons use_metal=yes
```

Build artifacts are written to `bin/`. The full `just check` suite also runs real-model tests and expects `tests/models/gemma-3-270m-it-F16.gguf`.

The pinned llama.cpp source is kept clean. `SConstruct` reconstructs it in `bin/vendor/llama.cpp/` and applies `patches/llama-resource-cleanup.patch` before compiling either backend or the provider. The patch closes the non-mmap asynchronous upload cleanup gap on cancellation, partial setup, and exceptions; it also releases a partially constructed compatibility batch when conversion allocation fails and preserves the correct GGML revision when CMake runs beneath the Chorus checkout. The source tree and CMake build identity include the vendor revision and patch/helper bytes. A patch conflict or dirty tracked submodule fails the build rather than using unpatched code. `.gitattributes` keeps the patch LF even in CRLF-configured checkouts, matching Git's pristine vendor archive. For an upstream revision change, check out the public revision in `third-party/llama.cpp`, rebase the three-file patch against it, and rerun both CPU and Vulkan build/tests plus the upload probe. Do not commit a local-only vendor revision. The weekly latest-llama canary intentionally fails if the patch drifts. On a Linux Vulkan machine with the existing GGUF fixture, `scons test use_vulkan=yes -j2` followed by `python tools/run_vendor_upload_probe.py` checks repeated upload cancellation, final cancellation, decode, bounded partial-setup failures, and conversion allocation faults against the exact built archive. The probe uses non-mmap loading only within the test; Chorus production defaults are unchanged.

### Godot

To build and stage the addon for Godot 4.4 or newer on Linux or Windows, run:

```sh
just godot
```

After a Metal build on macOS, stage the addon without rebuilding:

```sh
python tools/stage_godot.py
```

The staged addon is written to `plugin/addons/chorus`. To use it in a Godot project:

1. Copy that directory to `<project>/addons/chorus`.
2. Open the project and enable **Chorus LLM** under **Project Settings > Plugins**.
3. Add a `GodotChorus` node to a scene.
4. Set its `model_path` to a compatible GGUF model. 
5. Connect load signals, call `load_model()`, and check its typed `ChorusLoadResult.accepted` before waiting for the identified terminal.

```gdscript
var terminals := {}
chorus.model_loaded.connect(func(id, model_id): terminals[id] = true)
chorus.model_load_failed.connect(func(id, model_id, error_code, message):
    terminals[id] = false
    if error_code == GodotChorus.ERR_CANCELLED:
        print("Model load %d cancelled" % id)
    else:
        push_error("Model load %d failed (%d): %s" % [id, error_code, message]))
chorus.model_load_progress.connect(func(id, model_id, phase, has_fraction, fraction):
    if has_fraction:
        print("Load %d phase %d: %.1f%%" % [id, phase, fraction * 100.0])
    else:
        print("Load %d phase %d: progress unknown" % [id, phase]))
var load := chorus.load_model()
if not load.accepted:
    push_error("Load rejected: %s" % load.message)
else:
    # To abandon this attempt, call chorus.cancel_load(load.load_id) and still await its terminal.
    var deadline := Time.get_ticks_msec() + 30000
    while not terminals.has(load.load_id) and Time.get_ticks_msec() < deadline:
        await get_tree().process_frame
    if terminals.get(load.load_id, false) and chorus.is_loaded():
        var result := chorus.generate(ChorusRequest.stateless("Hello"))
        assert(result.accepted)
    elif not terminals.has(load.load_id):
        push_error("Timed out waiting for load %d" % load.load_id)
```

A weight-loading fraction of 100% is not readiness. A success signal records publication, but an earlier handler in the same polled batch can stop or replace the engine; correlate the ID and check current readiness when acting. Changing node properties while loaded or loading affects only the next accepted load. Cancellation admission does not wait for cleanup; explicit `stop_all()` and node destruction do. `model_path` accepts absolute paths and Godot `res://` or `user://` paths to loose GGUF files. A model packed inside a PCK must currently be extracted to the filesystem.

## Architecture
See [Architecture](docs/ARCHITECTURE.md) for the dependency model, service
contracts, provider boundaries, and threading invariants.

## License
Licensing terms are still being determined.

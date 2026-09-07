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
5. Call `load_model()` and inspect `last_load_error` and `last_load_error_message` if it returns `false`.

## Architecture
See [Architecture](docs/ARCHITECTURE.md) for the dependency model, service
contracts, provider boundaries, and threading invariants.

## License
Licensing terms are still being determined.

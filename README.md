# Chorus
**Chorus** is a local LLM inference runtime for games initially delivered as a Godot 4.4+ GDExtension backed by `llama.cpp`.

The project’s thesis is that one consumer GPU should support real-time inference for many concurrent game agents. Its core is independent of any host or inference provider, so `Godot` and `llama.cpp` are the first integrations rather than permanent boundaries.      

Chorus is under active development; it is currently the hobby project of one dev. The native runtime is available through a C ABI and a Godot adapter, with `llama.cpp` as the inference backend. 

## Features
<how is it useful?>

### Demo
<brief clip - possibly doesn't require a header?>

## Setup
The Chorus build requires Git, Python 3 with SCons, CMake 3.14 or newer, a C++20 compiler toolchain, and (optionally, but more simply) [Just](https://github.com/casey/just). The default Linux and Windows build also requires the Vulkan SDK.

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

Build artifacts are written to `bin/`. 

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

## Usage
<Link to Godot integration guidance and a minimal C example that each demonstrate working inference.>

## Architecture
See [Architecture](docs/ARCHITECTURE.md) for the dependency model, service
contracts, provider boundaries, and threading invariants.

## License
Licensing terms are still being determined.

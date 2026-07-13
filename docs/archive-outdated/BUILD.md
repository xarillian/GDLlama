# Building the Project
## Prerequisites
- SCons
- CMake 3.14+
- Git
- Platform-specific tools
    - Windows: Visual Studio with the "Desktop development with C++" workload (for MSVC and linkers)
    - Linux: A C++ compiler like `clang` or `gcc`
    - macOS: Xcode Command Line Tools
- GPU-Specific SDKs
    - Vulkan SDK: For GPU-accelerated builds on Windows and Linux
    - Xcode: Provides the Metal framework for GPU-accelerated builds on macOS

## Build Steps

1. Clone the repository and initialize its submodules.
```shell
git clone https://github.com/xarillian/GDLlama.git
cd chorus-llm
git submodule update --init --recursive
```

2. Build from the project root.

SCons will automatically build the Godot C++ bindings and compile llama.cpp via CMake before linking the final shared library into `bin/`.

Execute the `scons` command that matches your operating system (`windows`, `linux`, or `macos`) and desired build type (`template_debug` or `template_release`):

```shell
scons platform=linux target=template_debug
```

For a GPU-accelerated build, pass the appropriate backend flag:

```shell
# Linux / Windows (Vulkan)
scons platform=linux target=template_debug use_vulkan=yes

# macOS (Metal)
scons platform=macos target=template_debug use_metal=yes
```

Note: The first build may take a while, as `llama.cpp` is compiled from scratch. Subsequent builds are incremental.

## Running Tests

```shell
scons platform=linux target=template_debug test
```

This produces `bin/run_tests`, which you can execute directly.

## Add to Your Godot Project

Copy the compiled shared library from `bin/` along with `plugin/chorus.gdextension` into your Godot project's `addons/chorus/` directory.

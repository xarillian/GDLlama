# Building the Project
## Prerequisites
- CMake 3.14+
- Ninja build system
- Vulkan SDK (for GPU builds)
- Git
- Platform-specific tools
    - Windows: Visual Studio with the "Desktop development with C++" workload (for `clang-cl` and linkers)
    - Linux: A C++ compiler like `clang` or `gcc`
    - macOS: Xcode Command Line Tools
- GPU-Specific SDKs
    - Vulkan SDK: For GPU-accelerated builds on Windows and Linux
    - Xcode: Provides the Metal framework for GPU-accelerated builds on macOS

## Build Steps

1. Clone the repository and initialize its submodules.
```shell
git clone https://github.com/xarillian/GDLlama.git
cd godot-llm
git submodule update --init --recursive
```

2.  Generate Godot Bindings (if building for the first time or updating Godot)

You need to generate the C++ bindings for Godot. This step is run from the `godot-cpp` directory.

```shell
cd godot-cpp
```

Execute the `SCons` command that matches your operating system (`windows`, `linux`, or `macos`.) and desired build type (debug or release):

```shell
SCons platform=X target=template_release use_clang=yes
```

After the command finishes, return to the root directory:
```shell
cd ..
```

3. Configure the Build with CMake

GDLlama uses CMake Presets to simplify configuration. Create a build directory and run cmake from inside it.

```shell
mkdir build
cd build
```

**Windows**

GPU (Vulkan):
```shell
cmake --preset windows-vulkan-release ..
```

CPU Only:
```shell
cmake --preset windows-cpu-release ..
```

**Linux**

GPU (Vulkan):
```shell
cmake --preset linux-vulkan-release ..
```

CPU Only:
```shell
cmake --preset linux-cpu-release ..
```

**macOS**

GPU (Metal):
```shell
cmake --preset macos-metal-release ..
```

CPU Only:
```shell
cmake --preset macos-cpu-release ..
```

4. Compile and Install

Once CMake has configured the project, compile and install it using Ninja.

```shell
ninja
ninja install
```

This will place the final files in the install directory at the root of the project.

5. Add to Your Godot Project

The compiled addon is now ready to be used in a Godot project. The `ninja install` command creates an `install` directory in the product root. The addon is located inside, organized by backend:
    - CPU builds: `install/cpu/addons/godot_llm`
    - GPU builds: `install/gpu/addons/godot_llm`

Copy the `godot_llm` folder from the appropriate path into the `addons` folder of your Godot project.

## Running Tests
If you configure the project using a debug preset (e.g. `linux-vulkan-debug`), the test suite will be enabled.

After running `ninja`, you can execute the tests from the build directory using `ctest` or `ninja test`.

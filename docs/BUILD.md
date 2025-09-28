# Building the Project
## Prerequisites
- CMake 3.14+
- Ninja build system
- Vulkan SDK (for GPU builds)
- Git
- (for Windows): Visual Studio Build Tools with clang-cl
    - or some equivalent

## Build Steps

1. Install the necessary build tools (e.g. `clang`) and Vulkan SDK for your operating system, then clone this repository.

```shell
git clone https://github.com/xarillian/GDLlama.git
cd godot-llm
git submodule update --init --recursive
mkdir build
cd build
```

2. Run `cmake`.

### Windows
from preset (recommended):
```shell
cmake --preset windows-vulkan-release ..
```

or manually:
```shell
cmake .. -GNinja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_CXX_FLAGS="/EHsc" -DLLAMA_NATIVE=OFF -DLLAMA_VULKAN=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_COMMON=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release
```

### Linux
```shell
cmake .. -GNinja -DLLAMA_NATIVE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DLLAMA_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
```

### Android
I haven't tested this at all, sorry. Here's the advice from the original project:

For Android, set `$NDK_PATH` to your android ndk directory, then:

```shell
cmake .. -GNinja -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH\cmake\android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS="-mcpu=generic" -DCMAKE_CXX_FLAGS="-mcpu=generic" -DCMAKE_BUILD_TYPE=Release
```

3. Compile and install with `ninja`.

```shell
ninja
ninja install
```

4. The folder `godot-llm/install/gpu/addons/godot_llm` can be copied to the `addons` folder of your Godot project. On Windows at least, you will also need to copy the required DLL dependencies from `godot-llm/install/bin` into your Godot project's `addons/godot_llm/bin/` directory:
- `ggml.dll`
- `ggml-base.dll`
- `ggml-cpu.dll`
- `llama.dll`

Replace "gpu" with "cpu" for a CPU build.
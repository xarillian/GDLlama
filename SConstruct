#!/usr/bin/env python
import os
import sys
import subprocess
from SCons.Script import Alias, ARGUMENTS, COMMAND_LINE_TARGETS, Default, Glob, SConscript

def build_llama_with_cmake(target, source, env):
    source_dir = os.path.abspath("external/llama.cpp")
    build_dir = os.path.abspath("external/llama.cpp/build")

    cmake_config = [
        "cmake",
        "-S", source_dir,
        "-B", build_dir,
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_CURL=OFF",
        "-DGGML_NATIVE=ON"
    ]

    # --- GPU CONFIG --- #

    targets_to_build = ["llama", "common"]

    if env.get("use_vulkan", False):
        print(">>> [SCons] Enabling Vulkan Backend")
        cmake_config.append("-DLLAMA_VULKAN=ON")
    else:
        cmake_config.append("-DLLAMA_VULKAN=OFF")
    
    if env.get("use_metal", False):
        print(">>> [SCons] Enabling Metal Backend")
        cmake_config.append("-DLLAMA_METAL=ON")
        cmake_config.append("-DLLAMA_METAL_EMBED_LIBRARY=ON")
    else:
        cmake_config.append("-DLLAMA_METAL=OFF")

    # Build Type
    if sys.platform == "win32":
        cmake_config.append("-DCMAKE_CONFIGURATION_TYPES=Release")
    else:
        cmake_config.append("-DCMAKE_BUILD_TYPE=Release")

    if sys.platform == "darwin":
        cmake_config.append("-DCMAKE_OSX_ARCHITECTURES=x86_64;arm64")

    # --- EXECUTE CMAKE COMMANDS --- #

    cmake_build = [
            "cmake", 
            "--build", build_dir, 
            "--config", "Release", 
            "--target"
    ] + targets_to_build + ["-j", "16"]

    try:
        print(">>> [SCons] Configuring Llama.cpp via CMake...")
        subprocess.check_call(cmake_config)
        
        print(">>> [SCons] Compiling Llama.cpp...")
        subprocess.check_call(cmake_build)
        
    except subprocess.CalledProcessError as e:
        print(f">>> [SCons] Error: CMake failed with exit code {e.returncode}")
        return 1
    except FileNotFoundError:
        print(">>> [SCons] Error: 'cmake' command not found. Is it installed and in your PATH?")
        return 1

    return 0

# ----------------------------------------------------------------------
# BASE CONFIGURATION
# ----------------------------------------------------------------------
env = SConscript("external/godot-cpp/SConstruct")
use_vulkan = ARGUMENTS.get("use_vulkan", "no") == "yes"
use_metal = ARGUMENTS.get("use_metal", "no") == "yes"

env["use_vulkan"] = use_vulkan
env["use_metal"] = use_metal

if env["platform"] == "windows":
    lib_paths = [
        "external/llama.cpp/build/src/Release",
        "external/llama.cpp/build/ggml/src/Release",
        "external/llama.cpp/build/common/Release",
    ]
else:
    lib_paths = [
        "external/llama.cpp/build/src",
        "external/llama.cpp/build/ggml/src",
        "external/llama.cpp/build/common",
    ]

if env["platform"] == "windows":
    # Force /MD to match llama.cpp Release build
    for flag in ["/MT", "/MTd", "/MDd"]:
        if flag in env["CCFLAGS"]:
            env["CCFLAGS"].remove(flag)
    
    env.Append(CCFLAGS=["/MD"])

    # /std:c++17 : Enable C++17 features
    # /EHsc      : Enable C++ exceptions (Required by llama.cpp/json)
    # /bigobj    : Often needed for heavy template headers like json.hpp
    env.Append(CXXFLAGS=["/std:c++17", "/EHsc", "/bigobj"])
    env["LIBPATH"] = lib_paths
    env.Append(LIBS=["advapi32", "user32", "kernel32"])

    if use_vulkan:
        env.Append(LIBS=["vulkan-1"])
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if vulkan_sdk:
            print(f">>> [SCons] Found Vulkan SDK at: {vulkan_sdk}")
            env.Append(LIBPATH=[os.path.join(vulkan_sdk, "Lib")])
        else:
            print(">>> [SCons] WARNING: VULKAN_SDK env var not found. Linking might fail.")
else:
    # Linux / macOS settings
    env.Append(CXXFLAGS=["-std=c++17", "-fexceptions"])

    if sys.platform.startswith("linux"):
        env.Append(CXXFLAGS=["-fopenmp"])
        env.Append(LINKFLAGS=["-fopenmp"])

    env["LIBPATH"] = lib_paths

    if sys.platform == "darwin" or env["platform"] == "macos":
        env.Append(LINKFLAGS=["-framework", "Accelerate", "-framework", "Foundation"])
        if use_metal:
            env.Append(LINKFLAGS=["-framework", "Metal", "-framework", "MetalKit"])


env.Append(CPPPATH=[
    "include",
    "src",
    # Llama Paths
    "external/llama.cpp/include",
    "external/llama.cpp/common",
    "external/llama.cpp/src",
    "external/llama.cpp/ggml/include",
    "external/llama.cpp/ggml/src",
    # Dependencies
    "external/llama.cpp/vendor"
])

# ----------------------------------------------------------------------
# SOURCE DEFINITIONS
# ----------------------------------------------------------------------
sources_core = Glob("src/chorus_core/*.cpp")
sources_chorus_llama = Glob("src/chorus_llama/*.cpp")
sources_godot = Glob("src/godot_chorus/*.cpp")

# ----------------------------------------------------------------------
# CMAKE TARGET DEFINITION
# ----------------------------------------------------------------------
llama_libs = ["llama", "ggml", "ggml-cpu", "ggml-base", "common"]

if env["platform"] == "windows":
    llama_libs = [lib + ".lib" for lib in llama_libs]
    llama_lib_trigger = "external/llama.cpp/build/src/Release/llama.lib"
else:
    llama_lib_trigger = "external/llama.cpp/build/src/libllama.a"

if use_metal and env["platform"] == "macos":
    llama_libs.append("ggml-metal")

cmake_target = env.Command(
    target=llama_lib_trigger,
    source=[],
    action=build_llama_with_cmake
)

# ----------------------------------------------------------------------
# BUILD TARGETS
# ----------------------------------------------------------------------
if "test" in COMMAND_LINE_TARGETS:
    test_env = env.Clone()
    test_env.Append(CPPDEFINES=["TEST_BUILD"])
    if env["platform"] == "windows":
        test_env.Append(LINKFLAGS=["/SUBSYSTEM:CONSOLE"])

    test_libs = (
        Glob("tests/*.cpp") + 
        Glob("tests/chorus_core/*.cpp") + 
        Glob("tests/chorus_llama/*.cpp")
    )

    test_env.Append(LIBS=llama_libs)

    test_program = test_env.Program(
        target="bin/run_tests",
        source=sources_core + sources_chorus_llama + test_libs,
    )

    test_env.Depends(test_program, cmake_target)
    
    Alias("test", test_program)

else:
    # --- LIBRARY BUILD (DEFAULT) ---
    env.Append(LIBS=llama_libs)
    
    library = env.SharedLibrary(
        target="bin/libgodot_chorus",
        source=sources_core + sources_chorus_llama + sources_godot
    )
    env.Depends(library, cmake_target)
    Default(library)
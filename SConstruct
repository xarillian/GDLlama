#!/usr/bin/env python
import os
import shutil
import sys
import subprocess
from SCons.Script import Alias, ARGUMENTS, COMMAND_LINE_TARGETS, Default, Glob, SConscript

def build_llama_with_cmake(target, source, env):
    source_dir = os.path.abspath("external/llama.cpp")
    build_dir = os.path.abspath("external/llama.cpp/build")

    if os.path.exists(build_dir):
        # Clean up.
        shutil.rmtree(build_dir)


    cmake_config = [
        "cmake",
        "-S", source_dir,
        "-B", build_dir,
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_CURL=OFF",
        "-DGGML_NATIVE=ON"
    ]

    # --- GPU CONFIG --- #

    # llama.cpp b9934 renamed the `common` target to `llama-common` (+ `llama-common-base`).
    targets_to_build = ["llama", "llama-common"]

    if env.get("use_vulkan", False):
        print(">>> [SCons] Enabling Vulkan Backend")
        cmake_config.append("-DGGML_VULKAN=ON")
    else:
        cmake_config.append("-DGGML_VULKAN=OFF")
    
    if env.get("use_metal", False):
        print(">>> [SCons] Enabling Metal Backend")
        cmake_config.append("-DLLAMA_METAL=ON")
        cmake_config.append("-DLLAMA_METAL_EMBED_LIBRARY=ON")
    else:
        cmake_config.append("-DLLAMA_METAL=OFF")
        cmake_config.append("-DGGML_METAL=OFF")  # Taking a "belt and suspenders" approach with metal
        if sys.platform == "darwin":
            cmake_config.append("-DGGML_BLAS=OFF")

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

# Auto-detect the GPU backend the prebuilt llama.cpp was actually compiled with, 
# so that the link libs always match regardless of the CLI flags.
# e.g. A prebuilt Vulkan llama linked without ggml-vulkan fails: `undefined reference to ggml_backend_vk_reg`.
# An explicit use_vulkan=yes / use_metal=yes still forces the backend on for a fresh build.
def _llama_built_with(flag, build_dir="external/llama.cpp/build"):
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.exists(cache):
        return False
    with open(cache) as f:
        return (flag + ":BOOL=ON") in f.read()

use_vulkan = use_vulkan or _llama_built_with("GGML_VULKAN")
use_metal = use_metal or _llama_built_with("GGML_METAL")

env["use_vulkan"] = use_vulkan
env["use_metal"] = use_metal

if env["platform"] == "windows":
    lib_paths = [
        "external/llama.cpp/build/src/Release",
        "external/llama.cpp/build/ggml/src/Release",
        "external/llama.cpp/build/common/Release",
    ]
    if use_vulkan:
        lib_paths.append("external/llama.cpp/build/ggml/src/ggml-vulkan/Release")
else:
    lib_paths = [
        "external/llama.cpp/build/src",
        "external/llama.cpp/build/ggml/src",
        "external/llama.cpp/build/common",
    ]
    if use_vulkan:
        lib_paths.append("external/llama.cpp/build/ggml/src/ggml-vulkan")

if env["platform"] == "windows":
    # Force /MD to match llama.cpp Release build
    for flag in ["/MT", "/MTd", "/MDd"]:
        if flag in env["CCFLAGS"]:
            env["CCFLAGS"].remove(flag)
    
    env.Append(CCFLAGS=["/MD"])

    # /std:c++20 : Enable C++20 features (our code; vendored deps stay at C++17)
    # /EHsc      : Enable C++ exceptions (Required by llama.cpp/json)
    # /bigobj    : Often needed for heavy template headers like json.hpp
    env.Append(CXXFLAGS=["/std:c++20", "/EHsc", "/bigobj"])
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
    env.Append(CXXFLAGS=["-std=c++20", "-fexceptions"])

    if sys.platform.startswith("linux"):
        env.Append(CXXFLAGS=["-fopenmp"])
        env.Append(LINKFLAGS=["-fopenmp"])

    env["LIBPATH"] = lib_paths

    if use_vulkan and sys.platform.startswith("linux"):
        env.Append(LIBS=["vulkan"])

    if sys.platform == "darwin" or env["platform"] == "macos":
        env.Append(LINKFLAGS=[
            "-framework", "Accelerate",
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "MetalKit"
            ]
        )


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
# VariantDir redirects intermediate build artifacts (.os/.o) into bin/obj/
# so they don't clutter the source tree. duplicate=0 keeps sources in place.
VariantDir("bin/obj/chorus",       "src/chorus",       duplicate=0)
VariantDir("bin/obj/godot_chorus", "src/godot_chorus", duplicate=0)
VariantDir("bin/obj/tests",        "tests",            duplicate=0)

sources_factory = Glob("bin/obj/chorus/*.cpp")
sources_runtime = Glob("bin/obj/chorus/runtime/*.cpp")
sources_echo    = Glob("bin/obj/chorus/backends/echo/*.cpp")
sources_llama   = Glob("bin/obj/chorus/backends/llama/*.cpp")
sources_godot   = Glob("bin/obj/godot_chorus/*.cpp")
sources_tests   = (
    Glob("bin/obj/tests/*.cpp") +
    Glob("bin/obj/tests/core/*.cpp") +
    Glob("bin/obj/tests/backends/echo/*.cpp") +
    Glob("bin/obj/tests/backends/llama/*.cpp") +
    Glob("bin/obj/tests/runtime/*.cpp")
)

# Tests compile under their own env. Built by one helper so the compiledb section
# below mirrors the exact flags the real test build uses (clangd needs them too).
def make_test_env(base_env):
    test_env = base_env.Clone()
    test_env.Append(CPPDEFINES=["TEST_BUILD"])
    test_env.Append(CPPPATH=["tests"])
    return test_env

# ----------------------------------------------------------------------
# CMAKE TARGET DEFINITION
# ----------------------------------------------------------------------
# Static link order: llama-common pulls from llama-common-base and llama, so it comes first.
llama_libs = ["llama-common", "llama-common-base", "llama", "ggml", "ggml-cpu", "ggml-base"]

if env["platform"] == "windows":
    llama_libs = [lib + ".lib" for lib in llama_libs]
    llama_lib_trigger = "external/llama.cpp/build/src/Release/llama.lib"
else:
    llama_lib_trigger = "external/llama.cpp/build/src/libllama.a"

if use_metal and env["platform"] == "macos":
    llama_libs.append("ggml-metal")

if use_vulkan:
    llama_libs.append("ggml-vulkan")

cmake_target = env.Command(
    target=llama_lib_trigger,
    source=[],
    action=build_llama_with_cmake
)

# ----------------------------------------------------------------------
# COMPILE COMMANDS (for clangd / IDE tooling)
# ----------------------------------------------------------------------
# Run `scons compiledb` to regenerate compile_commands.json.
if "compiledb" in COMMAND_LINE_TARGETS:
    env.Tool("compilation_db")
    compiledb = env.CompilationDatabase("compile_commands.json")
    env.Object(sources_factory + sources_runtime + sources_echo + sources_llama + sources_godot)
    make_test_env(env).Object(sources_tests)
    Alias("compiledb", compiledb)
    Default(compiledb)

# ----------------------------------------------------------------------
# BUILD TARGETS
# ----------------------------------------------------------------------
if "test" in COMMAND_LINE_TARGETS:
    test_env = make_test_env(env)
    if env["platform"] == "windows":
        test_env.Append(LINKFLAGS=["/SUBSYSTEM:CONSOLE"])

    test_env.Append(LIBS=llama_libs)

    test_program = test_env.Program(
        target="bin/run_tests",
        source=sources_echo + sources_llama + sources_factory + sources_runtime + sources_tests,
    )

    test_env.Depends(test_program, cmake_target)
    
    Alias("test", test_program)

else:
    # --- LIBRARY BUILD (DEFAULT) ---
    env.Append(LIBS=llama_libs)

    # llama.cpp/ggml are archived in with a static libstdc++, so the .so carries a
    # full private copy of the C++ runtime. When Godot loads us it already has its
    # own dynamic libstdc++.so.6; exporting our copy's symbols lets them interpose,
    # and a locale facet built against one vtable layout gets dispatched through the
    # other, crashing inside ostream/codecvt during model load. Localizing every
    # symbol pulled from a static archive keeps the process on one libstdc++.
    if sys.platform.startswith("linux"):
        env.Append(LINKFLAGS=["-Wl,--exclude-libs,ALL"])

    library = env.SharedLibrary(
        target="bin/libgodot_chorus",
        source=sources_echo + sources_llama + sources_factory + sources_runtime + sources_godot
    )
    env.Depends(library, cmake_target)
    Default(library)

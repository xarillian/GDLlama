#!/usr/bin/env python
import os
import sys
import subprocess
from SCons.Script import Alias, ARGUMENTS, COMMAND_LINE_TARGETS, Default, Glob, SConscript, Value

def discover_llama_revision():
    try:
        return subprocess.check_output(
            ["git", "-C", "external/llama.cpp", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"

def build_llama_with_cmake(target, source, env):
    source_dir = os.path.abspath("external/llama.cpp")
    build_dir = os.path.abspath(env["llama_build_dir"])

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
use_vulkan = ARGUMENTS.pop("use_vulkan", "no") == "yes"
use_metal = ARGUMENTS.pop("use_metal", "no") == "yes"
env = SConscript("external/godot-cpp/SConstruct")

llama_variant_parts = []
if use_vulkan:
    llama_variant_parts.append("vulkan")
if use_metal:
    llama_variant_parts.append("metal")
llama_variant = "-".join(llama_variant_parts) or "cpu"
llama_platform = str(env["platform"])
llama_arch = str(env.get("arch", "unknown") or "unknown")
llama_build_identity = f"{llama_platform}-{llama_arch}"
llama_build_dir = os.path.join(
    "external", "llama.cpp", "build", "chorus", llama_build_identity, llama_variant
)
print(f">>> [SCons] llama.cpp variant: {os.path.abspath(llama_build_dir)}")

env["use_vulkan"] = use_vulkan
env["use_metal"] = use_metal
env["llama_build_dir"] = llama_build_dir

if env["platform"] == "windows":
    lib_paths = [
        os.path.join(llama_build_dir, "src", "Release"),
        os.path.join(llama_build_dir, "ggml", "src", "Release"),
        os.path.join(llama_build_dir, "common", "Release"),
    ]
    if use_vulkan:
        lib_paths.append(os.path.join(llama_build_dir, "ggml", "src", "ggml-vulkan", "Release"))
else:
    lib_paths = [
        os.path.join(llama_build_dir, "src"),
        os.path.join(llama_build_dir, "ggml", "src"),
        os.path.join(llama_build_dir, "common"),
    ]
    if use_vulkan:
        lib_paths.append(os.path.join(llama_build_dir, "ggml", "src", "ggml-vulkan"))
    if use_metal and env["platform"] == "macos":
        lib_paths.append(os.path.join(llama_build_dir, "ggml", "src", "ggml-metal"))

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


env.Append(CPPPATH=["include", "src"])

# Vendor headers reach only the objects allowed to see them: the llama provider
# and the llama tests. An #include of <llama.h> from the core, the runtime, the
# factory, or a host adapter fails to compile rather than waiting on review
# (ARCHITECTURE.md, include discipline). The matching link seam -- an
# inner-layer target that builds with no llama.cpp artifacts present -- is
# heavier and lands with #11, where a second heavyweight backend pays for it.
llama_cpppath = [
    "external/llama.cpp/include",
    "external/llama.cpp/common",
    "external/llama.cpp/src",
    "external/llama.cpp/ggml/include",
    "external/llama.cpp/ggml/src",
    "external/llama.cpp/vendor",  # nlohmann/json, vendored inside llama.cpp
]

def with_llama_includes(base_env):
    scoped = base_env.Clone()
    scoped.Append(CPPPATH=llama_cpppath)
    return scoped

# ----------------------------------------------------------------------
# SOURCE DEFINITIONS
# ----------------------------------------------------------------------
# VariantDir redirects intermediate build artifacts (.os/.o) into bin/obj/
# so they don't clutter the source tree. duplicate=0 keeps sources in place.
VariantDir("bin/obj/chorus",       "src/chorus",       duplicate=0)
VariantDir("bin/obj/godot_chorus", "src/godot_chorus", duplicate=0)
VariantDir("bin/obj/tests",        "tests",            duplicate=0)

sources_core    = Glob("bin/obj/chorus/core/*.cpp")
sources_factory = Glob("bin/obj/chorus/*.cpp")
sources_runtime = Glob("bin/obj/chorus/runtime/*.cpp")
sources_echo    = Glob("bin/obj/chorus/backends/echo/*.cpp")
sources_llama   = Glob("bin/obj/chorus/backends/llama/*.cpp")
sources_godot   = Glob("bin/obj/godot_chorus/*.cpp")
sources_tests   = (
    Glob("bin/obj/tests/native/*.cpp") +
    Glob("bin/obj/tests/native/support/*.cpp") +
    Glob("bin/obj/tests/native/wlib/*.cpp") +
    Glob("bin/obj/tests/native/core/*.cpp") +
    Glob("bin/obj/tests/native/factory/*.cpp") +
    Glob("bin/obj/tests/native/backends/echo/*.cpp") +
    Glob("bin/obj/tests/native/runtime/*.cpp")
)
# Kept apart from the rest: these are the only test objects that may see a
# vendor header, so they compile under the scoped env alongside the provider.
sources_tests_llama = Glob("bin/obj/tests/native/backends/llama/*.cpp")

# Tests compile under their own env. Built by one helper so the compiledb section
# below mirrors the exact flags the real test build uses (clangd needs them too).
def make_test_env(base_env):
    test_env = base_env.Clone()
    test_env.Append(CPPDEFINES=["TEST_BUILD"])
    test_env.Append(CPPPATH=["tests/native", "tests/native/support"])
    return test_env

# ----------------------------------------------------------------------
# CMAKE TARGET DEFINITION
# ----------------------------------------------------------------------
# Static link order: llama-common pulls from llama-common-base and llama, so it comes first.
llama_libs = ["llama-common", "llama-common-base", "llama", "ggml", "ggml-cpu", "ggml-base"]

if env["platform"] == "windows":
    llama_libs = [lib + ".lib" for lib in llama_libs]
    llama_lib_trigger = os.path.join(llama_build_dir, "src", "Release", "llama.lib")
else:
    llama_lib_trigger = os.path.join(llama_build_dir, "src", "libllama.a")

if use_metal and env["platform"] == "macos":
    llama_libs.append("ggml-metal")

if use_vulkan:
    llama_libs.append("ggml-vulkan")

llama_revision = discover_llama_revision()
llama_build_signature = Value(
    f"revision={llama_revision};variant={llama_variant};platform={env['platform']};arch={env.get('arch', '')}"
)
cmake_target = env.Command(
    target=llama_lib_trigger,
    source=[llama_build_signature],
    action=build_llama_with_cmake
)

# ----------------------------------------------------------------------
# COMPILE COMMANDS (for clangd / IDE tooling)
# ----------------------------------------------------------------------
# Run `scons compiledb` to regenerate compile_commands.json.
if "compiledb" in COMMAND_LINE_TARGETS:
    env.Tool("compilation_db")
    compiledb = env.CompilationDatabase("compile_commands.json")
    env.Object(sources_core + sources_factory + sources_runtime + sources_echo + sources_godot)
    with_llama_includes(env).Object(sources_llama)
    compiledb_test_env = make_test_env(env)
    compiledb_test_env.Object(sources_tests)
    with_llama_includes(compiledb_test_env).Object(sources_tests_llama)
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

    llama_test_objects = with_llama_includes(test_env).Object(sources_llama + sources_tests_llama)

    test_program = test_env.Program(
        target="bin/run_tests",
        source=sources_echo + sources_core + sources_factory + sources_runtime + sources_tests + llama_test_objects,
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

    llama_objects = with_llama_includes(env).SharedObject(sources_llama)

    library = env.SharedLibrary(
        target="bin/libgodot_chorus",
        source=sources_echo + sources_core + sources_factory + sources_runtime + sources_godot + llama_objects
    )
    env.Depends(library, cmake_target)
    Default(library)

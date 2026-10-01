#!/usr/bin/env python
import os
import sys
import subprocess
from SCons.Script import Alias, ARGUMENTS, CacheDir, COMMAND_LINE_TARGETS, Default, GetOption, Glob, SConscript, Value
sys.dont_write_bytecode = True
from tools.materialize_llama import materialize
from tools.materialize_godot_api import materialize as materialize_godot_api

needs_googletest = any(target in COMMAND_LINE_TARGETS for target in ("test", "compiledb"))
required_submodules = ["godot-cpp", "llama.cpp", "nlohmann-json"]
if needs_googletest:
    required_submodules.append("googletest")

missing_submodules = [
    module
    for module in required_submodules
    if not os.path.exists(
        f"third-party/{module}/single_include/nlohmann/json.hpp" if module == "nlohmann-json"
        else f"third-party/{module}/CMakeLists.txt"
    )
]
if missing_submodules:
    raise SystemExit(
        f">>> [SCons] third-party submodules missing: {', '.join(missing_submodules)}. "
        "Run: git submodule update --init --recursive"
    )

# Build variant
use_vulkan = ARGUMENTS.pop("use_vulkan", "no") == "yes"
use_metal = ARGUMENTS.pop("use_metal", "no") == "yes"
host_test = ARGUMENTS.pop("host_test", "no") == "yes"
# ThreadSanitizer cannot run alongside the others; AddressSanitizer and UBSan share a build.
SANITIZERS = {"thread": "thread", "address": "address,undefined"}
sanitize = ARGUMENTS.pop("sanitize", "")
if sanitize and sanitize not in SANITIZERS:
    raise SystemExit(f">>> [SCons] sanitize must be one of: {', '.join(SANITIZERS)}")
if not ARGUMENTS.get("custom_api_file"):
    try:
        ARGUMENTS["custom_api_file"] = str(materialize_godot_api())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        raise SystemExit(f">>> [SCons] Godot API materialization failed: {error}")
# Godot reads the same variable; CI points it at a directory restored between runs.
if os.environ.get("SCONS_CACHE"):
    CacheDir(os.environ["SCONS_CACHE"])
env = SConscript("third-party/godot-cpp/SConstruct")

llama_variant_parts = []
if use_vulkan:
    llama_variant_parts.append("vulkan")
if use_metal:
    llama_variant_parts.append("metal")
llama_variant = "-".join(llama_variant_parts) or "cpu"
llama_platform = str(env["platform"])
llama_arch = str(env.get("arch", "unknown") or "unknown")
llama_build_identity = f"{llama_platform}-{llama_arch}"
try:
    llama_source_dir, llama_revision, llama_patch_identity = materialize()
except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
    raise SystemExit(f">>> [SCons] llama.cpp materialization failed: {error}")
llama_build_dir = os.path.join(
    "bin", "vendor-build", "llama.cpp", llama_revision + "-" + llama_patch_identity,
    llama_build_identity, llama_variant
)
print(f">>> [SCons] llama.cpp source: {llama_source_dir}")
print(f">>> [SCons] llama.cpp variant: {os.path.abspath(llama_build_dir)}")

if host_test:
    env.Append(CPPDEFINES=["CHORUS_HOST_TEST"])
env["use_vulkan"] = use_vulkan
env["use_metal"] = use_metal
env["llama_build_dir"] = llama_build_dir

# Platform toolchain
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
    # llama.cpp's Release archive uses `/MD`, so every linked object must use the same CRT.
    for flag in ["/MT", "/MTd", "/MDd"]:
        if flag in env["CCFLAGS"]:
            env["CCFLAGS"].remove(flag)
    
    env.Append(CCFLAGS=["/MD"])

    # Template-heavy llama.cpp headers can exceed COFF's default section limit.
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


# Vendor boundary
env.Append(CPPPATH=["include", "src"])

# Vendor headers are scoped to the llama provider and its tests. An include
# from an inner layer or host adapter fails at compile time.
llama_cpppath = [
    str(llama_source_dir / directory)
    for directory in ("include", "common", "src", "ggml/include", "ggml/src", "vendor")
]

googletest_cpppath = [
    "third-party/googletest/googletest/include",
    "third-party/googletest/googletest",
]

def with_llama_includes(base_env):
    scoped = base_env.Clone()
    scoped.Append(CPPPATH=llama_cpppath)
    return scoped

def with_host_json_includes(base_env):
    scoped = base_env.Clone()
    scoped.Append(CPPPATH=["third-party/nlohmann-json/single_include"])
    return scoped

# Layer source sets
# VariantDir redirects intermediate build artifacts (.os/.o) into bin/obj/
# so they don't clutter the source tree. duplicate=0 keeps sources in place.
VariantDir("bin/obj/chorus",       "src/chorus",       duplicate=0)
VariantDir("bin/obj/godot_chorus", "src/godot_chorus", duplicate=0)
VariantDir("bin/obj/chorus_c",     "src/chorus_c",     duplicate=0)
VariantDir("bin/obj/host_settings", "src/host_settings", duplicate=0)
VariantDir("bin/obj/tests",        "tests",            duplicate=0)
if needs_googletest:
    VariantDir("bin/obj/googletest", "third-party/googletest/googletest", duplicate=0)

sources_core    = Glob("bin/obj/chorus/core/*.cpp")
sources_factory = Glob("bin/obj/chorus/*.cpp")
sources_runtime = Glob("bin/obj/chorus/runtime/*.cpp")
sources_echo    = Glob("bin/obj/chorus/providers/echo/*.cpp")
sources_llama   = Glob("bin/obj/chorus/providers/llama/*.cpp")
sources_godot   = Glob("bin/obj/godot_chorus/*.cpp")
sources_c       = Glob("bin/obj/chorus_c/*.cpp")
sources_host_settings = Glob("bin/obj/host_settings/*.cpp")
sources_chorus  = sources_echo + sources_core + sources_factory + sources_runtime
sources_tests   = (
    Glob("bin/obj/tests/native/*.cpp") +
    Glob("bin/obj/tests/native/support/*.cpp") +
    Glob("bin/obj/tests/native/wlib/*.cpp") +
    Glob("bin/obj/tests/native/core/*.cpp") +
    Glob("bin/obj/tests/native/chorus_c/*.cpp") +
    Glob("bin/obj/tests/native/host_settings/*.cpp") +
    Glob("bin/obj/tests/native/factory/*.cpp") +
    Glob("bin/obj/tests/native/providers/echo/*.cpp") +
    Glob("bin/obj/tests/native/runtime/*.cpp")
)
sources_tests_c = Glob("bin/obj/tests/native/chorus_c/*.c")
# llama tests stay separate so only they compile with vendor headers.
sources_tests_llama = Glob("bin/obj/tests/native/providers/llama/*.cpp")
sources_googletest = ["bin/obj/googletest/src/gtest-all.cc"]

# One helper keeps the real test build and clangd flags identical.
def make_test_env(base_env):
    test_env = base_env.Clone()
    test_env.Append(CPPDEFINES=["TEST_BUILD", "CHORUS_HOST_TEST"])
    if base_env.get("use_vulkan", False):
        test_env.Append(CPPDEFINES=["CHORUS_TEST_VULKAN"])
    if base_env["platform"] == "windows":
        test_env.Append(CFLAGS=["/std:c11"])
    else:
        test_env.Append(CFLAGS=["-std=c11"])
    test_env.Append(CPPPATH=["tests/native", "tests/native/support"])
    test_env.Append(CPPPATH=googletest_cpppath)
    if sanitize:
        if base_env["platform"] == "windows":
            raise SystemExit(">>> [SCons] sanitize is supported with GCC and Clang, not MSVC")
        flag = "-fsanitize=" + SANITIZERS[sanitize]
        test_env.Append(CCFLAGS=[flag, "-fno-omit-frame-pointer", "-g"], LINKFLAGS=[flag])
    return test_env

def make_chorus_c_build_env(base_env):
    chorus_c_env = base_env.Clone()
    if base_env["platform"] == "windows":
        chorus_c_env.Append(CPPDEFINES=["CHORUS_C_BUILD"])
    return chorus_c_env

# llama.cpp dependency
def build_llama_with_cmake(target, source, env):
    source_dir = str(llama_source_dir)
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
        "-DGGML_NATIVE=ON",
        f"-DLLAMA_BUILD_COMMIT={llama_revision[:12]}-chorus-{llama_patch_identity}",
        f"-DGGML_BUILD_COMMIT={llama_revision[:12]}-chorus-{llama_patch_identity}",
        "-DLLAMA_BUILD_NUMBER=0",
    ]

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
        cmake_config.append("-DGGML_METAL=OFF")
        if sys.platform == "darwin":
            cmake_config.append("-DGGML_BLAS=OFF")

    if sys.platform == "win32":
        cmake_config.append("-DCMAKE_CONFIGURATION_TYPES=Release")
    else:
        cmake_config.append("-DCMAKE_BUILD_TYPE=Release")

    if sys.platform == "darwin":
        cmake_config.append("-DCMAKE_OSX_ARCHITECTURES=x86_64;arm64")

    cmake_build = [
            "cmake", 
            "--build", build_dir, 
            "--config", "Release", 
            "--target"
    ] + targets_to_build + ["-j", str(GetOption("num_jobs"))]

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

llama_build_signature = Value(
    f"revision={llama_revision};patch={llama_patch_identity};variant={llama_variant};platform={env['platform']};arch={env.get('arch', '')}"
)
cmake_target = env.Command(
    target=llama_lib_trigger,
    source=[llama_build_signature, "patches/llama-resource-cleanup.patch", "tools/materialize_llama.py"],
    action=build_llama_with_cmake
)
# The CMake build writes more libraries than its one declared target, so it stays out of the
# SCons cache; CI caches its build directory instead.
env.NoCache(cmake_target)

# Tooling target
if "compiledb" in COMMAND_LINE_TARGETS:
    env.Tool("compilation_db")
    compiledb = env.CompilationDatabase("compile_commands.json")
    env.Object(sources_core + sources_factory + sources_runtime + sources_echo + sources_godot)
    with_host_json_includes(env).Object(sources_host_settings)
    make_chorus_c_build_env(env).SharedObject(sources_c)
    with_llama_includes(env).Object(sources_llama)
    compiledb_test_env = make_chorus_c_build_env(make_test_env(env))
    compiledb_test_env.Object(sources_tests + sources_tests_c)
    with_llama_includes(compiledb_test_env).Object(sources_tests_llama)
    compiledb_test_env.Object(sources_googletest)
    Alias("compiledb", compiledb)
    Default(compiledb)

# Product targets
if "test" in COMMAND_LINE_TARGETS:
    test_env = make_chorus_c_build_env(make_test_env(env))
    if env["platform"] == "windows":
        test_env.Append(LINKFLAGS=["/SUBSYSTEM:CONSOLE"])

    test_env.Append(LIBS=llama_libs)

    llama_test_objects = with_llama_includes(test_env).Object(sources_llama + sources_tests_llama)
    chorus_c_test_objects = test_env.Object(sources_c)
    host_codec_test_objects = with_host_json_includes(test_env).Object(sources_host_settings)
    googletest_objects = test_env.Object(sources_googletest)

    test_program = test_env.Program(
        target="bin/run_tests",
        source=sources_chorus + chorus_c_test_objects + host_codec_test_objects + sources_tests + sources_tests_c + llama_test_objects + googletest_objects,
    )

    test_env.Depends(test_program, cmake_target)
    
    Alias("test", test_program)

else:
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

    chorus_c_objects = make_chorus_c_build_env(env).SharedObject(sources_c)
    host_codec_objects = with_host_json_includes(env).SharedObject(sources_host_settings)

    if env["target"] in ("editor", "template_debug"):
        sources_godot += env.GodotCPPDocData(
            "bin/gen/doc_data.gen.cpp", source=Glob("plugin/doc_classes/*.xml")
        )

    godot_library = env.SharedLibrary(
        target="bin/libgodot_chorus",
        source=sources_chorus + sources_godot + host_codec_objects + llama_objects
    )
    chorus_c_library = env.SharedLibrary(
        target="bin/libchorus_c",
        source=sources_chorus + chorus_c_objects + host_codec_objects + llama_objects
    )
    env.Depends([godot_library, chorus_c_library], cmake_target)
    Default(godot_library, chorus_c_library)

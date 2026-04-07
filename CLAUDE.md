# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is
A Godot 4.4+ GDExtension that wraps `llama.cpp` to provide local LLM inference in games. Built with SCons + CMake, targeting Windows/Linux/macOS.

NOTE: We should treat the docs as stale for now and lean on the code. We're in the middle of a re-write.

## Architecture
Two-layer design: a backend-agnostic core + a llama.cpp implementation.

## Tests

Custom lightweight test framework in `tests/test_utils.hpp` (macros: `ASSERT_TRUE`, `ASSERT_EQ`, color output).

- `tests/chorus_core/test_core_mechanics.cpp` — unit tests using `MockInferenceEngine`
- `tests/chorus_llama/test_llama_integration.cpp` — integration tests hitting real llama.cpp (requires model)
- `tests/test_runner.cpp` — `main()` entry point

## Key Build Notes

- C++17 required
- Links against llama.cpp static libs: `llama`, `ggml`, `ggml-cpu`, `ggml-base`, `common`
- macOS needs Metal/Foundation/Accelerate frameworks
- Linux links OpenMP
- Output: `bin/libgodot_chorus` (shared lib) + `bin/run_tests` (test binary)

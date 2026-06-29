# CLAUDE.md

## What This Is
A Godot 4.4+ GDExtension that wraps `llama.cpp` to provide local LLM inference in games. Built with SCons, targeting Windows/Linux/macOS. The project's thesis is "Real-time inference at scale on one local GPU". We should be able to run 40+ NPC conversations at one time on consumer hardware.

## Architecture
Two-layer design: a backend-agnostic core + a llama.cpp implementation.

## Tests

Custom lightweight test framework in `tests/test_utils.hpp` (macros: `ASSERT_TRUE`, `ASSERT_EQ`, color output).

- `tests/chorus_core/test_core_mechanics.cpp` — unit tests using `MockInferenceEngine`
- `tests/chorus_llama/test_llama_integration.cpp` — integration tests hitting real llama.cpp (requires model)
- `tests/test_runner.cpp` — `main()` entry point

## Key Build Notes

- C++20 required
- `just`
- Links against llama.cpp static libs: `llama`, `ggml`, `ggml-cpu`, `ggml-base`, `common`
- macOS needs Metal/Foundation/Accelerate frameworks
- Linux links OpenMP
- Output: `bin/libgodot_chorus` (shared lib) + `bin/run_tests` (test binary)

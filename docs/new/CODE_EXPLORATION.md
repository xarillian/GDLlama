# Code Exploration — how to read chorus-llm

A Godot 4.4+ GDExtension running local LLMs (llama.cpp) in games. New here? Read in this
order — each step follows the one request from GDScript down to llama.cpp and back.

## Start here (a request's path through the code)
1. `plugin/chorus.gdextension` → `src/godot_chorus/register_types.cpp` — the entry symbol
   (`llm_library_init`); how Godot loads the extension and registers the `GodotChorus` node.
2. `include/godot_chorus/godot_chorus.hpp` — the public API a game dev sees (the surface).
3. `GodotChorus::generate()` in `godot_chorus.cpp` — a request enters: dict → `ChorusRequest`.
4. `include/chorus_core/inference_engine.hpp` + `chorus_common.hpp` — the backend contract
   and the plain data types crossing it (`ChorusRequest`, `ChorusSignal`, `ChorusError`).
5. `LlamaScheduler::worker_loop` in `src/chorus_llama/llama_scheduler.cpp` — the heart:
   queue → slot → batch-decode → sample → emit `ChorusSignal`.
6. `GodotChorus::_drain_signals` — results marshalled back to the main thread → Godot signals.

To watch the whole flow execute, read `tests/chorus_llama/test_llama_integration.cpp`
(`test_simple_generation`); `tests/test_runner.cpp` is the test `main()`.
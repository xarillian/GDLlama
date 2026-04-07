# Chorus TODO

Tracking remaining work for the 2026 rewrite. Goals: modularity, identity (rename to Chorus),
batching + thread safety, scheduling + priority, backward compat with deprecations.

See `CLAUDE.md` for architecture overview.

---

## Blocking (nothing ships without these)

### Godot Binding Layer (`src/godot_chorus/`)
The directory is empty. No GDExtension class exists — the extension can't be used from GDScript at all.

- [ ] Create `GodotChorus` node class (`godot_chorus.hpp` / `godot_chorus.cpp`)
  - Extends `Node` or `RefCounted`
  - Owns a `LlamaEngine` instance
  - Exposes `ChorusConfig` / `GenerationConfig` as Godot properties
  - Wraps `submit_request()` with a GDScript-friendly API
- [ ] Wire `std::function` callbacks → Godot signals
  - `token_generated(token: String, request_id: int)`
  - `generation_complete(request_id: int)`
  - `error(message: String, request_id: int)`
- [ ] Implement `register_types()` and entry symbol (`llm_library_init`)
- [ ] Expose request submission methods to GDScript
  - `generate(prompt: String, config: Dictionary) -> int` (returns request ID)
  - `generate_chat(messages: Array, config: Dictionary) -> int`
- [ ] Backward-compat shims for old `GDLlama` API (with deprecation warnings)
  - `run_generate_text(prompt)` → `generate(prompt, {})`
  - `run_generate_chat(messages)` → `generate_chat(messages, {})`
  - etc.
- [ ] Update `.gdextension` entry symbol if needed

---

## High Priority (core correctness)

- [ ] Wire slot count to `ChorusConfig` — `init_slots(4)` is hardcoded in `llama_scheduler.cpp:71`
  - Add `num_slots` (or `batch_size`) field to `ChorusConfig`
  - Protect `slots` vector access with a mutex if count becomes dynamic
- [ ] Fix tokenizer buffer size assumption in `llama_utils.hpp:51`
  - Currently allocates `text.length() + 2` tokens (can silently overflow)
  - Should use a proper resize loop or llama.cpp's recommended pattern
- [ ] Structured error reporting — currently just strings in `ChorusSignal.text`
  - Add error codes or an error enum so callers can distinguish fatal vs. recoverable

---

## Medium Priority (feature completeness)

- [ ] Implement `RequestType::Embedding` processing path in the scheduler
  - `EventType::Embedding` is declared and `ChorusSignal.embedding` field exists — just unused
- [ ] Implement grammar / GBNF support
  - `GenerationConfig.grammar` field exists but `build_sampler()` ignores it
  - Wire into llama.cpp's grammar sampler chain
- [ ] Async model loading with progress callback
  - Loading blocks the thread (can be 30s+ for large models)
  - `@todo` at `llama_engine.cpp:27`: "Add progress callback here for Godot UI feedback"
- [ ] KV cache reuse for repeated system prompts
  - `@todo` at `llama_scheduler.cpp:123`
  - Currently clears cache per request — cheap win if system prompt is static
- [ ] Request cancellation — once submitted, requests can't be cancelled
  - No request ID → slot mapping exposed to callers
  - Add `cancel_request(id: int)` to `InferenceEngine` interface

---

## Low Priority (polish)

- [ ] Priority aging / starvation prevention
  - Low-priority requests can starve indefinitely; no aging mechanism exists
- [ ] Rename output artifact `libgodot_llm` → `libgodot_chorus` in `SConstruct`
- [ ] Refactor worker loop `continue` statements (`llama_scheduler.cpp:213-217`)
- [ ] Update docs (currently stale, flagged in `CLAUDE.md`)
  - Write new Chorus Architecture doc
  - Update `README.md` examples to new API
  - Update `API_REFERENCE.md`

---

## Completed

- [x] Backend-agnostic `InferenceEngine` abstract interface
- [x] `LlamaEngine` + `LlamaScheduler` concrete implementation
- [x] Multi-slot KV cache batching
- [x] Priority queue (`std::priority_queue<ChorusRequest>`)
- [x] Worker thread with mutex / condvar / atomic shutdown
- [x] Per-slot samplers
- [x] Rename to "Chorus" (`chorus_core` / `chorus_llama` namespaces)
- [x] Custom test framework + unit tests (mock engine)
- [x] Integration test scaffold (real llama.cpp, requires model)
- [x] GPU build support (Vulkan / Metal / OpenMP) in SConstruct

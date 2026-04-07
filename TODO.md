# Chorus TODO

Tracking remaining work for the 2026 rewrite. Goals: modularity, identity (rename to Chorus),
batching + thread safety, scheduling + priority, backward compat with deprecations.

See `CLAUDE.md` for architecture overview.

---

## Blocking (nothing ships without these)

### Godot Binding Layer (`src/godot_chorus/`)

- [x] Create `GodotChorus` node class (`godot_chorus.hpp` / `godot_chorus.cpp`)
- [x] Wire `std::function` callbacks → Godot signals via thread-safe queue drained in `_process`
  - `token_generated(request_id: int, token: String)`
  - `generation_complete(request_id: int, full_text: String)`
  - `generation_error(request_id: int, message: String)`
- [x] Implement `register_types()` and entry symbol (`llm_library_init`)
- [x] `generate(request: Dictionary) -> int` — single dict API; prompt + all params in one place
  - `stream: bool` (default false) controls whether `token_generated` fires per token
  - `priority: int` (default 0) feeds directly into the priority queue
  - `generation_complete` always carries full accumulated text regardless of stream mode
- [x] Expose all `ChorusConfig` fields as Godot properties (`model_path`, `context_size`, `thread_count`, `use_gpu`, `gpu_layers`)
- [x] Backward-compat shim — full GDLlama API coverage with deprecation warnings:
  - `unload_model`, `is_model_loaded`, `generate_text_async`, `generate_chat_async`
  - `generate_text` / `generate_chat` (sync, blocking — discouraged, compat only)
  - `stop_generate_text`, `is_running`, `reset_context`
  - `compute_embedding`, `compute_embedding_async` (stubs — not yet implemented)
  - `similarity_cos` (fully implemented — pure math)
  - All old signals emitted alongside new ones (`generate_text_updated`, `generate_text_finished`, `generate_text_error`, `embedding_computed`, `embedding_failed`)
  - Old property aliases: `n_predict`, `temperature`, `top_k`, `top_p`, `penalty_repeat`, `seed`, `n_ctx`, `n_gpu_layers`
  - No-op stubs with warnings: `ignore_eos`, `penalty_last_n`, `chat_template`, `n_batch`, `main_gpu`
- [ ] `generate_chat` (new API) — blocked on chat template + conversation history (see below)

---

## High Priority (core correctness)

- [ ] Wire slot count to `ChorusConfig` — `init_slots(4)` is hardcoded in `llama_scheduler.cpp:71`
  - Add `num_slots` field to `ChorusConfig` with a sensible default
  - Protect `slots` vector access with a mutex if count becomes dynamic
  - Relates to "Max batch size handling" in Further Work
- [ ] Fix tokenizer buffer size assumption in `llama_utils.hpp:51`
  - Currently allocates `text.length() + 2` tokens (can silently overflow)
  - Should use a proper resize loop or llama.cpp's recommended pattern
- [ ] Structured error reporting — currently just strings in `ChorusSignal.text`
  - Add error codes or an error enum so callers can distinguish fatal vs. recoverable
- [ ] `tokens_per_tick` hardcoded at 512 in `llama_scheduler.cpp`
  - Same class of problem as the slot count — should be configurable via `ChorusConfig`

---

## Medium Priority (feature completeness)

### Chat support
- [ ] Chat template support — prerequisite for everything below
  - llama.cpp has Jinja2 template support via `common`; needs wiring into `LlamaScheduler`
  - Old API supported a custom `chat_template` string — that stub should eventually work
- [ ] Conversation history — `generate_chat` needs per-session state
  - Represent as `Array[Dictionary]` with `role` + `content` keys (OpenAI-style)
  - Expose `export_conversation_history() -> Array[Dictionary]`
  - Expose `import_conversation_history(history: Array[Dictionary])`
  - Expose `clear_conversation_history()`
  - Decide: history on the node (single convo) or per-request (multi-session)?

### Other features
- [ ] Implement `RequestType::Embedding` processing path in the scheduler
  - `EventType::Embedding` is declared and `ChorusSignal.embedding` field exists — just unused
  - Will un-stub `compute_embedding` / `compute_embedding_async` in the compat layer
- [ ] Implement grammar / GBNF support
  - `GenerationConfig.grammar` field exists but `build_sampler()` ignores it
  - Wire into llama.cpp's grammar sampler chain
- [ ] Async model loading with progress callback
  - Loading blocks the thread (can be 30s+ for large models)
  - `@todo` at `llama_engine.cpp:27`
- [ ] KV cache reuse for repeated system prompts
  - `@todo` at `llama_scheduler.cpp:123`
  - Currently clears cache per request — cheap win if system prompt is static
- [ ] Request cancellation — once submitted, requests can't be cancelled
  - Add `cancel_request(id: int)` to `InferenceEngine` interface and wire through

---

## Low Priority (polish)

- [ ] Priority aging / starvation prevention
  - Low-priority requests can starve indefinitely; no aging mechanism exists
- [ ] Refactor worker loop `continue` statements (`llama_scheduler.cpp:213-217`)
- [ ] Update docs (currently stale, flagged in `CLAUDE.md`)
  - Write new Chorus Architecture doc
  - Update `README.md` examples to new API
  - Update `API_REFERENCE.md`

---

## Build / Tooling

- [x] `scons compiledb` target — generates `compile_commands.json` for clangd / IDE tooling
- [x] `.clangd` config — fallback include paths + strips GCC-only `-fno-gnu-unique` flag
- [x] `-DCMAKE_POSITION_INDEPENDENT_CODE=ON` for llama.cpp — required for linking into a shared `.so`
- [x] Vulkan CMake flag corrected: `LLAMA_VULKAN` → `GGML_VULKAN` (llama.cpp moved it to ggml)
- [ ] Verify Metal CMake flag is still correct (`LLAMA_METAL` / `GGML_METAL`) — may have the same rename issue

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
- [x] Streaming opt-in via `stream` key in `generate()` request dict (default false)
- [x] `generate(Dictionary)` — the "JSON blob in / response out" shape from Additional Considerations
- [x] Responses delivered per-request as soon as they finish (per-slot callbacks in scheduler)

---

# Further Work

Further work beyond the core 2026 rewrite goals.

## Conversation Export / Import API
Expose conversation history to GDScript so users can save/restore sessions, display history in UI, etc.

- `export_conversation_history() -> Array[Dictionary]`
- `import_conversation_history(history: Array[Dictionary])`
- `clear_conversation_history()`
- Hisory needs to be scoped to a per request-id (user def'd or UUID, assumably) to support multiple parallel conversations

## Chat Support
Blocked on chat template support. See Medium Priority above.

- `generate({prompt=..., history=[...]})` or a dedicated `generate_chat()`
- probably `generate({prompt=..., history=[...]})`
- Old `generate_chat_async` compat stub already exists, just needs the real implementation behind it

## Max Batch Size
Add a configurable `num_slots` to `ChorusConfig` so users can tune parallelism. See High Priority above.

## Enhance Access Parameters and Signals
No description yet

## Additional Considerations

The `generate(Dictionary)` API is already a step toward the "JSON blob in, blob out" vision. Next questions:
- Should there be a typed `ChorusRequest` resource class exposed to GDScript (instead of a raw Dictionary)?
- How does batch submission work from GDScript — submit N requests and get N `generation_complete` signals back?
- Priority queue is already wired; do users need to inspect or reorder the queue at runtime?

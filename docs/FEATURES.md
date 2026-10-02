# Feature Set

## Current
### Inference
- Perform generation against a local model.
- Generate normalized embeddings from a local model.
- Opt a generation model into serving embeddings, if capable; otherwise it skips the memory embedding batches reserve.
- Execute multiple inference requests concurrently through shared `llama.cpp` batches.
- Support stateless prompt and stateful conversations.
- Submit ordered generation and embedding batches with independent admission.

### Request Lifecycle
- Stream generated content while inference is running.
- Cancel queued or active requests.
- Query whether a request is active and which active request owns a session.
- Deliver exactly one terminal result for every accepted request.
- Release the loaded model before loading its replacement.

### Structured Output
- Constrain generation with a GBNF grammar.
- Convert a JSON Schema into a generation constraint.
- Configure structured-output constraints separately for each request, including explicit unconstrained intent.
- Return structured-output failures through the request error model.

### Conversations
- Maintain independent conversation histories for concurrent agents.
- Inspect, import, export, edit, clear, and list conversation histories.
- Assign durable IDs to stored conversation messages and report prompt-fitting omissions.
- Regenerate the latest assistant response.
- Inject temporary messages into one turn without modifying stored history.
- Fit conversation history off-thread with cached content budgeting and exact rendered checks while preserving system messages, injected context, and response capacity.
- Count literal tokens for individual conversation messages asynchronously.
- Roll back conversation mutations when generation fails or is cancelled.
- Report whether the latest turn completed, failed, or was cancelled.

### Scheduling
- Admit queued requests according to priority.
- Preserve submission order among queued requests with equal priority.

### Continuous Inference Scheduling
- Admit and retire inference sequences continuously as capacity becomes available.
- Batch active inference sequences within a shared token budget.
- Process long prompts in bounded chunks alongside active generation.
- Prioritize active decoding while ensuring newly admitted prompts continue to make progress.
- Contain recoverable capacity exhaustion to one request rather than failing unrelated active requests.

### Generation Control
- Configure generation separately for each request.
- Configure maximum output length, temperature, top-k, top-p, and random seed.
- Configure frequency, presence, and repetition penalties.
- Configure stop sequences.
- Configure advanced `llama.cpp` samplers, sampler order, and logit bias.
- Apply request choices over injected host choices while leaving absent choices to the provider; assignment selects a value and clearing removes only the local choice.
- Persist explicit generation choices as portable JSON, without persisting provider defaults.

### Chat and Reasoning
- Render conversations with a model-provided chat template.
- Select a shared project chat template or a per-request template.
- Preview a frozen conversation's fitted prompt asynchronously without occupying its session.
- Enable or disable model thinking when supported by the template.
- Deliver reasoning and visible content through separate channels.
- Prevent reasoning output from being stored as visible conversation content.

### Models and Hardware
- Load GGUF models through `llama.cpp`.
- Run inference on CPU or GPU.
- Configure GPU offload, primary GPU, threads, context size, batch size, and concurrent sequence capacity.

### Model Loading
- Load models asynchronously.
- Report model-loading progress and completion.
- Cancel an in-progress model load.

### KV Cache
- Reclaim a sequence's KV-cache state when its request ends.
- Keep only the attention window in sliding-window attention layers.
- Keep a conversation's KV cache in its idle slot between turns and process only the tokens that changed.

### Host Integration and Diagnostics
- Use Chorus through a host-neutral C++ runtime.
- Use Chorus as a native Godot GDExtension.
- Use typed Godot request resources with directly assigned and locally cleared generation choices.
- Edit shared Godot generation defaults in native Project Settings, imported from JSON and written back automatically only in the editor or explicitly saved from gameplay after source checks.
- Load generation defaults from files or JSON content into selected C runtime instances; export and explicitly save runtime choices through the C ABI.
- Use Chorus through a stable C ABI over the public runtime API.
- Block the host thread until events or prepared work arrive, instead of polling on a timer.
- Emit structured logs with typed fields.
- Correlate logs with request and session identities.
- Report engine failure independently from request failure.

### Providers and Routing
- Inspect a provider's capability envelope before loading and its effective capabilities after loading.
- Query loaded-model format, family, quantization, context limits, size, and modalities.
- Reject unsupported model formats, modalities, constraints, and options explicitly.
- Configure providers through self-described option schemas.

## Priority Backlog
### Request Lifecycle
- Report whether generation completed through EOS, a caller stop marker, or output-budget exhaustion.
- Report terminal token usage and monotonic queue, preparation, and inference timings.
- Expire stale requests at an optional monotonic deadline with an explicit terminal outcome.
- Configure a runtime-wide pending-request limit and report overload distinctly from invalid input.
- Inspect active work and report why admitted requests are waiting.

### Scheduling
- Inspect and reorder pending requests.
- Age queued-request priorities so sustained higher-priority traffic cannot starve older work.

### Conversations
- Fork an idle conversation into a new session with defined message-ID semantics, without requiring retained KV state.

### KV Cache
- Reuse one physical KV-cache representation of a common prompt prefix across multiple agents.
- Evict least-recently-used inactive session caches when additional KV capacity is needed, without affecting active requests or shared prefixes still in use.
- Rebuild an evicted session's private KV state from conversation history while reusing any retained shared prefix.
- Reuse the deepest exact rendered-token prefix that survives conversation edits, truncation, or injected-context changes.

### Models & Hardware
- Inspect the planned and effective GPU offload, context, concurrency, batching, and KV-cache geometry.
- Offer a CUDA build for NVIDIA GPUs alongside the portable Vulkan default.

### Tool Calling
- Define tools with typed names, descriptions, and input schemas.
- Advertise available tools separately for each request.
- Restrict or include tool calls for any individual request.
- Constrain generated tool calls to the advertised definitions.
- Parse generated tool calls into structured request events.
- Validate generated arguments against each tool's input schema.
- Add tool results to conversation history as typed messages.
- Correlate tool results with stable tool-call identities, including multiple calls within one turn.
- Resume generation after one or more tool results.
- Support multiple tool calls within one conversation turn.

### Providers and Routing
- Perform generation against an API.
- Use local in-process and remote asynchronous providers through the same runtime.
- Use vLLM through a provider that declares its supported capabilities and preserves runtime lifecycle contracts.
- Distinguish transport failure from a confirmed backend outcome.
- Expose whether cancellation stops local delivery or confirms that backend execution has stopped.

### Host Event Backpressure
- Bound buffered host-event counts and bytes with an explicit overload policy.
- Coalesce adjacent streamed chunks within the same request and output channel while preserving output order.
- Apply backpressure or fail affected requests explicitly when buffering limits are reached, rather than silently dropping output or terminal events.

### Workload Groups
- Apply concurrency and pending-work limits to caller-defined workload groups.

## Backlog
### Unsorted
- Semantic search?
    - obviously under-specified
    - RAG helpers in general

### Host Integration and Diagnostics
- Use Chorus through a command-line host over the public runtime API.
- Produce an opt-in load-and-infer diagnostic report with build/provider identity, model facts, selected options, a synthetic request, and relevant diagnostics, excluding model weights, conversation content, and local paths by default.
- Optionally spin briefly before blocking in an event wait, trading one busy core for lower wake latency on latency-critical hosts.

### Inference
- Generate normalized embeddings from an API.
- Select greedy tokens on the GPU with the same tie and NaN handling as CPU selection, avoiding the per-step logits transfer.
- Carry text, image, and audio references as typed message content, with explicit rejection of unsupported modalities.

### Request Lifecycle
- Configure whether a busy session rejects new requests or queues them up to a developer-defined limit.
- Start inference on prepared requests without waiting for the next host poll.

### Continuous Inference Scheduling
- Offer a deterministic scheduling mode for replays, testing, and deterministic netcode.

### Frame-Aware Inference
- Pace inference against a host-provided frame-time budget.
- Target a configured output rate for each inference stream.
- Slow or defer lower-priority work under renderer contention while preserving accepted requests.
- Reserve a developer-defined share of GPU memory for the renderer when planning inference resources.
- Adapt inference pacing to observed frame cost instead of static hardware classifications.

### Generation Control
- Report effective provider generation defaults to hosts.
- Report each generated token's log probability and its top alternatives on request, for confidence scoring and classification.
- Define named reusable generation-default profiles that hosts can persist and select for different agents or workloads.

### Models & Hardware
- Detect available inference hardware.
- Allow developers to select models using hardware bands that describe their users' hardware constraints.
- Recommend model size, quantization, concurrency, and pacing for the detected hardware.
- Calibrate provider-specific hardware characteristics that materially affect resource planning.
- Select a safe model-load resource plan from the model, workload, measured hardware, and renderer reserve before allocation.
- Reject infeasible resource plans before model allocation while identifying the limiting resource.
- Run on Linux, Windows, and macOS.
- Run on NVIDIA, AMD, and Apple GPUs through supported compute backends.
- Configure flash attention and prompt micro-batch size for local inference.
- Compile GPU pipelines for common batch shapes during model loading so first requests do not stall on shader compilation.

### KV Cache
- Warn when a developer-selected KV-cache configuration provides implausibly little capacity for the requested context and concurrency.
- Quantize KV-cache data.
- Allocate active sequences dynamically from a shared KV pool when unified storage is selected.
- Let developers select unified storage for shared-prefix workloads or split storage for independent sequences.
- Keep new requests queued until sufficient KV-cache capacity is available to run them safely.
- Reconfigure KV-cache geometry at a quiescent engine boundary without losing runtime conversation history.

### Retrieval and Context Assembly
- Organize retrievable context into reusable collections (RAG).
- Associate collections with models, agents, conversations, and requests.
    - under-specified
- Retrieve content through keyword matching, regex, embedding similarity, etc.
- Control retrieval through a priority system, conditions, exclusion groups, probability, persistence, cooldowns, and explicit overrides.
- Recursively retrieve content referenced by other retrieved entries.
- Retrieve relevant content from earlier conversation history.
- Add, edit, activate, and remove retrievable content at runtime.
- Assemble retrieved content into configured prompt positions within a token budget.
- Inspect which entries were retrieved, why they matched, where they were inserted, and how many tokens they consumed.

### Providers and Routing
- Route individual requests across multiple engines and models.
    - Is this really necessary ... ?
- Select routes according to request requirements and developer policy.
    - Seems difficult. Maybe out of scope?
- Preserve session identity when routing between engines.
    - Under-specified
- Use smaller models for ambient agents and larger models for focal agents.
- Apply per-agent LoRA adapters while sharing one base model in memory.
- Fall back to another provider when the selected provider becomes unavailable.

### Tech Debt
- Log Bridges should be generized / made easier. Can be done once we have another provider, or could build it against echo.
- Refactor the echo engine and llama scheduler worker loops around named lifecycle operations so their shutdown conditions are explicit.
- Verify UTF-8 stream partition invariance across visible content, reasoning, and stop filtering by comparing final output under different chunk boundaries.

## Out of Scope
### Model Transformation & Observability
- Merge, remove, and compare LoRA adapters.
- Package model adapters and their required prompts, templates, and evaluation suites as game assets.
    - Not written as a feature.
- Steer or suppress selected model concepts during inference through supported activation interventions.
- Merge or interpolate model checkpoints.
- Apply reproducible tensor transformations without modifying the source artifact.
- Fit or load Jacobian lenses for supported open-weight models.
- Record residual-stream activations at selected layers and token positions.

### Timed Agent Work
- Schedule agent work to run once at a specified time.
- Schedule agent work to recur at a fixed cadence.
    - monotonic?
- Allow an agent to choose when its next iteration runs.
- Schedule work against wall time, game time, monotonic, or a host-provided clock.
- Associate scheduled work with a session, request priority, model-routing policy, and generation configuration.
- Inspect, run immediately, pause, resume, reschedule, and cancel scheduled work.
- Persist schedules independently from in-memory conversation state.


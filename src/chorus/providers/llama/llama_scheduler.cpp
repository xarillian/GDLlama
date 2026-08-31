#include "chorus/providers/llama/llama_scheduler.hpp"
#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_utils.hpp"

#include <algorithm>
#include <cassert>

LlamaScheduler::~LlamaScheduler() {
    shutdown();
}

bool LlamaScheduler::load_model_from_file(const Chorus::LlamaLoadConfig& config) {
    llama_model_params model_params = Chorus::make_llama_model_params(config, _no_offload_devices);

    model = llama_model_load_from_file(config.weights_path.c_str(), model_params);
    if (!model) {
        _log.error("Failed to load model weights", {{"path", config.weights_path}});
        return false;
    }

    return true;
}

bool LlamaScheduler::init_context(const Chorus::LlamaLoadConfig& config) {
    llama_context_params ctx_params = Chorus::make_llama_context_params(config);

    context = llama_init_from_model(model, ctx_params);
    if (!context) {
        _log.error("Failed to create the inference context", {{"context_size", (int64_t)config.context_size}});
        return false;
    }

    return true;
}

void LlamaScheduler::init_slots(uint32_t count) {
    slots.clear();
    slots.resize(count);
    for (uint32_t index = 0; index < count; ++index)
        slots[index].id = static_cast<int>(index);
}

std::optional<Chorus::ChorusError>
LlamaScheduler::initialize(const Chorus::ChorusConfig& config, Chorus::Logger logger) {
    _log = std::move(logger);
    // Acquired before anything is loaded, so llama's own account of a failed
    // model load reaches the host instead of the process's stderr.
    _llama_log_bridge = Chorus::LlamaLogBridge::acquire(_log);

    auto parsed = Chorus::parse_llama_load_config(config);
    if (auto* rejection = std::get_if<Chorus::RequestRejection>(&parsed)) {
        _log.error("Rejected load options", {{"detail", rejection->message}});
        return rejection->error;
    }
    const Chorus::LlamaLoadConfig& load_config = std::get<Chorus::LlamaLoadConfig>(parsed);

    if (!load_model_from_file(load_config))
        return Chorus::ChorusError::ModelLoad;
    if (!init_context(load_config))
        return Chorus::ChorusError::ContextInit;

    init_slots(load_config.num_slots);
    _tokens_per_tick = load_config.tokens_per_tick;
    _batch_capacity = static_cast<int32_t>(llama_n_batch(context));

    if (static_cast<int64_t>(load_config.num_slots) * load_config.tokens_per_tick > _batch_capacity) {
        _log.warn(
            "Per-tick batch demand exceeds the context's effective n_batch and will be clamped to it",
            {{"num_slots", (int64_t)load_config.num_slots},
             {"tokens_per_tick", (int64_t)load_config.tokens_per_tick},
             {"requested_n_batch", (int64_t)load_config.n_batch},
             {"effective_n_batch", (int64_t)_batch_capacity}}
        );
    }

    batch = new llama_batch(llama_batch_init(_batch_capacity, 0, 1));

    Chorus::LoadedModelInfo info;
    info.model_id = config.model.model_id;
    info.format = Chorus::ModelFormat::Gguf;
    char buf[256];
    if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) > 0)
        info.family = buf;
    if (llama_model_desc(model, buf, sizeof(buf)) > 0)
        info.quantization = buf;
    info.maximum_context = (uint32_t)llama_model_n_ctx_train(model);
    // Prompt fitting budgets against an even division of the context across
    // concurrent slots.
    info.per_request_context = (uint32_t)(llama_n_ctx(context) / std::max<size_t>(1, slots.size()));
    info.model_bytes = llama_model_size(model);
    info.input_modalities = {Chorus::Modality::Text};
    info.output_modalities = {Chorus::Modality::Text};
    _model_info = info;

    try {
        _model_default_chat_templates = common_chat_templates_init(model, /*chat_template_override=*/"");
    } catch (const std::exception& e) {
        // Explicit overrides still work without embedded templates.
        // Otherwise, chat rejects during ingest,
        // `LlamaScheduler::render_chat_prompt` returns `std::nullopt`, and raw
        // prompt generation remains available.
        _log.warn("Chat templates unavailable", {{"detail", e.what()}});
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        is_running = true;
        _cancel_requested.clear();
    }
    worker_thread = std::thread(&LlamaScheduler::worker_loop, this);

    return std::nullopt;
}

void LlamaScheduler::shutdown() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        is_running = false;
    }
    queue_cv.notify_all();

    if (worker_thread.joinable()) {
        worker_thread.join();
    }

    if (batch) {
        llama_batch_free(*batch);
        delete batch;
        batch = nullptr;
    }
    for (auto& slot : slots)
        slot.sampler.reset();
    if (context) {
        llama_free(context);
        context = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    _model_default_chat_templates.reset();
    _model_info = std::nullopt;
    // `::llama_model_free` and `::llama_free` log during teardown, and those
    // records belong to this engine's host, so the bridge is released last.
    _llama_log_bridge.reset();
}

std::optional<Chorus::RenderedPrompt> LlamaScheduler::render_chat_prompt(
    const std::vector<Chorus::ChatMessage>& messages, const std::string& template_override, bool enable_thinking
) const {
    if (!model)
        return std::nullopt;
    auto rendered = [&] {
        std::lock_guard<std::mutex> template_lock(_template_mutex);
        return Chorus::render_llama_chat(
            model, _model_default_chat_templates.get(), template_override, messages, enable_thinking
        );
    }();
    if (std::holds_alternative<Chorus::RequestRejection>(rendered))
        return std::nullopt;
    auto& render = std::get<Chorus::LlamaChatRender>(rendered);
    // Vocab tokenization is read-only and safe beside the running worker.
    std::vector<int32_t> tokens =
        Chorus::LlamaUtils::tokenize(context, render.prompt, /*add_special=*/true, /*parse_special=*/true);
    return Chorus::RenderedPrompt{std::move(render.prompt), (int32_t)tokens.size()};
}

bool LlamaScheduler::is_healthy() const {
    return is_running.load();
}

bool LlamaScheduler::push_request(const Chorus::ChorusRequest& req) {
    auto pending = std::make_shared<PendingRequest>();
    pending->request = req;
    bool accepted = false;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (is_running) {
            request_queue.push(std::move(pending));
            accepted = true;
        }
    }
    if (accepted) {
        queue_cv.notify_one();
    }
    return accepted;
}

void LlamaScheduler::cancel_request(Chorus::RequestId id) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (!is_running)
            return;
        _cancel_requested.insert(id);
    }
    queue_cv.notify_one();
}

void LlamaScheduler::worker_loop() {
    while (true) {
        if (process_control_requests())
            return;

        ingest_new_requests();

        if (process_control_requests())
            return;

        bool has_work = prepare_next_batch(_tokens_per_tick);

        if (!has_work) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait_for(lock, std::chrono::milliseconds(10), [this] {
                return !is_running || !request_queue.empty() || !_cancel_requested.empty();
            });
            continue;
        }

        int decode_rc = run_inference();
        if (process_control_requests())
            return;

        if (decode_rc != 0) {
            fail_busy_slots(Chorus::ChorusError::Decode);
            if (decode_rc < 0) {
                _log.fatal("Decode failed unrecoverably; stopping the engine", {{"code", (int64_t)decode_rc}});
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    is_running = false;
                }
                queue_cv.notify_all();
            }
            continue;
        }

        sample_batch();
    }
}

bool LlamaScheduler::process_control_requests() {
    std::vector<PendingSignal> terminals;
    bool shutting_down = false;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        shutting_down = !is_running;

        std::priority_queue<PendingRequestPtr, std::vector<PendingRequestPtr>, PendingRequestCompare> retained;
        while (!request_queue.empty()) {
            auto pending = request_queue.top();
            request_queue.pop();
            if (shutting_down || _cancel_requested.contains(pending->request.id)) {
                terminals.emplace_back(
                    std::move(pending->request),
                    Chorus::ChorusSignal::Error{
                        Chorus::ChorusError::Cancelled,
                        shutting_down ? "Request cancelled: engine stopped." : "Request cancelled.",
                    }
                );
            } else {
                retained.push(std::move(pending));
            }
        }
        request_queue = std::move(retained);

        for (auto& slot : slots) {
            if (!slot.is_busy)
                continue;
            if (!shutting_down && !_cancel_requested.contains(slot.current_request.id))
                continue;
            terminals.push_back(release_with_event(
                slot,
                Chorus::ChorusSignal::Error{
                    Chorus::ChorusError::Cancelled,
                    shutting_down ? "Request cancelled: engine stopped." : "Request cancelled.",
                }
            ));
        }

        _cancel_requested.clear();
    }
    emit_signals(std::move(terminals));
    return shutting_down;
}

std::optional<LlamaScheduler::PendingSignal>
LlamaScheduler::take_cancellation_terminal_locked(const Chorus::ChorusRequest& request) {
    const bool stopped = !is_running;
    if (!stopped && _cancel_requested.erase(request.id) == 0)
        return std::nullopt;

    return PendingSignal{
        request,
        Chorus::ChorusSignal::Error{
            Chorus::ChorusError::Cancelled,
            stopped ? "Request cancelled: engine stopped." : "Request cancelled.",
        },
    };
}

std::optional<LlamaScheduler::PendingSignal> LlamaScheduler::resolve_pending_request(PendingRequest& pending) {
    Chorus::ChorusRequest& request = pending.request;
    if (!pending.resolved) {
        auto resolution = Chorus::resolve_llama_generation(request.gen_config);
        if (const auto* rejection = std::get_if<Chorus::RequestRejection>(&resolution)) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (auto cancelled = take_cancellation_terminal_locked(request))
                return cancelled;
            return PendingSignal{request, Chorus::ChorusSignal::Error{rejection->error, rejection->message}};
        }
        pending.resolved.emplace(std::get<Chorus::ResolvedLlamaGeneration>(std::move(resolution)));
    }

    if (pending.resolved->max_tokens != 0)
        return std::nullopt;

    std::lock_guard<std::mutex> lock(queue_mutex);
    if (auto cancelled = take_cancellation_terminal_locked(request))
        return cancelled;
    return PendingSignal{request, Chorus::ChorusSignal::Stop{}};
}

LlamaScheduler::PreparedRequestResult LlamaScheduler::prepare_request(PendingRequest& pending) {
    PreparedRequest prepared;
    prepared.max_tokens = pending.resolved->max_tokens;
    prepared.stop_sequences = std::move(pending.resolved->stop);

    auto sampler = Chorus::make_llama_sampler(model, std::move(*pending.resolved));
    std::optional<Chorus::RequestRejection> render_rejection;

    if (!pending.request.messages.empty()) {
        auto rendered = [&] {
            std::lock_guard<std::mutex> template_lock(_template_mutex);
            return Chorus::render_llama_chat(
                model,
                _model_default_chat_templates.get(),
                pending.request.chat_template,
                pending.request.messages,
                pending.request.gen_config.show_thinking.value_or(true)
            );
        }();
        if (const auto* rejection = std::get_if<Chorus::RequestRejection>(&rendered)) {
            render_rejection = *rejection;
        } else {
            auto& render = std::get<Chorus::LlamaChatRender>(rendered);
            prepared.tokens =
                Chorus::LlamaUtils::tokenize(context, render.prompt, /*add_special=*/true, /*parse_special=*/true);
            for (auto& template_stop_sequence : render.template_stop_sequences)
                prepared.stop_sequences.push_back(std::move(template_stop_sequence));
            // The request flag controls template rendering, not channel
            // separation. Some reasoning templates ignore the flag and still
            // open a think block, so capability alone selects the parser.
            prepared.parse_stream = Chorus::make_llama_chat_parse_stream(render);
        }
    } else {
        prepared.tokens = Chorus::LlamaUtils::tokenize(context, pending.request.prompt, true);
    }

    // Preserve failure precedence: a rejected render leaves no tokens but is
    // not a tokenization failure, and sampler validation happened before both.
    if (render_rejection)
        return std::move(*render_rejection);
    if (const auto* rejection = std::get_if<Chorus::RequestRejection>(&sampler))
        return *rejection;
    if (prepared.tokens.empty())
        return Chorus::RequestRejection{Chorus::ChorusError::Tokenize, "Tokenization failed (empty result)."};

    prepared.sampler = std::get<common_sampler_ptr>(std::move(sampler));
    return prepared;
}

std::optional<LlamaScheduler::PendingSignal>
LlamaScheduler::admit_request(const Chorus::ChorusRequest& request, PreparedRequestResult prepared) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    if (auto cancelled = take_cancellation_terminal_locked(request))
        return cancelled;

    if (const auto* rejection = std::get_if<Chorus::RequestRejection>(&prepared))
        return PendingSignal{request, Chorus::ChorusSignal::Error{rejection->error, rejection->message}};

    const int slot_idx = find_free_slot();
    if (slot_idx == -1) {
        // Capacity is checked before preparation, and only this worker releases
        // slots. Keep the runtime guard so a broken invariant terminates the
        // request once instead of indexing `LlamaScheduler::slots[-1]`.
        assert(slot_idx != -1);
        return PendingSignal{
            request,
            Chorus::ChorusSignal::Error{
                Chorus::ChorusError::Unknown,
                "Internal scheduler error: no free slot after admission.",
            },
        };
    }

    PreparedRequest& admitted = std::get<PreparedRequest>(prepared);
    Slot& slot = slots[slot_idx];
    slot.is_busy = true;
    slot.current_request = request;
    slot.n_past = 0;
    slot.n_decoded = 0;
    slot.input_cursor = 0;
    slot.current_input_tokens = std::move(admitted.tokens);
    slot.max_tokens = admitted.max_tokens;
    slot.sampler = std::move(admitted.sampler);
    slot.parse_stream = std::move(admitted.parse_stream);
    if (!admitted.stop_sequences.empty())
        slot.stop_filter.emplace(std::move(admitted.stop_sequences));
    return std::nullopt;
}

void LlamaScheduler::ingest_new_requests() {
    std::vector<PendingRequestPtr> pending_requests;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        pending_requests.reserve(request_queue.size());
        while (!request_queue.empty()) {
            pending_requests.push_back(request_queue.top());
            request_queue.pop();
        }
    }

    for (auto& pending : pending_requests) {
        Chorus::ChorusRequest& request = pending->request;

        std::optional<PendingSignal> cancellation;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            cancellation = take_cancellation_terminal_locked(request);
        }
        if (cancellation) {
            emit_signal(std::move(*cancellation));
            continue;
        }

        if (auto terminal = resolve_pending_request(*pending)) {
            emit_signal(std::move(*terminal));
            continue;
        }

        std::optional<PendingSignal> controlled_terminal;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (auto cancelled = take_cancellation_terminal_locked(request)) {
                controlled_terminal = std::move(*cancelled);
            } else if (find_free_slot() == -1) {
                request_queue.push(std::move(pending));
            }
        }
        if (controlled_terminal) {
            emit_signal(std::move(*controlled_terminal));
            continue;
        }
        if (!pending)
            continue;

        auto terminal = admit_request(request, prepare_request(*pending));
        if (!terminal)
            continue;

        const auto* error = std::get_if<Chorus::ChorusSignal::Error>(&terminal->event);
        if (error && error->code == Chorus::ChorusError::Tokenize)
            _log.for_request(request.id, request.session_id)
                .error("Tokenization produced no tokens; dropping the request");
        emit_signal(std::move(*terminal));
    }
}

bool LlamaScheduler::prepare_next_batch(int32_t tokens_per_tick) {
    llama_batch& curr_batch = *batch;
    curr_batch.n_tokens = 0;

    for (auto& slot : slots) {
        if (!slot.is_busy)
            continue;

        // `LlamaScheduler::batch` has `_batch_capacity` entries; an overrun
        // corrupts the heap.
        size_t capacity_left = (size_t)_batch_capacity - curr_batch.n_tokens;
        if (capacity_left == 0)
            break;

        if (slot.input_cursor < slot.current_input_tokens.size()) {

            size_t n_remaining = slot.current_input_tokens.size() - slot.input_cursor;
            size_t n_chunk = std::min({n_remaining, (size_t)tokens_per_tick, capacity_left});

            const int32_t chunk_size = static_cast<int32_t>(n_chunk);
            for (int32_t offset = 0; offset < chunk_size; ++offset) {
                const size_t token_index = slot.input_cursor + static_cast<size_t>(offset);
                const int32_t pos = slot.n_past + offset;
                const bool is_last_in_sequence = token_index == slot.current_input_tokens.size() - 1;

                Chorus::LlamaUtils::batch_add_seq(
                    curr_batch, slot.current_input_tokens[token_index], slot.id, pos, is_last_in_sequence
                );
            }

            slot.n_past += chunk_size;
            slot.input_cursor += n_chunk;
        }
    }

    return curr_batch.n_tokens > 0;
}

int LlamaScheduler::run_inference() {
    int rc = llama_decode(context, *batch);
    if (rc != 0) {
        _log.error("Decode failed", {{"code", (int64_t)rc}});
    }
    return rc;
}

void LlamaScheduler::sample_batch() {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    llama_batch& curr_batch = *batch;
    for (int i = 0; i < curr_batch.n_tokens; ++i) {
        if (!curr_batch.logits[i])
            continue;

        int seq_id = curr_batch.seq_id[i][0];
        Slot& slot = slots[seq_id];
        if (!slot.is_busy)
            continue;

        llama_token new_token_id = common_sampler_sample(slot.sampler.get(), context, i);
        common_sampler_accept(slot.sampler.get(), new_token_id, true);
        slot.n_decoded++;

        bool is_eos = llama_vocab_is_eog(vocab, new_token_id);
        bool is_limit = (slot.max_tokens > 0 && slot.n_decoded >= slot.max_tokens);

        if (is_eos) {
            complete_slot(slot, true);
            continue;
        }

        std::string piece = Chorus::LlamaUtils::token_to_piece(context, new_token_id);
        if (slot.parse_stream) {
            auto delta = slot.parse_stream->push(piece);
            if (!delta.reasoning.empty())
                emit_token(slot, slot.reasoning_chunker.push(delta.reasoning), Chorus::TokenChannel::Reasoning);
            // Only content meets the stop filter. An all-reasoning piece leaves
            // it empty: both emission paths no-op while completion bookkeeping
            // still runs.
            piece = std::move(delta.content);
        }
        if (slot.stop_filter) {
            auto filtered = slot.stop_filter->push(piece);
            emit_token(slot, std::move(filtered.safe_text));
            if (filtered.matched) {
                complete_slot(slot, false);
                continue;
            }
        } else {
            emit_token(slot, slot.content_chunker.push(piece));
        }

        if (is_limit)
            complete_slot(slot, true);
        else
            slot.current_input_tokens.push_back(new_token_id);
    }
}

int LlamaScheduler::find_free_slot() {
    for (size_t index = 0; index < slots.size(); ++index) {
        if (!slots[index].is_busy)
            return static_cast<int>(index);
    }
    return -1;
}

void LlamaScheduler::release_slot(int slot_id) {
    Slot& slot = slots[slot_id];

    slot.sampler.reset();
    slot.stop_filter.reset();
    slot.parse_stream.reset();
    slot.reasoning_chunker.reset();
    slot.content_chunker.reset();

    // Reclaim this sequence's KV cache so freed capacity is available to other slots.
    if (context) {
        llama_memory_t mem = llama_get_memory(context);
        llama_memory_seq_rm(mem, slot.id, 0, -1);
    }

    slot.current_request = {};
    slot.current_input_tokens.clear();
    slot.is_busy = false;
}

LlamaScheduler::PendingSignal LlamaScheduler::release_with_event(Slot& slot, Chorus::ChorusSignal::Event event) {
    PendingSignal pending{slot.current_request, std::move(event)};
    release_slot(slot.id);
    return pending;
}

void LlamaScheduler::emit_signal(PendingSignal pending) {
    if (!pending.request.on_event)
        return;

    Chorus::ChorusSignal signal{pending.request.id, std::move(pending.event)};
    pending.request.on_event(signal);
}

void LlamaScheduler::emit_signals(std::vector<PendingSignal> pending) {
    for (auto& signal : pending)
        emit_signal(std::move(signal));
}

void LlamaScheduler::emit_token(Slot& slot, std::string text, Chorus::TokenChannel channel) {
    if (text.empty() || !slot.current_request.on_event)
        return;

    Chorus::ChorusSignal signal{
        slot.current_request.id,
        Chorus::ChorusSignal::Token{channel, std::move(text)},
    };
    slot.current_request.on_event(signal);
}

void LlamaScheduler::complete_slot(Slot& slot, bool flush_pending_text) {
    // Finalize the reasoning split before taking the queue lock: it is a full
    // non-partial parse of the whole response, the parse stream is worker-owned,
    // and the host thread waits on queue_mutex in push/cancel. A request that
    // turns out cancelled below pays for a discarded parse; cancels are rare.
    Chorus::LlamaChatParseStream::Delta residual;
    if (slot.parse_stream) {
        residual = slot.parse_stream->finalize();
        // Reasoning residuals pass the same UTF-8 guard as the streamed path.
        // Content stays raw until finish_content_stream so malformed-byte
        // recovery cannot join non-contiguous stop-marker fragments.
        residual.reasoning = slot.reasoning_chunker.push(residual.reasoning);
    }

    std::vector<PendingSignal> buffered_tokens;
    std::optional<PendingSignal> terminal;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        const bool cancelled = !is_running || _cancel_requested.erase(slot.current_request.id) > 0;
        if (cancelled) {
            terminal = release_with_event(
                slot, Chorus::ChorusSignal::Error{Chorus::ChorusError::Cancelled, "Request cancelled."}
            );
        } else {
            // Residuals surface before the terminal: reasoning first, then
            // residual content passes through the stop filter before its
            // withheld tail is released.
            if (!residual.reasoning.empty()) {
                buffered_tokens.emplace_back(
                    slot.current_request,
                    Chorus::ChorusSignal::Token{
                        Chorus::TokenChannel::Reasoning,
                        std::move(residual.reasoning),
                    }
                );
            }
            if (flush_pending_text) {
                auto filtered = Chorus::finish_content_stream(
                    slot.stop_filter ? &*slot.stop_filter : nullptr, slot.content_chunker, residual.content
                );
                std::string text = std::move(filtered.safe_text);
                if (!text.empty())
                    buffered_tokens.emplace_back(
                        slot.current_request,
                        Chorus::ChorusSignal::Token{Chorus::TokenChannel::Content, std::move(text)}
                    );
            }
            terminal = release_with_event(slot, Chorus::ChorusSignal::Stop{});
        }
    }
    emit_signals(std::move(buffered_tokens));
    emit_signal(std::move(*terminal));
}

void LlamaScheduler::fail_busy_slots(Chorus::ChorusError code) {
    std::vector<PendingSignal> terminals;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        for (auto& slot : slots) {
            if (!slot.is_busy)
                continue;

            const bool cancelled = !is_running || _cancel_requested.erase(slot.current_request.id) > 0;
            terminals.push_back(
                cancelled ? release_with_event(
                                slot, Chorus::ChorusSignal::Error{Chorus::ChorusError::Cancelled, "Request cancelled."}
                            )
                          : release_with_event(slot, Chorus::ChorusSignal::Error{code, "Inference decode failed."})
            );
        }
    }
    emit_signals(std::move(terminals));
}

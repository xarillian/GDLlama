#include "chorus/providers/llama/llama_scheduler.hpp"
#include "chorus/providers/llama/llama_utils.hpp"
#include "chorus/core/common.hpp"

#include <algorithm>
#include <cassert>

// --------------------------------------------------------------------------
// LIFECYCLE
// --------------------------------------------------------------------------

LlamaScheduler::LlamaScheduler() {}

LlamaScheduler::~LlamaScheduler() {
    stop();
}

bool LlamaScheduler::load_model_from_file(const Chorus::LlamaLoadConfig& config) {
    if (model)
        return true;

    llama_model_params model_params = Chorus::make_llama_model_params(config, _no_offload_devices);

    // @todo Add a host-agnostic progress callback here (via ChorusConfig, like log_callback);
    //       the binding layer wires it to whatever UI the host uses. Core must not know the host.

    model = llama_model_load_from_file(config.weights_path.c_str(), model_params);
    if (!model) {
        Chorus::chorus_log(_log, Chorus::LogLevel::Error, "Failed to load model from " + config.weights_path);
        return false;
    }

    return true;
}

bool LlamaScheduler::init_context(const Chorus::LlamaLoadConfig& config) {
    if (context)
        return true;

    llama_context_params ctx_params = Chorus::make_llama_context_params(config);

    context = llama_init_from_model(model, ctx_params);
    if (!context) {
        Chorus::chorus_log(_log, Chorus::LogLevel::Error, "Failed to create Llama context.");
        return false;
    }

    return true;
}

void LlamaScheduler::init_slots(int count) {
    slots.clear();
    slots.resize(count);
    for (int i = 0; i < count; ++i) {
        slots[i].id = i;
        slots[i].is_busy = false;
        slots[i].n_past = 0;
        slots[i].n_decoded = 0;
        slots[i].input_cursor = 0;
    }
}

std::optional<Chorus::ChorusError> LlamaScheduler::initialize(const Chorus::ChorusConfig& config) {
    _log = config.log_callback;

    auto parsed = Chorus::parse_llama_load_config(config);
    if (auto* rejection = std::get_if<Chorus::RequestRejection>(&parsed)) {
        Chorus::chorus_log(_log, Chorus::LogLevel::Error, rejection->message);
        return rejection->error;
    }
    const Chorus::LlamaLoadConfig& load_config = std::get<Chorus::LlamaLoadConfig>(parsed);

    if (!load_model_from_file(load_config))
        return Chorus::ChorusError::ModelLoad;
    if (!init_context(load_config))
        return Chorus::ChorusError::ContextInit;

    init_slots(load_config.num_slots);
    _tokens_per_tick = load_config.tokens_per_tick;
    _batch_capacity = static_cast<int32_t>(load_config.n_batch);

    if (static_cast<int64_t>(load_config.num_slots) * load_config.tokens_per_tick > load_config.n_batch) {
        Chorus::chorus_log(
            _log,
            Chorus::LogLevel::Warn,
            "num_slots * tokens_per_tick exceeds n_batch; per-tick batch demand will be clamped to the logical "
            "batch capacity (n_batch)."
        );
    }

    batch = new llama_batch(llama_batch_init(static_cast<int32_t>(load_config.n_batch), 0, 1));

    Chorus::LoadedModelInfo info;
    info.model_id = config.model.model_id;
    info.format = Chorus::ModelFormat::Gguf;
    char buf[256];
    if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) > 0)
        info.family = buf;
    if (llama_model_desc(model, buf, sizeof(buf)) > 0)
        info.quantization = buf; // modest by contract: the desc string, e.g. "gemma3 270M F16"
    info.maximum_context = (uint32_t)llama_model_n_ctx_train(model);
    // What prompt fitting budgets against: today's slot model divides the
    // context evenly across concurrent requests.
    info.per_request_context = (uint32_t)(llama_n_ctx(context) / std::max<size_t>(1, slots.size()));
    info.model_bytes = llama_model_size(model);
    info.input_modalities = {Chorus::Modality::Text};
    info.output_modalities = {Chorus::Modality::Text};
    _model_info = info;

    try {
        _chat_templates = common_chat_templates_init(model, /*chat_template_override=*/"");
    } catch (const std::exception& e) {
        // No usable template: chat requests will reject at ingest and
        // render_chat_prompt returns nullopt; raw-prompt generation still works.
        Chorus::chorus_log(_log, Chorus::LogLevel::Warn, std::string("Chat templates unavailable: ") + e.what());
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        is_running = true;
        _cancel_requested.clear();
    }
    worker_thread = std::thread(&LlamaScheduler::worker_loop, this);

    return std::nullopt;
}

void LlamaScheduler::stop() {
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
    _chat_templates.reset();
    _model_info = std::nullopt;
}

std::optional<Chorus::RenderedPrompt> LlamaScheduler::render_chat_prompt(
    const std::vector<Chorus::ChatMessage>& messages, const std::string& template_override, bool enable_thinking
) const {
    if (!model || !_chat_templates)
        return std::nullopt;
    // Serialize template application against the worker's ingest path -- no
    // upstream thread-safety guarantee for common_chat_templates_apply.
    std::lock_guard<std::mutex> template_lock(_template_mutex);
    const common_chat_templates* tmpls = _chat_templates.get();
    common_chat_templates_ptr override_templates;
    if (!template_override.empty()) {
        try {
            override_templates = common_chat_templates_init(model, template_override);
        } catch (const std::exception&) {
            return std::nullopt;
        }
        tmpls = override_templates.get();
    }
    auto rendered = Chorus::render_llama_chat(tmpls, messages, enable_thinking);
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
    slot.n_past = 0;
    slot.n_decoded = 0;
    slot.input_cursor = 0;
    slot.max_tokens = -1;
    slot.is_busy = false;
}

LlamaScheduler::TerminalEvent
LlamaScheduler::release_with_terminal(Slot& slot, Chorus::EventType type, Chorus::ChorusError error, std::string text) {
    TerminalEvent terminal{slot.current_request, type, error, std::move(text)};
    release_slot(slot.id);
    return terminal;
}

void LlamaScheduler::emit_terminal(TerminalEvent terminal) {
    if (!terminal.request.on_event)
        return;

    Chorus::ChorusSignal signal;
    signal.request_id = terminal.request.id;
    signal.type = terminal.type;
    signal.error_code = terminal.error;
    signal.channel = terminal.channel;
    signal.text = std::move(terminal.text);
    terminal.request.on_event(signal);
}

void LlamaScheduler::emit_terminals(std::vector<TerminalEvent> terminals) {
    for (auto& terminal : terminals)
        emit_terminal(std::move(terminal));
}

void LlamaScheduler::emit_token(Slot& slot, std::string text, Chorus::TokenChannel channel) {
    if (text.empty() || !slot.current_request.on_event)
        return;

    Chorus::ChorusSignal signal;
    signal.request_id = slot.current_request.id;
    signal.type = Chorus::EventType::Token;
    signal.channel = channel;
    signal.text = std::move(text);
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

    std::vector<TerminalEvent> buffered_tokens;
    TerminalEvent terminal;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        const bool cancelled = !is_running || _cancel_requested.erase(slot.current_request.id) > 0;
        if (cancelled) {
            terminal = release_with_terminal(
                slot, Chorus::EventType::Error, Chorus::ChorusError::Cancelled, "Request cancelled."
            );
        } else {
            // Residuals surface before the terminal: reasoning first, then
            // residual content passes through the stop filter before its
            // withheld tail is released.
            if (!residual.reasoning.empty()) {
                buffered_tokens.push_back(
                    TerminalEvent{
                        slot.current_request,
                        Chorus::EventType::Token,
                        {},
                        std::move(residual.reasoning),
                        Chorus::TokenChannel::Reasoning,
                    }
                );
            }
            if (flush_pending_text) {
                auto filtered = Chorus::finish_content_stream(
                    slot.stop_filter ? &*slot.stop_filter : nullptr, slot.content_chunker, residual.content
                );
                std::string text = std::move(filtered.safe_text);
                if (!text.empty())
                    buffered_tokens.push_back(
                        TerminalEvent{slot.current_request, Chorus::EventType::Token, {}, std::move(text)}
                    );
            }
            terminal = release_with_terminal(slot, Chorus::EventType::Stop);
        }
    }
    emit_terminals(std::move(buffered_tokens));
    emit_terminal(std::move(terminal));
}

void LlamaScheduler::fail_busy_slots(Chorus::ChorusError code) {
    std::vector<TerminalEvent> terminals;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        for (auto& slot : slots) {
            if (!slot.is_busy)
                continue;

            const bool cancelled = !is_running || _cancel_requested.erase(slot.current_request.id) > 0;
            terminals.push_back(
                cancelled ? release_with_terminal(
                                slot, Chorus::EventType::Error, Chorus::ChorusError::Cancelled, "Request cancelled."
                            )
                          : release_with_terminal(slot, Chorus::EventType::Error, code, "Inference decode failed.")
            );
        }
    }
    emit_terminals(std::move(terminals));
}

bool LlamaScheduler::process_control_requests() {
    std::vector<TerminalEvent> terminals;
    bool shutting_down = false;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        shutting_down = !is_running;

        std::priority_queue<PendingRequestPtr, std::vector<PendingRequestPtr>, PendingRequestCompare> retained;
        while (!request_queue.empty()) {
            auto pending = request_queue.top();
            request_queue.pop();
            if (shutting_down || _cancel_requested.contains(pending->request.id)) {
                terminals.push_back(
                    TerminalEvent{
                        std::move(pending->request),
                        Chorus::EventType::Error,
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
            terminals.push_back(release_with_terminal(
                slot,
                Chorus::EventType::Error,
                Chorus::ChorusError::Cancelled,
                shutting_down ? "Request cancelled: engine stopped." : "Request cancelled."
            ));
        }

        _cancel_requested.clear();
    }
    emit_terminals(std::move(terminals));
    return shutting_down;
}

// --------------------------------------------------------------------------
// CORE PROCESSING
// --------------------------------------------------------------------------

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
        Chorus::ChorusRequest& chorus_request = pending->request;

        std::optional<TerminalEvent> controlled_terminal;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!is_running || _cancel_requested.erase(chorus_request.id) > 0) {
                controlled_terminal = TerminalEvent{
                    chorus_request,
                    Chorus::EventType::Error,
                    Chorus::ChorusError::Cancelled,
                    is_running ? "Request cancelled." : "Request cancelled: engine stopped.",
                };
            }
        }
        if (controlled_terminal) {
            emit_terminal(std::move(*controlled_terminal));
            continue;
        }

        if (!pending->resolved) {
            auto resolution = Chorus::resolve_llama_generation(chorus_request.gen_config);
            if (const auto* rejection = std::get_if<Chorus::RequestRejection>(&resolution)) {
                TerminalEvent terminal;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    const bool cancelled = !is_running || _cancel_requested.erase(chorus_request.id) > 0;
                    terminal = cancelled
                                   ? TerminalEvent{
                                         chorus_request,
                                         Chorus::EventType::Error,
                                         Chorus::ChorusError::Cancelled,
                                         is_running ? "Request cancelled." : "Request cancelled: engine stopped.",
                                     }
                                   : TerminalEvent{
                                         chorus_request,
                                         Chorus::EventType::Error,
                                         rejection->error,
                                         rejection->message,
                                     };
                }
                emit_terminal(std::move(terminal));
                continue;
            }
            pending->resolved.emplace(std::get<Chorus::ResolvedLlamaGeneration>(std::move(resolution)));
        }
        Chorus::ResolvedLlamaGeneration& resolved = *pending->resolved;
        const int32_t max_tokens = resolved.max_tokens;
        if (max_tokens == 0) {
            TerminalEvent terminal;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                const bool cancelled = !is_running || _cancel_requested.erase(chorus_request.id) > 0;
                terminal = cancelled
                               ? TerminalEvent{
                                     chorus_request,
                                     Chorus::EventType::Error,
                                     Chorus::ChorusError::Cancelled,
                                     is_running ? "Request cancelled." : "Request cancelled: engine stopped.",
                                 }
                               : TerminalEvent{chorus_request, Chorus::EventType::Stop, Chorus::ChorusError::None, {}};
            }
            emit_terminal(std::move(terminal));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!is_running || _cancel_requested.erase(chorus_request.id) > 0) {
                controlled_terminal = TerminalEvent{
                    chorus_request,
                    Chorus::EventType::Error,
                    Chorus::ChorusError::Cancelled,
                    is_running ? "Request cancelled." : "Request cancelled: engine stopped.",
                };
            } else if (find_free_slot() == -1) {
                request_queue.push(std::move(pending));
            }
        }
        if (controlled_terminal) {
            emit_terminal(std::move(*controlled_terminal));
            continue;
        }
        if (!pending)
            continue;

        Chorus::ResolvedLlamaGeneration admitted = std::move(resolved);
        pending->resolved.reset();
        std::vector<std::string> stop_sequences = std::move(admitted.stop);
        auto sampler = Chorus::make_llama_sampler(model, std::move(admitted));

        // #5: a non-empty messages list renders through the chat template
        // (embedded or per-request override) and supersedes the raw prompt.
        std::vector<int32_t> tokens;
        std::optional<Chorus::LlamaChatParseStream> parse_stream;
        std::optional<Chorus::RequestRejection> render_rejection;
        if (!chorus_request.messages.empty()) {
            std::lock_guard<std::mutex> template_lock(_template_mutex);
            const common_chat_templates* tmpls = _chat_templates.get();
            common_chat_templates_ptr override_templates;
            if (!chorus_request.chat_template.empty()) {
                try {
                    override_templates = common_chat_templates_init(model, chorus_request.chat_template);
                    tmpls = override_templates.get();
                } catch (const std::exception& e) {
                    render_rejection = Chorus::RequestRejection{
                        Chorus::ChorusError::InvalidRequest, std::string("Invalid chat_template: ") + e.what()
                    };
                }
            }
            if (!render_rejection && !tmpls) {
                render_rejection = Chorus::RequestRejection{
                    Chorus::ChorusError::InvalidRequest, "No chat template available for messages."
                };
            }
            if (!render_rejection) {
                const bool thinking = chorus_request.gen_config.thinking.value_or(true);
                auto rendered = Chorus::render_llama_chat(tmpls, chorus_request.messages, thinking);
                if (const auto* rejection = std::get_if<Chorus::RequestRejection>(&rendered)) {
                    render_rejection = *rejection;
                } else {
                    auto& render = std::get<Chorus::LlamaChatRender>(rendered);
                    tokens = Chorus::LlamaUtils::tokenize(
                        context, render.prompt, /*add_special=*/true, /*parse_special=*/true
                    );
                    for (auto& stop : render.additional_stops)
                        stop_sequences.push_back(std::move(stop));
                    // The request flag controls template rendering, not channel
                    // separation. Some reasoning templates ignore the flag and
                    // still open a think block, so capability alone selects the
                    // parser.
                    parse_stream = Chorus::make_llama_chat_parse_stream(render);
                }
            }
        } else {
            tokens = Chorus::LlamaUtils::tokenize(context, chorus_request.prompt, true);
        }

        std::optional<TerminalEvent> terminal;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!is_running || _cancel_requested.erase(chorus_request.id) > 0) {
                terminal = TerminalEvent{
                    chorus_request,
                    Chorus::EventType::Error,
                    Chorus::ChorusError::Cancelled,
                    is_running ? "Request cancelled." : "Request cancelled: engine stopped.",
                };
            } else if (render_rejection) {
                // Checked before tokens.empty(): a failed render leaves tokens
                // empty and must not masquerade as a Tokenize error.
                terminal = TerminalEvent{
                    chorus_request,
                    Chorus::EventType::Error,
                    render_rejection->error,
                    render_rejection->message,
                };
            } else if (const auto* rejection = std::get_if<Chorus::RequestRejection>(&sampler)) {
                terminal = TerminalEvent{
                    chorus_request,
                    Chorus::EventType::Error,
                    rejection->error,
                    rejection->message,
                };
            } else if (tokens.empty()) {
                terminal = TerminalEvent{
                    chorus_request,
                    Chorus::EventType::Error,
                    Chorus::ChorusError::Tokenize,
                    "Tokenization failed (empty result).",
                };
            } else if (const int slot_idx = find_free_slot(); slot_idx == -1) {
                // Unreachable: admission is capacity-gated above and this
                // single-threaded worker frees no slot in between. Guard the
                // index anyway so a broken invariant yields one Error terminal
                // rather than indexing slots[-1] under NDEBUG.
                assert(slot_idx != -1);
                terminal = TerminalEvent{
                    chorus_request,
                    Chorus::EventType::Error,
                    Chorus::ChorusError::Unknown,
                    "Internal scheduler error: no free slot after admission.",
                };
            } else {
                Slot& slot = slots[slot_idx];
                slot.is_busy = true;
                slot.current_request = chorus_request;
                slot.n_past = 0;
                slot.n_decoded = 0;
                slot.input_cursor = 0;
                slot.current_input_tokens = std::move(tokens);
                slot.max_tokens = max_tokens;
                slot.sampler = std::get<common_sampler_ptr>(std::move(sampler));
                slot.parse_stream = std::move(parse_stream);
                if (!stop_sequences.empty())
                    slot.stop_filter.emplace(std::move(stop_sequences));
            }
        }
        if (terminal) {
            if (terminal->error == Chorus::ChorusError::Tokenize)
                Chorus::chorus_log(_log, Chorus::LogLevel::Error, "Tokenization produced no tokens; dropping request.");
            emit_terminal(std::move(*terminal));
        }
    }
}

bool LlamaScheduler::prepare_next_batch(int32_t tokens_per_tick) {
    llama_batch& curr_batch = *batch;
    curr_batch.n_tokens = 0; // Reset for this tick

    for (auto& slot : slots) {
        if (!slot.is_busy)
            continue;

        // The batch buffers hold _batch_capacity entries; overrunning them is heap corruption.
        size_t capacity_left = (size_t)_batch_capacity - curr_batch.n_tokens;
        if (capacity_left == 0)
            break;

        if (slot.input_cursor < slot.current_input_tokens.size()) {

            size_t n_remaining = slot.current_input_tokens.size() - slot.input_cursor;
            size_t n_chunk = std::min({n_remaining, (size_t)tokens_per_tick, capacity_left});

            for (size_t i = 0; i < n_chunk; ++i) {
                int32_t pos = slot.n_past + i;
                bool is_last_in_sequence = (slot.input_cursor + i == slot.current_input_tokens.size() - 1);

                Chorus::LlamaUtils::batch_add_seq(
                    curr_batch, slot.current_input_tokens[slot.input_cursor + i], slot.id, pos, is_last_in_sequence
                );
            }

            slot.n_past += n_chunk;
            slot.input_cursor += n_chunk;
        }
    }

    return curr_batch.n_tokens > 0;
}

int LlamaScheduler::run_inference() {
    int rc = llama_decode(context, *batch);
    if (rc != 0) {
        Chorus::chorus_log(_log, Chorus::LogLevel::Error, "llama_decode failed with code " + std::to_string(rc) + ".");
    }
    return rc;
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
                Chorus::chorus_log(_log, Chorus::LogLevel::Fatal, "Fatal decode error; stopping engine.");
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    is_running = false;
                }
                queue_cv.notify_all();
            }
            continue;
        }

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
                // Only content meets the stop filter. An all-reasoning piece
                // leaves it empty: both emit paths below no-op on empty text,
                // while the EOS/limit bookkeeping still runs.
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
}

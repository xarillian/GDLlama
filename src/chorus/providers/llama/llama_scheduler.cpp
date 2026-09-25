#include "chorus/providers/llama/llama_scheduler.hpp"
#include "chorus/providers/llama/llama_engine.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

LlamaScheduler::~LlamaScheduler() {
    shutdown();
}

bool LlamaScheduler::load_model_from_file(const Chorus::LlamaLoadConfig& config, const Chorus::InitializationControl& control,
                                          bool& callback_cancelled, bool& callback_failed) {
    struct ProgressState {
        const Chorus::InitializationControl& control;
        bool& cancelled;
        bool& failed;
    } state{control, callback_cancelled, callback_failed};
    auto params = Chorus::make_llama_model_params(config, _no_offload_devices);
    params.progress_callback = [](float fraction, void* user_data) noexcept -> bool {
        auto& progress = *static_cast<ProgressState*>(user_data);
        try {
            if (progress.control.on_progress)
                progress.control.on_progress({Chorus::LoadPhase::LoadingModel, fraction});
        } catch (...) {
            progress.failed = true;
            return false;
        }
        if (progress.control.stop_token.stop_requested()) {
            progress.cancelled = true;
            return false;
        }
        return true;
    };
    params.progress_callback_user_data = &state;
    model = llama_model_load_from_file(config.weights_path.c_str(), params);
    if (!model)
        _log.error("Failed to load model weights", {{"path", config.weights_path}});
    return model != nullptr;
}

bool LlamaScheduler::init_context(const Chorus::LlamaLoadConfig& config) {
    context = llama_init_from_model(model, Chorus::make_llama_context_params(config));
    if (!context)
        _log.error("Failed to create the inference context", {{"context_size", (int64_t)config.context_size}});
    return context != nullptr;
}

std::optional<Chorus::InitializationFailure>
LlamaScheduler::initialize(const Chorus::ChorusConfig& config, Chorus::Logger logger, const Chorus::InitializationControl& control) {
    _log = std::move(logger);
    try {
        _llama_log_bridge = Chorus::LlamaLogBridge::acquire(_log);
        const auto cancelled = [] { return Chorus::InitializationFailure{Chorus::ChorusError::Cancelled, "Llama initialization cancelled."}; };
        const auto fail = [this](Chorus::InitializationFailure failure) {
            shutdown();
            return failure;
        };
        if (control.stop_token.stop_requested())
            return fail(cancelled());
        auto parsed = Chorus::parse_llama_load_config(config);
        if (auto* rejection = std::get_if<Chorus::RequestRejection>(&parsed))
            return fail({rejection->error, rejection->message});
        if (control.stop_token.stop_requested())
            return fail(cancelled());
        if (control.on_progress)
            control.on_progress({Chorus::LoadPhase::LoadingModel, std::nullopt});
        const auto& load_config = std::get<Chorus::LlamaLoadConfig>(parsed);
        bool callback_cancelled = false;
        bool callback_failed = false;
        if (!load_model_from_file(load_config, control, callback_cancelled, callback_failed)) {
            if (callback_failed)
                return fail({Chorus::ChorusError::Unknown, "Model progress callback failed."});
            if (callback_cancelled || control.stop_token.stop_requested())
                return fail(cancelled());
            return fail({Chorus::ChorusError::ModelLoad, "Failed to load model weights."});
        }
        if (callback_failed)
            return fail({Chorus::ChorusError::Unknown, "Model progress callback failed."});
        if (control.stop_token.stop_requested())
            return fail(cancelled());
        if (control.on_progress)
            control.on_progress({Chorus::LoadPhase::InitializingEngine, std::nullopt});
        if (control.stop_token.stop_requested())
            return fail(cancelled());
        if (!init_context(load_config))
            return fail(control.stop_token.stop_requested()
                            ? cancelled()
                            : Chorus::InitializationFailure{Chorus::ChorusError::ContextInit,
                                                            "Failed to create the inference context."});
        if (control.stop_token.stop_requested())
            return fail(cancelled());

        _batch_capacity = static_cast<int32_t>(llama_n_batch(context));
        _micro_batch_capacity = static_cast<int32_t>(llama_n_ubatch(context));
        _max_concurrent_requests = load_config.max_concurrent_requests;
        if (_max_concurrent_requests > static_cast<uint32_t>(_batch_capacity)) {
            const std::string message = "This model configuration supports at most " + std::to_string(_batch_capacity) +
                                        " concurrent requests; requested " + std::to_string(_max_concurrent_requests) + ".";
            return fail({Chorus::ChorusError::UnsupportedOption, message});
        }
        _pooling = llama_pooling_type(context);
        _embedding_dimensions = llama_model_n_embd_out(model);
        _has_encoder = llama_model_has_encoder(model);
        _has_decoder = llama_model_has_decoder(model);
        if (_pooling == LLAMA_POOLING_TYPE_RANK) {
            return fail({Chorus::ChorusError::UnsupportedFeature, "The model's rank pooling is not an embedding output."});
        }
        if (control.stop_token.stop_requested())
            return fail(cancelled());
        batch.initialize(_batch_capacity, 0, _max_concurrent_requests);
        _sequence_ids.reset(static_cast<int>(_max_concurrent_requests));

        Chorus::LoadedModelInfo info;
        info.model_id = config.model.model_id;
        info.format = Chorus::ModelFormat::Gguf;
        info.family = Chorus::LlamaUtils::model_metadata(model, "general.architecture");
        info.quantization = Chorus::LlamaUtils::model_description(model);
        info.maximum_context = static_cast<uint32_t>(llama_model_n_ctx_train(model));
        info.per_request_context = static_cast<uint32_t>(llama_n_ctx(context) / _max_concurrent_requests);
        info.model_bytes = llama_model_size(model);
        info.input_modalities = {Chorus::Modality::Text};
        info.output_modalities = {Chorus::Modality::Text};
        _model_info = std::move(info);
        try {
            _model_default_chat_templates = common_chat_templates_init(model, "");
        } catch (const std::exception& e) {
            _log.warn("Chat templates unavailable", {{"detail", e.what()}});
        }
        if (control.stop_token.stop_requested())
            return fail(cancelled());
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            is_running = true;
            _cancel_requested.clear();
        }
        worker_thread = std::thread(&LlamaScheduler::worker_loop, this);
        if (control.stop_token.stop_requested())
            return fail(cancelled());
        return std::nullopt;
    } catch (...) {
        shutdown();
        throw;
    }
}

void LlamaScheduler::shutdown() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        is_running = false;
    }
    queue_cv.notify_all();
    if (worker_thread.joinable())
        worker_thread.join();
    std::unique_lock<std::shared_mutex> preparation_lock(_preparation_fence);
    request_queue = {};
    _terminal_deliveries.clear();
    _cancel_requested.clear();
    batch.reset();
    _active_sequences.clear();
    _sequence_ids.reset();
    if (context) {
        llama_free(context);
        context = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    _model_default_chat_templates.reset();
    _model_info.reset();
    _llama_log_bridge.reset();
}

std::variant<Chorus::RenderedPrompt, Chorus::RequestRejection> LlamaScheduler::render_chat_prompt(
    const std::vector<Chorus::ChatMessage>& messages, const std::string& template_override, bool enable_thinking
) const {
    std::shared_lock<std::shared_mutex> resource_lock(_preparation_fence);
    if (!is_healthy())
        return Chorus::RequestRejection{Chorus::ChorusError::EngineNotReady, "Llama preparation is closed."};
    std::lock_guard<std::mutex> lock(_template_mutex);
    auto rendered = Chorus::render_llama_chat(model, _model_default_chat_templates.get(), template_override, messages, enable_thinking);
    if (auto* rejection = std::get_if<Chorus::RequestRejection>(&rendered))
        return *rejection;
    auto& value = std::get<Chorus::LlamaChatRender>(rendered);
    auto tokens = Chorus::LlamaUtils::tokenize_vocabulary(llama_model_get_vocab(model), value.prompt, true, true);
    if (!tokens)
        return Chorus::RequestRejection{Chorus::ChorusError::Tokenize, "Rendered prompt tokenization failed."};
    return Chorus::RenderedPrompt{std::move(value.prompt), static_cast<int32_t>(tokens->size())};
}

std::variant<int64_t, Chorus::RequestRejection> LlamaScheduler::count_message_tokens(const std::string& text) const {
    std::shared_lock<std::shared_mutex> lock(_preparation_fence);
    if (!is_healthy())
        return Chorus::RequestRejection{Chorus::ChorusError::EngineNotReady, "Llama preparation is closed."};
    auto tokens = Chorus::LlamaUtils::tokenize_vocabulary(llama_model_get_vocab(model), text, false, false);
    if (!tokens)
        return Chorus::RequestRejection{Chorus::ChorusError::Tokenize, "Message tokenization failed."};
    return static_cast<int64_t>(tokens->size());
}

std::optional<Chorus::RequestRejection> LlamaScheduler::validate_request(const Chorus::ChorusRequest& request) const {
    std::shared_lock<std::shared_mutex> lock(_preparation_fence);
    if (!is_healthy())
        return Chorus::RequestRejection{Chorus::ChorusError::EngineNotReady, "Llama preparation is closed."};
    if (request.type == Chorus::RequestType::Embedding)
        return validate_embedding(request);
    if (!_has_decoder)
        return Chorus::RequestRejection{Chorus::ChorusError::UnsupportedFeature, "The loaded model cannot generate text."};
    return Chorus::validate_llama_request(request);
}

bool LlamaScheduler::is_healthy() const { return is_running.load(); }

Chorus::EngineCapabilities LlamaScheduler::capabilities() const {
    auto caps = Chorus::llama_provider_capabilities();
    caps.streaming = _has_decoder;
    caps.embeddings = Chorus::llama_embedding_architecture_supported(_has_encoder, _has_decoder) &&
                      _embedding_dimensions > 0 && _pooling != LLAMA_POOLING_TYPE_RANK;
    caps.prompt_rendering = _has_decoder;
    return caps;
}

std::optional<Chorus::RequestRejection> LlamaScheduler::validate_embedding(const Chorus::ChorusRequest& request) const {
    if (!capabilities().embeddings)
        return Chorus::RequestRejection{Chorus::ChorusError::UnsupportedFeature, "The loaded model cannot produce embeddings."};
    if (request.prompt.empty())
        return Chorus::RequestRejection{Chorus::ChorusError::InvalidRequest, "Embedding prompts must not be empty."};
    const auto tokens = Chorus::LlamaUtils::tokenize_vocabulary(llama_model_get_vocab(model), request.prompt, true);
    if (!tokens || tokens->empty())
        return Chorus::RequestRejection{Chorus::ChorusError::Tokenize, "Tokenization failed (empty result)."};
    if (tokens->size() > static_cast<size_t>(_micro_batch_capacity))
        return Chorus::RequestRejection{Chorus::ChorusError::InvalidRequest, "Embedding prompt exceeds the physical micro-batch capacity."};
    return std::nullopt;
}

bool LlamaScheduler::push_request(Chorus::ChorusRequest req) {
    auto pending = std::make_shared<PendingRequest>();
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (!is_running)
            return false;
        pending->submission_sequence = _next_submission_sequence++;
        auto [delivery, inserted] = _terminal_deliveries.emplace(req.id, TerminalDelivery{
            req.on_event,
            {req.id, Chorus::ChorusSignal::Error{Chorus::ChorusError::Unknown, "Inference worker failed."}},
            req.priority, pending->submission_sequence,
        });
        if (!inserted)
            return false;
        try {
            pending->request = std::move(req);
            request_queue.push(std::move(pending));
        } catch (...) {
            _terminal_deliveries.erase(delivery);
            throw;
        }
    }
    queue_cv.notify_one();
    return true;
}

void LlamaScheduler::cancel_request(Chorus::RequestId id) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (!is_running || !_terminal_deliveries.contains(id))
            return;
        _cancel_requested.insert(id);
    }
    queue_cv.notify_one();
}

bool LlamaScheduler::has_active_exclusive() const {
    return std::ranges::any_of(_active_sequences, [](const auto& entry) {
        return entry.second.request.execution == Chorus::ExecutionMode::Exclusive;
    });
}

bool LlamaScheduler::process_control_requests() {
    std::vector<PendingSignal> signals;
    bool stopped = false;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stopped = !is_running;
        std::priority_queue<PendingRequestPtr, std::vector<PendingRequestPtr>, PendingRequestCompare> retained;
        while (!request_queue.empty()) {
            auto pending = request_queue.top();
            request_queue.pop();
            if (stopped || _cancel_requested.erase(pending->request.id)) {
                signals.emplace_back(std::move(pending->request), Chorus::ChorusSignal::Error{
                    Chorus::ChorusError::Cancelled, stopped ? "Request cancelled: engine stopped." : "Request cancelled."});
            } else {
                retained.push(std::move(pending));
            }
        }
        request_queue = std::move(retained);
    }
    std::vector<int> cancelled;
    for (int id : ordered_active_sequence_ids()) {
        const auto found = _active_sequences.find(id);
        if (found == _active_sequences.end())
            continue;
        bool request_stopped = false;
        if (take_cancellation(found->second.request.id, request_stopped))
            cancelled.push_back(id);
        stopped = stopped || request_stopped;
    }
    for (int id : cancelled)
        retire_sequence(id, Chorus::ChorusSignal::Error{Chorus::ChorusError::Cancelled, stopped ? "Request cancelled: engine stopped." : "Request cancelled."}, signals);
    emit_signals(std::move(signals));
    return stopped;
}

bool LlamaScheduler::take_cancellation(Chorus::RequestId id, bool& stopped) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    stopped = !is_running;
    return stopped || _cancel_requested.erase(id) > 0;
}

std::optional<LlamaScheduler::PendingSignal> LlamaScheduler::resolve_pending_request(PendingRequest& pending) {
    if (pending.request.type == Chorus::RequestType::Embedding)
        return std::nullopt;
    if (!pending.resolved) {
        auto resolution = Chorus::resolve_llama_generation(pending.request.gen_config);
        if (auto* rejection = std::get_if<Chorus::RequestRejection>(&resolution))
            return PendingSignal{pending.request, Chorus::ChorusSignal::Error{rejection->error, rejection->message}};
        pending.resolved.emplace(std::get<Chorus::ResolvedLlamaGeneration>(std::move(resolution)));
    }
    if (pending.resolved->max_tokens == 0)
        return PendingSignal{pending.request, Chorus::ChorusSignal::Stop{}};
    return std::nullopt;
}

LlamaScheduler::PreparedRequestResult LlamaScheduler::prepare_request(PendingRequest& pending) {
    PreparedRequest prepared;
    prepared.submission_sequence = pending.submission_sequence;
    if (pending.request.type == Chorus::RequestType::Embedding) {
        prepared.tokens = Chorus::LlamaUtils::tokenize(context, pending.request.prompt, true);
        if (prepared.tokens.empty())
            return Chorus::RequestRejection{Chorus::ChorusError::Tokenize, "Tokenization failed (empty result)."};
        if (prepared.tokens.size() > static_cast<size_t>(_micro_batch_capacity))
            return Chorus::RequestRejection{Chorus::ChorusError::InvalidRequest, "Embedding prompt exceeds the physical micro-batch capacity."};
        return prepared;
    }
    prepared.max_tokens = pending.resolved->max_tokens;
    prepared.stop_sequences = std::move(pending.resolved->stop);
    auto sampler = Chorus::make_llama_sampler(model, std::move(*pending.resolved));
    std::optional<Chorus::RequestRejection> render_rejection;
    if (!pending.request.messages.empty()) {
        std::lock_guard<std::mutex> lock(_template_mutex);
        auto rendered = Chorus::render_llama_chat(model, _model_default_chat_templates.get(), pending.request.chat_template,
                                                   pending.request.messages, pending.request.gen_config.show_thinking.value_or(true));
        if (auto* rejection = std::get_if<Chorus::RequestRejection>(&rendered))
            render_rejection = *rejection;
        else {
            auto& value = std::get<Chorus::LlamaChatRender>(rendered);
            prepared.tokens = Chorus::LlamaUtils::tokenize(context, value.prompt, true, true);
            for (auto& stop : value.template_stop_sequences)
                prepared.stop_sequences.push_back(std::move(stop));
            prepared.parse_stream = Chorus::make_llama_chat_parse_stream(value);
        }
    } else {
        prepared.tokens = Chorus::LlamaUtils::tokenize(context, pending.request.prompt, true);
    }
    if (render_rejection)
        return *render_rejection;
    if (auto* rejection = std::get_if<Chorus::RequestRejection>(&sampler))
        return *rejection;
    if (prepared.tokens.empty())
        return Chorus::RequestRejection{Chorus::ChorusError::Tokenize, "Tokenization failed (empty result)."};
    if (pending.request.exact_prompt_budget &&
        static_cast<int64_t>(prepared.tokens.size()) > *pending.request.exact_prompt_budget)
        return Chorus::RequestRejection{Chorus::ChorusError::InvalidRequest, "Provider rendering exceeds the prepared prompt budget."};
    prepared.sampler = std::get<common_sampler_ptr>(std::move(sampler));
    return prepared;
}

void LlamaScheduler::admit_available() {
    std::vector<PendingSignal> signals;
    while (true) {
        PendingRequestPtr pending;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!is_running || request_queue.empty() || has_active_exclusive())
                break;
            if (request_queue.top()->request.execution == Chorus::ExecutionMode::Exclusive && !_active_sequences.empty())
                break;
            pending = request_queue.top();
            request_queue.pop();
            if (_cancel_requested.erase(pending->request.id)) {
                signals.emplace_back(std::move(pending->request), Chorus::ChorusSignal::Error{Chorus::ChorusError::Cancelled, "Request cancelled."});
                continue;
            }
        }
#ifdef TEST_BUILD
        std::function<void(Chorus::RequestId)> observer;
        { std::lock_guard<std::mutex> lock(_batch_observer_mutex); observer = _admission_observer; }
        if (observer)
            observer(pending->request.id);
#endif
        if (auto terminal = resolve_pending_request(*pending)) {
            bool stopped = false;
            if (take_cancellation(pending->request.id, stopped))
                signals.emplace_back(std::move(pending->request), Chorus::ChorusSignal::Error{
                    Chorus::ChorusError::Cancelled, stopped ? "Request cancelled: engine stopped." : "Request cancelled."});
            else
                signals.push_back(std::move(*terminal));
            continue;
        }
        if (!pending->prepared) {
            auto prepared = prepare_request(*pending);
            if (auto* rejection = std::get_if<Chorus::RequestRejection>(&prepared)) {
                bool stopped = false;
                if (take_cancellation(pending->request.id, stopped))
                    signals.emplace_back(std::move(pending->request), Chorus::ChorusSignal::Error{
                        Chorus::ChorusError::Cancelled, stopped ? "Request cancelled: engine stopped." : "Request cancelled."});
                else
                    signals.emplace_back(std::move(pending->request), Chorus::ChorusSignal::Error{rejection->error, rejection->message});
                continue;
            }
            pending->prepared = std::make_shared<PreparedRequest>(std::get<PreparedRequest>(std::move(prepared)));
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!is_running || _cancel_requested.erase(pending->request.id)) {
                signals.emplace_back(std::move(pending->request), Chorus::ChorusSignal::Error{Chorus::ChorusError::Cancelled, "Request cancelled."});
                continue;
            }
            if (_sequence_ids.empty() || has_active_exclusive() ||
                (!request_queue.empty() && PendingRequestCompare{}(pending, request_queue.top()))) {
                request_queue.push(std::move(pending));
                break;
            }
            if (pending->request.execution == Chorus::ExecutionMode::Exclusive && !_active_sequences.empty()) {
                request_queue.push(std::move(pending));
                break;
            }
            const int id = *_sequence_ids.acquire();
            auto prepared = std::move(*pending->prepared);
            Sequence sequence;
            sequence.id = id;
            sequence.request = std::move(pending->request);
            sequence.submission_sequence = pending->submission_sequence;
            sequence.prompt_tokens = std::move(prepared.tokens);
            sequence.max_tokens = prepared.max_tokens;
            sequence.sampler = std::move(prepared.sampler);
            sequence.parse_stream = std::move(prepared.parse_stream);
            if (!prepared.stop_sequences.empty())
                sequence.stop_filter.emplace(std::move(prepared.stop_sequences));
            _active_sequences.emplace(id, std::move(sequence));
        }
    }
    emit_signals(std::move(signals));
}

std::vector<int> LlamaScheduler::ordered_active_sequence_ids() const {
    std::vector<int> ids;
    ids.reserve(_active_sequences.size());
    for (const auto& entry : _active_sequences)
        ids.push_back(entry.first);
    std::ranges::sort(ids, [this](int left, int right) {
        const Sequence& left_sequence = _active_sequences.at(left);
        const Sequence& right_sequence = _active_sequences.at(right);
        if (left_sequence.request.priority != right_sequence.request.priority)
            return left_sequence.request.priority > right_sequence.request.priority;
        return left_sequence.submission_sequence < right_sequence.submission_sequence;
    });
    return ids;
}

std::vector<LlamaScheduler::Sequence*> LlamaScheduler::ordered_runnable() const {
    std::vector<Sequence*> values;
    for (int id : ordered_active_sequence_ids()) {
        const auto found = _active_sequences.find(id);
        if (found == _active_sequences.end())
            continue;
        auto& sequence = const_cast<Sequence&>(found->second);
        if (sequence.request.type == Chorus::RequestType::Embedding || sequence.phase == GenerationPhase::Decode ||
            sequence.prompt_cursor < sequence.prompt_tokens.size())
            values.push_back(&sequence);
    }
    return values;
}

std::optional<LlamaScheduler::BatchPlan> LlamaScheduler::build_plan(int32_t generation_budget, int32_t embedding_budget) const {
    std::vector<Chorus::LlamaPlannerSequence> planner_sequences;
    for (const Sequence* sequence : ordered_runnable()) {
        planner_sequences.push_back({
            sequence->id,
            sequence->request.type,
            sequence->request.priority,
            sequence->submission_sequence,
            sequence->phase == GenerationPhase::Decode ? Chorus::LlamaPlannerPhase::Decode : Chorus::LlamaPlannerPhase::Prefill,
            sequence->prompt_cursor,
            sequence->prompt_tokens.size(),
            sequence->request.execution == Chorus::ExecutionMode::Exclusive,
        });
    }
    const auto planned = Chorus::llama_plan_batch(
        std::move(planner_sequences), generation_budget, embedding_budget, _decode_fairness_cursor,
        _has_contested_type ? std::optional<Chorus::RequestType>{_last_contested_type} : std::nullopt
    );
    if (!planned)
        return std::nullopt;

    BatchPlan plan;
    plan.type = planned->type;
    plan.contested_type = planned->contested_type;
    plan.decode_fairness = planned->decode_fairness;
    auto delta_for = [&plan](int id) -> SequenceDelta& {
        auto found = std::ranges::find(plan.deltas, id, &SequenceDelta::sequence_id);
        if (found == plan.deltas.end()) {
            plan.participants.push_back(id);
            plan.deltas.push_back({id});
            return plan.deltas.back();
        }
        return *found;
    };
    for (const auto& entry : planned->entries) {
        const auto found = _active_sequences.find(entry.sequence_id);
        if (found == _active_sequences.end())
            throw std::logic_error("planner selected an unknown sequence");
        const Sequence& sequence = found->second;
        auto& delta = delta_for(sequence.id);
        const int32_t token = entry.decode ? sequence.pending_token : sequence.prompt_tokens.at(entry.prompt_offset);
        const int32_t position = entry.decode ? sequence.n_past
                                               : sequence.n_past + static_cast<int32_t>(entry.prompt_offset - sequence.prompt_cursor);
        const bool logits = plan.type == Chorus::RequestType::Embedding && _pooling != LLAMA_POOLING_TYPE_NONE ? true : entry.logits;
        plan.entries.push_back({sequence.id, token, position, logits});
        ++delta.kv_advance;
        if (!entry.decode)
            ++delta.prompt_advance;
        if (logits) {
            delta.sampled = plan.type == Chorus::RequestType::Generate;
            if (plan.type == Chorus::RequestType::Embedding && _pooling == LLAMA_POOLING_TYPE_NONE)
                delta.embedding_output_index = static_cast<int32_t>(plan.entries.size() - 1);
        }
    }
    return plan;
}

void LlamaScheduler::populate_batch(const BatchPlan& plan) {
    auto& value = batch.get();
    value.n_tokens = 0;
    for (const auto& entry : plan.entries)
        Chorus::LlamaUtils::batch_add_seq(value, entry.token, entry.sequence_id, entry.position, entry.logits);
}

int LlamaScheduler::run_inference(const BatchPlan& plan) {
#ifdef TEST_BUILD
    std::function<void(const Chorus::LlamaBatchRecord&)> observer;
    { std::lock_guard<std::mutex> lock(_batch_observer_mutex); observer = _batch_observer; }
    if (observer) {
        Chorus::LlamaBatchRecord record{plan.type, static_cast<int32_t>(plan.entries.size()), {}, {}};
        for (const auto& delta : plan.deltas) {
            const auto& sequence = _active_sequences.at(delta.sequence_id);
            record.request_ids.push_back(sequence.request.id);
            if (plan.type == Chorus::RequestType::Embedding)
                record.embedding_output_indices.push_back(delta.embedding_output_index);
        }
        observer(record);
    }
#endif
    llama_set_embeddings(context, plan.type == Chorus::RequestType::Embedding);
    const int rc = llama_decode(context, batch.get());
    if (rc != 0)
        _log.error("Decode failed", {{"code", (int64_t)rc}});
    return rc;
}

void LlamaScheduler::commit_plan(const BatchPlan& plan) {
    for (const auto& delta : plan.deltas) {
        auto found = _active_sequences.find(delta.sequence_id);
        if (found == _active_sequences.end())
            throw std::logic_error("batch participant disappeared before commit");
        found->second.prompt_cursor += delta.prompt_advance;
        found->second.n_past += delta.kv_advance;
    }
    if (plan.contested_type) {
        _last_contested_type = *plan.contested_type;
        _has_contested_type = true;
    }
    if (plan.decode_fairness)
        _decode_fairness_cursor = *plan.decode_fairness;
}

void LlamaScheduler::process_generation_plan(const BatchPlan& plan) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    for (size_t index = 0; index < plan.entries.size(); ++index) {
        if (!plan.entries[index].logits)
            continue;
        auto found = _active_sequences.find(plan.entries[index].sequence_id);
        if (found == _active_sequences.end())
            continue;
        Sequence& sequence = found->second;
        llama_token token = common_sampler_sample(sequence.sampler.get(), context, static_cast<int>(index));
        common_sampler_accept(sequence.sampler.get(), token, true);
        ++sequence.n_decoded;
        if (llama_vocab_is_eog(vocab, token)) {
            complete_sequence(sequence.id, true);
            continue;
        }
        std::string piece = Chorus::LlamaUtils::token_to_piece(context, token);
        if (sequence.parse_stream) {
            auto delta = sequence.parse_stream->push(piece);
            if (!delta.reasoning.empty())
                emit_token(sequence, sequence.reasoning_chunker.push(delta.reasoning), Chorus::TokenChannel::Reasoning);
            piece = std::move(delta.content);
        }
        bool stopped = false;
        if (sequence.stop_filter) {
            auto filtered = sequence.stop_filter->push(piece);
            emit_token(sequence, std::move(filtered.safe_text));
            stopped = filtered.matched;
        } else {
            emit_token(sequence, sequence.content_chunker.push(piece));
        }
        if (stopped || (sequence.max_tokens > 0 && sequence.n_decoded >= sequence.max_tokens)) {
            complete_sequence(sequence.id, !stopped);
            continue;
        }
        sequence.phase = GenerationPhase::Decode;
        sequence.pending_token = token;
    }
}

void LlamaScheduler::process_embedding_plan(const BatchPlan& plan) {
    std::vector<PendingSignal> signals;
    for (const auto& delta : plan.deltas) {
        auto found = _active_sequences.find(delta.sequence_id);
        if (found == _active_sequences.end())
            continue;
        const float* source = _pooling == LLAMA_POOLING_TYPE_NONE ? llama_get_embeddings_ith(context, delta.embedding_output_index)
                                                                   : llama_get_embeddings_seq(context, delta.sequence_id);
        std::vector<float> values;
        bool valid = source && _embedding_dimensions > 0;
        double squared_norm = 0.0;
        if (valid) {
            values.assign(source, source + _embedding_dimensions);
            for (float value : values) {
                if (!std::isfinite(value)) { valid = false; break; }
                squared_norm += static_cast<double>(value) * value;
            }
            valid = valid && std::isfinite(squared_norm) && squared_norm > 0.0;
        }
        if (valid) {
            const double norm = std::sqrt(squared_norm);
            for (float& value : values)
                value = static_cast<float>(value / norm);
        }
        bool cancelled;
        { std::lock_guard<std::mutex> lock(queue_mutex); cancelled = _cancel_requested.erase(found->second.request.id) > 0 || !is_running; }
        if (cancelled)
            retire_sequence(delta.sequence_id, Chorus::ChorusSignal::Error{Chorus::ChorusError::Cancelled, "Request cancelled."}, signals);
        else if (!valid)
            retire_sequence(delta.sequence_id, Chorus::ChorusSignal::Error{Chorus::ChorusError::Decode, "Embedding output was missing or invalid."}, signals);
        else {
            const Chorus::ChorusRequest request = found->second.request;
            signals.emplace_back(request, Chorus::ChorusSignal::Embedding{std::move(values)});
            retire_sequence(delta.sequence_id, Chorus::ChorusSignal::Stop{}, signals);
        }
    }
    emit_signals(std::move(signals));
}

std::optional<LlamaScheduler::BatchPlan>
LlamaScheduler::recovery_plan(const BatchPlan& failed, const Chorus::LlamaRecoverySelection& selection) const {
    if (selection.sequence_ids.empty() || selection.sequence_ids.size() != selection.entry_limits.size())
        return std::nullopt;

    std::map<int, size_t> limits;
    for (size_t index = 0; index < selection.sequence_ids.size(); ++index) {
        if (!_active_sequences.contains(selection.sequence_ids[index]) || !limits.emplace(selection.sequence_ids[index], selection.entry_limits[index]).second)
            return std::nullopt;
    }

    BatchPlan retry;
    retry.type = failed.type;
    retry.contested_type = failed.contested_type;
    retry.decode_fairness = failed.decode_fairness;
    for (int id : selection.sequence_ids) {
        retry.participants.push_back(id);
        retry.deltas.push_back({id});
    }
    std::map<int, size_t> selected;
    for (const auto& entry : failed.entries) {
        auto limit = limits.find(entry.sequence_id);
        if (limit == limits.end() || selected[entry.sequence_id] >= limit->second)
            continue;
        ++selected[entry.sequence_id];
        auto delta = std::ranges::find(retry.deltas, entry.sequence_id, &SequenceDelta::sequence_id);
        if (delta == retry.deltas.end())
            throw std::logic_error("recovery entry has no contributor");
        const Sequence& sequence = _active_sequences.at(entry.sequence_id);
        ++delta->kv_advance;
        if (failed.type == Chorus::RequestType::Embedding || sequence.phase == GenerationPhase::Prefill)
            ++delta->prompt_advance;
        if (entry.logits) {
            delta->sampled = failed.type == Chorus::RequestType::Generate;
            if (failed.type == Chorus::RequestType::Embedding && _pooling == LLAMA_POOLING_TYPE_NONE)
                delta->embedding_output_index = static_cast<int32_t>(retry.entries.size());
        }
        retry.entries.push_back(entry);
    }
    return retry.entries.empty() ? std::nullopt : std::optional<BatchPlan>{std::move(retry)};
}

std::string LlamaScheduler::recovery_capacity_message() const {
    const uint32_t effective_context = _model_info && _model_info->per_request_context
                                           ? *_model_info->per_request_context
                                           : static_cast<uint32_t>(llama_n_ctx(context) / _max_concurrent_requests);
    return "Inference capacity is exhausted for this request at max_concurrent_requests=" +
           std::to_string(_max_concurrent_requests) + " with effective per-request context " +
           std::to_string(effective_context) + ". Reduce max_concurrent_requests or increase context_size.";
}

bool LlamaScheduler::recover_decode(const BatchPlan& failed) {
    std::vector<Chorus::LlamaRecoveryContribution> contributors;
    for (int id : failed.participants) {
        const auto sequence = _active_sequences.find(id);
        if (sequence == _active_sequences.end())
            return false;
        const size_t entries = static_cast<size_t>(std::ranges::count(failed.entries, id, &BatchEntry::sequence_id));
        const size_t minimum_entries = failed.type == Chorus::RequestType::Embedding ? entries : size_t{1};
        contributors.push_back({id, entries, minimum_entries});
    }
    for (const auto& selection : Chorus::llama_recovery_reductions(contributors)) {
        auto retry = recovery_plan(failed, selection);
        if (!retry)
            return false;
        populate_batch(*retry);
        const int rc = run_inference(*retry);
        if (rc == 0) {
            commit_plan(*retry);
            if (retry->type == Chorus::RequestType::Embedding)
                process_embedding_plan(*retry);
            else
                process_generation_plan(*retry);
            return true;
        }
        if (rc != 1)
            return false;
    }
    std::vector<PendingSignal> signals;
    if (!failed.participants.empty()) {
        retire_sequence(failed.participants.front(), Chorus::ChorusSignal::Error{Chorus::ChorusError::Decode,
                                                                                   recovery_capacity_message()}, signals);
    }
    emit_signals(std::move(signals));
    return true;
}

void LlamaScheduler::worker_loop() {
    try {
        run_worker();
    } catch (...) {
        fail_all(Chorus::ChorusError::Unknown);
    }
}

void LlamaScheduler::run_worker() {
    while (true) {
#ifdef TEST_BUILD
        ++_worker_iterations;
#endif
        if (process_control_requests())
            return;
        admit_available();
        if (process_control_requests())
            return;
        auto plan = build_plan(_batch_capacity, _micro_batch_capacity);
        if (!plan) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait_for(lock, std::chrono::milliseconds(10), [this] {
                return !is_running || !request_queue.empty() || !_cancel_requested.empty();
            });
            continue;
        }
        populate_batch(*plan);
        const int rc = run_inference(*plan);
        if (rc == 0) {
            commit_plan(*plan);
            if (plan->type == Chorus::RequestType::Embedding)
                process_embedding_plan(*plan);
            else
                process_generation_plan(*plan);
            continue;
        }
        if (rc == 1 && recover_decode(*plan))
            continue;
        fail_all(Chorus::ChorusError::Decode);
        return;
    }
}

void LlamaScheduler::retire_sequence(int sequence_id, Chorus::ChorusSignal::Event event, std::vector<PendingSignal>& signals) {
    auto found = _active_sequences.find(sequence_id);
    if (found == _active_sequences.end())
        return;
    Chorus::ChorusRequest request = std::move(found->second.request);
    found->second.sampler.reset();
    found->second.stop_filter.reset();
    found->second.parse_stream.reset();
    found->second.reasoning_chunker.reset();
    found->second.content_chunker.reset();
    if (context)
        llama_memory_seq_rm(llama_get_memory(context), sequence_id, 0, -1);
    _active_sequences.erase(found);
    _sequence_ids.release(sequence_id);
    signals.emplace_back(std::move(request), std::move(event));
}

void LlamaScheduler::complete_sequence(int sequence_id, bool flush_pending_text) {
    auto found = _active_sequences.find(sequence_id);
    if (found == _active_sequences.end())
        return;
    Sequence& sequence = found->second;
    Chorus::LlamaChatParseStream::Delta residual;
    if (sequence.parse_stream) {
        residual = sequence.parse_stream->finalize();
        residual.reasoning = sequence.reasoning_chunker.push(residual.reasoning);
    }
    std::vector<PendingSignal> signals;
    bool cancelled;
    { std::lock_guard<std::mutex> lock(queue_mutex); cancelled = _cancel_requested.erase(sequence.request.id) > 0 || !is_running; }
    if (cancelled) {
        retire_sequence(sequence_id, Chorus::ChorusSignal::Error{Chorus::ChorusError::Cancelled, "Request cancelled."}, signals);
    } else {
        if (!residual.reasoning.empty())
            signals.emplace_back(sequence.request, Chorus::ChorusSignal::Token{Chorus::TokenChannel::Reasoning, std::move(residual.reasoning)});
        if (flush_pending_text) {
            auto filtered = Chorus::finish_content_stream(sequence.stop_filter ? &*sequence.stop_filter : nullptr,
                                                           sequence.content_chunker, residual.content);
            if (!filtered.safe_text.empty())
                signals.emplace_back(sequence.request, Chorus::ChorusSignal::Token{Chorus::TokenChannel::Content, std::move(filtered.safe_text)});
        }
        retire_sequence(sequence_id, Chorus::ChorusSignal::Stop{}, signals);
    }
    emit_signals(std::move(signals));
}

void LlamaScheduler::fail_all(Chorus::ChorusError code) {
    decltype(_terminal_deliveries) deliveries;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        is_running = false;
        deliveries.swap(_terminal_deliveries);
        _cancel_requested.clear();
    }
    queue_cv.notify_all();
    // Do not attempt KV cleanup or resume inference after an unexpected worker failure.
    while (!deliveries.empty()) {
        auto next = std::max_element(deliveries.begin(), deliveries.end(), [](const auto& left, const auto& right) {
            if (left.second.priority != right.second.priority)
                return left.second.priority < right.second.priority;
            return left.second.submission_sequence > right.second.submission_sequence;
        });
        auto delivery = deliveries.extract(next);
        std::get<Chorus::ChorusSignal::Error>(delivery.mapped().failure.event).code = code;
        try {
            if (delivery.mapped().on_event)
                delivery.mapped().on_event(delivery.mapped().failure);
        } catch (...) {
            // A contract-violating sink must not suppress other requests' terminal callbacks.
        }
    }
}

void LlamaScheduler::emit_signal(PendingSignal pending) {
    if (std::holds_alternative<Chorus::ChorusSignal::Stop>(pending.event) ||
        std::holds_alternative<Chorus::ChorusSignal::Error>(pending.event)) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (!_terminal_deliveries.erase(pending.request.id))
            return;
        _cancel_requested.erase(pending.request.id);
    }
    if (!pending.request.on_event)
        return;
    Chorus::ChorusSignal signal{pending.request.id, std::move(pending.event)};
    pending.request.on_event(signal);
}
void LlamaScheduler::emit_signals(std::vector<PendingSignal> pending) {
    for (auto& signal : pending)
        emit_signal(std::move(signal));
}
void LlamaScheduler::emit_token(Sequence& sequence, std::string text, Chorus::TokenChannel channel) {
    if (text.empty() || !sequence.request.on_event)
        return;
    Chorus::ChorusSignal signal{sequence.request.id, Chorus::ChorusSignal::Token{channel, std::move(text)}};
    sequence.request.on_event(signal);
}
#ifdef TEST_BUILD
void LlamaScheduler::set_batch_observer(std::function<void(const Chorus::LlamaBatchRecord&)> observer) {
    std::lock_guard<std::mutex> lock(_batch_observer_mutex);
    _batch_observer = std::move(observer);
}
void LlamaScheduler::set_admission_observer(std::function<void(Chorus::RequestId)> observer) {
    std::lock_guard<std::mutex> lock(_batch_observer_mutex);
    _admission_observer = std::move(observer);
}
#endif

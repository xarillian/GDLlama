#include "chorus/runtime/runtime.hpp"

#include <cassert>

namespace Chorus {

ChorusRuntime::~ChorusRuntime() {
    // Pending events die with us; nobody can poll a destroyed runtime.
    unload_engine();
}

std::optional<InitializationFailure>
ChorusRuntime::load_engine(std::unique_ptr<InferenceEngine> engine, const ChorusConfig& config) {
    assert_host_thread();
    if (!engine)
        return InitializationFailure{ChorusError::InvalidRequest, "An inference engine is required."};

    if (_engine) {
        unload_engine();
        cancel_live_requests();
    }

    // Loggers share the channel's lifetime, so provider logging cannot outlive its sink.
    Logger logger(_log_channel, config.log_level);
    auto err = engine->initialize(config, std::move(logger));
    if (err.has_value())
        return err; // engine destroyed on scope exit; runtime stays unloaded

    _engine = std::move(engine);
    _engine_failure_reported = false;
    return std::nullopt;
}

bool ChorusRuntime::is_loaded() const {
    return _engine && _engine->is_initialized();
}

std::optional<LoadedModelInfo> ChorusRuntime::loaded_model_info() const {
    assert_host_thread();
    return is_loaded() ? _engine->loaded_model_info() : std::nullopt;
}

std::optional<EngineCapabilities> ChorusRuntime::capabilities() const {
    assert_host_thread();
    if (!is_loaded())
        return std::nullopt;
    return _engine->capabilities();
}

void ChorusRuntime::set_host_defaults(HostDefaults defaults) {
    assert_host_thread();
    _host_defaults = std::move(defaults);
}

ChorusRuntime::ResolvedRequest ChorusRuntime::resolve_request(const GenerationRequest& request) const {
    ResolvedRequest resolved{
        request,
        apply_generation_patch(apply_generation_patch(GenerationConfig{}, _host_defaults.config), request.overrides),
        request.chat_template
    };

    if (!request.session_id) {
        // If it's a stateless, one-off request, don't auto-apply the host's default chat settings
        // (like 'show_thinking'). However, if the caller explicitly requested it anyway,
        // we leave it in the config so the validation step below can rightfully reject it.
        if (request.overrides.show_thinking.action == PatchAction::Inherit)
            resolved.config.show_thinking.reset();
    } else if (resolved.chat_template.empty()) {
        resolved.chat_template = _host_defaults.chat_template;
    }
    return resolved;
}

ChorusRequest ChorusRuntime::make_engine_request(const ResolvedRequest& resolved) {
    ChorusRequest engine_request;
    engine_request.session_id = resolved.request.session_id;
    engine_request.priority = resolved.request.priority;
    engine_request.execution = resolved.request.execution;
    engine_request.prompt = resolved.request.prompt;
    engine_request.gen_config = resolved.config;
    engine_request.chat_template = resolved.chat_template;
    return engine_request;
}

SubmitResult ChorusRuntime::not_ready() const {
    return _engine ? SubmitResult{-1, ChorusError::EngineNotReady, "The engine has failed; load it again."}
                   : SubmitResult{-1, ChorusError::EngineNotReady, "No engine is loaded."};
}

SubmitResult ChorusRuntime::submit(const GenerationRequest& request) {
    assert_host_thread();
    if (!is_loaded())
        return not_ready();
    if (request.session_id && request.session_id->empty())
        return SubmitResult{
            -1,
            ChorusError::InvalidRequest,
            "session_id must be non-empty when present; omit it for stateless requests."
        };
    if (request.session_id && _request_by_session.contains(*request.session_id))
        return SubmitResult{
            -1, ChorusError::SessionBusy, "Session '" + *request.session_id + "' already has a live request."
        };

    const ResolvedRequest resolved = resolve_request(request);

    // We should not silently discard. If this is a stateless request
    // but they're trying to sneak in chat-specific features, don't just ignore the features.
    // Bounce the whole request so the developer knows their payload is flawed.
    if (!request.session_id &&
        (!request.inject.empty() || !resolved.chat_template.empty() || resolved.config.show_thinking.has_value()))
        return SubmitResult{
            -1,
            ChorusError::InvalidRequest,
            "inject/chat_template/show_thinking are chat controls; they require a session."
        };

    ChorusRequest engine_request = make_engine_request(resolved);

    if (request.session_id) {
        if (request.prompt.empty())
            return SubmitResult{-1, ChorusError::InvalidRequest, "Chat turns require a non-empty prompt."};
        std::vector<ChatMessage> prospective;
        if (auto it = _histories.find(*request.session_id); it != _histories.end())
            prospective = it->second.messages;
        prospective.push_back({"user", request.prompt});
        auto fitted = fit_turn_messages(resolved, std::move(prospective));
        if (std::holds_alternative<SubmitResult>(fitted))
            return std::get<SubmitResult>(fitted);
        auto& turn = std::get<FittedTurn>(fitted);
        engine_request.messages = std::move(turn.messages);
        return submit_engine_request(resolved, std::move(engine_request), turn.dropped);
    }

    return submit_engine_request(resolved, std::move(engine_request));
}

SubmitResult ChorusRuntime::submit(const EmbeddingRequest& request) {
    assert_host_thread();
    if (!is_loaded())
        return not_ready();

    const RequestId id = _next_request_id++;
    ChorusRequest engine_request;
    engine_request.id = id;
    engine_request.type = RequestType::Embedding;
    engine_request.prompt = request.prompt;
    engine_request.priority = request.priority;
    engine_request.execution = request.execution;
    engine_request.on_event = [this](ChorusSignal& sig) { enqueue_signal(sig); };

    if (auto rejection = _engine->validate_request(engine_request))
        return SubmitResult{-1, rejection->error, rejection->message};

    LiveRequest live;
    live.type = RequestType::Embedding;
    _live_requests.emplace(id, std::move(live));
    _engine->submit_request(engine_request);
    return SubmitResult{id, ChorusError::None, ""};
}

SubmitResult ChorusRuntime::submit_engine_request(
    const ResolvedRequest& resolved,
    ChorusRequest engine_request,
    int32_t dropped,
    std::optional<ChatMessage> replaced_reply
) {
    const GenerationRequest& request = resolved.request;
    const RequestId id = _next_request_id++;
    engine_request.id = id;
    engine_request.on_event = [this](ChorusSignal& sig) { enqueue_signal(sig); };

    if (auto rejection = _engine->validate_request(engine_request))
        return SubmitResult{-1, rejection->error, rejection->message};

    const bool is_regenerate = replaced_reply.has_value();
    LiveRequest live;
    live.type = RequestType::Generate;
    live.streaming = request.stream;
    live.session_id = request.session_id;
    live.replaced_reply = std::move(replaced_reply);
    _live_requests.emplace(id, std::move(live));
    if (request.session_id) {
        _request_by_session.emplace(*request.session_id, id);
        if (is_regenerate)
            _histories[*request.session_id].messages.pop_back();
        else
            _histories[*request.session_id].messages.push_back({"user", request.prompt});
        if (dropped > 0) {
            RuntimeEvent event{id, request.session_id, RuntimeEvent::Kind::HistoryTruncated, "", ChorusError::None};
            event.dropped = dropped;
            _host_events.push_back(std::move(event));
        }
    }
    _engine->submit_request(engine_request);
    return SubmitResult{id, ChorusError::None, ""};
}

bool ChorusRuntime::cancel(RequestId id) {
    assert_host_thread();
    if (_live_requests.find(id) == _live_requests.end())
        return false;
    if (_engine)
        _engine->cancel_request(id);
    return true;
}

bool ChorusRuntime::is_request_active(RequestId id) const {
    assert_host_thread();
    return _live_requests.find(id) != _live_requests.end();
}

std::optional<RequestId> ChorusRuntime::active_request_for_session(const SessionId& session_id) const {
    assert_host_thread();
    const auto request = _request_by_session.find(session_id);
    return request == _request_by_session.end() ? std::nullopt : std::optional<RequestId>{request->second};
}

std::vector<RuntimeEvent> ChorusRuntime::poll() {
    assert_host_thread();
    std::vector<ChorusSignal> signals = drain_pending_signals();
    std::vector<RuntimeEvent> events = std::move(_host_events);
    _host_events.clear();

    for (const auto& signal : signals)
        append_signal_events(signal, events);
    append_engine_failure(events);
    return events;
}

std::vector<ChorusSignal> ChorusRuntime::drain_pending_signals() {
    std::vector<ChorusSignal> signals;
    std::lock_guard<std::mutex> lock(_pending_mutex);
    signals.swap(_pending_signals);
    return signals;
}

void ChorusRuntime::append_signal_events(const ChorusSignal& signal, std::vector<RuntimeEvent>& events) {
    const RequestId id = signal.request_id;
    auto request = _live_requests.find(id);
    if (request == _live_requests.end())
        return;

    LiveRequest& live = request->second;
    if (const auto* embedding = std::get_if<ChorusSignal::Embedding>(&signal.event)) {
        if (live.type != RequestType::Embedding) {
            finish_turn(live, TurnOutcome::Errored, "");
            events.push_back({id, live.session_id, RuntimeEvent::Kind::Error, "Provider emitted an embedding for a generation request.", ChorusError::Unknown});
            retire_request(id);
            return;
        }
        live.embedding = embedding->values;
        return;
    }

    if (const auto* token = std::get_if<ChorusSignal::Token>(&signal.event)) {
        if (live.type != RequestType::Generate) {
            finish_turn(live, TurnOutcome::Errored, "");
            events.push_back({id, live.session_id, RuntimeEvent::Kind::Error, "Provider emitted a token for an embedding request.", ChorusError::Unknown});
            retire_request(id);
            return;
        }
        if (token->channel == TokenChannel::Reasoning) {
            live.accumulated_reasoning += token->text;
            if (live.streaming)
                events.push_back(
                    {id, live.session_id, RuntimeEvent::Kind::StreamedReasoningToken, token->text, ChorusError::None}
                );
        } else {
            live.accumulated_text += token->text;
            if (live.streaming)
                events.push_back(
                    {id, live.session_id, RuntimeEvent::Kind::StreamedToken, token->text, ChorusError::None}
                );
        }
        return;
    }

    if (std::holds_alternative<ChorusSignal::Stop>(signal.event)) {
        if (live.type == RequestType::Embedding) {
            if (!live.embedding) {
                events.push_back({id, std::nullopt, RuntimeEvent::Kind::Error, "Provider stopped an embedding request without a vector.", ChorusError::Unknown});
            } else {
                RuntimeEvent event{id, std::nullopt, RuntimeEvent::Kind::Embedding, "", ChorusError::None};
                event.embedding = std::move(*live.embedding);
                events.push_back(std::move(event));
            }
            retire_request(id);
            return;
        }

        finish_turn(live, TurnOutcome::Completed, live.accumulated_text);
        RuntimeEvent event{
            id, live.session_id, RuntimeEvent::Kind::Complete, std::move(live.accumulated_text), ChorusError::None
        };
        event.reasoning = std::move(live.accumulated_reasoning);
        events.push_back(std::move(event));
        retire_request(id);
        return;
    }

    if (const auto* error = std::get_if<ChorusSignal::Error>(&signal.event)) {
        finish_turn(live, error->code == ChorusError::Cancelled ? TurnOutcome::Cancelled : TurnOutcome::Errored, "");
        events.push_back({id, live.session_id, RuntimeEvent::Kind::Error, error->message, error->code});
        retire_request(id);
        return;
    }
}

void ChorusRuntime::append_engine_failure(std::vector<RuntimeEvent>& events) {
    if (!_engine || _engine_failure_reported || _engine->is_initialized())
        return;

    _engine_failure_reported = true;
    events.push_back(
        {-1, std::nullopt, RuntimeEvent::Kind::EngineFailed, "The engine has failed.", ChorusError::EngineNotReady}
    );
}

std::vector<LogRecord> ChorusRuntime::poll_logs() {
    assert_host_thread();
    return _log_channel->drain();
}

void ChorusRuntime::stop_all() {
    assert_host_thread();
    unload_engine();
    cancel_live_requests();
}

void ChorusRuntime::unload_engine() {
    if (_engine) {
        // Port contract: after shutdown() returns, the engine never invokes a
        // previously supplied on_event again.
        _engine->shutdown();
        _engine.reset();
    }
    // An unload is not a death, and there is no longer an engine to report on.
    _engine_failure_reported = false;
}

void ChorusRuntime::cancel_live_requests() {
    if (_live_requests.empty())
        return;
    std::lock_guard<std::mutex> lock(_pending_mutex);
    for (const auto& request : _live_requests) {
        _pending_signals.emplace_back(
            request.first, ChorusSignal::Error{ChorusError::Cancelled, "Request cancelled: engine stopped."}
        );
    }
}

void ChorusRuntime::retire_request(RequestId id) {
    const auto request = _live_requests.find(id);
    if (request == _live_requests.end())
        return;
    if (request->second.session_id)
        _request_by_session.erase(*request->second.session_id);
    _live_requests.erase(request);
}

void ChorusRuntime::enqueue_signal(const ChorusSignal& signal) {
    std::lock_guard<std::mutex> lock(_pending_mutex);
    _pending_signals.push_back(signal);
}

void ChorusRuntime::assert_host_thread() const {
    // Keep the field in every build so class layout cannot differ across translation units.
#ifndef NDEBUG
    std::thread::id expected{};
    _host_thread.compare_exchange_strong(expected, std::this_thread::get_id());
    assert(_host_thread.load() == std::this_thread::get_id() && "ChorusRuntime public methods are host-thread-only");
#endif
}

} // namespace Chorus

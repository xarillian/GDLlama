#include "chorus/runtime/runtime.hpp"

#include <cassert>

namespace Chorus {

ChorusRuntime::~ChorusRuntime() {
    // Pending events die with us; nobody can poll a destroyed runtime.
    unload_engine();
}

std::optional<ChorusError>
ChorusRuntime::load_engine(std::unique_ptr<InferenceEngine> engine, const ChorusConfig& config) {
    assert_host_thread();
    if (!engine)
        return ChorusError::InvalidRequest;

    if (_engine) {
        unload_engine();
        cancel_live_requests();
    }

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
        // Ambient chat controls are inapplicable here, not erroneous: a host's
        // node-level show_thinking default must not make a raw-prompt call fail.
        // A request that named the control itself keeps it, and meets the
        // rejection below.
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

    // Chat-only controls are meaningless on a stateless raw-prompt request and
    // must not be silently discarded (project no-silent-discard rule). By this
    // point only controls the caller set deliberately survive.
    if (!request.session_id &&
        (!request.inject.empty() || !resolved.chat_template.empty() || resolved.config.show_thinking.has_value()))
        return SubmitResult{
            -1, ChorusError::InvalidRequest, "inject/chat_template/show_thinking are chat controls; they require a session."
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

    // Live state exists before the engine sees the request: engines may invoke
    // on_event inline during submit_request. The durable history commit obeys
    // the same invariant -- the turn's mutation must exist before dispatch so
    // an inline terminal finds it.
    const bool is_regenerate = replaced_reply.has_value();
    _live_requests.emplace(id, LiveRequest{request.stream, {}, request.session_id, {}, std::move(replaced_reply)});
    if (request.session_id) {
        _request_by_session.emplace(*request.session_id, id);
        if (is_regenerate)
            _histories[*request.session_id].messages.pop_back(); // the reply being rerolled
        else
            _histories[*request.session_id].messages.push_back({"user", request.prompt});
        if (dropped > 0) {
            // Fitted copy only: durable history keeps every message. The host
            // still hears about the truncation on its next poll().
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
    std::vector<ChorusSignal> batch;
    {
        std::lock_guard<std::mutex> lock(_pending_mutex);
        batch.swap(_pending_signals);
    }

    std::vector<RuntimeEvent> events;
    if (!_host_events.empty()) {
        events = std::move(_host_events);
        _host_events.clear();
    }
    for (const auto& sig : batch) {
        const int64_t id = sig.request_id;
        auto request = _live_requests.find(id);
        if (request == _live_requests.end())
            continue; // no live state: late token, duplicate terminal, or unknown id

        LiveRequest& live = request->second;

        if (const auto* token = std::get_if<ChorusSignal::Token>(&sig.event)) {
            if (token->channel == TokenChannel::Reasoning) {
                live.accumulated_reasoning += token->text;
                if (live.streaming)
                    events.push_back(
                        {id, live.session_id, RuntimeEvent::Kind::ReasoningToken, token->text, ChorusError::None}
                    );
            } else {
                live.accumulated_text += token->text;
                if (live.streaming)
                    events.push_back({id, live.session_id, RuntimeEvent::Kind::Token, token->text, ChorusError::None});
            }
            continue;
        }

        if (std::holds_alternative<ChorusSignal::Stop>(sig.event)) {
            // finish_turn first, while live is untouched; the moves below gut it.
            finish_turn(live, TurnOutcome::Completed, live.accumulated_text);
            RuntimeEvent event{
                id, live.session_id, RuntimeEvent::Kind::Complete, std::move(live.accumulated_text), ChorusError::None
            };
            event.reasoning = std::move(live.accumulated_reasoning);
            events.push_back(std::move(event));
            retire_request(id);
            continue;
        }

        if (const auto* error = std::get_if<ChorusSignal::Error>(&sig.event)) {
            finish_turn(
                live, error->code == ChorusError::Cancelled ? TurnOutcome::Cancelled : TurnOutcome::Errored, ""
            );
            events.push_back({id, live.session_id, RuntimeEvent::Kind::Error, error->message, error->code});
            retire_request(id);
        }
        // `ChorusSignal::Embedding` has no runtime consumer yet.
    }

    // Last, so the terminals of the requests that died with the engine lead the
    // batch. An engine that dies on its own announces nothing, so this is the
    // only moment the host can learn of it.
    if (_engine && !_engine_failure_reported && !_engine->is_initialized()) {
        _engine_failure_reported = true;
        RuntimeEvent failure{
            -1, std::nullopt, RuntimeEvent::Kind::EngineFailed, "The engine has failed.", ChorusError::EngineNotReady
        };
        events.push_back(std::move(failure));
    }
    return events;
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
    // The engine is already stopped (contract: no further callbacks), but the
    // mutex keeps this correct even if a future caller reorders.
    std::lock_guard<std::mutex> lock(_pending_mutex);
    for (const auto& request : _live_requests) {
        _pending_signals.emplace_back(
            request.first,
            ChorusSignal::Error{ChorusError::Cancelled, "Request cancelled: engine stopped."}
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
#ifndef NDEBUG
    std::thread::id expected{};
    _host_thread.compare_exchange_strong(expected, std::this_thread::get_id());
    assert(_host_thread.load() == std::this_thread::get_id() && "ChorusRuntime public methods are host-thread-only");
#endif
}

} // namespace Chorus

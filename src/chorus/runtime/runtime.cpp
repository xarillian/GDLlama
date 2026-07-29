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

    auto err = engine->initialize(config);
    if (err.has_value())
        return err; // engine destroyed on scope exit; runtime stays unloaded

    _engine = std::move(engine);
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
        // node-level thinking default must not make a raw-prompt call fail.
        // A request that named the control itself keeps it, and meets the
        // rejection below.
        if (request.overrides.thinking.action == PatchAction::Inherit)
            resolved.config.thinking.reset();
    } else if (resolved.chat_template.empty()) {
        resolved.chat_template = _host_defaults.chat_template;
    }
    return resolved;
}

SubmitResult ChorusRuntime::submit(const GenerationRequest& request) {
    assert_host_thread();
    if (!is_loaded())
        return SubmitResult{-1, ChorusError::EngineNotReady, "No engine is loaded."};
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
        (!request.inject.empty() || !resolved.chat_template.empty() || resolved.config.thinking.has_value()))
        return SubmitResult{
            -1, ChorusError::InvalidRequest, "inject/chat_template/thinking are chat controls; they require a session."
        };

    ChorusRequest engine_request;
    engine_request.session_id = request.session_id;
    engine_request.priority = request.priority;
    engine_request.prompt = request.prompt;
    engine_request.gen_config = resolved.config;
    engine_request.chat_template = resolved.chat_template;

    if (request.session_id) {
        // Chat turn (#5): presence of a session means continuation. Build the
        // prospective message list WITHOUT mutating the store (find, never
        // operator[] -- a rejection must leave no trace, not even an empty
        // lane in list_conversations()); commit happens on acceptance.
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
    const int64_t id = _next_request_id.fetch_add(1);
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

        switch (sig.type) {
        case EventType::Token:
            if (sig.channel == TokenChannel::Reasoning) {
                live.accumulated_reasoning += sig.text;
                if (live.streaming)
                    events.push_back(
                        {id, live.session_id, RuntimeEvent::Kind::ReasoningToken, sig.text, ChorusError::None}
                    );
            } else {
                live.accumulated_text += sig.text;
                if (live.streaming)
                    events.push_back({id, live.session_id, RuntimeEvent::Kind::Token, sig.text, ChorusError::None});
            }
            break;
        case EventType::Stop: {
            // finish_turn first, while live is untouched; the moves below gut it.
            finish_turn(live, TurnOutcome::Completed, live.accumulated_text);
            RuntimeEvent event{
                id, live.session_id, RuntimeEvent::Kind::Complete, std::move(live.accumulated_text), ChorusError::None
            };
            event.reasoning = std::move(live.accumulated_reasoning);
            events.push_back(std::move(event));
            retire_request(id);
            break;
        }
        case EventType::Error:
            finish_turn(
                live, sig.error_code == ChorusError::Cancelled ? TurnOutcome::Cancelled : TurnOutcome::Errored, ""
            );
            events.push_back({id, live.session_id, RuntimeEvent::Kind::Error, sig.text, sig.error_code});
            retire_request(id);
            break;
        default:
            break; // EventType::Embedding et al.: no consumer yet
        }
    }
    return events;
}

void ChorusRuntime::stop_all() {
    assert_host_thread();
    unload_engine();
    cancel_live_requests();
}

void ChorusRuntime::unload_engine() {
    if (_engine) {
        // Port contract: after stop() returns, the engine never invokes a
        // previously supplied on_event again.
        _engine->stop();
        _engine.reset();
    }
}

void ChorusRuntime::cancel_live_requests() {
    if (_live_requests.empty())
        return;
    // The engine is already stopped (contract: no further callbacks), but the
    // mutex keeps this correct even if a future caller reorders.
    std::lock_guard<std::mutex> lock(_pending_mutex);
    for (const auto& request : _live_requests) {
        ChorusSignal sig;
        sig.request_id = request.first;
        sig.type = EventType::Error;
        sig.error_code = ChorusError::Cancelled;
        sig.text = "Request cancelled: engine stopped.";
        _pending_signals.push_back(sig);
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

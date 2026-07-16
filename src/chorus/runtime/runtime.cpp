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
    if (request.session_id && _active_sessions.count(*request.session_id))
        return SubmitResult{
            -1, ChorusError::SessionBusy, "Session '" + *request.session_id + "' already has a live request."
        };

    const int64_t id = _next_request_id.fetch_add(1);
    ChorusRequest engine_request;
    engine_request.id = id;
    engine_request.session_id = request.session_id;
    engine_request.priority = request.priority;
    engine_request.prompt = request.prompt;
    engine_request.gen_config = request.config;
    engine_request.on_event = [this](ChorusSignal& sig) { enqueue_signal(sig); };

    if (auto rejection = _engine->validate_request(engine_request))
        return SubmitResult{-1, rejection->error, rejection->message};

    // Live state exists before the engine sees the request: engines may invoke
    // on_event inline during submit_request.
    _request_streaming[id] = request.stream;
    if (request.session_id) {
        _request_sessions[id] = *request.session_id;
        _active_sessions.insert(*request.session_id);
    }
    _engine->submit_request(engine_request);
    return SubmitResult{id, ChorusError::None, ""};
}

bool ChorusRuntime::cancel(RequestId id) {
    assert_host_thread();
    if (_request_streaming.find(id) == _request_streaming.end())
        return false;
    if (_engine)
        _engine->cancel_request(id);
    return true;
}

bool ChorusRuntime::is_request_active(RequestId id) const {
    assert_host_thread();
    return _request_streaming.find(id) != _request_streaming.end();
}

std::optional<RequestId> ChorusRuntime::active_request_for_session(const SessionId& session_id) const {
    assert_host_thread();
    for (const auto& [id, request_session] : _request_sessions) {
        if (request_session == session_id)
            return id;
    }
    return std::nullopt;
}

std::vector<RuntimeEvent> ChorusRuntime::poll() {
    assert_host_thread();
    std::vector<ChorusSignal> batch;
    {
        std::lock_guard<std::mutex> lock(_pending_mutex);
        batch.swap(_pending_signals);
    }

    std::vector<RuntimeEvent> events;
    for (const auto& sig : batch) {
        const int64_t id = sig.request_id;
        auto streaming_it = _request_streaming.find(id);
        if (streaming_it == _request_streaming.end())
            continue; // no live state: late token, duplicate terminal, or unknown id

        auto session_it = _request_sessions.find(id);
        std::optional<SessionId> session;
        if (session_it != _request_sessions.end())
            session = session_it->second;

        switch (sig.type) {
        case EventType::Token:
            _accumulator[id] += sig.text;
            if (streaming_it->second)
                events.push_back({id, session, RuntimeEvent::Kind::Token, sig.text, ChorusError::None});
            break;
        case EventType::Stop:
            events.push_back(
                {id, session, RuntimeEvent::Kind::Complete, std::move(_accumulator[id]), ChorusError::None}
            );
            _accumulator.erase(id);
            _request_streaming.erase(id);
            if (session_it != _request_sessions.end()) {
                _active_sessions.erase(session_it->second);
                _request_sessions.erase(session_it);
            }
            break;
        case EventType::Error:
            events.push_back({id, session, RuntimeEvent::Kind::Error, sig.text, sig.error_code});
            _accumulator.erase(id);
            _request_streaming.erase(id);
            if (session_it != _request_sessions.end()) {
                _active_sessions.erase(session_it->second);
                _request_sessions.erase(session_it);
            }
            break;
        default:
            break; // EventType::Embedding et al.: not consumed pre-#6
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
    if (_request_streaming.empty())
        return;
    // The engine is already stopped (contract: no further callbacks), but the
    // mutex keeps this correct even if a future caller reorders.
    std::lock_guard<std::mutex> lock(_pending_mutex);
    for (const auto& [id, streaming] : _request_streaming) {
        ChorusSignal sig;
        sig.request_id = id;
        sig.type = EventType::Error;
        sig.error_code = ChorusError::Cancelled;
        sig.text = "Request cancelled: engine stopped.";
        _pending_signals.push_back(sig);
    }
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

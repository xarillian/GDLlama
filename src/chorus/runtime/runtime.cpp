#include "chorus/runtime/runtime.hpp"

#include <cassert>

namespace Chorus {
namespace {

SubmitResult submission_rejection(ChorusError error, std::string message) {
    SubmitResult result;
    result.error = error;
    result.message = std::move(message);
    return result;
}

} // namespace

ChorusRuntime::~ChorusRuntime() {
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
    Logger logger(_log_channel, config.log_level);
    auto err = engine->initialize(config, std::move(logger));
    if (err)
        return err;
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
    return is_loaded() ? std::optional<EngineCapabilities>{_engine->capabilities()} : std::nullopt;
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
    return submission_rejection(
        ChorusError::EngineNotReady, _engine ? "The engine has failed; load it again." : "No engine is loaded."
    );
}

SubmitResult ChorusRuntime::submit(const GenerationRequest& request) {
    assert_host_thread();
    if (!is_loaded())
        return not_ready();
    if (request.session_id && request.session_id->empty())
        return submission_rejection(ChorusError::InvalidRequest, "session_id must be non-empty when present.");
    if (request.session_id && _request_by_session.contains(*request.session_id))
        return submission_rejection(ChorusError::SessionBusy, "Session '" + *request.session_id + "' already has a live request.");

    const ResolvedRequest resolved = resolve_request(request);
    if (!request.session_id &&
        (!request.inject.empty() || !resolved.chat_template.empty() || resolved.config.show_thinking.has_value()))
        return submission_rejection(
            ChorusError::InvalidRequest, "inject/chat_template/show_thinking are chat controls; they require a session."
        );

    ChorusRequest engine_request = make_engine_request(resolved);
    if (!request.session_id)
        return submit_engine_request(resolved, std::move(engine_request));
    if (request.prompt.empty())
        return submission_rejection(ChorusError::InvalidRequest, "Chat turns require a non-empty prompt.");

    std::vector<ConversationMessage> history;
    if (auto it = _histories.find(*request.session_id); it != _histories.end())
        history = it->second.messages;
    auto fitted = fit_turn_messages(
        resolved, std::move(history), ChatMessage{MessageRole::User, MessageContent::text(request.prompt)}
    );
    if (std::holds_alternative<SubmitResult>(fitted))
        return std::get<SubmitResult>(std::move(fitted));
    auto turn = std::get<FittedTurn>(std::move(fitted));
    engine_request.messages = std::move(turn.messages);
    return submit_engine_request(resolved, std::move(engine_request), std::move(turn.omitted_message_ids));
}

SubmitResult ChorusRuntime::submit(const EmbeddingRequest& request) {
    assert_host_thread();
    if (!is_loaded())
        return not_ready();
    if (request.session_id && request.session_id->empty())
        return submission_rejection(ChorusError::InvalidRequest, "session_id must be non-empty when present.");
    if (request.session_id && _request_by_session.contains(*request.session_id))
        return submission_rejection(ChorusError::SessionBusy, "Session '" + *request.session_id + "' already has a live request.");

    ChorusRequest engine_request;
    engine_request.type = RequestType::Embedding;
    engine_request.session_id = request.session_id;
    engine_request.prompt = request.prompt;
    engine_request.priority = request.priority;
    engine_request.execution = request.execution;
    if (auto rejection = _engine->validate_request(engine_request))
        return submission_rejection(rejection->error, rejection->message);

    const RequestId id = _next_request_id++;
    engine_request.id = id;
    engine_request.on_event = [this](ChorusSignal& signal) { enqueue_signal(signal); };
    LiveRequest live;
    live.type = RequestType::Embedding;
    live.session_id = request.session_id;
    _live_requests.emplace(id, std::move(live));
    if (request.session_id)
        _request_by_session.emplace(*request.session_id, id);
    _engine->submit_request(engine_request);
    SubmitResult result;
    result.request_id = id;
    return result;
}

std::vector<SubmitResult> ChorusRuntime::submit_batch(const std::vector<GenerationRequest>& requests) {
    assert_host_thread();
    std::vector<SubmitResult> results;
    results.reserve(requests.size());
    for (const auto& request : requests)
        results.push_back(submit(request));
    return results;
}

std::vector<SubmitResult> ChorusRuntime::submit_batch(const std::vector<EmbeddingRequest>& requests) {
    assert_host_thread();
    std::vector<SubmitResult> results;
    results.reserve(requests.size());
    for (const auto& request : requests)
        results.push_back(submit(request));
    return results;
}

SubmitResult ChorusRuntime::submit_engine_request(
    const ResolvedRequest& resolved,
    ChorusRequest engine_request,
    std::vector<MessageId> omitted_message_ids,
    std::optional<ConversationMessage> replaced_reply
) {
    const GenerationRequest& request = resolved.request;
    if (auto rejection = _engine->validate_request(engine_request))
        return submission_rejection(rejection->error, rejection->message);

    const bool sessioned = request.session_id.has_value();
    const bool regenerating = replaced_reply.has_value();
    std::optional<MessageId> user_id;
    std::optional<MessageId> assistant_id;
    if (sessioned && !regenerating) {
        if (!_next_message_id || * _next_message_id > INT64_MAX - 1)
            return submission_rejection(ChorusError::InvalidRequest, "Message identity capacity is exhausted.");
        user_id = *_next_message_id;
        assistant_id = *user_id + 1;
        _next_message_id = *assistant_id == INT64_MAX ? std::nullopt : std::optional<MessageId>{*assistant_id + 1};
    } else if (regenerating) {
        assistant_id = replaced_reply->id;
    }

    const RequestId id = _next_request_id++;
    engine_request.id = id;
    engine_request.on_event = [this](ChorusSignal& signal) { enqueue_signal(signal); };
    LiveRequest live;
    live.type = RequestType::Generate;
    live.streaming = request.stream;
    live.session_id = request.session_id;
    live.pending_user_id = user_id;
    live.reserved_assistant_id = regenerating ? std::nullopt : assistant_id;
    live.replaced_reply = replaced_reply;
    _live_requests.emplace(id, std::move(live));

    if (sessioned) {
        _request_by_session.emplace(*request.session_id, id);
        auto& history = _histories[*request.session_id];
        if (regenerating) {
            auto reply = std::find_if(history.messages.begin(), history.messages.end(), [&](const auto& item) {
                return item.id == replaced_reply->id;
            });
            if (reply != history.messages.end())
                history.messages.erase(reply);
        } else {
            history.messages.push_back({*user_id, {MessageRole::User, MessageContent::text(request.prompt)}});
        }
        if (!omitted_message_ids.empty()) {
            RuntimeEvent event{id, request.session_id, RuntimeEvent::Kind::HistoryTruncated};
            event.omitted_message_ids = std::move(omitted_message_ids);
            _host_events.push_back(std::move(event));
        }
    }
    _engine->submit_request(engine_request);
    SubmitResult result;
    result.request_id = id;
    result.request_message_id = user_id;
    result.response_message_id = assistant_id;
    return result;
}

bool ChorusRuntime::cancel(RequestId id) {
    assert_host_thread();
    if (!_live_requests.contains(id))
        return false;
    if (_engine)
        _engine->cancel_request(id);
    return true;
}

bool ChorusRuntime::is_request_active(RequestId id) const {
    assert_host_thread();
    return _live_requests.contains(id);
}

std::optional<RequestId> ChorusRuntime::active_request_for_session(const SessionId& session_id) const {
    assert_host_thread();
    const auto it = _request_by_session.find(session_id);
    return it == _request_by_session.end() ? std::nullopt : std::optional<RequestId>{it->second};
}

std::vector<RuntimeEvent> ChorusRuntime::poll() {
    assert_host_thread();
    auto signals = drain_pending_signals();
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
            events.push_back({id, live.session_id, RuntimeEvent::Kind::Error, "Provider emitted a token for an embedding request.", ChorusError::Unknown});
            retire_request(id);
            return;
        }
        if (token->channel == TokenChannel::Reasoning) {
            live.accumulated_reasoning += token->text;
            if (live.streaming)
                events.push_back({id, live.session_id, RuntimeEvent::Kind::StreamedReasoningToken, token->text});
        } else {
            live.accumulated_text += token->text;
            if (live.streaming)
                events.push_back({id, live.session_id, RuntimeEvent::Kind::StreamedToken, token->text});
        }
        return;
    }
    if (std::holds_alternative<ChorusSignal::Stop>(signal.event)) {
        if (live.type == RequestType::Embedding) {
            if (!live.embedding) {
                events.push_back({id, live.session_id, RuntimeEvent::Kind::Error, "Provider stopped an embedding request without a vector.", ChorusError::Unknown});
            } else {
                RuntimeEvent event{id, live.session_id, RuntimeEvent::Kind::Embedding};
                event.embedding = std::move(*live.embedding);
                events.push_back(std::move(event));
            }
            retire_request(id);
            return;
        }
        const std::optional<MessageId> completed_id = live.replaced_reply ? std::optional<MessageId>{live.replaced_reply->id}
                                                                           : live.reserved_assistant_id;
        finish_turn(live, TurnOutcome::Completed, live.accumulated_text);
        RuntimeEvent event{id, live.session_id, RuntimeEvent::Kind::Complete, std::move(live.accumulated_text)};
        event.reasoning = std::move(live.accumulated_reasoning);
        event.message_id = completed_id;
        events.push_back(std::move(event));
        retire_request(id);
        return;
    }
    if (const auto* error = std::get_if<ChorusSignal::Error>(&signal.event)) {
        if (live.type == RequestType::Generate)
            finish_turn(live, error->code == ChorusError::Cancelled ? TurnOutcome::Cancelled : TurnOutcome::Errored, "");
        events.push_back({id, live.session_id, RuntimeEvent::Kind::Error, error->message, error->code});
        retire_request(id);
    }
}

void ChorusRuntime::append_engine_failure(std::vector<RuntimeEvent>& events) {
    if (!_engine || _engine_failure_reported || _engine->is_initialized())
        return;
    _engine_failure_reported = true;
    events.push_back({-1, std::nullopt, RuntimeEvent::Kind::EngineFailed, "The engine has failed.", ChorusError::EngineNotReady});
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
        _engine->shutdown();
        _engine.reset();
    }
    _engine_failure_reported = false;
}

void ChorusRuntime::cancel_live_requests() {
    if (_live_requests.empty())
        return;
    std::lock_guard<std::mutex> lock(_pending_mutex);
    for (const auto& request : _live_requests)
        _pending_signals.emplace_back(request.first, ChorusSignal::Error{ChorusError::Cancelled, "Request cancelled: engine stopped."});
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

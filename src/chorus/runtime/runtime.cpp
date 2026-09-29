#include "chorus/runtime/runtime.hpp"
#include "chorus/runtime/runtime_preparation.hpp"
#include "chorus/runtime/runtime_loading.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

namespace Chorus {
namespace {

SubmitResult rejection(ChorusError error, std::string message) {
    SubmitResult result;
    result.error = error;
    result.message = std::move(message);
    return result;
}

bool valid_content(const MessageContent& content) {
    return std::ranges::all_of(content.parts, [](const auto& part) { return std::holds_alternative<std::string>(part); });
}

GenerationConfig compose_choices(const GenerationConfig& defaults, const GenerationConfig& request) {
    GenerationConfig result = defaults;
    if (request.max_tokens) result.max_tokens = request.max_tokens;
    if (request.temperature) result.temperature = request.temperature;
    if (request.top_k) result.top_k = request.top_k;
    if (request.top_p) result.top_p = request.top_p;
    if (request.seed) result.seed = request.seed;
    if (request.frequency_penalty) result.frequency_penalty = request.frequency_penalty;
    if (request.presence_penalty) result.presence_penalty = request.presence_penalty;
    if (request.stop) result.stop = request.stop;
    if (request.constraint) result.constraint = request.constraint;
    if (request.show_thinking) result.show_thinking = request.show_thinking;

    for (auto it = result.provider_options.begin(); it != result.provider_options.end();) {
        const auto* entries = std::get_if<ProviderOptionMap>(&it->second);
        if (entries && entries->empty()) it = result.provider_options.erase(it);
        else ++it;
    }
    for (const auto& [provider, choice] : request.provider_options) {
        const auto* incoming = std::get_if<ProviderOptionMap>(&choice);
        if (incoming && incoming->empty()) continue;
        auto existing = result.provider_options.find(provider);
        if (existing != result.provider_options.end() && incoming) {
            if (auto* entries = std::get_if<ProviderOptionMap>(&existing->second)) {
                for (const auto& [key, value] : *incoming)
                    (*entries)[key] = value;
                continue;
            }
        }
        result.provider_options[provider] = choice;
    }
    return result;
}

} // namespace

ChorusRuntime::ChorusRuntime() : _wakeup(std::make_shared<Wakeup>()) {}

ChorusRuntime::~ChorusRuntime() {
    stop_all();
    if (_loading) {
        {
            std::lock_guard lock(_loading->mutex);
            _loading->stopping = true;
        }
        _loading->cv.notify_all();
        _loading->worker.join();
    }
}

bool ChorusRuntime::is_loaded() const {
    return _lifetime && !_lifetime->preparation->failed && _lifetime->engine->is_initialized();
}

std::optional<LoadedModelInfo> ChorusRuntime::loaded_model_info() const {
    assert_host_thread();
    return is_loaded() ? _lifetime->preparation->model_info : std::nullopt;
}

std::optional<EngineCapabilities> ChorusRuntime::capabilities() const {
    assert_host_thread();
    return is_loaded() ? std::optional<EngineCapabilities>{_lifetime->preparation->capabilities} : std::nullopt;
}

void ChorusRuntime::set_generation_defaults(GenerationDefaults defaults) {
    assert_host_thread();
    _generation_defaults = std::move(defaults);
}

ChorusRuntime::ResolvedRequest ChorusRuntime::resolve_request(const GenerationRequest& request) const {
    ResolvedRequest resolved{request, compose_choices(_generation_defaults.options, request.options), request.chat_template};
    if (!request.session_id) {
        if (!request.options.show_thinking)
            resolved.config.show_thinking.reset();
    } else if (!resolved.chat_template) {
        resolved.chat_template = _generation_defaults.chat_template;
    }
    return resolved;
}

ChorusRequest ChorusRuntime::make_engine_request(const ResolvedRequest& resolved) {
    ChorusRequest request;
    request.session_id = resolved.request.session_id;
    request.priority = resolved.request.priority;
    request.execution = resolved.request.execution;
    return request;
}

SubmitResult ChorusRuntime::not_ready() const {
    return rejection(ChorusError::EngineNotReady, active_load_id() ? "An engine is loading." :
                     _lifetime ? "The engine has failed; load it again." : "No engine is loaded.");
}

SubmitResult ChorusRuntime::submit(const GenerationRequest& request) {
    return admit_generation(request, Operation::Generate);
}

SubmitResult ChorusRuntime::render_prompt(const GenerationRequest& request) {
    return admit_generation(request, Operation::Preview);
}

SubmitResult ChorusRuntime::regenerate(const GenerationRequest& request) {
    return admit_generation(request, Operation::Generate, true);
}

SubmitResult ChorusRuntime::admit_generation(const GenerationRequest& request, Operation operation, bool regenerate) {
    assert_host_thread();
    if (!is_loaded())
        return not_ready();
    if (request.session_id && request.session_id->empty())
        return rejection(ChorusError::InvalidRequest, "session_id must be non-empty when present.");
    if (operation == Operation::Generate && request.session_id && _request_by_session.contains(*request.session_id))
        return rejection(ChorusError::SessionBusy, "Session already has a live request.");
    if (regenerate && (!request.session_id || !request.prompt.empty()))
        return rejection(ChorusError::InvalidRequest, "Regeneration requires a session and no prompt.");
    if (!regenerate && request.session_id && request.prompt.empty())
        return rejection(ChorusError::InvalidRequest, "Chat turns require a non-empty prompt.");
    if (operation == Operation::Preview && request.session_id && !_lifetime->preparation->capabilities.prompt_rendering)
        return rejection(ChorusError::UnsupportedFeature, "The provider cannot render chat prompts.");
    for (const auto& injected : request.inject) {
        if (!message_role_name(injected.message.role) || !valid_content(injected.message.content))
            return rejection(ChorusError::InvalidRequest, "An injected message is invalid.");
    }
    auto job = std::make_unique<PreparationJob>();
    job->operation = operation;
    job->resolved = resolve_request(request);
    const auto& resolved = job->resolved;
    if (!request.session_id && (!request.inject.empty() || request.chat_template.has_value() || request.options.show_thinking.has_value()))
        return rejection(ChorusError::InvalidRequest, "inject/chat_template/show_thinking require a session.");
    job->request = make_engine_request(resolved);
    MessageNodePtr replaced;
    if (request.session_id) {
        auto it = _histories.find(*request.session_id);
        job->history = it == _histories.end() ? std::make_shared<const HistoryNodes>() : it->second.messages;
        if (regenerate) {
            if (job->history->empty() || job->history->back()->value.message.role != MessageRole::Assistant)
                return rejection(ChorusError::InvalidRequest, "Regeneration needs history ending in an assistant reply.");
            replaced = job->history->back();
            auto prospective = std::make_shared<HistoryNodes>(job->history->begin(), job->history->end() - 1);
            job->history = std::move(prospective);
        }
    }
    return admit(std::move(job), std::move(replaced));
}

SubmitResult ChorusRuntime::submit(const EmbeddingRequest& request) {
    assert_host_thread();
    if (!is_loaded())
        return not_ready();
    if (request.session_id && request.session_id->empty())
        return rejection(ChorusError::InvalidRequest, "session_id must be non-empty when present.");
    if (request.session_id && _request_by_session.contains(*request.session_id))
        return rejection(ChorusError::SessionBusy, "Session already has a live request.");
    if (!_lifetime->preparation->capabilities.embeddings)
        return rejection(ChorusError::UnsupportedFeature, "The loaded engine does not serve embeddings.");
    if (request.prompt.empty())
        return rejection(ChorusError::InvalidRequest, "Embedding prompts must not be empty.");
    auto job = std::make_unique<PreparationJob>();
    job->operation = Operation::Embed;
    job->request.type = RequestType::Embedding;
    job->request.session_id = request.session_id;
    job->request.prompt = request.prompt;
    job->request.priority = request.priority;
    job->request.execution = request.execution;
    return admit(std::move(job));
}

SubmitResult ChorusRuntime::count_message_tokens(MessageContent content) {
    assert_host_thread();
    if (!is_loaded())
        return not_ready();
    if (!_lifetime->preparation->capabilities.message_token_counting)
        return rejection(ChorusError::UnsupportedFeature, "The provider has no message tokenizer.");
    if (!valid_content(content))
        return rejection(ChorusError::UnsupportedFeature, "Message content must be text.");
    auto job = std::make_unique<PreparationJob>();
    job->operation = Operation::Count;
    job->content = std::move(content);
    return admit(std::move(job));
}

SubmitResult ChorusRuntime::admit(std::unique_ptr<PreparationJob> job, MessageNodePtr replaced_reply) {
    auto& state = *_lifetime->preparation;
    std::unique_lock<std::mutex> lock(state.mutex);
    if (state.closing || state.failed)
        return not_ready();
    if (state.outstanding >= kPreparationCapacity)
        return rejection(ChorusError::InvalidRequest, "Preparation capacity exhausted (256 outstanding jobs).");
    if (_next_request_id == INT64_MAX)
        return rejection(ChorusError::InvalidRequest, "Request identity capacity is exhausted.");
    const auto session = job->request.session_id;
    const bool generation = job->operation == Operation::Generate;
    const bool occupies = generation || job->operation == Operation::Embed;
    SubmitResult result;
    result.request_id = _next_request_id;
    if (generation && session) {
        if (replaced_reply) {
            result.response_message_id = replaced_reply->value.id;
        } else {
            if (!_next_message_id || *_next_message_id > INT64_MAX - 1)
                return rejection(ChorusError::InvalidRequest, "Message identity capacity is exhausted.");
            result.request_message_id = *_next_message_id;
            result.response_message_id = *_next_message_id + 1;
        }
    }
    if (session && !replaced_reply) {
        job->pending = make_node({result.request_message_id.value_or(-1),
                                 {MessageRole::User, MessageContent::text(std::move(job->resolved.request.prompt))}});
    }
    HistorySnapshot changed_history;
    if (generation && session) {
        if (replaced_reply) {
            changed_history = job->history;
        } else {
            auto nodes = std::make_shared<HistoryNodes>(*job->history);
            nodes->push_back(job->pending);
            changed_history = std::move(nodes);
        }
    }
    job->request.id = result.request_id;
    const auto control = job->control;
    control->id = result.request_id;
    std::weak_ptr<PreparationState> output_state = _lifetime->preparation;
    job->request.on_event = [output_state, control](ChorusSignal& signal) {
        if (auto state = output_state.lock())
            enqueue_signal(*state, signal, control);
    };
    LiveRequest live;
    live.operation = job->operation;
    live.control = control;
    live.preparation = _lifetime->preparation;
    live.streaming = job->resolved.request.stream;
    live.session_id = session;
    live.pending_user_id = result.request_message_id;
    live.reserved_assistant_id = replaced_reply ? std::nullopt : result.response_message_id;
    live.replaced_reply = std::move(replaced_reply);
    bool history_created = false;
    try {
        _live_requests.emplace(result.request_id, std::move(live));
        state.controls.emplace(result.request_id, control);
        if (session && occupies)
            _request_by_session.emplace(*session, result.request_id);
        if (changed_history)
            history_created = _histories.try_emplace(*session).second;
        state.jobs.push_back(std::move(job));
    } catch (...) {
        _live_requests.erase(result.request_id);
        state.controls.erase(result.request_id);
        if (session && occupies)
            _request_by_session.erase(*session);
        if (history_created)
            _histories.erase(*session);
        throw;
    }
    if (changed_history)
        _histories.at(*session).messages = std::move(changed_history);
    if (result.request_message_id)
        _next_message_id = *result.response_message_id == INT64_MAX ? std::nullopt : std::optional<MessageId>{*result.response_message_id + 1};
    ++_next_request_id;
    ++state.outstanding;
    lock.unlock();
    state.cv.notify_one();
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

bool ChorusRuntime::cancel(RequestId id) {
    assert_host_thread();
    auto found = _live_requests.find(id);
    if (found == _live_requests.end())
        return false;
    auto control = found->second.control;
    control->cancelled = true;
    if (control->provider_active) {
        if (_lifetime && found->second.preparation == _lifetime->preparation)
            _lifetime->engine->cancel_request(id);
    } else {
        publish_error(*found->second.preparation, id, control, ChorusError::Cancelled, "Request cancelled.");
    }
    return true;
}

bool ChorusRuntime::is_request_active(RequestId id) const {
    assert_host_thread();
    return _live_requests.contains(id);
}

std::optional<RequestId> ChorusRuntime::active_request_for_session(const SessionId& session) const {
    assert_host_thread();
    auto it = _request_by_session.find(session);
    return it == _request_by_session.end() ? std::nullopt : std::optional<RequestId>{it->second};
}

std::vector<RuntimeEvent> ChorusRuntime::poll() {
    assert_host_thread();
    _wakeup->clear();
    std::vector<RuntimeEvent> events;
    if (_lifetime && !is_loaded()) {
        fail_preparation(*_lifetime->preparation);
        for (const auto& [id, live] : _live_requests)
            if (live.preparation == _lifetime->preparation && live.control->provider_active)
                _lifetime->engine->cancel_request(id);
    }
    auto drain = [&](const std::shared_ptr<PreparationState>& state) {
        std::vector<PreparationState::Output> output;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            output.swap(state->output);
        }
        for (auto& item : output) {
            if (auto* signal = std::get_if<ChorusSignal>(&item)) {
                append_signal_events(*signal, events);
            } else if (auto* event = std::get_if<RuntimeEvent>(&item)) {
                if (_live_requests.contains(event->request_id)) {
                    retire_request(event->request_id);
                    events.push_back(std::move(*event));
                }
            } else {
                auto job = std::move(std::get<std::unique_ptr<PreparationJob>>(item));
                bool forward;
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    job->control->preparation_drained = true;
                    state->release_preparation(job->control);
                    forward = !job->control->terminal && !job->control->cancelled &&
                              _lifetime && _lifetime->preparation == state && is_loaded();
                    if (forward)
                        job->control->provider_active = true;
                }
                if (!forward)
                    continue;
                if (!job->omitted.empty()) {
                    RuntimeEvent truncated{job->request.id, job->request.session_id, RuntimeEvent::Kind::HistoryTruncated};
                    truncated.omitted_message_ids = std::move(job->omitted);
                    events.push_back(std::move(truncated));
                }
                _lifetime->engine->submit_request(std::move(job->request));
            }
        }
    };
    for (const auto& state : _draining_states)
        drain(state);
    if (_lifetime)
        drain(_lifetime->preparation);
    append_load_events(events, [&] {
        for (const auto& state : _draining_states)
            drain(state);
    });
    std::erase_if(_draining_states, [&](const auto& state) {
        return std::none_of(_live_requests.begin(), _live_requests.end(),
                            [&](const auto& item) { return item.second.preparation == state; });
    });
    append_engine_failure(events);
    return events;
}

void ChorusRuntime::append_signal_events(const ChorusSignal& signal, std::vector<RuntimeEvent>& events) {
    const RequestId id = signal.request_id;
    auto request = _live_requests.find(id);
    if (request == _live_requests.end())
        return;
    LiveRequest& live = request->second;
    if (const auto* embedding = std::get_if<ChorusSignal::Embedding>(&signal.event)) {
        if (live.operation != Operation::Embed) {
            finish_turn(live, TurnOutcome::Errored, "");
            events.push_back({id, live.session_id, RuntimeEvent::Kind::Error, "Provider emitted an embedding for a generation request.", ChorusError::Unknown});
            retire_request(id);
            return;
        }
        live.embedding = embedding->values;
        return;
    }
    if (const auto* token = std::get_if<ChorusSignal::Token>(&signal.event)) {
        if (live.operation != Operation::Generate) {
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
        if (live.operation == Operation::Embed) {
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
        const auto completed_id = live.replaced_reply ? std::optional<MessageId>{live.replaced_reply->value.id} : live.reserved_assistant_id;
        finish_turn(live, TurnOutcome::Completed, live.accumulated_text);
        RuntimeEvent event{id, live.session_id, RuntimeEvent::Kind::Complete, std::move(live.accumulated_text)};
        event.reasoning = std::move(live.accumulated_reasoning);
        event.message_id = completed_id;
        events.push_back(std::move(event));
        retire_request(id);
        return;
    }
    if (const auto* error = std::get_if<ChorusSignal::Error>(&signal.event)) {
        if (live.operation == Operation::Generate)
            finish_turn(live, error->code == ChorusError::Cancelled ? TurnOutcome::Cancelled : TurnOutcome::Errored, "");
        events.push_back({id, live.session_id, RuntimeEvent::Kind::Error, error->message, error->code});
        retire_request(id);
    }
}

void ChorusRuntime::append_engine_failure(std::vector<RuntimeEvent>& events) {
    if (!_lifetime || _engine_failure_reported || is_loaded())
        return;
    _engine_failure_reported = true;
    events.push_back({-1, std::nullopt, RuntimeEvent::Kind::EngineFailed, "The engine has failed.", ChorusError::EngineNotReady});
}

bool ChorusRuntime::wait_for_events(std::chrono::nanoseconds timeout) {
    assert_host_thread();
    return _wakeup->wait_for(timeout);
}

std::vector<LogRecord> ChorusRuntime::poll_logs() {
    assert_host_thread();
    return _log_channel->drain();
}

void ChorusRuntime::stop_all() {
    assert_host_thread();
    if (_loading) {
        std::unique_lock lock(_loading->mutex);
        if (auto attempt = _loading->attempt) {
            if (!attempt->committed) {
                attempt->cancelled = true;
                attempt->stop.request_stop();
                _loading->cv.notify_all();
                _loading->cv.wait(lock, [&] { return attempt->finished; });
            }
        }
    }
    unload_engine();
}

void ChorusRuntime::unload_engine() {
    if (!_lifetime)
        return;
    close_preparation(*_lifetime->preparation);
    fence_engine(*_lifetime);
    if (!_lifetime->preparation->controls.empty() || !_lifetime->preparation->output.empty())
        _draining_states.push_back(_lifetime->preparation);
    _lifetime.reset();
    _engine_failure_reported = false;
}

void ChorusRuntime::close_preparation(PreparationState& state) {
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.closing = true;
        for (const auto& [id, control] : state.controls) {
            control->cancelled = true;
            if (!control->provider_active && !control->terminal) {
                state.publish(ChorusSignal{id, ChorusSignal::Error{ChorusError::Cancelled, "Request cancelled: engine stopped."}});
                control->terminal = true;
            }
        }
    }
    state.cv.notify_all();
}

void ChorusRuntime::fence_engine(EngineLifetime& lifetime) {
    auto& state = *lifetime.preparation;
    if (state.worker.joinable())
        state.worker.join();
    std::vector<std::unique_ptr<PreparationJob>> discarded;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        for (auto& output : state.output) {
            if (auto* job = std::get_if<std::unique_ptr<PreparationJob>>(&output)) {
                (*job)->control->preparation_drained = true;
                state.release_preparation((*job)->control);
                discarded.push_back(std::move(*job));
            }
        }
        std::erase_if(state.output, [](const auto& output) {
            return std::holds_alternative<std::unique_ptr<PreparationJob>>(output);
        });
    }
    discarded.clear();
    state.service.reset();
    state.clear_caches();
    state.capabilities = {};
    state.model_info.reset();
    if (lifetime.engine) {
        lifetime.engine->shutdown();
        lifetime.engine.reset();
    }
    cancel_live_requests(state);
}

void ChorusRuntime::cancel_live_requests(PreparationState& state) {
    std::vector<std::pair<RequestId, std::shared_ptr<Control>>> pending;
    {
        std::lock_guard lock(state.mutex);
        for (const auto& [id, control] : state.controls)
            if (!control->terminal)
                pending.emplace_back(id, control);
    }
    for (const auto& [id, control] : pending)
        publish_error(state, id, control, ChorusError::Cancelled, "Request cancelled: engine stopped.");
}

void ChorusRuntime::retire_request(RequestId id) {
    auto request = _live_requests.find(id);
    if (request == _live_requests.end())
        return;
    if (request->second.session_id) {
        auto lane = _request_by_session.find(*request->second.session_id);
        if (lane != _request_by_session.end() && lane->second == id)
            _request_by_session.erase(lane);
    }
    {
        auto& state = *request->second.preparation;
        std::lock_guard<std::mutex> lock(state.mutex);
        request->second.control->terminal = true;
        request->second.control->preparation_drained = true;
        state.release_preparation(request->second.control);
        state.controls.erase(id);
    }
    _live_requests.erase(request);
}

void ChorusRuntime::enqueue_signal(PreparationState& state, const ChorusSignal& signal, const std::shared_ptr<Control>& control) {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (control->terminal || signal.request_id != control->id)
        return;
    state.publish(signal);
    if (std::holds_alternative<ChorusSignal::Stop>(signal.event) || std::holds_alternative<ChorusSignal::Error>(signal.event))
        control->terminal = true;
}

void ChorusRuntime::publish_error(PreparationState& state, RequestId id, const std::shared_ptr<Control>& control, ChorusError error, std::string message) {
    enqueue_signal(state, {id, ChorusSignal::Error{error, std::move(message)}}, control);
}

void ChorusRuntime::assert_host_thread() const {
#ifndef NDEBUG
    std::thread::id expected{};
    _host_thread.compare_exchange_strong(expected, std::this_thread::get_id());
    assert(_host_thread.load() == std::this_thread::get_id() && "ChorusRuntime public methods are host-thread-only");
#endif
}

} // namespace Chorus

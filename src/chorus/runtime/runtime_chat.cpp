#include "chorus/runtime/runtime.hpp"

#include "chorus/runtime/prompt_fitting.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_set>

namespace Chorus {
namespace {

constexpr int32_t kFallbackResponseReservation = 512;

bool valid_message(const ChatMessage& message) {
    return message_role_name(message.role).has_value() && joined_text(message.content).has_value();
}

SubmitResult rejected(std::string message) {
    SubmitResult result;
    result.error = ChorusError::InvalidRequest;
    result.message = std::move(message);
    return result;
}

} // namespace

std::optional<ChorusError>
ChorusRuntime::import_conversation_history(const SessionId& session, std::vector<ConversationMessage> history) {
    assert_host_thread();
    if (session.empty())
        return ChorusError::InvalidRequest;
    if (_request_by_session.contains(session))
        return ChorusError::SessionBusy;

    std::unordered_set<MessageId> ids;
    std::optional<MessageId> largest;
    for (const auto& entry : history) {
        if (entry.id < 0 || !ids.insert(entry.id).second || !valid_message(entry.message))
            return ChorusError::InvalidRequest;
        if (!largest || entry.id > *largest)
            largest = entry.id;
    }

    std::optional<MessageId> next = _next_message_id;
    if (largest && next) {
        if (*largest == INT64_MAX)
            next.reset();
        else if (*next <= *largest)
            next = *largest + 1;
    }
    _histories[session] = ConversationHistory{std::move(history), TurnOutcome::None};
    _next_message_id = next;
    return std::nullopt;
}

std::vector<ConversationMessage> ChorusRuntime::export_conversation_history(const SessionId& session) const {
    assert_host_thread();
    auto it = _histories.find(session);
    return it == _histories.end() ? std::vector<ConversationMessage>{} : it->second.messages;
}

std::optional<ChorusError> ChorusRuntime::clear_conversation_history(const SessionId& session) {
    assert_host_thread();
    if (_request_by_session.contains(session))
        return ChorusError::SessionBusy;
    _histories.erase(session);
    return std::nullopt;
}

std::optional<ChorusError>
ChorusRuntime::edit_message(const SessionId& session, MessageId message_id, MessageContent content) {
    assert_host_thread();
    if (_request_by_session.contains(session))
        return ChorusError::SessionBusy;
    if (message_id < 0 || !joined_text(content))
        return ChorusError::InvalidRequest;
    auto it = _histories.find(session);
    if (it == _histories.end())
        return ChorusError::InvalidRequest;
    auto message = std::find_if(it->second.messages.begin(), it->second.messages.end(), [message_id](const auto& item) {
        return item.id == message_id;
    });
    if (message == it->second.messages.end())
        return ChorusError::InvalidRequest;
    message->message.content = std::move(content);
    return std::nullopt;
}

std::vector<SessionId> ChorusRuntime::list_conversations() const {
    assert_host_thread();
    std::vector<SessionId> sessions;
    sessions.reserve(_histories.size());
    for (const auto& [session, conversation] : _histories)
        sessions.push_back(session);
    return sessions;
}

std::optional<ChorusError> ChorusRuntime::reset_context() {
    assert_host_thread();
    if (!_request_by_session.empty())
        return ChorusError::SessionBusy;
    _histories.clear();
    return std::nullopt;
}

TurnOutcome ChorusRuntime::last_turn_outcome(const SessionId& session) const {
    assert_host_thread();
    auto it = _histories.find(session);
    return it == _histories.end() ? TurnOutcome::None : it->second.last_turn_outcome;
}

RenderPromptResult ChorusRuntime::render_prompt(const GenerationRequest& request) const {
    assert_host_thread();
    if (!is_loaded())
        return {"", {}, ChorusError::EngineNotReady, "No initialized engine is available."};
    if (request.session_id && request.session_id->empty())
        return {"", {}, ChorusError::InvalidRequest, "session_id must be non-empty when present."};
    const ResolvedRequest resolved = resolve_request(request);
    if (!request.session_id &&
        (!request.inject.empty() || !resolved.chat_template.empty() || resolved.config.show_thinking.has_value()))
        return {"", {}, ChorusError::InvalidRequest, "inject/chat_template/show_thinking are chat controls; they require a session."};

    ChorusRequest engine_request = make_engine_request(resolved);
    if (request.session_id) {
        if (request.prompt.empty())
            return {"", {}, ChorusError::InvalidRequest, "Chat turns require a non-empty prompt."};
        std::vector<ConversationMessage> history;
        if (auto it = _histories.find(*request.session_id); it != _histories.end())
            history = it->second.messages;
        auto fitted = fit_turn_messages(resolved, std::move(history), ChatMessage{MessageRole::User, MessageContent::text(request.prompt)});
        if (std::holds_alternative<SubmitResult>(fitted)) {
            const auto& failure = std::get<SubmitResult>(fitted);
            return {"", {}, failure.error, failure.message};
        }
        auto turn = std::get<FittedTurn>(std::move(fitted));
        engine_request.messages = std::move(turn.messages);
        if (auto rejection = _engine->validate_request(engine_request))
            return {"", {}, rejection->error, rejection->message};
        if (turn.rendered_text)
            return {std::move(*turn.rendered_text), std::move(turn.omitted_message_ids), ChorusError::None, {}};
        auto rendered = _engine->render_chat_prompt(
            engine_request.messages, resolved.chat_template, resolved.config.show_thinking.value_or(true)
        );
        if (!rendered)
            return {"", {}, ChorusError::InvalidRequest, "The engine cannot render this chat prompt."};
        return {std::move(rendered->text), std::move(turn.omitted_message_ids), ChorusError::None, {}};
    }

    if (auto rejection = _engine->validate_request(engine_request))
        return {"", {}, rejection->error, rejection->message};
    return {engine_request.prompt, {}, ChorusError::None, {}};
}

SubmitResult ChorusRuntime::regenerate(const GenerationRequest& request) {
    assert_host_thread();
    if (!is_loaded())
        return not_ready();
    if (!request.session_id || request.session_id->empty())
        return rejected("regenerate() requires a session.");
    if (!request.prompt.empty())
        return rejected("regenerate() takes no prompt; it rerolls the last assistant reply.");
    if (_request_by_session.contains(*request.session_id)) {
        SubmitResult result = rejected("Session '" + *request.session_id + "' already has a live request.");
        result.error = ChorusError::SessionBusy;
        return result;
    }
    auto history_it = _histories.find(*request.session_id);
    if (history_it == _histories.end() || history_it->second.messages.empty() ||
        history_it->second.messages.back().message.role != MessageRole::Assistant)
        return rejected("regenerate() needs a conversation ending in an assistant reply.");

    const ResolvedRequest resolved = resolve_request(request);
    std::vector<ConversationMessage> prospective(history_it->second.messages.begin(), history_it->second.messages.end() - 1);
    auto fitted = fit_turn_messages(resolved, std::move(prospective));
    if (std::holds_alternative<SubmitResult>(fitted))
        return std::get<SubmitResult>(std::move(fitted));
    auto turn = std::get<FittedTurn>(std::move(fitted));
    ChorusRequest engine_request = make_engine_request(resolved);
    engine_request.messages = std::move(turn.messages);
    return submit_engine_request(
        resolved, std::move(engine_request), std::move(turn.omitted_message_ids), history_it->second.messages.back()
    );
}

std::variant<ChorusRuntime::FittedTurn, SubmitResult> ChorusRuntime::fit_turn_messages(
    const ResolvedRequest& resolved,
    std::vector<ConversationMessage> history,
    std::optional<ChatMessage> pending
) const {
    const GenerationRequest& request = resolved.request;
    std::vector<FittingCandidate> candidates;
    candidates.reserve(history.size() + pending.has_value());
    for (const auto& item : history) {
        if (!valid_message(item.message))
            return rejected("Conversation history contains an invalid message.");
        candidates.push_back({item.message, item.id});
    }
    if (pending) {
        if (!valid_message(*pending))
            return rejected("The pending message is invalid.");
        candidates.push_back({std::move(*pending), std::nullopt});
    }
    for (const auto& injected : request.inject)
        if (!valid_message(injected.message))
            return rejected("An injected message is invalid.");

    const bool show_thinking = resolved.config.show_thinking.value_or(true);
    auto info = _engine->loaded_model_info();
    if (!info || !info->per_request_context) {
        std::vector<ChatMessage> messages;
        messages.reserve(candidates.size());
        for (const auto& candidate : candidates)
            messages.push_back(candidate.message);
        return FittedTurn{place_injections(std::move(messages), request.inject), {}, std::nullopt};
    }

    const int32_t reservation = resolved.config.max_tokens && *resolved.config.max_tokens >= 0
                                    ? *resolved.config.max_tokens
                                    : kFallbackResponseReservation;
    const int32_t context = static_cast<int32_t>(std::min<uint32_t>(*info->per_request_context, INT32_MAX));
    const int32_t budget = context - reservation;
    if (budget <= 0)
        return rejected("Response reservation leaves no prompt room in the per-request context.");

    std::optional<std::string> last_render;
    RenderProbe probe = [&](const std::vector<ChatMessage>& candidate) -> std::optional<int32_t> {
        auto rendered = _engine->render_chat_prompt(candidate, resolved.chat_template, show_thinking);
        if (!rendered) {
            last_render.reset();
            return std::nullopt;
        }
        last_render = std::move(rendered->text);
        return rendered->token_count;
    };
    auto fit = fit_messages_to_budget(candidates, request.inject, budget, probe);
    if (std::holds_alternative<ChorusError>(fit))
        return rejected("Conversation does not fit the context window even after truncation.");
    auto result = std::get<FitResult>(std::move(fit));
    return FittedTurn{std::move(result.fitted), std::move(result.omitted_message_ids), std::move(last_render)};
}

void ChorusRuntime::finish_turn(const LiveRequest& live, TurnOutcome outcome, const std::string& text) {
    if (!live.session_id)
        return;
    auto history_it = _histories.find(*live.session_id);
    if (history_it == _histories.end())
        return;
    auto& history = history_it->second;
    history.last_turn_outcome = outcome;
    if (outcome == TurnOutcome::Completed) {
        if (live.reserved_assistant_id)
            history.messages.push_back({*live.reserved_assistant_id, {MessageRole::Assistant, MessageContent::text(text)}});
        else if (live.replaced_reply) {
            ConversationMessage replacement = *live.replaced_reply;
            replacement.message.content = MessageContent::text(text);
            history.messages.push_back(std::move(replacement));
        }
        return;
    }
    if (live.replaced_reply) {
        history.messages.push_back(*live.replaced_reply);
    } else if (live.pending_user_id) {
        auto pending = std::find_if(history.messages.begin(), history.messages.end(), [&](const auto& item) {
            return item.id == *live.pending_user_id;
        });
        if (pending != history.messages.end())
            history.messages.erase(pending);
    }
}

} // namespace Chorus

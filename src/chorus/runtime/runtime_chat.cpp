#include "chorus/runtime/runtime.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace Chorus {
namespace {
bool valid_content(const MessageContent& content) {
    return std::ranges::all_of(content.parts, [](const auto& part) { return std::holds_alternative<std::string>(part); });
}
}

ChorusRuntime::MessageNodePtr ChorusRuntime::make_node(ConversationMessage message) {
    if (_next_content_identity == UINT64_MAX)
        throw std::overflow_error("Content identity capacity is exhausted.");
    return std::make_shared<const MessageNode>(MessageNode{std::move(message), _next_content_identity++});
}

std::optional<ChorusError>
ChorusRuntime::import_conversation_history(const SessionId& session, std::vector<ConversationMessage> history) {
    assert_host_thread();
    if (session.empty())
        return ChorusError::InvalidRequest;
    if (_request_by_session.contains(session))
        return ChorusError::SessionBusy;
    if (history.size() > UINT64_MAX - _next_content_identity)
        return ChorusError::InvalidRequest;
    std::unordered_set<MessageId> ids;
    std::optional<MessageId> largest;
    for (const auto& entry : history) {
        if (entry.id < 0 || !ids.insert(entry.id).second || !message_role_name(entry.message.role) || !valid_content(entry.message.content))
            return ChorusError::InvalidRequest;
        if (!largest || entry.id > *largest)
            largest = entry.id;
    }
    auto nodes = std::make_shared<HistoryNodes>();
    nodes->reserve(history.size());
    for (auto& entry : history)
        nodes->push_back(make_node(std::move(entry)));
    std::optional<MessageId> next = _next_message_id;
    if (largest && next) {
        if (*largest == INT64_MAX)
            next.reset();
        else if (*next <= *largest)
            next = *largest + 1;
    }
    _histories[session] = ConversationHistory{std::move(nodes), TurnOutcome::None};
    _next_message_id = next;
    return std::nullopt;
}

std::vector<ConversationMessage> ChorusRuntime::export_conversation_history(const SessionId& session) const {
    assert_host_thread();
    std::vector<ConversationMessage> result;
    if (auto it = _histories.find(session); it != _histories.end()) {
        result.reserve(it->second.messages->size());
        for (const auto& node : *it->second.messages)
            result.push_back(node->value);
    }
    return result;
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
    if (message_id < 0 || !valid_content(content) || _next_content_identity == UINT64_MAX)
        return ChorusError::InvalidRequest;
    auto it = _histories.find(session);
    if (it == _histories.end())
        return ChorusError::InvalidRequest;
    const auto& current = *it->second.messages;
    auto message = std::find_if(current.begin(), current.end(), [message_id](const auto& node) { return node->value.id == message_id; });
    if (message == current.end())
        return ChorusError::InvalidRequest;
    auto nodes = std::make_shared<HistoryNodes>(current);
    (*nodes)[static_cast<size_t>(message - current.begin())] = make_node({message_id, {(*message)->value.message.role, std::move(content)}});
    it->second.messages = std::move(nodes);
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

void ChorusRuntime::finish_turn(const LiveRequest& live, TurnOutcome outcome, const std::string& text) {
    if (live.operation != Operation::Generate || !live.session_id)
        return;
    auto history_it = _histories.find(*live.session_id);
    if (history_it == _histories.end())
        return;
    auto& history = history_it->second;
    auto nodes = std::make_shared<HistoryNodes>(*history.messages);
    if (outcome == TurnOutcome::Completed) {
        const auto id = live.replaced_reply ? std::optional<MessageId>{live.replaced_reply->value.id} : live.reserved_assistant_id;
        if (id)
            nodes->push_back(make_node({*id, {MessageRole::Assistant, MessageContent::text(text)}}));
    } else if (live.replaced_reply) {
        nodes->push_back(live.replaced_reply);
    } else if (live.pending_user_id) {
        std::erase_if(*nodes, [&](const auto& node) { return node->value.id == *live.pending_user_id; });
    }
    history.messages = std::move(nodes);
    history.last_turn_outcome = outcome;
}

} // namespace Chorus

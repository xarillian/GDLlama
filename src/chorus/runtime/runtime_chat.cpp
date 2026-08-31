#include "chorus/runtime/runtime.hpp"

#include "chorus/runtime/prompt_fitting.hpp"

#include <algorithm>
#include <cstdint>

namespace Chorus {
namespace {

// Fitting reservation when the request leaves max_tokens unset (llama's
// resolved default is unbounded, which cannot be budgeted against).
constexpr int32_t kFallbackResponseReservation = 512;

} // namespace

std::optional<ChorusError>
ChorusRuntime::import_conversation_history(const SessionId& session, std::vector<ChatMessage> history) {
    assert_host_thread();
    if (session.empty())
        return ChorusError::InvalidRequest;
    if (_request_by_session.count(session))
        return ChorusError::SessionBusy;
    auto& conversation = _histories[session];
    conversation.messages = std::move(history);
    conversation.last_turn_outcome = TurnOutcome::None;
    return std::nullopt;
}

std::vector<ChatMessage> ChorusRuntime::export_conversation_history(const SessionId& session) const {
    assert_host_thread();
    auto it = _histories.find(session);
    return it == _histories.end() ? std::vector<ChatMessage>{} : it->second.messages;
}

std::optional<ChorusError> ChorusRuntime::clear_conversation_history(const SessionId& session) {
    assert_host_thread();
    if (_request_by_session.count(session))
        return ChorusError::SessionBusy;
    _histories.erase(session);
    return std::nullopt;
}

std::optional<ChorusError> ChorusRuntime::edit_message(const SessionId& session, int64_t index, std::string content) {
    assert_host_thread();
    if (_request_by_session.count(session))
        return ChorusError::SessionBusy;
    auto it = _histories.find(session);
    if (it == _histories.end())
        return ChorusError::InvalidRequest;

    auto& messages = it->second.messages;
    const int64_t size = (int64_t)messages.size();
    const int64_t resolved = index < 0 ? size + index : index;
    if (resolved < 0 || resolved >= size)
        return ChorusError::InvalidRequest;

    messages[(size_t)resolved].content = std::move(content);
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

std::optional<std::string> ChorusRuntime::render_prompt(
    const SessionId& session,
    const std::string& template_override,
    const std::vector<InjectedMessage>& inject,
    const GenerationConfigPatch& overrides
) const {
    assert_host_thread();
    if (!_engine)
        return std::nullopt;
    auto it = _histories.find(session);
    if (it == _histories.end())
        return std::nullopt;
    // The SAME fitting and the SAME layering a real turn would apply, so
    // inspection == consumption: the caller passes the overrides it generates
    // with, host defaults resolve underneath exactly as they would on submit,
    // and the resolved max_tokens and show_thinking drive the reservation.
    GenerationRequest probe_request;
    probe_request.session_id = session; // a render is a chat turn: ambient chat controls apply
    probe_request.chat_template = template_override;
    probe_request.inject = inject;
    probe_request.overrides = overrides;
    const ResolvedRequest resolved = resolve_request(probe_request);

    auto fitted = fit_turn_messages(resolved, it->second.messages);
    if (std::holds_alternative<SubmitResult>(fitted))
        return std::nullopt;
    auto& turn = std::get<FittedTurn>(fitted);
    if (turn.rendered_text) // fitting already rendered the winning candidate
        return std::move(turn.rendered_text);
    auto rendered =
        _engine->render_chat_prompt(turn.messages, resolved.chat_template, resolved.config.show_thinking.value_or(true));
    return rendered ? std::optional<std::string>(std::move(rendered->text)) : std::nullopt;
}

SubmitResult ChorusRuntime::regenerate(const GenerationRequest& request) {
    assert_host_thread();
    if (!is_loaded())
        return not_ready();
    if (!request.session_id || request.session_id->empty())
        return SubmitResult{-1, ChorusError::InvalidRequest, "regenerate() requires a session."};
    if (!request.prompt.empty())
        return SubmitResult{
            -1, ChorusError::InvalidRequest, "regenerate() takes no prompt; it rerolls the last assistant reply."
        };
    if (_request_by_session.count(*request.session_id))
        return SubmitResult{
            -1, ChorusError::SessionBusy, "Session '" + *request.session_id + "' already has a live request."
        };
    auto history_it = _histories.find(*request.session_id);
    if (history_it == _histories.end() || history_it->second.messages.empty() ||
        history_it->second.messages.back().role != "assistant")
        return SubmitResult{
            -1, ChorusError::InvalidRequest, "regenerate() needs a conversation ending in an assistant reply."
        };

    const ResolvedRequest resolved = resolve_request(request);

    std::vector<ChatMessage> prospective(history_it->second.messages.begin(), history_it->second.messages.end() - 1);
    auto fitted = fit_turn_messages(resolved, std::move(prospective));
    if (std::holds_alternative<SubmitResult>(fitted))
        return std::get<SubmitResult>(fitted);
    auto& turn = std::get<FittedTurn>(fitted);

    ChorusRequest engine_request = make_engine_request(resolved);
    engine_request.messages = std::move(turn.messages);

    // The pop happens inside submit_engine_request's commit point, after
    // validation and before dispatch.
    return submit_engine_request(resolved, std::move(engine_request), turn.dropped, history_it->second.messages.back());
}

std::variant<ChorusRuntime::FittedTurn, SubmitResult>
ChorusRuntime::fit_turn_messages(const ResolvedRequest& resolved, std::vector<ChatMessage> prospective) const {
    const GenerationRequest& request = resolved.request;
    const bool show_thinking = resolved.config.show_thinking.value_or(true);

    auto info = _engine->loaded_model_info();
    if (!info || !info->per_request_context) // no budget known: fitting doesn't apply
        return FittedTurn{place_injections(std::move(prospective), request.inject), 0};

    const auto& max_tokens = resolved.config.max_tokens;
    const int32_t reservation =
        (max_tokens.has_value() && *max_tokens >= 0) ? *max_tokens : kFallbackResponseReservation;
    // Clamp before the signed subtraction; a window over INT32_MAX would wrap
    // the budget negative and reject every turn.
    const int32_t context = (int32_t)std::min<uint32_t>(*info->per_request_context, (uint32_t)INT32_MAX);
    const int32_t budget = context - reservation;
    if (budget <= 0)
        return SubmitResult{
            -1,
            ChorusError::InvalidRequest,
            "Response reservation (" + std::to_string(reservation) + " tokens, from max_tokens or the " +
                std::to_string(kFallbackResponseReservation) +
                "-token default) leaves no prompt room in the per-request context (" + std::to_string(context) +
                " tokens)."
        };

    // The probe keeps the text of its most recent successful render; when the
    // fit succeeds that is exactly the winning candidate's render, which
    // render_prompt can then reuse. A failed render invalidates it so a
    // skipped fit can never pair stale text with a different message list.
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

    auto fit = fit_messages_to_budget(prospective, request.inject, budget, probe);
    if (std::holds_alternative<ChorusError>(fit))
        return SubmitResult{
            -1, std::get<ChorusError>(fit), "Conversation does not fit the context window even after truncation."
        };
    auto& result = std::get<FitResult>(fit);
    return FittedTurn{std::move(result.fitted), result.dropped, std::move(last_render)};
}

void ChorusRuntime::finish_turn(const LiveRequest& live, TurnOutcome outcome, const std::string& text) {
    if (!live.session_id)
        return;
    auto history_it = _histories.find(*live.session_id);
    if (history_it == _histories.end())
        return; // defensive: lane vanished (should be impossible under busy-mutation rule)
    auto& conversation = history_it->second;
    conversation.last_turn_outcome = outcome;

    if (outcome == TurnOutcome::Completed) {
        conversation.messages.push_back({"assistant", text});
        return;
    }
    // Cancelled/Errored: undo this turn's mutation. SessionBusy guarantees the
    // trailing message is exactly what this turn added.
    if (live.replaced_reply) {
        conversation.messages.push_back(*live.replaced_reply);
    } else if (!conversation.messages.empty() && conversation.messages.back().role == "user") {
        conversation.messages.pop_back();
    }
}

} // namespace Chorus

#include "chorus_c/chorus_c.hpp"

#include <cstdlib>
#include <exception>
#include <utility>
#include <vector>

using namespace chorus_c;

namespace {

chorus_message_role to_c_role(Chorus::MessageRole role) noexcept {
    switch (role) {
    case Chorus::MessageRole::System:
        return CHORUS_ROLE_SYSTEM;
    case Chorus::MessageRole::User:
        return CHORUS_ROLE_USER;
    case Chorus::MessageRole::Assistant:
        return CHORUS_ROLE_ASSISTANT;
    }
    return CHORUS_ROLE_USER;
}

chorus_turn_outcome to_c_outcome(Chorus::TurnOutcome outcome) noexcept {
    switch (outcome) {
    case Chorus::TurnOutcome::None:
        return CHORUS_TURN_NONE;
    case Chorus::TurnOutcome::Completed:
        return CHORUS_TURN_COMPLETED;
    case Chorus::TurnOutcome::Cancelled:
        return CHORUS_TURN_CANCELLED;
    case Chorus::TurnOutcome::Errored:
        return CHORUS_TURN_ERRORED;
    }
    return CHORUS_TURN_NONE;
}

chorus_error runtime_result(const chorus_runtime* rt, Chorus::ChorusError error) noexcept {
    const chorus_error mapped = to_c_error(error);
    replace_last_error(rt, error_name(mapped));
    return mapped;
}

void free_conversation_messages(chorus_conversation_message* messages, size_t count) noexcept {
    if (!messages)
        return;
    for (size_t i = 0; i < count; ++i)
        std::free(const_cast<char*>(messages[i].content));
    std::free(messages);
}

void free_string_list(char** strings, size_t count) noexcept {
    if (!strings)
        return;
    for (size_t i = 0; i < count; ++i)
        std::free(strings[i]);
    std::free(static_cast<void*>(strings));
}

} // namespace

extern "C" {

chorus_error chorus_history_import(
    chorus_runtime* rt, const char* session, const chorus_conversation_message* history, size_t count
) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!session || (count != 0 && !history))
        return invalid_request(rt, "session and history storage are required.");

    try {
        std::vector<Chorus::ConversationMessage> copied;
        copied.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto role = to_cpp_role(history[i].role);
            if (!role || !history[i].content || history[i].id < 0)
                return invalid_request(rt, "Every history message requires a nonnegative ID, known role, and content.");
            copied.push_back({history[i].id, {*role, Chorus::MessageContent::text(history[i].content)}});
        }
        auto error = rt->value.import_conversation_history(session, std::move(copied));
        if (error)
            return runtime_result(rt, *error);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while importing history.");
    }
}

chorus_error chorus_history_export(
    const chorus_runtime* rt, const char* session, chorus_conversation_message** out_messages, size_t* out_count
) {
    if (out_messages)
        *out_messages = nullptr;
    if (out_count)
        *out_count = 0;
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!session || !out_messages || !out_count)
        return invalid_request(rt, "session, out_messages, and out_count are required.");

    try {
        const auto history = rt->value.export_conversation_history(session);
        bool known = !history.empty();
        if (!known) {
            for (const auto& candidate : rt->value.list_conversations()) {
                if (candidate == session) {
                    known = true;
                    break;
                }
            }
        }
        if (!known) {
            clear_last_error(rt);
            return CHORUS_OK;
        }

        const size_t allocation_count = history.empty() ? 1 : history.size();
        auto* output = static_cast<chorus_conversation_message*>(
            std::calloc(allocation_count, sizeof(chorus_conversation_message))
        );
        if (!output)
            return unknown_exception(rt, "Unable to allocate the history snapshot.");

        for (size_t i = 0; i < history.size(); ++i) {
            const auto content = Chorus::joined_text(history[i].message.content);
            output[i].id = history[i].id;
            output[i].role = to_c_role(history[i].message.role);
            output[i].content = content ? copy_owned_string(*content) : nullptr;
            if (!content || !output[i].content) {
                free_conversation_messages(output, history.size());
                return unknown_exception(rt, "Unable to allocate the history snapshot.");
            }
        }

        *out_messages = output;
        *out_count = history.size();
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while exporting history.");
    }
}

void chorus_conversation_messages_free(chorus_conversation_message* messages, size_t count) {
    free_conversation_messages(messages, count);
}

chorus_error chorus_history_clear(chorus_runtime* rt, const char* session) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!session)
        return invalid_request(rt, "session is required.");
    try {
        auto error = rt->value.clear_conversation_history(session);
        if (error)
            return runtime_result(rt, *error);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while clearing history.");
    }
}

chorus_error chorus_history_edit_message(
    chorus_runtime* rt, const char* session, chorus_message_id message_id, const char* content
) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!session || !content || message_id < 0)
        return invalid_request(rt, "session, nonnegative message_id, and content are required.");
    try {
        auto error = rt->value.edit_message(session, message_id, Chorus::MessageContent::text(content));
        if (error)
            return runtime_result(rt, *error);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while editing history.");
    }
}

chorus_error chorus_list_conversations(const chorus_runtime* rt, char*** out_sessions, size_t* out_count) {
    if (out_sessions)
        *out_sessions = nullptr;
    if (out_count)
        *out_count = 0;
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!out_sessions || !out_count)
        return invalid_request(rt, "out_sessions and out_count are required.");

    try {
        const auto conversations = rt->value.list_conversations();
        if (conversations.empty()) {
            clear_last_error(rt);
            return CHORUS_OK;
        }
        auto* output = static_cast<char**>(std::calloc(conversations.size(), sizeof(char*)));
        if (!output)
            return unknown_exception(rt, "Unable to allocate the conversation list.");
        for (size_t i = 0; i < conversations.size(); ++i) {
            output[i] = copy_owned_string(conversations[i]);
            if (!output[i]) {
                free_string_list(output, conversations.size());
                return unknown_exception(rt, "Unable to allocate the conversation list.");
            }
        }
        *out_sessions = output;
        *out_count = conversations.size();
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while listing conversations.");
    }
}

void chorus_string_list_free(char** strings, size_t count) {
    free_string_list(strings, count);
}

chorus_error chorus_reset_context(chorus_runtime* rt) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    try {
        auto error = rt->value.reset_context();
        if (error)
            return runtime_result(rt, *error);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while resetting context.");
    }
}

chorus_turn_outcome chorus_last_turn_outcome(const chorus_runtime* rt, const char* session) {
    if (!rt || !session) {
        if (rt)
            invalid_request(rt, "session is required.");
        return CHORUS_TURN_NONE;
    }
    try {
        return to_c_outcome(rt->value.last_turn_outcome(session));
    } catch (const std::exception& error) {
        unknown_exception(rt, error.what());
        return CHORUS_TURN_NONE;
    } catch (...) {
        unknown_exception(rt, "Unknown exception while reading the turn outcome.");
        return CHORUS_TURN_NONE;
    }
}

} // extern "C"

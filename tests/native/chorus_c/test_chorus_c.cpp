#include "chorus_c/chorus_c.h"
#include "test_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

extern "C" int chorus_c_header_smoke(void);

namespace {

struct RuntimeDeleter {
    void operator()(chorus_runtime* runtime) const { chorus_runtime_free(runtime); }
};

struct OptionsDeleter {
    void operator()(chorus_options* options) const { chorus_options_free(options); }
};

struct RequestDeleter {
    void operator()(chorus_request* request) const { chorus_request_free(request); }
};

using RuntimePtr = std::unique_ptr<chorus_runtime, RuntimeDeleter>;
using OptionsPtr = std::unique_ptr<chorus_options, OptionsDeleter>;
using RequestPtr = std::unique_ptr<chorus_request, RequestDeleter>;

struct ChatSnapshot {
    chorus_chat_message* messages = nullptr;
    size_t count = 0;

    ~ChatSnapshot() { chorus_chat_messages_free(messages, count); }
};

struct ConversationList {
    char** sessions = nullptr;
    size_t count = 0;

    ~ConversationList() { chorus_string_list_free(sessions, count); }
};

struct OwnedEvent {
    chorus_event_kind kind;
    chorus_request_id request_id;
    bool has_session;
    std::string session;
    bool has_text;
    std::string text;
    chorus_error error;
    bool has_reasoning;
    std::string reasoning;
    int32_t dropped;
};

struct PolledEvents {
    std::vector<OwnedEvent> events;
    bool saw_null_array = false;
    bool saw_terminal = false;
};

OwnedEvent copy_event(const chorus_event& event) {
    return OwnedEvent{
        event.kind,
        event.request_id,
        event.session != nullptr,
        event.session != nullptr ? event.session : "",
        event.text != nullptr,
        event.text != nullptr ? event.text : "",
        event.error,
        event.reasoning != nullptr,
        event.reasoning != nullptr ? event.reasoning : "",
        event.dropped,
    };
}

PolledEvents poll_until_terminal(chorus_runtime* runtime, chorus_request_id request_id) {
    PolledEvents result;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline && !result.saw_terminal) {
        size_t count = 0;
        const chorus_event* events = chorus_poll(runtime, &count);
        if (events == nullptr) {
            result.saw_null_array = true;
            break;
        }
        for (size_t index = 0; index < count; ++index) {
            result.events.push_back(copy_event(events[index]));
            if (events[index].request_id == request_id &&
                (events[index].kind == CHORUS_EVENT_COMPLETE || events[index].kind == CHORUS_EVENT_ERROR))
                result.saw_terminal = true;
        }
        if (!result.saw_terminal)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return result;
}

void test_chorus_c_header_and_error_vocabulary() {
    ASSERT_EQ(chorus_c_header_smoke(), 0);
    ASSERT_EQ(chorus_abi_version(), uint32_t{1});

    const struct {
        chorus_error error;
        const char* name;
    } cases[] = {
        {CHORUS_OK, "None"},
        {CHORUS_ERR_MODEL_LOAD, "ModelLoad"},
        {CHORUS_ERR_CONTEXT_INIT, "ContextInit"},
        {CHORUS_ERR_DECODE, "Decode"},
        {CHORUS_ERR_TOKENIZE, "Tokenize"},
        {CHORUS_ERR_INVALID_REQUEST, "InvalidRequest"},
        {CHORUS_ERR_ENGINE_NOT_READY, "EngineNotReady"},
        {CHORUS_ERR_CANCELLED, "Cancelled"},
        {CHORUS_ERR_UNSUPPORTED_MODEL_FORMAT, "UnsupportedModelFormat"},
        {CHORUS_ERR_UNSUPPORTED_FEATURE, "UnsupportedFeature"},
        {CHORUS_ERR_UNSUPPORTED_OPTION, "UnsupportedOption"},
        {CHORUS_ERR_SESSION_BUSY, "SessionBusy"},
        {CHORUS_ERR_UNKNOWN, "Unknown"},
    };
    for (const auto& item : cases) {
        ASSERT_TRUE(chorus_error_name(item.error) != nullptr);
        ASSERT_TRUE(std::strcmp(chorus_error_name(item.error), item.name) == 0);
    }
    ASSERT_TRUE(std::strcmp(chorus_error_name(static_cast<chorus_error>(999)), "Unknown") == 0);
}

void test_chorus_c_load_stop_and_empty_polls() {
    RuntimePtr runtime(chorus_runtime_new());
    ASSERT_TRUE(runtime != nullptr);
    ASSERT_TRUE(!chorus_is_loaded(runtime.get()));

    size_t event_count = 99;
    const chorus_event* events = chorus_poll(runtime.get(), &event_count);
    ASSERT_TRUE(events != nullptr);
    ASSERT_EQ(event_count, size_t{0});

    size_t log_count = 99;
    const chorus_log_record* logs = chorus_poll_logs(runtime.get(), &log_count);
    ASSERT_TRUE(logs != nullptr);
    ASSERT_EQ(log_count, size_t{0});

    ASSERT_EQ(
        chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_LEVEL_DEFAULT), CHORUS_OK
    );
    ASSERT_TRUE(chorus_is_loaded(runtime.get()));

    chorus_stop_all(runtime.get());
    ASSERT_TRUE(!chorus_is_loaded(runtime.get()));

    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF), CHORUS_OK);
    ASSERT_TRUE(chorus_is_loaded(runtime.get()));
}

void test_chorus_c_builder_streams_and_completes() {
    RuntimePtr runtime(chorus_runtime_new());
    ASSERT_TRUE(runtime != nullptr);
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF), CHORUS_OK);

    RequestPtr request(chorus_request_new());
    ASSERT_TRUE(request != nullptr);
    char prompt[] = "hello C adapter";
    ASSERT_EQ(chorus_request_set_prompt(request.get(), prompt), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_priority(request.get(), 7), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_stream(request.get(), true), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_max_tokens(request.get(), -1), CHORUS_OK);
    prompt[0] = 'j';

    chorus_request_id request_id = -1;
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &request_id), CHORUS_OK);
    ASSERT_TRUE(request_id >= 0);
    ASSERT_TRUE(chorus_is_request_active(runtime.get(), request_id));

    const PolledEvents streamed = poll_until_terminal(runtime.get(), request_id);
    ASSERT_TRUE(!streamed.saw_null_array);
    ASSERT_TRUE(streamed.saw_terminal);
    ASSERT_EQ(streamed.events.size(), size_t{4});
    ASSERT_EQ(streamed.events[0].kind, CHORUS_EVENT_TOKEN);
    ASSERT_TRUE(!streamed.events[0].has_session);
    ASSERT_TRUE(streamed.events[0].has_text);
    ASSERT_EQ(streamed.events[0].text, std::string("hello "));
    ASSERT_TRUE(!streamed.events[0].has_reasoning);
    ASSERT_EQ(streamed.events[1].text, std::string("C "));
    ASSERT_EQ(streamed.events[2].text, std::string("adapter"));
    ASSERT_EQ(streamed.events[3].kind, CHORUS_EVENT_COMPLETE);
    ASSERT_TRUE(streamed.events[3].has_text);
    ASSERT_EQ(streamed.events[3].text, std::string("hello C adapter"));
    ASSERT_TRUE(streamed.events[3].has_reasoning);
    ASSERT_EQ(streamed.events[3].reasoning, std::string{});
    ASSERT_TRUE(!chorus_is_request_active(runtime.get(), request_id));

    ASSERT_EQ(chorus_request_set_prompt(request.get(), "again"), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_stream(request.get(), false), CHORUS_OK);
    chorus_request_id second_id = -1;
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &second_id), CHORUS_OK);
    const PolledEvents nonstreamed = poll_until_terminal(runtime.get(), second_id);
    ASSERT_TRUE(nonstreamed.saw_terminal);
    ASSERT_EQ(nonstreamed.events.size(), size_t{1});
    ASSERT_EQ(nonstreamed.events[0].kind, CHORUS_EVENT_COMPLETE);
    ASSERT_EQ(nonstreamed.events[0].text, std::string("again"));

    size_t idle_count = 99;
    ASSERT_TRUE(chorus_poll(runtime.get(), &idle_count) != nullptr);
    ASSERT_EQ(idle_count, size_t{0});
}

void test_chorus_c_rejections_replace_and_success_clears_last_error() {
    RuntimePtr runtime(chorus_runtime_new());
    RequestPtr request(chorus_request_new());
    ASSERT_TRUE(runtime != nullptr);
    ASSERT_TRUE(request != nullptr);
    ASSERT_EQ(chorus_request_set_prompt(request.get(), "hello"), CHORUS_OK);

    chorus_request_id request_id = 42;
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &request_id), CHORUS_ERR_ENGINE_NOT_READY);
    ASSERT_EQ(request_id, chorus_request_id{-1});
    ASSERT_TRUE(chorus_last_error_message(runtime.get()) != nullptr);
    ASSERT_TRUE(std::strlen(chorus_last_error_message(runtime.get())) > 0);

    OptionsPtr options(chorus_options_new());
    ASSERT_TRUE(options != nullptr);
    ASSERT_EQ(chorus_options_set_int(options.get(), "integer", 1), CHORUS_OK);
    ASSERT_EQ(chorus_options_set_float(options.get(), "float", 1.5), CHORUS_OK);
    ASSERT_EQ(chorus_options_set_bool(options.get(), "bool", true), CHORUS_OK);
    ASSERT_EQ(chorus_options_set_string(options.get(), "string", "copied"), CHORUS_OK);
    ASSERT_EQ(
        chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, options.get(), CHORUS_LOG_OFF),
        CHORUS_ERR_UNSUPPORTED_OPTION
    );
    ASSERT_TRUE(std::strlen(chorus_last_error_message(runtime.get())) > 0);

    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF), CHORUS_OK);
    ASSERT_EQ(std::strlen(chorus_last_error_message(runtime.get())), size_t{0});

    ASSERT_EQ(chorus_request_set_session(request.get(), ""), CHORUS_OK);
    request_id = 42;
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &request_id), CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(request_id, chorus_request_id{-1});
    ASSERT_TRUE(std::strlen(chorus_last_error_message(runtime.get())) > 0);

    ASSERT_EQ(chorus_history_import(runtime.get(), "known", nullptr, 0), CHORUS_OK);
    ASSERT_EQ(std::strlen(chorus_last_error_message(runtime.get())), size_t{0});

    RequestPtr provider_request(chorus_request_new());
    ASSERT_TRUE(provider_request != nullptr);
    ASSERT_EQ(chorus_request_set_prompt(provider_request.get(), "provider options"), CHORUS_OK);
    ASSERT_EQ(
        chorus_request_set_provider_option_bool(provider_request.get(), "echo", "enabled", true), CHORUS_OK
    );
    ASSERT_EQ(
        chorus_request_set_provider_option_string(provider_request.get(), "echo", "mode", "strict"), CHORUS_OK
    );
    request_id = 42;
    ASSERT_EQ(chorus_generate(runtime.get(), provider_request.get(), &request_id), CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(request_id, chorus_request_id{-1});
    ASSERT_TRUE(std::strlen(chorus_last_error_message(runtime.get())) > 0);

    ASSERT_EQ(chorus_history_clear(runtime.get(), "known"), CHORUS_OK);
    ASSERT_EQ(std::strlen(chorus_last_error_message(runtime.get())), size_t{0});
}

void test_chorus_c_cancellation_stays_active_until_terminal_poll() {
    RuntimePtr runtime(chorus_runtime_new());
    ASSERT_TRUE(runtime != nullptr);
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF), CHORUS_OK);

    const chorus_chat_message baseline[] = {{"system", "persona"}};
    ASSERT_EQ(chorus_history_import(runtime.get(), "cancelled", baseline, 1), CHORUS_OK);

    std::string long_prompt;
    long_prompt.reserve(100000);
    for (int index = 0; index < 20000; ++index)
        long_prompt += "word ";

    RequestPtr blocker(chorus_request_new());
    ASSERT_TRUE(blocker != nullptr);
    ASSERT_EQ(chorus_request_set_prompt(blocker.get(), long_prompt.c_str()), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_stream(blocker.get(), false), CHORUS_OK);
    chorus_request_id blocker_id = -1;
    ASSERT_EQ(chorus_generate(runtime.get(), blocker.get(), &blocker_id), CHORUS_OK);

    RequestPtr target(chorus_request_new());
    ASSERT_TRUE(target != nullptr);
    ASSERT_EQ(chorus_request_set_prompt(target.get(), "cancel me"), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_session(target.get(), "cancelled"), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_stream(target.get(), false), CHORUS_OK);
    chorus_request_id target_id = -1;
    ASSERT_EQ(chorus_generate(runtime.get(), target.get(), &target_id), CHORUS_OK);
    ASSERT_TRUE(chorus_is_request_active(runtime.get(), target_id));
    ASSERT_EQ(chorus_active_request_for_session(runtime.get(), "cancelled"), target_id);

    ASSERT_TRUE(chorus_cancel(runtime.get(), target_id));
    ASSERT_TRUE(chorus_is_request_active(runtime.get(), target_id));
    ASSERT_EQ(chorus_active_request_for_session(runtime.get(), "cancelled"), target_id);
    ASSERT_TRUE(chorus_cancel(runtime.get(), blocker_id));

    const PolledEvents polled = poll_until_terminal(runtime.get(), target_id);
    ASSERT_TRUE(polled.saw_terminal);
    const auto terminal = std::find_if(polled.events.begin(), polled.events.end(), [target_id](const OwnedEvent& event) {
        return event.request_id == target_id && event.kind == CHORUS_EVENT_ERROR;
    });
    ASSERT_TRUE(terminal != polled.events.end());
    ASSERT_EQ(terminal->error, CHORUS_ERR_CANCELLED);
    ASSERT_TRUE(terminal->has_session);
    ASSERT_EQ(terminal->session, std::string("cancelled"));
    ASSERT_TRUE(!chorus_is_request_active(runtime.get(), target_id));
    ASSERT_EQ(chorus_active_request_for_session(runtime.get(), "cancelled"), chorus_request_id{-1});
    ASSERT_TRUE(!chorus_cancel(runtime.get(), target_id));
    ASSERT_EQ(chorus_last_turn_outcome(runtime.get(), "cancelled"), CHORUS_TURN_CANCELLED);

    ChatSnapshot history;
    ASSERT_EQ(chorus_history_export(runtime.get(), "cancelled", &history.messages, &history.count), CHORUS_OK);
    ASSERT_EQ(history.count, size_t{1});
    ASSERT_TRUE(std::strcmp(history.messages[0].role, "system") == 0);
    ASSERT_TRUE(std::strcmp(history.messages[0].content, "persona") == 0);
}

void test_chorus_c_conversation_snapshots_edits_and_outcome() {
    RuntimePtr runtime(chorus_runtime_new());
    ASSERT_TRUE(runtime != nullptr);
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF), CHORUS_OK);

    char role[] = "system";
    char content[] = "persona";
    const chorus_chat_message imported[] = {{role, content}, {"user", "old question"}, {"assistant", "old answer"}};
    ASSERT_EQ(chorus_history_import(runtime.get(), "npc", imported, 3), CHORUS_OK);
    content[0] = 'X';
    ASSERT_EQ(chorus_last_turn_outcome(runtime.get(), "npc"), CHORUS_TURN_NONE);

    ConversationList conversations;
    ASSERT_EQ(chorus_list_conversations(runtime.get(), &conversations.sessions, &conversations.count), CHORUS_OK);
    ASSERT_EQ(conversations.count, size_t{1});
    ASSERT_TRUE(std::strcmp(conversations.sessions[0], "npc") == 0);

    {
        ChatSnapshot exported;
        ASSERT_EQ(chorus_history_export(runtime.get(), "npc", &exported.messages, &exported.count), CHORUS_OK);
        ASSERT_EQ(exported.count, size_t{3});
        ASSERT_TRUE(std::strcmp(exported.messages[0].content, "persona") == 0);
    }

    ASSERT_EQ(chorus_history_edit_message(runtime.get(), "npc", -1, "edited answer"), CHORUS_OK);
    {
        ChatSnapshot edited;
        ASSERT_EQ(chorus_history_export(runtime.get(), "npc", &edited.messages, &edited.count), CHORUS_OK);
        ASSERT_EQ(edited.count, size_t{3});
        ASSERT_TRUE(std::strcmp(edited.messages[2].role, "assistant") == 0);
        ASSERT_TRUE(std::strcmp(edited.messages[2].content, "edited answer") == 0);
    }

    ASSERT_EQ(chorus_history_clear(runtime.get(), "npc"), CHORUS_OK);
    {
        ChatSnapshot cleared;
        ASSERT_EQ(chorus_history_export(runtime.get(), "npc", &cleared.messages, &cleared.count), CHORUS_OK);
        ASSERT_EQ(cleared.count, size_t{0});
        ASSERT_TRUE(cleared.messages == nullptr);
    }

    const chorus_chat_message baseline[] = {{"system", "persona"}};
    ASSERT_EQ(chorus_history_import(runtime.get(), "npc", baseline, 1), CHORUS_OK);
    RequestPtr request(chorus_request_new());
    ASSERT_TRUE(request != nullptr);
    ASSERT_EQ(chorus_request_set_prompt(request.get(), "hello history"), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_session(request.get(), "npc"), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_stream(request.get(), false), CHORUS_OK);
    chorus_request_id request_id = -1;
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &request_id), CHORUS_OK);

    const PolledEvents completed = poll_until_terminal(runtime.get(), request_id);
    ASSERT_TRUE(completed.saw_terminal);
    ASSERT_EQ(completed.events.size(), size_t{1});
    ASSERT_EQ(completed.events[0].kind, CHORUS_EVENT_COMPLETE);
    ASSERT_EQ(completed.events[0].text, std::string("hello history"));
    ASSERT_EQ(chorus_last_turn_outcome(runtime.get(), "npc"), CHORUS_TURN_COMPLETED);

    ChatSnapshot history;
    ASSERT_EQ(chorus_history_export(runtime.get(), "npc", &history.messages, &history.count), CHORUS_OK);
    ASSERT_EQ(history.count, size_t{3});
    ASSERT_TRUE(std::strcmp(history.messages[1].role, "user") == 0);
    ASSERT_TRUE(std::strcmp(history.messages[1].content, "hello history") == 0);
    ASSERT_TRUE(std::strcmp(history.messages[2].role, "assistant") == 0);
    ASSERT_TRUE(std::strcmp(history.messages[2].content, "hello history") == 0);

    char* rendered = chorus_render_prompt(runtime.get(), "npc", nullptr);
    ASSERT_TRUE(rendered == nullptr);
}

void test_chorus_c_log_polling_exposes_owned_record_shape() {
    RuntimePtr runtime(chorus_runtime_new());
    ASSERT_TRUE(runtime != nullptr);
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_WARN), CHORUS_OK);

    size_t count = 0;
    ASSERT_TRUE(chorus_poll_logs(runtime.get(), &count) != nullptr);
    ASSERT_EQ(count, size_t{0});

    RequestPtr request(chorus_request_new());
    ASSERT_TRUE(request != nullptr);
    ASSERT_EQ(chorus_request_set_prompt(request.get(), "log shape"), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_temperature(request.get(), 0.5F), CHORUS_OK);
    chorus_request_id request_id = -1;
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &request_id), CHORUS_OK);

    const chorus_log_record* records = chorus_poll_logs(runtime.get(), &count);
    ASSERT_TRUE(records != nullptr);
    ASSERT_TRUE(count > 0);
    bool saw_warning = false;
    for (size_t record_index = 0; record_index < count; ++record_index) {
        const chorus_log_record& record = records[record_index];
        ASSERT_TRUE(record.message != nullptr);
        ASSERT_TRUE(record.produced_at > 0.0);
        if (record.level == CHORUS_LOG_WARN)
            saw_warning = true;
        if (record.field_count == 0)
            continue;
        ASSERT_TRUE(record.fields != nullptr);
        for (size_t field_index = 0; field_index < record.field_count; ++field_index) {
            ASSERT_TRUE(record.fields[field_index].key != nullptr);
            if (record.fields[field_index].type == CHORUS_FIELD_STRING)
                ASSERT_TRUE(record.fields[field_index].value.string_value != nullptr);
        }
    }
    ASSERT_TRUE(saw_warning);

    ASSERT_TRUE(chorus_poll_logs(runtime.get(), &count) != nullptr);
    ASSERT_EQ(count, size_t{0});

    ASSERT_TRUE(chorus_cancel(runtime.get(), request_id));
    (void)poll_until_terminal(runtime.get(), request_id);
}

} // namespace

int run_chorus_c_tests() {
    std::cout << "\n--- CHORUS C ABI SUITE ---\n";

    run_test("ChorusC_header_and_error_vocabulary", test_chorus_c_header_and_error_vocabulary);
    run_test("ChorusC_load_stop_and_empty_polls", test_chorus_c_load_stop_and_empty_polls);
    run_test("ChorusC_builder_streams_and_completes", test_chorus_c_builder_streams_and_completes);
    run_test(
        "ChorusC_rejections_replace_and_success_clears_last_error",
        test_chorus_c_rejections_replace_and_success_clears_last_error
    );
    run_test(
        "ChorusC_cancellation_stays_active_until_terminal_poll",
        test_chorus_c_cancellation_stays_active_until_terminal_poll
    );
    run_test(
        "ChorusC_conversation_snapshots_edits_and_outcome", test_chorus_c_conversation_snapshots_edits_and_outcome
    );
    run_test("ChorusC_log_polling_exposes_owned_record_shape", test_chorus_c_log_polling_exposes_owned_record_shape);

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << RED << "SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED." << RESET
                  << "\n";
        return 1;
    }
    std::cout << GREEN << "SUMMARY: ALL TESTS PASSED." << RESET << "\n";
    return 0;
}

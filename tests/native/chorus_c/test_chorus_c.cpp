#include "chorus_c/chorus_c.h"
#include "gtest_utils.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

extern "C" int chorus_c_header_smoke(void);

namespace {

struct RuntimeDeleter {
    void operator()(chorus_runtime* runtime) const { chorus_runtime_free(runtime); }
};
struct RequestDeleter {
    void operator()(chorus_request* request) const { chorus_request_free(request); }
};
using RuntimePtr = std::unique_ptr<chorus_runtime, RuntimeDeleter>;
using RequestPtr = std::unique_ptr<chorus_request, RequestDeleter>;

RuntimePtr loaded_runtime() {
    RuntimePtr runtime(chorus_runtime_new());
    EXPECT_TRUE(runtime != nullptr);
    if (runtime) {
        chorus_load_result result{};
        EXPECT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF, &result), CHORUS_OK);
        EXPECT_EQ(result.error, CHORUS_OK);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        bool loaded = false;
        while (!loaded && std::chrono::steady_clock::now() < deadline) {
            size_t count = 0;
            const auto* events = chorus_poll(runtime.get(), &count);
            for (size_t i = 0; i < count; ++i) {
                EXPECT_EQ(events[i].load_id, result.load_id);
                EXPECT_NE(events[i].kind, CHORUS_EVENT_MODEL_LOAD_FAILED);
                loaded |= events[i].kind == CHORUS_EVENT_MODEL_LOADED;
            }
            if (!loaded) std::this_thread::yield();
        }
        EXPECT_TRUE(loaded);
    }
    return runtime;
}

RequestPtr request_with_prompt(const char* prompt) {
    RequestPtr request(chorus_request_new());
    EXPECT_TRUE(request != nullptr);
    if (request)
        EXPECT_EQ(chorus_request_set_prompt(request.get(), prompt), CHORUS_OK);
    return request;
}

struct EventSnapshot {
    chorus_request_id id;
    chorus_event_kind kind;
    chorus_error error;
    std::string text;
};

std::vector<EventSnapshot> wait_events(chorus_runtime* runtime, size_t expected = 1) {
    std::vector<EventSnapshot> result;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (result.size() < expected && std::chrono::steady_clock::now() < deadline) {
        size_t count = 0;
        const auto* events = chorus_poll(runtime, &count);
        for (size_t i = 0; i < count; ++i)
            result.push_back({events[i].request_id, events[i].kind, events[i].error, events[i].text});
        std::this_thread::yield();
    }
    EXPECT_EQ(result.size(), expected);
    return result;
}

TEST(ChorusC, Header_is_pure_c_and_uses_abi_nine) {
    ASSERT_EQ(chorus_c_header_smoke(), 0);
    ASSERT_EQ(chorus_abi_version(), uint32_t{9});
}

TEST(ChorusC, Builder_local_clears_restore_fresh_echo_request_behavior) {
    RuntimePtr runtime = loaded_runtime();
    RequestPtr request = request_with_prompt("echoed");
    chorus_submit_result result{};
    auto generate = [&](chorus_error expected_error, const char* expected_text = nullptr) {
        EXPECT_EQ(chorus_generate(runtime.get(), request.get(), &result), CHORUS_OK);
        EXPECT_EQ(result.error, CHORUS_OK);
        if (result.error != CHORUS_OK)
            return;
        const auto events = wait_events(runtime.get());
        ASSERT_EQ(events.size(), 1U);
        EXPECT_EQ(events[0].error, expected_error);
        if (expected_text)
            EXPECT_EQ(events[0].text, expected_text);
    };

    ASSERT_EQ(chorus_request_set_max_tokens(request.get(), 0), CHORUS_OK);
    generate(CHORUS_OK, "");
    ASSERT_EQ(chorus_request_clear_max_tokens(request.get()), CHORUS_OK);
    generate(CHORUS_OK, "echoed");

    ASSERT_EQ(chorus_request_set_empty_stop(request.get()), CHORUS_OK);
    generate(CHORUS_OK, "echoed");
    ASSERT_EQ(chorus_request_add_stop(request.get(), "x"), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_stop(request.get()), CHORUS_OK);
    generate(CHORUS_OK, "echoed");
    ASSERT_EQ(chorus_request_add_stop(request.get(), "y"), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_set_empty_stop(request.get()), CHORUS_OK);
    generate(CHORUS_OK, "echoed");
    ASSERT_EQ(chorus_request_clear_stop(request.get()), CHORUS_OK);

    ASSERT_EQ(chorus_request_set_constraint(request.get(), CHORUS_CONSTRAINT_GBNF, "root ::= 'x'"), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_FEATURE);
    ASSERT_EQ(chorus_request_set_unconstrained(request.get()), CHORUS_OK);
    generate(CHORUS_OK, "echoed");
    ASSERT_EQ(chorus_request_clear_constraint(request.get()), CHORUS_OK);
    generate(CHORUS_OK, "echoed");

    ASSERT_EQ(chorus_request_set_temperature(request.get(), 0), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_temperature(request.get()), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_top_k(request.get(), 0), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_top_k(request.get()), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_top_p(request.get(), 0), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_top_p(request.get()), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_seed(request.get(), 0), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_seed(request.get()), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_frequency_penalty(request.get(), 0), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_frequency_penalty(request.get()), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_presence_penalty(request.get(), 0), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_presence_penalty(request.get()), CHORUS_OK);
    generate(CHORUS_OK, "echoed");

    ASSERT_EQ(chorus_request_set_provider_option_bool(request.get(), "echo", "flag", false), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_provider_option(request.get(), "echo", "flag"), CHORUS_OK);
    generate(CHORUS_OK, "echoed");
    ASSERT_EQ(chorus_request_set_provider_option_int(request.get(), "echo", "count", 0), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_provider_option_string(request.get(), "echo", "text", ""), CHORUS_OK);
    ASSERT_EQ(chorus_request_clear_provider_option(request.get(), "echo", "count"), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_provider_options(request.get()), CHORUS_OK);
    generate(CHORUS_OK, "echoed");
}

TEST(ChorusC, Chat_choices_clear_locally_and_invalid_builder_inputs_do_not_change_them) {
    RuntimePtr runtime = loaded_runtime();
    RequestPtr request = request_with_prompt("chat");
    ASSERT_EQ(chorus_request_set_session(request.get(), "npc"), CHORUS_OK);
    chorus_submit_result result{};
    auto generate = [&](chorus_error expected) {
        ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &result), CHORUS_OK);
        ASSERT_EQ(result.error, CHORUS_OK);
        const auto events = wait_events(runtime.get());
        ASSERT_EQ(events.size(), 1U);
        EXPECT_EQ(events[0].error, expected);
    };
    ASSERT_EQ(chorus_request_set_show_thinking(request.get(), false), CHORUS_OK);
    generate(CHORUS_OK);
    ASSERT_EQ(chorus_request_clear_show_thinking(request.get()), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_chat_template(request.get(), ""), CHORUS_OK);
    generate(CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(chorus_request_set_chat_template(request.get(), nullptr), CHORUS_ERR_INVALID_REQUEST);
    generate(CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(chorus_request_clear_chat_template(request.get()), CHORUS_OK);
    generate(CHORUS_OK);
    ASSERT_EQ(chorus_request_set_chat_template(request.get(), "template"), CHORUS_OK);
    generate(CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_chat_template(request.get()), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_constraint(request.get(), static_cast<chorus_constraint_format>(99), "x"), CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(chorus_request_set_constraint(request.get(), CHORUS_CONSTRAINT_GBNF, nullptr), CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(chorus_request_add_stop(request.get(), nullptr), CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(chorus_request_clear_provider_option(request.get(), nullptr, "key"), CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(chorus_request_clear_provider_option(request.get(), "echo", nullptr), CHORUS_ERR_INVALID_REQUEST);
    generate(CHORUS_OK);
}


TEST(ChorusC, Load_admission_cancellation_and_retry_are_identified_and_keep_event_storage) {
    RuntimePtr runtime = loaded_runtime();
    RequestPtr request = request_with_prompt("hello");
    chorus_submit_result submitted{};
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &submitted), CHORUS_OK);
    ASSERT_EQ(submitted.error, CHORUS_OK);
    auto generated = wait_events(runtime.get());
    ASSERT_EQ(generated[0].kind, CHORUS_EVENT_COMPLETE);

    chorus_load_result rejected{42, CHORUS_OK, nullptr};
    ASSERT_EQ(chorus_load(runtime.get(), static_cast<chorus_provider>(99), nullptr, nullptr, CHORUS_LOG_OFF, &rejected), CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(rejected.load_id, -1);
    ASSERT_EQ(rejected.error, CHORUS_ERR_INVALID_REQUEST);
    ASSERT_TRUE(chorus_is_loaded(runtime.get()));
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF, nullptr), CHORUS_ERR_INVALID_REQUEST);
    ASSERT_TRUE(chorus_is_loaded(runtime.get()));

    chorus_load_result cancelled{};
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF, &cancelled), CHORUS_OK);
    ASSERT_EQ(cancelled.error, CHORUS_OK);
    ASSERT_FALSE(chorus_is_loaded(runtime.get()));
    ASSERT_EQ(chorus_active_load_id(runtime.get()), cancelled.load_id);
    chorus_load_result overlap{};
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF, &overlap), CHORUS_OK);
    ASSERT_EQ(overlap.load_id, -1);
    ASSERT_EQ(overlap.error, CHORUS_ERR_INVALID_REQUEST);
    ASSERT_NE(overlap.message, nullptr);
    size_t log_count = 0;
    chorus_poll_logs(runtime.get(), &log_count);
    ASSERT_FALSE(std::string(overlap.message).empty());
    ASSERT_TRUE(chorus_cancel_load(runtime.get(), cancelled.load_id));
    bool failed = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!failed && std::chrono::steady_clock::now() < deadline) {
        size_t count = 0;
        const auto* events = chorus_poll(runtime.get(), &count);
        for (size_t i = 0; i < count; ++i) {
            if (events[i].kind != CHORUS_EVENT_MODEL_LOAD_FAILED) continue;
            EXPECT_EQ(events[i].load_id, cancelled.load_id);
            EXPECT_EQ(events[i].error, CHORUS_ERR_CANCELLED);
            EXPECT_STREQ(events[i].model_id, "");
            EXPECT_EQ(events[i].request_id, -1);
            failed = true;
        }
        if (!failed) std::this_thread::yield();
    }
    ASSERT_TRUE(failed);
    ASSERT_EQ(chorus_active_load_id(runtime.get()), -1);

    chorus_load_result retry{};
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF, &retry), CHORUS_OK);
    ASSERT_EQ(retry.error, CHORUS_OK);
    bool loaded = false;
    const auto retry_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!loaded && std::chrono::steady_clock::now() < retry_deadline) {
        size_t count = 0;
        const auto* events = chorus_poll(runtime.get(), &count);
        for (size_t i = 0; i < count; ++i) {
            if (events[i].kind != CHORUS_EVENT_MODEL_LOADED) continue;
            EXPECT_EQ(events[i].load_id, retry.load_id);
            EXPECT_EQ(events[i].request_id, -1);
            const char* model = events[i].model_id;
            ASSERT_NE(model, nullptr);
            size_t log_count = 0;
            chorus_poll_logs(runtime.get(), &log_count);
            EXPECT_STREQ(model, "");
            loaded = true;
        }
        if (!loaded) std::this_thread::yield();
    }
    ASSERT_TRUE(loaded);
    ASSERT_TRUE(chorus_is_loaded(runtime.get()));
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &submitted), CHORUS_OK);
    ASSERT_EQ(wait_events(runtime.get())[0].kind, CHORUS_EVENT_COMPLETE);
}

TEST(ChorusC, Waiting_then_polling_alone_delivers_a_generation_from_a_provider_thread) {
    auto runtime = loaded_runtime();
    auto request = request_with_prompt("hello");
    chorus_submit_result submitted{};
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &submitted), CHORUS_OK);

    bool complete = false;
    while (!complete) {
        ASSERT_TRUE(chorus_wait(runtime.get(), 5'000'000));
        size_t count = 0;
        const auto* events = chorus_poll(runtime.get(), &count);
        for (size_t i = 0; i < count; ++i)
            complete |= events[i].request_id == submitted.request_id && events[i].kind == CHORUS_EVENT_COMPLETE;
    }
}

TEST(ChorusC, Request_event_snapshot_survives_load_submission_and_log_poll) {
    RuntimePtr runtime = loaded_runtime();
    RequestPtr request = request_with_prompt("snapshot");
    chorus_submit_result submitted{};
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &submitted), CHORUS_OK);
    ASSERT_EQ(submitted.error, CHORUS_OK);
    const chorus_event* snapshot = nullptr;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!snapshot && std::chrono::steady_clock::now() < deadline) {
        size_t count = 0;
        const auto* events = chorus_poll(runtime.get(), &count);
        if (count && events[0].kind == CHORUS_EVENT_COMPLETE)
            snapshot = events;
        if (!snapshot) std::this_thread::yield();
    }
    ASSERT_NE(snapshot, nullptr);
    const char* text = snapshot[0].text;
    chorus_load_result load{};
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF, &load), CHORUS_OK);
    size_t log_count = 0;
    chorus_poll_logs(runtime.get(), &log_count);
    ASSERT_STREQ(text, "snapshot");
    ASSERT_EQ(snapshot[0].load_id, -1);
    ASSERT_TRUE(chorus_cancel_load(runtime.get(), load.load_id));
    chorus_stop_all(runtime.get());
}

TEST(ChorusC, Missing_model_fails_after_admission_and_diagnostic_survives_other_calls) {
    RuntimePtr runtime = loaded_runtime();
    chorus_load_result result{};
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_LLAMA, "tests/models/not-present.gguf", nullptr, CHORUS_LOG_OFF, &result), CHORUS_OK);
    ASSERT_EQ(result.error, CHORUS_OK);
    const auto id = result.load_id;
    ASSERT_FALSE(chorus_is_loaded(runtime.get()));
    bool failed = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!failed && std::chrono::steady_clock::now() < deadline) {
        size_t count = 0;
        const auto* events = chorus_poll(runtime.get(), &count);
        for (size_t i = 0; i < count; ++i) {
            if (events[i].kind != CHORUS_EVENT_MODEL_LOAD_FAILED) continue;
            ASSERT_EQ(events[i].load_id, id);
            ASSERT_EQ(events[i].error, CHORUS_ERR_MODEL_LOAD);
            ASSERT_NE(events[i].model_id, nullptr);
            ASSERT_STREQ(events[i].model_id, "tests/models/not-present.gguf");
            const char* text = events[i].text;
            ASSERT_NE(text, nullptr);
            size_t logs = 0;
            chorus_poll_logs(runtime.get(), &logs);
            ASSERT_FALSE(std::string(text).empty());
            failed = true;
        }
        if (!failed) std::this_thread::yield();
    }
    ASSERT_TRUE(failed);
    ASSERT_FALSE(chorus_is_loaded(runtime.get()));
}

TEST(ChorusC, Typed_injection_rejects_unknown_roles) {
    RequestPtr request = request_with_prompt("hello");
    ASSERT_EQ(
        chorus_request_add_inject(request.get(), static_cast<chorus_message_role>(99), "context", 0),
        CHORUS_ERR_INVALID_REQUEST
    );
    ASSERT_EQ(chorus_request_add_inject(request.get(), CHORUS_ROLE_SYSTEM, "context", 0), CHORUS_OK);
}

TEST(ChorusC, Generation_batch_initializes_every_slot_and_preserves_diagnostics) {
    RuntimePtr runtime = loaded_runtime();
    RequestPtr accepted = request_with_prompt("accepted");
    RequestPtr rejected = request_with_prompt("rejected");
    ASSERT_EQ(chorus_request_set_provider_option_bool(rejected.get(), "echo", "unsupported", true), CHORUS_OK);
    RequestPtr later = request_with_prompt("later");
    const chorus_request* requests[] = {accepted.get(), rejected.get(), later.get()};
    chorus_submit_result results[3] = {};

    ASSERT_EQ(chorus_generate_batch(runtime.get(), requests, 3, results), CHORUS_OK);
    ASSERT_EQ(results[0].error, CHORUS_OK);
    ASSERT_TRUE(results[0].request_id >= 0);
    ASSERT_EQ(results[0].request_message_id, chorus_message_id{-1});
    ASSERT_EQ(results[1].error, CHORUS_OK);
    ASSERT_TRUE(results[1].request_id >= 0);
    ASSERT_EQ(results[2].error, CHORUS_OK);
    ASSERT_TRUE(results[2].request_id >= 0);
    const auto events = wait_events(runtime.get(), 3);
    const auto failed = std::find_if(events.begin(), events.end(), [&](const auto& event) { return event.id == results[1].request_id; });
    ASSERT_NE(failed, events.end());
    ASSERT_EQ(failed->error, CHORUS_ERR_UNSUPPORTED_OPTION);
}

TEST(ChorusC, Batch_result_strings_survive_internal_singular_submissions) {
    RuntimePtr runtime = loaded_runtime();
    RequestPtr rejected = request_with_prompt("rejected");
    ASSERT_EQ(chorus_request_set_session(rejected.get(), ""), CHORUS_OK);
    RequestPtr accepted = request_with_prompt("accepted");
    const chorus_request* requests[] = {rejected.get(), accepted.get()};
    chorus_submit_result results[2] = {};

    ASSERT_EQ(chorus_generate_batch(runtime.get(), requests, 2, results), CHORUS_OK);
    ASSERT_EQ(results[0].error, CHORUS_ERR_INVALID_REQUEST);
    ASSERT_TRUE(results[0].message != nullptr);
    const std::string diagnostic = results[0].message;
    ASSERT_EQ(std::string(results[0].message), diagnostic);
    ASSERT_EQ(results[1].error, CHORUS_OK);
}

TEST(ChorusC, Embedding_batch_preserves_positions_and_rejects_invalid_entries) {
    RuntimePtr runtime = loaded_runtime();
    const chorus_embedding_request requests[] = {
        {"first", nullptr, 0, CHORUS_EXECUTION_SHARED},
        {"", nullptr, 0, CHORUS_EXECUTION_SHARED},
        {"third", nullptr, 0, CHORUS_EXECUTION_SHARED},
    };
    chorus_submit_result results[3] = {};

    ASSERT_EQ(chorus_embed_batch(runtime.get(), requests, 3, results), CHORUS_OK);
    ASSERT_EQ(results[0].error, CHORUS_OK);
    ASSERT_EQ(results[1].error, CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(results[2].error, CHORUS_OK);
    ASSERT_TRUE(results[0].request_id >= 0);
    ASSERT_TRUE(results[2].request_id >= 0);
}

TEST(ChorusC, Result_storage_serves_each_submission_render_and_poll_boundary) {
    RuntimePtr runtime = loaded_runtime();
    RequestPtr rejected = request_with_prompt("rejected");
    ASSERT_EQ(chorus_request_set_session(rejected.get(), ""), CHORUS_OK);
    chorus_submit_result submission{};
    ASSERT_EQ(chorus_generate(runtime.get(), rejected.get(), &submission), CHORUS_OK);
    ASSERT_EQ(submission.error, CHORUS_ERR_INVALID_REQUEST);
    ASSERT_NE(submission.message, nullptr);
    ASSERT_FALSE(std::string(submission.message).empty());

    RequestPtr rendered_request = request_with_prompt("rendered");
    chorus_submit_result rendered{};
    ASSERT_EQ(chorus_render_prompt(runtime.get(), rendered_request.get(), &rendered), CHORUS_OK);
    ASSERT_EQ(rendered.error, CHORUS_OK);
    ASSERT_EQ(chorus_request_set_prompt(rendered_request.get(), "mutated"), CHORUS_OK);
    rendered_request.reset();
    const auto events = wait_events(runtime.get());
    ASSERT_EQ(events.back().id, rendered.request_id);
    ASSERT_EQ(events.back().kind, CHORUS_EVENT_PROMPT_RENDERED);
    ASSERT_EQ(events.back().text, "rendered");
}

TEST(ChorusC, Typed_history_rejects_duplicate_and_negative_ids_atomically) {
    RuntimePtr runtime = loaded_runtime();
    const chorus_conversation_message original[] = {{7, CHORUS_ROLE_USER, "kept"}};
    ASSERT_EQ(chorus_history_import(runtime.get(), "npc", original, 1), CHORUS_OK);
    const chorus_conversation_message invalid[] = {
        {8, CHORUS_ROLE_USER, "not committed"}, {8, CHORUS_ROLE_ASSISTANT, "duplicate"}
    };
    ASSERT_EQ(chorus_history_import(runtime.get(), "npc", invalid, 2), CHORUS_ERR_INVALID_REQUEST);
    chorus_conversation_message* exported = nullptr;
    size_t count = 0;
    ASSERT_EQ(chorus_history_export(runtime.get(), "npc", &exported, &count), CHORUS_OK);
    ASSERT_EQ(count, size_t{1});
    ASSERT_EQ(exported[0].id, chorus_message_id{7});
    chorus_conversation_messages_free(exported, count);
    const chorus_conversation_message negative[] = {{-1, CHORUS_ROLE_USER, "negative"}};
    ASSERT_EQ(chorus_history_import(runtime.get(), "npc", negative, 1), CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(chorus_history_export(runtime.get(), "npc", &exported, &count), CHORUS_OK);
    ASSERT_EQ(count, size_t{1});
    chorus_conversation_messages_free(exported, count);
}

TEST(ChorusC, Typed_history_preserves_ids_and_edits_by_id) {
    RuntimePtr runtime = loaded_runtime();
    const chorus_conversation_message history[] = {
        {41, CHORUS_ROLE_SYSTEM, "persona"},
        {52, CHORUS_ROLE_USER, "question"},
        {63, CHORUS_ROLE_ASSISTANT, "answer"},
    };
    ASSERT_EQ(chorus_history_import(runtime.get(), "npc", history, 3), CHORUS_OK);
    ASSERT_EQ(chorus_history_edit_message(runtime.get(), "npc", 63, "edited"), CHORUS_OK);
    ASSERT_EQ(chorus_history_edit_message(runtime.get(), "npc", 2, "wrong"), CHORUS_ERR_INVALID_REQUEST);

    chorus_conversation_message* exported = nullptr;
    size_t count = 0;
    ASSERT_EQ(chorus_history_export(runtime.get(), "npc", &exported, &count), CHORUS_OK);
    ASSERT_EQ(count, size_t{3});
    ASSERT_EQ(exported[2].id, chorus_message_id{63});
    ASSERT_EQ(exported[2].role, CHORUS_ROLE_ASSISTANT);
    ASSERT_EQ(std::string(exported[2].content), std::string("edited"));
    chorus_conversation_messages_free(exported, count);
}

TEST(ChorusC, Sessioned_generation_reports_reserved_and_completion_message_ids) {
    RuntimePtr runtime = loaded_runtime();
    ASSERT_EQ(chorus_history_import(runtime.get(), "npc", nullptr, 0), CHORUS_OK);
    RequestPtr request = request_with_prompt("answer");
    ASSERT_EQ(chorus_request_set_session(request.get(), "npc"), CHORUS_OK);
    chorus_submit_result submission{};
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &submission), CHORUS_OK);
    ASSERT_EQ(submission.error, CHORUS_OK);
    ASSERT_TRUE(submission.request_message_id >= 0);
    ASSERT_TRUE(submission.response_message_id >= 0);

    size_t count = 0;
    const chorus_event* events = nullptr;
    for (int attempt = 0; attempt < 100 && count == 0; ++attempt) {
        events = chorus_poll(runtime.get(), &count);
        if (count == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    ASSERT_EQ(count, size_t{1});
    ASSERT_EQ(events[0].kind, CHORUS_EVENT_COMPLETE);
    ASSERT_EQ(events[0].message_id, submission.response_message_id);
}

TEST(ChorusC, Sessioned_embedding_reports_session_and_occupies_its_lane) {
    RuntimePtr runtime = loaded_runtime();
    const chorus_embedding_request embedding = {"memory", "npc", 0, CHORUS_EXECUTION_SHARED};
    chorus_submit_result result{};
    ASSERT_EQ(chorus_embed(runtime.get(), &embedding, &result), CHORUS_OK);
    ASSERT_EQ(result.error, CHORUS_OK);
    ASSERT_EQ(chorus_active_request_for_session(runtime.get(), "npc"), result.request_id);

    size_t count = 0;
    const chorus_event* events = nullptr;
    for (int attempt = 0; attempt < 100 && count == 0; ++attempt) {
        events = chorus_poll(runtime.get(), &count);
        if (count == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    ASSERT_TRUE(events != nullptr);
    ASSERT_EQ(count, size_t{1});
    ASSERT_EQ(events[0].kind, CHORUS_EVENT_EMBEDDING);
    ASSERT_EQ(events[0].request_id, result.request_id);
    ASSERT_EQ(std::string(events[0].session), std::string("npc"));
    ASSERT_EQ(events[0].message_id, chorus_message_id{-1});
}

TEST(ChorusC, Render_result_returns_rejection_as_initialized_value) {
    RuntimePtr runtime = loaded_runtime();
    ASSERT_EQ(chorus_history_import(runtime.get(), "npc", nullptr, 0), CHORUS_OK);
    RequestPtr request = request_with_prompt("preview");
    ASSERT_EQ(chorus_request_set_session(request.get(), "npc"), CHORUS_OK);
    chorus_submit_result result{};
    ASSERT_EQ(chorus_render_prompt(runtime.get(), request.get(), &result), CHORUS_OK);
    ASSERT_EQ(result.error, CHORUS_ERR_UNSUPPORTED_FEATURE);
    ASSERT_TRUE(result.message != nullptr);
    ASSERT_EQ(result.request_id, -1);
    ASSERT_EQ(result.request_message_id, -1);
    ASSERT_EQ(result.response_message_id, -1);
}

TEST(ChorusC, Count_rejection_initializes_outputs_and_echo_does_not_claim_a_tokenizer) {
    chorus_submit_result result{99, 99, 99, CHORUS_OK, "stale"};
    ASSERT_EQ(chorus_count_message_tokens(nullptr, "x", &result), CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(result.request_id, -1);
    ASSERT_EQ(result.message, nullptr);
    auto runtime = loaded_runtime();
    chorus_capabilities caps{};
    ASSERT_TRUE(chorus_get_capabilities(runtime.get(), &caps));
    ASSERT_FALSE(caps.message_token_counting);
    ASSERT_EQ(chorus_count_message_tokens(runtime.get(), "x", &result), CHORUS_OK);
    ASSERT_EQ(result.error, CHORUS_ERR_UNSUPPORTED_FEATURE);
    ASSERT_EQ(result.request_id, -1);
}

std::string defaults_json(chorus_runtime* runtime) {
    char* output = nullptr;
    EXPECT_EQ(chorus_generation_defaults_export_json(runtime, &output), CHORUS_OK);
    std::string result = output ? output : "";
    chorus_string_free(output);
    return result;
}

std::string read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_bytes(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::filesystem::path disposable_dir() {
    static std::atomic<unsigned> next{0};
    auto path = std::filesystem::path("_project/verification/adr-compliance-2026-09-25/adr-007") /
        ("c-files-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         "-" + std::to_string(next++));
    std::filesystem::create_directories(path);
    return path;
}

EventSnapshot generate_echo(chorus_runtime* runtime, chorus_request* request) {
    chorus_submit_result result{};
    EXPECT_EQ(chorus_generate(runtime, request, &result), CHORUS_OK);
    EXPECT_EQ(result.error, CHORUS_OK);
    auto events = wait_events(runtime);
    return events.empty() ? EventSnapshot{-1, CHORUS_EVENT_ERROR, CHORUS_ERR_UNKNOWN, ""} : events.front();
}

TEST(ChorusC, Defaults_content_file_and_two_runtimes_preserve_presence_and_priority) {
    auto first = loaded_runtime();
    auto second = loaded_runtime();
    auto request = request_with_prompt("hello");
    const auto directory = disposable_dir();
    const auto path = directory / "selected.json";
    const std::string content = R"({"version":1,"generation":{"max_tokens":0,"stop":[],"show_thinking":false,"provider_options":{"llama":{"logit_bias":{}}}}})";
    write_bytes(path, content);
    ASSERT_EQ(chorus_generation_defaults_load_file(first.get(), path.c_str()), CHORUS_OK);
    ASSERT_EQ(chorus_generation_defaults_apply_json(second.get(), content.data(), content.size()), CHORUS_OK);
    EXPECT_EQ(defaults_json(first.get()), defaults_json(second.get()));
    EXPECT_NE(defaults_json(first.get()).find("\"logit_bias\":{}"), std::string::npos);
    EXPECT_EQ(generate_echo(first.get(), request.get()).text, "");
    EXPECT_EQ(generate_echo(second.get(), request.get()).text, "");
    ASSERT_EQ(chorus_request_set_max_tokens(request.get(), 2), CHORUS_OK);
    EXPECT_EQ(generate_echo(first.get(), request.get()).text, "hello");
    ASSERT_EQ(chorus_request_clear_max_tokens(request.get()), CHORUS_OK);
    EXPECT_EQ(generate_echo(first.get(), request.get()).text, "");
    const std::string empty = R"({"version":1,"generation":{}})";
    ASSERT_EQ(chorus_generation_defaults_apply_json(second.get(), empty.data(), empty.size()), CHORUS_OK);
    EXPECT_EQ(generate_echo(second.get(), request.get()).text, "hello");
    EXPECT_EQ(generate_echo(first.get(), request.get()).text, "");
    EXPECT_EQ(read_bytes(path), content);
}

TEST(ChorusC, Defaults_validation_failure_does_not_replace_runtime_or_destination) {
    auto runtime = loaded_runtime();
    auto request = request_with_prompt("hello");
    const std::string selected = R"({"version":1,"generation":{"max_tokens":0}})";
    ASSERT_EQ(chorus_generation_defaults_apply_json(runtime.get(), selected.data(), selected.size()), CHORUS_OK);
    const std::string escaped_nul = R"({"version":1,"generation":{"max_tokens":0,"provider_options":{"foreign":{"a\u0000b":"x\u0000y"}}}})";
    ASSERT_EQ(chorus_generation_defaults_apply_json(runtime.get(), escaped_nul.data(), escaped_nul.size()), CHORUS_OK);
    EXPECT_NE(defaults_json(runtime.get()).find("\\u0000"), std::string::npos);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).text, "");
    ASSERT_EQ(chorus_generation_defaults_apply_json(runtime.get(), selected.data(), selected.size()), CHORUS_OK);
    const auto directory = disposable_dir();
    const auto invalid = directory / "invalid.json";
    const std::string bad = R"({"version":1,"generation":{"max_tokens":1,"max_tokens":2}})";
    write_bytes(invalid, bad);
    EXPECT_EQ(chorus_generation_defaults_load_file(runtime.get(), invalid.c_str()), CHORUS_ERR_INVALID_REQUEST);
    EXPECT_FALSE(std::string(chorus_last_error_message(runtime.get())).empty());
    EXPECT_EQ(chorus_generation_defaults_save_file(runtime.get(), invalid.c_str()), CHORUS_ERR_INVALID_REQUEST);
    EXPECT_EQ(read_bytes(invalid), bad);
#if !defined(_WIN32)
    const auto unreadable = directory / "unreadable.json";
    write_bytes(unreadable, selected);
    std::filesystem::permissions(unreadable, std::filesystem::perms::none);
    if (!std::ifstream(unreadable)) {
        EXPECT_EQ(chorus_generation_defaults_load_file(runtime.get(), unreadable.c_str()), CHORUS_ERR_UNKNOWN);
        EXPECT_EQ(chorus_generation_defaults_save_file(runtime.get(), unreadable.c_str()), CHORUS_ERR_UNKNOWN);
    }
    std::filesystem::permissions(unreadable, std::filesystem::perms::owner_all);
    EXPECT_EQ(read_bytes(unreadable), selected);
#endif
    EXPECT_EQ(chorus_generation_defaults_apply_json(runtime.get(), bad.data(), bad.size()), CHORUS_ERR_INVALID_REQUEST);
    for (const auto& suffix : {std::string(1, '\0'), std::string("\0garbage", 8)}) {
        const auto invalid_bytes = selected + suffix;
        EXPECT_EQ(chorus_generation_defaults_apply_json(runtime.get(), invalid_bytes.data(), invalid_bytes.size()), CHORUS_ERR_INVALID_REQUEST);
        EXPECT_EQ(defaults_json(runtime.get()), selected);
    }
    EXPECT_EQ(chorus_generation_defaults_apply_json(runtime.get(), "", 0), CHORUS_ERR_INVALID_REQUEST);
    EXPECT_EQ(chorus_generation_defaults_load_file(runtime.get(), directory.c_str()), CHORUS_ERR_UNKNOWN);
    EXPECT_EQ(chorus_generation_defaults_save_file(runtime.get(), directory.c_str()), CHORUS_ERR_UNKNOWN);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).text, "");
    EXPECT_EQ(defaults_json(runtime.get()), selected);
    EXPECT_STREQ(chorus_last_error_message(runtime.get()), "");
    EXPECT_EQ(read_bytes(invalid), bad);
    EXPECT_EQ(chorus_generation_defaults_load_file(runtime.get(), nullptr), CHORUS_ERR_INVALID_REQUEST);
    EXPECT_EQ(chorus_generation_defaults_save_file(runtime.get(), ""), CHORUS_ERR_INVALID_REQUEST);
    char* output = reinterpret_cast<char*>(1);
    EXPECT_EQ(chorus_generation_defaults_export_json(runtime.get(), nullptr), CHORUS_ERR_INVALID_REQUEST);
    EXPECT_EQ(chorus_generation_defaults_export_json(nullptr, &output), CHORUS_ERR_INVALID_REQUEST);
    EXPECT_EQ(output, nullptr);
}

TEST(ChorusC, Defaults_missing_file_creation_and_explicit_save_do_not_write_on_apply) {
    RuntimePtr runtime(chorus_runtime_new());
    const auto directory = disposable_dir();
    const auto selected = directory / "nested" / "missing.json";
    ASSERT_EQ(chorus_generation_defaults_load_file(runtime.get(), selected.c_str()), CHORUS_OK);
    EXPECT_EQ(read_bytes(selected), R"({"version":1,"generation":{}})");
    EXPECT_EQ(defaults_json(runtime.get()), read_bytes(selected));
    const std::string content = R"({"version":1,"generation":{"max_tokens":0,"seed":18446744073709551615,"constraint":{"kind":"unconstrained"},"chat_template":""}})";
    ASSERT_EQ(chorus_generation_defaults_apply_json(runtime.get(), content.data(), content.size()), CHORUS_OK);
    EXPECT_EQ(read_bytes(selected), R"({"version":1,"generation":{}})");
    ASSERT_EQ(chorus_generation_defaults_save_file(runtime.get(), selected.c_str()), CHORUS_OK);
    EXPECT_EQ(read_bytes(selected), defaults_json(runtime.get()));
    const auto other = directory / "new.json";
    ASSERT_EQ(chorus_generation_defaults_save_file(runtime.get(), other.c_str()), CHORUS_OK);
    EXPECT_EQ(read_bytes(other), read_bytes(selected));
    EXPECT_EQ(chorus_generation_defaults_load_file(runtime.get(), other.c_str()), CHORUS_OK);
    const auto cannot_create = directory / "blocker" / "child.json";
    write_bytes(directory / "blocker", "unchanged");
    EXPECT_EQ(chorus_generation_defaults_load_file(runtime.get(), cannot_create.c_str()), CHORUS_ERR_UNKNOWN);
    EXPECT_EQ(chorus_generation_defaults_save_file(runtime.get(), cannot_create.c_str()), CHORUS_ERR_UNKNOWN);
    EXPECT_EQ(read_bytes(directory / "blocker"), "unchanged");
    EXPECT_EQ(defaults_json(runtime.get()), read_bytes(other));
}

TEST(ChorusC, Defaults_concurrent_missing_file_selection_adopts_one_complete_document) {
    const auto selected = disposable_dir() / "shared.json";
    constexpr int callers = 8;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    std::vector<chorus_error> errors(callers, CHORUS_ERR_UNKNOWN);
    std::vector<std::string> exports(callers);
    std::vector<std::string> diagnostics(callers);
    for (int i = 0; i < callers; ++i) {
        threads.emplace_back([&, i] {
            RuntimePtr runtime(chorus_runtime_new());
            ready.fetch_add(1);
            while (!start.load()) std::this_thread::yield();
            errors[i] = chorus_generation_defaults_load_file(runtime.get(), selected.c_str());
            if (errors[i] != CHORUS_OK)
                diagnostics[i] = chorus_last_error_message(runtime.get());
            if (errors[i] == CHORUS_OK) {
                char* json = nullptr;
                errors[i] = chorus_generation_defaults_export_json(runtime.get(), &json);
                if (json) exports[i] = json;
                chorus_string_free(json);
            }
        });
    }
    while (ready.load() != callers) std::this_thread::yield();
    start.store(true);
    for (auto& thread : threads) thread.join();
    for (int i = 0; i < callers; ++i) {
        EXPECT_EQ(errors[i], CHORUS_OK) << i << ": " << diagnostics[i];
        EXPECT_EQ(exports[i], R"({"version":1,"generation":{}})") << i;
    }
    EXPECT_EQ(read_bytes(selected), R"({"version":1,"generation":{}})");
}

TEST(ChorusC, Defaults_freeze_admitted_work_and_provider_validation_stays_async) {
    auto runtime = loaded_runtime();
    auto request = request_with_prompt("hello");
    const std::string zero = R"({"version":1,"generation":{"max_tokens":0}})";
    const std::string empty = R"({"version":1,"generation":{}})";
    ASSERT_EQ(chorus_generation_defaults_apply_json(runtime.get(), zero.data(), zero.size()), CHORUS_OK);
    chorus_submit_result single{}, batch[2]{};
    const chorus_request* requests[] = {request.get(), request.get()};
    ASSERT_EQ(chorus_generate(runtime.get(), request.get(), &single), CHORUS_OK);
    ASSERT_EQ(chorus_generate_batch(runtime.get(), requests, 2, batch), CHORUS_OK);
    ASSERT_EQ(chorus_generation_defaults_apply_json(runtime.get(), empty.data(), empty.size()), CHORUS_OK);
    const auto frozen = wait_events(runtime.get(), 3);
    ASSERT_EQ(frozen.size(), 3U);
    for (const auto& event : frozen) {
        EXPECT_EQ(event.error, CHORUS_OK);
        EXPECT_EQ(event.text, "");
    }
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).text, "hello");
    const std::string grammar = R"({"version":1,"generation":{"constraint":{"kind":"gbnf","source":"root ::= 'x'"}}})";
    ASSERT_EQ(chorus_generation_defaults_apply_json(runtime.get(), grammar.data(), grammar.size()), CHORUS_OK);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).error, CHORUS_ERR_UNSUPPORTED_FEATURE);
    ASSERT_EQ(chorus_request_set_unconstrained(request.get()), CHORUS_OK);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).text, "hello");
    ASSERT_EQ(chorus_request_clear_constraint(request.get()), CHORUS_OK);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).error, CHORUS_ERR_UNSUPPORTED_FEATURE);
    const std::string chat = R"({"version":1,"generation":{"chat_template":""}})";
    ASSERT_EQ(chorus_generation_defaults_apply_json(runtime.get(), chat.data(), chat.size()), CHORUS_OK);
    ASSERT_EQ(chorus_request_set_session(request.get(), "npc"), CHORUS_OK);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).error, CHORUS_ERR_INVALID_REQUEST);
    ASSERT_EQ(chorus_request_set_chat_template(request.get(), ""), CHORUS_OK);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).error, CHORUS_ERR_INVALID_REQUEST);
}

TEST(ChorusC, Defaults_echo_namespace_is_provider_validated_and_request_overrides_one_option) {
    auto runtime = loaded_runtime();
    auto request = request_with_prompt("hello");
    const std::string foreign = R"({"version":1,"generation":{"provider_options":{"llama":{"repeat_penalty":1.0}}}})";
    ASSERT_EQ(chorus_generation_defaults_apply_json(runtime.get(), foreign.data(), foreign.size()), CHORUS_OK);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).text, "hello");
    const std::string echo = R"({"version":1,"generation":{"provider_options":{"echo":{"flag":true}}}})";
    ASSERT_EQ(chorus_generation_defaults_apply_json(runtime.get(), echo.data(), echo.size()), CHORUS_OK);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).error, CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_set_provider_option_bool(request.get(), "echo", "flag", false), CHORUS_OK);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).error, CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(chorus_request_clear_provider_option(request.get(), "echo", "flag"), CHORUS_OK);
    EXPECT_EQ(generate_echo(runtime.get(), request.get()).error, CHORUS_ERR_UNSUPPORTED_OPTION);
}

class ChorusCModelTest : public ChorusModelTest {};
TEST_F(ChorusCModelTest, Async_count_copies_input_and_event_storage_survives_submission_and_log_polling) {
    RuntimePtr runtime(chorus_runtime_new());
    chorus_load_result load{};
    ASSERT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_LLAMA, "tests/models/gemma-3-270m-it-F16.gguf", nullptr, CHORUS_LOG_OFF, &load), CHORUS_OK);
    ASSERT_EQ(load.error, CHORUS_OK);
    const auto load_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    bool loaded = false;
    while (!loaded && std::chrono::steady_clock::now() < load_deadline) {
        size_t count = 0;
        const auto* events = chorus_poll(runtime.get(), &count);
        for (size_t i = 0; i < count; ++i) {
            ASSERT_NE(events[i].kind, CHORUS_EVENT_MODEL_LOAD_FAILED);
            loaded |= events[i].kind == CHORUS_EVENT_MODEL_LOADED && events[i].load_id == load.load_id;
        }
        if (!loaded) std::this_thread::yield();
    }
    ASSERT_TRUE(loaded);
    chorus_capabilities caps{};
    ASSERT_TRUE(chorus_get_capabilities(runtime.get(), &caps));
    ASSERT_TRUE(caps.message_token_counting);
    chorus_submit_result submitted{};
    {
        std::string content = "日本語 😀 <bos> literal content";
        ASSERT_EQ(chorus_count_message_tokens(runtime.get(), content.c_str(), &submitted), CHORUS_OK);
        content.assign("changed");
    }
    ASSERT_EQ(submitted.error, CHORUS_OK);
    const chorus_event* events = nullptr;
    size_t count = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!count && std::chrono::steady_clock::now() < deadline) {
        events = chorus_poll(runtime.get(), &count);
        std::this_thread::yield();
    }
    ASSERT_EQ(count, 1U);
    ASSERT_EQ(events[0].kind, CHORUS_EVENT_MESSAGE_TOKEN_COUNT);
    ASSERT_EQ(events[0].request_id, submitted.request_id);
    const auto original_count = events[0].token_count;
    ASSERT_GT(original_count, 0);
    ASSERT_EQ(events[0].session, nullptr);
    ASSERT_EQ(events[0].message_id, -1);
    chorus_submit_result repeated{};
    ASSERT_EQ(chorus_count_message_tokens(runtime.get(), "日本語 😀 <bos> literal content", &repeated), CHORUS_OK);
    size_t logs = 0;
    chorus_poll_logs(runtime.get(), &logs);
    ASSERT_EQ(events[0].token_count, original_count);
    ASSERT_EQ(events[0].request_id, submitted.request_id);
    count = 0;
    while (!count && std::chrono::steady_clock::now() < deadline) {
        events = chorus_poll(runtime.get(), &count);
        std::this_thread::yield();
    }
    ASSERT_EQ(count, 1U);
    ASSERT_EQ(events[0].request_id, repeated.request_id);
    ASSERT_EQ(events[0].token_count, original_count);
    ASSERT_EQ(chorus_count_message_tokens(runtime.get(), "", &repeated), CHORUS_OK);
    count = 0;
    while (!count && std::chrono::steady_clock::now() < deadline) {
        events = chorus_poll(runtime.get(), &count);
        std::this_thread::yield();
    }
    ASSERT_EQ(count, 1U);
    ASSERT_EQ(events[0].kind, CHORUS_EVENT_MESSAGE_TOKEN_COUNT);
    ASSERT_EQ(events[0].token_count, 0);
}

} // namespace

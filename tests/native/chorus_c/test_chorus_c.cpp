#include "chorus_c/chorus_c.h"
#include "gtest_utils.hpp"

#include <cstring>
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
    if (runtime)
        EXPECT_EQ(chorus_load(runtime.get(), CHORUS_PROVIDER_ECHO, nullptr, nullptr, CHORUS_LOG_OFF), CHORUS_OK);
    return runtime;
}

RequestPtr request_with_prompt(const char* prompt) {
    RequestPtr request(chorus_request_new());
    EXPECT_TRUE(request != nullptr);
    if (request)
        EXPECT_EQ(chorus_request_set_prompt(request.get(), prompt), CHORUS_OK);
    return request;
}

TEST(ChorusC, Header_is_pure_c_and_uses_abi_four) {
    ASSERT_EQ(chorus_c_header_smoke(), 0);
    ASSERT_EQ(chorus_abi_version(), uint32_t{4});
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
    ASSERT_EQ(results[1].error, CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_EQ(results[1].request_id, chorus_request_id{-1});
    ASSERT_TRUE(results[1].message != nullptr);
    ASSERT_EQ(results[2].error, CHORUS_OK);
    ASSERT_TRUE(results[2].request_id >= 0);
}

TEST(ChorusC, Batch_result_strings_survive_internal_singular_submissions) {
    RuntimePtr runtime = loaded_runtime();
    RequestPtr rejected = request_with_prompt("rejected");
    ASSERT_EQ(chorus_request_set_provider_option_bool(rejected.get(), "echo", "unsupported", true), CHORUS_OK);
    RequestPtr accepted = request_with_prompt("accepted");
    const chorus_request* requests[] = {rejected.get(), accepted.get()};
    chorus_submit_result results[2] = {};

    ASSERT_EQ(chorus_generate_batch(runtime.get(), requests, 2, results), CHORUS_OK);
    ASSERT_EQ(results[0].error, CHORUS_ERR_UNSUPPORTED_OPTION);
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
    ASSERT_EQ(chorus_request_set_provider_option_bool(rejected.get(), "echo", "unsupported", true), CHORUS_OK);
    chorus_submit_result submission{};
    ASSERT_EQ(chorus_generate(runtime.get(), rejected.get(), &submission), CHORUS_OK);
    ASSERT_EQ(submission.error, CHORUS_ERR_UNSUPPORTED_OPTION);
    ASSERT_NE(submission.message, nullptr);
    ASSERT_FALSE(std::string(submission.message).empty());

    RequestPtr rendered_request = request_with_prompt("rendered");
    chorus_render_result rendered{};
    ASSERT_EQ(chorus_render_prompt(runtime.get(), rendered_request.get(), &rendered), CHORUS_OK);
    ASSERT_EQ(rendered.error, CHORUS_OK);
    ASSERT_EQ(std::string(rendered.text), "rendered");

    size_t count = 0;
    ASSERT_NE(chorus_poll(runtime.get(), &count), nullptr);
    ASSERT_EQ(count, size_t{0});
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
    chorus_render_result result{};
    ASSERT_EQ(chorus_render_prompt(runtime.get(), request.get(), &result), CHORUS_OK);
    ASSERT_EQ(result.error, CHORUS_ERR_INVALID_REQUEST);
    ASSERT_TRUE(result.message != nullptr);
    ASSERT_EQ(result.text, nullptr);
    ASSERT_EQ(result.omitted_message_ids, nullptr);
    ASSERT_EQ(result.omitted_message_id_count, size_t{0});
}

} // namespace

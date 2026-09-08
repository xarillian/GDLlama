#include "chorus/runtime/runtime.hpp"
#include "support/gtest_utils.hpp"
#include "support/sync_mock_engine.hpp"

#include <limits>

namespace {
using namespace Chorus;

ChatMessage message(MessageRole role, const char* content) {
    return {role, MessageContent::text(content)};
}
ConversationMessage record(MessageId id, MessageRole role, const char* content) {
    return {id, message(role, content)};
}
std::string text(const ConversationMessage& item) {
    return *joined_text(item.message.content);
}

TEST(ChatHistory, import_is_atomic_and_preserves_ids) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ASSERT_FALSE(runtime.import_conversation_history("npc", {record(41, MessageRole::System, "persona")}).has_value());
    ASSERT_EQ(runtime.import_conversation_history("npc", {record(9, MessageRole::User, "bad"), record(9, MessageRole::Assistant, "duplicate")}), ChorusError::InvalidRequest);
    const auto history = runtime.export_conversation_history("npc");
    ASSERT_EQ(history.size(), 1U);
    ASSERT_EQ(history[0].id, 41);
}

TEST(ChatHistory, accepted_turn_reports_and_stores_reserved_ids) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->tokens = {"answer"};
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    GenerationRequest request;
    request.session_id = "npc";
    request.prompt = "hello";
    const auto submitted = runtime.submit(request);
    ASSERT_TRUE(submitted.ok());
    ASSERT_EQ(submitted.request_message_id, 0);
    ASSERT_EQ(submitted.response_message_id, 1);
    const auto events = runtime.poll();
    ASSERT_EQ(events.back().message_id, 1);
    const auto history = runtime.export_conversation_history("npc");
    ASSERT_EQ(history.size(), 2U);
    ASSERT_EQ(history[0].id, 0);
    ASSERT_EQ(history[1].id, 1);
    ASSERT_EQ(text(history[1]), "answer");
}

TEST(ChatHistory, imported_ids_advance_the_mint_and_exhaust_the_final_pair) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    const MessageId maximum = std::numeric_limits<MessageId>::max();
    ASSERT_FALSE(runtime.import_conversation_history("near-limit", {record(maximum - 2, MessageRole::System, "persona")}).has_value());

    GenerationRequest request;
    request.session_id = "near-limit";
    request.prompt = "last turn";
    const auto final_pair = runtime.submit(request);
    ASSERT_TRUE(final_pair.ok());
    ASSERT_EQ(final_pair.request_message_id, maximum - 1);
    ASSERT_EQ(final_pair.response_message_id, maximum);
    runtime.poll();

    request.session_id = "exhausted";
    const auto exhausted = runtime.submit(request);
    ASSERT_FALSE(exhausted.ok());
    ASSERT_EQ(exhausted.error, ChorusError::InvalidRequest);
}

TEST(ChatHistory, failed_first_turn_keeps_empty_history_and_outcome) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->fail_submit_with = ChorusError::Decode;
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());

    GenerationRequest request;
    request.session_id = "absent";
    request.prompt = "new";
    ASSERT_TRUE(runtime.submit(request).ok());
    runtime.poll();
    const auto sessions = runtime.list_conversations();
    ASSERT_EQ(sessions.size(), 1U);
    ASSERT_EQ(sessions[0], "absent");
    ASSERT_TRUE(runtime.export_conversation_history("absent").empty());
    ASSERT_EQ(runtime.last_turn_outcome("absent"), TurnOutcome::Errored);

    ASSERT_FALSE(runtime.import_conversation_history("imported-empty", {}).has_value());
    request.session_id = "imported-empty";
    ASSERT_TRUE(runtime.submit(request).ok());
    runtime.poll();
    ASSERT_TRUE(runtime.export_conversation_history("imported-empty").empty());
}

TEST(ChatHistory, cancelled_first_turn_keeps_empty_history_and_outcome) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->hold_requests = true;
    engine->emit_cancelled_on_cancel = true;
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());

    GenerationRequest request;
    request.session_id = "cancelled";
    request.prompt = "new";
    const auto submitted = runtime.submit(request);
    ASSERT_TRUE(submitted.ok());
    ASSERT_TRUE(runtime.cancel(submitted.request_id));
    const auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1U);
    ASSERT_EQ(events[0].error, ChorusError::Cancelled);
    ASSERT_TRUE(runtime.export_conversation_history("cancelled").empty());
    ASSERT_EQ(runtime.last_turn_outcome("cancelled"), TurnOutcome::Cancelled);
    ASSERT_EQ(runtime.list_conversations().size(), 1U);
}

TEST(ChatHistory, failed_turn_erases_exact_pending_id) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->fail_submit_with = ChorusError::Decode;
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ASSERT_FALSE(runtime.import_conversation_history("npc", {record(77, MessageRole::User, "old")}).has_value());
    GenerationRequest request;
    request.session_id = "npc";
    request.prompt = "new";
    ASSERT_TRUE(runtime.submit(request).ok());
    runtime.poll();
    const auto history = runtime.export_conversation_history("npc");
    ASSERT_EQ(history.size(), 1U);
    ASSERT_EQ(history[0].id, 77);
}

TEST(ChatHistory, regeneration_reuses_assistant_identity_and_restores_on_error) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->fail_submit_with = ChorusError::Decode;
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ASSERT_FALSE(runtime.import_conversation_history("npc", {record(4, MessageRole::User, "old"), record(5, MessageRole::Assistant, "reply")}).has_value());
    GenerationRequest request;
    request.session_id = "npc";
    const auto submitted = runtime.regenerate(request);
    ASSERT_TRUE(submitted.ok());
    ASSERT_FALSE(submitted.request_message_id.has_value());
    ASSERT_EQ(submitted.response_message_id, 5);
    runtime.poll();
    const auto history = runtime.export_conversation_history("npc");
    ASSERT_EQ(history.back().id, 5);
    ASSERT_EQ(text(history.back()), "reply");
}

TEST(ChatHistory, successful_regeneration_replaces_content_without_changing_identity) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->tokens = {"rerolled"};
    runtime.load_engine(std::move(engine), {});
    ASSERT_FALSE(runtime.import_conversation_history("npc", {
        record(4, MessageRole::User, "old"), record(5, MessageRole::Assistant, "reply")
    }).has_value());

    GenerationRequest request;
    request.session_id = "npc";
    const auto submitted = runtime.regenerate(request);
    ASSERT_TRUE(submitted.ok());
    ASSERT_EQ(submitted.response_message_id, 5);
    const auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1U);
    ASSERT_EQ(events[0].message_id, 5);
    const auto history = runtime.export_conversation_history("npc");
    ASSERT_EQ(history.size(), 2U);
    ASSERT_EQ(history[1].id, 5);
    ASSERT_EQ(text(history[1]), "rerolled");
}

TEST(ChatHistory, regeneration_restores_the_exact_reply_after_cancellation) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->hold_requests = true;
    engine->emit_cancelled_on_cancel = true;
    runtime.load_engine(std::move(engine), {});
    ASSERT_FALSE(runtime.import_conversation_history("npc", {
        record(4, MessageRole::User, "old"), record(5, MessageRole::Assistant, "reply")
    }).has_value());

    GenerationRequest request;
    request.session_id = "npc";
    const auto submitted = runtime.regenerate(request);
    ASSERT_TRUE(submitted.ok());
    ASSERT_TRUE(runtime.cancel(submitted.request_id));
    const auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1U);
    ASSERT_EQ(events[0].error, ChorusError::Cancelled);
    const auto history = runtime.export_conversation_history("npc");
    ASSERT_EQ(history.size(), 2U);
    ASSERT_EQ(history[1].id, 5);
    ASSERT_EQ(history[1].message.role, MessageRole::Assistant);
    ASSERT_EQ(text(history[1]), "reply");
}

TEST(ChatHistory, edit_addresses_id_and_keeps_role) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ASSERT_FALSE(runtime.import_conversation_history("npc", {record(42, MessageRole::Assistant, "old")}).has_value());
    ASSERT_FALSE(runtime.edit_message("npc", 42, MessageContent::text("new")).has_value());
    ASSERT_EQ(runtime.edit_message("npc", 43, MessageContent::text("bad")), ChorusError::InvalidRequest);
    const auto history = runtime.export_conversation_history("npc");
    ASSERT_EQ(history[0].message.role, MessageRole::Assistant);
    ASSERT_EQ(text(history[0]), "new");
}
} // namespace

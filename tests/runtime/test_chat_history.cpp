#include "chorus/runtime/runtime.hpp"
#include "sync_mock_engine.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

static Chorus::ChorusConfig make_config() {
    Chorus::ChorusConfig config;
    config.model.model_id = "mock.bin";
    return config;
}

void test_import_export_roundtrip_and_list() {
    Chorus::ChorusRuntime runtime;
    std::vector<Chorus::ChatMessage> history{{"system", "persona"}, {"user", "hi"}};
    ASSERT_TRUE(!runtime.import_conversation_history("npc_1", history).has_value());

    auto exported = runtime.export_conversation_history("npc_1");
    ASSERT_EQ((int)exported.size(), 2);
    ASSERT_EQ(exported[0].role, std::string("system"));
    ASSERT_EQ(exported[1].content, std::string("hi"));

    auto sessions = runtime.list_conversations();
    ASSERT_EQ((int)sessions.size(), 1);
    ASSERT_EQ(sessions[0], std::string("npc_1"));

    ASSERT_TRUE(runtime.export_conversation_history("unknown").empty());
    ASSERT_TRUE(runtime.last_turn_outcome("npc_1") == Chorus::TurnOutcome::None);
}

void test_clear_and_reset() {
    Chorus::ChorusRuntime runtime;
    (void)runtime.import_conversation_history("a", {{"user", "1"}});
    (void)runtime.import_conversation_history("b", {{"user", "2"}});

    ASSERT_TRUE(!runtime.clear_conversation_history("a").has_value());
    ASSERT_TRUE(runtime.export_conversation_history("a").empty());
    ASSERT_TRUE(!runtime.clear_conversation_history("never_existed").has_value()); // idempotent

    ASSERT_TRUE(!runtime.reset_context().has_value());
    ASSERT_TRUE(runtime.list_conversations().empty());
}

void test_mutation_rejected_while_session_busy() {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->hold_requests = true; // request stays live
    ASSERT_TRUE(!runtime.load_engine(std::move(engine), make_config()).has_value());

    Chorus::GenerationRequest request;
    request.prompt = "hi";
    request.session_id = "npc_1";
    auto submitted = runtime.submit(request);
    ASSERT_TRUE(submitted.ok());

    auto import_err = runtime.import_conversation_history("npc_1", {{"user", "x"}});
    ASSERT_TRUE(import_err.has_value() && *import_err == Chorus::ChorusError::SessionBusy);
    auto clear_err = runtime.clear_conversation_history("npc_1");
    ASSERT_TRUE(clear_err.has_value() && *clear_err == Chorus::ChorusError::SessionBusy);
    auto reset_err = runtime.reset_context();
    ASSERT_TRUE(reset_err.has_value() && *reset_err == Chorus::ChorusError::SessionBusy);

    // Other sessions are still mutable while npc_1 is busy.
    ASSERT_TRUE(!runtime.import_conversation_history("npc_2", {{"user", "y"}}).has_value());
}

static Chorus::GenerationRequest chat_turn(const std::string& prompt, const std::string& session, bool stream = false) {
    Chorus::GenerationRequest request;
    request.prompt = prompt;
    request.session_id = session;
    request.stream = stream;
    return request;
}

void test_sessioned_submit_builds_messages_and_appends_on_complete() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    auto* engine = owned.get();
    engine->tokens = {}; // default-tokens empty => mock echoes last user message for chat
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    (void)runtime.import_conversation_history("npc_1", {{"system", "persona"}});

    auto submitted = runtime.submit(chat_turn("hello", "npc_1"));
    ASSERT_TRUE(submitted.ok());

    auto events = runtime.poll();
    ASSERT_EQ((int)events.size(), 1);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[0].text, std::string("hello")); // mock echoes last user msg

    auto history = runtime.export_conversation_history("npc_1");
    ASSERT_EQ((int)history.size(), 3); // system + user + assistant
    ASSERT_EQ(history[1].role, std::string("user"));
    ASSERT_EQ(history[1].content, std::string("hello"));
    ASSERT_EQ(history[2].role, std::string("assistant"));
    ASSERT_EQ(history[2].content, std::string("hello"));
    ASSERT_TRUE(runtime.last_turn_outcome("npc_1") == Chorus::TurnOutcome::Completed);
}

void test_stateless_submit_untouched_by_chat() {
    Chorus::ChorusRuntime runtime;
    ASSERT_TRUE(!runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config()).has_value());
    Chorus::GenerationRequest request;
    request.prompt = "hi";
    ASSERT_TRUE(runtime.submit(request).ok());
    auto events = runtime.poll();
    ASSERT_EQ((int)events.size(), 1);
    ASSERT_TRUE(runtime.list_conversations().empty()); // no session, no history
}

void test_cancelled_turn_rolls_back_user_message() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    auto* engine = owned.get();
    engine->hold_requests = true;
    engine->emit_cancelled_on_cancel = true;
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    (void)runtime.import_conversation_history("npc_1", {{"system", "persona"}});

    auto submitted = runtime.submit(chat_turn("doomed", "npc_1"));
    ASSERT_TRUE(submitted.ok());
    ASSERT_TRUE(runtime.cancel(submitted.request_id));
    auto events = runtime.poll();
    ASSERT_EQ((int)events.size(), 1);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Error);

    auto history = runtime.export_conversation_history("npc_1");
    ASSERT_EQ((int)history.size(), 1); // user message rolled back
    ASSERT_TRUE(runtime.last_turn_outcome("npc_1") == Chorus::TurnOutcome::Cancelled);
}

void test_errored_turn_rolls_back_and_marks_errored() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->emit_error_instead_of_stop = true;
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());

    auto submitted = runtime.submit(chat_turn("q", "npc_1"));
    ASSERT_TRUE(submitted.ok());
    (void)runtime.poll();
    ASSERT_TRUE(runtime.export_conversation_history("npc_1").empty());
    ASSERT_TRUE(runtime.last_turn_outcome("npc_1") == Chorus::TurnOutcome::Errored);
}

void test_reasoning_tokens_route_to_reasoning_channel() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->scripted_channel_tokens = {
        {Chorus::TokenChannel::Reasoning, "thinking... "},
        {Chorus::TokenChannel::Content, "four"},
    };
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());

    auto submitted = runtime.submit(chat_turn("2+2?", "npc_1", /*stream=*/true));
    ASSERT_TRUE(submitted.ok());
    auto events = runtime.poll();
    // stream=true: ReasoningToken, Token, Complete
    ASSERT_EQ((int)events.size(), 3);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::ReasoningToken);
    ASSERT_EQ(events[0].text, std::string("thinking... "));
    ASSERT_TRUE(events[1].kind == Chorus::RuntimeEvent::Kind::Token);
    ASSERT_TRUE(events[2].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[2].text, std::string("four"));
    ASSERT_EQ(events[2].reasoning, std::string("thinking... "));

    // History gets content only.
    auto history = runtime.export_conversation_history("npc_1");
    ASSERT_EQ(history.back().content, std::string("four"));
}

void test_nonstreaming_suppresses_reasoning_tokens_but_terminal_carries_reasoning() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->scripted_channel_tokens = {{Chorus::TokenChannel::Reasoning, "hmm"}, {Chorus::TokenChannel::Content, "ok"}};
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    auto submitted = runtime.submit(chat_turn("q", "npc_1", /*stream=*/false));
    auto events = runtime.poll();
    ASSERT_EQ((int)events.size(), 1);
    ASSERT_EQ(events[0].reasoning, std::string("hmm"));
    ASSERT_EQ(events[0].text, std::string("ok"));
}

// Budget math with mock render: probe counts 1 + words per message.
void test_truncation_drops_oldest_and_emits_event() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->tokens = {};
    owned->supports_render = true;
    owned->mock_per_request_context = 522; // budget = 522 - 512 fallback = 10 probe-tokens
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    // system(1+1) + 6 turns of one word (2 each) = 14 > 10; drop until fits.
    (void)runtime.import_conversation_history(
        "npc_1",
        {{"system", "persona"},
         {"user", "a"},
         {"assistant", "b"},
         {"user", "c"},
         {"assistant", "d"},
         {"user", "e"},
         {"assistant", "f"}}
    );

    auto submitted = runtime.submit(chat_turn("g", "npc_1"));
    ASSERT_TRUE(submitted.ok());

    auto events = runtime.poll();
    ASSERT_TRUE(events.size() >= 2);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::HistoryTruncated);
    // Deterministic drop count: probe = 1 + words per message. Prospective is
    // 8 messages x 2 = 16 against budget 10; system (2) pinned, newest user
    // (2) protected. Drop 3 reaches 10 but strands assistant "d" at the front
    // of the window; user-boundary alignment extends the drop to 4.
    ASSERT_EQ(events[0].dropped, 4);
    ASSERT_TRUE(events[0].session_id.has_value() && *events[0].session_id == std::string("npc_1"));
    // Durable history is NOT truncated: import(7) + user + assistant = 9.
    ASSERT_EQ((int)runtime.export_conversation_history("npc_1").size(), 9);
}

void test_oversized_single_turn_hard_fails_without_mutation() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->supports_render = true;
    owned->mock_per_request_context = 513; // budget = 1 probe-token
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());

    auto submitted = runtime.submit(chat_turn("many words in one message", "npc_1"));
    ASSERT_TRUE(!submitted.ok());
    ASSERT_TRUE(submitted.error == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(runtime.export_conversation_history("npc_1").empty()); // no trace
}

void test_fitting_skipped_when_engine_cannot_render() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->tokens = {};
    owned->mock_per_request_context = 513; // tiny, but no render => no fitting
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    auto submitted = runtime.submit(chat_turn("many words in one message", "npc_1"));
    ASSERT_TRUE(submitted.ok()); // passes through unfitted
    auto events = runtime.poll();
    for (const auto& event : events)
        ASSERT_TRUE(event.kind != Chorus::RuntimeEvent::Kind::HistoryTruncated);
}

void test_render_prompt_passthrough_and_injections() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->supports_render = true;
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    (void)runtime.import_conversation_history("npc_1", {{"system", "persona"}, {"user", "hi"}});

    auto rendered = runtime.render_prompt("npc_1");
    ASSERT_TRUE(rendered.has_value());
    ASSERT_TRUE(rendered->find("persona") != std::string::npos);

    auto with_inject = runtime.render_prompt("npc_1", "", {{{"system", "it rains"}, 0}});
    ASSERT_TRUE(with_inject.has_value());
    ASSERT_TRUE(with_inject->find("it rains") != std::string::npos);

    ASSERT_TRUE(!runtime.render_prompt("unknown_session").has_value());
}

void test_injected_messages_reach_engine_but_not_history() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    auto* engine = owned.get();
    engine->tokens = {};
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());

    Chorus::GenerationRequest request = chat_turn("hello", "npc_1");
    request.inject = {{{"system", "the tavern is on fire"}, 0}};
    ASSERT_TRUE(runtime.submit(request).ok());
    (void)runtime.poll();

    // Injection reached the engine but durable history must NOT contain it.
    for (const auto& message : runtime.export_conversation_history("npc_1"))
        ASSERT_TRUE(message.content.find("tavern") == std::string::npos);
    ASSERT_TRUE(!engine->last_messages.empty());
    bool saw_injection = false;
    for (const auto& message : engine->last_messages)
        if (message.content.find("tavern") != std::string::npos)
            saw_injection = true;
    ASSERT_TRUE(saw_injection);
}

void test_render_reservation_matches_generation() {
    // Same session, two different max_tokens: the fitted window must differ,
    // and render_prompt must track the config it is given (Codex 1.4).
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->supports_render = true;
    owned->mock_per_request_context = 526; // budget: 526 - max_tokens
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    (void)runtime.import_conversation_history(
        "npc_1",
        {{"system", "persona"}, {"user", "a"}, {"assistant", "b"}, {"user", "c"}, {"assistant", "d"}, {"user", "e"}}
    );
    // Probe: 6 messages x 2 = 12. max_tokens=514 => budget 12: nothing dropped.
    Chorus::GenerationConfig roomy;
    roomy.common.max_tokens = 514;
    auto full = runtime.render_prompt("npc_1", "", {}, roomy);
    ASSERT_TRUE(full.has_value() && full->find("<user>a") != std::string::npos);
    // max_tokens=518 => budget 8: oldest turns fall out of the window.
    Chorus::GenerationConfig tight;
    tight.common.max_tokens = 518;
    auto fitted = runtime.render_prompt("npc_1", "", {}, tight);
    ASSERT_TRUE(fitted.has_value());
    ASSERT_TRUE(fitted->find("<user>a") == std::string::npos); // dropped
    ASSERT_TRUE(fitted->find("<user>e") != std::string::npos); // newest protected
}

void test_inspection_equals_consumption() {
    // render_prompt(config X) must equal, byte for byte, what an X-configured
    // turn hands the engine. A turn fits history + the pending user message,
    // so inspection is performed on a STAGED copy of that same state.
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    auto* engine = owned.get();
    engine->tokens = {};
    engine->supports_render = true;
    engine->mock_per_request_context = 526;
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());

    Chorus::GenerationConfig config;
    config.common.max_tokens = 516; // budget 10; staged list 6x2=12 -> drops oldest turn
    std::vector<Chorus::ChatMessage> base{
        {"system", "persona"}, {"user", "a"}, {"assistant", "b"}, {"user", "c"}, {"assistant", "d"}
    };

    // Inspect: the state submit will fit (history + pending user message).
    auto staged = base;
    staged.push_back({"user", "e"});
    (void)runtime.import_conversation_history("npc_inspect", staged);
    auto inspected = runtime.render_prompt("npc_inspect", "", {}, config);
    ASSERT_TRUE(inspected.has_value());

    // Consume: the real turn on the un-staged history.
    (void)runtime.import_conversation_history("npc_turn", base);
    Chorus::GenerationRequest turn = chat_turn("e", "npc_turn");
    turn.config = config;
    ASSERT_TRUE(runtime.submit(turn).ok());
    (void)runtime.poll();
    auto consumed = engine->render_chat_prompt(engine->last_messages, "", true);
    ASSERT_TRUE(consumed.has_value());
    ASSERT_EQ(*inspected, consumed->text); // byte-for-byte
}

void test_two_sessions_accumulate_independent_histories() {
    // Spec testing bullet: multi-session isolation through real turns.
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->tokens = {};
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());

    ASSERT_TRUE(runtime.submit(chat_turn("alpha", "npc_1")).ok());
    ASSERT_TRUE(runtime.submit(chat_turn("beta", "npc_2")).ok());
    (void)runtime.poll();

    auto first = runtime.export_conversation_history("npc_1");
    auto second = runtime.export_conversation_history("npc_2");
    ASSERT_EQ((int)first.size(), 2);
    ASSERT_EQ((int)second.size(), 2);
    ASSERT_EQ(first[0].content, std::string("alpha"));
    ASSERT_EQ(first[1].content, std::string("alpha")); // mock echo reply
    ASSERT_EQ(second[0].content, std::string("beta"));
    ASSERT_EQ(second[1].content, std::string("beta"));
}

void test_stateless_chat_controls_rejected() {
    Chorus::ChorusRuntime runtime;
    ASSERT_TRUE(!runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config()).has_value());
    Chorus::GenerationRequest request;
    request.prompt = "hi"; // no session
    request.inject = {{{"system", "x"}, 0}};
    ASSERT_TRUE(runtime.submit(request).error == Chorus::ChorusError::InvalidRequest);
    request.inject.clear();
    request.chat_template = "{{ x }}";
    ASSERT_TRUE(runtime.submit(request).error == Chorus::ChorusError::InvalidRequest);
    request.chat_template.clear();
    request.config.common.thinking = false;
    ASSERT_TRUE(runtime.submit(request).error == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(runtime.list_conversations().empty());
}

void test_rejected_chat_submit_leaves_no_session_trace() {
    // Codex 1.3: a rejected sessioned submit must not create an empty lane.
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->reject_with = Chorus::RequestRejection{Chorus::ChorusError::UnsupportedOption, "nope"};
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    ASSERT_TRUE(!runtime.submit(chat_turn("hello", "npc_ghost")).ok());
    ASSERT_TRUE(runtime.list_conversations().empty());
    ASSERT_TRUE(runtime.export_conversation_history("npc_ghost").empty());
}

void test_regenerate_replaces_last_assistant_message() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->tokens = {};
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    (void)runtime.import_conversation_history(
        "npc_1", {{"system", "persona"}, {"user", "hello"}, {"assistant", "old reply"}}
    );

    Chorus::GenerationRequest request;
    request.session_id = "npc_1";
    auto submitted = runtime.regenerate(request);
    ASSERT_TRUE(submitted.ok());
    (void)runtime.poll();

    auto history = runtime.export_conversation_history("npc_1");
    ASSERT_EQ((int)history.size(), 3);
    ASSERT_EQ(history.back().role, std::string("assistant"));
    ASSERT_EQ(history.back().content, std::string("hello")); // mock echoes last user msg
    ASSERT_TRUE(runtime.last_turn_outcome("npc_1") == Chorus::TurnOutcome::Completed);
}

void test_regenerate_restores_old_reply_on_cancel() {
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    auto* engine = owned.get();
    engine->hold_requests = true;
    engine->emit_cancelled_on_cancel = true;
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    (void)runtime.import_conversation_history(
        "npc_1", {{"system", "persona"}, {"user", "hello"}, {"assistant", "precious reply"}}
    );

    Chorus::GenerationRequest request;
    request.session_id = "npc_1";
    auto submitted = runtime.regenerate(request);
    ASSERT_TRUE(submitted.ok());
    ASSERT_TRUE(runtime.cancel(submitted.request_id));
    (void)runtime.poll();

    auto history = runtime.export_conversation_history("npc_1");
    ASSERT_EQ((int)history.size(), 3);
    ASSERT_EQ(history.back().content, std::string("precious reply")); // restored
    ASSERT_TRUE(runtime.last_turn_outcome("npc_1") == Chorus::TurnOutcome::Cancelled);
}

void test_regenerate_restores_old_reply_on_engine_error() {
    // Spec: Cancelled AND Errored both restore (Codex 3.2 flagged the gap).
    Chorus::ChorusRuntime runtime;
    auto owned = std::make_unique<SyncMockEngine>();
    owned->emit_error_instead_of_stop = true;
    ASSERT_TRUE(!runtime.load_engine(std::move(owned), make_config()).has_value());
    (void)runtime.import_conversation_history(
        "npc_1", {{"system", "persona"}, {"user", "hello"}, {"assistant", "precious reply"}}
    );

    Chorus::GenerationRequest request;
    request.session_id = "npc_1";
    ASSERT_TRUE(runtime.regenerate(request).ok());
    (void)runtime.poll();

    auto history = runtime.export_conversation_history("npc_1");
    ASSERT_EQ((int)history.size(), 3);
    ASSERT_EQ(history.back().content, std::string("precious reply"));
    ASSERT_TRUE(runtime.last_turn_outcome("npc_1") == Chorus::TurnOutcome::Errored);
}

void test_regenerate_guards() {
    Chorus::ChorusRuntime runtime;
    ASSERT_TRUE(!runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config()).has_value());

    Chorus::GenerationRequest no_session;
    ASSERT_TRUE(runtime.regenerate(no_session).error == Chorus::ChorusError::InvalidRequest);

    Chorus::GenerationRequest with_prompt;
    with_prompt.session_id = "npc_1";
    with_prompt.prompt = "not allowed";
    ASSERT_TRUE(runtime.regenerate(with_prompt).error == Chorus::ChorusError::InvalidRequest);

    Chorus::GenerationRequest unknown;
    unknown.session_id = "never_spoke";
    ASSERT_TRUE(runtime.regenerate(unknown).error == Chorus::ChorusError::InvalidRequest);

    (void)runtime.import_conversation_history("npc_1", {{"user", "hi"}}); // trailing user, not assistant
    Chorus::GenerationRequest no_assistant;
    no_assistant.session_id = "npc_1";
    ASSERT_TRUE(runtime.regenerate(no_assistant).error == Chorus::ChorusError::InvalidRequest);
}

int run_chat_history_tests() {
    std::cout << "\n--- Runtime Chat History Tests ---" << std::endl;
    run_test("ChatHistory_import_export_roundtrip_and_list", test_import_export_roundtrip_and_list);
    run_test("ChatHistory_clear_and_reset", test_clear_and_reset);
    run_test("ChatHistory_mutation_rejected_while_busy", test_mutation_rejected_while_session_busy);
    run_test(
        "ChatHistory_sessioned_submit_appends_on_complete",
        test_sessioned_submit_builds_messages_and_appends_on_complete
    );
    run_test("ChatHistory_stateless_submit_untouched", test_stateless_submit_untouched_by_chat);
    run_test("ChatHistory_cancelled_turn_rolls_back", test_cancelled_turn_rolls_back_user_message);
    run_test("ChatHistory_errored_turn_rolls_back", test_errored_turn_rolls_back_and_marks_errored);
    run_test("ChatHistory_reasoning_channel_routing", test_reasoning_tokens_route_to_reasoning_channel);
    run_test(
        "ChatHistory_nonstream_reasoning_on_terminal",
        test_nonstreaming_suppresses_reasoning_tokens_but_terminal_carries_reasoning
    );
    run_test("ChatHistory_truncation_drops_oldest", test_truncation_drops_oldest_and_emits_event);
    run_test("ChatHistory_oversized_single_turn_hard_fails", test_oversized_single_turn_hard_fails_without_mutation);
    run_test("ChatHistory_fitting_skipped_without_render", test_fitting_skipped_when_engine_cannot_render);
    run_test("ChatHistory_render_prompt_passthrough", test_render_prompt_passthrough_and_injections);
    run_test("ChatHistory_injections_reach_engine_not_history", test_injected_messages_reach_engine_but_not_history);
    run_test("ChatHistory_render_reservation_matches_generation", test_render_reservation_matches_generation);
    run_test("ChatHistory_inspection_equals_consumption", test_inspection_equals_consumption);
    run_test("ChatHistory_two_sessions_independent", test_two_sessions_accumulate_independent_histories);
    run_test("ChatHistory_stateless_chat_controls_rejected", test_stateless_chat_controls_rejected);
    run_test("ChatHistory_rejected_submit_no_session_trace", test_rejected_chat_submit_leaves_no_session_trace);
    run_test("ChatHistory_regenerate_replaces_last_assistant", test_regenerate_replaces_last_assistant_message);
    run_test("ChatHistory_regenerate_restores_on_cancel", test_regenerate_restores_old_reply_on_cancel);
    run_test("ChatHistory_regenerate_restores_on_error", test_regenerate_restores_old_reply_on_engine_error);
    run_test("ChatHistory_regenerate_guards", test_regenerate_guards);
    return g_tests_failed > 0 ? 1 : 0;
}

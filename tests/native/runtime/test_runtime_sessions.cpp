#include "chorus/runtime/runtime.hpp"
#include "sync_mock_engine.hpp"
#include "gtest_utils.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <string>

static Chorus::GenerationRequest sessioned(const char* prompt, const char* session) {
    Chorus::GenerationRequest req;
    req.prompt = prompt;
    req.session_id = std::string(session);
    return req;
}

static Chorus::GenerationRequest stateless(const char* prompt, bool stream = false) {
    Chorus::GenerationRequest req;
    req.prompt = prompt;
    req.stream = stream;
    return req;
}

static Chorus::EmbeddingRequest embedding(const char* prompt, const char* session) {
    Chorus::EmbeddingRequest req;
    req.prompt = prompt;
    req.session_id = std::string(session);
    return req;
}

TEST(RuntimeSessions, Runtime_session_ids_monotonic_across_sessioned_and_stateless) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});

    // Inline mock completes at submit; poll() drains the terminal and frees the
    // session, so the same lane could be reused. Ids stay monotonic regardless.
    auto a = runtime.submit(sessioned("a", "s1"));
    runtime.poll();
    auto b = runtime.submit(stateless("b"));
    runtime.poll();
    auto c = runtime.submit(sessioned("c", "s2"));
    runtime.poll();

    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    ASSERT_TRUE(c.ok());
    ASSERT_EQ(a.request_id, 0);
    ASSERT_EQ(b.request_id, 1);
    ASSERT_EQ(c.request_id, 2);
}

TEST(RuntimeSessions, Runtime_session_id_present_on_all_event_kinds) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});

    auto req = sessioned("hi", "s1");
    req.stream = true;
    auto result = runtime.submit(req);
    ASSERT_TRUE(result.ok());
    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 3); // StreamedToken, StreamedToken, Complete
    for (const auto& ev : events) {
        ASSERT_TRUE(ev.session_id.has_value());
        ASSERT_EQ(*ev.session_id, "s1");
    }

    // The synthesized terminal path must also carry the session id.
    auto held_mock = std::make_unique<SyncMockEngine>();
    held_mock->hold_requests = true;
    runtime.load_engine(std::move(held_mock), Chorus::ChorusConfig{});

    auto held = runtime.submit(sessioned("still going", "s1"));
    ASSERT_TRUE(held.ok());
    runtime.stop_all();
    auto cancelled = runtime.poll();
    ASSERT_EQ(cancelled.size(), 1);
    ASSERT_TRUE(cancelled[0].kind == Chorus::RuntimeEvent::Kind::Error);
    ASSERT_TRUE(cancelled[0].error == Chorus::ChorusError::Cancelled);
    ASSERT_TRUE(cancelled[0].session_id.has_value());
    ASSERT_EQ(*cancelled[0].session_id, "s1");
}

TEST(RuntimeSessions, Runtime_session_second_live_request_rejected_SessionBusy) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    auto* mock_ptr = mock.get();
    mock_ptr->hold_requests = true;
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});

    auto first = runtime.submit(sessioned("a", "npc"));
    ASSERT_TRUE(first.ok());

    auto second = runtime.submit(sessioned("b", "npc"));
    ASSERT_TRUE(!second.ok());
    ASSERT_TRUE(second.error == Chorus::ChorusError::SessionBusy);
    ASSERT_TRUE(!second.message.empty());
    ASSERT_EQ(mock_ptr->submitted_ids.size(), 1); // engine never saw the second
}

TEST(RuntimeSessions, Runtime_session_released_only_when_terminal_drained) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    mock->hold_requests = true;
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});

    auto first = runtime.submit(sessioned("a", "npc"));
    ASSERT_TRUE(first.ok());

    // Replacing the engine synthesizes a Cancelled terminal but does NOT drain
    // it: the session stays busy until poll() consumes the terminal.
    runtime.load_engine(std::make_unique<SyncMockEngine>(), Chorus::ChorusConfig{});

    auto before_drain = runtime.submit(sessioned("b", "npc"));
    ASSERT_TRUE(!before_drain.ok());
    ASSERT_TRUE(before_drain.error == Chorus::ChorusError::SessionBusy);

    auto events = runtime.poll(); // drains the Cancelled terminal, frees the lane
    ASSERT_EQ(events.size(), 1);
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::Cancelled);
    ASSERT_TRUE(events[0].session_id.has_value());
    ASSERT_EQ(*events[0].session_id, "npc");

    auto after_drain = runtime.submit(sessioned("c", "npc"));
    ASSERT_TRUE(after_drain.ok());
}

TEST(RuntimeSessions, Runtime_session_reusable_after_complete_and_error) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    auto* mock_ptr = mock.get();
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});

    auto first = runtime.submit(sessioned("a", "npc"));
    ASSERT_TRUE(first.ok());
    auto done = runtime.poll();
    ASSERT_EQ(done.size(), 1);
    ASSERT_TRUE(done[0].kind == Chorus::RuntimeEvent::Kind::Complete);

    auto reuse = runtime.submit(sessioned("b", "npc"));
    ASSERT_TRUE(reuse.ok()); // Complete freed the lane
    runtime.poll();

    // A request that validates but fails inside the engine still frees the lane
    // once its Error terminal drains.
    mock_ptr->fail_submit_with = Chorus::ChorusError::Decode;
    auto failing = runtime.submit(sessioned("c", "npc"));
    ASSERT_TRUE(failing.ok()); // validate passes; failure arrives as an event
    auto err = runtime.poll();
    ASSERT_EQ(err.size(), 1);
    ASSERT_TRUE(err[0].kind == Chorus::RuntimeEvent::Kind::Error);

    mock_ptr->fail_submit_with = Chorus::ChorusError::None;
    auto after_error = runtime.submit(sessioned("d", "npc"));
    ASSERT_TRUE(after_error.ok());
}

TEST(RuntimeSessions, Runtime_session_stateless_requests_admit_concurrently) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    mock->hold_requests = true;
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});

    auto a = runtime.submit(stateless("a"));
    auto b = runtime.submit(stateless("b"));
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok()); // no session lane, no collision
}

TEST(RuntimeSessions, Runtime_session_empty_string_is_invalid) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    auto* mock_ptr = mock.get();
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});

    auto result = runtime.submit(sessioned("a", ""));
    ASSERT_TRUE(!result.ok());
    ASSERT_TRUE(result.error == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(mock_ptr->submitted_ids.empty());
}

TEST(RuntimeSessions, Runtime_session_validate_rejection_creates_no_state) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    auto* mock_ptr = mock.get();
    mock_ptr->reject_with = Chorus::RequestRejection{Chorus::ChorusError::UnsupportedOption, "nope"};
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});

    auto rejected = runtime.submit(sessioned("a", "npc"));
    ASSERT_TRUE(!rejected.ok());
    ASSERT_TRUE(rejected.error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_EQ(rejected.message, "nope");
    ASSERT_TRUE(mock_ptr->submitted_ids.empty());
    ASSERT_EQ(runtime.poll().size(), 0);

    // The rejection left no live state, so the session is still free.
    mock_ptr->reject_with = std::nullopt;
    auto accepted = runtime.submit(sessioned("b", "npc"));
    ASSERT_TRUE(accepted.ok());
}

TEST(RuntimeSessions, Runtime_session_two_sessions_concurrently_live) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    mock->hold_requests = true;
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});

    auto s1 = runtime.submit(sessioned("a", "s1"));
    auto s2 = runtime.submit(sessioned("b", "s2"));
    ASSERT_TRUE(s1.ok());
    ASSERT_TRUE(s2.ok()); // distinct sessions never collide, no SessionBusy
    ASSERT_EQ(s1.request_id, 0);
    ASSERT_EQ(s2.request_id, 1);

    std::map<int64_t, std::string> session_of_id{{s1.request_id, "s1"}, {s2.request_id, "s2"}};

    runtime.stop_all();
    auto events = runtime.poll();
    // Per-id accumulation independence is covered by
    // test_interleaved_requests_accumulate_independently in test_runtime.cpp.
    ASSERT_EQ(events.size(), 2);
    for (const auto& ev : events) {
        ASSERT_TRUE(ev.kind == Chorus::RuntimeEvent::Kind::Error);
        ASSERT_TRUE(ev.error == Chorus::ChorusError::Cancelled);
        ASSERT_TRUE(ev.session_id.has_value());
        ASSERT_EQ(*ev.session_id, session_of_id[ev.request_id]);
    }

    // Both lanes were freed by the drained Cancelled terminals; a fresh engine
    // accepts new submissions for both sessions.
    runtime.load_engine(std::make_unique<SyncMockEngine>(), Chorus::ChorusConfig{});
    auto s1_again = runtime.submit(sessioned("c", "s1"));
    auto s2_again = runtime.submit(sessioned("d", "s2"));
    ASSERT_TRUE(s1_again.ok());
    ASSERT_TRUE(s2_again.ok());
}

TEST(RuntimeSessions, Runtime_sessioned_embedding_owns_lane_until_terminal_delivery) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    mock->hold_requests = true;
    mock->emit_cancelled_on_cancel = true;
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});
    const auto submitted = runtime.submit(embedding("memory", "s1"));
    ASSERT_TRUE(submitted.ok());
    ASSERT_EQ(runtime.active_request_for_session("s1"), submitted.request_id);
    ASSERT_EQ(runtime.submit(sessioned("chat", "s1")).error, Chorus::ChorusError::SessionBusy);
    ASSERT_TRUE(runtime.cancel(submitted.request_id));
    ASSERT_EQ(runtime.submit(embedding("still busy", "s1")).error, Chorus::ChorusError::SessionBusy);
    const auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1U);
    ASSERT_EQ(events[0].session_id, std::optional<Chorus::SessionId>{"s1"});
    ASSERT_TRUE(runtime.submit(embedding("reusable", "s1")).ok());
    ASSERT_TRUE(runtime.export_conversation_history("s1").empty());
}

TEST(RuntimeSessions, Runtime_embedding_batch_cancellation_is_independent) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    mock->hold_requests = true;
    mock->emit_cancelled_on_cancel = true;
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});

    const auto results = runtime.submit_batch(std::vector<Chorus::EmbeddingRequest>{
        embedding("first", "one"), embedding("second", "two")
    });
    ASSERT_EQ(results.size(), 2U);
    ASSERT_TRUE(results[0].ok());
    ASSERT_TRUE(results[1].ok());
    ASSERT_TRUE(runtime.cancel(results[0].request_id));
    ASSERT_TRUE(runtime.is_request_active(results[1].request_id));
    const auto first_terminal = runtime.poll();
    ASSERT_EQ(first_terminal.size(), 1U);
    ASSERT_EQ(first_terminal[0].request_id, results[0].request_id);
    ASSERT_TRUE(runtime.is_request_active(results[1].request_id));
    ASSERT_TRUE(runtime.cancel(results[1].request_id));
    ASSERT_EQ(runtime.poll().size(), 1U);
}

TEST(RuntimeSessions, Runtime_batches_follow_singular_input_order) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    mock->hold_requests = true;
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});
    const auto results = runtime.submit_batch(std::vector<Chorus::GenerationRequest>{sessioned("one", "s"), sessioned("two", "s"), stateless("three")});
    ASSERT_EQ(results.size(), 3U);
    ASSERT_TRUE(results[0].ok());
    ASSERT_EQ(results[1].error, Chorus::ChorusError::SessionBusy);
    ASSERT_TRUE(results[2].ok());
}

TEST(RuntimeSessions, Runtime_cancel_status_preserves_session_until_terminal_drain) {
    Chorus::ChorusRuntime runtime;
    auto mock = std::make_unique<SyncMockEngine>();
    mock->hold_requests = true;
    mock->emit_cancelled_on_cancel = true;
    runtime.load_engine(std::move(mock), Chorus::ChorusConfig{});
    auto result = runtime.submit(sessioned("hello", "npc"));

    ASSERT_TRUE(runtime.active_request_for_session("npc") == result.request_id);
    ASSERT_TRUE(!runtime.active_request_for_session("other").has_value());
    ASSERT_TRUE(runtime.cancel(result.request_id));
    ASSERT_TRUE(runtime.active_request_for_session("npc") == result.request_id);

    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1);
    ASSERT_EQ(events[0].request_id, result.request_id);
    ASSERT_TRUE(events[0].session_id.has_value());
    ASSERT_EQ(*events[0].session_id, "npc");
    ASSERT_TRUE(!runtime.active_request_for_session("npc").has_value());
}

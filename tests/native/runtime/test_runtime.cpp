#include "chorus/runtime/runtime.hpp"
#include "sync_mock_engine.hpp"
#include "gtest_utils.hpp"

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

static Chorus::ChorusConfig make_config(const std::string& model_id = "mock.bin") {
    Chorus::ChorusConfig config;
    config.model.model_id = model_id;
    return config;
}

static Chorus::GenerationRequest make_request(const std::string& prompt, bool stream = false) {
    Chorus::GenerationRequest request;
    request.prompt = prompt;
    request.stream = stream;
    return request;
}

static size_t terminal_count(const std::vector<Chorus::RuntimeEvent>& events, Chorus::RequestId id) {
    size_t count = 0;
    for (const auto& event : events)
        count += event.request_id == id &&
                 (event.kind == Chorus::RuntimeEvent::Kind::Complete || event.kind == Chorus::RuntimeEvent::Kind::Embedding ||
                  event.kind == Chorus::RuntimeEvent::Kind::Error);
    return count;
}

static Chorus::EmbeddingRequest make_embedding_request(const std::string& prompt) {
    Chorus::EmbeddingRequest request;
    request.prompt = prompt;
    return request;
}

TEST(Runtime, Runtime_submit_before_load_returns_EngineNotReady) {
    Chorus::ChorusRuntime runtime;
    ASSERT_TRUE(!runtime.is_loaded());

    auto result = runtime.submit(make_request("hi"));
    ASSERT_TRUE(!result.ok());
    ASSERT_TRUE(result.error == Chorus::ChorusError::EngineNotReady);
    ASSERT_EQ(result.request_id, -1);
}

TEST(Runtime, Runtime_load_engine_success_and_is_loaded) {
    Chorus::ChorusRuntime runtime;
    auto err = runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());
    ASSERT_TRUE(!err.has_value());
    ASSERT_TRUE(runtime.is_loaded());
}

TEST(Runtime, Runtime_load_engine_null_engine_is_InvalidRequest) {
    Chorus::ChorusRuntime runtime;
    auto err = runtime.load_engine(nullptr, make_config());
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(err.value() == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(!runtime.is_loaded());
}

TEST(Runtime, Runtime_failed_load_leaves_runtime_unloaded) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->fail_initialize_with = Chorus::ChorusError::ModelLoad;

    auto err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(err.value() == Chorus::ChorusError::ModelLoad);
    ASSERT_EQ(err->message, "mock initialization failure");
    ASSERT_TRUE(!runtime.is_loaded());
}

TEST(Runtime, Runtime_execution_mode_reaches_the_engine) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    auto* observed = engine.get();
    ASSERT_FALSE(runtime.load_engine(std::move(engine), make_config()).has_value());
    auto request = make_request("isolated");
    request.execution = Chorus::ExecutionMode::Exclusive;
    ASSERT_TRUE(runtime.submit(request).ok());
    ASSERT_EQ(observed->last_execution, Chorus::ExecutionMode::Exclusive);
}

TEST(Runtime, Runtime_poll_on_idle_runtime_returns_empty) {
    Chorus::ChorusRuntime runtime;
    ASSERT_EQ(runtime.poll().size(), 0);
    runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());
    ASSERT_EQ(runtime.poll().size(), 0);
}

TEST(Runtime, Runtime_non_streaming_yields_only_Complete) {
    Chorus::ChorusRuntime runtime;
    runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());

    auto result = runtime.submit(make_request("hi", /*stream=*/false));
    ASSERT_TRUE(result.ok());
    auto events = runtime.poll();

    ASSERT_EQ(events.size(), 1);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[0].request_id, result.request_id);
    ASSERT_EQ(events[0].text, "Hello world");
    ASSERT_EQ(terminal_count(events, result.request_id), size_t{1});
}

TEST(Runtime, Runtime_streaming_yields_tokens_then_Complete) {
    Chorus::ChorusRuntime runtime;
    runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());

    auto result = runtime.submit(make_request("hi", /*stream=*/true));
    ASSERT_TRUE(result.ok());
    auto events = runtime.poll();

    ASSERT_EQ(events.size(), 3);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::StreamedToken);
    ASSERT_EQ(events[0].text, "Hello ");
    ASSERT_TRUE(events[1].kind == Chorus::RuntimeEvent::Kind::StreamedToken);
    ASSERT_EQ(events[1].text, "world");
    ASSERT_TRUE(events[2].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[2].text, "Hello world");
    ASSERT_EQ(terminal_count(events, result.request_id), size_t{1});
}

TEST(Runtime, Runtime_interleaved_requests_accumulate_independently) {
    Chorus::ChorusRuntime runtime;
    runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());

    auto a = runtime.submit(make_request("a"));
    auto b = runtime.submit(make_request("b"));
    auto events = runtime.poll();

    ASSERT_EQ(events.size(), 2);
    ASSERT_EQ(events[0].request_id, a.request_id);
    ASSERT_EQ(events[1].request_id, b.request_id);
    ASSERT_EQ(events[0].text, "Hello world");
    ASSERT_EQ(events[1].text, "Hello world");
}

TEST(Runtime, Runtime_inline_rejection_delivered_on_next_poll) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->fail_submit_with = Chorus::ChorusError::Tokenize;
    runtime.load_engine(std::move(engine), make_config());

    auto result = runtime.submit(make_request("hi"));
    ASSERT_TRUE(result.ok()); // engine accepted the call; failure arrives as an event
    auto events = runtime.poll();

    ASSERT_EQ(events.size(), 1);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Error);
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::Tokenize);
    ASSERT_EQ(events[0].text, "mock failure");
    ASSERT_EQ(runtime.poll().size(), 0); // state erased: nothing further
}

TEST(Runtime, Runtime_error_after_tokens_discards_partial) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->emit_error_instead_of_stop = true; // "Hello ", "world", then Error
    runtime.load_engine(std::move(engine), make_config());

    auto result = runtime.submit(make_request("hi", /*stream=*/false));
    ASSERT_TRUE(result.ok());
    auto events = runtime.poll();

    ASSERT_EQ(events.size(), 1); // no Complete; the accumulated "Hello world" is discarded
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Error);
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::Decode);
    ASSERT_EQ(events[0].text, "failed after partial output");
    ASSERT_EQ(terminal_count(events, result.request_id), size_t{1});
    ASSERT_EQ(runtime.poll().size(), 0);
}

enum class BrokenEventDefect { DuplicateTerminal, TokenAfterTerminal, UnknownRequestId };

struct BrokenEventCase {
    const char* name;
    BrokenEventDefect defect;
};

class RuntimeBrokenEventDefense : public ::testing::TestWithParam<BrokenEventCase> {};

TEST_P(RuntimeBrokenEventDefense, Surfaces_only_the_accepted_request_stream_and_terminal) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    switch (GetParam().defect) {
        case BrokenEventDefect::DuplicateTerminal:
            engine->emit_duplicate_stop = true;
            break;
        case BrokenEventDefect::TokenAfterTerminal:
            engine->emit_token_after_stop = true;
            break;
        case BrokenEventDefect::UnknownRequestId:
            engine->rogue_extra_id = 999999;
            break;
    }
    const auto load_error = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_error.has_value());

    const auto result = runtime.submit(make_request("hi", /*stream=*/true));
    ASSERT_TRUE(result.ok());
    const auto events = runtime.poll();

    ASSERT_EQ(events.size(), size_t{3});
    ASSERT_EQ(events[0].request_id, result.request_id);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::StreamedToken);
    ASSERT_EQ(events[0].text, "Hello ");
    ASSERT_EQ(events[1].request_id, result.request_id);
    ASSERT_TRUE(events[1].kind == Chorus::RuntimeEvent::Kind::StreamedToken);
    ASSERT_EQ(events[1].text, "world");
    ASSERT_EQ(events[2].request_id, result.request_id);
    ASSERT_TRUE(events[2].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[2].text, "Hello world");
    ASSERT_EQ(terminal_count(events, result.request_id), size_t{1});
    ASSERT_TRUE(runtime.poll().empty());
}

INSTANTIATE_TEST_SUITE_P(
    ProviderDefects,
    RuntimeBrokenEventDefense,
    ::testing::Values(
        BrokenEventCase{"DuplicateTerminal", BrokenEventDefect::DuplicateTerminal},
        BrokenEventCase{"TokenAfterTerminal", BrokenEventDefect::TokenAfterTerminal},
        BrokenEventCase{"UnknownRequestId", BrokenEventDefect::UnknownRequestId}
    ),
    [](const ::testing::TestParamInfo<BrokenEventCase>& info) { return info.param.name; }
);

TEST(Runtime, Runtime_embedding_waits_for_Stop_then_emits_its_vector) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->emit_embedding_event = true;
    engine->embedding_values = {3.0F, 4.0F};
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    const auto result = runtime.submit(make_embedding_request("hi"));
    ASSERT_TRUE(result.ok());
    const auto events = runtime.poll();

    ASSERT_EQ(events.size(), size_t{1});
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Embedding);
    ASSERT_EQ(events[0].request_id, result.request_id);
    ASSERT_EQ(events[0].embedding, std::vector<float>({3.0F, 4.0F}));
    ASSERT_TRUE(events[0].text.empty());
    ASSERT_EQ(terminal_count(events, result.request_id), size_t{1});
}

TEST(Runtime, Runtime_embedding_error_discards_a_pending_vector) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->emit_embedding_event = true;
    engine->emit_error_instead_of_stop = true;
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    const auto result = runtime.submit(make_embedding_request("hi"));
    ASSERT_TRUE(result.ok());
    const auto events = runtime.poll();

    ASSERT_EQ(events.size(), size_t{1});
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Error);
    ASSERT_TRUE(events[0].embedding.empty());
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::Decode);
    ASSERT_EQ(terminal_count(events, result.request_id), size_t{1});
}

TEST(Runtime, Runtime_embedding_does_not_create_conversation_history) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->emit_embedding_event = true;
    ASSERT_TRUE(!runtime.load_engine(std::move(engine), make_config()).has_value());

    const auto result = runtime.submit(make_embedding_request("hi"));
    ASSERT_TRUE(result.ok());
    (void)runtime.poll();
    ASSERT_TRUE(runtime.list_conversations().empty());
}

TEST(Runtime, Runtime_mismatched_embedding_rolls_back_the_generation_turn) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->emit_embedding_event = true;
    ASSERT_TRUE(!runtime.load_engine(std::move(engine), make_config()).has_value());

    auto request = make_request("discard this user turn");
    request.session_id = "session";
    const auto submitted = runtime.submit(request);
    ASSERT_TRUE(submitted.ok());

    const auto events = runtime.poll();
    ASSERT_EQ(events.size(), size_t{1});
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Error);
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::Unknown);
    ASSERT_TRUE(runtime.export_conversation_history("session").empty());
    ASSERT_TRUE(runtime.last_turn_outcome("session") == Chorus::TurnOutcome::Errored);
    ASSERT_TRUE(!runtime.active_request_for_session("session").has_value());
}

TEST(Runtime, Runtime_capabilities_copy_the_loaded_engine_effective_value) {
    Chorus::ChorusRuntime runtime;
    ASSERT_TRUE(!runtime.capabilities().has_value());
    ASSERT_TRUE(!runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config()).has_value());
    const auto capabilities = runtime.capabilities();
    ASSERT_TRUE(capabilities.has_value());
    ASSERT_TRUE(capabilities->embeddings);
}

TEST(Runtime, Runtime_cancel_unknown_request_returns_false) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    auto* seen = engine.get();
    runtime.load_engine(std::move(engine), make_config());

    ASSERT_TRUE(!runtime.cancel(404));
    ASSERT_TRUE(!runtime.is_request_active(404));
    ASSERT_TRUE(seen->cancelled_ids.empty());
}

TEST(Runtime, Runtime_cancel_forwards_each_time_while_live) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    auto* seen = engine.get();
    engine->hold_requests = true;
    runtime.load_engine(std::move(engine), make_config());
    auto result = runtime.submit(make_request("held"));

    ASSERT_TRUE(runtime.cancel(result.request_id));
    ASSERT_TRUE(runtime.cancel(result.request_id));
    ASSERT_EQ(seen->cancelled_ids.size(), 2);
    ASSERT_EQ(seen->cancelled_ids[0], result.request_id);
    ASSERT_EQ(seen->cancelled_ids[1], result.request_id);
    ASSERT_TRUE(runtime.is_request_active(result.request_id));
}

TEST(Runtime, Runtime_cancelled_request_active_until_terminal_drain) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    auto* seen = engine.get();
    engine->hold_requests = true;
    engine->emit_cancelled_on_cancel = true;
    runtime.load_engine(std::move(engine), make_config());
    auto result = runtime.submit(make_request("held"));

    ASSERT_TRUE(runtime.cancel(result.request_id));
    ASSERT_TRUE(runtime.is_request_active(result.request_id));
    ASSERT_TRUE(runtime.cancel(result.request_id));
    ASSERT_EQ(seen->cancelled_ids.size(), 2);

    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1);
    ASSERT_EQ(events[0].request_id, result.request_id);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Error);
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::Cancelled);
    ASSERT_TRUE(!runtime.is_request_active(result.request_id));
    ASSERT_TRUE(!runtime.cancel(result.request_id));
    ASSERT_EQ(seen->cancelled_ids.size(), 2);
}

TEST(Runtime, Runtime_cancel_forwards_after_terminal_enqueue_before_poll) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    auto* seen = engine.get();
    runtime.load_engine(std::move(engine), make_config());
    auto result = runtime.submit(make_request("completing"));

    ASSERT_TRUE(runtime.is_request_active(result.request_id));
    ASSERT_TRUE(runtime.cancel(result.request_id));
    ASSERT_EQ(seen->cancelled_ids.size(), 1);
    ASSERT_EQ(seen->cancelled_ids[0], result.request_id);

    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_TRUE(!runtime.is_request_active(result.request_id));
}

TEST(Runtime, Runtime_cancel_two_request_isolation) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    auto* seen = engine.get();
    engine->hold_requests = true;
    engine->emit_cancelled_on_cancel = true;
    runtime.load_engine(std::move(engine), make_config());
    auto cancelled = runtime.submit(make_request("cancelled"));
    auto untouched = runtime.submit(make_request("untouched"));

    ASSERT_TRUE(runtime.cancel(cancelled.request_id));
    auto events = runtime.poll();

    ASSERT_EQ(events.size(), 1);
    ASSERT_EQ(events[0].request_id, cancelled.request_id);
    ASSERT_TRUE(!runtime.is_request_active(cancelled.request_id));
    ASSERT_TRUE(runtime.is_request_active(untouched.request_id));
    ASSERT_EQ(seen->cancelled_ids.size(), 1);
    ASSERT_EQ(seen->cancelled_ids[0], cancelled.request_id);
}

TEST(Runtime, Runtime_stop_all_yields_one_Cancelled_per_live_request) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->hold_requests = true;
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    auto a = runtime.submit(make_request("a"));
    auto b = runtime.submit(make_request("b"));
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    runtime.stop_all();
    ASSERT_TRUE(!runtime.is_loaded());

    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 2);
    ASSERT_EQ(terminal_count(events, a.request_id), size_t{1});
    ASSERT_EQ(terminal_count(events, b.request_id), size_t{1});
    for (const auto& ev : events) {
        ASSERT_TRUE(ev.kind == Chorus::RuntimeEvent::Kind::Error);
        ASSERT_TRUE(ev.error == Chorus::ChorusError::Cancelled);
        ASSERT_TRUE(ev.request_id == a.request_id || ev.request_id == b.request_id);
    }
    ASSERT_EQ(runtime.poll().size(), 0);
}

TEST(Runtime, Runtime_replacing_engine_cancels_and_new_engine_works) {
    Chorus::ChorusRuntime runtime;
    int first_stops = 0;
    auto first = std::make_unique<SyncMockEngine>();
    first->hold_requests = true;
    first->shutdown_count_sink = &first_stops;
    auto load_err = runtime.load_engine(std::move(first), make_config());
    ASSERT_TRUE(!load_err.has_value());
    auto held = runtime.submit(make_request("held"));
    ASSERT_TRUE(held.ok());

    auto err = runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());
    ASSERT_TRUE(!err.has_value());
    ASSERT_EQ(first_stops, 1); // old engine stopped before replacement
    ASSERT_TRUE(runtime.is_loaded());

    auto fresh = runtime.submit(make_request("fresh"));
    ASSERT_TRUE(fresh.ok());
    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 2);
    ASSERT_EQ(events[0].request_id, held.request_id);
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::Cancelled);
    ASSERT_EQ(events[1].request_id, fresh.request_id);
    ASSERT_TRUE(events[1].kind == Chorus::RuntimeEvent::Kind::Complete);
}

TEST(Runtime, Runtime_failed_replacement_cancels_and_unloads) {
    Chorus::ChorusRuntime runtime;
    auto first = std::make_unique<SyncMockEngine>();
    first->hold_requests = true;
    auto load_err = runtime.load_engine(std::move(first), make_config());
    ASSERT_TRUE(!load_err.has_value());
    auto held = runtime.submit(make_request("held"));
    ASSERT_TRUE(held.ok());

    auto bad = std::make_unique<SyncMockEngine>();
    bad->fail_initialize_with = Chorus::ChorusError::ModelLoad;
    auto err = runtime.load_engine(std::move(bad), make_config());
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(!runtime.is_loaded());

    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1);
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::Cancelled);
}

TEST(Runtime, Runtime_engine_death_yields_one_EngineFailed_once) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    SyncMockEngine* mock = engine.get();
    const auto load_error = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_error.has_value());
    ASSERT_TRUE(runtime.is_loaded());

    mock->die();
    const auto events = runtime.poll();
    ASSERT_EQ(events.size(), size_t{1});
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::EngineFailed);
    ASSERT_EQ(events[0].request_id, Chorus::RequestId{-1});
    ASSERT_EQ(events[0].text, "The engine has failed.");
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::EngineNotReady);
    ASSERT_TRUE(events[0].reasoning.empty());
    ASSERT_TRUE(events[0].omitted_message_ids.empty());
    ASSERT_TRUE(!events[0].session_id.has_value());
    ASSERT_TRUE(runtime.poll().empty());
    ASSERT_TRUE(runtime.poll().empty());
}

TEST(Runtime, Runtime_ordinary_unload_emits_no_EngineFailed) {
    Chorus::ChorusRuntime runtime;
    auto load_err = runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());
    ASSERT_TRUE(!load_err.has_value());

    // stop_all() also takes is_initialized() true -> false; only a death speaks.
    runtime.stop_all();
    ASSERT_EQ(runtime.poll().size(), 0);

    // Nor does replacing a healthy engine.
    ASSERT_TRUE(!runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config()).has_value());
    ASSERT_EQ(runtime.poll().size(), 0);
}

TEST(Runtime, Runtime_dying_terminal_precedes_EngineFailed) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->hold_requests = true;
    SyncMockEngine* mock = engine.get();
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    auto held = runtime.submit(make_request("held"));
    ASSERT_TRUE(held.ok());

    mock->die();
    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 2);
    ASSERT_EQ(events[0].request_id, held.request_id);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Error);
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::Decode);
    ASSERT_TRUE(events[1].kind == Chorus::RuntimeEvent::Kind::EngineFailed);
}

TEST(Runtime, Runtime_submit_after_engine_death_is_EngineNotReady) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    SyncMockEngine* mock = engine.get();
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    mock->die();
    ASSERT_TRUE(!runtime.is_loaded());

    auto result = runtime.submit(make_request("hi"));
    ASSERT_TRUE(!result.ok());
    ASSERT_TRUE(result.error == Chorus::ChorusError::EngineNotReady);
    ASSERT_EQ(result.request_id, -1);
    // A dead engine is not an absent one; the host is told which it is.
    ASSERT_TRUE(result.message != "No engine is loaded.");
}

TEST(Runtime, Runtime_engine_error_during_stop_wins_over_Cancelled) {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->hold_requests = true;
    engine->emit_error_during_shutdown = true;
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    auto held = runtime.submit(make_request("held"));
    ASSERT_TRUE(held.ok());
    runtime.stop_all();

    // The engine's own Error (emitted inside shutdown(), before it returned) is
    // drained first; the synthesized Cancelled for the same id is then dropped
    // by the liveness rule. Exactly one terminal.
    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Error);
    ASSERT_TRUE(events[0].error == Chorus::ChorusError::Decode);
    ASSERT_EQ(runtime.poll().size(), 0);
}

TEST(Runtime, Runtime_consecutive_loads_apply_each_config) {
    Chorus::ChorusRuntime runtime;
    std::string path_a, path_b;
    auto e1 = std::make_unique<SyncMockEngine>();
    e1->seen_model_id = &path_a;
    auto e2 = std::make_unique<SyncMockEngine>();
    e2->seen_model_id = &path_b;

    auto err_a = runtime.load_engine(std::move(e1), make_config("a.gguf"));
    ASSERT_TRUE(!err_a.has_value());
    auto err_b = runtime.load_engine(std::move(e2), make_config("b.gguf"));
    ASSERT_TRUE(!err_b.has_value());

    ASSERT_EQ(path_a, "a.gguf");
    ASSERT_EQ(path_b, "b.gguf"); // no reuse: the second load fully applies
}

// A record produced on a provider thread reaches the runtime log channel and
// drains exactly once.
TEST(Runtime, Runtime_provider_thread_log_drains_once) {
    Chorus::ChorusConfig config = make_config();
    config.log_level = Chorus::LogLevel::Debug;

    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->log_on_initialize_from_worker = true;
    runtime.load_engine(std::move(engine), config);

    auto records = runtime.poll_logs();
    ASSERT_EQ(records.size(), size_t{1});
    ASSERT_EQ(records[0].message, "From worker");
    ASSERT_TRUE(records[0].level == Chorus::LogLevel::Info);

    // Drained means drained: a second poll sees nothing.
    ASSERT_EQ(runtime.poll_logs().size(), size_t{0});
}

// The host's declared threshold travels with the config, and the provider that
// would emit the record is the one that declines to build it.
TEST(Runtime, Runtime_log_level_from_the_config_silences_a_level) {
    Chorus::ChorusConfig config = make_config();
    config.log_level = Chorus::log_level_default; // Warn and above, so Info is out

    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->log_on_initialize_from_worker = true;
    runtime.load_engine(std::move(engine), config);

    ASSERT_EQ(runtime.poll_logs().size(), size_t{0});
}

// The shutdown fence cuts callbacks, not the record already in the channel:
// the channel outlives every engine, so what was enqueued still drains.
TEST(Runtime, Runtime_records_enqueued_before_shutdown_drain_afterward) {
    Chorus::ChorusConfig config = make_config();
    config.log_level = Chorus::LogLevel::Debug;

    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->log_on_initialize_from_worker = true;
    runtime.load_engine(std::move(engine), config);
    runtime.stop_all();

    auto records = runtime.poll_logs();
    ASSERT_EQ(records.size(), size_t{1});
    ASSERT_EQ(records[0].message, "From worker");
}

// Shutdown diagnostics remain ordinary records. A host that wants them stops
// explicitly, then drains before destroying the runtime.
TEST(Runtime, Runtime_records_produced_during_stop_drain_afterward) {
    Chorus::ChorusConfig config = make_config();
    config.log_level = Chorus::LogLevel::Debug;

    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->log_on_shutdown = true;
    runtime.load_engine(std::move(engine), config);
    ASSERT_EQ(runtime.poll_logs().size(), size_t{0});

    runtime.stop_all();
    const auto records = runtime.poll_logs();
    ASSERT_EQ(records.size(), size_t{2});
    ASSERT_EQ(records[0].message, "Shutting down with work outstanding");
    ASSERT_EQ(records[1].message, "Releasing weights");
}

TEST(Runtime, Runtime_destruction_with_active_requests_shuts_down_engine_once) {
    int shutdown_count = 0;
    {
        Chorus::ChorusRuntime runtime;
        auto engine = std::make_unique<SyncMockEngine>();
        engine->hold_requests = true;
        engine->shutdown_count_sink = &shutdown_count;
        const auto load_error = runtime.load_engine(std::move(engine), make_config());
        ASSERT_TRUE(!load_error.has_value());
        const auto held = runtime.submit(make_request("held"));
        ASSERT_TRUE(held.ok());
    }

    ASSERT_EQ(shutdown_count, 1);
}


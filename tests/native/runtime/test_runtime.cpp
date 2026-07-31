#include "chorus/runtime/runtime.hpp"
#include "sync_mock_engine.hpp"
#include "test_utils.hpp"

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

void test_submit_before_load_returns_EngineNotReady_without_touching_engine() {
    Chorus::ChorusRuntime runtime;
    ASSERT_TRUE(!runtime.is_loaded());

    auto result = runtime.submit(make_request("hi"));
    ASSERT_TRUE(!result.ok());
    ASSERT_TRUE(result.error == Chorus::ChorusError::EngineNotReady);
    ASSERT_EQ(result.request_id, -1);
}

void test_load_engine_success_and_is_loaded() {
    Chorus::ChorusRuntime runtime;
    auto err = runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());
    ASSERT_TRUE(!err.has_value());
    ASSERT_TRUE(runtime.is_loaded());
}

void test_load_engine_null_engine_is_InvalidRequest() {
    Chorus::ChorusRuntime runtime;
    auto err = runtime.load_engine(nullptr, make_config());
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(err.value() == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(!runtime.is_loaded());
}

void test_failed_load_leaves_runtime_unloaded() {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->fail_initialize_with = Chorus::ChorusError::ModelLoad;

    auto err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(err.value() == Chorus::ChorusError::ModelLoad);
    ASSERT_TRUE(!runtime.is_loaded());
}

void test_ids_are_monotonic_and_unique() {
    Chorus::ChorusRuntime runtime;
    runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());

    auto a = runtime.submit(make_request("a"));
    auto b = runtime.submit(make_request("b"));
    auto c = runtime.submit(make_request("c"));
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    ASSERT_TRUE(c.ok());
    ASSERT_TRUE(a.request_id < b.request_id);
    ASSERT_TRUE(b.request_id < c.request_id);
}

void test_poll_on_idle_runtime_returns_empty() {
    Chorus::ChorusRuntime runtime;
    ASSERT_EQ(runtime.poll().size(), 0);
    runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());
    ASSERT_EQ(runtime.poll().size(), 0);
}

void test_non_streaming_request_yields_only_Complete_with_full_text() {
    Chorus::ChorusRuntime runtime;
    runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());

    auto result = runtime.submit(make_request("hi", /*stream=*/false));
    auto events = runtime.poll();

    ASSERT_EQ(events.size(), 1);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[0].request_id, result.request_id);
    ASSERT_EQ(events[0].text, "Hello world");
}

void test_streaming_request_yields_tokens_in_order_then_Complete_with_full_text() {
    Chorus::ChorusRuntime runtime;
    runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());

    auto result = runtime.submit(make_request("hi", /*stream=*/true));
    ASSERT_TRUE(result.ok());
    auto events = runtime.poll();

    ASSERT_EQ(events.size(), 3);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Token);
    ASSERT_EQ(events[0].text, "Hello ");
    ASSERT_TRUE(events[1].kind == Chorus::RuntimeEvent::Kind::Token);
    ASSERT_EQ(events[1].text, "world");
    ASSERT_TRUE(events[2].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[2].text, "Hello world");
}

void test_interleaved_requests_accumulate_independently() {
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

void test_inline_rejection_during_submit_is_delivered_on_next_poll() {
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

void test_error_after_tokens_discards_partial_text() {
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
    ASSERT_EQ(runtime.poll().size(), 0);
}

void test_duplicate_terminal_is_dropped() {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->emit_duplicate_stop = true;
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    auto result = runtime.submit(make_request("hi", /*stream=*/true));
    ASSERT_TRUE(result.ok());
    auto events = runtime.poll();

    // 2 tokens + 1 Complete. The duplicate Stop is dropped.
    ASSERT_EQ(events.size(), 3);
    ASSERT_TRUE(events[2].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[2].text, "Hello world");
    ASSERT_EQ(runtime.poll().size(), 0);
}

void test_token_after_terminal_is_dropped() {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->emit_token_after_stop = true;
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    auto result = runtime.submit(make_request("hi", /*stream=*/true));
    ASSERT_TRUE(result.ok());
    auto events = runtime.poll();

    // 2 tokens + 1 Complete. The post-Stop token is dropped.
    ASSERT_EQ(events.size(), 3);
    ASSERT_TRUE(events[2].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[2].text, "Hello world");
    ASSERT_EQ(runtime.poll().size(), 0);
}

void test_unknown_id_events_are_dropped() {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->rogue_extra_id = 999999;
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    auto result = runtime.submit(make_request("hi", /*stream=*/true));
    ASSERT_TRUE(result.ok());
    auto events = runtime.poll();

    // 2 tokens + 1 Complete. The rogue-id token is dropped.
    ASSERT_EQ(events.size(), 3);
    ASSERT_TRUE(events[2].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[2].text, "Hello world");
    ASSERT_EQ(runtime.poll().size(), 0);
}

void test_embedding_events_are_dropped_without_corrupting_accumulation() {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->emit_embedding_event = true;
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    auto result = runtime.submit(make_request("hi"));
    ASSERT_TRUE(result.ok());
    auto events = runtime.poll();

    ASSERT_EQ(events.size(), 1);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_EQ(events[0].text, "Hello world");
}

void test_cancel_unknown_request_returns_false_without_forwarding() {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    auto* seen = engine.get();
    runtime.load_engine(std::move(engine), make_config());

    ASSERT_TRUE(!runtime.cancel(404));
    ASSERT_TRUE(!runtime.is_request_active(404));
    ASSERT_TRUE(seen->cancelled_ids.empty());
}

void test_cancel_forwards_each_time_while_request_remains_live() {
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

void test_cancelled_request_stays_active_until_terminal_is_drained() {
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

void test_cancel_forwards_after_engine_terminal_enqueue_before_poll() {
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

void test_cancel_one_request_does_not_affect_another() {
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

void test_stop_all_yields_exactly_one_Cancelled_terminal_per_live_request() {
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
    for (const auto& ev : events) {
        ASSERT_TRUE(ev.kind == Chorus::RuntimeEvent::Kind::Error);
        ASSERT_TRUE(ev.error == Chorus::ChorusError::Cancelled);
        ASSERT_TRUE(ev.request_id == a.request_id || ev.request_id == b.request_id);
    }
    ASSERT_EQ(runtime.poll().size(), 0);
}

void test_replacing_engine_cancels_live_requests_and_new_engine_works() {
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

void test_failed_replacement_cancels_live_requests_and_unloads() {
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

void test_engine_death_yields_one_EngineFailed_on_next_poll() {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    SyncMockEngine* mock = engine.get();
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());
    ASSERT_TRUE(runtime.is_loaded());

    mock->die();
    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1);
    ASSERT_TRUE(events[0].kind == Chorus::RuntimeEvent::Kind::EngineFailed);
    ASSERT_EQ(events[0].request_id, -1);
    ASSERT_TRUE(!events[0].session_id.has_value());
}

void test_engine_death_is_reported_once_across_polls() {
    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    SyncMockEngine* mock = engine.get();
    auto load_err = runtime.load_engine(std::move(engine), make_config());
    ASSERT_TRUE(!load_err.has_value());

    mock->die();
    ASSERT_EQ(runtime.poll().size(), 1);
    ASSERT_EQ(runtime.poll().size(), 0);
    ASSERT_EQ(runtime.poll().size(), 0);
}

void test_ordinary_unload_emits_no_EngineFailed() {
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

void test_dying_request_terminal_precedes_EngineFailed() {
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

void test_submit_after_engine_death_is_EngineNotReady_and_says_so() {
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

void test_engine_error_during_stop_wins_over_synthesized_Cancelled() {
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

void test_two_consecutive_loads_apply_each_config() {
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

void test_log_callback_passes_through_from_worker_thread() {
    std::mutex log_mutex;
    std::vector<std::string> log_lines;

    Chorus::ChorusConfig config = make_config();
    config.log_callback = [&](Chorus::LogLevel, const std::string& msg) {
        std::lock_guard<std::mutex> lock(log_mutex);
        log_lines.push_back(msg);
    };

    Chorus::ChorusRuntime runtime;
    auto engine = std::make_unique<SyncMockEngine>();
    engine->log_on_initialize_from_worker = true;
    runtime.load_engine(std::move(engine), config);

    std::lock_guard<std::mutex> lock(log_mutex);
    ASSERT_EQ(log_lines.size(), 1);
    ASSERT_EQ(log_lines[0], "from worker");
}

void test_destruction_with_active_requests_is_clean() {
    {
        Chorus::ChorusRuntime runtime;
        auto engine = std::make_unique<SyncMockEngine>();
        engine->hold_requests = true;
        auto load_err = runtime.load_engine(std::move(engine), make_config());
        ASSERT_TRUE(!load_err.has_value());
        auto held = runtime.submit(make_request("held"));
        ASSERT_TRUE(held.ok());
    } // destructor stops the engine; pending events die with the runtime
    ASSERT_TRUE(true);
}

// Terminal invariant at the runtime layer: for each accepted request id, poll yields
// exactly one terminal event (Complete or Error) across success, decode failure,
// duplicate-terminal defense, and engine stop. This is the cross-layer companion to
// the engine-level terminal sweeps in the Echo and Llama suites.
void test_runtime_exactly_one_terminal_per_accepted_request() {
    const auto terminal_count = [](const std::vector<Chorus::RuntimeEvent>& events, Chorus::RequestId id) {
        size_t count = 0;
        for (const auto& event : events)
            count += event.request_id == id && (event.kind == Chorus::RuntimeEvent::Kind::Complete ||
                                                event.kind == Chorus::RuntimeEvent::Kind::Error);
        return count;
    };

    // Success: one Complete terminal.
    {
        Chorus::ChorusRuntime runtime;
        runtime.load_engine(std::make_unique<SyncMockEngine>(), make_config());
        auto result = runtime.submit(make_request("hi", /*stream=*/true));
        ASSERT_TRUE(result.ok());
        auto events = runtime.poll();
        ASSERT_EQ(terminal_count(events, result.request_id), size_t{1});
        ASSERT_TRUE(events.back().kind == Chorus::RuntimeEvent::Kind::Complete);
    }

    // Decode failure after partial output: one Error terminal, no Complete.
    {
        Chorus::ChorusRuntime runtime;
        auto engine = std::make_unique<SyncMockEngine>();
        engine->emit_error_instead_of_stop = true;
        runtime.load_engine(std::move(engine), make_config());
        auto result = runtime.submit(make_request("hi", /*stream=*/false));
        ASSERT_TRUE(result.ok());
        auto events = runtime.poll();
        ASSERT_EQ(terminal_count(events, result.request_id), size_t{1});
        ASSERT_TRUE(events.back().kind == Chorus::RuntimeEvent::Kind::Error);
        ASSERT_TRUE(events.back().error == Chorus::ChorusError::Decode);
    }

    // Broken provider emitting a duplicate terminal: the runtime still surfaces one.
    {
        Chorus::ChorusRuntime runtime;
        auto engine = std::make_unique<SyncMockEngine>();
        engine->emit_duplicate_stop = true;
        runtime.load_engine(std::move(engine), make_config());
        auto result = runtime.submit(make_request("hi", /*stream=*/true));
        ASSERT_TRUE(result.ok());
        auto events = runtime.poll();
        ASSERT_EQ(terminal_count(events, result.request_id), size_t{1});
    }

    // Engine stop: each live request gets exactly one synthesized Cancelled terminal.
    {
        Chorus::ChorusRuntime runtime;
        auto engine = std::make_unique<SyncMockEngine>();
        engine->hold_requests = true;
        runtime.load_engine(std::move(engine), make_config());
        auto a = runtime.submit(make_request("a"));
        auto b = runtime.submit(make_request("b"));
        ASSERT_TRUE(a.ok());
        ASSERT_TRUE(b.ok());
        runtime.stop_all();
        auto events = runtime.poll();
        ASSERT_EQ(terminal_count(events, a.request_id), size_t{1});
        ASSERT_EQ(terminal_count(events, b.request_id), size_t{1});
        for (const auto& event : events) {
            ASSERT_TRUE(event.kind == Chorus::RuntimeEvent::Kind::Error);
            ASSERT_TRUE(event.error == Chorus::ChorusError::Cancelled);
        }
    }
}

int run_runtime_tests() {
    std::cout << "\n--- RUNTIME TEST SUITE ---\n";

    run_test(
        "Runtime_submit_before_load_returns_EngineNotReady",
        test_submit_before_load_returns_EngineNotReady_without_touching_engine
    );
    run_test("Runtime_load_engine_success_and_is_loaded", test_load_engine_success_and_is_loaded);
    run_test("Runtime_load_engine_null_engine_is_InvalidRequest", test_load_engine_null_engine_is_InvalidRequest);
    run_test("Runtime_failed_load_leaves_runtime_unloaded", test_failed_load_leaves_runtime_unloaded);
    run_test("Runtime_ids_are_monotonic_and_unique", test_ids_are_monotonic_and_unique);
    run_test("Runtime_poll_on_idle_runtime_returns_empty", test_poll_on_idle_runtime_returns_empty);
    run_test(
        "Runtime_non_streaming_yields_only_Complete", test_non_streaming_request_yields_only_Complete_with_full_text
    );
    run_test(
        "Runtime_streaming_yields_tokens_then_Complete",
        test_streaming_request_yields_tokens_in_order_then_Complete_with_full_text
    );
    run_test(
        "Runtime_interleaved_requests_accumulate_independently", test_interleaved_requests_accumulate_independently
    );
    run_test(
        "Runtime_inline_rejection_delivered_on_next_poll", test_inline_rejection_during_submit_is_delivered_on_next_poll
    );
    run_test("Runtime_error_after_tokens_discards_partial", test_error_after_tokens_discards_partial_text);
    run_test("Runtime_duplicate_terminal_is_dropped", test_duplicate_terminal_is_dropped);
    run_test("Runtime_token_after_terminal_is_dropped", test_token_after_terminal_is_dropped);
    run_test("Runtime_unknown_id_events_are_dropped", test_unknown_id_events_are_dropped);
    run_test("Runtime_embedding_events_are_dropped", test_embedding_events_are_dropped_without_corrupting_accumulation);
    run_test(
        "Runtime_cancel_unknown_request_returns_false", test_cancel_unknown_request_returns_false_without_forwarding
    );
    run_test("Runtime_cancel_forwards_each_time_while_live", test_cancel_forwards_each_time_while_request_remains_live);
    run_test(
        "Runtime_cancelled_request_active_until_terminal_drain",
        test_cancelled_request_stays_active_until_terminal_is_drained
    );
    run_test(
        "Runtime_cancel_forwards_after_terminal_enqueue_before_poll",
        test_cancel_forwards_after_engine_terminal_enqueue_before_poll
    );
    run_test("Runtime_cancel_two_request_isolation", test_cancel_one_request_does_not_affect_another);
    run_test(
        "Runtime_stop_all_yields_one_Cancelled_per_live_request",
        test_stop_all_yields_exactly_one_Cancelled_terminal_per_live_request
    );
    run_test(
        "Runtime_replacing_engine_cancels_and_new_engine_works",
        test_replacing_engine_cancels_live_requests_and_new_engine_works
    );
    run_test(
        "Runtime_failed_replacement_cancels_and_unloads", test_failed_replacement_cancels_live_requests_and_unloads
    );
    run_test("Runtime_engine_death_yields_one_EngineFailed", test_engine_death_yields_one_EngineFailed_on_next_poll);
    run_test("Runtime_engine_death_reported_once", test_engine_death_is_reported_once_across_polls);
    run_test("Runtime_ordinary_unload_emits_no_EngineFailed", test_ordinary_unload_emits_no_EngineFailed);
    run_test("Runtime_dying_terminal_precedes_EngineFailed", test_dying_request_terminal_precedes_EngineFailed);
    run_test(
        "Runtime_submit_after_engine_death_is_EngineNotReady",
        test_submit_after_engine_death_is_EngineNotReady_and_says_so
    );
    run_test(
        "Runtime_engine_error_during_stop_wins_over_Cancelled",
        test_engine_error_during_stop_wins_over_synthesized_Cancelled
    );
    run_test("Runtime_consecutive_loads_apply_each_config", test_two_consecutive_loads_apply_each_config);
    run_test("Runtime_destruction_with_active_requests_is_clean", test_destruction_with_active_requests_is_clean);
    run_test(
        "Runtime_log_callback_passes_through_from_worker_thread", test_log_callback_passes_through_from_worker_thread
    );
    run_test(
        "Runtime_exactly_one_terminal_per_accepted_request", test_runtime_exactly_one_terminal_per_accepted_request
    );

    return g_tests_failed > 0 ? 1 : 0;
}

#include "chorus/core/common.hpp"
#include "chorus/backends/echo/echo_engine.hpp"
#include "test_utils.hpp"

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// No model, no skips: this suite must pass under CHORUS_SKIP_MODEL_TESTS=1.

void test_echo_initializes_without_a_model_file() {
    Chorus::EchoEngine engine;
    ASSERT_TRUE(!engine.is_initialized());

    Chorus::ChorusConfig config; // model_path deliberately empty
    ASSERT_TRUE(!engine.initialize(config).has_value());
    ASSERT_TRUE(engine.is_initialized());

    engine.stop();
    ASSERT_TRUE(!engine.is_initialized());
}

void test_echo_streams_prompt_word_by_word_then_stops() {
    std::mutex sig_mutex;
    std::vector<Chorus::ChorusSignal> sigs;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest req;
    req.id = 7;
    req.prompt = "hello chorus seam";
    req.on_event = [&](Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(sig_mutex);
        sigs.push_back(sig);
    };
    engine.submit_request(req);

    int timeout_ms = 2000;
    while (timeout_ms > 0) {
        {
            std::lock_guard<std::mutex> lock(sig_mutex);
            if (!sigs.empty() && sigs.back().type == Chorus::EventType::Stop)
                break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        timeout_ms -= 10;
    }

    std::lock_guard<std::mutex> lock(sig_mutex);
    // 3 words -> 3 Token signals + 1 Stop, all carrying the request id.
    ASSERT_EQ(sigs.size(), 4);
    std::string reassembled;
    for (size_t i = 0; i + 1 < sigs.size(); ++i) {
        ASSERT_TRUE(sigs[i].type == Chorus::EventType::Token);
        ASSERT_EQ(sigs[i].request_id, 7);
        reassembled += sigs[i].text;
    }
    ASSERT_TRUE(reassembled == "hello chorus seam");
    ASSERT_TRUE(sigs.back().type == Chorus::EventType::Stop);
    ASSERT_EQ(sigs.back().request_id, 7);

    engine.stop();
}

void test_echo_submit_before_initialize_signals_engine_not_ready() {
    // Pinned in the spec (2026-07-08): pre-init submit mirrors LlamaEngine exactly,
    // so a test passing on Echo cannot silently fail on Llama.
    Chorus::EchoEngine engine;

    bool errored = false;
    Chorus::ChorusError code = Chorus::ChorusError::None;

    Chorus::ChorusRequest req;
    req.id = 3;
    req.prompt = "ignored";
    req.on_event = [&](Chorus::ChorusSignal& sig) {
        // pre-init failure is emitted synchronously on the caller thread
        errored = sig.is_error();
        code = sig.error_code;
    };
    engine.submit_request(req);

    ASSERT_TRUE(errored);
    ASSERT_TRUE(code == Chorus::ChorusError::EngineNotReady);
}

int run_echo_engine_tests() {
    std::cout << "\n--- ECHO ENGINE SUITE ---\n";

    run_test("Echo_initializes_without_a_model_file", test_echo_initializes_without_a_model_file);
    run_test("Echo_streams_prompt_word_by_word_then_stops", test_echo_streams_prompt_word_by_word_then_stops);
    run_test(
        "Echo_submit_before_initialize_signals_EngineNotReady",
        test_echo_submit_before_initialize_signals_engine_not_ready
    );

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << RED << "SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED." << RESET
                  << "\n";
        return 1;
    }
    std::cout << GREEN << "SUMMARY: ALL TESTS PASSED." << RESET << "\n";
    return 0;
}

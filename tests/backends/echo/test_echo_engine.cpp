#include "chorus/backends/echo/echo_engine.hpp"
#include "chorus/core/common.hpp"
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

    Chorus::ChorusConfig config; // empty ModelSpec deliberately: Echo needs no model
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

void test_echo_capabilities_deterministic_across_init() {
    Chorus::EchoEngine engine;
    auto before = engine.capabilities();
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());
    auto after = engine.capabilities();
    ASSERT_EQ(before.backend_id, std::string("echo"));
    ASSERT_EQ(after.backend_id, before.backend_id);
    ASSERT_TRUE(before.streaming && after.streaming);
    ASSERT_TRUE(before.scheduling == Chorus::SchedulingAuthority::BackendManaged);
    ASSERT_TRUE(before.portable_generation_options.empty());
    ASSERT_TRUE(!engine.loaded_model_info().has_value()); // model-free by design
    engine.stop();
}

void test_echo_rejects_set_but_unsupported_controls() {
    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = "hi";

    req.gen_config.common.temperature = 0.5f; // set but unsupported
    auto r1 = engine.validate_request(req);
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r1->error == Chorus::ChorusError::UnsupportedOption);

    req = {};
    req.id = 2;
    req.prompt = "hi";
    req.gen_config.common.constraint = Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, "root ::= \"x\""};
    auto r2 = engine.validate_request(req);
    ASSERT_TRUE(r2.has_value());
    ASSERT_TRUE(r2->error == Chorus::ChorusError::UnsupportedFeature);

    req = {};
    req.id = 3;
    req.prompt = "hi";
    req.gen_config.backend_options["echo"] = Chorus::OptionMap{{"volume", int64_t{11}}};
    auto r3 = engine.validate_request(req);
    ASSERT_TRUE(r3.has_value());
    ASSERT_TRUE(r3->error == Chorus::ChorusError::UnsupportedOption);
    engine.stop();
}

void test_echo_accepts_session_id_as_correlation() {
    // Correlation is universal: native_sessions=false must not reject it.
    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());
    Chorus::ChorusRequest req;
    req.id = 4;
    req.prompt = "hi";
    req.session_id = "npc_42/dialogue";
    ASSERT_TRUE(!engine.validate_request(req).has_value());
    ASSERT_TRUE(!engine.capabilities().native_sessions);
    engine.stop();
}

int run_echo_engine_tests() {
    std::cout << "\n--- ECHO ENGINE SUITE ---\n";

    run_test("Echo_initializes_without_a_model_file", test_echo_initializes_without_a_model_file);
    run_test("Echo_streams_prompt_word_by_word_then_stops", test_echo_streams_prompt_word_by_word_then_stops);
    run_test(
        "Echo_submit_before_initialize_signals_EngineNotReady",
        test_echo_submit_before_initialize_signals_engine_not_ready
    );
    run_test("Echo_capabilities_deterministic_across_init", test_echo_capabilities_deterministic_across_init);
    run_test("Echo_rejects_set_but_unsupported_controls", test_echo_rejects_set_but_unsupported_controls);
    run_test("Echo_accepts_session_id_as_correlation", test_echo_accepts_session_id_as_correlation);

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << RED << "SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED." << RESET
                  << "\n";
        return 1;
    }
    std::cout << GREEN << "SUMMARY: ALL TESTS PASSED." << RESET << "\n";
    return 0;
}

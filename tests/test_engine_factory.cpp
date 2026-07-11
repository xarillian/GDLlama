#include "chorus/core/common.hpp"
#include "chorus/engine_factory.hpp"
#include "test_utils.hpp"

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Model-free: exercises the factory seam, not llama.cpp.

void test_factory_echo_engine_round_trips_through_interface() {
    std::mutex sig_mutex;
    std::vector<Chorus::ChorusSignal> sigs;

    std::unique_ptr<Chorus::InferenceEngine> engine = Chorus::make_engine(Chorus::Backend::Echo);
    ASSERT_TRUE(engine != nullptr);
    ASSERT_TRUE(!engine->is_initialized());

    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine->initialize(config).has_value());

    Chorus::ChorusRequest req;
    req.id = 11;
    req.prompt = "seam proof";
    req.on_event = [&](Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(sig_mutex);
        sigs.push_back(sig);
    };
    engine->submit_request(req);

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
    ASSERT_EQ(sigs.size(), 3); // "seam " + "proof" + Stop
    std::string reassembled = sigs[0].text + sigs[1].text;
    ASSERT_TRUE(sigs[0].type == Chorus::EventType::Token);
    ASSERT_TRUE(sigs[1].type == Chorus::EventType::Token);
    ASSERT_TRUE(reassembled == "seam proof");
    ASSERT_TRUE(sigs[2].type == Chorus::EventType::Stop);

    engine->stop();
}

void test_factory_creates_llama_engine_uninitialized() {
    // No model load here: this only proves the factory constructs the real backend.
    std::unique_ptr<Chorus::InferenceEngine> engine = Chorus::make_engine(Chorus::Backend::Llama);
    ASSERT_TRUE(engine != nullptr);
    ASSERT_TRUE(!engine->is_initialized());
}

int run_engine_factory_tests() {
    std::cout << "\n--- ENGINE FACTORY SUITE ---\n";

    run_test(
        "Factory_echo_engine_round_trips_through_interface", test_factory_echo_engine_round_trips_through_interface
    );
    run_test("Factory_creates_llama_engine_uninitialized", test_factory_creates_llama_engine_uninitialized);

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << RED << "SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED." << RESET
                  << "\n";
        return 1;
    }
    std::cout << GREEN << "SUMMARY: ALL TESTS PASSED." << RESET << "\n";
    return 0;
}

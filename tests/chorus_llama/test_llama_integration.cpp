#include "../../include/chorus_core/chorus_common.hpp"
#include "../../include/chorus_llama/llama_engine.hpp"
#include "../test_utils.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

const std::string MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

void test_model_loading() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config;

    config.model_path = MODEL_PATH;
    config.context_size = 1024;
    config.use_gpu = false;

    std::cout << "  [INFO] Loading model: " << MODEL_PATH << std::endl;

    ASSERT_TRUE(!engine.initialize(config).has_value());
    ASSERT_TRUE(engine.is_initialized());

    engine.stop();
    ASSERT_TRUE(!engine.is_initialized());
}

void test_simple_generation() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config;
    config.model_path = MODEL_PATH;
    config.use_gpu = false;

    std::atomic<bool> done{false};
    std::string full_response = "";

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest chorus_request;
    chorus_request.id = 1;
    chorus_request.prompt = "<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n";
    chorus_request.gen_config.max_tokens = 20;
    chorus_request.gen_config.temperature = 0.7f;

    chorus_request.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (sig.type == Chorus::EventType::Token) {
            std::cout << sig.text << std::flush; // Print tokens as they arrive!
            full_response += sig.text;
        } else if (sig.type == Chorus::EventType::Stop) {
            done = true;
        } else if (sig.type == Chorus::EventType::Error) {
            std::cerr << "\n[ERROR] " << sig.text << "\n";
            done = true;
        }
    };

    std::cout << "  [INFO] Sending Prompt: 'Hello, Chorus!'\n";
    std::cout << "  [GENERATION] > ";

    engine.submit_request(chorus_request);

    // Wait loop with timeout (e.g., 10 seconds)
    int timeout_ms = 10000;
    while (!done && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        timeout_ms -= 100;
    }

    std::cout << "\n"; // Newline after generation

    if (timeout_ms <= 0) {
        std::cerr << RED << "[FAILED] Timed out waiting for generation." << RESET << "\n";
        g_tests_failed++;
    } else {
        ASSERT_TRUE(full_response.length() > 0);
        std::cout << "  [INFO] Received " << full_response.length() << " characters.\n";
    }
}

void test_concurrent_requests_complete_with_multiple_slots() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config;
    config.model_path = MODEL_PATH;
    config.use_gpu = false;
    config.num_slots = 2;

    std::atomic<int> completed_count{0};
    std::string responses[2];
    std::mutex responses_mutex;

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config).has_value());

    for (int slot_index = 0; slot_index < 2; ++slot_index) {
        Chorus::ChorusRequest request;
        request.id = slot_index;
        request.prompt = "<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n";
        request.gen_config.max_tokens = 10;
        request.on_event = [&, slot_index](const Chorus::ChorusSignal& sig) {
            if (sig.type == Chorus::EventType::Token) {
                std::lock_guard<std::mutex> lock(responses_mutex);
                responses[slot_index] += sig.text;
            } else if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error) {
                completed_count++;
            }
        };
        engine.submit_request(request);
    }

    int timeout_ms = 30000;
    while (completed_count < 2 && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        timeout_ms -= 100;
    }

    if (timeout_ms <= 0) {
        std::cerr << RED << "[FAILED] Timed out waiting for concurrent generation." << RESET << "\n";
        g_tests_failed++;
        return;
    }

    ASSERT_TRUE(!responses[0].empty());
    ASSERT_TRUE(!responses[1].empty());
}

void test_max_tokens_counts_generated_not_prompt_tokens() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config;
    config.model_path = MODEL_PATH;
    config.use_gpu = false;
    config.context_size = 1024;

    std::atomic<int> token_count{0};
    std::atomic<bool> done{false};

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config).has_value());

    std::string long_prompt = "<start_of_turn>user\n";
    for (int i = 0; i < 40; ++i)
        long_prompt += "Tell me a long and detailed story about dragons and castles. ";
    long_prompt += "<end_of_turn>\n<start_of_turn>model\n";

    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = long_prompt;
    req.gen_config.max_tokens = 8;
    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (sig.type == Chorus::EventType::Token)
            token_count++;
        else if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error)
            done = true;
    };

    engine.submit_request(req);

    int timeout_ms = 15000;
    while (!done && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        timeout_ms -= 100;
    }

    ASSERT_TRUE(done);
    ASSERT_TRUE(token_count > 1);
    ASSERT_TRUE(token_count <= 8);
}

void test_engine_reinitializes_and_generates_after_stop() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config;
    config.model_path = MODEL_PATH;
    config.use_gpu = false;

    std::atomic<int> tokens{0};
    std::atomic<bool> done{false};

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config).has_value());
    engine.stop();
    ASSERT_TRUE(!engine.is_initialized());

    // Re-init must rebuild cleanly (robust stop() freed everything) and still generate.
    ASSERT_TRUE(!engine.initialize(config).has_value());
    ASSERT_TRUE(engine.is_initialized());

    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n";
    req.gen_config.max_tokens = 5;
    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (sig.type == Chorus::EventType::Token)
            tokens++;
        else if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error)
            done = true;
    };
    engine.submit_request(req);

    int timeout_ms = 15000;
    while (!done && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        timeout_ms -= 100;
    }

    ASSERT_TRUE(done);
    ASSERT_TRUE(tokens > 0);
    engine.stop();
}

int run_llama_integration_tests() {
    std::cout << "\n--- LLAMA INTEGRATION SUITE ---\n";

    run_test("Llama_Model_Load", test_model_loading);
    run_test("Llama_Generation_Stream", test_simple_generation);
    run_test(
        "Llama_ConcurrentRequestsCompleteWithMultipleSlots", test_concurrent_requests_complete_with_multiple_slots
    );
    run_test("Max_tokens_counts_generated_not_prompt_tokens", test_max_tokens_counts_generated_not_prompt_tokens);
    run_test("Engine_reinitializes_and_generates_after_stop", test_engine_reinitializes_and_generates_after_stop);

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << RED << "SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED." << RESET
                  << "\n";
        return 1;
    } else {
        std::cout << GREEN << "SUMMARY: ALL TESTS PASSED." << RESET << "\n";
        return 0;
    }
}

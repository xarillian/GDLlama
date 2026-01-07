#include "../../include/chorus_llama/llama_engine.hpp"
#include "../../include/chorus_core/chorus_common.hpp"
#include "../test_utils.hpp"

#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>

const std::string MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

void test_model_loading() {
    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config;

    config.model_path = MODEL_PATH;
    config.context_size = 1024;
    config.use_gpu = false;

    std::cout << "  [INFO] Loading model: " << MODEL_PATH << std::endl;
    bool success = engine.initialize(config);

    ASSERT_TRUE(success);
    ASSERT_TRUE(engine.is_initialized());

    engine.stop();
    ASSERT_TRUE(!engine.is_initialized());
}

void test_simple_generation() {
    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config;
    config.model_path = MODEL_PATH;
    config.use_gpu = false;
    
    if (!engine.initialize(config)) {
        std::cerr << RED << "[SKIP] Could not load model. Check path." << RESET << "\n";
        return;
    }

    Chorus::ChorusRequest chorus_request;
    chorus_request.id = 1;
    chorus_request.prompt = "<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n";
    chorus_request.gen_config.max_tokens = 20;
    chorus_request.gen_config.temperature = 0.7f;

    std::atomic<bool> done{false};
    std::string full_response = "";

    chorus_request.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (sig.type == Chorus::EventType::Token) {
            std::cout << sig.text << std::flush; // Print tokens as they arrive!
            full_response += sig.text;
        } 
        else if (sig.type == Chorus::EventType::Stop) {
            done = true;
        }
        else if (sig.type == Chorus::EventType::Error) {
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

int run_llama_integration_tests() {
    std::cout << "\n--- LLAMA INTEGRATION SUITE ---\n"; 

    FILE* f = fopen(MODEL_PATH.c_str(), "rb");
    if (!f) {
        std::cout << RED << "[ERROR] Test model not found at: " << MODEL_PATH << RESET << "\n";
        return 1;
    }
    fclose(f);

    run_test("Llama_Model_Load", test_model_loading);
    run_test("Llama_Generation_Stream", test_simple_generation);

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << RED << "SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED." << RESET << "\n";
        return 1;
    } else {
        std::cout << GREEN << "SUMMARY: ALL TESTS PASSED." << RESET << "\n";
        return 0;
    }
}

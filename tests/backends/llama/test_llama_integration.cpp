#include "chorus/backends/llama/llama_engine.hpp"
#include "chorus/core/common.hpp"
#include "test_utils.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

const std::string MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

static Chorus::ChorusConfig make_gguf_config(const std::string& path) {
    Chorus::ChorusConfig config;
    config.model.model_id = "test-model";
    config.model.format = Chorus::ModelFormat::Gguf;
    config.model.assets.push_back({"weights", path, std::nullopt, std::nullopt});
    return config;
}

void test_unsupported_model_format_is_rejected() {
    // No model needed: the format gate fires before any file I/O.
    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config = make_gguf_config("/nonexistent.safetensors");
    config.model.format = Chorus::ModelFormat::SafeTensors;
    auto err = engine.initialize(config);
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(*err == Chorus::ChorusError::UnsupportedModelFormat);
}

void test_unknown_llama_load_option_is_rejected() {
    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH); // existing macro/constant in this file
    config.backend_options["llama"] = Chorus::OptionMap{{"warp_factor", int64_t{9}}};
    auto err = engine.initialize(config);
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(*err == Chorus::ChorusError::UnsupportedOption);
}

void test_model_loading() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{
        {"context_size", int64_t{1024}},
        {"use_gpu", false},
    };

    std::cout << "  [INFO] Loading model: " << MODEL_PATH << std::endl;

    ASSERT_TRUE(!engine.initialize(config).has_value());
    ASSERT_TRUE(engine.is_initialized());

    engine.stop();
    ASSERT_TRUE(!engine.is_initialized());
}

void test_simple_generation() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{{"use_gpu", false}};

    std::atomic<bool> done{false};
    std::string full_response = "";

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest chorus_request;
    chorus_request.id = 1;
    chorus_request.prompt = "<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n";
    chorus_request.gen_config.common.max_tokens = 20;
    chorus_request.gen_config.common.temperature = 0.7f;

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
        return;
    }

    ASSERT_TRUE(full_response.length() > 0);
    std::cout << "  [INFO] Received " << full_response.length() << " characters.\n";
}

void test_concurrent_requests_complete_with_multiple_slots() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{
        {"use_gpu", false},
        {"num_slots", int64_t{2}},
    };

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
        request.gen_config.common.max_tokens = 10;
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

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{
        {"use_gpu", false},
        {"context_size", int64_t{1024}},
    };

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
    req.gen_config.common.max_tokens = 8;
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

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{{"use_gpu", false}};

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
    req.gen_config.common.max_tokens = 5;
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

void test_llama_declares_gguf_and_chorus_managed() {
    Chorus::LlamaEngine engine; // pre-init envelope
    auto caps = engine.capabilities();
    ASSERT_EQ(caps.backend_id, std::string("llama"));
    ASSERT_TRUE(caps.scheduling == Chorus::SchedulingAuthority::ChorusManaged);
    ASSERT_EQ(caps.model_formats.size(), (size_t)1);
    ASSERT_TRUE(caps.model_formats[0] == Chorus::ModelFormat::Gguf);
    ASSERT_TRUE(caps.constraint_formats.empty()); // honesty until #4
    ASSERT_EQ(caps.portable_generation_options.size(), (size_t)5);
}

void test_llama_loaded_model_info_populated() {
    SKIP_IF_MODEL_TESTS_DISABLED();
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.loaded_model_info().has_value()); // pre-init: empty
    ASSERT_TRUE(!engine.initialize(make_gguf_config(MODEL_PATH)).has_value());
    auto info = engine.loaded_model_info();
    ASSERT_TRUE(info.has_value());
    ASSERT_EQ(info->model_id, std::string("test-model"));
    ASSERT_TRUE(info->format == Chorus::ModelFormat::Gguf); // never Auto
    ASSERT_TRUE(!info->family.empty());
    ASSERT_TRUE(info->maximum_context.has_value() && *info->maximum_context > 0);
    ASSERT_TRUE(info->model_bytes.has_value() && *info->model_bytes > 0);
    engine.stop();
    ASSERT_TRUE(!engine.loaded_model_info().has_value()); // teardown clears it
}

void test_llama_rejects_unwired_controls_explicitly() {
    SKIP_IF_MODEL_TESTS_DISABLED();
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(make_gguf_config(MODEL_PATH)).has_value());
    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = "hi";
    req.gen_config.common.constraint = Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, "root ::= \"x\""};
    auto r = engine.validate_request(req);
    ASSERT_TRUE(r.has_value());
    ASSERT_TRUE(r->error == Chorus::ChorusError::UnsupportedFeature);

    req.gen_config.common.constraint.reset();
    req.gen_config.backend_options["llama"] = Chorus::OptionMap{{"mirostat", int64_t{2}}};
    auto r2 = engine.validate_request(req);
    ASSERT_TRUE(r2.has_value());
    ASSERT_TRUE(r2->error == Chorus::ChorusError::UnsupportedOption);

    // A known key with the wrong value type must also reject: Task 3 left
    // mistyped repeat_penalty failing open in resolve_sampling; the port
    // closes it here (known key, but int64 where a double is required).
    req.gen_config.backend_options["llama"] = Chorus::OptionMap{{"repeat_penalty", int64_t{2}}};
    auto r3 = engine.validate_request(req);
    ASSERT_TRUE(r3.has_value());
    ASSERT_TRUE(r3->error == Chorus::ChorusError::UnsupportedOption);
    engine.stop();
}

// Conformance: every option listed as supported provably does something.
// seed+temperature: same set seed twice => identical text. top_k: =1 forces
// greedy => deterministic without a seed. max_tokens: bounds output length.
// top_p: forwarded to the chain, but its behavioral distinctness alone is not
// stable enough to assert on a 270M model; top_k and top_p set-field
// forwarding is proven by the resolve_sampling override test in
// test_llama_scheduler.cpp instead.
void test_llama_conformance_seed_and_temperature() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{{"use_gpu", false}};

    auto run_once = [&](std::string& text) {
        std::mutex sig_mutex;
        std::condition_variable cv;
        bool done = false;

        // declared after the state its worker callbacks capture, so the engine
        // (and its worker thread) is destroyed first
        Chorus::LlamaEngine engine;
        ASSERT_TRUE(!engine.initialize(config).has_value());

        Chorus::ChorusRequest req;
        req.id = 1;
        req.prompt = "<start_of_turn>user\nTell me about dragons.<end_of_turn>\n<start_of_turn>model\n";
        req.gen_config.common.seed = 42;
        req.gen_config.common.temperature = 0.9f;
        req.gen_config.common.max_tokens = 24;
        req.on_event = [&](const Chorus::ChorusSignal& sig) {
            std::lock_guard<std::mutex> lock(sig_mutex);
            if (sig.type == Chorus::EventType::Token) {
                text += sig.text;
            } else if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error) {
                done = true;
                cv.notify_one();
            }
        };
        engine.submit_request(req);

        {
            std::unique_lock<std::mutex> lock(sig_mutex);
            cv.wait_for(lock, std::chrono::seconds(15), [&] { return done; });
        }
        engine.stop();
    };

    std::string text_a;
    std::string text_b;
    run_once(text_a);
    run_once(text_b);
    ASSERT_TRUE(!text_a.empty());
    ASSERT_EQ(text_a, text_b);
}

void test_llama_conformance_max_tokens_bounds_output() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{{"use_gpu", false}};

    std::mutex sig_mutex;
    std::condition_variable cv;
    bool done = false;
    int token_count = 0;

    // declared after the state its worker callbacks capture, so the engine
    // (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = "<start_of_turn>user\nTell me a long story.<end_of_turn>\n<start_of_turn>model\n";
    req.gen_config.common.max_tokens = 8;
    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(sig_mutex);
        if (sig.type == Chorus::EventType::Token) {
            token_count++;
        } else if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error) {
            done = true;
            cv.notify_one();
        }
    };
    engine.submit_request(req);

    {
        std::unique_lock<std::mutex> lock(sig_mutex);
        cv.wait_for(lock, std::chrono::seconds(15), [&] { return done; });
    }
    engine.stop();

    ASSERT_TRUE(done);
    ASSERT_TRUE(token_count <= 8);
}

int run_llama_integration_tests() {
    std::cout << "\n--- LLAMA INTEGRATION SUITE ---\n";

    run_test("Llama: unsupported model format is rejected", test_unsupported_model_format_is_rejected);
    run_test("Llama: unknown load option is rejected", test_unknown_llama_load_option_is_rejected);
    run_test("Llama_Model_Load", test_model_loading);
    run_test("Llama_Generation_Stream", test_simple_generation);
    run_test(
        "Llama_ConcurrentRequestsCompleteWithMultipleSlots", test_concurrent_requests_complete_with_multiple_slots
    );
    run_test("Max_tokens_counts_generated_not_prompt_tokens", test_max_tokens_counts_generated_not_prompt_tokens);
    run_test("Engine_reinitializes_and_generates_after_stop", test_engine_reinitializes_and_generates_after_stop);
    run_test("Llama_declares_gguf_capabilities_chorus_managed", test_llama_declares_gguf_and_chorus_managed);
    run_test("Llama loaded model info populated", test_llama_loaded_model_info_populated);
    run_test("Llama_rejects_unwired_controls_explicitly", test_llama_rejects_unwired_controls_explicitly);
    run_test("Llama_conformance_seed_and_temperature", test_llama_conformance_seed_and_temperature);
    run_test("Llama_conformance_max_tokens_bounds_output", test_llama_conformance_max_tokens_bounds_output);

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

#include "chorus/providers/llama/llama_engine.hpp"
#include "chorus/core/common.hpp"
#include "test_utils.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

static const char* kReasoningModelPath = "tests/models/Qwen3-0.6B-Q8_0.gguf";

static Chorus::ChorusConfig make_reasoning_config() {
    Chorus::ChorusConfig config;
    config.model.model_id = "qwen3-0.6b";
    config.model.format = Chorus::ModelFormat::Gguf;
    config.model.assets.push_back({Chorus::AssetRole::Weights, kReasoningModelPath, std::nullopt, std::nullopt});
    config.provider_options["llama"] = Chorus::OptionMap{{"context_size", int64_t{2048}}, {"num_slots", int64_t{1}}};
    return config;
}

void test_reasoning_model_splits_channels() {
    SKIP_IF_MODEL_TESTS_DISABLED();
    // Loud failure when the prerequisite model is absent (never a silent skip).
    if (FILE* f = std::fopen(kReasoningModelPath, "rb")) {
        std::fclose(f);
    } else {
        ASSERT_TRUE(!"Reasoning model missing: tests/models/Qwen3-0.6B-Q8_0.gguf (plan prerequisite)");
        return;
    }

    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(make_reasoning_config()).has_value());

    std::mutex mutex;
    std::string content, reasoning;
    std::atomic<bool> done{false};
    std::atomic<bool> stopped{false};
    Chorus::ChorusRequest request;
    request.id = 1001;
    request.messages = {{"user", "What is 2+2? Answer with just the number."}};
    request.gen_config.max_tokens = 512; // room for the think block
    request.on_event = [&](Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(mutex);
        if (sig.type == Chorus::EventType::Token) {
            (sig.channel == Chorus::TokenChannel::Reasoning ? reasoning : content) += sig.text;
        }
        if (sig.type == Chorus::EventType::Stop) {
            stopped = true;
            done = true;
        } else if (sig.type == Chorus::EventType::Error) {
            done = true;
        }
    };
    engine.submit_request(request);
    for (int i = 0; i < 1200 && !done; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine.stop();

    ASSERT_TRUE(done.load());
    ASSERT_TRUE(stopped.load());
    // Qwen3 thinks by default: reasoning must be non-empty and content clean.
    ASSERT_TRUE(!reasoning.empty());
    ASSERT_TRUE(!content.empty());
    ASSERT_TRUE(content.find("<think>") == std::string::npos);
    ASSERT_TRUE(content.find("</think>") == std::string::npos);
}

void test_thinking_disabled_yields_no_reasoning() {
    SKIP_IF_MODEL_TESTS_DISABLED();
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(make_reasoning_config()).has_value());

    std::mutex mutex;
    std::string content, reasoning;
    std::atomic<bool> done{false};
    std::atomic<bool> stopped{false};
    Chorus::ChorusRequest request;
    request.id = 1002;
    request.messages = {{"user", "Say hello."}};
    request.gen_config.thinking = false;
    request.gen_config.max_tokens = 64;
    request.on_event = [&](Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(mutex);
        if (sig.type == Chorus::EventType::Token)
            (sig.channel == Chorus::TokenChannel::Reasoning ? reasoning : content) += sig.text;
        if (sig.type == Chorus::EventType::Stop) {
            stopped = true;
            done = true;
        } else if (sig.type == Chorus::EventType::Error) {
            done = true;
        }
    };
    engine.submit_request(request);
    for (int i = 0; i < 600 && !done; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine.stop();
    ASSERT_TRUE(done.load());
    ASSERT_TRUE(stopped.load());
    ASSERT_TRUE(reasoning.empty());
    ASSERT_TRUE(!content.empty());
    // The template pre-closes the think block, so no tags may leak into
    // content even though reasoning-aware templates retain their parser.
    ASSERT_TRUE(content.find("<think>") == std::string::npos);
    ASSERT_TRUE(content.find("</think>") == std::string::npos);
}

int run_llama_reasoning_tests() {
    std::cout << "\n--- Llama Reasoning (Qwen3, model-gated) Tests ---" << std::endl;
    run_test("LlamaReasoning_channels_split", test_reasoning_model_splits_channels);
    run_test("LlamaReasoning_thinking_off_no_reasoning", test_thinking_disabled_yields_no_reasoning);
    return g_tests_failed > 0 ? 1 : 0;
}

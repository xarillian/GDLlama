#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_engine.hpp"
#include "gtest_utils.hpp"

class LlamaReasoningModelTest : public ChorusModelTest {};

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
    config.model.assets.push_back({Chorus::AssetRole::Weights, kReasoningModelPath});
    config.provider_options["llama"] =
        Chorus::ProviderOptionMap{{"context_size", int64_t{2048}}, {"max_concurrent_requests", int64_t{1}}};
    return config;
}

TEST_F(LlamaReasoningModelTest, LlamaReasoning_channels_split) {
    // Loud failure when the prerequisite model is absent (never a silent skip).
    if (FILE* f = std::fopen(kReasoningModelPath, "rb")) {
        std::fclose(f);
    } else {
        ASSERT_TRUE(!"Reasoning model missing: tests/models/Qwen3-0.6B-Q8_0.gguf (plan prerequisite)");
        return;
    }

    std::mutex mutex;
    std::string content, reasoning;
    std::atomic<bool> done{false};
    std::atomic<bool> stopped{false};
    std::atomic<int64_t> sampling_steps{0};
    Chorus::GenerationUsage usage;
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(make_reasoning_config(), {}, {}).has_value());
    engine.set_logits_observer([&](Chorus::RequestId id, std::span<const float>) {
        if (id == 1001)
            ++sampling_steps;
    });
    Chorus::ChorusRequest request;
    request.id = 1001;
    request.messages = {
        {Chorus::MessageRole::User, Chorus::MessageContent::text("What is 2+2? Answer with just the number.")}
    };
    request.gen_config.max_tokens = 512; // room for the think block
    request.on_event = [&](Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event)) {
            (std::get<Chorus::ChorusSignal::Token>(sig.event).channel == Chorus::TokenChannel::Reasoning ? reasoning
                                                                                                         : content) +=
                std::get<Chorus::ChorusSignal::Token>(sig.event).text;
        }
        if (sig.is_terminal()) {
            stopped = std::holds_alternative<Chorus::ChorusSignal::Completion>(sig.event);
            if (const auto* completion = std::get_if<Chorus::ChorusSignal::Completion>(&sig.event))
                usage = completion->usage;
            done = true;
        }
    };
    engine.submit_request(request);
    for (int i = 0; i < 1200 && !done; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine.shutdown();

    ASSERT_TRUE(done.load());
    ASSERT_TRUE(stopped.load());
    EXPECT_GT(sampling_steps.load(), 0);
    EXPECT_EQ(usage.generated_tokens, sampling_steps.load());
    // Qwen3 thinks by default: reasoning must be non-empty and content clean.
    ASSERT_TRUE(!reasoning.empty());
    ASSERT_TRUE(!content.empty());
    ASSERT_TRUE(content.find("<think>") == std::string::npos);
    ASSERT_TRUE(content.find("</think>") == std::string::npos);
}

TEST_F(LlamaReasoningModelTest, LlamaReasoning_show_thinking_off_no_reasoning) {
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(make_reasoning_config(), {}, {}).has_value());

    std::mutex mutex;
    std::string content, reasoning;
    std::atomic<bool> done{false};
    std::atomic<bool> stopped{false};
    Chorus::ChorusRequest request;
    request.id = 1002;
    request.messages = {{Chorus::MessageRole::User, Chorus::MessageContent::text("Say hello.")}};
    request.gen_config.show_thinking = false;
    request.gen_config.max_tokens = 64;
    request.on_event = [&](Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event))
            (std::get<Chorus::ChorusSignal::Token>(sig.event).channel == Chorus::TokenChannel::Reasoning ? reasoning
                                                                                                         : content) +=
                std::get<Chorus::ChorusSignal::Token>(sig.event).text;
        if (sig.is_terminal()) {
            stopped = std::holds_alternative<Chorus::ChorusSignal::Completion>(sig.event);
            done = true;
        }
    };
    engine.submit_request(request);
    for (int i = 0; i < 600 && !done; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine.shutdown();
    ASSERT_TRUE(done.load());
    ASSERT_TRUE(stopped.load());
    ASSERT_TRUE(reasoning.empty());
    ASSERT_TRUE(!content.empty());
    // The template pre-closes the think block, so no tags may leak into
    // content even though reasoning-aware templates retain their parser.
    ASSERT_TRUE(content.find("<think>") == std::string::npos);
    ASSERT_TRUE(content.find("</think>") == std::string::npos);
}

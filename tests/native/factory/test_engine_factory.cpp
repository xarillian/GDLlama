#include "chorus/core/common.hpp"
#include "chorus/engine_factory.hpp"
#include "gtest_utils.hpp"

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Model-free: exercises the factory seam, not llama.cpp.

TEST(EngineFactory, Factory_echo_engine_round_trips_through_interface) {
    std::mutex sig_mutex;
    std::vector<Chorus::ChorusSignal> sigs;

    std::unique_ptr<Chorus::InferenceEngine> engine = Chorus::make_engine(Chorus::Provider::Echo);
    ASSERT_TRUE(engine != nullptr);
    ASSERT_TRUE(!engine->is_initialized());

    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine->initialize(config, {}).has_value());

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
            if (!sigs.empty() && std::holds_alternative<Chorus::ChorusSignal::Stop>(sigs.back().event))
                break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        timeout_ms -= 10;
    }

    std::lock_guard<std::mutex> lock(sig_mutex);
    ASSERT_EQ(sigs.size(), 3); // "seam " + "proof" + Stop
    std::string reassembled = std::get<Chorus::ChorusSignal::Token>(sigs[0].event).text + std::get<Chorus::ChorusSignal::Token>(sigs[1].event).text;
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Token>(sigs[0].event));
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Token>(sigs[1].event));
    ASSERT_TRUE(reassembled == "seam proof");
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(sigs[2].event));

    engine->shutdown();
}

TEST(EngineFactory, Factory_creates_llama_engine_uninitialized) {
    // No model load here: this only proves the factory constructs the real provider.
    std::unique_ptr<Chorus::InferenceEngine> engine = Chorus::make_engine(Chorus::Provider::Llama);
    ASSERT_TRUE(engine != nullptr);
    ASSERT_TRUE(!engine->is_initialized());
}

TEST(EngineFactory, Factory_shapes_initial_models_by_provider_policy) {
    const auto llama = Chorus::make_initial_model_spec(Chorus::Provider::Llama, "hero-model", "models/hero.gguf");
    ASSERT_EQ(llama.model_id, std::string("hero-model"));
    ASSERT_TRUE(llama.format == Chorus::ModelFormat::Gguf);
    ASSERT_EQ(llama.assets.size(), size_t{1});
    ASSERT_TRUE(llama.assets[0].role == Chorus::AssetRole::Weights);
    ASSERT_EQ(llama.assets[0].source, std::string("models/hero.gguf"));

    const auto echo = Chorus::make_initial_model_spec(Chorus::Provider::Echo, "ignored", "ignored.gguf");
    ASSERT_TRUE(echo.model_id.empty());
    ASSERT_TRUE(echo.format == Chorus::ModelFormat::Auto);
    ASSERT_TRUE(echo.assets.empty());
}

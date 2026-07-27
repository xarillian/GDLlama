#include "chorus/backends/llama/llama_load_config.hpp"
#include "godot_chorus/llama_load_options.hpp"

#include "test_utils.hpp"

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace {

std::variant<Chorus::LlamaLoadConfig, Chorus::RequestRejection> parse_godot_llama_options(Chorus::OptionMap options) {
    Chorus::ChorusConfig config;
    config.model.format = Chorus::ModelFormat::Gguf;
    config.model.assets.push_back({"weights", "model.gguf", std::nullopt, std::nullopt});
    config.backend_options["llama"] = std::move(options);
    return Chorus::parse_llama_load_config(config);
}

void test_godot_llama_load_options_use_cpu_without_gpu_controls() {
    Chorus::GodotAdapter::LlamaLoadOptions properties;
    properties.use_gpu = false;
    properties.gpu_layers = 17;
    properties.main_gpu = 2;

    const auto options = Chorus::GodotAdapter::make_llama_load_options(properties);

    ASSERT_TRUE(std::get<bool>(options.at("use_gpu")) == false);
    ASSERT_TRUE(options.find("gpu_layers") == options.end());
    ASSERT_TRUE(options.find("main_gpu") == options.end());

    const auto parsed = parse_godot_llama_options(options);
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaLoadConfig>(parsed));
    const auto& load = std::get<Chorus::LlamaLoadConfig>(parsed);
    ASSERT_TRUE(!load.use_gpu);
    ASSERT_TRUE(!load.gpu_layers_explicit);
    ASSERT_TRUE(!load.main_gpu_explicit);
}

void test_godot_llama_load_options_forward_gpu_controls() {
    Chorus::GodotAdapter::LlamaLoadOptions properties;
    properties.use_gpu = true;
    properties.gpu_layers = -1;
    properties.main_gpu = 2;

    const auto options = Chorus::GodotAdapter::make_llama_load_options(properties);

    ASSERT_EQ(std::get<int64_t>(options.at("gpu_layers")), int64_t{-1});
    ASSERT_EQ(std::get<int64_t>(options.at("main_gpu")), int64_t{2});

    const auto parsed = parse_godot_llama_options(options);
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaLoadConfig>(parsed));
    const auto& load = std::get<Chorus::LlamaLoadConfig>(parsed);
    ASSERT_TRUE(load.use_gpu);
    ASSERT_TRUE(load.gpu_layers_explicit);
    ASSERT_TRUE(load.main_gpu_explicit);
    ASSERT_EQ(load.gpu_layers, int32_t{-1});
    ASSERT_EQ(load.main_gpu, int32_t{2});
}

void test_godot_llama_load_options_expose_all_layers_sentinel() {
    ASSERT_EQ(Chorus::GodotAdapter::DEFAULT_GPU_LAYERS, int32_t{-1});
    ASSERT_EQ(std::string(Chorus::GodotAdapter::GPU_LAYERS_PROPERTY_HINT), std::string("-1,999,1"));
}

} // namespace

int run_godot_llama_load_options_tests() {
    std::cout << "\n--- Godot Llama Load Option Tests ---\n";
    run_test(
        "Godot llama load options use CPU without GPU controls",
        test_godot_llama_load_options_use_cpu_without_gpu_controls
    );
    run_test("Godot llama load options forward GPU controls", test_godot_llama_load_options_forward_gpu_controls);
    run_test(
        "Godot llama load options expose all-layers sentinel", test_godot_llama_load_options_expose_all_layers_sentinel
    );
    return 0;
}

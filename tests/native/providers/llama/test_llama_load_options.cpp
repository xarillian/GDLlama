#include "chorus/providers/llama/llama_generation_options.hpp"
#include "chorus/providers/llama/llama_load_config.hpp"

#include "gtest_utils.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

std::variant<Chorus::LlamaLoadConfig, Chorus::RequestRejection> parse(Chorus::ProviderOptionMap options) {
    Chorus::ChorusConfig config;
    config.model.format = Chorus::ModelFormat::Gguf;
    config.model.assets.push_back({Chorus::AssetRole::Weights, "model.gguf"});
    config.provider_options["llama"] = std::move(options);
    return Chorus::parse_llama_load_config(config);
}

// One option set to `value`, every other at its declared default -- the shape a
// host produces when the user touches a single control.
std::variant<Chorus::LlamaLoadConfig, Chorus::RequestRejection>
parse_with(const std::string& key, Chorus::ProviderOptionValue value) {
    const auto& descriptors = Chorus::llama_load_option_descriptors();
    Chorus::ProviderOptionMap stored{{key, std::move(value)}};
    return parse(Chorus::resolve_option_defaults(descriptors, stored));
}

struct MappingCase {
    std::string key;
    Chorus::ProviderOptionValue value;
    std::function<void(const Chorus::LlamaLoadConfig&)> verify;
};

struct InvalidCase {
    std::string key;
    Chorus::ProviderOptionValue value;
};

TEST(LlamaLoadOptions, Llama_load_options_map_to_distinct_config_fields) {
    const auto defaults = parse(Chorus::resolve_option_defaults(Chorus::llama_load_option_descriptors(), {}));
    const auto* parsed_defaults = std::get_if<Chorus::LlamaLoadConfig>(&defaults);
    ASSERT_TRUE(parsed_defaults != nullptr);
    if (!parsed_defaults)
        return;

    const Chorus::LlamaLoadConfig expected_defaults{};
    ASSERT_EQ(parsed_defaults->context_size, expected_defaults.context_size);
    ASSERT_EQ(parsed_defaults->thread_count, expected_defaults.thread_count);
    ASSERT_EQ(parsed_defaults->use_gpu, expected_defaults.use_gpu);
    ASSERT_EQ(parsed_defaults->gpu_layers, expected_defaults.gpu_layers);
    ASSERT_EQ(parsed_defaults->num_slots, expected_defaults.num_slots);
    ASSERT_EQ(parsed_defaults->tokens_per_tick, expected_defaults.tokens_per_tick);
    ASSERT_EQ(parsed_defaults->n_batch, expected_defaults.n_batch);
    ASSERT_EQ(parsed_defaults->n_ubatch, expected_defaults.n_ubatch);
    ASSERT_EQ(parsed_defaults->main_gpu, expected_defaults.main_gpu);
    ASSERT_EQ(parsed_defaults->pooling, LLAMA_POOLING_TYPE_UNSPECIFIED);

    const std::vector<MappingCase> cases{
        {"context_size", int64_t{4096}, [](const auto& load) { ASSERT_EQ(load.context_size, uint32_t{4096}); }},
        {"thread_count", int64_t{6}, [](const auto& load) { ASSERT_EQ(load.thread_count, int32_t{6}); }},
        {"use_gpu", false, [](const auto& load) { ASSERT_TRUE(!load.use_gpu); }},
        {"gpu_layers", int64_t{17}, [](const auto& load) {
             ASSERT_EQ(load.gpu_layers, int32_t{17});
             ASSERT_TRUE(load.gpu_layers_explicit);
         }},
        {"num_slots", int64_t{3}, [](const auto& load) { ASSERT_EQ(load.num_slots, uint32_t{3}); }},
        {"tokens_per_tick", int64_t{96}, [](const auto& load) { ASSERT_EQ(load.tokens_per_tick, int32_t{96}); }},
        {"n_batch", int64_t{1024}, [](const auto& load) { ASSERT_EQ(load.n_batch, uint32_t{1024}); }},
        {"n_ubatch", int64_t{32}, [](const auto& load) { ASSERT_EQ(load.n_ubatch, uint32_t{32}); }},
        {"main_gpu", int64_t{2}, [](const auto& load) {
             ASSERT_EQ(load.main_gpu, int32_t{2});
             ASSERT_TRUE(load.main_gpu_explicit);
         }},
        {"pooling", std::string{"none"}, [](const auto& load) { ASSERT_EQ(load.pooling, LLAMA_POOLING_TYPE_NONE); }},
    };

    for (const auto& mapping : cases) {
        const auto parsed = parse_with(mapping.key, mapping.value);
        const auto* load = std::get_if<Chorus::LlamaLoadConfig>(&parsed);
        ASSERT_TRUE(load != nullptr) << "failed to map " << mapping.key;
        if (load)
            mapping.verify(*load);
    }
}

// The contract's half of a widget bound: whatever a host lets a user pick
// inside the declared range, the provider accepts. A hint tightened past the
// provider's own limits, or a limit tightened past the hint, fails here rather
// than in a game at load time.
TEST(LlamaLoadOptions, Llama_descriptor_bounds_are_accepted) {
    for (const auto& descriptor : Chorus::llama_load_option_descriptors()) {
        if (!descriptor.minimum || !descriptor.maximum)
            continue;
        for (const double bound : {*descriptor.minimum, *descriptor.maximum}) {
            const auto parsed = parse_with(descriptor.key, int64_t(bound));
            if (!std::holds_alternative<Chorus::LlamaLoadConfig>(parsed)) {
                std::cout << "    rejected " << descriptor.key << " = " << bound << ": "
                          << std::get<Chorus::RequestRejection>(parsed).message << "\n";
                ASSERT_TRUE(false);
            }
        }
    }
}

TEST(LlamaLoadOptions, Llama_load_options_reject_invalid_values_with_key_context) {
    const int64_t above_int32 = int64_t{std::numeric_limits<int32_t>::max()} + 1;
    const int64_t above_uint32 = int64_t{std::numeric_limits<uint32_t>::max()} + 1;
    const std::vector<InvalidCase> cases{
        {"not_a_real_option", int64_t{1}},
        {"warp_factor", int64_t{9}},
        {"context_size", std::string("large")},
        {"n_batch", true},
        {"n_ubatch", 32.0},
        {"main_gpu", std::string("0")},
        {"pooling", std::string("rank")},
        {"pooling", std::string("unknown")},
        {"pooling", int64_t{1}},
        {"context_size", int64_t{0}},
        {"thread_count", int64_t{0}},
        {"gpu_layers", int64_t{-2}},
        {"num_slots", int64_t{0}},
        {"tokens_per_tick", int64_t{0}},
        {"n_batch", int64_t{0}},
        {"n_batch", int64_t{-1}},
        {"n_ubatch", int64_t{0}},
        {"n_ubatch", int64_t{-1}},
        {"main_gpu", int64_t{-1}},
        {"context_size", above_uint32},
        {"thread_count", above_int32},
        {"gpu_layers", above_int32},
        {"num_slots", above_uint32},
        {"tokens_per_tick", above_int32},
        {"n_batch", above_int32},
        {"n_ubatch", above_uint32},
        {"main_gpu", above_int32},
    };

    for (const auto& invalid : cases) {
        const auto parsed = parse({{invalid.key, invalid.value}});
        const auto* rejection = std::get_if<Chorus::RequestRejection>(&parsed);
        ASSERT_TRUE(rejection != nullptr) << "accepted invalid " << invalid.key;
        if (!rejection)
            continue;
        ASSERT_EQ(rejection->error, Chorus::ChorusError::UnsupportedOption);
        ASSERT_TRUE(rejection->message.find(invalid.key) != std::string::npos)
            << "rejection lost key context for " << invalid.key;
    }
}

TEST(LlamaLoadOptions, Llama_pooling_descriptor_lists_only_embedding_modes) {
    const auto* descriptor = Chorus::find_option_descriptor(Chorus::llama_load_option_descriptors(), "pooling");
    ASSERT_TRUE(descriptor != nullptr);
    ASSERT_EQ(descriptor->choices, std::vector<std::string>({"model", "none", "mean", "cls", "last"}));
    for (const auto& [value, expected] : std::vector<std::pair<std::string, enum llama_pooling_type>>{
             {"model", LLAMA_POOLING_TYPE_UNSPECIFIED},
             {"none", LLAMA_POOLING_TYPE_NONE},
             {"mean", LLAMA_POOLING_TYPE_MEAN},
             {"cls", LLAMA_POOLING_TYPE_CLS},
             {"last", LLAMA_POOLING_TYPE_LAST},
         }) {
        const auto parsed = parse_with("pooling", value);
        ASSERT_TRUE(std::holds_alternative<Chorus::LlamaLoadConfig>(parsed));
        ASSERT_EQ(std::get<Chorus::LlamaLoadConfig>(parsed).pooling, expected);
    }
}

TEST(LlamaLoadOptions, Llama_n_ubatch_must_not_exceed_n_batch) {
    const auto parsed = parse({{"n_batch", int64_t{32}}, {"n_ubatch", int64_t{33}}});
    const auto* rejection = std::get_if<Chorus::RequestRejection>(&parsed);
    ASSERT_TRUE(rejection != nullptr);
    if (rejection) {
        ASSERT_EQ(rejection->error, Chorus::ChorusError::UnsupportedOption);
        ASSERT_TRUE(rejection->message.find("n_ubatch") != std::string::npos);
    }
}

TEST(LlamaLoadOptions, Llama_CPU_rejects_explicit_GPU_controls) {
    for (const auto& [key, value] : std::vector<std::pair<std::string, int64_t>>{
             {"main_gpu", 0},
             {"gpu_layers", 17},
             {"gpu_layers", -1},
         }) {
        const auto parsed = parse({{"use_gpu", false}, {key, value}});
        const auto* rejection = std::get_if<Chorus::RequestRejection>(&parsed);
        ASSERT_TRUE(rejection != nullptr);
        if (rejection) {
            ASSERT_EQ(rejection->error, Chorus::ChorusError::UnsupportedOption);
            ASSERT_TRUE(rejection->message.find(key) != std::string::npos);
        }
    }
}

TEST(LlamaLoadOptions, Llama_CPU_accepts_explicit_zero_GPU_layers) {
    const auto parsed = parse({{"use_gpu", false}, {"gpu_layers", int64_t{0}}});
    const auto* load = std::get_if<Chorus::LlamaLoadConfig>(&parsed);
    ASSERT_TRUE(load != nullptr);
    if (load) {
        ASSERT_TRUE(!load->use_gpu);
        ASSERT_TRUE(load->gpu_layers_explicit);
        ASSERT_TRUE(!load->main_gpu_explicit);
        ASSERT_EQ(load->gpu_layers, int32_t{0});
    }
}

TEST(LlamaLoadOptions, Llama_GPU_prerequisites_are_dropped_when_CPU_is_selected) {
    const auto resolved =
        Chorus::resolve_option_defaults(Chorus::llama_load_option_descriptors(), {{"use_gpu", false}});
    ASSERT_TRUE(resolved.find("main_gpu") == resolved.end());
    ASSERT_TRUE(resolved.find("gpu_layers") == resolved.end());
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaLoadConfig>(parse(resolved)));
}

} // namespace

#include "chorus/core/generation_config.hpp"
#include "gtest_utils.hpp"

TEST(GenerationConfig, Generation_patch_overrides_and_clears) {
    Chorus::GenerationConfig base;
    base.max_tokens = 128;
    base.temperature = 0.8f;
    base.stop = {"<END>"};
    base.seed = 7;
    base.frequency_penalty = 0.1f;
    base.presence_penalty = 0.2f;
    base.show_thinking = true;

    Chorus::GenerationConfigPatch patch;
    patch.max_tokens = Chorus::ConfigPatch<int32_t>::clear();
    patch.temperature = Chorus::ConfigPatch<float>::set(0.2f);
    patch.stop = Chorus::ConfigPatch<std::vector<std::string>>::set({});
    patch.seed = Chorus::ConfigPatch<uint64_t>::set(11);
    patch.frequency_penalty = Chorus::ConfigPatch<float>::set(0.4f);
    patch.presence_penalty = Chorus::ConfigPatch<float>::set(-0.2f);
    patch.show_thinking = Chorus::ConfigPatch<bool>::set(false);

    const auto merged = Chorus::apply_generation_patch(base, patch);
    ASSERT_TRUE(!merged.max_tokens.has_value());
    ASSERT_TRUE(merged.temperature.has_value());
    ASSERT_TRUE(*merged.temperature == 0.2f);
    ASSERT_TRUE(merged.stop.empty());
    ASSERT_TRUE(merged.seed.has_value());
    ASSERT_EQ(*merged.seed, uint64_t{11});
    ASSERT_TRUE(merged.frequency_penalty.has_value());
    ASSERT_TRUE(*merged.frequency_penalty == 0.4f);
    ASSERT_TRUE(merged.presence_penalty.has_value());
    ASSERT_TRUE(*merged.presence_penalty == -0.2f);
    ASSERT_TRUE(merged.show_thinking.has_value());
    ASSERT_TRUE(!*merged.show_thinking);

    Chorus::GenerationConfigPatch clear_patch;
    clear_patch.show_thinking = Chorus::ConfigPatch<bool>::clear();
    const auto cleared = Chorus::apply_generation_patch(merged, clear_patch);
    ASSERT_TRUE(!cleared.show_thinking.has_value());
}

TEST(GenerationConfig, Generation_patch_deep_merges_namespaces) {
    Chorus::GenerationConfig base;
    base.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"repeat_penalty", 1.1}, {"dry", Chorus::ProviderOptionMap{{"base", 1.75}, {"length", int64_t{2}}}}
    };
    Chorus::GenerationConfigPatch patch;
    patch.provider_options = Chorus::ProviderOptionMap{
        {"llama", Chorus::ProviderOptionMap{{"dry", Chorus::ProviderOptionMap{{"length", int64_t{4}}}}}}
    };
    const auto merged = Chorus::apply_generation_patch(base, patch);
    const auto& llama = std::get<Chorus::ProviderOptionMap>(merged.provider_options.at("llama"));
    const auto& dry = std::get<Chorus::ProviderOptionMap>(llama.at("dry"));
    ASSERT_TRUE(std::get<double>(llama.at("repeat_penalty")) == 1.1);
    ASSERT_TRUE(std::get<double>(dry.at("base")) == 1.75);
    ASSERT_EQ(std::get<int64_t>(dry.at("length")), 4);
}

TEST(GenerationConfig, Generation_patch_erases_inherited_options) {
    Chorus::GenerationConfig base;
    base.provider_options = Chorus::ProviderOptionMap{
        {"llama",
         Chorus::ProviderOptionMap{
             {"repeat_penalty", 1.1},
             {"top_n_sigma", 2.0},
             {"dry", Chorus::ProviderOptionMap{{"base", 1.75}, {"length", int64_t{2}}}},
         }}
    };

    Chorus::GenerationConfigPatch patch;
    patch.provider_option_erasures = {"llama.repeat_penalty", "llama.dry.base"};

    const auto merged = Chorus::apply_generation_patch(base, patch);
    const auto& llama = std::get<Chorus::ProviderOptionMap>(merged.provider_options.at("llama"));
    ASSERT_TRUE(llama.find("repeat_penalty") == llama.end());
    ASSERT_TRUE(std::get<double>(llama.at("top_n_sigma")) == 2.0); // siblings survive
    const auto& dry = std::get<Chorus::ProviderOptionMap>(llama.at("dry"));
    ASSERT_TRUE(dry.find("base") == dry.end());
    ASSERT_EQ(std::get<int64_t>(dry.at("length")), int64_t{2});
}

TEST(GenerationConfig, Generation_patch_erasure_runs_after_the_merge) {
    // Both spellings in one patch: the merge adds, the erasure removes. Order
    // is what makes an explicit null mean "drop it" rather than "drop it
    // unless something else in this same patch set it".
    Chorus::GenerationConfig base;
    Chorus::GenerationConfigPatch patch;
    patch.provider_options = Chorus::ProviderOptionMap{{"llama", Chorus::ProviderOptionMap{{"repeat_penalty", 1.3}}}};
    patch.provider_option_erasures = {"llama.repeat_penalty"};

    const auto merged = Chorus::apply_generation_patch(base, patch);
    const auto& llama = std::get<Chorus::ProviderOptionMap>(merged.provider_options.at("llama"));
    ASSERT_TRUE(llama.find("repeat_penalty") == llama.end());
}

TEST(GenerationConfig, Erase_option_path_tolerates_bad_paths) {
    Chorus::ProviderOptionMap options{
        {"llama", Chorus::ProviderOptionMap{{"repeat_penalty", 1.1}}},
        {"scalar", int64_t{3}},
    };

    Chorus::erase_option_path(options, "llama.not_here"); // missing leaf
    Chorus::erase_option_path(options, "nowhere.at.all"); // missing namespace
    Chorus::erase_option_path(options, "scalar.deeper");  // scalar where a map is expected
    ASSERT_TRUE(std::get<double>(std::get<Chorus::ProviderOptionMap>(options.at("llama")).at("repeat_penalty")) == 1.1);
    ASSERT_EQ(std::get<int64_t>(options.at("scalar")), int64_t{3});

    Chorus::erase_option_path(options, "scalar"); // a top-level leaf
    ASSERT_TRUE(options.find("scalar") == options.end());
}

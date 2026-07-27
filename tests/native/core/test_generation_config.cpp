#include "chorus/core/generation_config.hpp"
#include "test_utils.hpp"

void test_generation_patch_overrides_and_clears_portable_fields() {
    Chorus::GenerationConfig base;
    base.common.max_tokens = 128;
    base.common.temperature = 0.8f;
    base.common.stop = {"<END>"};

    Chorus::GenerationConfigPatch patch;
    patch.max_tokens = Chorus::OptionalPatch<int32_t>::clear();
    patch.temperature = Chorus::OptionalPatch<float>::set(0.2f);
    patch.stop = Chorus::ValuePatch<std::vector<std::string>>::set({});

    const auto merged = Chorus::apply_generation_patch(base, patch);
    ASSERT_TRUE(!merged.common.max_tokens.has_value());
    ASSERT_TRUE(merged.common.temperature.has_value());
    ASSERT_TRUE(*merged.common.temperature == 0.2f);
    ASSERT_TRUE(merged.common.stop.empty());
}

void test_generation_patch_deep_merges_backend_namespaces() {
    Chorus::GenerationConfig base;
    base.backend_options["llama"] =
        Chorus::OptionMap{{"repeat_penalty", 1.1}, {"dry", Chorus::OptionMap{{"base", 1.75}, {"length", int64_t{2}}}}};
    Chorus::GenerationConfigPatch patch;
    patch.backend_options =
        Chorus::OptionMap{{"llama", Chorus::OptionMap{{"dry", Chorus::OptionMap{{"length", int64_t{4}}}}}}};
    const auto merged = Chorus::apply_generation_patch(base, patch);
    const auto& llama = std::get<Chorus::OptionMap>(merged.backend_options.at("llama"));
    const auto& dry = std::get<Chorus::OptionMap>(llama.at("dry"));
    ASSERT_TRUE(std::get<double>(llama.at("repeat_penalty")) == 1.1);
    ASSERT_TRUE(std::get<double>(dry.at("base")) == 1.75);
    ASSERT_EQ(std::get<int64_t>(dry.at("length")), 4);
}

// Pins the three-layer composition the Godot request normalizer relies on:
// engine/backend defaults, then a shared ChorusGenerationDefaults-style
// overlay, then a per-request overlay, applied as two sequential
// apply_generation_patch calls. Covers backend defaults, shared defaults,
// request override, a scalar clear, an empty-stop replacement, and a
// recursive backend namespace merge in one pass.
void test_three_layer_patch_composition() {
    // Layer 1: engine/backend defaults -- nothing set.
    Chorus::GenerationConfig backend_defaults;

    // Layer 2: shared resource defaults.
    Chorus::GenerationConfigPatch shared_defaults;
    shared_defaults.max_tokens = Chorus::OptionalPatch<int32_t>::set(128);
    shared_defaults.temperature = Chorus::OptionalPatch<float>::set(0.7f);
    shared_defaults.stop = Chorus::ValuePatch<std::vector<std::string>>::set({"<END>", "<STOP>"});
    shared_defaults.backend_options = Chorus::OptionMap{
        {"llama",
         Chorus::OptionMap{{"repeat_penalty", 1.1}, {"dry", Chorus::OptionMap{{"base", 1.75}, {"length", int64_t{2}}}}}}
    };

    // Layer 3: per-request overlay.
    Chorus::GenerationConfigPatch request_override;
    request_override.max_tokens = Chorus::OptionalPatch<int32_t>::set(256);        // request wins over shared
    request_override.temperature = Chorus::OptionalPatch<float>::clear();          // scalar clear
    request_override.stop = Chorus::ValuePatch<std::vector<std::string>>::set({}); // empty-stop replacement
    request_override.backend_options = Chorus::OptionMap{
        {"llama", Chorus::OptionMap{{"dry", Chorus::OptionMap{{"length", int64_t{4}}}}, {"n_batch", int64_t{1024}}}}
    };

    const auto after_shared = Chorus::apply_generation_patch(backend_defaults, shared_defaults);
    const auto merged = Chorus::apply_generation_patch(after_shared, request_override);

    ASSERT_TRUE(merged.common.max_tokens.has_value());
    ASSERT_EQ(*merged.common.max_tokens, 256);

    ASSERT_TRUE(!merged.common.temperature.has_value());

    ASSERT_TRUE(merged.common.stop.empty());

    const auto& llama = std::get<Chorus::OptionMap>(merged.backend_options.at("llama"));
    ASSERT_TRUE(std::get<double>(llama.at("repeat_penalty")) == 1.1);
    const auto& dry = std::get<Chorus::OptionMap>(llama.at("dry"));
    ASSERT_TRUE(std::get<double>(dry.at("base")) == 1.75);
    ASSERT_EQ(std::get<int64_t>(dry.at("length")), 4);
    ASSERT_EQ(std::get<int64_t>(llama.at("n_batch")), 1024);
}

int run_generation_config_tests() {
    run_test("Generation patch overrides and clears", test_generation_patch_overrides_and_clears_portable_fields);
    run_test("Generation patch deep merges namespaces", test_generation_patch_deep_merges_backend_namespaces);
    run_test("Three-layer patch composition", test_three_layer_patch_composition);
    return g_tests_failed;
}

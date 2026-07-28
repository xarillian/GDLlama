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

void test_generation_patch_erases_inherited_backend_options() {
    Chorus::GenerationConfig base;
    base.backend_options = Chorus::OptionMap{
        {"llama",
         Chorus::OptionMap{
             {"repeat_penalty", 1.1},
             {"top_n_sigma", 2.0},
             {"dry", Chorus::OptionMap{{"base", 1.75}, {"length", int64_t{2}}}},
         }}
    };

    Chorus::GenerationConfigPatch patch;
    patch.backend_option_erasures = {"llama.repeat_penalty", "llama.dry.base"};

    const auto merged = Chorus::apply_generation_patch(base, patch);
    const auto& llama = std::get<Chorus::OptionMap>(merged.backend_options.at("llama"));
    ASSERT_TRUE(llama.find("repeat_penalty") == llama.end());
    ASSERT_TRUE(std::get<double>(llama.at("top_n_sigma")) == 2.0); // siblings survive
    const auto& dry = std::get<Chorus::OptionMap>(llama.at("dry"));
    ASSERT_TRUE(dry.find("base") == dry.end());
    ASSERT_EQ(std::get<int64_t>(dry.at("length")), int64_t{2});
}

void test_generation_patch_erasure_runs_after_the_merge() {
    // Both spellings in one patch: the merge adds, the erasure removes. Order
    // is what makes an explicit null mean "drop it" rather than "drop it
    // unless something else in this same patch set it".
    Chorus::GenerationConfig base;
    Chorus::GenerationConfigPatch patch;
    patch.backend_options = Chorus::OptionMap{{"llama", Chorus::OptionMap{{"repeat_penalty", 1.3}}}};
    patch.backend_option_erasures = {"llama.repeat_penalty"};

    const auto merged = Chorus::apply_generation_patch(base, patch);
    const auto& llama = std::get<Chorus::OptionMap>(merged.backend_options.at("llama"));
    ASSERT_TRUE(llama.find("repeat_penalty") == llama.end());
}

void test_erase_option_path_tolerates_missing_and_malformed_paths() {
    Chorus::OptionMap options{
        {"llama", Chorus::OptionMap{{"repeat_penalty", 1.1}}},
        {"scalar", int64_t{3}},
    };

    Chorus::erase_option_path(options, "llama.not_here"); // missing leaf
    Chorus::erase_option_path(options, "nowhere.at.all"); // missing namespace
    Chorus::erase_option_path(options, "scalar.deeper");  // scalar where a map is expected
    ASSERT_TRUE(std::get<double>(std::get<Chorus::OptionMap>(options.at("llama")).at("repeat_penalty")) == 1.1);
    ASSERT_EQ(std::get<int64_t>(options.at("scalar")), int64_t{3});

    Chorus::erase_option_path(options, "scalar"); // a top-level leaf
    ASSERT_TRUE(options.find("scalar") == options.end());
}

int run_generation_config_tests() {
    run_test("Generation patch overrides and clears", test_generation_patch_overrides_and_clears_portable_fields);
    run_test("Generation patch deep merges namespaces", test_generation_patch_deep_merges_backend_namespaces);
    run_test("Three-layer patch composition", test_three_layer_patch_composition);
    run_test("Generation patch erases inherited options", test_generation_patch_erases_inherited_backend_options);
    run_test("Generation patch erasure runs after the merge", test_generation_patch_erasure_runs_after_the_merge);
    run_test("Erase option path tolerates bad paths", test_erase_option_path_tolerates_missing_and_malformed_paths);
    return g_tests_failed;
}

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"
#include "chorus/core/model_spec.hpp"
#include "test_utils.hpp"

// No model, no skips: this suite must pass under CHORUS_SKIP_MODEL_TESTS=1.

void test_engine_capabilities_defaults_are_conservative() {
    Chorus::EngineCapabilities caps;
    // BackendManaged is the deliberate conservative default: an engine that
    // forgets to set it must not claim Chorus-managed frame guarantees.
    ASSERT_TRUE(caps.scheduling == Chorus::SchedulingAuthority::BackendManaged);
    ASSERT_TRUE(!caps.streaming);
    ASSERT_TRUE(!caps.cancellation);
    ASSERT_TRUE(!caps.native_sessions);
    ASSERT_TRUE(!caps.embeddings);
    ASSERT_TRUE(!caps.speculative_decoding);
    ASSERT_TRUE(!caps.dynamic_adapters);
    ASSERT_TRUE(!caps.prompt_rendering);
}

void test_model_spec_defaults() {
    Chorus::ModelSpec spec;
    ASSERT_TRUE(spec.format == Chorus::ModelFormat::Auto);
    ASSERT_TRUE(spec.assets.empty());
    ASSERT_TRUE(spec.backend_options.empty());
}

void test_new_error_categories_exist() {
    // Compile-level lock: these categories are part of the 3c contract.
    Chorus::ChorusError errs[] = {
        Chorus::ChorusError::UnsupportedModelFormat,
        Chorus::ChorusError::UnsupportedFeature,
        Chorus::ChorusError::UnsupportedOption,
        Chorus::ChorusError::SessionBusy,
    };
    ASSERT_EQ(sizeof(errs) / sizeof(errs[0]), (size_t)4);
}

int run_contract_type_tests() {
    std::cout << "\n--- Contract Type Tests ---\n";
    run_test("EngineCapabilities: conservative defaults", test_engine_capabilities_defaults_are_conservative);
    run_test("ModelSpec: defaults", test_model_spec_defaults);
    run_test("ChorusError: 3c categories", test_new_error_categories_exist);
    return g_tests_failed;
}

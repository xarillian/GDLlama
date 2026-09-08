#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"
#include "chorus/core/model_spec.hpp"
#include "gtest_utils.hpp"

// No model, no skips: this suite must pass under CHORUS_SKIP_MODEL_TESTS=1.

TEST(ContractTypes, EngineCapabilities_conservative_defaults) {
    Chorus::EngineCapabilities caps;
    // ProviderManaged is the deliberate conservative default: an engine that
    // forgets to set it must not claim Chorus-managed frame guarantees.
    ASSERT_TRUE(caps.scheduling == Chorus::SchedulingAuthority::ProviderManaged);
    ASSERT_TRUE(!caps.streaming);
    ASSERT_TRUE(!caps.cancellation);
    ASSERT_TRUE(!caps.native_sessions);
    ASSERT_TRUE(!caps.embeddings);
    ASSERT_TRUE(!caps.speculative_decoding);
    ASSERT_TRUE(!caps.dynamic_adapters);
    ASSERT_TRUE(!caps.prompt_rendering);
}

TEST(ContractTypes, Typed_messages_validate_roles_and_join_text_parts) {
    ASSERT_EQ(Chorus::message_role_name(Chorus::MessageRole::System), "system");
    ASSERT_FALSE(Chorus::message_role_name(static_cast<Chorus::MessageRole>(99)).has_value());
    const Chorus::MessageContent content{{std::string("a"), std::string(""), std::string("b")}};
    ASSERT_EQ(Chorus::joined_text(content), "ab");
}

TEST(ContractTypes, InitialModelSpec_defaults) {
    Chorus::InitialModelSpec spec;
    ASSERT_TRUE(spec.format == Chorus::ModelFormat::Auto);
    ASSERT_TRUE(spec.assets.empty());
    ASSERT_TRUE(spec.provider_options.empty());
}


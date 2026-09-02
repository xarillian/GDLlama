#include "chorus/core/provider_option_value.hpp"
#include "gtest_utils.hpp"

#include <variant>

// No model, no skips: this suite must pass under CHORUS_SKIP_MODEL_TESTS=1.

TEST(ProviderOptionValue, Option_value_literal_is_string) {
    // Classic variant footgun: const char* prefers the bool conversion.
    Chorus::ProviderOptionValue v = "gemma";
    ASSERT_TRUE(std::holds_alternative<std::string>(v));
    ASSERT_EQ(std::get<std::string>(v), std::string("gemma"));
}


#include "chorus/core/generation_config.hpp"
#include "gtest_utils.hpp"

TEST(GenerationConfig, Default_choices_are_absent) {
    Chorus::GenerationConfig choices;
    ASSERT_FALSE(choices.max_tokens);
    ASSERT_FALSE(choices.temperature);
    ASSERT_FALSE(choices.top_k);
    ASSERT_FALSE(choices.top_p);
    ASSERT_FALSE(choices.seed);
    ASSERT_FALSE(choices.frequency_penalty);
    ASSERT_FALSE(choices.presence_penalty);
    ASSERT_FALSE(choices.stop);
    ASSERT_FALSE(choices.constraint);
    ASSERT_FALSE(choices.show_thinking);
    ASSERT_TRUE(choices.provider_options.empty());
}

#include "gtest_utils.hpp"

TEST(CoreMechanics, Model_tests_disabled_by_env_only_for_explicit_1) {
    ASSERT_TRUE(!chorus_model_tests_disabled_by_value(nullptr)); // unset  -> enabled
    ASSERT_TRUE(!chorus_model_tests_disabled_by_value("0"));     // "0"    -> enabled
    ASSERT_TRUE(!chorus_model_tests_disabled_by_value(""));      // empty  -> enabled
    ASSERT_TRUE(chorus_model_tests_disabled_by_value("1"));      // "1"    -> disabled
}


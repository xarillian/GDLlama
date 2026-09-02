#pragma once

#include <cstdlib>
#include <string_view>

#include <gtest/gtest.h>

inline bool chorus_model_tests_disabled_by_value(const char* value) {
    return value != nullptr && std::string_view(value) == "1";
}

inline bool chorus_model_tests_disabled() {
    return chorus_model_tests_disabled_by_value(std::getenv("CHORUS_SKIP_MODEL_TESTS"));
}

class ChorusModelTest : public ::testing::Test {
  protected:
    void skip_if_model_tests_disabled() {
        if (chorus_model_tests_disabled())
            GTEST_SKIP() << "model tests disabled (CHORUS_SKIP_MODEL_TESTS=1)";
    }

    void SetUp() override {
        skip_if_model_tests_disabled();
    }
};

class ChorusGpuModelTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (chorus_model_tests_disabled())
            GTEST_SKIP() << "model tests disabled (CHORUS_SKIP_MODEL_TESTS=1)";
#if !defined(CHORUS_TEST_VULKAN)
        GTEST_SKIP() << "GPU tests require a Vulkan test build";
#endif
    }
};

#pragma once

#include "gtest_utils.hpp"

#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"

#include <functional>
#include <memory>
#include <string>

/*
 * One provider, dressed for the contract battery.
 *
 * The suite holds engines by contract type and names no provider, so a subject
 * supplies the three things the contract cannot state for it: how to build the
 * engine, a config that brings it up, and request shapes that stream for a
 * while or finish promptly.
 */
struct EngineUnderTest {
    std::string label;
    bool model_gated = false; // subjects needing a model skip under CHORUS_SKIP_MODEL_TESTS=1

    std::function<std::unique_ptr<Chorus::InferenceEngine>()> make_engine;
    std::function<Chorus::ChorusConfig()> make_config;

    // Emits at least two Token signals; the suite paces it by blocking inside
    // the callback rather than by trusting a clock.
    std::function<void(Chorus::ChorusRequest&)> shape_long_request;
    // Runs to its own Stop quickly.
    std::function<void(Chorus::ChorusRequest&)> shape_short_request;
};

class EngineContractTest : public ChorusModelTest, public ::testing::WithParamInterface<EngineUnderTest> {
  protected:
    void SetUp() override {
        if (GetParam().model_gated)
            skip_if_model_tests_disabled();
    }
};

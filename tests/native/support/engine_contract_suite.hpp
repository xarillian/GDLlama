#pragma once

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
    std::string label;        // prefixes every case name in the runner output
    bool model_gated = false; // subjects needing a model skip under CHORUS_SKIP_MODEL_TESTS=1

    std::function<std::unique_ptr<Chorus::InferenceEngine>()> make_engine;
    std::function<Chorus::ChorusConfig()> make_config;

    // Emits at least two Token signals; the suite paces it by blocking inside
    // the callback rather than by trusting a clock.
    std::function<void(Chorus::ChorusRequest&)> shape_long_request;
    // Runs to its own Stop quickly.
    std::function<void(Chorus::ChorusRequest&)> shape_short_request;
};

/*
 * The obligations InferenceEngine states in prose, executed.
 *
 * Every provider runs this battery, so an obligation stays a shared property
 * instead of a paragraph each implementation reads for itself. A failure here
 * is a contract violation in the provider, or a contract that moved without
 * its header.
 */
void run_engine_contract_suite(const EngineUnderTest& subject);

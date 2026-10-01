#pragma once

#include "chorus/runtime/runtime.hpp"
#include "gtest/gtest.h"

#include <chrono>
#include <memory>
#include <thread>

struct ObservedLoad {
    Chorus::LoadId load_id = -1;
    Chorus::ChorusError error = Chorus::ChorusError::Unknown;
    std::string message;
    std::vector<Chorus::RuntimeEvent> events;

    bool ok() const { return error == Chorus::ChorusError::None; }
};

inline ObservedLoad wait_load_terminal(Chorus::ChorusRuntime& runtime, Chorus::LoadSubmitResult admission) {
    ObservedLoad observed;
    observed.load_id = admission.load_id;
    observed.error = admission.error;
    observed.message = std::move(admission.message);
    if (!admission.ok())
        return observed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool terminal = false;
    while (!terminal && std::chrono::steady_clock::now() < deadline) {
        for (auto& event : runtime.poll()) {
            if (event.load_id == admission.load_id && (event.kind == Chorus::RuntimeEvent::Kind::ModelLoaded ||
                                                       event.kind == Chorus::RuntimeEvent::Kind::ModelLoadFailed)) {
                observed.error = event.error;
                observed.message = event.text;
                terminal = true;
            }
            observed.events.push_back(std::move(event));
        }
        if (!terminal)
            std::this_thread::yield();
    }
    EXPECT_TRUE(terminal) << "Load terminal delivery timed out for ID " << admission.load_id;
    if (!terminal)
        observed.error = Chorus::ChorusError::Unknown;
    return observed;
}

inline ObservedLoad load_runtime(
    Chorus::ChorusRuntime& runtime, std::unique_ptr<Chorus::InferenceEngine> engine, const Chorus::ChorusConfig& config
) {
    auto observed = wait_load_terminal(runtime, runtime.load_engine(std::move(engine), config));
    for (const auto& event : observed.events)
        EXPECT_TRUE(event.load_id.has_value()) << "Fixture load discarded a request event";
    return observed;
}

inline bool runtime_terminal(Chorus::RuntimeEvent::Kind kind) {
    using Kind = Chorus::RuntimeEvent::Kind;
    return kind == Kind::Complete || kind == Kind::Embedding || kind == Kind::Error || kind == Kind::PromptRendered ||
           kind == Kind::MessageTokenCount;
}

inline std::vector<Chorus::RuntimeEvent> drain_runtime_events(Chorus::ChorusRuntime& runtime, size_t terminals = 1) {
    std::vector<Chorus::RuntimeEvent> events;
    size_t count = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    do {
        for (auto& event : runtime.poll()) {
            count += runtime_terminal(event.kind);
            events.push_back(std::move(event));
        }
        if (count >= terminals)
            break;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    EXPECT_GE(count, terminals) << "Runtime terminal delivery timed out";
    return events;
}

template <typename Predicate> void forward_runtime_until(Chorus::ChorusRuntime& runtime, Predicate done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!done() && std::chrono::steady_clock::now() < deadline) {
        EXPECT_TRUE(runtime.poll().empty());
        std::this_thread::yield();
    }
    EXPECT_TRUE(done()) << "Runtime provider forwarding timed out";
}

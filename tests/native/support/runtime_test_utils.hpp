#pragma once

#include "chorus/runtime/runtime.hpp"
#include "gtest/gtest.h"

#include <chrono>
#include <thread>

inline bool runtime_terminal(Chorus::RuntimeEvent::Kind kind) {
    using Kind = Chorus::RuntimeEvent::Kind;
    return kind == Kind::Complete || kind == Kind::Embedding || kind == Kind::Error ||
           kind == Kind::PromptRendered || kind == Kind::MessageTokenCount;
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

template <typename Predicate>
void forward_runtime_until(Chorus::ChorusRuntime& runtime, Predicate done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!done() && std::chrono::steady_clock::now() < deadline) {
        EXPECT_TRUE(runtime.poll().empty());
        std::this_thread::yield();
    }
    EXPECT_TRUE(done()) << "Runtime provider forwarding timed out";
}

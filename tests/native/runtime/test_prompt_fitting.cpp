#include "chorus/runtime/prompt_fitting.hpp" // adjust to "../src..." only if include fails; see step 3
#include "gtest_utils.hpp"

#include <string>
#include <variant>
#include <vector>

using Chorus::ChatMessage;
using Chorus::InjectedMessage;

// Probe: 1 token per message (content ignored) -- budgets read as message counts.
static std::optional<int32_t> one_per_message(const std::vector<ChatMessage>& msgs) {
    return (int32_t)msgs.size();
}

static std::vector<ChatMessage> turns(int n) {
    std::vector<ChatMessage> out{{"system", "persona"}};
    for (int i = 0; i < n; ++i)
        out.push_back({i % 2 == 0 ? "user" : "assistant", "t" + std::to_string(i)});
    return out;
}

TEST(PromptFitting, Fitting_noop_under_budget) {
    auto result = Chorus::fit_messages_to_budget(turns(4), {}, 10, one_per_message);
    auto& fit = std::get<Chorus::FitResult>(result);
    ASSERT_EQ(fit.dropped, 0);
    ASSERT_EQ((int)fit.fitted.size(), 5);
}

TEST(PromptFitting, Fitting_drops_oldest_pins_system) {
    auto result = Chorus::fit_messages_to_budget(turns(6), {}, 5, one_per_message);
    auto& fit = std::get<Chorus::FitResult>(result);
    ASSERT_EQ(fit.dropped, 2);
    ASSERT_EQ(fit.fitted.front().role, std::string("system"));
    ASSERT_EQ(fit.fitted[1].content, std::string("t2")); // t0, t1 dropped
    ASSERT_EQ(fit.fitted.back().content, std::string("t5"));
}

TEST(PromptFitting, Fitting_hard_fail_oversized) {
    std::vector<ChatMessage> history{{"system", "persona"}, {"user", "question"}};
    auto result = Chorus::fit_messages_to_budget(history, {}, 1, one_per_message);
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusError>(result));
    ASSERT_TRUE(std::get<Chorus::ChorusError>(result) == Chorus::ChorusError::InvalidRequest);
}

TEST(PromptFitting, Fitting_injections_survive_and_place) {
    std::vector<InjectedMessage> inject{{{"system", "it rains"}, 1}};
    auto result = Chorus::fit_messages_to_budget(turns(6), inject, 6, one_per_message);
    auto& fit = std::get<Chorus::FitResult>(result);
    ASSERT_EQ(fit.dropped, 2); // 7 history + 1 injection, budget 6
    // depth 1 => before the last message
    ASSERT_EQ(fit.fitted[fit.fitted.size() - 2].content, std::string("it rains"));
    ASSERT_EQ(fit.fitted.back().content, std::string("t5"));
}

TEST(PromptFitting, Fitting_depth_zero_and_clamp) {
    std::vector<ChatMessage> history{{"system", "persona"}, {"user", "q"}};
    std::vector<InjectedMessage> inject{{{"system", "end note"}, 0}, {{"system", "deep note"}, 99}};
    auto placed = Chorus::place_injections(history, inject);
    ASSERT_EQ((int)placed.size(), 4);
    ASSERT_EQ(placed.back().content, std::string("end note"));
    ASSERT_EQ(placed[1].content, std::string("deep note")); // clamped after system pin
}

TEST(PromptFitting, Fitting_equal_clamped_depths_preserve_order) {
    // Two injections that BOTH clamp to the pin boundary must keep array
    // order. Naive clamping inserts at the same index and reverses it.
    std::vector<ChatMessage> history{{"system", "persona"}, {"user", "q"}};
    std::vector<InjectedMessage> inject{{{"system", "first"}, 99}, {{"system", "second"}, 99}};
    auto placed = Chorus::place_injections(history, inject);
    ASSERT_EQ((int)placed.size(), 4);
    ASSERT_EQ(placed[1].content, std::string("first"));
    ASSERT_EQ(placed[2].content, std::string("second"));
    ASSERT_EQ(placed.back().content, std::string("q"));
}

TEST(PromptFitting, Fitting_drop_aligns_to_user_boundary) {
    // Dropping must reopen the window on a user turn: a lone drop that leaves
    // an assistant-led window (strict-alternation templates reject it) is
    // extended to the next user message.
    auto result = Chorus::fit_messages_to_budget(turns(4), {}, 4, one_per_message);
    auto& fit = std::get<Chorus::FitResult>(result);
    ASSERT_EQ(fit.dropped, 2); // 1 would fit the count but strand assistant t1
    ASSERT_EQ(fit.fitted[1].role, std::string("user"));
    ASSERT_EQ(fit.fitted[1].content, std::string("t2"));
}

TEST(PromptFitting, Fitting_nullopt_probe_unfitted) {
    auto never = [](const std::vector<ChatMessage>&) -> std::optional<int32_t> { return std::nullopt; };
    auto result = Chorus::fit_messages_to_budget(turns(6), {}, 1, never);
    auto& fit = std::get<Chorus::FitResult>(result);
    ASSERT_EQ(fit.dropped, 0);
    ASSERT_EQ((int)fit.fitted.size(), 7);
}

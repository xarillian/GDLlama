#include "chorus/runtime/prompt_fitting.hpp"
#include "support/gtest_utils.hpp"

namespace {
using Chorus::ChatMessage;
using Chorus::FittingCandidate;
using Chorus::InjectedMessage;
using Chorus::MessageContent;
using Chorus::MessageId;
using Chorus::MessageRole;

ChatMessage message(MessageRole role, const char* text) {
    return {role, MessageContent::text(text)};
}

std::optional<int32_t> one_per_message(const std::vector<ChatMessage>& messages) {
    return static_cast<int32_t>(messages.size());
}

TEST(PromptFitting, omits_only_durable_ids_in_history_order) {
    std::vector<FittingCandidate> history{
        {message(MessageRole::System, "persona"), 41},
        {message(MessageRole::User, "old question"), 7},
        {message(MessageRole::Assistant, "old answer"), 88},
        {message(MessageRole::User, "pending"), std::nullopt},
    };
    std::vector<InjectedMessage> inject{{message(MessageRole::System, "note"), 99}};

    auto fitted = Chorus::fit_messages_to_budget(history, inject, 3, one_per_message);
    ASSERT_TRUE(std::holds_alternative<Chorus::FitResult>(fitted));
    const auto& result = std::get<Chorus::FitResult>(fitted);
    ASSERT_EQ(result.omitted_message_ids, (std::vector<MessageId>{7, 88}));
    ASSERT_EQ(result.fitted.size(), 3U);
    ASSERT_EQ(result.fitted.back().role, MessageRole::User);
}

TEST(PromptFitting, unavailable_probe_preserves_all_messages_without_omissions) {
    std::vector<FittingCandidate> history{{message(MessageRole::User, "old"), 9}, {message(MessageRole::User, "pending"), std::nullopt}};
    auto never = [](const std::vector<ChatMessage>&) -> std::optional<int32_t> { return std::nullopt; };
    auto fitted = Chorus::fit_messages_to_budget(history, {}, 1, never);
    ASSERT_TRUE(std::holds_alternative<Chorus::FitResult>(fitted));
    const auto& result = std::get<Chorus::FitResult>(fitted);
    ASSERT_TRUE(result.omitted_message_ids.empty());
    ASSERT_EQ(result.fitted.size(), 2U);
}
} // namespace

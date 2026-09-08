#pragma once

#include "chorus/runtime/runtime.hpp"

#include <functional>
#include <optional>
#include <variant>
#include <vector>

namespace Chorus {

struct FittingCandidate {
    ChatMessage message;
    std::optional<MessageId> id;
};

struct FitResult {
    std::vector<ChatMessage> fitted;
    std::vector<MessageId> omitted_message_ids;
};

using RenderProbe = std::function<std::optional<int32_t>(const std::vector<ChatMessage>&)>;

std::vector<ChatMessage> place_injections(std::vector<ChatMessage> base, const std::vector<InjectedMessage>& inject);

std::variant<FitResult, ChorusError> fit_messages_to_budget(
    const std::vector<FittingCandidate>& history,
    const std::vector<InjectedMessage>& inject,
    int32_t budget,
    const RenderProbe& probe
);

} // namespace Chorus

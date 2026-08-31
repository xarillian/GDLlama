#pragma once

#include "chorus/core/common.hpp"

#include <functional>
#include <optional>
#include <variant>
#include <vector>

namespace Chorus {

struct FitResult {
    std::vector<ChatMessage> fitted;
    int32_t dropped = 0;
};

using RenderProbe = std::function<std::optional<int32_t>(const std::vector<ChatMessage>&)>;

/*
 * Places injected messages by depth from the end of a conversation.
 *
 * Depth zero appends. Entries that would cross the leading system-message run
 * are clamped just after it, and entries at equal depths retain array order.
 */
std::vector<ChatMessage> place_injections(std::vector<ChatMessage> base, const std::vector<InjectedMessage>& inject);

/*
 * Fits a conversation within a rendered-token budget.
 *
 * The leading system-message run and every injection remain pinned while the
 * oldest eligible messages are removed. After removal begins, the surviving
 * history starts at a user turn. A probe returning `std::nullopt` disables
 * fitting and preserves the complete history.
 *
 * Returns:
 *  - `Chorus::FitResult`: the fitted messages and number of removed history messages.
 *  - `Chorus::ChorusError`: the conversation cannot fit.
 *
 * Errors:
 *  - `Chorus::ChorusError::InvalidRequest`: pinned messages exceed the budget.
 */
std::variant<FitResult, ChorusError> fit_messages_to_budget(
    const std::vector<ChatMessage>& history,
    const std::vector<InjectedMessage>& inject,
    int32_t budget,
    const RenderProbe& probe
);

} // namespace Chorus

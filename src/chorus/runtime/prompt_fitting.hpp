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

// Returns `base` with `inject` entries placed by depth-from-end (0 = append;
// clamped to just after the leading system run). Equal depths keep array order.
std::vector<ChatMessage> place_injections(std::vector<ChatMessage> base, const std::vector<InjectedMessage>& inject);

// Spec #5-F: pin the leading system run and all injections, drop the oldest
// non-pinned message until the probe fits the budget. Once dropping begins,
// the window is advanced to the next user turn (strict-alternation templates
// reject assistant-led windows; orphaned replies are dangling context).
// InvalidRequest when nothing droppable remains. A nullopt probe
// short-circuits to "unfitted".
std::variant<FitResult, ChorusError> fit_messages_to_budget(
    const std::vector<ChatMessage>& history,
    const std::vector<InjectedMessage>& inject,
    int32_t budget,
    const RenderProbe& probe
);

} // namespace Chorus

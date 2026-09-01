#include "chorus/runtime/prompt_fitting.hpp"

#include <algorithm>

namespace Chorus {
namespace {

size_t leading_system_run(const std::vector<ChatMessage>& messages) {
    size_t n = 0;
    while (n < messages.size() && messages[n].role == "system")
        ++n;
    return n;
}

} // namespace

std::vector<ChatMessage> place_injections(std::vector<ChatMessage> base, const std::vector<InjectedMessage>& inject) {
    const size_t pin = leading_system_run(base);
    size_t clamped = 0;
    using Difference = std::vector<ChatMessage>::difference_type;
    for (const auto& injected : inject) {
        const size_t pin_effective = pin + clamped;
        const size_t max_from_end = base.size() - std::min(pin_effective, base.size());
        const size_t wanted = (size_t)std::max<int32_t>(injected.depth, 0);
        const size_t from_end = std::min(wanted, max_from_end);
        if (wanted > max_from_end)
            ++clamped;
        base.insert(base.end() - static_cast<Difference>(from_end), injected.message);
    }
    return base;
}

std::variant<FitResult, ChorusError> fit_messages_to_budget(
    const std::vector<ChatMessage>& history,
    const std::vector<InjectedMessage>& inject,
    int32_t budget,
    const RenderProbe& probe
) {
    const size_t pin = leading_system_run(history);
    const size_t droppable = history.size() > pin ? history.size() - pin - 1 : 0;
    for (size_t drop = 0; drop <= droppable; ++drop) {
        // Strict-alternation templates reject assistant-led retained windows.
        while (drop > 0 && drop < droppable && history[pin + drop].role != "user")
            ++drop;

        std::vector<ChatMessage> candidate;
        candidate.reserve(history.size() - drop + inject.size());
        candidate.insert(candidate.end(), history.begin(), history.begin() + pin);
        candidate.insert(candidate.end(), history.begin() + pin + drop, history.end());
        candidate = place_injections(std::move(candidate), inject);

        auto count = probe(candidate);
        if (!count.has_value())
            return FitResult{place_injections(history, inject), 0};
        if (*count <= budget)
            return FitResult{std::move(candidate), static_cast<int32_t>(drop)};
    }

    return ChorusError::InvalidRequest;
}

} // namespace Chorus

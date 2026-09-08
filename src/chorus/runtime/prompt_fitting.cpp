#include "chorus/runtime/prompt_fitting.hpp"

#include <algorithm>

namespace Chorus {
namespace {

size_t leading_system_run(const std::vector<FittingCandidate>& messages) {
    size_t n = 0;
    while (n < messages.size() && messages[n].message.role == MessageRole::System)
        ++n;
    return n;
}

std::vector<ChatMessage> messages_of(const std::vector<FittingCandidate>& candidates) {
    std::vector<ChatMessage> messages;
    messages.reserve(candidates.size());
    for (const auto& candidate : candidates)
        messages.push_back(candidate.message);
    return messages;
}

} // namespace

std::vector<ChatMessage> place_injections(std::vector<ChatMessage> base, const std::vector<InjectedMessage>& inject) {
    size_t pin = 0;
    while (pin < base.size() && base[pin].role == MessageRole::System)
        ++pin;
    size_t clamped = 0;
    using Difference = std::vector<ChatMessage>::difference_type;
    for (const auto& injected : inject) {
        const size_t pin_effective = pin + clamped;
        const size_t max_from_end = base.size() - std::min(pin_effective, base.size());
        const size_t wanted = static_cast<size_t>(std::max<int32_t>(injected.depth, 0));
        const size_t from_end = std::min(wanted, max_from_end);
        if (wanted > max_from_end)
            ++clamped;
        base.insert(base.end() - static_cast<Difference>(from_end), injected.message);
    }
    return base;
}

std::variant<FitResult, ChorusError> fit_messages_to_budget(
    const std::vector<FittingCandidate>& history,
    const std::vector<InjectedMessage>& inject,
    int32_t budget,
    const RenderProbe& probe
) {
    const size_t pin = leading_system_run(history);
    const size_t droppable = history.size() > pin ? history.size() - pin - 1 : 0;
    for (size_t drop = 0; drop <= droppable; ++drop) {
        while (drop > 0 && drop < droppable && history[pin + drop].message.role != MessageRole::User)
            ++drop;

        std::vector<FittingCandidate> candidate;
        candidate.reserve(history.size() - drop);
        candidate.insert(candidate.end(), history.begin(), history.begin() + static_cast<std::ptrdiff_t>(pin));
        candidate.insert(candidate.end(), history.begin() + static_cast<std::ptrdiff_t>(pin + drop), history.end());
        auto messages = place_injections(messages_of(candidate), inject);

        auto count = probe(messages);
        if (!count)
            return FitResult{place_injections(messages_of(history), inject), {}};
        if (*count <= budget) {
            std::vector<MessageId> omitted;
            for (size_t i = pin; i < pin + drop; ++i)
                if (history[i].id.has_value())
                    omitted.push_back(*history[i].id);
            return FitResult{std::move(messages), std::move(omitted)};
        }
    }
    return ChorusError::InvalidRequest;
}

} // namespace Chorus

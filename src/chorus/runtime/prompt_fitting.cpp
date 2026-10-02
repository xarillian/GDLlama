#include "chorus/runtime/prompt_fitting.hpp"

#include <algorithm>

namespace Chorus {

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

} // namespace Chorus

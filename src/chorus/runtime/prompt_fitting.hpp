#pragma once

#include "chorus/runtime/runtime.hpp"

namespace Chorus {

std::vector<ChatMessage> place_injections(std::vector<ChatMessage> base, const std::vector<InjectedMessage>& inject);

} // namespace Chorus

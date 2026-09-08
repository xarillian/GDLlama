#include "chorus/core/common.hpp"

namespace Chorus {

std::optional<std::string_view> message_role_name(MessageRole role) {
    switch (role) {
    case MessageRole::System:
        return "system";
    case MessageRole::User:
        return "user";
    case MessageRole::Assistant:
        return "assistant";
    }
    return std::nullopt;
}

std::optional<std::string> joined_text(const MessageContent& content) {
    std::string joined;
    for (const auto& part : content.parts) {
        const auto* text = std::get_if<std::string>(&part);
        if (!text)
            return std::nullopt;
        joined += *text;
    }
    return joined;
}

} // namespace Chorus

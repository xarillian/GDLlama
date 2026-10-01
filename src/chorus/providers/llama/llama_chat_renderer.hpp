#pragma once

#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_chat.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

struct llama_model;

namespace Chorus {

struct LlamaPreparedChat {
    LlamaChatRender render;
    std::vector<int32_t> tokens;
};

/*
 * Renders and tokenizes chat prompts for one loaded model.
 *
 * Runtime fitting renders every chat request before the provider prepares it, so each
 * render is remembered until the matching request takes it. Parsed override templates
 * are kept between requests. All methods are thread-safe, and concurrent renders run
 * in parallel on separate template instances.
 */
class LlamaChatRenderer {
  public:
    LlamaChatRenderer(const llama_model* model, common_chat_templates_ptr model_default_templates);

    /*
     * Renders a prompt for fitting or preview and remembers it for submission.
     *
     * Errors:
     *  - `Chorus::ChorusError::InvalidRequest`: template selection or application failed.
     *  - `Chorus::ChorusError::Tokenize`: the rendered prompt could not be tokenized.
     */
    std::variant<LlamaPreparedChat, RequestRejection> render(
        const std::vector<ChatMessage>& messages, const std::optional<std::string>& template_override,
        std::optional<bool> enable_thinking
    ) const;

    /*
     * Returns the remembered render for a submitted request, rendering it if absent.
     *
     * Errors:
     *  - `Chorus::ChorusError::InvalidRequest`: template selection or application failed.
     *  - `Chorus::ChorusError::Tokenize`: the rendered prompt could not be tokenized.
     */
    std::variant<LlamaPreparedChat, RequestRejection> take(
        const std::vector<ChatMessage>& messages, const std::optional<std::string>& template_override,
        std::optional<bool> enable_thinking
    ) const;

  private:
    struct Key {
        std::vector<ChatMessage> messages;
        std::optional<std::string> template_override;
        std::optional<bool> enable_thinking;
    };

    struct TemplateSet {
        using Override = std::pair<std::string, common_chat_templates_ptr>;
        common_chat_templates_ptr model_default;
        std::deque<Override> overrides;
    };

    std::variant<LlamaPreparedChat, RequestRejection> render_borrowed(
        const std::vector<ChatMessage>& messages, const std::optional<std::string>& template_override,
        std::optional<bool> enable_thinking
    ) const;
    std::variant<LlamaPreparedChat, RequestRejection> render_with(
        TemplateSet& templates, const std::vector<ChatMessage>& messages,
        const std::optional<std::string>& template_override, std::optional<bool> enable_thinking
    ) const;
    std::variant<const common_chat_templates*, RequestRejection>
    templates_for(TemplateSet& templates, const std::optional<std::string>& source) const;

    const llama_model* _model;
    bool _has_model_default;
    // llama.cpp does not guarantee that one parsed template can be applied concurrently,
    // so each render borrows a set of its own; sets grow to the peak number of renderers.
    mutable std::mutex _sets_mutex;
    mutable std::vector<std::unique_ptr<TemplateSet>> _idle_sets;
    mutable std::mutex _memo_mutex;
    mutable std::deque<std::pair<Key, LlamaPreparedChat>> _remembered;
};

} // namespace Chorus

#include "chorus/providers/llama/llama_chat_renderer.hpp"
#include "chorus/providers/llama/llama_utils.hpp"

#include <algorithm>

namespace Chorus {
namespace {

constexpr std::size_t kOverrideTemplateEntries = 8;
constexpr std::size_t kRememberedRenders = 64;

bool same_messages(const std::vector<ChatMessage>& left, const std::vector<ChatMessage>& right) {
    return std::ranges::equal(left, right, [](const ChatMessage& a, const ChatMessage& b) {
        return a.role == b.role && a.content.parts == b.content.parts;
    });
}

} // namespace

LlamaChatRenderer::LlamaChatRenderer(const llama_model* model, common_chat_templates_ptr model_default_templates)
    : _model(model), _has_model_default(model_default_templates != nullptr) {
    _idle_sets.push_back(std::make_unique<TemplateSet>(TemplateSet{std::move(model_default_templates), {}}));
}

std::variant<LlamaPreparedChat, RequestRejection> LlamaChatRenderer::render(
    const std::vector<ChatMessage>& messages, const std::optional<std::string>& template_override,
    std::optional<bool> enable_thinking
) const {
    auto rendered = render_borrowed(messages, template_override, enable_thinking);
    if (const auto* prepared = std::get_if<LlamaPreparedChat>(&rendered)) {
        std::lock_guard<std::mutex> lock(_memo_mutex);
        if (_remembered.size() >= kRememberedRenders)
            _remembered.pop_front();
        _remembered.emplace_back(Key{messages, template_override, enable_thinking}, *prepared);
    }
    return rendered;
}

std::variant<LlamaPreparedChat, RequestRejection> LlamaChatRenderer::take(
    const std::vector<ChatMessage>& messages, const std::optional<std::string>& template_override,
    std::optional<bool> enable_thinking
) const {
    {
        std::lock_guard<std::mutex> lock(_memo_mutex);
        auto found = std::ranges::find_if(_remembered, [&](const auto& entry) {
            const Key& key = entry.first;
            return key.enable_thinking == enable_thinking && key.template_override == template_override &&
                   same_messages(key.messages, messages);
        });
        if (found != _remembered.end()) {
            LlamaPreparedChat prepared = std::move(found->second);
            _remembered.erase(found);
            return prepared;
        }
    }
    return render_borrowed(messages, template_override, enable_thinking);
}

std::variant<LlamaPreparedChat, RequestRejection> LlamaChatRenderer::render_borrowed(
    const std::vector<ChatMessage>& messages, const std::optional<std::string>& template_override,
    std::optional<bool> enable_thinking
) const {
    std::unique_ptr<TemplateSet> templates;
    {
        std::lock_guard<std::mutex> lock(_sets_mutex);
        if (!_idle_sets.empty()) {
            templates = std::move(_idle_sets.back());
            _idle_sets.pop_back();
        } else {
            templates = std::make_unique<TemplateSet>();
            if (_has_model_default)
                templates->model_default = common_chat_templates_init(_model, "");
        }
    }
    auto rendered = render_with(*templates, messages, template_override, enable_thinking);
    std::lock_guard<std::mutex> lock(_sets_mutex);
    _idle_sets.push_back(std::move(templates));
    return rendered;
}

std::variant<LlamaPreparedChat, RequestRejection> LlamaChatRenderer::render_with(
    TemplateSet& templates, const std::vector<ChatMessage>& messages,
    const std::optional<std::string>& template_override, std::optional<bool> enable_thinking
) const {
    auto selected = templates_for(templates, template_override);
    if (auto* rejection = std::get_if<RequestRejection>(&selected))
        return *rejection;
    auto rendered = render_llama_chat(_model, std::get<const common_chat_templates*>(selected), std::nullopt, messages,
                                      enable_thinking);
    if (auto* rejection = std::get_if<RequestRejection>(&rendered))
        return *rejection;
    auto& render = std::get<LlamaChatRender>(rendered);
    auto tokens = LlamaUtils::tokenize_vocabulary(llama_model_get_vocab(_model), render.prompt, true, true);
    if (!tokens)
        return RequestRejection{ChorusError::Tokenize, "Rendered prompt tokenization failed."};
    return LlamaPreparedChat{std::move(render), std::move(*tokens)};
}

std::variant<const common_chat_templates*, RequestRejection>
LlamaChatRenderer::templates_for(TemplateSet& templates, const std::optional<std::string>& source) const {
    if (!source)
        return templates.model_default.get();
    auto& overrides = templates.overrides;
    auto found = std::ranges::find(overrides, *source, &TemplateSet::Override::first);
    if (found != overrides.end())
        return found->second.get();
    auto loaded = load_llama_chat_template(_model, *source);
    if (auto* rejection = std::get_if<RequestRejection>(&loaded))
        return *rejection;
    if (overrides.size() >= kOverrideTemplateEntries)
        overrides.pop_front();
    overrides.emplace_back(*source, std::get<common_chat_templates_ptr>(std::move(loaded)));
    return overrides.back().second.get();
}

} // namespace Chorus

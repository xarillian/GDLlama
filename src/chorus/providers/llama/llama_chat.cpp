#include "chorus/providers/llama/llama_chat.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <utility>

namespace Chorus {
namespace {

common_chat_msg to_common(const ChatMessage& message) {
    common_chat_msg out;
    out.role = message.role;
    out.content = message.content;
    return out;
}

} // namespace

std::variant<LlamaChatRender, RequestRejection> render_llama_chat(
    const llama_model* model,
    const common_chat_templates* defaults,
    const std::string& template_override,
    const std::vector<ChatMessage>& messages,
    bool enable_thinking
) {
    common_chat_templates_ptr override_templates;
    const common_chat_templates* templates = defaults;
    if (!template_override.empty()) {
        try {
            override_templates = common_chat_templates_init(model, template_override);
            templates = override_templates.get();
        } catch (const std::exception& e) {
            return RequestRejection{ChorusError::InvalidRequest, std::string("Invalid chat_template: ") + e.what()};
        }
    }
    if (!templates)
        return RequestRejection{ChorusError::InvalidRequest, "No chat template available for messages."};

    common_chat_templates_inputs inputs;
    inputs.messages.reserve(messages.size());
    for (const auto& message : messages)
        inputs.messages.push_back(to_common(message));
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = enable_thinking;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    try {
        common_chat_params params = common_chat_templates_apply(templates, inputs);
        LlamaChatRender render;
        render.prompt = std::move(params.prompt);
        render.additional_stops = std::move(params.additional_stops);
        render.supports_thinking = params.supports_thinking;
        render.parser_params = common_chat_parser_params(params);
        render.parser_params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        // The convenience constructor does not retain the compiled PEG parser.
        // Without it, response parsing silently treats reasoning as visible content.
        render.parser_params.parser.load(params.parser);
        return render;
    } catch (const std::exception& e) {
        return RequestRejection{
            ChorusError::InvalidRequest, std::string("Chat template application failed: ") + e.what()
        };
    }
}

std::optional<LlamaChatParseStream> make_llama_chat_parse_stream(const LlamaChatRender& render) {
    if (!render.supports_thinking)
        return std::nullopt;
    return LlamaChatParseStream(render.parser_params);
}

LlamaChatParseStream::LlamaChatParseStream(common_chat_parser_params params) : _params(std::move(params)) {}

LlamaChatParseStream::Delta LlamaChatParseStream::push(const std::string& piece) {
    if (_passthrough)
        return Delta{piece, {}};
    _raw_output += piece;

    // Re-parsing the complete buffer after every reasoning token would make
    // streaming quadratic. Waiting for about one-sixteenth of the parsed size
    // keeps total parsing work amortized linear. Deferred bytes surface in the
    // next delta or during finalization. Once visible content begins, every
    // piece is parsed until the parser can be retired.
    const bool content_started = !_previous_parse.content.empty();
    if (!content_started && _raw_output.size() - _parsed_bytes < std::max<std::size_t>(1, _parsed_bytes / 16))
        return {};

    constexpr bool is_partial = true;
    try {
        _parsed_bytes = _raw_output.size();
        Delta delta = diff_against_previous(common_chat_parse(_raw_output, is_partial, _params));
        // A verbatim visible-content delta means the complete buffer has been
        // attributed and the parser has reached its terminal content region. Chorus
        // does not accept tool calls, so subsequent bytes are also visible content
        // and can pass through without another full-buffer parse.
        if (content_started && !piece.empty() && delta.reasoning.empty() && delta.content == piece)
            _passthrough = true;
        return delta;
    } catch (const std::exception&) {
        // Surface only the triggering piece, then pass subsequent pieces through.
        // Re-emitting the complete buffer would duplicate previous deltas and could
        // expose reasoning as visible content. Any buffered bytes that were never
        // emitted are discarded. This boundary is reached only when the parser
        // rejects output produced from its own template.
        _passthrough = true;
        return Delta{piece, {}};
    }
}

LlamaChatParseStream::Delta LlamaChatParseStream::finalize() {
    if (_passthrough)
        return {};
    constexpr bool is_partial = false;
    try {
        return diff_against_previous(common_chat_parse(_raw_output, is_partial, _params));
    } catch (const std::exception&) {
        // Discard residual output rather than risk exposing reasoning as content.
        _passthrough = true;
        return {};
    }
}

LlamaChatParseStream::Delta LlamaChatParseStream::diff_against_previous(const common_chat_msg& parsed) {
    // Partial parsing can report an empty message when input ends inside a tag.
    // Preserve prior state so the next successful parse cannot re-emit prior output.
    if (parsed.empty() && !_previous_parse.empty())
        return {};
    Delta delta;
    for (const auto& diff : common_chat_msg_diff::compute_diffs(_previous_parse, parsed)) {
        delta.content += diff.content_delta;
        delta.reasoning += diff.reasoning_content_delta;
    }
    _previous_parse = parsed;
    return delta;
}

} // namespace Chorus

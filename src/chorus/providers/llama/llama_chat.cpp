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

std::variant<LlamaChatRender, RequestRejection>
render_llama_chat(const common_chat_templates* tmpls, const std::vector<ChatMessage>& messages, bool enable_thinking) {
    common_chat_templates_inputs inputs;
    inputs.messages.reserve(messages.size());
    for (const auto& message : messages)
        inputs.messages.push_back(to_common(message));
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = enable_thinking;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    try {
        common_chat_params params = common_chat_templates_apply(tmpls, inputs);
        LlamaChatRender render;
        render.prompt = std::move(params.prompt);
        render.additional_stops = std::move(params.additional_stops);
        render.supports_thinking = params.supports_thinking;
        render.parser_params = common_chat_parser_params(params);
        render.parser_params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        // The convenience ctor above copies only format + generation_prompt;
        // without loading the PEG arena, common_chat_parse silently falls back
        // to a content-only parser and reasoning never splits (chat.cpp:2859).
        render.parser_params.parser.load(params.parser);
        return render;
    } catch (const std::exception& e) {
        return RequestRejection{
            ChorusError::InvalidRequest, std::string("Chat template application failed: ") + e.what()
        };
    }
}

LlamaChatParseStream::LlamaChatParseStream(common_chat_parser_params params) : _params(std::move(params)) {}

std::optional<LlamaChatParseStream> make_llama_chat_parse_stream(const LlamaChatRender& render) {
    if (!render.supports_thinking)
        return std::nullopt;
    return LlamaChatParseStream(render.parser_params);
}

LlamaChatParseStream::Delta LlamaChatParseStream::diff_against_previous(const common_chat_msg& parsed) {
    // Upstream guard (server-task.cpp:158-173): a partial parse can yield an
    // EMPTY message (e.g. the raw buffer ends mid-tag). Replacing a non-empty
    // previous state with empty would re-emit everything on the next parse.
    if (parsed.empty() && !_previous.empty())
        return {};
    Delta delta;
    for (const auto& diff : common_chat_msg_diff::compute_diffs(_previous, parsed)) {
        delta.content += diff.content_delta;
        delta.reasoning += diff.reasoning_content_delta;
    }
    _previous = parsed;
    return delta;
}

LlamaChatParseStream::Delta LlamaChatParseStream::push(const std::string& piece) {
    if (_passthrough)
        return Delta{piece, {}};
    _raw += piece;

    // Pre-content throttle (spec F2): while the think block is open, parse
    // only once ~1/16th of the already-parsed size has newly accumulated,
    // keeping the parse-over-everything cost amortized-linear per request.
    // Costs only reasoning-channel latency; skipped bytes surface on the next
    // parse or in finalize. Content streams token-granular: once it starts,
    // every piece parses until the identity flip below retires the parser.
    const bool content_started = !_previous.content.empty();
    if (!content_started && _raw.size() - _parsed_bytes < std::max<std::size_t>(1, _parsed_bytes / 16))
        return {};

    try {
        _parsed_bytes = _raw.size();
        Delta delta = diff_against_previous(common_chat_parse(_raw, /*is_partial=*/true, _params));
        // Identity flip (spec F2): the parser passed a pure-content piece
        // through verbatim, so everything so far is attributed. We field no
        // tool calls, and every format's no-tools grammar ends in
        // content(rest) (chat.cpp builders), so everything after is content
        // too: retire the per-piece full reparse. Revisit if tool-call
        // support ever lands in ChorusRequest.
        if (content_started && !piece.empty() && delta.reasoning.empty() && delta.content == piece)
            _passthrough = true;
        return delta;
    } catch (const std::exception&) {
        // Break-once policy: surface ONLY the triggering piece as content and
        // pass every future piece straight through. We deliberately do NOT
        // re-emit _raw (already partially surfaced as deltas -- re-emitting
        // duplicates it, and it may contain reasoning that must not leak into
        // content). The un-surfaced middle, if any, is lost at this single
        // boundary; a break here means the parser rejected its own
        // template-described output, which the tests treat as exceptional.
        _passthrough = true;
        return Delta{piece, {}};
    }
}

LlamaChatParseStream::Delta LlamaChatParseStream::finalize() {
    if (_passthrough)
        return {};
    try {
        return diff_against_previous(common_chat_parse(_raw, /*is_partial=*/false, _params));
    } catch (const std::exception&) {
        _passthrough = true;
        return {};
    }
}

} // namespace Chorus

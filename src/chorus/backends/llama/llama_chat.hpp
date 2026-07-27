#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"

#include <chat.h>

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace Chorus {

// What the ingest path needs from one template application.
struct LlamaChatRender {
    std::string prompt;
    std::vector<std::string> additional_stops;
    bool supports_thinking = false;
    common_chat_parser_params parser_params;
};

// Renders messages through llama.cpp common's jinja path (use_jinja = true,
// add_generation_prompt = true). Throws nothing: template failures come back
// as RequestRejection{InvalidRequest}.
std::variant<LlamaChatRender, RequestRejection>
render_llama_chat(const common_chat_templates* tmpls, const std::vector<ChatMessage>& messages, bool enable_thinking);

// Incremental reasoning/content splitter over accumulated raw output.
// Wraps common_chat_parse(is_partial=true) + common_chat_msg_diff.
class LlamaChatParseStream {
  public:
    struct Delta {
        std::string content;
        std::string reasoning;
    };

    explicit LlamaChatParseStream(common_chat_parser_params params);
    Delta push(const std::string& piece); // accumulate + partial parse + diff
    Delta finalize();                     // final full parse + diff
    // True once the parser has retired (identity flip or break-once) and
    // pieces flow straight through as content.
    bool content_passthrough() const { return _passthrough; }

  private:
    Delta diff_against_previous(const common_chat_msg& parsed);

    common_chat_parser_params _params;
    std::string _raw;
    common_chat_msg _previous;
    // Parser retired: pieces pass through as content, finalize is a no-op.
    // Entered deliberately (identity flip on a single-think format, spec F2)
    // or defensively (a partial parse threw; break-once policy).
    bool _passthrough = false;
    std::size_t _parsed_bytes = 0; // _raw size at the last parse (throttle bookkeeping)
};

std::optional<LlamaChatParseStream> make_llama_chat_parse_stream(const LlamaChatRender& render);

} // namespace Chorus

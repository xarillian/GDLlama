#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"

#include <chat.h>

#include <optional>
#include <string>
#include <variant>
#include <vector>

struct llama_model;

namespace Chorus {

// What the ingest path needs from one template application.
struct LlamaChatRender {
    std::string prompt;
    std::vector<std::string> additional_stops;
    bool supports_thinking = false;
    common_chat_parser_params parser_params;
};

// Selects an explicit override or the model's defaults, then renders messages
// through llama.cpp common's jinja path. Throws nothing: template failures
// come back as RequestRejection{InvalidRequest}.
// Callers sharing `model` or `defaults` must serialize calls because llama.cpp
// does not guarantee thread-safe chat-template initialization or application.
std::variant<LlamaChatRender, RequestRejection> render_llama_chat(
    const llama_model* model,
    const common_chat_templates* defaults,
    const std::string& template_override,
    const std::vector<ChatMessage>& messages,
    bool enable_thinking
);

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

  private:
    Delta diff_against_previous(const common_chat_msg& parsed);

    common_chat_parser_params _params;
    std::string _raw;
    common_chat_msg _previous;
    // Parser retired: pieces pass through as content, finalize is a no-op.
    // Entered deliberately after an identity flip on a single-think format,
    // or defensively (a partial parse threw; break-once policy).
    bool _passthrough = false;
    std::size_t _parsed_bytes = 0; // _raw size at the last parse (throttle bookkeeping)
};

std::optional<LlamaChatParseStream> make_llama_chat_parse_stream(const LlamaChatRender& render);

} // namespace Chorus

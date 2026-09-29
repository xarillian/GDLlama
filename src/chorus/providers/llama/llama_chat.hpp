#pragma once

#include "chorus/core/common.hpp"

#include <chat.h>

#include <optional>
#include <string>
#include <variant>
#include <vector>

struct llama_model;

namespace Chorus {

struct LlamaChatRender {
    std::string prompt;
    std::vector<std::string> template_stop_sequences;
    bool supports_thinking = false;
    common_chat_parser_params parser_params;
};

/*
 * Parses a caller-supplied chat template for one model.
 *
 * Returns:
 *  - `::common_chat_templates_ptr`: the parsed template, reusable for later renders.
 *  - `Chorus::RequestRejection`: the template is empty or llama.cpp rejects it.
 *
 * Errors:
 *  - `Chorus::ChorusError::InvalidRequest`: the template is empty or invalid.
 */
std::variant<common_chat_templates_ptr, RequestRejection>
load_llama_chat_template(const llama_model* model, const std::string& template_source);

/*
 * Renders messages through an explicit or model-provided llama.cpp chat template.
 *
 * Calls sharing the same model or template instances must be serialized because
 * llama.cpp does not guarantee thread-safe template initialization or application.
 * llama.cpp template failures are returned as `Chorus::RequestRejection`.
 *
 * Returns:
 *  - `Chorus::LlamaChatRender`: the rendered prompt and response parser state.
 *  - `Chorus::RequestRejection`: no template is available or llama.cpp rejects template initialization or application.
 *
 * Errors:
 *  - `Chorus::ChorusError::InvalidRequest`: template selection or application failed.
 */
std::variant<LlamaChatRender, RequestRejection> render_llama_chat(
    const llama_model* model,
    const common_chat_templates* model_default_chat_templates,
    const std::optional<std::string>& template_override,
    const std::vector<ChatMessage>& messages,
    std::optional<bool> enable_thinking
);

/*
 * Separates accumulated model output into visible-content and reasoning deltas.
 *
 * `Chorus::LlamaChatParseStream::push` may return an empty delta while reasoning
 * accumulates between throttled partial parses. During successful parsing, each
 * classified fragment is returned at most once. `Chorus::LlamaChatParseStream::finalize`
 * performs the final parse and returns any residual output.
 *
 * Once parsing reaches terminal visible content, subsequent pieces pass through
 * as content and `Chorus::LlamaChatParseStream::finalize` becomes a no-op. If a
 * partial parse fails, the triggering and subsequent pieces pass through while
 * buffered but unsurfaced output is discarded. A failed final parse likewise
 * discards residual output. These policies prevent duplicate content and reasoning
 * from leaking into the visible channel.
 */
class LlamaChatParseStream {
  public:
    struct Delta {
        std::string content;
        std::string reasoning;
    };

    explicit LlamaChatParseStream(common_chat_parser_params params);
    /// Adds one output piece and returns newly classified output.
    Delta push(const std::string& piece);
    /// Returns output remaining after the final, non-partial parse.
    Delta finalize();

  private:
    Delta diff_against_previous(const common_chat_msg& parsed);

    common_chat_parser_params _params;
    std::string _raw_output;
    common_chat_msg _previous_parse;
    bool _passthrough = false;
    std::size_t _parsed_bytes = 0;
};

/*
 * Creates a response parser when the rendered template supports reasoning.
 *
 * Returns:
 *  - `Chorus::LlamaChatParseStream`: the template supports reasoning separation.
 *  - `std::nullopt`: the template does not support reasoning separation.
 */
std::optional<LlamaChatParseStream> make_llama_chat_parse_stream(const LlamaChatRender& render);

} // namespace Chorus

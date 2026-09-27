#include "chorus/providers/llama/llama_chat.hpp"
#include "gtest_utils.hpp"

#include <chat.h> // match the include style llama_chat.hpp settles on

#include <fstream>
#include <iterator>
#include <string>
#include <variant>
#include <vector>

// Minimal ChatML template with a thinking block, model-free. Assistant
// messages round-trip reasoning_content inside literal <think>...</think>
// tags so llama.cpp's differential autoparser (compare_reasoning_presence,
// chat-diff-analyzer.cpp:443) learns <think> and </think> as separate
// open/close reasoning delimiters instead of one fused literal.
static const char* kThinkTemplate =
    "{%- for message in messages -%}"
    "<|im_start|>{{ message.role }}\n"
    "{%- if message.role == 'assistant' and message.reasoning_content is defined and message.reasoning_content -%}"
    "<think>\n{{ message.reasoning_content }}\n</think>\n"
    "{%- endif -%}"
    "{{ message.content }}<|im_end|>\n"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|im_start|>assistant\n"
    "{%- if enable_thinking is defined and not enable_thinking -%}<think>\n\n</think>\n\n{%- endif -%}"
    "{%- endif -%}";

static common_chat_templates_ptr make_templates() {
    return common_chat_templates_init(/*model=*/nullptr, kThinkTemplate);
}

// Parser params must come from a real template application. The convenience
// common_chat_parser_params(common_chat_params&) ctor copies only format and
// generation_prompt, but drops the PEG
// parser; common_chat_parse falls back to a content-only parser on an empty
// arena (chat.cpp:2859) and never splits <think>. The arena is loaded from
// params.parser explicitly (upstream idiom: test-chat.cpp:1003).
static common_chat_parser_params think_parser_params() {
    auto tmpls = make_templates();
    common_chat_templates_inputs inputs;
    common_chat_msg user;
    user.role = "user";
    user.content = "hi";
    inputs.messages = {user};
    inputs.use_jinja = true;
    inputs.enable_thinking = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
    common_chat_parser_params parser_params(params);
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    parser_params.parser.load(params.parser);
    return parser_params;
}

TEST(LlamaChat, LlamaChat_render_produces_role_scaffolding) {
    auto tmpls = make_templates();
    std::vector<Chorus::ChatMessage> messages{{Chorus::MessageRole::System, Chorus::MessageContent::text("You are Brunn.")}, {Chorus::MessageRole::User, Chorus::MessageContent::text("Hello there!")}};
    auto result = Chorus::render_llama_chat(nullptr, tmpls.get(), std::nullopt, messages, /*enable_thinking=*/true);
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaChatRender>(result));
    const auto& render = std::get<Chorus::LlamaChatRender>(result);
    ASSERT_TRUE(render.prompt.find("<|im_start|>system") != std::string::npos);
    ASSERT_TRUE(render.prompt.find("You are Brunn.") != std::string::npos);
    ASSERT_TRUE(render.prompt.find("<|im_start|>user") != std::string::npos);
    // Generation prompt appended (assistant turn opened, not closed).
    ASSERT_TRUE(render.prompt.rfind("<|im_start|>assistant") != std::string::npos);
}

TEST(LlamaChat, LlamaChat_absent_thinking_defaults_to_true_and_false_is_honored) {
    auto tmpls = make_templates();
    const std::vector<Chorus::ChatMessage> messages{
        {Chorus::MessageRole::User, Chorus::MessageContent::text("Hello there!")}
    };
    auto absent = Chorus::render_llama_chat(nullptr, tmpls.get(), std::nullopt, messages, std::nullopt);
    auto enabled = Chorus::render_llama_chat(nullptr, tmpls.get(), std::nullopt, messages, true);
    auto disabled = Chorus::render_llama_chat(nullptr, tmpls.get(), std::nullopt, messages, false);
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaChatRender>(absent));
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaChatRender>(enabled));
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaChatRender>(disabled));
    EXPECT_EQ(std::get<Chorus::LlamaChatRender>(absent).prompt, std::get<Chorus::LlamaChatRender>(enabled).prompt);
    EXPECT_NE(std::get<Chorus::LlamaChatRender>(enabled).prompt, std::get<Chorus::LlamaChatRender>(disabled).prompt);
}

TEST(LlamaChat, LlamaChat_explicit_override_without_defaults) {
    auto result = Chorus::render_llama_chat(
        nullptr, nullptr, kThinkTemplate, {{Chorus::MessageRole::User, Chorus::MessageContent::text("Hello there!")}}, /*enable_thinking=*/true
    );
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaChatRender>(result));
    const auto& render = std::get<Chorus::LlamaChatRender>(result);
    ASSERT_TRUE(render.prompt.find("<|im_start|>user") != std::string::npos);
}

TEST(LlamaChat, LlamaChat_selected_empty_template_does_not_fall_back_to_model_default) {
    auto tmpls = make_templates();
    auto result = Chorus::render_llama_chat(
        nullptr, tmpls.get(), std::string{},
        {{Chorus::MessageRole::User, Chorus::MessageContent::text("Hello there!")}}, std::nullopt
    );
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(result));
    EXPECT_EQ(std::get<Chorus::RequestRejection>(result).error, Chorus::ChorusError::InvalidRequest);
}

TEST(LlamaChat, LlamaChat_missing_template_rejected) {
    auto result = Chorus::render_llama_chat(nullptr, nullptr, std::nullopt, {{Chorus::MessageRole::User, Chorus::MessageContent::text("Hello there!")}}, /*enable_thinking=*/true);
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(result));
    const auto& rejection = std::get<Chorus::RequestRejection>(result);
    ASSERT_TRUE(rejection.error == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(rejection.message.find("No chat template") != std::string::npos);
}

TEST(LlamaChat, LlamaChat_invalid_override_rejected) {
    auto result =
        Chorus::render_llama_chat(nullptr, nullptr, "{% if", {{Chorus::MessageRole::User, Chorus::MessageContent::text("Hello there!")}}, /*enable_thinking=*/true);
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(result));
    const auto& rejection = std::get<Chorus::RequestRejection>(result);
    ASSERT_TRUE(rejection.error == Chorus::ChorusError::InvalidRequest);
}

TEST(LlamaChat, LlamaChat_parse_stream_splits_reasoning) {
    Chorus::LlamaChatParseStream stream(think_parser_params());

    std::string reasoning, content;
    // Feed a synthetic think-marker stream in awkward chunks (tag split mid-piece).
    for (const std::string piece : {"<th", "ink>let me ", "reason</think", ">the answer ", "is 4"}) {
        auto delta = stream.push(piece);
        reasoning += delta.reasoning;
        content += delta.content;
    }
    auto tail = stream.finalize();
    reasoning += tail.reasoning;
    content += tail.content;

    ASSERT_TRUE(reasoning.find("let me reason") != std::string::npos);
    ASSERT_TRUE(content.find("the answer is 4") != std::string::npos);
    ASSERT_TRUE(content.find("<think>") == std::string::npos);
    ASSERT_TRUE(content.find("</think>") == std::string::npos);
}

TEST(LlamaChat, LlamaChat_parse_stream_plain_passthrough) {
    Chorus::LlamaChatParseStream stream(think_parser_params());

    std::string content;
    for (const std::string piece : {"plain ", "text ", "reply"})
        content += stream.push(piece).content;
    content += stream.finalize().content;
    ASSERT_EQ(content, std::string("plain text reply"));
}

TEST(LlamaChat, LlamaChat_parse_stream_no_dup_after_empty_partial) {
    // A lone "<" can partial-parse to an EMPTY message; upstream only replaces
    // its previous state when the new parse is non-empty (server-task.cpp:158).
    // Regression guard: total surfaced content must never repeat earlier text.
    Chorus::LlamaChatParseStream stream(think_parser_params());
    std::string content;
    for (const std::string piece : {"hello ", "<", "b>world"})
        content += stream.push(piece).content;
    content += stream.finalize().content;
    // Exact reassembly may vary with markup handling; the invariant is no
    // duplication of the already-surfaced prefix.
    ASSERT_TRUE(content.find("hello hello") == std::string::npos);
    ASSERT_TRUE(content.find("hello ") == 0);
}

TEST(LlamaChat, LlamaChat_parse_stream_throttles_pre_content) {
    // While the think block is open, partial parses amortize (about 1/16th
    // of parsed size must accumulate between parses) so the full-buffer
    // reparse cost stays linear per request. Skipped pushes return empty
    // deltas; nothing may be lost or reordered across the throttle.
    Chorus::LlamaChatParseStream stream(think_parser_params());
    std::string reasoning, content, expected;
    auto absorb = [&](const Chorus::LlamaChatParseStream::Delta& delta) {
        reasoning += delta.reasoning;
        content += delta.content;
    };
    absorb(stream.push("<think>"));
    int empty_reasoning_deltas = 0;
    for (int i = 0; i < 200; ++i) {
        const std::string piece = "w" + std::to_string(i) + " ";
        expected += piece;
        auto delta = stream.push(piece);
        if (delta.reasoning.empty())
            ++empty_reasoning_deltas;
        absorb(delta);
    }
    expected += "end";
    absorb(stream.push("end"));
    absorb(stream.push("</think>done"));
    absorb(stream.finalize());
    ASSERT_TRUE(empty_reasoning_deltas > 50); // the throttle actually skipped parses
    ASSERT_EQ(reasoning, expected);           // lossless across skips + finalize
    ASSERT_EQ(content, std::string("done"));
}

TEST(LlamaChat, LlamaChat_parse_stream_keeps_streaming_after_content_starts) {
    // Once the think block closes, subsequent content must surface exactly
    // once. Pieces mirror the splits test: this template's partial parse only
    // moves to content when the close tag straddles pieces, which is also the
    // realistic token-stream shape.
    Chorus::LlamaChatParseStream stream(think_parser_params());
    std::string content;
    for (const std::string piece : {"<th", "ink>let me ", "reason</think", ">the answer "})
        content += stream.push(piece).content;
    content += stream.push("is 4").content;
    auto delta = stream.push(" tail");
    ASSERT_EQ(delta.content, std::string(" tail"));
    ASSERT_TRUE(delta.reasoning.empty());
    // Everything already surfaced: finalize is a no-op after the flip.
    auto residual = stream.finalize();
    ASSERT_TRUE(residual.content.empty() && residual.reasoning.empty());
    content += delta.content + residual.content;
    ASSERT_EQ(content, std::string("the answer is 4 tail"));
}

TEST(LlamaChat, LlamaChat_deepseek_thinking_off_separates_reasoning) {
    std::ifstream file("third-party/llama.cpp/models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja");
    ASSERT_TRUE(file.good());
    const std::string source{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    auto templates = common_chat_templates_init(nullptr, source);
    auto result = Chorus::render_llama_chat(nullptr, templates.get(), std::nullopt, {{Chorus::MessageRole::User, Chorus::MessageContent::text("hello")}}, false);
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaChatRender>(result));
    const auto& render = std::get<Chorus::LlamaChatRender>(result);
    ASSERT_TRUE(render.supports_thinking);
    ASSERT_TRUE(render.prompt.find("<｜Assistant｜><think>") != std::string::npos);

    auto stream = Chorus::make_llama_chat_parse_stream(render);
    ASSERT_TRUE(stream.has_value());
    std::string content;
    std::string reasoning;
    for (const std::string piece : {"reasoning ", "here</think>", "answer"}) {
        auto delta = stream->push(piece);
        content += delta.content;
        reasoning += delta.reasoning;
    }
    auto tail = stream->finalize();
    content += tail.content;
    reasoning += tail.reasoning;

    ASSERT_TRUE(reasoning.find("reasoning here") != std::string::npos);
    ASSERT_EQ(content, std::string("answer"));
}

#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "chorus/core/generation_config.hpp"
#include "chorus/core/identity.hpp"
#include "chorus/core/log.hpp"
#include "chorus/core/model_spec.hpp"
#include "chorus/core/provider_option_value.hpp"

namespace Chorus {

enum class ChorusError {
    None,
    Unknown,
    ModelLoad,
    ContextInit,
    Decode,
    Tokenize,
    InvalidRequest,
    EngineNotReady,
    Cancelled,
    UnsupportedModelFormat,
    UnsupportedFeature,
    UnsupportedOption,
    SessionBusy,
};

enum class RequestType { Generate, Embedding };

enum class ExecutionMode { Shared, Exclusive };

enum class TokenChannel { Content, Reasoning };

enum class LoadPhase { ReleasingEngine, LoadingModel, InitializingEngine };

struct LoadProgress {
    LoadPhase phase = LoadPhase::LoadingModel;
    std::optional<float> fraction;
};

struct InitializationControl {
    std::stop_token stop_token;
    std::function<void(const LoadProgress&)> on_progress;
};

/*
 * Engine-wide configuration, fixed for the life of one engine instance.
 */
struct ChorusConfig {
    InitialModelSpec model;
    ProviderOptionMap provider_options; // Values keyed first by provider name.
    LogLevel log_level = log_level_default;
};

enum class MessageRole { System, User, Assistant };

using MessagePart = std::variant<std::string>;

struct MessageContent {
    std::vector<MessagePart> parts;

    static MessageContent text(std::string text) { return {{std::move(text)}}; }
};

struct ChatMessage {
    MessageRole role = MessageRole::User;
    MessageContent content;
};

std::optional<std::string_view> message_role_name(MessageRole role);
std::optional<std::string> joined_text(const MessageContent& content);

struct RenderedPrompt {
    std::string text;
    int32_t token_count = 0;
};

struct RequestRejection {
    ChorusError error = ChorusError::Unknown;
    std::string message;
};

struct InitializationFailure {
    ChorusError error = ChorusError::Unknown;
    std::string message;

    bool operator==(ChorusError value) const { return error == value; }
};

inline bool operator==(ChorusError value, const InitializationFailure& failure) {
    return failure == value;
}

/// Token counts for one completed generation.
struct GenerationUsage {
    // Tokens in the generation's prompt, special and reused tokens included.
    int64_t prompt_tokens = 0;
    // Leading prompt tokens reused from retained KV instead of being evaluated again.
    int64_t cached_prompt_tokens = 0;
    // Sampled tokens across both channels, including any end-of-generation or stop-marker token absent from the text.
    int64_t generated_tokens = 0;
};

/*
 * Every accepted request emits exactly one terminal and nothing follows it.
 *
 * Generations emit zero or more `Chorus::ChorusSignal::Token` signals, then
 * `Chorus::ChorusSignal::Completion` with usage. Embedding requests emit only
 * `Chorus::ChorusSignal::Embedding` with their normalized vector. Failure or
 * cancellation ends either kind with `Chorus::ChorusSignal::Error`, without
 * usage or result; only generation tokens may precede it. A signal of the wrong
 * request kind ends the request with `Chorus::ChorusError::Unknown` at the
 * runtime boundary.
 *
 * Usage counts are nonnegative and `Chorus::GenerationUsage::cached_prompt_tokens`
 * never exceeds `Chorus::GenerationUsage::prompt_tokens`. A generation capped at
 * `Chorus::GenerationConfig::max_tokens == 0` completes without running the model
 * and reports zero for every count, even with a nonempty prompt.
 */
struct ChorusSignal {
    struct Token {
        TokenChannel channel;
        std::string text;
    };

    struct Completion {
        GenerationUsage usage;
    };

    struct Embedding {
        std::vector<float> values;
    };

    struct Error {
        ChorusError code;
        std::string message;
    };

    using Event = std::variant<Token, Completion, Embedding, Error>;

    ChorusSignal(RequestId id, Event event) : request_id(id), event(std::move(event)) {}

    /// Whether this signal ends its request.
    bool is_terminal() const;

    RequestId request_id;
    Event event;
};

/*
 * When `Chorus::ChorusRequest::messages` is nonempty, the provider renders them
 * through a chat template and ignores `Chorus::ChorusRequest::prompt`.
 */
struct ChorusRequest {
    RequestId id;
    // Unset for stateless requests.
    std::optional<SessionId> session_id;

    RequestType type = RequestType::Generate;
    std::string prompt;

    int priority = 0;
    ExecutionMode execution = ExecutionMode::Shared;

    std::vector<ChatMessage> messages;
    std::optional<std::string> chat_template; // A selected template overrides the model's embedded template.

    GenerationConfig gen_config;

    // The provider must check its actual rendered prompt before prefill.
    std::optional<int64_t> exact_prompt_budget;

    // Provider worker threads may invoke this callback; it must be thread-safe.
    std::function<void(ChorusSignal&)> on_event;
};
} // namespace Chorus

#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
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

enum class TokenChannel { Content, Reasoning };

/*
 * Engine-wide configuration, fixed for the life of one engine instance.
 */
struct ChorusConfig {
    InitialModelSpec model;
    ProviderOptionMap provider_options; // Values keyed first by provider name.
    LogLevel log_level = log_level_default;
};

struct ChatMessage {
    std::string role;
    std::string content;
};

struct RenderedPrompt {
    std::string text;
    int32_t token_count = 0;
};

struct RequestRejection {
    ChorusError error = ChorusError::Unknown;
    std::string message;
};

/*
 * A message spliced into a conversation at a fixed distance from its end.
 *
 * `InjectedMessage::depth == 0` indicates it should go after the last message;
 * larger values move it that many messages toward the front.
 */
struct InjectedMessage {
    ChatMessage message;
    int32_t depth = 0;
};

/*
 * An accepted request may emit any number of `ChorusSignal::Token` or
 * `ChorusSignal::Embedding` events, followed by exactly one terminal
 * `ChorusSignal::Stop` or `ChorusSignal::Error` event.
 */
struct ChorusSignal {
    struct Token {
        TokenChannel channel;
        std::string text;
    };

    struct Embedding {
        std::vector<float> values;
    };

    struct Stop {};

    struct Error {
        ChorusError code;
        std::string message;
    };

    using Event = std::variant<Token, Embedding, Stop, Error>;

    ChorusSignal(RequestId id, Event event) : request_id(id), event(std::move(event)) {}

    RequestId request_id;
    Event event;
};

/*
 * When `ChorusRequest::messages` is non-empty, the provider renders them
 * through a chat template and ignores `ChorusRequest::prompt`. Every accepted
 * request emits exactly one terminal `ChorusSignal::Stop` or
 * `ChorusSignal::Error`.
 */
struct ChorusRequest {
    RequestId id;
    // Unset for stateless requests.
    std::optional<SessionId> session_id;

    RequestType type = RequestType::Generate;
    std::string prompt;

    int priority = 0;

    std::vector<ChatMessage> messages;
    std::string chat_template; // Overrides the model's embedded template when non-empty.

    GenerationConfig gen_config;

    // Provider worker threads may invoke this callback; it must be thread-safe.
    std::function<void(ChorusSignal&)> on_event;
};
} // namespace Chorus

#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "chorus/core/generation_config.hpp"
#include "chorus/core/identity.hpp"
#include "chorus/core/log.hpp"
#include "chorus/core/model_spec.hpp"
#include "chorus/core/provider_option_value.hpp"

namespace Chorus {

/// Error codes for Chorus operations.
enum class ChorusError {
    None,
    Unknown,                // Catch-all for unexpected failures
    ModelLoad,              // Failed to load model file
    ContextInit,            // Failed to create inference context
    Decode,                 // Inference decode step failed
    Tokenize,               // Tokenization failed
    InvalidRequest,         // Bad input from caller (missing prompt, bad config, etc.)
    EngineNotReady,         // Operation attempted before engine is initialized
    Cancelled,              // Request terminated by stop_all()/engine replacement before completion
    UnsupportedModelFormat, // Provider cannot load the requested artifact format
    UnsupportedFeature,     // Requested capability (constraint, modality, ...) not supported
    UnsupportedOption,      // A set option is unknown or not honored; never silently dropped
    SessionBusy,            // Session already has a live request
};

enum class RequestType { Generate, Embedding };

enum class EventType {
    Token,
    Embedding,
    Stop,
    Error,
};

enum class TokenChannel { Content, Reasoning };

/*
 * Engine-wide configuration, fixed for the life of one engine instance.
 */
struct ChorusConfig {
    InitialModelSpec model;
    ProviderOptionMap provider_options;     // Engine-wide options, namespaced: provider_options["llama"]
    LogLevel log_level = log_level_default; // least severe level worth reporting
};

/// An individual turn of a conversation.
struct ChatMessage {
    std::string role;
    std::string content;
};

/// A provider's render of a conversation (see, e.g., InferenceEngine::render_chat_prompt).
struct RenderedPrompt {
    std::string text;
    int32_t token_count = 0;
};

/// Why an engine refused a request up front (see InferenceEngine::validate_request).
struct RequestRejection {
    ChorusError error = ChorusError::Unknown;
    std::string message;
};

/*
 * A message spliced into a conversation at a fixed distance from its end.
 *
 * depth == 0 lands after the last message, depth == N lands N messages earlier.
 */
struct InjectedMessage {
    ChatMessage message;
    int32_t depth = 0;
};

/*
 * One event in a request's lifetime.
 *
 * A request might emit any number of Token or Embedding signals.
 * It will only emit one terminal signal: Stop or Error.
 */
struct ChorusSignal {
    RequestId request_id;
    EventType type;
    TokenChannel channel = TokenChannel::Content;

    ChorusError error_code = ChorusError::None;

    std::string text; // Token: the token text; Error: the message
    std::vector<float> embedding;

    bool is_error() const { return type == EventType::Error; }
    bool is_embedding() const { return type == EventType::Embedding; }
};

/*
 * One unit of work handed to an engine.
 *
 * A request is a raw prompt or a chat. When messages is non-empty the provider
 * renders it through a chat template and prompt is ignored. Every accepted request
 * is guaranteed to emit one terminal signal, either Stop or Error.
 */
struct ChorusRequest {
    RequestId id;
    // The conversation this request continues; unset for stateless requests.
    std::optional<SessionId> session_id;

    RequestType type = RequestType::Generate;
    std::string prompt;

    int priority = 0;

    std::vector<ChatMessage> messages;
    std::string chat_template; // Overrides the model's embedded template when non-empty.

    GenerationConfig gen_config;

    // Receives every signal for this request, possibly from an engine worker thread;
    // must be thread-safe.
    std::function<void(ChorusSignal&)> on_event;
};
} // namespace Chorus

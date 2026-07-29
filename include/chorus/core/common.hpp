#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "chorus/core/generation_config.hpp"
#include "chorus/core/log.hpp"
#include "chorus/core/model_spec.hpp"
#include "chorus/core/options.hpp"

namespace Chorus {

using RequestId = int64_t;
using SessionId = std::string;

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

/*
 * Engine-wide configuration, fixed for the life of one engine instance.
 */
struct ChorusConfig {
    InitialModelSpec model;
    OptionMap provider_options; // Engine-wide options, namespaced: provider_options["llama"]
    LogCallback log_callback;   // Optional; falls back to stderr if not set
};

/// An individual turn of a conversation.
struct ChatMessage {
    std::string role;
    std::string content;
};

/*
 * A message spliced into a conversation at a fixed distance from its end.
 *
 * depth == 0 lands after the last message, depth == N lands N messages earlier.
 * @todo this is a code smell. It should live deeper in the arch.
 */
struct InjectedMessage {
    ChatMessage message;
    int32_t depth = 0;
};

enum class EventType {
    Token,
    Embedding,
    Stop,
    Error,
};

enum class TokenChannel { Content, Reasoning };

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

enum class RequestType { Generate, Embedding };

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

    // Priority-queue ordering: higher priority is served first.
    bool operator<(const ChorusRequest& other) const { return priority < other.priority; }
};
} // namespace Chorus

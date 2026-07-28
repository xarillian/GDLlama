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

// ---- Errors -----------------------------------------------------------------

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
    UnsupportedModelFormat, // Backend cannot load the requested artifact format
    UnsupportedFeature,     // Requested capability (constraint, modality, ...) not supported
    UnsupportedOption,      // A set option is unknown or not honored; never silently dropped
    SessionBusy,            // Session already has a live request
};

// Everything needed to bring an engine up: the model artifact to load and the
// provider-specific options to load it with.
struct ChorusConfig {
    ModelSpec model;
    OptionMap backend_options; // Engine-wide options, namespaced: backend_options["llama"]
    LogCallback log_callback;  // Optional; falls back to stderr if not set
};

// An individual turn of a conversation.
struct ChatMessage {
    std::string role;
    std::string content;
};

// A message spliced into a conversation at a fixed distance from its end.
// depth == 0 lands after the last message, depth == N lands N messages earlier.
// @todo this is a code smell. It should live deeper in the arch.
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

// A single event in a request's lifetime.
struct ChorusSignal {
    RequestId request_id;
    EventType type;
    TokenChannel channel = TokenChannel::Content;

    ChorusError error_code = ChorusError::None;

    std::string text;
    std::vector<float> embedding;

    bool is_error() const { return type == EventType::Error; }
    bool is_embedding() const { return type == EventType::Embedding; }
};

// ---- Requests ---------------------------------------------------------------

enum class RequestType { Generate, Embedding };

// One unit of work handed to an engine.
struct ChorusRequest {
    RequestId id;
    std::optional<SessionId> session_id;

    RequestType type = RequestType::Generate;
    std::string prompt;

    int priority = 0;

    std::vector<ChatMessage> messages;
    std::string chat_template;
    GenerationConfig gen_config;

    std::function<void(ChorusSignal&)> on_event;

    // Priority-queue ordering: higher priority is served first.
    bool operator<(const ChorusRequest& other) const { return priority < other.priority; }
};
} // namespace Chorus

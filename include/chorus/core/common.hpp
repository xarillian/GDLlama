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

enum class ChorusError {
    None,                   // No error
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
    Unknown,                // Catch-all for unexpected failures
};

struct ChorusConfig {
    ModelSpec model;
    OptionMap backend_options; // engine-wide options, namespaced: backend_options["llama"]
    LogCallback log_callback;  // optional; falls back to stderr if not set
};

enum class EventType {
    Token,
    Embedding,
    Stop,
    Error,
};

enum class RequestType { Generate, Embedding };

struct ChorusSignal {
    int64_t request_id;
    EventType type;
    ChorusError error_code = ChorusError::None;

    std::string text;
    std::vector<float> embedding;

    bool is_error() const { return type == EventType::Error; }
    bool is_embedding() const { return type == EventType::Embedding; }
};

using RequestId = int64_t;
using SessionId = std::string;

struct ChorusRequest {
    int64_t id;
    int priority = 0;

    // Caller-owned continuity lane. Engines must preserve it as request
    // identity; they may use it only per declared native_sessions capability.
    std::optional<SessionId> session_id;

    RequestType type = RequestType::Generate; // Replaces 'bool is_embedding'

    std::string prompt;
    GenerationConfig gen_config;

    std::function<void(ChorusSignal&)> on_event;

    bool operator<(const ChorusRequest& other) const { return priority < other.priority; }
};
} // namespace Chorus

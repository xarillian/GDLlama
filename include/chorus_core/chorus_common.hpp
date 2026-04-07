#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace Chorus {

struct ChorusConfig {
    std::string model_path;
    int32_t context_size = 2048;   // llama.cpp's original default
    int32_t thread_count = 4;      // conservative enough to avoid over-subscribing most machines out of the box
    bool use_gpu = true;           // GPU is almost always faster; safer to opt-out than opt-in
    int32_t gpu_layers = 99;       // use all layers on GPU by default
    int32_t num_slots = 1;         // matches llama.cpp's n_seq_max default; increase for concurrent requests
    int32_t tokens_per_tick = 512; // large enough for throughput, small enough not to stall a game tick
};

struct GenerationConfig {
    int32_t max_tokens = 128;    // -1 for infinite
    float temperature = 0.8f;    // less boring than 1.0, hopefully customized often by users
    int32_t top_k = 40;          // llama.cpp default
    float top_p = 0.95f;         // llama.cpp default
    float repeat_penalty = 1.1f; // light penalty to discourage loops without distorting the distribution much
    uint32_t seed = 1337;        // -1 for random

    std::string grammar; // GBNF grammar string for constrained output
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

    std::string text;
    std::vector<float> embedding;

    bool is_error() const { return type == EventType::Error; }
    bool is_embedding() const { return type == EventType::Embedding; }
};

struct ChorusRequest {
    int64_t id;
    int priority = 0;

    RequestType type = RequestType::Generate; // Replaces 'bool is_embedding'

    std::string prompt;
    GenerationConfig gen_config;

    std::function<void(ChorusSignal&)> on_event;

    bool operator<(const ChorusRequest& other) const { return priority < other.priority; }
};
} // namespace Chorus

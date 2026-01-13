#pragma once

#include <functional>
#include <string>

namespace Chorus {

    struct ChorusConfig {
        std::string model_path;
        int32_t context_size = 2048;
        int32_t thread_count = 4;
        bool use_gpu = true;
        int32_t gpu_layers = 99; // Use all layers on GPU by default
    };

    struct GenerationConfig {
        int32_t max_tokens = 128;  // -1 for infinite
        float temperature = 0.8f;
        int32_t top_k = 40;
        float top_p = 0.95f;
        float repeat_penalty = 1.1f;
        uint32_t seed = 1337;  // -1 for random

        std::string grammar;  // GBNF grammar string for constrained output
    };

    enum class EventType {
        Token, 
        Embedding,
        Stop,
        Error,
    };

    enum class RequestType {
        Generate,
        Embedding
    };

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

        bool operator<(const ChorusRequest& other) const {
            return priority < other.priority; 
        }
    };
}


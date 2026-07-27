#pragma once

#include "chorus/core/common.hpp"

#include <cstdint>

namespace Chorus::GodotAdapter {

inline constexpr int32_t DEFAULT_GPU_LAYERS = -1;
inline constexpr const char* GPU_LAYERS_PROPERTY_HINT = "-1,999,1";

struct LlamaLoadOptions {
    int32_t context_size = 2048;
    int32_t thread_count = 4;
    bool use_gpu = true;
    int32_t gpu_layers = DEFAULT_GPU_LAYERS;
    int32_t num_slots = 1;
    int32_t tokens_per_tick = 512;
    int32_t n_batch = 2048;
    int32_t n_ubatch = 512;
    int32_t main_gpu = 0;
};

inline OptionMap make_llama_load_options(const LlamaLoadOptions& properties) {
    OptionMap options{
        {"context_size", int64_t{properties.context_size}},
        {"thread_count", int64_t{properties.thread_count}},
        {"use_gpu", properties.use_gpu},
        {"num_slots", int64_t{properties.num_slots}},
        {"tokens_per_tick", int64_t{properties.tokens_per_tick}},
        {"n_batch", int64_t{properties.n_batch}},
        {"n_ubatch", int64_t{properties.n_ubatch}},
    };
    if (properties.use_gpu) {
        options["gpu_layers"] = int64_t{properties.gpu_layers};
        options["main_gpu"] = int64_t{properties.main_gpu};
    }
    return options;
}

} // namespace Chorus::GodotAdapter

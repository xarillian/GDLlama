#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"

#include <cstdint>
#include <string>
#include <variant>

#include <llama.h>

namespace Chorus {

struct LlamaLoadConfig {
    std::string weights_path;
    uint32_t context_size = 2048;
    int32_t thread_count = 4;
    bool use_gpu = true;
    int32_t gpu_layers = 99;
    uint32_t num_slots = 1;
    int32_t tokens_per_tick = 512;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
    int32_t main_gpu = 0;
};

std::variant<LlamaLoadConfig, RequestRejection> parse_llama_load_config(const ChorusConfig& config);
llama_model_params make_llama_model_params(const LlamaLoadConfig& config);
llama_context_params make_llama_context_params(const LlamaLoadConfig& config);

} // namespace Chorus

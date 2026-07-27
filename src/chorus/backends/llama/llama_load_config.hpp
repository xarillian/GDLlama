#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <variant>

#include <llama.h>

namespace Chorus {

/**
 * @brief Configures model loading, device placement, and inference capacity.
 *
 * The context window is shared by all slots. Disabling GPU use guarantees
 * CPU-only execution. The physical batch size must not exceed the logical
 * batch size.
 */
struct LlamaLoadConfig {
    // Path to the GGUF model file.
    std::string weights_path;

    // Total context window in tokens, shared by all slots.
    uint32_t context_size = 2048;

    // Number of CPU threads available to inference.
    int32_t thread_count = 4;

    // Whether GPU acceleration is allowed. False guarantees CPU-only execution.
    bool use_gpu = true;

    // Model layers assigned to the GPU. Zero means none; a negative value means all.
    int32_t gpu_layers = -1;

    // Whether gpu_layers is a requested value rather than a default.
    bool gpu_layers_explicit = false;

    // Maximum number of concurrent conversations.
    uint32_t num_slots = 1;

    // Maximum prompt tokens consumed from each active conversation per scheduler pass.
    int32_t tokens_per_tick = 512;

    // Maximum total tokens combined into one inference batch.
    uint32_t n_batch = 2048;

    // Maximum physical sub-batch size. Must not exceed n_batch.
    uint32_t n_ubatch = 512;

    // Requested zero-based primary GPU index.
    int32_t main_gpu = 0;

    // Whether main_gpu is a requested value rather than a default.
    bool main_gpu_explicit = false;
};

/**
 * @brief Stores the empty, null-terminated device list used by CPU-only loads.
 *
 * Value-initialize this array so its sole element is nullptr. llama.cpp treats
 * that list as "use no offload devices"; the storage must survive the model-load call.
 */
using LlamaOffloadDeviceList = std::array<ggml_backend_dev_t, 1>;

std::variant<LlamaLoadConfig, RequestRejection> parse_llama_load_config(const ChorusConfig& config);
llama_model_params make_llama_model_params(const LlamaLoadConfig& config, LlamaOffloadDeviceList& no_offload_devices);
llama_context_params make_llama_context_params(const LlamaLoadConfig& config);

} // namespace Chorus

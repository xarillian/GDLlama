#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <variant>

#include <llama.h>

namespace Chorus {

struct LlamaLoadConfig {
    std::string weights_path;         // Path to the GGUF model file.
    uint32_t context_size = 2048;     // Total context window in tokens, shared by all slots.
    int32_t thread_count = 4;         // Number of CPU threads available to inference.
    bool use_gpu = true;              // Whether GPU acceleration is allowed. False guarantees CPU-only execution.
    int32_t gpu_layers = -1;          // Model layers assigned to the GPU. Zero means none; a negative value means all.
    bool gpu_layers_explicit = false; // Whether gpu_layers is a requested value rather than a default.
    uint32_t num_slots = 1;           // Maximum number of concurrent conversations.
    int32_t tokens_per_tick = 512;  // Maximum prompt tokens consumed from each active conversation per scheduler pass.
    uint32_t n_batch = 2048;        // Maximum total tokens combined into one inference batch.
    uint32_t n_ubatch = 512;        // Maximum physical sub-batch size. Must not exceed n_batch.
    int32_t main_gpu = 0;           // Requested zero-based primary GPU index.
    bool main_gpu_explicit = false; // Whether main_gpu is a requested value rather than a default.
};

/*
 * Provides stable storage for CPU-only offload device selection.
 *
 * A value-initialized `Chorus::LlamaOffloadDeviceList` contains one null
 * device. That list tells `::llama_model_load_from_file` not to use an offload
 * device, and its storage must survive the model-load call.
 */
using LlamaOffloadDeviceList = std::array<ggml_backend_dev_t, 1>;

/*
 * Declares llama load options for host rendering and parser validation.
 *
 * Defaults come from a default-constructed `Chorus::LlamaLoadConfig`, keeping
 * its member initializers as the single source of truth. For every supplied
 * option, the parser requires a matching descriptor, value type, and
 * implementation binding.
 */
const ProviderOptionDescriptors& llama_load_option_descriptors();

/*
 * Parses provider configuration into normalized llama load settings.
 *
 * Validation failures are returned as `Chorus::RequestRejection`.
 *
 * Returns:
 *  - `Chorus::LlamaLoadConfig`: the validated load settings.
 *  - `Chorus::RequestRejection`: the reason the configuration was rejected.
 *
 * Errors:
 *  - `Chorus::ChorusError::InvalidRequest`: the model has no weights asset.
 *  - `Chorus::ChorusError::UnsupportedOption`: an asset or option is unsupported.
 */
std::variant<LlamaLoadConfig, RequestRejection> parse_llama_load_config(const ChorusConfig& config);

/*
 * Builds llama.cpp model parameters from normalized load settings.
 *
 * During CPU-only execution, the returned `::llama_model_params::devices`
 * borrows `no_offload_devices`; its storage must survive the model-load call.
 */
llama_model_params make_llama_model_params(const LlamaLoadConfig& config, LlamaOffloadDeviceList& no_offload_devices);

/*
 * Builds llama.cpp context parameters from normalized load settings.
 */
llama_context_params make_llama_context_params(const LlamaLoadConfig& config);

} // namespace Chorus

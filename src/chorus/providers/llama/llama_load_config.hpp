#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <variant>

#include <llama.h>

namespace Chorus {

/*
 * Returns the default CPU thread count: four, or fewer on machines with fewer cores.
 *
 * ggml waits for every thread at each step of a decode, and without OpenMP the waiting threads
 * spin. A thread without a core of its own then stalls every step; one extra thread made
 * generation roughly 90 times slower.
 */
int32_t llama_default_thread_count();

struct LlamaLoadConfig {
    std::string weights_path;     // Path to the GGUF model file.
    uint32_t context_size = 2048; // Total context window in tokens, divided among active sequences.
    int32_t thread_count = llama_default_thread_count(); // Number of CPU threads available to inference.
    bool use_gpu = true;              // Whether GPU acceleration is allowed. False guarantees CPU-only execution.
    int32_t gpu_layers = -1;          // Model layers assigned to the GPU. Zero means none; a negative value means all.
    bool gpu_layers_explicit = false; // Whether gpu_layers is a requested value rather than a default.
    uint32_t max_concurrent_requests = 1;
    uint32_t n_batch = 2048;        // Maximum total tokens combined into one inference batch.
    uint32_t n_ubatch = 512;        // Maximum physical sub-batch size. Must not exceed n_batch.
    int32_t main_gpu = 0;           // Requested zero-based primary GPU index.
    bool main_gpu_explicit = false; // Whether main_gpu is a requested value rather than a default.
    enum llama_pooling_type pooling = LLAMA_POOLING_TYPE_UNSPECIFIED;
    bool embeddings = false; // Whether a generation model also serves embedding requests.
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
 * Reports whether an engine for `model` serves embedding requests.
 *
 * Embedding models always do: those that cannot decode and those whose GGUF declares a
 * pooling type. A generation model does only when `Chorus::LlamaLoadConfig::embeddings`
 * turns it on.
 */
bool llama_model_serves_embeddings(const llama_model* model, const LlamaLoadConfig& config);

/*
 * Builds llama.cpp context parameters from normalized load settings.
 *
 * llama.cpp outputs every token of an embedding batch. Without embeddings, each sequence
 * outputs at most one token per batch, and the reserved logits shrink to match.
 */
llama_context_params make_llama_context_params(const LlamaLoadConfig& config, bool serves_embeddings);

} // namespace Chorus

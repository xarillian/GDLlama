#include "chorus/backends/llama/llama_load_config.hpp"

#include <limits>
#include <utility>

namespace Chorus {
namespace {

RequestRejection unsupported(std::string message) {
    return RequestRejection{ChorusError::UnsupportedOption, std::move(message)};
}

const int64_t* require_int64(const std::string& key, const OptionValue& value, RequestRejection& rejection) {
    const auto* integer = std::get_if<int64_t>(&value);
    if (!integer)
        rejection = unsupported("Llama load option '" + key + "' must be an int64.");
    return integer;
}

} // namespace

std::variant<LlamaLoadConfig, RequestRejection> parse_llama_load_config(const ChorusConfig& config) {
    LlamaLoadConfig out;
    for (const auto& asset : config.model.assets) {
        if (asset.role == "weights") {
            out.weights_path = asset.location;
        } else {
            return unsupported("LlamaEngine does not use asset role '" + asset.role + "'");
        }
    }
    if (out.weights_path.empty())
        return RequestRejection{ChorusError::InvalidRequest, "ModelSpec has no 'weights' asset."};
    if (!config.model.backend_options.empty())
        return unsupported("LlamaEngine defines no artifact-scoped model options.");

    for (const auto& [ns, value] : config.backend_options) {
        if (ns != "llama")
            return unsupported("Unknown option namespace '" + ns + "'");
        const auto* opts = std::get_if<OptionMap>(&value);
        if (!opts)
            return unsupported("'llama' options must be a map.");

        for (const auto& [key, option] : *opts) {
            const auto* as_int = std::get_if<int64_t>(&option);
            const auto* as_bool = std::get_if<bool>(&option);
            if (key == "context_size" && as_int)
                out.context_size = static_cast<uint32_t>(*as_int);
            else if (key == "thread_count" && as_int)
                out.thread_count = static_cast<int32_t>(*as_int);
            else if (key == "use_gpu" && as_bool)
                out.use_gpu = *as_bool;
            else if (key == "gpu_layers" && as_int) {
                out.gpu_layers = static_cast<int32_t>(*as_int);
                out.gpu_layers_explicit = true;
            } else if (key == "num_slots" && as_int)
                out.num_slots = static_cast<uint32_t>(*as_int);
            else if (key == "tokens_per_tick" && as_int)
                out.tokens_per_tick = static_cast<int32_t>(*as_int);
            else if (key == "n_batch") {
                RequestRejection rejection;
                as_int = require_int64(key, option, rejection);
                if (!as_int)
                    return rejection;
                if (*as_int <= 0)
                    return unsupported("Llama load option 'n_batch' must be greater than zero.");
                if (*as_int > std::numeric_limits<int32_t>::max())
                    return unsupported("Llama load option 'n_batch' does not fit llama_batch_init's capacity.");
                out.n_batch = static_cast<uint32_t>(*as_int);
            } else if (key == "n_ubatch") {
                RequestRejection rejection;
                as_int = require_int64(key, option, rejection);
                if (!as_int)
                    return rejection;
                if (*as_int <= 0)
                    return unsupported("Llama load option 'n_ubatch' must be greater than zero.");
                if (static_cast<uint64_t>(*as_int) > std::numeric_limits<uint32_t>::max())
                    return unsupported("Llama load option 'n_ubatch' does not fit llama_context_params::n_ubatch.");
                out.n_ubatch = static_cast<uint32_t>(*as_int);
            } else if (key == "main_gpu") {
                RequestRejection rejection;
                as_int = require_int64(key, option, rejection);
                if (!as_int)
                    return rejection;
                if (*as_int < 0)
                    return unsupported("Llama load option 'main_gpu' must not be negative.");
                if (*as_int > std::numeric_limits<int32_t>::max())
                    return unsupported("Llama load option 'main_gpu' does not fit llama_model_params::main_gpu.");
                out.main_gpu = static_cast<int32_t>(*as_int);
                out.main_gpu_explicit = true;
            } else {
                return unsupported("Unknown or mistyped llama load option '" + key + "'");
            }
        }
    }

    if (out.n_ubatch > out.n_batch)
        return unsupported("Llama load option 'n_ubatch' must not exceed 'n_batch'.");
    if (!out.use_gpu && out.gpu_layers_explicit && out.gpu_layers != 0)
        return unsupported("Llama load option 'gpu_layers' must be zero when 'use_gpu' is false.");
    if (!out.use_gpu && out.main_gpu_explicit)
        return unsupported("Llama load option 'main_gpu' cannot be set when 'use_gpu' is false.");

    return out;
}

llama_model_params make_llama_model_params(const LlamaLoadConfig& config, LlamaOffloadDeviceList& no_offload_devices) {
    llama_model_params params = llama_model_default_params();
    params.devices = config.use_gpu ? nullptr : no_offload_devices.data();
    params.n_gpu_layers = config.use_gpu ? config.gpu_layers : 0;
    params.main_gpu = config.main_gpu;
    return params;
}

llama_context_params make_llama_context_params(const LlamaLoadConfig& config) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = config.context_size;
    params.n_seq_max = config.num_slots;
    params.n_threads = config.thread_count;
    params.n_threads_batch = config.thread_count;
    params.n_batch = config.n_batch;
    params.n_ubatch = config.n_ubatch;
    params.offload_kqv = config.use_gpu;
    params.op_offload = config.use_gpu;
    return params;
}

} // namespace Chorus

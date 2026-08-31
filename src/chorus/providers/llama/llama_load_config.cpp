#include "chorus/providers/llama/llama_load_config.hpp"

#include <limits>
#include <optional>
#include <utility>

namespace Chorus {
namespace {

RequestRejection unsupported(std::string message) {
    return RequestRejection{ChorusError::UnsupportedOption, std::move(message)};
}

const char* value_shape(const ProviderOptionValue& value) {
    if (std::holds_alternative<bool>(value))
        return "a bool";
    if (std::holds_alternative<int64_t>(value))
        return "an int64";
    if (std::holds_alternative<double>(value))
        return "a double";
    if (std::holds_alternative<std::string>(value))
        return "a string";
    if (std::holds_alternative<ProviderOptionList>(value))
        return "a list";
    return "a map";
}

std::optional<RequestRejection> check_against_schema(const std::string& key, const ProviderOptionValue& value) {
    const auto* descriptor = find_option_descriptor(llama_load_option_descriptors(), key);
    if (!descriptor)
        return unsupported("Unknown llama load option '" + key + "'");
    if (value.index() != descriptor->default_value.index())
        return unsupported("Llama load option '" + key + "' must be " + value_shape(descriptor->default_value) + ".");
    return std::nullopt;
}

std::optional<RequestRejection> apply_load_option(
        LlamaLoadConfig& config, const std::string& key, const ProviderOptionValue& option) {
    if (auto rejection = check_against_schema(key, option))
        return rejection;

    const auto* as_int = std::get_if<int64_t>(&option);
    const auto* as_bool = std::get_if<bool>(&option);
    if (key == "context_size" && as_int) {
        if (*as_int <= 0 || static_cast<uint64_t>(*as_int) > std::numeric_limits<uint32_t>::max())
            return unsupported("Llama load option 'context_size' must be a positive uint32.");
        config.context_size = static_cast<uint32_t>(*as_int);
    } else if (key == "thread_count" && as_int) {
        if (*as_int <= 0 || *as_int > std::numeric_limits<int32_t>::max())
            return unsupported("Llama load option 'thread_count' must be a positive int32.");
        config.thread_count = static_cast<int32_t>(*as_int);
    } else if (key == "use_gpu" && as_bool) {
        config.use_gpu = *as_bool;
    } else if (key == "gpu_layers" && as_int) {
        if (*as_int < -1 || *as_int > std::numeric_limits<int32_t>::max())
            return unsupported("Llama load option 'gpu_layers' must fit an int32 and be -1 or greater.");
        config.gpu_layers = static_cast<int32_t>(*as_int);
        config.gpu_layers_explicit = true;
    } else if (key == "num_slots" && as_int) {
        if (*as_int <= 0 || static_cast<uint64_t>(*as_int) > std::numeric_limits<uint32_t>::max())
            return unsupported("Llama load option 'num_slots' must be a positive uint32.");
        config.num_slots = static_cast<uint32_t>(*as_int);
    } else if (key == "tokens_per_tick" && as_int) {
        if (*as_int <= 0 || *as_int > std::numeric_limits<int32_t>::max())
            return unsupported("Llama load option 'tokens_per_tick' must be a positive int32.");
        config.tokens_per_tick = static_cast<int32_t>(*as_int);
    } else if (key == "n_batch" && as_int) {
        if (*as_int <= 0)
            return unsupported("Llama load option 'n_batch' must be greater than zero.");
        if (*as_int > std::numeric_limits<int32_t>::max())
            return unsupported("Llama load option 'n_batch' does not fit llama_batch_init's capacity.");
        config.n_batch = static_cast<uint32_t>(*as_int);
    } else if (key == "n_ubatch" && as_int) {
        if (*as_int <= 0)
            return unsupported("Llama load option 'n_ubatch' must be greater than zero.");
        if (static_cast<uint64_t>(*as_int) > std::numeric_limits<uint32_t>::max())
            return unsupported("Llama load option 'n_ubatch' does not fit llama_context_params::n_ubatch.");
        config.n_ubatch = static_cast<uint32_t>(*as_int);
    } else if (key == "main_gpu" && as_int) {
        if (*as_int < 0)
            return unsupported("Llama load option 'main_gpu' must not be negative.");
        if (*as_int > std::numeric_limits<int32_t>::max())
            return unsupported("Llama load option 'main_gpu' does not fit llama_model_params::main_gpu.");
        config.main_gpu = static_cast<int32_t>(*as_int);
        config.main_gpu_explicit = true;
    } else {
        return unsupported("Llama load option '" + key + "' is declared but not applied; this is a bug.");
    }
    return std::nullopt;
}

} // namespace

const ProviderOptionDescriptors& llama_load_option_descriptors() {
    static const ProviderOptionDescriptors descriptors = [] {
        const LlamaLoadConfig d{};
        return ProviderOptionDescriptors{
            {"context_size",
             "Context Size",
             "Total context window in tokens, shared by all slots.",
             int64_t{d.context_size},
             128,
             65536,
             128,
             std::nullopt},
            {"thread_count",
             "Thread Count",
             "CPU threads available to inference.",
             int64_t{d.thread_count},
             1,
             32,
             1,
             std::nullopt},
            {"use_gpu",
             "Use GPU",
             "Allow GPU acceleration. Off guarantees CPU-only execution.",
             d.use_gpu,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt},
            {"gpu_layers",
             "GPU Layers",
             "Model layers offloaded to the GPU. Zero means none, -1 means all.",
             int64_t{d.gpu_layers},
             -1,
             999,
             1,
             "use_gpu"},
            {"num_slots",
             "Slots",
             "Maximum number of concurrent conversations.",
             int64_t{d.num_slots},
             1,
             32,
             1,
             std::nullopt},
            {"tokens_per_tick",
             "Tokens Per Tick",
             "Prompt tokens consumed from each active conversation per scheduler pass.",
             int64_t{d.tokens_per_tick},
             1,
             4096,
             1,
             std::nullopt},
            // Widget bounds cannot depend on sibling values, so each bound uses
            // the other option's default. The parser validates the final pair.
            {"n_batch",
             "Batch Size",
             "Maximum total tokens combined into one inference batch.",
             int64_t{d.n_batch},
             int64_t{d.n_ubatch},
             65536,
             1,
             std::nullopt},
            {"n_ubatch",
             "Micro-Batch Size",
             "Maximum physical sub-batch size. Must not exceed the batch size.",
             int64_t{d.n_ubatch},
             1,
             int64_t{d.n_batch},
             1,
             std::nullopt},
            {"main_gpu", "Main GPU", "Zero-based index of the primary GPU.", int64_t{d.main_gpu}, 0, 15, 1, "use_gpu"},
        };
    }();
    return descriptors;
}

std::variant<LlamaLoadConfig, RequestRejection> parse_llama_load_config(const ChorusConfig& config) {
    LlamaLoadConfig out;
    for (const auto& asset : config.model.assets) {
        if (asset.role == AssetRole::Weights) {
            out.weights_path = asset.source;
        } else {
            return unsupported("LlamaEngine only uses the 'weights' asset role.");
        }
    }
    if (out.weights_path.empty())
        return RequestRejection{ChorusError::InvalidRequest, "InitialModelSpec has no 'weights' asset."};
    if (!config.model.provider_options.empty())
        return unsupported("LlamaEngine defines no artifact-scoped model options.");

    for (const auto& [ns, value] : config.provider_options) {
        if (ns != "llama")
            return unsupported("Unknown option namespace '" + ns + "'");
        const auto* opts = std::get_if<ProviderOptionMap>(&value);
        if (!opts)
            return unsupported("'llama' options must be a map.");

        for (const auto& [key, option] : *opts) {
            if (auto rejection = apply_load_option(out, key, option))
                return *rejection;
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

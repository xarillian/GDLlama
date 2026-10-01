#include "chorus/providers/llama/llama_load_config.hpp"
#include "chorus/providers/llama/llama_utils.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <optional>
#include <string_view>
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

template <typename Target>
std::optional<RequestRejection> assign_bounded_integer(
    Target& destination,
    int64_t value,
    int64_t minimum,
    int64_t maximum,
    const char* below_minimum_message,
    const char* above_maximum_message
) {
    if (value < minimum)
        return unsupported(below_minimum_message);
    if (value > maximum)
        return unsupported(above_maximum_message);
    destination = static_cast<Target>(value);
    return std::nullopt;
}

template <typename Target>
std::optional<RequestRejection> assign_bounded_integer(
    Target& destination,
    int64_t value,
    int64_t minimum,
    int64_t maximum,
    const char* rejection_message
) {
    return assign_bounded_integer(
        destination, value, minimum, maximum, rejection_message, rejection_message
    );
}

std::optional<RequestRejection> apply_context_size(LlamaLoadConfig& config, const ProviderOptionValue& option) {
    return assign_bounded_integer(
        config.context_size,
        std::get<int64_t>(option),
        1,
        std::numeric_limits<uint32_t>::max(),
        "Llama load option 'context_size' must be a positive uint32."
    );
}

std::optional<RequestRejection> apply_thread_count(LlamaLoadConfig& config, const ProviderOptionValue& option) {
    return assign_bounded_integer(
        config.thread_count,
        std::get<int64_t>(option),
        1,
        std::numeric_limits<int32_t>::max(),
        "Llama load option 'thread_count' must be a positive int32."
    );
}

std::optional<RequestRejection> apply_use_gpu(LlamaLoadConfig& config, const ProviderOptionValue& option) {
    config.use_gpu = std::get<bool>(option);
    return std::nullopt;
}

std::optional<RequestRejection> apply_gpu_layers(LlamaLoadConfig& config, const ProviderOptionValue& option) {
    auto rejection = assign_bounded_integer(
        config.gpu_layers,
        std::get<int64_t>(option),
        -1,
        std::numeric_limits<int32_t>::max(),
        "Llama load option 'gpu_layers' must fit an int32 and be -1 or greater."
    );
    if (!rejection)
        config.gpu_layers_explicit = true;
    return rejection;
}

std::optional<RequestRejection> apply_max_concurrent_requests(
    LlamaLoadConfig& config, const ProviderOptionValue& option
) {
    return assign_bounded_integer(
        config.max_concurrent_requests,
        std::get<int64_t>(option),
        1,
        std::numeric_limits<uint32_t>::max(),
        "Llama load option 'max_concurrent_requests' must be a positive uint32."
    );
}

std::optional<RequestRejection> apply_n_batch(LlamaLoadConfig& config, const ProviderOptionValue& option) {
    return assign_bounded_integer(
        config.n_batch,
        std::get<int64_t>(option),
        1,
        std::numeric_limits<int32_t>::max(),
        "Llama load option 'n_batch' must be greater than zero.",
        "Llama load option 'n_batch' does not fit llama_batch_init's capacity."
    );
}

std::optional<RequestRejection> apply_n_ubatch(LlamaLoadConfig& config, const ProviderOptionValue& option) {
    return assign_bounded_integer(
        config.n_ubatch,
        std::get<int64_t>(option),
        1,
        std::numeric_limits<uint32_t>::max(),
        "Llama load option 'n_ubatch' must be greater than zero.",
        "Llama load option 'n_ubatch' does not fit llama_context_params::n_ubatch."
    );
}

std::optional<RequestRejection> apply_pooling(LlamaLoadConfig& config, const ProviderOptionValue& option) {
    const auto& value = std::get<std::string>(option);
    if (value == "model")
        config.pooling = LLAMA_POOLING_TYPE_UNSPECIFIED;
    else if (value == "none")
        config.pooling = LLAMA_POOLING_TYPE_NONE;
    else if (value == "mean")
        config.pooling = LLAMA_POOLING_TYPE_MEAN;
    else if (value == "cls")
        config.pooling = LLAMA_POOLING_TYPE_CLS;
    else if (value == "last")
        config.pooling = LLAMA_POOLING_TYPE_LAST;
    else
        return unsupported("Llama load option 'pooling' must be model, none, mean, cls, or last.");
    return std::nullopt;
}

std::optional<RequestRejection> apply_embeddings(LlamaLoadConfig& config, const ProviderOptionValue& option) {
    config.embeddings = std::get<bool>(option);
    return std::nullopt;
}

std::optional<RequestRejection> apply_main_gpu(LlamaLoadConfig& config, const ProviderOptionValue& option) {
    auto rejection = assign_bounded_integer(
        config.main_gpu,
        std::get<int64_t>(option),
        0,
        std::numeric_limits<int32_t>::max(),
        "Llama load option 'main_gpu' must not be negative.",
        "Llama load option 'main_gpu' does not fit llama_model_params::main_gpu."
    );
    if (!rejection)
        config.main_gpu_explicit = true;
    return rejection;
}

using LoadOptionApplier = std::optional<RequestRejection> (*)(LlamaLoadConfig&, const ProviderOptionValue&);

struct LoadOptionBinding {
    std::string_view key;
    LoadOptionApplier apply;
};

constexpr std::array load_option_bindings{
    LoadOptionBinding{"context_size", apply_context_size},
    LoadOptionBinding{"thread_count", apply_thread_count},
    LoadOptionBinding{"use_gpu", apply_use_gpu},
    LoadOptionBinding{"gpu_layers", apply_gpu_layers},
    LoadOptionBinding{"max_concurrent_requests", apply_max_concurrent_requests},
    LoadOptionBinding{"n_batch", apply_n_batch},
    LoadOptionBinding{"n_ubatch", apply_n_ubatch},
    LoadOptionBinding{"main_gpu", apply_main_gpu},
    LoadOptionBinding{"pooling", apply_pooling},
    LoadOptionBinding{"embeddings", apply_embeddings},
};

std::optional<RequestRejection> apply_load_option(
    LlamaLoadConfig& config,
    const std::string& key,
    const ProviderOptionValue& option
) {
    if (auto rejection = check_against_schema(key, option))
        return rejection;

    const auto binding = std::ranges::find(load_option_bindings, key, &LoadOptionBinding::key);
    if (binding == load_option_bindings.end())
        return unsupported("Llama load option '" + key + "' is declared but not applied; this is a bug.");
    return binding->apply(config, option);
}

} // namespace

const ProviderOptionDescriptors& llama_load_option_descriptors() {
    static const ProviderOptionDescriptors descriptors = [] {
        const LlamaLoadConfig d{};
        return ProviderOptionDescriptors{
            {"context_size",
             "Context Size",
             "Total context window in tokens, divided among concurrent requests.",
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
            {"max_concurrent_requests",
             "Max Concurrent Requests",
             "Maximum generation and embedding requests processed concurrently. One processes requests individually; higher values automatically co-batch compatible work. Additional requests remain queued. Higher values divide Context Size among concurrent requests and may increase resource use. Idle slots keep recent conversations cached, so their next turn processes only new messages. Takes effect on the next load_model().",
             int64_t{d.max_concurrent_requests},
             1,
             32,
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
             std::nullopt,
             {},
             ProviderOptionPresentation::Advanced},
            {"n_ubatch",
             "Micro-Batch Size",
             "Maximum physical sub-batch size, and the most prompt tokens loaded per step while replies stream; "
             "smaller values stream more smoothly. Must not exceed the batch size.",
             int64_t{d.n_ubatch},
             1,
             int64_t{d.n_batch},
             1,
             std::nullopt,
             {},
             ProviderOptionPresentation::Advanced},
            {"main_gpu", "Main GPU", "Zero-based index of the primary GPU.", int64_t{d.main_gpu}, 0, 15, 1, "use_gpu"},
            {"pooling",
             "Pooling",
             "Embedding pooling strategy. Model uses the GGUF metadata.",
             std::string{"model"},
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             {"model", "none", "mean", "cls", "last"}},
            {"embeddings",
             "Embeddings",
             "Also serve embedding requests from a generation model. Off saves the memory those batches need. Embedding models always serve embeddings.",
             d.embeddings,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt},
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

bool llama_model_serves_embeddings(const llama_model* model, const LlamaLoadConfig& config) {
    if (config.embeddings || !llama_model_has_decoder(model))
        return true;
    const std::string architecture = LlamaUtils::model_metadata(model, "general.architecture");
    const std::string declared = LlamaUtils::model_metadata(model, (architecture + ".pooling_type").c_str());
    int pooling = LLAMA_POOLING_TYPE_NONE;
    std::from_chars(declared.data(), declared.data() + declared.size(), pooling);
    return pooling != LLAMA_POOLING_TYPE_NONE && pooling != LLAMA_POOLING_TYPE_UNSPECIFIED;
}

llama_context_params make_llama_context_params(const LlamaLoadConfig& config, bool serves_embeddings) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = config.context_size;
    params.n_seq_max = config.max_concurrent_requests;
    params.n_threads = config.thread_count;
    params.n_threads_batch = config.thread_count;
    params.n_batch = config.n_batch;
    params.n_ubatch = config.n_ubatch;
    params.pooling_type = config.pooling;
    params.n_outputs_max = serves_embeddings ? 0 : config.max_concurrent_requests;
    // Chorus never rewinds or reuses cached tokens, so sliding-window layers need only
    // their window. Prefix reuse would need the full cache back.
    params.swa_full = false;
    params.offload_kqv = config.use_gpu;
    params.op_offload = config.use_gpu;
    return params;
}

} // namespace Chorus

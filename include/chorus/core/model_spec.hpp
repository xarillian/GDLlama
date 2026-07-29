#pragma once

#include "chorus/core/provider_option_value.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Chorus {

enum class Modality {
    Text,
    Image,
    Audio,
};

enum class ModelFormat {
    Auto, // format left unstated; the provider resolves it
    Gguf,
    LiteRtLm,
    SafeTensors,
    Remote,
};

enum class AssetRole {
    Weights,
    Projector, // multimodal projector (e.g. mmproj)
    Tokenizer,
    Drafter, // draft model for speculative decoding
    Package, // bundled archive carrying several roles
};

/*
 * One artifact of a model.
 *
 * Models can be sets of artifacts. The provider loads them all and uses them
 * according to their role.
 */
struct ModelAsset {
    AssetRole role = AssetRole::Weights;
    std::string source; // local path, URI, repository reference, provider identifier
};

/*
 * The model an engine boots with.
 *
 * "Initial" is a lifetime rule. An engine's base model is fixed at boot time and
 * cannot be changed without restarting the engine. Loading a different model means
 * building a different engine, which keeps two models from ever being resident at once.
 */
struct InitialModelSpec {
    std::string model_id; // Caller-selected model name, e.g. "gemma3 270M F16"
    ModelFormat format = ModelFormat::Auto;
    std::vector<ModelAsset> assets;
    ProviderOptionMap provider_options;
};

/*
 * Information about a model loaded into memory.
 *
 * The engine fills this post-initialization, and it is held for the engine's lifetime.
 * Fields state observed facts instead of requested configuration. For example, `format` is
 * always concrete, never `ModelFormat::Auto`. Optional fields are absent when the provider
 * cannot determine them.
 */
struct LoadedModelInfo {
    std::string model_id;
    std::string family; // architecture family, e.g. "gemma3"
    ModelFormat format = ModelFormat::Auto;
    std::string quantization; // display string, e.g. "gemma3 270M F16"

    std::vector<Modality> input_modalities;
    std::vector<Modality> output_modalities;

    std::optional<uint32_t> maximum_context;     // the context window the model was trained for
    std::optional<uint32_t> per_request_context; // the context one request may assume
    // weight tensors as loaded, in bytes;
    // the engine's full memory cost adds KV cache and compute buffers
    std::optional<uint64_t> model_bytes;
};

} // namespace Chorus

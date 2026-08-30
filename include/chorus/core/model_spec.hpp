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
 * Models may comprise multiple artifacts. The provider loads each one according
 * to its `ModelAsset::role`.
 */
struct ModelAsset {
    AssetRole role = AssetRole::Weights;
    std::string source; // local path, URI, repository reference, provider identifier
};

/*
 * The model an engine boots with.
 *
 * "Initial" is a lifetime rule. An engine's base model is fixed at boot time
 * and cannot be changed without restarting the engine. Loading a different
 * model means building a different engine, so two models are never resident at once.
 */
struct InitialModelSpec {
    /// Caller-defined label for this model.
    std::string model_id;
    ModelFormat format = ModelFormat::Auto;
    std::vector<ModelAsset> assets;
    ProviderOptionMap provider_options;
};

/*
 * Information about a model loaded into memory.
 *
 * The engine fills this after initialization and holds it for the engine's
 * lifetime. Fields report observed facts rather than requested configuration.
 * `LoadedModelInfo::format` is always concrete, never `ModelFormat::Auto`.
 * Optional fields are absent when the provider cannot determine them.
 */
struct LoadedModelInfo {
    /// Value copied from `InitialModelSpec::model_id`.
    std::string model_id;
    std::string family;
    ModelFormat format = ModelFormat::Auto;
    /// Provider description of the loaded weights, not necessarily a bare quantization tag.
    std::string quantization;

    std::vector<Modality> input_modalities;
    std::vector<Modality> output_modalities;

    /// Context window the model was trained to support.
    std::optional<uint32_t> maximum_context;
    /// Context capacity available to one request.
    std::optional<uint32_t> per_request_context;
    /// Loaded weight tensors in bytes, excluding KV cache and compute buffers.
    std::optional<uint64_t> model_bytes;
};

} // namespace Chorus

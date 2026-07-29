#pragma once

#include "chorus/core/provider_option_value.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Chorus {

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
    Drafter, // dratft model for speculative decoding
    Package, // bundled archive carrying serveral roles
};

/*
 * One artifact of a model.
 *
 * Models can be sets of artifcts. The provider loads them all and uses them
 * according to their role.
 */
struct ModelAsset {
    AssetRole role = AssetRole::Weights;
    std::string source; // local path, URI, repository reference, provider identifier
    std::optional<uint64_t> declared_size_bytes;
    std::optional<std::string> checksum;
};

/*
 * The model an engine boots with.
 *
 * "Initial" is a lifetime rule. An engine's base model is fixed at boot time and
 * cannot be changed without restarting the engine. Loading a different model means
 * building a different engine, which keeps two models from ever being resident at once.
 */
struct InitialModelSpec {
    std::string model_id; // Caller-selected model name, e.g. "gpt-3.5-turbo"
    ModelFormat format = ModelFormat::Auto;
    std::vector<ModelAsset> assets;
    ProviderOptionMap provider_options;
};

} // namespace Chorus

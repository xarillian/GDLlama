#pragma once

#include "chorus/core/options.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Chorus {

enum class ModelFormat {
    Auto, // request-side wildcard; a populated LoadedModelInfo never reports it
    Gguf,
    LiteRtLm,
    SafeTensors,
    Remote,
};

struct ModelAsset {
    std::string role;     // "weights", "projector", "tokenizer", "drafter", "package"
    std::string location; // local path, URI, repository reference, or backend identifier
    std::optional<uint64_t> declared_size_bytes;
    std::optional<std::string> checksum;
};

// A load request, not an unquestioned source of truth: after initialization
// the engine reports what it actually loaded via LoadedModelInfo.
struct ModelSpec {
    std::string model_id; // stable logical name chosen by the developer or asset manager
    ModelFormat format = ModelFormat::Auto;
    std::vector<ModelAsset> assets;
    OptionMap backend_options; // artifact-scoped hints; engine-wide options live on ChorusConfig
};

} // namespace Chorus

#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/model_spec.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Chorus {

enum class Modality {
    Text,
    Image,
    Audio,
    // Embedding is deliberately absent: it is a request type with its own
    // `embeddings` capability flag, not a content modality (spec 3c-C).
};

enum class SchedulingAuthority {
    ChorusManaged,  // Chorus owns batching, prefill, KV, decode cadence
    BackendManaged, // backend runtime owns its own scheduling
    Hybrid,         // Chorus influences coarse budgets only
};

struct EngineCapabilities {
    std::string backend_id;
    std::vector<ModelFormat> model_formats;
    std::vector<Modality> input_modalities;
    std::vector<Modality> output_modalities;
    std::vector<ConstraintFormat> constraint_formats;

    // Conservative default: an engine must opt in to frame-budget claims.
    SchedulingAuthority scheduling = SchedulingAuthority::BackendManaged;

    bool streaming = false;
    bool cancellation = false;
    bool native_sessions = false; // engine keeps per-session native execution state;
                                  // session-id *correlation* is universal, not gated by this
    bool embeddings = false;
    bool speculative_decoding = false;
    bool dynamic_adapters = false;
    bool prompt_rendering = false;

    std::vector<std::string> portable_generation_options;
    std::vector<std::string> backend_generation_options;
};

struct LoadedModelInfo {
    std::string model_id;
    std::string family;
    ModelFormat format = ModelFormat::Auto; // contract: populated info never reports Auto
    std::string quantization;
    std::vector<Modality> input_modalities;
    std::vector<Modality> output_modalities;
    std::optional<uint32_t> maximum_context;
    // Usable context per concurrent request under the current configuration
    // (today's slot model divides n_ctx across num_slots). This -- not
    // maximum_context, which reports the model's training window -- is what
    // prompt fitting budgets against.
    std::optional<uint32_t> per_request_context;
    std::optional<uint64_t> model_bytes;
    bool has_speculative_assets = false;
};

struct RequestRejection {
    ChorusError error = ChorusError::Unknown;
    std::string message;
};

} // namespace Chorus

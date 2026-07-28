#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/model_spec.hpp"
#include "chorus/core/options.hpp"

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

/**
 * @brief One configurable option, declared by the provider that honors it.
 *
 * Configuration metadata belongs to the provider (ARCHITECTURE.md): a host
 * renders its settings surface from these descriptors rather than re-encoding
 * option names, defaults, and ranges it would then have to keep in sync.
 *
 * `default_value` doubles as the type declaration -- the alternative it holds
 * is the only value shape the option accepts -- so every option must name the
 * value it resolves to when absent.
 *
 * `minimum`, `maximum`, and `step` are host widget bounds, not the provider's
 * validation limits. The provider guarantees the reverse inclusion: every
 * value inside the declared bounds is accepted. A provider may accept more
 * (llama takes any non-negative `main_gpu`, while the useful widget stops at
 * 15); it must never accept less, or a host offers a control that fails on use.
 */
struct OptionDescriptor {
    std::string key; // key within the provider's option namespace
    std::string display_name;
    std::string description;
    OptionValue default_value;
    std::optional<double> minimum;
    std::optional<double> maximum;
    std::optional<double> step;
    // When set: the key of a bool option in the same namespace that must be
    // true for this one to apply. Hosts omit a gated-off option; the provider
    // still rejects it if sent, so the gate is advice, not enforcement.
    std::optional<std::string> enabled_by;
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

    // Options this backend accepts under its own namespace in
    // ChorusConfig::backend_options at load time. Full descriptors rather than
    // names because hosts render configuration surfaces from them; the
    // generation lists above stay names, being per-request rather than
    // configured, with defaults that are backend sampler internals.
    std::vector<OptionDescriptor> load_options;
};

// Resolution over a declared option schema. This lives beside the vocabulary
// rather than in a host because every host resolves a stored option map
// against a provider's declaration identically; a second adapter should
// translate the result, not re-derive it.

const OptionDescriptor*
find_option_descriptor(const std::vector<OptionDescriptor>& descriptors, const std::string& key);

// Whether this option's gate is satisfied, reading the gate from `stored` and
// falling back to its declared default where the caller never set it.
bool option_is_enabled(
    const std::vector<OptionDescriptor>& descriptors, const OptionDescriptor& descriptor, const OptionMap& stored
);

// The map to hand the provider: every declared option at its stored or
// default value, minus the ones their gate switches off.
OptionMap resolve_option_defaults(const std::vector<OptionDescriptor>& descriptors, const OptionMap& stored);

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

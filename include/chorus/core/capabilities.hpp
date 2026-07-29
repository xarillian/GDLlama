#pragma once

#include "chorus/core/generation_config.hpp"
#include "chorus/core/model_spec.hpp"
#include "chorus/core/provider_option_value.hpp"

#include <optional>
#include <string>
#include <vector>

namespace Chorus {

enum class SchedulingAuthority {
    ChorusManaged,
    ProviderManaged,
    Hybrid,
};

/*
 * One knob in a provider's generation config schema.
 *
 * Hosts build their configuration UI from these descriptors instead of hardcoding the knobs
 * per provider. The descriptor is a single source of truth for a knob's name, default, and
 * editor hints.
 */
struct ProviderOptionDescriptor {
    std::string key;
    std::string display_name;
    std::string description;

    ProviderOptionValue default_value;

    std::optional<double> minimum;
    std::optional<double> maximum;
    std::optional<double> step;

    std::optional<std::string> enabled_by;
};

using ProviderOptionDescriptors = std::vector<ProviderOptionDescriptor>;

/*
 * A provider's honest self-description.
 *
 * Consumers adapt to what is declared here instead of branching on provider
 * identity. Pre-init this is the provider's envelope; post-init it narrows to
 * the effective intersection of provider, model, and load configuration.
 */
struct EngineCapabilities {
    std::string provider_id;
    std::vector<ModelFormat> model_formats;
    std::vector<Modality> input_modalities;
    std::vector<Modality> output_modalities;
    std::vector<ConstraintFormat> constraint_formats;

    SchedulingAuthority scheduling = SchedulingAuthority::ProviderManaged;

    bool streaming = false;
    bool cancellation = false;
    bool native_sessions = false;
    bool embeddings = false;
    bool speculative_decoding = false;
    bool dynamic_adapters = false;
    bool prompt_rendering = false;

    std::vector<std::string> common_generation_options;
    std::vector<std::string> provider_generation_options;

    ProviderOptionDescriptors load_options;
};

/// Finds a descriptor by key; nullptr when the schema doesn't declare it.
const ProviderOptionDescriptor* find_option_descriptor(const ProviderOptionDescriptors& schema, const std::string& key);

bool option_is_enabled(
    const ProviderOptionDescriptors& schema, const ProviderOptionDescriptor& descriptor, const ProviderOptionMap& stored
);

ProviderOptionMap resolve_option_defaults(const ProviderOptionDescriptors& schema, const ProviderOptionMap& stored);

} // namespace Chorus

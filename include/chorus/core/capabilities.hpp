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
 * One option in a provider's declared option schema.
 *
 * Hosts build their configuration UI from these descriptors instead of hardcoding the options
 * per provider. The descriptor is a single source of truth for an option's name, default, and
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

    /*
     * Names another option in this same schema that must be on for this one to apply.
     *
     * The named option must be declared bool. This option applies only while that bool
     * resolves true, whether from a configured value or from the named option's own default.
     * Empty in the common case: an option with no prerequisite is always live.
     *
     * For example, llama's `gpu_layers` names `use_gpu`, so a CPU-only load drops the layer
     * count rather than sending a number the provider would reject.
     */
    std::optional<std::string> prerequisite_option;
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

/*
 * Looks up one option in a provider's option schema.
 *
 * Returns:
 *  - A pointer to the descriptor whose key matches
 *  - A nullptr when the schema declares no such key.
 *
 * The pointer borrows the schema's storage and stays valid
 * until the schema is modified or destroyed; callers must not free it.
 */
const ProviderOptionDescriptor*
find_option_descriptor(const ProviderOptionDescriptors& declared_options, const std::string& key);

/*
 * Answers whether the option named as a prerequisite is switched on.
 *
 * An option that names no prerequisite is always live, so this is true for most of a schema.
 * Otherwise the named option must resolve to boolean true, from a configured value if the
 * host set one and from its own declared default if not. A prerequisite the provider never
 * declared, or one declared as some type other than bool, can never be satisfied, and the
 * option naming it stays off.
 *
 * Callers who need to know if an option is enabled should use this function.
 */
bool is_prerequisite_option_enabled(
    const ProviderOptionDescriptors& declared_options,
    const ProviderOptionDescriptor& option,
    const ProviderOptionMap& configured_values
);

/*
 * Flattens a schema and the host's configured values into the map a provider is sent.
 *
 * Every declared option resolves to its configured value, or to its declared default when the
 * host set none. Options whose prerequisite is unmet are dropped, and configured values
 * the schema does not declare are ignored: a provider receives what it declared and nothing
 * else.
 */
ProviderOptionMap
resolve_option_defaults(const ProviderOptionDescriptors& declared_options, const ProviderOptionMap& configured_values);

} // namespace Chorus

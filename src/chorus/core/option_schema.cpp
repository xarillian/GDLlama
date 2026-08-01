#include "chorus/core/capabilities.hpp"

#include <variant>

namespace Chorus {

const ProviderOptionDescriptor*
find_option_descriptor(const ProviderOptionDescriptors& declared_options, const std::string& key) {
    for (const auto& option : declared_options) {
        if (option.key == key)
            return &option;
    }

    return nullptr;
}

bool is_prerequisite_option_enabled(
    const ProviderOptionDescriptors& declared_options,
    const ProviderOptionDescriptor& option,
    const ProviderOptionMap& configured_values
) {
    if (!option.prerequisite_option)
        return true; // an option with no prerequisite is always live

    if (const auto it = configured_values.find(*option.prerequisite_option); it != configured_values.end()) {
        const auto* as_bool = std::get_if<bool>(&it->second);
        return as_bool && *as_bool;
    }
    if (const auto* prerequisite = find_option_descriptor(declared_options, *option.prerequisite_option)) {
        const auto* as_bool = std::get_if<bool>(&prerequisite->default_value);
        return as_bool && *as_bool;
    }

    return false;
}

ProviderOptionMap
resolve_option_defaults(const ProviderOptionDescriptors& declared_options, const ProviderOptionMap& configured_values) {
    ProviderOptionMap resolved;
    for (const auto& option : declared_options) {
        if (!is_prerequisite_option_enabled(declared_options, option, configured_values))
            continue;
        const auto it = configured_values.find(option.key);
        resolved[option.key] = it != configured_values.end() ? it->second : option.default_value;
    }
    return resolved;
}

} // namespace Chorus

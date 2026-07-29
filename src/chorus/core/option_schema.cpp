#include "chorus/core/capabilities.hpp"

#include <variant>

namespace Chorus {

const ProviderOptionDescriptor*
find_option_descriptor(const ProviderOptionDescriptors& schema, const std::string& key) {
    for (const auto& descriptor : schema) {
        if (descriptor.key == key)
            return &descriptor;
    }
    return nullptr;
}

bool option_is_enabled(
    const ProviderOptionDescriptors& schema, const ProviderOptionDescriptor& descriptor, const ProviderOptionMap& stored
) {
    if (!descriptor.enabled_by)
        return true;

    if (const auto it = stored.find(*descriptor.enabled_by); it != stored.end()) {
        const auto* as_bool = std::get_if<bool>(&it->second);
        return as_bool && *as_bool;
    }
    if (const auto* gate = find_option_descriptor(schema, *descriptor.enabled_by)) {
        const auto* as_bool = std::get_if<bool>(&gate->default_value);
        return as_bool && *as_bool;
    }
    return false; // a gate naming an option the provider does not declare
}

ProviderOptionMap resolve_option_defaults(const ProviderOptionDescriptors& schema, const ProviderOptionMap& stored) {
    ProviderOptionMap resolved;
    for (const auto& descriptor : schema) {
        if (!option_is_enabled(schema, descriptor, stored))
            continue;
        const auto it = stored.find(descriptor.key);
        resolved[descriptor.key] = it != stored.end() ? it->second : descriptor.default_value;
    }
    return resolved;
}

} // namespace Chorus

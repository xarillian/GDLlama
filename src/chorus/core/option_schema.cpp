#include "chorus/core/capabilities.hpp"

#include <variant>

namespace Chorus {

const OptionDescriptor*
find_option_descriptor(const std::vector<OptionDescriptor>& descriptors, const std::string& key) {
    for (const auto& descriptor : descriptors) {
        if (descriptor.key == key)
            return &descriptor;
    }
    return nullptr;
}

bool option_is_enabled(
    const std::vector<OptionDescriptor>& descriptors, const OptionDescriptor& descriptor, const OptionMap& stored
) {
    if (!descriptor.enabled_by)
        return true;

    if (const auto it = stored.find(*descriptor.enabled_by); it != stored.end()) {
        const auto* as_bool = std::get_if<bool>(&it->second);
        return as_bool && *as_bool;
    }
    if (const auto* gate = find_option_descriptor(descriptors, *descriptor.enabled_by)) {
        const auto* as_bool = std::get_if<bool>(&gate->default_value);
        return as_bool && *as_bool;
    }
    return false; // a gate naming an option the provider does not declare
}

OptionMap resolve_option_defaults(const std::vector<OptionDescriptor>& descriptors, const OptionMap& stored) {
    OptionMap resolved;
    for (const auto& descriptor : descriptors) {
        if (!option_is_enabled(descriptors, descriptor, stored))
            continue;
        const auto it = stored.find(descriptor.key);
        resolved[descriptor.key] = it != stored.end() ? it->second : descriptor.default_value;
    }
    return resolved;
}

} // namespace Chorus

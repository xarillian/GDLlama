#pragma once

#include <optional>

#include <godot_cpp/core/property_info.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include "chorus/core/capabilities.hpp"
#include "chorus/core/provider_option_value.hpp"

// Godot inspector representations of provider-owned option descriptors and values.
namespace godot_chorus {

/*
 * Builds `godot::PropertyInfo` for `descriptor`.
 *
 * When `enabled` is `false`, the property remains visible but read-only so
 * its prerequisite relationship remains apparent in the inspector.
 */
godot::PropertyInfo property_info_for(const Chorus::ProviderOptionDescriptor& descriptor, bool enabled);

/*
 * Converts a provider option value for the Godot inspector.
 *
 * Returns:
 *  - `godot::Variant`: the corresponding scalar value.
 *  - `godot::Variant::NIL`: `value` is a list or map, which the inspector cannot render faithfully.
 */
godot::Variant option_value_to_variant(const Chorus::ProviderOptionValue& value);

/*
 * Coerces a Godot value to the scalar type declared by `descriptor`.
 *
 * Integer and floating-point options accept either numeric `godot::Variant`
 * type. All other scalar options require an exact type match.
 *
 * Returns:
 *  - `std::optional<Chorus::ProviderOptionValue>`: the coerced scalar value.
 *  - `std::nullopt`: `value` is incompatible or `descriptor` declares a list or map.
 */
std::optional<Chorus::ProviderOptionValue>
coerce_to_descriptor(const Chorus::ProviderOptionDescriptor& descriptor, const godot::Variant& value);

} // namespace godot_chorus

#pragma once

#include <optional>

#include <godot_cpp/core/property_info.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include "chorus/core/capabilities.hpp"
#include "chorus/core/provider_option_value.hpp"

// Godot's half of the option-schema seam: PropertyInfo and Variant translation
// only. The provider owns option names, types, defaults, and bounds
// (ARCHITECTURE.md, configuration metadata) and the core owns resolving them
// (Chorus::resolve_option_defaults); this file restates a descriptor in the
// inspector's vocabulary and nothing else.
namespace godot_chorus {

// `enabled` false marks the property read-only rather than hiding it: a gated
// option that vanishes reads as a bug, one that greys out reads as a gate.
godot::PropertyInfo property_info_for(const Chorus::ProviderOptionDescriptor& descriptor, bool enabled);

godot::Variant option_value_to_variant(const Chorus::ProviderOptionValue& value);

// nullopt when the Variant cannot be the shape the descriptor declares.
std::optional<Chorus::ProviderOptionValue>
coerce_to_descriptor(const Chorus::ProviderOptionDescriptor& descriptor, const godot::Variant& value);

} // namespace godot_chorus

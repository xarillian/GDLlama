#include "godot_chorus/provider_option_properties.hpp"

#include <algorithm>
#include <string>
#include <variant>

#include "godot_chorus/option_conversion.hpp"

using namespace godot;

namespace godot_chorus {
namespace {

// Godot range hints guide the editor widget but do not validate provider
// options. `or_greater` leaves the provider authoritative above the suggested
// maximum.
String range_hint(const Chorus::ProviderOptionDescriptor& descriptor) {
    if (!descriptor.minimum || !descriptor.maximum)
        return String();
    String hint =
        String::num_int64((int64_t)*descriptor.minimum) + "," + String::num_int64((int64_t)*descriptor.maximum);
    if (descriptor.step)
        hint += "," + String::num_int64((int64_t)*descriptor.step);
    return hint + ",or_greater";
}

Variant::Type variant_type_for(const Chorus::ProviderOptionValue& value) {
    if (std::holds_alternative<bool>(value))
        return Variant::BOOL;
    if (std::holds_alternative<int64_t>(value))
        return Variant::INT;
    if (std::holds_alternative<double>(value))
        return Variant::FLOAT;
    if (std::holds_alternative<std::string>(value))
        return Variant::STRING;
    if (std::holds_alternative<Chorus::ProviderOptionList>(value))
        return Variant::ARRAY;
    return Variant::DICTIONARY;
}

} // namespace

PropertyInfo property_info_for(const Chorus::ProviderOptionDescriptor& descriptor, bool enabled) {
    const Variant::Type type = variant_type_for(descriptor.default_value);
    String hint_string = (type == Variant::INT || type == Variant::FLOAT) ? range_hint(descriptor) : String();
    PropertyHint hint = hint_string.is_empty() ? PROPERTY_HINT_NONE : PROPERTY_HINT_RANGE;
    if (type == Variant::STRING && !descriptor.choices.empty()) {
        hint = PROPERTY_HINT_ENUM;
        for (size_t i = 0; i < descriptor.choices.size(); ++i) {
            if (i)
                hint_string += ",";
            hint_string += godot_chorus::to_godot_string(descriptor.choices[i]);
        }
    }
    uint32_t usage = PROPERTY_USAGE_DEFAULT;
    if (!enabled)
        usage |= PROPERTY_USAGE_READ_ONLY;
    return PropertyInfo(type, godot_chorus::to_godot_string(descriptor.key), hint, hint_string, usage);
}

Variant option_value_to_variant(const Chorus::ProviderOptionValue& value) {
    if (const auto* as_bool = std::get_if<bool>(&value))
        return Variant(*as_bool);
    if (const auto* as_int = std::get_if<int64_t>(&value))
        return Variant(*as_int);
    if (const auto* as_double = std::get_if<double>(&value))
        return Variant(*as_double);
    if (const auto* as_string = std::get_if<std::string>(&value))
        return Variant(godot_chorus::to_godot_string(*as_string));
    // Collection-valued options lack the element metadata needed for a
    // faithful inspector representation, so they surface as
    // `godot::Variant::NIL`.
    return Variant();
}

std::optional<Chorus::ProviderOptionValue>
coerce_to_descriptor(const Chorus::ProviderOptionDescriptor& descriptor, const Variant& value) {
    switch (variant_type_for(descriptor.default_value)) {
    case Variant::BOOL:
        if (value.get_type() != Variant::BOOL)
            return std::nullopt;
        return Chorus::ProviderOptionValue{(bool)value};
    case Variant::INT:
        // GDScript uses `godot::Variant::FLOAT` for a literal with a decimal
        // point; the inspector uses `godot::Variant::INT`.
        if (value.get_type() != Variant::INT && value.get_type() != Variant::FLOAT)
            return std::nullopt;
        return Chorus::ProviderOptionValue{(int64_t)value};
    case Variant::FLOAT:
        if (value.get_type() != Variant::INT && value.get_type() != Variant::FLOAT)
            return std::nullopt;
        return Chorus::ProviderOptionValue{(double)value};
    case Variant::STRING: {
        if (value.get_type() != Variant::STRING)
            return std::nullopt;
        std::string converted(((String)value).utf8().get_data());
        if (!descriptor.choices.empty() && std::ranges::find(descriptor.choices, converted) == descriptor.choices.end())
            return std::nullopt;
        return Chorus::ProviderOptionValue{std::move(converted)};
    }
    default:
        return std::nullopt;
    }
}

} // namespace godot_chorus

#include "godot_chorus/backend_option_properties.hpp"

#include <string>
#include <variant>

using namespace godot;

namespace godot_chorus {
namespace {

// A declared maximum is a widget bound, not the provider's limit, so every
// numeric range renders with or_greater: the spinner suggests the useful span
// while a script may still ask for more and earn the backend's own answer.
// Bounds render as whole numbers; the first float option with fractional
// bounds or step needs this to stop truncating.
String range_hint(const Chorus::OptionDescriptor& descriptor) {
    if (!descriptor.minimum || !descriptor.maximum)
        return String();
    String hint =
        String::num_int64((int64_t)*descriptor.minimum) + "," + String::num_int64((int64_t)*descriptor.maximum);
    if (descriptor.step)
        hint += "," + String::num_int64((int64_t)*descriptor.step);
    return hint + ",or_greater";
}

Variant::Type variant_type_for(const Chorus::OptionValue& value) {
    if (std::holds_alternative<bool>(value))
        return Variant::BOOL;
    if (std::holds_alternative<int64_t>(value))
        return Variant::INT;
    if (std::holds_alternative<double>(value))
        return Variant::FLOAT;
    if (std::holds_alternative<std::string>(value))
        return Variant::STRING;
    if (std::holds_alternative<Chorus::OptionList>(value))
        return Variant::ARRAY;
    return Variant::DICTIONARY;
}

} // namespace

PropertyInfo property_info_for(const Chorus::OptionDescriptor& descriptor, bool enabled) {
    const Variant::Type type = variant_type_for(descriptor.default_value);
    const String hint_string = (type == Variant::INT || type == Variant::FLOAT) ? range_hint(descriptor) : String();
    uint32_t usage = PROPERTY_USAGE_DEFAULT;
    if (!enabled)
        usage |= PROPERTY_USAGE_READ_ONLY;
    return PropertyInfo(
        type,
        String(descriptor.key.c_str()),
        hint_string.is_empty() ? PROPERTY_HINT_NONE : PROPERTY_HINT_RANGE,
        hint_string,
        usage
    );
}

Variant option_value_to_variant(const Chorus::OptionValue& value) {
    if (const auto* as_bool = std::get_if<bool>(&value))
        return Variant(*as_bool);
    if (const auto* as_int = std::get_if<int64_t>(&value))
        return Variant(*as_int);
    if (const auto* as_double = std::get_if<double>(&value))
        return Variant(*as_double);
    if (const auto* as_string = std::get_if<std::string>(&value))
        return Variant(String(as_string->c_str()));
    // List- and map-valued options have no inspector rendering yet: they read
    // as null here, and coerce_to_descriptor refuses every write, so the first
    // backend to declare one fails loudly rather than quietly.
    return Variant();
}

std::optional<Chorus::OptionValue>
coerce_to_descriptor(const Chorus::OptionDescriptor& descriptor, const Variant& value) {
    switch (variant_type_for(descriptor.default_value)) {
    case Variant::BOOL:
        if (value.get_type() != Variant::BOOL)
            return std::nullopt;
        return Chorus::OptionValue{(bool)value};
    case Variant::INT:
        // GDScript writes a float for any literal carrying a decimal point;
        // through the inspector an int option always arrives as INT.
        if (value.get_type() != Variant::INT && value.get_type() != Variant::FLOAT)
            return std::nullopt;
        return Chorus::OptionValue{(int64_t)value};
    case Variant::FLOAT:
        if (value.get_type() != Variant::INT && value.get_type() != Variant::FLOAT)
            return std::nullopt;
        return Chorus::OptionValue{(double)value};
    case Variant::STRING:
        if (value.get_type() != Variant::STRING)
            return std::nullopt;
        return Chorus::OptionValue{std::string(((String)value).utf8().get_data())};
    default:
        return std::nullopt;
    }
}

} // namespace godot_chorus

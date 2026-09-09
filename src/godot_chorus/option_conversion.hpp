#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "chorus/core/provider_option_value.hpp"

// Conversions used at the boundary between Godot and host-neutral Chorus types.
namespace godot_chorus {

/*
 * Converts UTF-8 text to a `godot::String`.
 *
 * Godot's `godot::String(const char*)` constructor reads Latin-1, so it
 * corrupts multibyte text. Every `std::string` the adapter hands to Godot
 * must pass through this function; `godot::String::utf8()` is its inverse.
 */
inline godot::String to_godot_string(const std::string& text) {
    return godot::String::utf8(text.c_str(), static_cast<int64_t>(text.size()));
}

namespace detail {

class OptionValueConversion {
  public:
    explicit OptionValueConversion(std::string& error) : _error(error) {}

    std::optional<Chorus::ProviderOptionValue> convert(const godot::Variant& value) {
        using godot::String;
        using godot::Variant;
        switch (value.get_type()) {
        case Variant::BOOL:
            return Chorus::ProviderOptionValue{(bool)value};
        case Variant::INT:
            return Chorus::ProviderOptionValue{(int64_t)value};
        case Variant::FLOAT:
            return Chorus::ProviderOptionValue{(double)value};
        case Variant::STRING:
            return Chorus::ProviderOptionValue{std::string(((String)value).utf8().get_data())};
        case Variant::ARRAY:
        case Variant::DICTIONARY:
            break;
        default:
            _error = "contains an unsupported value type.";
            return std::nullopt;
        }

        // Identity checks must not traverse the very graph being validated.
        // Only ancestors count: sibling references are valid acyclic sharing.
        for (const auto& ancestor : _ancestors) {
            if (godot::UtilityFunctions::is_same(ancestor, value)) {
                _error = "contains a container cycle.";
                return std::nullopt;
            }
        }
        if (_ancestors.size() >= 64) {
            _error = "exceeds the maximum container nesting depth of 64.";
            return std::nullopt;
        }
        _ancestors.push_back(value);
        auto result = convert_container(value);
        _ancestors.pop_back();
        return result;
    }

  private:
    std::optional<Chorus::ProviderOptionValue> convert_container(const godot::Variant& value) {
        using godot::Array;
        using godot::Dictionary;
        using godot::String;
        using godot::Variant;
        if (value.get_type() == Variant::ARRAY) {
            Chorus::ProviderOptionList list;
            const Array arr = value;
            for (int64_t i = 0; i < arr.size(); ++i) {
                auto item = convert(arr[i]);
                if (!item)
                    return std::nullopt;
                list.push_back(std::move(*item));
            }
            return Chorus::ProviderOptionValue{std::move(list)};
        }
        Chorus::ProviderOptionMap map;
        const Dictionary dict = value;
        const Array keys = dict.keys();
        for (int64_t i = 0; i < keys.size(); ++i) {
            const Variant key = keys[i];
            if (key.get_type() != Variant::STRING) {
                _error = "contains a non-string dictionary key.";
                return std::nullopt;
            }
            auto item = convert(dict[key]);
            if (!item)
                return std::nullopt;
            map[std::string(((String)key).utf8().get_data())] = std::move(*item);
        }
        return Chorus::ProviderOptionValue{std::move(map)};
    }

    std::string& _error;
    std::vector<godot::Variant> _ancestors;
};

} // namespace detail

/*
 * Converts a `godot::Variant` graph to a `Chorus::ProviderOptionValue` tree.
 *
 * Container nesting is limited to 64, including the root. Shared acyclic
 * containers are copied independently; cycles cannot be represented.
 *
 * Returns:
 *  - `std::optional<Chorus::ProviderOptionValue>`: the converted value.
 *  - `std::nullopt`: invalid input, with a diagnostic in `error`.
 */
inline std::optional<Chorus::ProviderOptionValue> variant_to_option_value(
    const godot::Variant& value, std::string& error
) {
    error.clear();
    return detail::OptionValueConversion(error).convert(value);
}

} // namespace godot_chorus

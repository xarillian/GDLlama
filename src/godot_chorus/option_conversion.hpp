#pragma once

#include <optional>
#include <string>
#include <utility>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

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

/*
 * Converts a `godot::Variant` tree to `Chorus::ProviderOptionValue`.
 *
 * Arrays and dictionaries are converted recursively. An unsupported value
 * makes the entire conversion fail so callers can reject it explicitly.
 *
 * Returns:
 *  - `std::optional<Chorus::ProviderOptionValue>`: the converted value.
 *  - `std::nullopt`: the tree contains an unsupported `godot::Variant` type.
 */
inline std::optional<Chorus::ProviderOptionValue> variant_to_option_value(const godot::Variant& value) {
    using godot::Array;
    using godot::Dictionary;
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
    case Variant::ARRAY: {
        Chorus::ProviderOptionList list;
        Array arr = value;
        for (int i = 0; i < arr.size(); ++i) {
            auto item = variant_to_option_value(arr[i]);
            if (!item)
                return std::nullopt;
            list.push_back(std::move(*item));
        }
        return Chorus::ProviderOptionValue{std::move(list)};
    }
    case Variant::DICTIONARY: {
        Chorus::ProviderOptionMap map;
        Dictionary dict = value;
        Array keys = dict.keys();
        for (int i = 0; i < keys.size(); ++i) {
            auto item = variant_to_option_value(dict[keys[i]]);
            if (!item)
                return std::nullopt;
            map[std::string(((String)keys[i]).utf8().get_data())] = std::move(*item);
        }
        return Chorus::ProviderOptionValue{std::move(map)};
    }
    default:
        return std::nullopt;
    }
}

} // namespace godot_chorus

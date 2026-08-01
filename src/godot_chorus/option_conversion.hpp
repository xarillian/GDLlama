#pragma once

#include <optional>
#include <string>
#include <utility>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include "chorus/core/provider_option_value.hpp"

// Private helpers shared by the Godot adapter's translation units, converting
// between Variant and the host-neutral Chorus types in both directions.
namespace godot_chorus {

/*
 * Builds a godot::String from UTF-8 text.
 *
 * Every string crossing this boundary (model output, prompts, stop markers,
 * grammar source, paths, session ids) is UTF-8, and godot-cpp's
 * `String(const char*)` constructor reads Latin-1, so it garbles anything
 * multi-byte. Use this for every std::string the adapter hands to Godot; the
 * `String::utf8()` accessor is its inverse.
 */
inline godot::String to_godot_string(const std::string& text) {
    return godot::String::utf8(text.c_str(), static_cast<int64_t>(text.size()));
}

/*
 * Recursively converts a Variant (bool/int/float/String/Array/Dictionary) into
 * the host-neutral Chorus::ProviderOptionValue.
 *
 * Returns:
 *  - `std::optional<Chorus::ProviderOptionValue>`: the converted value.
 *  - `std::nullopt`: the Variant holds an unsupported type, so callers can
 *    reject rather than silently drop it.
 */

inline std::optional<Chorus::ProviderOptionValue> variant_to_option_value(const godot::Variant& v) {
    using godot::Array;
    using godot::Dictionary;
    using godot::String;
    using godot::Variant;

    switch (v.get_type()) {
    case Variant::BOOL:
        return Chorus::ProviderOptionValue{(bool)v};
    case Variant::INT:
        return Chorus::ProviderOptionValue{(int64_t)v};
    case Variant::FLOAT:
        return Chorus::ProviderOptionValue{(double)v};
    case Variant::STRING:
        return Chorus::ProviderOptionValue{std::string(((String)v).utf8().get_data())};
    case Variant::ARRAY: {
        Chorus::ProviderOptionList list;
        Array arr = v;
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
        Dictionary dict = v;
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

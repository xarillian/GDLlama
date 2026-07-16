#pragma once

#include <optional>
#include <string>
#include <utility>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include "chorus/core/options.hpp"

// Private helper shared by the Godot adapter's translation units: recursively
// converts a Variant (bool/int/float/String/Array/Dictionary) into the
// host-neutral Chorus::OptionValue. Returns nullopt on any unsupported type
// so callers can reject rather than silently drop values.
namespace godot_chorus {

inline std::optional<Chorus::OptionValue> variant_to_option_value(const godot::Variant& v) {
    using godot::Array;
    using godot::Dictionary;
    using godot::String;
    using godot::Variant;

    switch (v.get_type()) {
    case Variant::BOOL:
        return Chorus::OptionValue{(bool)v};
    case Variant::INT:
        return Chorus::OptionValue{(int64_t)v};
    case Variant::FLOAT:
        return Chorus::OptionValue{(double)v};
    case Variant::STRING:
        return Chorus::OptionValue{std::string(((String)v).utf8().get_data())};
    case Variant::ARRAY: {
        Chorus::OptionList list;
        Array arr = v;
        for (int i = 0; i < arr.size(); ++i) {
            auto item = variant_to_option_value(arr[i]);
            if (!item)
                return std::nullopt;
            list.push_back(std::move(*item));
        }
        return Chorus::OptionValue{std::move(list)};
    }
    case Variant::DICTIONARY: {
        Chorus::OptionMap map;
        Dictionary dict = v;
        Array keys = dict.keys();
        for (int i = 0; i < keys.size(); ++i) {
            auto item = variant_to_option_value(dict[keys[i]]);
            if (!item)
                return std::nullopt;
            map[std::string(((String)keys[i]).utf8().get_data())] = std::move(*item);
        }
        return Chorus::OptionValue{std::move(map)};
    }
    default:
        return std::nullopt;
    }
}

} // namespace godot_chorus

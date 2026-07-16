#include "godot_chorus/generation_request_normalizer.hpp"

#include <array>
#include <optional>
#include <string>
#include <utility>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include "godot_chorus/option_conversion.hpp"

using namespace godot;

namespace godot_chorus {
namespace {

// ===========================================================================
// Portable scalar keys: absent inherits, null clears, a present value sets.
// ===========================================================================

template <typename T, typename Caster>
void apply_scalar_overlay(const Dictionary& request, const char* key, Chorus::OptionalPatch<T>& field, Caster caster) {
    if (!request.has(key))
        return;
    const Variant value = request[key];
    if (value.get_type() == Variant::NIL)
        field = Chorus::OptionalPatch<T>::clear();
    else
        field = Chorus::OptionalPatch<T>::set(caster(value));
}

void apply_portable_scalar_overlays(const Dictionary& request, Chorus::GenerationConfigPatch& patch) {
    apply_scalar_overlay(request, "max_tokens", patch.max_tokens, [](const Variant& v) { return (int32_t)(int64_t)v; });
    apply_scalar_overlay(request, "temperature", patch.temperature, [](const Variant& v) { return (float)v; });
    apply_scalar_overlay(request, "top_k", patch.top_k, [](const Variant& v) { return (int32_t)(int64_t)v; });
    apply_scalar_overlay(request, "top_p", patch.top_p, [](const Variant& v) { return (float)v; });
    apply_scalar_overlay(request, "seed", patch.seed, [](const Variant& v) { return (uint64_t)(int64_t)v; });
    apply_scalar_overlay(request, "frequency_penalty", patch.frequency_penalty, [](const Variant& v) {
        return (float)v;
    });
    apply_scalar_overlay(request, "presence_penalty", patch.presence_penalty, [](const Variant& v) {
        return (float)v;
    });
}

// ===========================================================================
// stop: null clears to the backend default; [] explicitly disables inherited
// stops; a non-empty array replaces them.
// ===========================================================================

std::optional<String> apply_stop_overlay(const Dictionary& request, Chorus::GenerationConfigPatch& patch) {
    if (!request.has("stop"))
        return std::nullopt;
    const Variant value = request["stop"];
    if (value.get_type() == Variant::NIL) {
        patch.stop = Chorus::ValuePatch<std::vector<std::string>>::clear();
        return std::nullopt;
    }
    if (value.get_type() != Variant::ARRAY)
        return String("[Chorus] generate(): 'stop' must be an Array of Strings, an empty Array, or null.");

    const Array arr = value;
    std::vector<std::string> stops;
    stops.reserve(arr.size());
    for (int i = 0; i < arr.size(); ++i) {
        const Variant entry = arr[i];
        if (entry.get_type() != Variant::STRING)
            return String("[Chorus] generate(): 'stop' entries must be Strings.");
        stops.emplace_back(((String)entry).utf8().get_data());
    }
    patch.stop = Chorus::ValuePatch<std::vector<std::string>>::set(std::move(stops));
    return std::nullopt;
}

// ===========================================================================
// constraint: the canonical {"format", "source"} spelling plus the grammar /
// json_schema / json conveniences. At most one spelling may be present; null
// clears an inherited constraint.
// ===========================================================================

constexpr std::array<const char*, 4> kConstraintKeys = {"constraint", "grammar", "json_schema", "json"};

std::optional<String> apply_constraint_overlay(const Dictionary& request, Chorus::GenerationConfigPatch& patch) {
    std::vector<const char*> present;
    for (const char* key : kConstraintKeys) {
        if (request.has(key))
            present.push_back(key);
    }
    if (present.size() > 1) {
        String joined;
        for (size_t i = 0; i < present.size(); ++i) {
            if (i)
                joined += ", ";
            joined += present[i];
        }
        return String("[Chorus] generate(): at most one constraint spelling may be given (got ") + joined +
               String(").");
    }
    if (present.empty())
        return std::nullopt;

    const std::string key = present.front();
    const Variant value = request[String(key.c_str())];

    if (value.get_type() == Variant::NIL) {
        patch.constraint = Chorus::OptionalPatch<Chorus::OutputConstraint>::clear();
        return std::nullopt;
    }

    Chorus::OutputConstraint constraint;
    if (key == "constraint") {
        if (value.get_type() != Variant::DICTIONARY)
            return String(
                "[Chorus] generate(): 'constraint' must be a Dictionary with 'format' and 'source', or null."
            );
        const Dictionary dict = value;
        if (!dict.has("format") || !dict.has("source"))
            return String("[Chorus] generate(): 'constraint' requires both 'format' and 'source' keys.");
        const String format = dict["format"];
        if (format == "gbnf")
            constraint.format = Chorus::ConstraintFormat::Gbnf;
        else if (format == "json_schema")
            constraint.format = Chorus::ConstraintFormat::JsonSchema;
        else
            return String("[Chorus] generate(): 'constraint.format' must be 'gbnf' or 'json_schema'.");
        constraint.source = std::string(((String)dict["source"]).utf8().get_data());
    } else if (key == "grammar") {
        constraint.format = Chorus::ConstraintFormat::Gbnf;
        constraint.source = std::string(((String)value).utf8().get_data());
    } else { // "json_schema" or "json": two spellings of the same schema-text convenience.
        constraint.format = Chorus::ConstraintFormat::JsonSchema;
        constraint.source = std::string(((String)value).utf8().get_data());
    }

    patch.constraint = Chorus::OptionalPatch<Chorus::OutputConstraint>::set(constraint);
    return std::nullopt;
}

// ===========================================================================
// backend_options: recursive namespace merge onto the inherited OptionMap. A
// null leaf erases the corresponding inherited key (checked before any
// OptionMap conversion is attempted); a nested Dictionary deep-merges; any
// other value overwrites the leaf via option_conversion's Variant->OptionValue.
// ===========================================================================

std::optional<String> merge_backend_dictionary_overrides(Chorus::OptionMap& target, const Dictionary& overrides) {
    const Array keys = overrides.keys();
    for (int i = 0; i < keys.size(); ++i) {
        const Variant key_variant = keys[i];
        if (key_variant.get_type() != Variant::STRING)
            return String("[Chorus] generate(): backend_options keys must be Strings.");
        const std::string key = std::string(((String)key_variant).utf8().get_data());
        const Variant value = overrides[key_variant];

        if (value.get_type() == Variant::NIL) {
            target.erase(key);
            continue;
        }

        if (value.get_type() == Variant::DICTIONARY) {
            const auto existing = target.find(key);
            Chorus::OptionMap* nested;
            if (existing != target.end() && std::holds_alternative<Chorus::OptionMap>(existing->second)) {
                nested = &std::get<Chorus::OptionMap>(existing->second);
            } else {
                const auto [it, inserted] = target.insert_or_assign(key, Chorus::OptionMap{});
                nested = &std::get<Chorus::OptionMap>(it->second);
            }
            if (auto error = merge_backend_dictionary_overrides(*nested, (Dictionary)value))
                return error;
            continue;
        }

        auto converted = godot_chorus::variant_to_option_value(value);
        if (!converted)
            return String("[Chorus] generate(): backend_options key '") + String(key.c_str()) +
                   String("' has an unsupported value.");
        target[key] = std::move(*converted);
    }
    return std::nullopt;
}

// ===========================================================================
// repeat_penalty: convenience for backend_options["llama"]["repeat_penalty"],
// applied after backend_options so it wins; null erases the inherited leaf.
// ===========================================================================

std::optional<String> apply_repeat_penalty_convenience(Chorus::OptionMap& backend, const Dictionary& request) {
    if (!request.has("repeat_penalty"))
        return std::nullopt;
    const Variant value = request["repeat_penalty"];

    if (value.get_type() == Variant::NIL) {
        const auto existing = backend.find("llama");
        if (existing != backend.end() && std::holds_alternative<Chorus::OptionMap>(existing->second))
            std::get<Chorus::OptionMap>(existing->second).erase("repeat_penalty");
        return std::nullopt;
    }
    if (value.get_type() != Variant::FLOAT && value.get_type() != Variant::INT)
        return String("[Chorus] generate(): 'repeat_penalty' must be a float or null.");

    const auto existing = backend.find("llama");
    Chorus::OptionMap* llama;
    if (existing != backend.end() && std::holds_alternative<Chorus::OptionMap>(existing->second)) {
        llama = &std::get<Chorus::OptionMap>(existing->second);
    } else {
        const auto [it, inserted] = backend.insert_or_assign("llama", Chorus::OptionMap{});
        llama = &std::get<Chorus::OptionMap>(it->second);
    }
    (*llama)["repeat_penalty"] = (double)(float)value;
    return std::nullopt;
}

} // namespace

std::variant<Chorus::GenerationRequest, String>
normalize_generation_request(const Dictionary& request, const Chorus::GenerationConfig& defaults) {
    if (!request.has("prompt"))
        return String("[Chorus] generate() requires a 'prompt' key in the request dictionary.");
    const Variant prompt_value = request["prompt"];
    if (prompt_value.get_type() == Variant::NIL)
        return String("[Chorus] generate(): 'prompt' must not be null.");

    Chorus::GenerationConfigPatch patch;
    apply_portable_scalar_overlays(request, patch);
    if (auto error = apply_stop_overlay(request, patch))
        return *error;
    if (auto error = apply_constraint_overlay(request, patch))
        return *error;

    Chorus::GenerationConfig config = Chorus::apply_generation_patch(defaults, patch);

    if (request.has("backend_options")) {
        const Variant backend_options = request["backend_options"];
        if (backend_options.get_type() != Variant::DICTIONARY)
            return String("[Chorus] generate(): 'backend_options' must be a Dictionary.");
        if (auto error = merge_backend_dictionary_overrides(config.backend_options, (Dictionary)backend_options))
            return *error;
    }
    if (auto error = apply_repeat_penalty_convenience(config.backend_options, request))
        return *error;

    Chorus::GenerationRequest gen_request;
    gen_request.prompt = std::string(((String)prompt_value).utf8().get_data());
    gen_request.stream = request.has("stream") ? (bool)request["stream"] : false;
    gen_request.priority = request.has("priority") ? (int)(int64_t)request["priority"] : 0;
    if (request.has("session"))
        gen_request.session_id = std::string(((String)request["session"]).utf8().get_data());
    gen_request.config = std::move(config);
    return gen_request;
}

} // namespace godot_chorus

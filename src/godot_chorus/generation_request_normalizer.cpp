#include "godot_chorus/generation_request_normalizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
// Common scalar keys: absent inherits, null clears, a present value sets.
// ===========================================================================

String scalar_error(const char* key, const char* requirement) {
    return String("[Chorus] generate(): '") + key + "' " + requirement;
}

std::optional<String>
apply_int32_overlay(const Dictionary& request, const char* key, Chorus::ConfigPatch<int32_t>& field) {
    if (!request.has(key))
        return std::nullopt;
    const Variant value = request[key];
    if (value.get_type() == Variant::NIL) {
        field = Chorus::ConfigPatch<int32_t>::clear();
        return std::nullopt;
    }
    if (value.get_type() != Variant::INT)
        return scalar_error(key, "must be an int or null.");

    const int64_t converted = value;
    if (converted < std::numeric_limits<int32_t>::min() || converted > std::numeric_limits<int32_t>::max())
        return scalar_error(key, "must fit a signed 32-bit integer.");
    field = Chorus::ConfigPatch<int32_t>::set(static_cast<int32_t>(converted));
    return std::nullopt;
}

std::optional<String>
apply_float_overlay(const Dictionary& request, const char* key, Chorus::ConfigPatch<float>& field) {
    if (!request.has(key))
        return std::nullopt;
    const Variant value = request[key];
    if (value.get_type() == Variant::NIL) {
        field = Chorus::ConfigPatch<float>::clear();
        return std::nullopt;
    }
    if (value.get_type() != Variant::FLOAT && value.get_type() != Variant::INT)
        return scalar_error(key, "must be a number or null.");

    const double converted = value;
    if (!std::isfinite(converted) || std::abs(converted) > std::numeric_limits<float>::max())
        return scalar_error(key, "must be finite and fit a 32-bit float.");
    field = Chorus::ConfigPatch<float>::set(static_cast<float>(converted));
    return std::nullopt;
}

std::optional<String>
apply_seed_overlay(const Dictionary& request, Chorus::ConfigPatch<uint64_t>& field) {
    if (!request.has("seed"))
        return std::nullopt;
    const Variant value = request["seed"];
    if (value.get_type() == Variant::NIL) {
        field = Chorus::ConfigPatch<uint64_t>::clear();
        return std::nullopt;
    }
    if (value.get_type() != Variant::INT)
        return scalar_error("seed", "must be a non-negative int or null.");

    const int64_t converted = value;
    if (converted < 0)
        return scalar_error("seed", "must be non-negative.");
    field = Chorus::ConfigPatch<uint64_t>::set(static_cast<uint64_t>(converted));
    return std::nullopt;
}

std::optional<String> apply_common_scalar_overlays(const Dictionary& request, Chorus::GenerationConfigPatch& patch) {
    if (auto error = apply_int32_overlay(request, "max_tokens", patch.max_tokens))
        return error;
    if (auto error = apply_float_overlay(request, "temperature", patch.temperature))
        return error;
    if (auto error = apply_int32_overlay(request, "top_k", patch.top_k))
        return error;
    if (auto error = apply_float_overlay(request, "top_p", patch.top_p))
        return error;
    if (auto error = apply_seed_overlay(request, patch.seed))
        return error;
    if (auto error = apply_float_overlay(request, "frequency_penalty", patch.frequency_penalty))
        return error;
    return apply_float_overlay(request, "presence_penalty", patch.presence_penalty);
}

// ===========================================================================
// stop: null clears to the provider default; [] explicitly disables inherited
// stops; a non-empty array replaces them.
// ===========================================================================

std::optional<String> apply_stop_overlay(const Dictionary& request, Chorus::GenerationConfigPatch& patch) {
    if (!request.has("stop"))
        return std::nullopt;
    const Variant value = request["stop"];
    if (value.get_type() == Variant::NIL) {
        patch.stop = Chorus::ConfigPatch<std::vector<std::string>>::clear();
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
    patch.stop = Chorus::ConfigPatch<std::vector<std::string>>::set(std::move(stops));
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
    const Variant value = request[godot_chorus::to_godot_string(key)];

    if (value.get_type() == Variant::NIL) {
        patch.constraint = Chorus::ConfigPatch<Chorus::OutputConstraint>::clear();
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

    patch.constraint = Chorus::ConfigPatch<Chorus::OutputConstraint>::set(constraint);
    return std::nullopt;
}

// ===========================================================================
// provider_options: recursive namespace merge into the patch's ProviderOptionMap. A
// nested Dictionary deep-merges; any other value overwrites the leaf via
// option_conversion's Variant->ProviderOptionValue. A null leaf records a dotted
// erasure path instead of a value: the layer it has to remove lives below the
// patch, in the runtime's host defaults, so the removal travels as an
// instruction rather than happening here.
// ===========================================================================

std::optional<String> merge_provider_dictionary_overrides(
    Chorus::GenerationConfigPatch& patch,
    Chorus::ProviderOptionMap& target,
    const std::string& prefix,
    const Dictionary& overrides
) {
    const Array keys = overrides.keys();
    for (int i = 0; i < keys.size(); ++i) {
        const Variant key_variant = keys[i];
        if (key_variant.get_type() != Variant::STRING)
            return String("[Chorus] generate(): provider_options keys must be Strings.");
        const std::string key = std::string(((String)key_variant).utf8().get_data());
        const std::string path = prefix.empty() ? key : prefix + "." + key;
        const Variant value = overrides[key_variant];

        if (value.get_type() == Variant::NIL) {
            patch.provider_option_erasures.push_back(path);
            continue;
        }

        if (value.get_type() == Variant::DICTIONARY) {
            const auto existing = target.find(key);
            Chorus::ProviderOptionMap* nested;
            if (existing != target.end() && std::holds_alternative<Chorus::ProviderOptionMap>(existing->second)) {
                nested = &std::get<Chorus::ProviderOptionMap>(existing->second);
            } else {
                const auto [it, inserted] = target.insert_or_assign(key, Chorus::ProviderOptionMap{});
                nested = &std::get<Chorus::ProviderOptionMap>(it->second);
            }
            if (auto error = merge_provider_dictionary_overrides(patch, *nested, path, (Dictionary)value))
                return error;
            continue;
        }

        auto converted = godot_chorus::variant_to_option_value(value);
        if (!converted)
            return String("[Chorus] generate(): provider_options key '") + godot_chorus::to_godot_string(key) +
                   String("' has an unsupported value.");
        target[key] = std::move(*converted);
    }
    return std::nullopt;
}

// ===========================================================================
// repeat_penalty: convenience for provider_options["llama"]["repeat_penalty"],
// applied after provider_options so it wins; null erases the inherited leaf.
// ===========================================================================

constexpr const char* kRepeatPenaltyPath = "llama.repeat_penalty";

void drop_erasure(Chorus::GenerationConfigPatch& patch, const std::string& path) {
    auto& paths = patch.provider_option_erasures;
    paths.erase(std::remove(paths.begin(), paths.end(), path), paths.end());
}

Chorus::ProviderOptionMap& llama_namespace(Chorus::ProviderOptionMap& provider) {
    const auto existing = provider.find("llama");
    if (existing != provider.end() && std::holds_alternative<Chorus::ProviderOptionMap>(existing->second))
        return std::get<Chorus::ProviderOptionMap>(existing->second);
    const auto [it, inserted] = provider.insert_or_assign("llama", Chorus::ProviderOptionMap{});
    return std::get<Chorus::ProviderOptionMap>(it->second);
}

std::optional<String>
apply_repeat_penalty_convenience(Chorus::GenerationConfigPatch& patch, const Dictionary& request) {
    if (!request.has("repeat_penalty"))
        return std::nullopt;
    const Variant value = request["repeat_penalty"];

    if (value.get_type() == Variant::NIL) {
        llama_namespace(patch.provider_options).erase("repeat_penalty");
        drop_erasure(patch, kRepeatPenaltyPath);
        patch.provider_option_erasures.emplace_back(kRepeatPenaltyPath);
        return std::nullopt;
    }
    if (value.get_type() != Variant::FLOAT && value.get_type() != Variant::INT)
        return String("[Chorus] generate(): 'repeat_penalty' must be a float or null.");

    // The convenience spelling wins over a provider_options entry for the same
    // leaf, including one that spelled itself as an erasure.
    drop_erasure(patch, kRepeatPenaltyPath);
    llama_namespace(patch.provider_options)["repeat_penalty"] = (double)(float)value;
    return std::nullopt;
}

// ===========================================================================
// show_thinking: reasoning-model toggle. Absent inherits, null clears an
// inherited value back to the template/provider default, a bool sets it.
// ===========================================================================

std::optional<String> apply_show_thinking_overlay(const Dictionary& request, Chorus::GenerationConfigPatch& patch) {
    if (!request.has("show_thinking"))
        return std::nullopt;
    const Variant value = request["show_thinking"];
    if (value.get_type() == Variant::NIL) {
        patch.show_thinking = Chorus::ConfigPatch<bool>::clear();
        return std::nullopt;
    }
    if (value.get_type() != Variant::BOOL)
        return String("[Chorus] generate(): 'show_thinking' must be a bool or null.");
    patch.show_thinking = Chorus::ConfigPatch<bool>::set((bool)value);
    return std::nullopt;
}

// ===========================================================================
// inject: per-request ephemeral messages, {role, content, depth?} entries.
// ===========================================================================

std::optional<String> apply_inject_overlay(const Dictionary& request, Chorus::GenerationRequest& gen_request) {
    if (!request.has("inject"))
        return std::nullopt;
    const Variant value = request["inject"];
    if (value.get_type() != Variant::ARRAY)
        return String("[Chorus] generate(): 'inject' must be an Array of {role, content, depth} Dictionaries.");
    auto parsed = normalize_inject_array((Array)value);
    if (std::holds_alternative<String>(parsed))
        return std::get<String>(parsed);
    gen_request.inject = std::move(std::get<std::vector<Chorus::InjectedMessage>>(parsed));
    return std::nullopt;
}

// ===========================================================================
// chat_template: per-request jinja override. Null reads as absent (the node
// default may still apply downstream).
// ===========================================================================

std::optional<String> apply_chat_template_overlay(const Dictionary& request, Chorus::GenerationRequest& gen_request) {
    if (!request.has("chat_template"))
        return std::nullopt;
    const Variant value = request["chat_template"];
    if (value.get_type() == Variant::NIL)
        return std::nullopt;
    if (value.get_type() != Variant::STRING)
        return String("[Chorus] generate(): 'chat_template' must be a String or null.");
    gen_request.chat_template = std::string(((String)value).utf8().get_data());
    return std::nullopt;
}

} // namespace

std::variant<std::vector<Chorus::InjectedMessage>, String> normalize_inject_array(const Array& entries) {
    std::vector<Chorus::InjectedMessage> inject;
    inject.reserve(entries.size());
    for (int i = 0; i < entries.size(); ++i) {
        if (entries[i].get_type() != Variant::DICTIONARY)
            return String("[Chorus] generate(): 'inject' entries must be Dictionaries.");
        const Dictionary entry = entries[i];
        if (!entry.has("role") || !entry.has("content"))
            return String("[Chorus] generate(): 'inject' entries require 'role' and 'content'.");
        if (entry["role"].get_type() != Variant::STRING || entry["content"].get_type() != Variant::STRING)
            return String("[Chorus] generate(): 'inject' role and content must be Strings.");
        if (entry.has("depth") && entry["depth"].get_type() != Variant::INT)
            return String("[Chorus] generate(): 'inject' depth must be an int.");
        Chorus::InjectedMessage injected;
        injected.message.role = std::string(((String)entry["role"]).utf8().get_data());
        injected.message.content = std::string(((String)entry["content"]).utf8().get_data());
        injected.depth = entry.has("depth") ? (int32_t)(int64_t)entry["depth"] : 0;
        inject.push_back(std::move(injected));
    }
    return inject;
}

std::variant<Chorus::GenerationRequest, String> normalize_generation_request(const Dictionary& request) {
    if (!request.has("prompt"))
        return String("[Chorus] generate() requires a 'prompt' key in the request dictionary.");
    if (request["prompt"].get_type() == Variant::NIL)
        return String("[Chorus] generate(): 'prompt' must not be null.");
    return normalize_generation_overrides(request);
}

std::variant<Chorus::GenerationRequest, String> normalize_generation_overrides(const Dictionary& request) {
    Chorus::GenerationConfigPatch patch;
    if (auto error = apply_common_scalar_overlays(request, patch))
        return *error;
    if (auto error = apply_stop_overlay(request, patch))
        return *error;
    if (auto error = apply_constraint_overlay(request, patch))
        return *error;
    if (auto error = apply_show_thinking_overlay(request, patch))
        return *error;

    if (request.has("provider_options")) {
        const Variant provider_options = request["provider_options"];
        if (provider_options.get_type() != Variant::DICTIONARY)
            return String("[Chorus] generate(): 'provider_options' must be a Dictionary.");
        if (auto error =
                merge_provider_dictionary_overrides(patch, patch.provider_options, "", (Dictionary)provider_options))
            return *error;
    }
    if (auto error = apply_repeat_penalty_convenience(patch, request))
        return *error;

    Chorus::GenerationRequest gen_request;
    if (request.has("prompt") && request["prompt"].get_type() != Variant::NIL) {
        if (request["prompt"].get_type() != Variant::STRING)
            return scalar_error("prompt", "must be a String.");
        gen_request.prompt = std::string(((String)request["prompt"]).utf8().get_data());
    }
    if (request.has("stream")) {
        if (request["stream"].get_type() != Variant::BOOL)
            return scalar_error("stream", "must be a bool.");
        gen_request.stream = request["stream"];
    }
    if (request.has("priority")) {
        if (request["priority"].get_type() != Variant::INT)
            return scalar_error("priority", "must be an int.");
        const int64_t priority = request["priority"];
        if (priority < std::numeric_limits<int>::min() || priority > std::numeric_limits<int>::max())
            return scalar_error("priority", "must fit a signed 32-bit integer.");
        gen_request.priority = static_cast<int>(priority);
    }
    if (request.has("session")) {
        if (request["session"].get_type() != Variant::STRING)
            return scalar_error("session", "must be a String.");
        gen_request.session_id = std::string(((String)request["session"]).utf8().get_data());
    }
    if (auto error = apply_inject_overlay(request, gen_request))
        return *error;
    if (auto error = apply_chat_template_overlay(request, gen_request))
        return *error;
    gen_request.overrides = std::move(patch);
    return gen_request;
}

} // namespace godot_chorus

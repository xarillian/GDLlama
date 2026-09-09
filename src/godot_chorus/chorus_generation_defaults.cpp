#include "godot_chorus/chorus_generation_defaults.hpp"

#include <cstdint>
#include <limits>
#include <utility>

#include <godot_cpp/core/class_db.hpp>

#include "godot_chorus/option_conversion.hpp"

using namespace godot;

namespace {

std::vector<std::string> to_stop_vector(const PackedStringArray& value) {
    std::vector<std::string> stop;
    stop.reserve(value.size());
    for (int i = 0; i < value.size(); ++i)
        stop.emplace_back(value[i].utf8().get_data());
    return stop;
}

PackedStringArray to_packed_string_array(const std::vector<std::string>& value) {
    PackedStringArray stop;
    for (const auto& entry : value)
        stop.push_back(godot_chorus::to_godot_string(entry));
    return stop;
}

} // namespace

ChorusGenerationDefaults::ChorusGenerationDefaults() {
    _patch.max_tokens = Chorus::ConfigPatch<int32_t>::set(128);
}

std::optional<Chorus::GenerationConfigPatch> ChorusGenerationDefaults::to_patch(std::string& error) const {
    error.clear();
    Chorus::GenerationConfigPatch patch = _patch;
    if (patch.seed.action == Chorus::PatchAction::Set && patch.seed.value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        error = "seed must be nonnegative when set.";
        return std::nullopt;
    }
    if (patch.constraint.action == Chorus::PatchAction::Set) {
        switch (patch.constraint.value.format) {
        case Chorus::ConstraintFormat::Gbnf:
        case Chorus::ConstraintFormat::JsonSchema:
        case Chorus::ConstraintFormat::Regex:
        case Chorus::ConstraintFormat::Lark: break;
        default:
            error = "constraint_format is invalid.";
            return std::nullopt;
        }
    }
    if (!_provider_options.is_empty()) {
        auto converted = godot_chorus::variant_to_option_value(_provider_options, error);
        if (!converted) {
            error = "provider_options " + error;
            return std::nullopt;
        }
        patch.provider_options = std::get<Chorus::ProviderOptionMap>(std::move(*converted));
    }
    return patch;
}

void ChorusGenerationDefaults::set_override_max_tokens(bool enabled) {
    _patch.max_tokens.action = enabled ? Chorus::PatchAction::Set : Chorus::PatchAction::Inherit;
}
bool ChorusGenerationDefaults::get_override_max_tokens() const {
    return _patch.max_tokens.action == Chorus::PatchAction::Set;
}
void ChorusGenerationDefaults::set_max_tokens(int32_t value) {
    _patch.max_tokens.value = value;
}
int32_t ChorusGenerationDefaults::get_max_tokens() const {
    return _patch.max_tokens.value;
}

void ChorusGenerationDefaults::set_override_temperature(bool enabled) {
    _patch.temperature.action = enabled ? Chorus::PatchAction::Set : Chorus::PatchAction::Inherit;
}
bool ChorusGenerationDefaults::get_override_temperature() const {
    return _patch.temperature.action == Chorus::PatchAction::Set;
}
void ChorusGenerationDefaults::set_temperature(float value) {
    _patch.temperature.value = value;
}
float ChorusGenerationDefaults::get_temperature() const {
    return _patch.temperature.value;
}

void ChorusGenerationDefaults::set_override_top_k(bool enabled) {
    _patch.top_k.action = enabled ? Chorus::PatchAction::Set : Chorus::PatchAction::Inherit;
}
bool ChorusGenerationDefaults::get_override_top_k() const {
    return _patch.top_k.action == Chorus::PatchAction::Set;
}
void ChorusGenerationDefaults::set_top_k(int32_t value) {
    _patch.top_k.value = value;
}
int32_t ChorusGenerationDefaults::get_top_k() const {
    return _patch.top_k.value;
}

void ChorusGenerationDefaults::set_override_top_p(bool enabled) {
    _patch.top_p.action = enabled ? Chorus::PatchAction::Set : Chorus::PatchAction::Inherit;
}
bool ChorusGenerationDefaults::get_override_top_p() const {
    return _patch.top_p.action == Chorus::PatchAction::Set;
}
void ChorusGenerationDefaults::set_top_p(float value) {
    _patch.top_p.value = value;
}
float ChorusGenerationDefaults::get_top_p() const {
    return _patch.top_p.value;
}

void ChorusGenerationDefaults::set_override_seed(bool enabled) {
    _patch.seed.action = enabled ? Chorus::PatchAction::Set : Chorus::PatchAction::Inherit;
}
bool ChorusGenerationDefaults::get_override_seed() const {
    return _patch.seed.action == Chorus::PatchAction::Set;
}
void ChorusGenerationDefaults::set_seed(int64_t value) {
    _patch.seed.value = (uint64_t)value;
}
int64_t ChorusGenerationDefaults::get_seed() const {
    return (int64_t)_patch.seed.value;
}

void ChorusGenerationDefaults::set_override_frequency_penalty(bool enabled) {
    _patch.frequency_penalty.action = enabled ? Chorus::PatchAction::Set : Chorus::PatchAction::Inherit;
}
bool ChorusGenerationDefaults::get_override_frequency_penalty() const {
    return _patch.frequency_penalty.action == Chorus::PatchAction::Set;
}
void ChorusGenerationDefaults::set_frequency_penalty(float value) {
    _patch.frequency_penalty.value = value;
}
float ChorusGenerationDefaults::get_frequency_penalty() const {
    return _patch.frequency_penalty.value;
}

void ChorusGenerationDefaults::set_override_presence_penalty(bool enabled) {
    _patch.presence_penalty.action = enabled ? Chorus::PatchAction::Set : Chorus::PatchAction::Inherit;
}
bool ChorusGenerationDefaults::get_override_presence_penalty() const {
    return _patch.presence_penalty.action == Chorus::PatchAction::Set;
}
void ChorusGenerationDefaults::set_presence_penalty(float value) {
    _patch.presence_penalty.value = value;
}
float ChorusGenerationDefaults::get_presence_penalty() const {
    return _patch.presence_penalty.value;
}

void ChorusGenerationDefaults::set_override_stop(bool enabled) {
    _patch.stop.action = enabled ? Chorus::PatchAction::Set : Chorus::PatchAction::Inherit;
}
bool ChorusGenerationDefaults::get_override_stop() const {
    return _patch.stop.action == Chorus::PatchAction::Set;
}
void ChorusGenerationDefaults::set_stop(const PackedStringArray& value) {
    _patch.stop.value = to_stop_vector(value);
}
PackedStringArray ChorusGenerationDefaults::get_stop() const {
    return to_packed_string_array(_patch.stop.value);
}

void ChorusGenerationDefaults::set_override_constraint(bool enabled) {
    _patch.constraint.action = enabled ? Chorus::PatchAction::Set : Chorus::PatchAction::Inherit;
}
bool ChorusGenerationDefaults::get_override_constraint() const {
    return _patch.constraint.action == Chorus::PatchAction::Set;
}
void ChorusGenerationDefaults::set_constraint_format(ChorusConstraintFormat::Value format) {
    _patch.constraint.value.format = (Chorus::ConstraintFormat)format;
}
ChorusConstraintFormat::Value ChorusGenerationDefaults::get_constraint_format() const {
    return (ChorusConstraintFormat::Value)_patch.constraint.value.format;
}
void ChorusGenerationDefaults::set_constraint_source(const String& source) {
    _patch.constraint.value.source = std::string(source.utf8().get_data());
}
String ChorusGenerationDefaults::get_constraint_source() const {
    return godot_chorus::to_godot_string(_patch.constraint.value.source);
}

void ChorusGenerationDefaults::set_override_show_thinking(bool enabled) {
    _patch.show_thinking.action = enabled ? Chorus::PatchAction::Set : Chorus::PatchAction::Inherit;
}
bool ChorusGenerationDefaults::get_override_show_thinking() const {
    return _patch.show_thinking.action == Chorus::PatchAction::Set;
}
void ChorusGenerationDefaults::set_show_thinking(bool value) {
    _patch.show_thinking.value = value;
}
bool ChorusGenerationDefaults::get_show_thinking() const {
    return _patch.show_thinking.value;
}

void ChorusGenerationDefaults::set_provider_options(const Dictionary& options) {
    _provider_options = options;
}
Dictionary ChorusGenerationDefaults::get_provider_options() const {
    return _provider_options;
}

void ChorusGenerationDefaults::_bind_methods() {

    ClassDB::bind_method(
        D_METHOD("set_override_max_tokens", "enabled"), &ChorusGenerationDefaults::set_override_max_tokens
    );
    ClassDB::bind_method(D_METHOD("get_override_max_tokens"), &ChorusGenerationDefaults::get_override_max_tokens);
    ClassDB::bind_method(D_METHOD("set_max_tokens", "value"), &ChorusGenerationDefaults::set_max_tokens);
    ClassDB::bind_method(D_METHOD("get_max_tokens"), &ChorusGenerationDefaults::get_max_tokens);
    ADD_GROUP("Max Tokens", "");
    ADD_PROPERTY(
        PropertyInfo(Variant::BOOL, "override_max_tokens"), "set_override_max_tokens", "get_override_max_tokens"
    );
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tokens"), "set_max_tokens", "get_max_tokens");

    ClassDB::bind_method(
        D_METHOD("set_override_temperature", "enabled"), &ChorusGenerationDefaults::set_override_temperature
    );
    ClassDB::bind_method(D_METHOD("get_override_temperature"), &ChorusGenerationDefaults::get_override_temperature);
    ClassDB::bind_method(D_METHOD("set_temperature", "value"), &ChorusGenerationDefaults::set_temperature);
    ClassDB::bind_method(D_METHOD("get_temperature"), &ChorusGenerationDefaults::get_temperature);
    ADD_GROUP("Temperature", "");
    ADD_PROPERTY(
        PropertyInfo(Variant::BOOL, "override_temperature"), "set_override_temperature", "get_override_temperature"
    );
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temperature"), "set_temperature", "get_temperature");

    ClassDB::bind_method(D_METHOD("set_override_top_k", "enabled"), &ChorusGenerationDefaults::set_override_top_k);
    ClassDB::bind_method(D_METHOD("get_override_top_k"), &ChorusGenerationDefaults::get_override_top_k);
    ClassDB::bind_method(D_METHOD("set_top_k", "value"), &ChorusGenerationDefaults::set_top_k);
    ClassDB::bind_method(D_METHOD("get_top_k"), &ChorusGenerationDefaults::get_top_k);
    ADD_GROUP("Top K", "");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_top_k"), "set_override_top_k", "get_override_top_k");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "top_k"), "set_top_k", "get_top_k");

    ClassDB::bind_method(D_METHOD("set_override_top_p", "enabled"), &ChorusGenerationDefaults::set_override_top_p);
    ClassDB::bind_method(D_METHOD("get_override_top_p"), &ChorusGenerationDefaults::get_override_top_p);
    ClassDB::bind_method(D_METHOD("set_top_p", "value"), &ChorusGenerationDefaults::set_top_p);
    ClassDB::bind_method(D_METHOD("get_top_p"), &ChorusGenerationDefaults::get_top_p);
    ADD_GROUP("Top P", "");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_top_p"), "set_override_top_p", "get_override_top_p");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_p"), "set_top_p", "get_top_p");

    ClassDB::bind_method(D_METHOD("set_override_seed", "enabled"), &ChorusGenerationDefaults::set_override_seed);
    ClassDB::bind_method(D_METHOD("get_override_seed"), &ChorusGenerationDefaults::get_override_seed);
    ClassDB::bind_method(D_METHOD("set_seed", "value"), &ChorusGenerationDefaults::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"), &ChorusGenerationDefaults::get_seed);
    ADD_GROUP("Seed", "");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_seed"), "set_override_seed", "get_override_seed");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");

    ClassDB::bind_method(
        D_METHOD("set_override_frequency_penalty", "enabled"), &ChorusGenerationDefaults::set_override_frequency_penalty
    );
    ClassDB::bind_method(
        D_METHOD("get_override_frequency_penalty"), &ChorusGenerationDefaults::get_override_frequency_penalty
    );
    ClassDB::bind_method(D_METHOD("set_frequency_penalty", "value"), &ChorusGenerationDefaults::set_frequency_penalty);
    ClassDB::bind_method(D_METHOD("get_frequency_penalty"), &ChorusGenerationDefaults::get_frequency_penalty);
    ADD_GROUP("Frequency Penalty", "");
    ADD_PROPERTY(
        PropertyInfo(Variant::BOOL, "override_frequency_penalty"),
        "set_override_frequency_penalty",
        "get_override_frequency_penalty"
    );
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "frequency_penalty"), "set_frequency_penalty", "get_frequency_penalty");

    ClassDB::bind_method(
        D_METHOD("set_override_presence_penalty", "enabled"), &ChorusGenerationDefaults::set_override_presence_penalty
    );
    ClassDB::bind_method(
        D_METHOD("get_override_presence_penalty"), &ChorusGenerationDefaults::get_override_presence_penalty
    );
    ClassDB::bind_method(D_METHOD("set_presence_penalty", "value"), &ChorusGenerationDefaults::set_presence_penalty);
    ClassDB::bind_method(D_METHOD("get_presence_penalty"), &ChorusGenerationDefaults::get_presence_penalty);
    ADD_GROUP("Presence Penalty", "");
    ADD_PROPERTY(
        PropertyInfo(Variant::BOOL, "override_presence_penalty"),
        "set_override_presence_penalty",
        "get_override_presence_penalty"
    );
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "presence_penalty"), "set_presence_penalty", "get_presence_penalty");

    ClassDB::bind_method(D_METHOD("set_override_stop", "enabled"), &ChorusGenerationDefaults::set_override_stop);
    ClassDB::bind_method(D_METHOD("get_override_stop"), &ChorusGenerationDefaults::get_override_stop);
    ClassDB::bind_method(D_METHOD("set_stop", "value"), &ChorusGenerationDefaults::set_stop);
    ClassDB::bind_method(D_METHOD("get_stop"), &ChorusGenerationDefaults::get_stop);
    ADD_GROUP("Stop", "");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_stop"), "set_override_stop", "get_override_stop");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "stop"), "set_stop", "get_stop");

    ClassDB::bind_method(
        D_METHOD("set_override_constraint", "enabled"), &ChorusGenerationDefaults::set_override_constraint
    );
    ClassDB::bind_method(D_METHOD("get_override_constraint"), &ChorusGenerationDefaults::get_override_constraint);
    ClassDB::bind_method(D_METHOD("set_constraint_format", "format"), &ChorusGenerationDefaults::set_constraint_format);
    ClassDB::bind_method(D_METHOD("get_constraint_format"), &ChorusGenerationDefaults::get_constraint_format);
    ClassDB::bind_method(D_METHOD("set_constraint_source", "source"), &ChorusGenerationDefaults::set_constraint_source);
    ClassDB::bind_method(D_METHOD("get_constraint_source"), &ChorusGenerationDefaults::get_constraint_source);
    ADD_GROUP("Constraint", "");
    ADD_PROPERTY(
        PropertyInfo(Variant::BOOL, "override_constraint"), "set_override_constraint", "get_override_constraint"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "constraint_format", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "ChorusConstraintFormat.Value"),
        "set_constraint_format",
        "get_constraint_format"
    );
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "constraint_source"), "set_constraint_source", "get_constraint_source");

    ClassDB::bind_method(
        D_METHOD("set_override_show_thinking", "enabled"), &ChorusGenerationDefaults::set_override_show_thinking
    );
    ClassDB::bind_method(D_METHOD("get_override_show_thinking"), &ChorusGenerationDefaults::get_override_show_thinking);
    ClassDB::bind_method(D_METHOD("set_show_thinking", "value"), &ChorusGenerationDefaults::set_show_thinking);
    ClassDB::bind_method(D_METHOD("get_show_thinking"), &ChorusGenerationDefaults::get_show_thinking);
    ADD_GROUP("Thinking", "");
    ADD_PROPERTY(
        PropertyInfo(Variant::BOOL, "override_show_thinking"),
        "set_override_show_thinking",
        "get_override_show_thinking"
    );
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_thinking"), "set_show_thinking", "get_show_thinking");

    ClassDB::bind_method(D_METHOD("set_provider_options", "options"), &ChorusGenerationDefaults::set_provider_options);
    ClassDB::bind_method(D_METHOD("get_provider_options"), &ChorusGenerationDefaults::get_provider_options);
    ADD_GROUP("Provider Options", "");
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "provider_options"), "set_provider_options", "get_provider_options");
}

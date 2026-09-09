#include "godot_chorus/request_conversion.hpp"

#include <cmath>
#include <limits>

#include "godot_chorus/option_conversion.hpp"

namespace godot_chorus {
namespace {

std::optional<Chorus::PatchAction> patch_action(ChorusOverrideState::Value value) {
    switch (value) {
    case ChorusOverrideState::INHERIT: return Chorus::PatchAction::Inherit;
    case ChorusOverrideState::SET: return Chorus::PatchAction::Set;
    case ChorusOverrideState::CLEAR: return Chorus::PatchAction::Clear;
    }
    return std::nullopt;
}

std::optional<Chorus::MessageRole> message_role(ChorusRole::Value value) {
    switch (value) {
    case ChorusRole::SYSTEM: return Chorus::MessageRole::System;
    case ChorusRole::USER: return Chorus::MessageRole::User;
    case ChorusRole::ASSISTANT: return Chorus::MessageRole::Assistant;
    }
    return std::nullopt;
}

std::optional<Chorus::ExecutionMode> execution_mode(ChorusExecution::Value value) {
    switch (value) {
    case ChorusExecution::SHARED: return Chorus::ExecutionMode::Shared;
    case ChorusExecution::EXCLUSIVE: return Chorus::ExecutionMode::Exclusive;
    }
    return std::nullopt;
}

std::optional<Chorus::ConstraintFormat> constraint_format(ChorusConstraintFormat::Value value) {
    switch (value) {
    case ChorusConstraintFormat::GBNF: return Chorus::ConstraintFormat::Gbnf;
    case ChorusConstraintFormat::JSON_SCHEMA: return Chorus::ConstraintFormat::JsonSchema;
    case ChorusConstraintFormat::REGEX: return Chorus::ConstraintFormat::Regex;
    case ChorusConstraintFormat::LARK: return Chorus::ConstraintFormat::Lark;
    }
    return std::nullopt;
}

template <typename T>
bool set_action(Chorus::ConfigPatch<T>& target, ChorusOverrideState::Value state, const T& value, const char* name, std::string& error) {
    const auto action = patch_action(state);
    if (!action) { error = std::string(name) + "_state is invalid."; return false; }
    target.action = *action;
    if (*action == Chorus::PatchAction::Set)
        target.value = value;
    return true;
}

bool set_int32_action(Chorus::ConfigPatch<int32_t>& target, ChorusOverrideState::Value state, int64_t value, const char* name, std::string& error) {
    const auto action = patch_action(state);
    if (!action) { error = std::string(name) + "_state is invalid."; return false; }
    if (*action == Chorus::PatchAction::Set &&
        (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())) {
        error = std::string(name) + " is outside the supported int32 range.";
        return false;
    }
    target.action = *action;
    if (*action == Chorus::PatchAction::Set)
        target.value = static_cast<int32_t>(value);
    return true;
}

bool set_float_action(Chorus::ConfigPatch<float>& target, ChorusOverrideState::Value state, double value, const char* name, std::string& error) {
    const auto action = patch_action(state);
    if (!action) { error = std::string(name) + "_state is invalid."; return false; }
    if (*action == Chorus::PatchAction::Set && (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())) {
        error = std::string(name) + " must be finite and fit a 32-bit float.";
        return false;
    }
    target.action = *action;
    if (*action == Chorus::PatchAction::Set)
        target.value = static_cast<float>(value);
    return true;
}

std::vector<std::string> stop_values(const godot::PackedStringArray& source) {
    std::vector<std::string> out;
    out.reserve(source.size());
    for (int i = 0; i < source.size(); ++i)
        out.emplace_back(source[i].utf8().get_data());
    return out;
}

bool copy_common(const ChorusInferenceRequest& source, Chorus::InferenceRequest& target, std::string& error) {
    if (source.get_priority() < std::numeric_limits<int>::min() || source.get_priority() > std::numeric_limits<int>::max()) {
        error = "priority is outside the supported int range.";
        return false;
    }
    const auto execution = execution_mode(source.get_execution());
    if (!execution) { error = "execution is invalid."; return false; }
    target.prompt = std::string(source.get_content().utf8().get_data());
    const godot::String session = source.get_session();
    if (!session.is_empty())
        target.session_id = std::string(session.utf8().get_data());
    target.priority = static_cast<int>(source.get_priority());
    target.execution = *execution;
    return true;
}

} // namespace

std::optional<Chorus::GenerationConfigPatch> generation_patch_from_request(const ChorusRequest& request, std::string& error) {
    Chorus::GenerationConfigPatch patch;
    if (!set_int32_action(patch.max_tokens, request.get_max_tokens_state(), request.get_max_tokens(), "max_tokens", error))
        return std::nullopt;
    if (!set_float_action(patch.temperature, request.get_temperature_state(), request.get_temperature(), "temperature", error) ||
        !set_int32_action(patch.top_k, request.get_top_k_state(), request.get_top_k(), "top_k", error) ||
        !set_float_action(patch.top_p, request.get_top_p_state(), request.get_top_p(), "top_p", error))
        return std::nullopt;
    if (request.get_seed_state() == ChorusOverrideState::SET && request.get_seed() < 0) { error = "seed must be nonnegative when set."; return std::nullopt; }
    if (!set_action(patch.seed, request.get_seed_state(), static_cast<uint64_t>(request.get_seed()), "seed", error) || !set_float_action(patch.frequency_penalty, request.get_frequency_penalty_state(), request.get_frequency_penalty(), "frequency_penalty", error) || !set_float_action(patch.presence_penalty, request.get_presence_penalty_state(), request.get_presence_penalty(), "presence_penalty", error) || !set_action(patch.stop, request.get_stop_state(), stop_values(request.get_stop()), "stop", error) || !set_action(patch.show_thinking, request.get_show_thinking_state(), request.get_show_thinking(), "show_thinking", error)) return std::nullopt;
    const auto constraint_action = patch_action(request.get_constraint_state());
    if (!constraint_action) { error = "constraint_state is invalid."; return std::nullopt; }
    patch.constraint.action = *constraint_action;
    if (*constraint_action == Chorus::PatchAction::Set) {
        const auto format = constraint_format(request.get_constraint_format());
        if (!format) { error = "constraint_format is invalid."; return std::nullopt; }
        patch.constraint.value = {*format, std::string(request.get_constraint_source().utf8().get_data())};
    }
    auto converted = variant_to_option_value(request.get_provider_options(), error);
    if (!converted) { error = "provider_options " + error; return std::nullopt; }
    patch.provider_options = std::get<Chorus::ProviderOptionMap>(std::move(*converted));
    const auto erasures = request.get_provider_option_erasures();
    patch.provider_option_erasures.reserve(erasures.size());
    for (int i = 0; i < erasures.size(); ++i)
        patch.provider_option_erasures.emplace_back(erasures[i].utf8().get_data());
    return patch;
}

std::variant<Chorus::GenerationRequest, std::string> generation_request_from_resource(const godot::Ref<ChorusRequest>& request) {
    if (request.is_null()) return std::string("request must not be null.");
    Chorus::GenerationRequest out;
    std::string error;
    if (!copy_common(**request, out, error)) return error;
    if (request->requires_nonempty_session() && request->get_session().is_empty())
        return std::string("ChorusRequest.chat requires a non-empty session.");
    out.stream = request->get_stream();
    auto patch = generation_patch_from_request(**request, error);
    if (!patch) return error;
    out.overrides = std::move(*patch);
    out.chat_template = std::string(request->get_chat_template().utf8().get_data());
    const auto inject = request->get_inject();
    out.inject.reserve(inject.size());
    for (int i = 0; i < inject.size(); ++i) {
        const godot::Ref<ChorusInjectedMessage> item = inject[i];
        if (item.is_null()) return std::string("inject contains a null ChorusInjectedMessage.");
        const auto role = message_role(item->get_role());
        if (!role) return std::string("inject role is invalid.");
        if (item->get_depth() < std::numeric_limits<int32_t>::min() || item->get_depth() > std::numeric_limits<int32_t>::max()) return std::string("inject depth is outside the supported int32 range.");
        out.inject.push_back({{*role, Chorus::MessageContent::text(std::string(item->get_content().utf8().get_data()))}, static_cast<int32_t>(item->get_depth())});
    }
    return out;
}

std::variant<Chorus::EmbeddingRequest, std::string> embedding_request_from_resource(const godot::Ref<ChorusEmbeddingRequest>& request) {
    if (request.is_null()) return std::string("request must not be null.");
    Chorus::EmbeddingRequest out;
    std::string error;
    if (!copy_common(**request, out, error)) return error;
    return out;
}

} // namespace godot_chorus

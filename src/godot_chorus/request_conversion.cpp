#include "godot_chorus/request_conversion.hpp"

#include <cmath>
#include <limits>

#include "godot_chorus/option_conversion.hpp"

namespace godot_chorus {
namespace {

std::optional<Chorus::MessageRole> message_role(ChorusRole::Value value) {
    switch (value) {
    case ChorusRole::SYSTEM:
        return Chorus::MessageRole::System;
    case ChorusRole::USER:
        return Chorus::MessageRole::User;
    case ChorusRole::ASSISTANT:
        return Chorus::MessageRole::Assistant;
    }
    return std::nullopt;
}

std::optional<Chorus::ExecutionMode> execution_mode(ChorusExecution::Value value) {
    switch (value) {
    case ChorusExecution::SHARED:
        return Chorus::ExecutionMode::Shared;
    case ChorusExecution::EXCLUSIVE:
        return Chorus::ExecutionMode::Exclusive;
    }
    return std::nullopt;
}

std::optional<Chorus::ConstraintFormat> constraint_format(ChorusConstraintFormat::Value value) {
    switch (value) {
    case ChorusConstraintFormat::GBNF:
        return Chorus::ConstraintFormat::Gbnf;
    case ChorusConstraintFormat::JSON_SCHEMA:
        return Chorus::ConstraintFormat::JsonSchema;
    case ChorusConstraintFormat::REGEX:
        return Chorus::ConstraintFormat::Regex;
    case ChorusConstraintFormat::LARK:
        return Chorus::ConstraintFormat::Lark;
    }
    return std::nullopt;
}

bool set_int32(std::optional<int32_t>& target, int64_t value, const char* name, std::string& error) {
    if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
        error = std::string(name) + " is outside the supported int32 range.";
        return false;
    }
    target = static_cast<int32_t>(value);
    return true;
}

bool set_float(std::optional<float>& target, double value, const char* name, std::string& error) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
        error = std::string(name) + " must be finite and fit a 32-bit float.";
        return false;
    }
    target = static_cast<float>(value);
    return true;
}

bool copy_common(const ChorusInferenceRequest& source, Chorus::InferenceRequest& target, std::string& error) {
    if (source.get_priority() < std::numeric_limits<int>::min() ||
        source.get_priority() > std::numeric_limits<int>::max()) {
        error = "priority is outside the supported int range.";
        return false;
    }
    const auto execution = execution_mode(source.get_execution());
    if (!execution) {
        error = "execution is invalid.";
        return false;
    }
    target.prompt = std::string(source.get_content().utf8().get_data());
    const godot::String session = source.get_session();
    if (!session.is_empty())
        target.session_id = std::string(session.utf8().get_data());
    target.priority = static_cast<int>(source.get_priority());
    target.execution = *execution;
    return true;
}

} // namespace

std::optional<Chorus::GenerationConfig>
generation_config_from_request(const ChorusRequest& request, std::string& error) {
    Chorus::GenerationConfig options;
    if (request.has_max_tokens() && !set_int32(options.max_tokens, request.get_max_tokens(), "max_tokens", error))
        return std::nullopt;
    if (request.has_temperature() && !set_float(options.temperature, request.get_temperature(), "temperature", error))
        return std::nullopt;
    if (request.has_top_k() && !set_int32(options.top_k, request.get_top_k(), "top_k", error))
        return std::nullopt;
    if (request.has_top_p() && !set_float(options.top_p, request.get_top_p(), "top_p", error))
        return std::nullopt;
    if (request.has_seed()) {
        if (request.get_seed() < 0) {
            error = "seed must be nonnegative when set.";
            return std::nullopt;
        }
        options.seed = static_cast<uint64_t>(request.get_seed());
    }
    if (request.has_frequency_penalty() &&
        !set_float(options.frequency_penalty, request.get_frequency_penalty(), "frequency_penalty", error))
        return std::nullopt;
    if (request.has_presence_penalty() &&
        !set_float(options.presence_penalty, request.get_presence_penalty(), "presence_penalty", error))
        return std::nullopt;
    if (request.has_stop()) {
        const auto stop = request.get_stop();
        std::vector<std::string> values;
        values.reserve(stop.size());
        for (int i = 0; i < stop.size(); ++i)
            values.emplace_back(stop[i].utf8().get_data());
        options.stop = std::move(values);
    }
    if (request.has_show_thinking())
        options.show_thinking = request.get_show_thinking();
    if (request.has_constraint()) {
        if (request.is_unconstrained())
            options.constraint = Chorus::UnconstrainedOutput{};
        else {
            const auto format = constraint_format(request.get_constraint_format());
            if (!format) {
                error = "constraint_format is invalid.";
                return std::nullopt;
            }
            options.constraint =
                Chorus::OutputConstraint{*format, std::string(request.get_constraint_source().utf8().get_data())};
        }
    }
    auto converted = variant_to_option_value(request.get_provider_options(), error);
    if (!converted) {
        error = "provider_options " + error;
        return std::nullopt;
    }
    options.provider_options = std::get<Chorus::ProviderOptionMap>(std::move(*converted));
    return options;
}

std::variant<Chorus::GenerationRequest, std::string>
generation_request_from_resource(const godot::Ref<ChorusRequest>& request) {
    if (request.is_null())
        return std::string("request must not be null.");
    Chorus::GenerationRequest out;
    std::string error;
    if (!copy_common(**request, out, error))
        return error;
    if (request->requires_nonempty_session() && request->get_session().is_empty())
        return std::string("ChorusRequest.chat requires a non-empty session.");
    out.stream = request->get_stream();
    auto options = generation_config_from_request(**request, error);
    if (!options)
        return error;
    out.options = std::move(*options);
    if (request->has_chat_template()) {
        if (request->get_chat_template().is_empty())
            return std::string("chat_template must not be empty when selected.");
        out.chat_template = std::string(request->get_chat_template().utf8().get_data());
    }
    const auto inject = request->get_inject();
    out.inject.reserve(inject.size());
    for (int i = 0; i < inject.size(); ++i) {
        const godot::Ref<ChorusInjectedMessage> item = inject[i];
        if (item.is_null())
            return std::string("inject contains a null ChorusInjectedMessage.");
        const auto role = message_role(item->get_role());
        if (!role)
            return std::string("inject role is invalid.");
        if (item->get_depth() < std::numeric_limits<int32_t>::min() ||
            item->get_depth() > std::numeric_limits<int32_t>::max())
            return std::string("inject depth is outside the supported int32 range.");
        out.inject.push_back(
            {{*role, Chorus::MessageContent::text(std::string(item->get_content().utf8().get_data()))},
             static_cast<int32_t>(item->get_depth())}
        );
    }
    return out;
}

std::variant<Chorus::EmbeddingRequest, std::string>
embedding_request_from_resource(const godot::Ref<ChorusEmbeddingRequest>& request) {
    if (request.is_null())
        return std::string("request must not be null.");
    Chorus::EmbeddingRequest out;
    std::string error;
    if (!copy_common(**request, out, error))
        return error;
    return out;
}

} // namespace godot_chorus

#include "chorus_c/chorus_c_internal.hpp"

#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace chorus_c;

namespace {

bool to_cpp_constraint(chorus_constraint_format format, Chorus::ConstraintFormat& out) noexcept {
    switch (format) {
    case CHORUS_CONSTRAINT_GBNF:
        out = Chorus::ConstraintFormat::Gbnf;
        return true;
    case CHORUS_CONSTRAINT_JSON_SCHEMA:
        out = Chorus::ConstraintFormat::JsonSchema;
        return true;
    case CHORUS_CONSTRAINT_REGEX:
        out = Chorus::ConstraintFormat::Regex;
        return true;
    case CHORUS_CONSTRAINT_LARK:
        out = Chorus::ConstraintFormat::Lark;
        return true;
    }
    return false;
}

template <typename Value>
void set_request_provider_option(
    Chorus::GenerationRequest& request, const char* provider, const char* key, Value&& value
) {
    auto [namespace_it, inserted] = request.options.provider_options.try_emplace(provider, Chorus::ProviderOptionMap{});
    if (!inserted && !std::holds_alternative<Chorus::ProviderOptionMap>(namespace_it->second))
        namespace_it->second = Chorus::ProviderOptionMap{};
    std::get<Chorus::ProviderOptionMap>(namespace_it->second)[key] = std::forward<Value>(value);
}

} // namespace

extern "C" {

chorus_request* chorus_request_new(void) {
    try {
        return new chorus_request;
    } catch (...) {
        return nullptr;
    }
}

void chorus_request_free(chorus_request* req) {
    try {
        delete req;
    } catch (...) {
    }
}

chorus_error chorus_request_set_prompt(chorus_request* req, const char* prompt) {
    if (!req || !prompt)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.prompt = prompt; });
}

chorus_error chorus_request_set_session(chorus_request* req, const char* session) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] {
        if (session)
            req->value.session_id = std::string(session);
        else
            req->value.session_id.reset();
    });
}

chorus_error chorus_request_set_priority(chorus_request* req, int32_t priority) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.priority = priority; });
}

chorus_error chorus_request_set_execution_mode(chorus_request* req, chorus_execution_mode execution) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    Chorus::ExecutionMode mode;
    if (!to_cpp_execution_mode(execution, mode))
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.execution = mode; });
}

chorus_error chorus_request_set_stream(chorus_request* req, bool stream) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.stream = stream; });
}

chorus_error chorus_request_set_max_tokens(chorus_request* req, int32_t max_tokens) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.options.max_tokens = max_tokens; });
}

chorus_error chorus_request_clear_max_tokens(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.max_tokens.reset();
    return CHORUS_OK;
}

chorus_error chorus_request_set_temperature(chorus_request* req, float temperature) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.options.temperature = temperature; });
}

chorus_error chorus_request_clear_temperature(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.temperature.reset();
    return CHORUS_OK;
}

chorus_error chorus_request_set_top_k(chorus_request* req, int32_t top_k) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.options.top_k = top_k; });
}

chorus_error chorus_request_clear_top_k(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.top_k.reset();
    return CHORUS_OK;
}

chorus_error chorus_request_set_top_p(chorus_request* req, float top_p) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.options.top_p = top_p; });
}

chorus_error chorus_request_clear_top_p(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.top_p.reset();
    return CHORUS_OK;
}

chorus_error chorus_request_set_seed(chorus_request* req, uint64_t seed) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.options.seed = seed; });
}

chorus_error chorus_request_clear_seed(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.seed.reset();
    return CHORUS_OK;
}

chorus_error chorus_request_set_frequency_penalty(chorus_request* req, float penalty) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.options.frequency_penalty = penalty; });
}

chorus_error chorus_request_clear_frequency_penalty(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.frequency_penalty.reset();
    return CHORUS_OK;
}

chorus_error chorus_request_set_presence_penalty(chorus_request* req, float penalty) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.options.presence_penalty = penalty; });
}

chorus_error chorus_request_clear_presence_penalty(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.presence_penalty.reset();
    return CHORUS_OK;
}

chorus_error chorus_request_add_stop(chorus_request* req, const char* sequence) {
    if (!req || !sequence)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] {
        auto& stop = req->value.options.stop;
        if (!stop)
            stop.emplace();
        stop->emplace_back(sequence);
    });
}

chorus_error chorus_request_set_empty_stop(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.options.stop = std::vector<std::string>{}; });
}

chorus_error chorus_request_clear_stop(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.stop.reset();
    return CHORUS_OK;
}

chorus_error chorus_request_set_constraint(chorus_request* req, chorus_constraint_format format, const char* source) {
    if (!req || !source)
        return CHORUS_ERR_INVALID_REQUEST;
    Chorus::ConstraintFormat cpp_format = Chorus::ConstraintFormat::Gbnf;
    if (!to_cpp_constraint(format, cpp_format))
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] {
        req->value.options.constraint = Chorus::OutputConstraint{cpp_format, std::string(source)};
    });
}

chorus_error chorus_request_set_unconstrained(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.constraint = Chorus::UnconstrainedOutput{};
    return CHORUS_OK;
}

chorus_error chorus_request_clear_constraint(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.constraint.reset();
    return CHORUS_OK;
}

chorus_error chorus_request_set_show_thinking(chorus_request* req, bool show_thinking) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.show_thinking = show_thinking;
    return CHORUS_OK;
}

chorus_error chorus_request_clear_show_thinking(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.show_thinking.reset();
    return CHORUS_OK;
}

chorus_error
chorus_request_set_provider_option_float(chorus_request* req, const char* provider, const char* key, double value) {
    if (!req || !provider || !key)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { set_request_provider_option(req->value, provider, key, value); });
}

chorus_error
chorus_request_set_provider_option_int(chorus_request* req, const char* provider, const char* key, int64_t value) {
    if (!req || !provider || !key)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { set_request_provider_option(req->value, provider, key, value); });
}

chorus_error
chorus_request_set_provider_option_bool(chorus_request* req, const char* provider, const char* key, bool value) {
    if (!req || !provider || !key)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { set_request_provider_option(req->value, provider, key, value); });
}

chorus_error chorus_request_set_provider_option_string(
    chorus_request* req, const char* provider, const char* key, const char* value
) {
    if (!req || !provider || !key || !value)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { set_request_provider_option(req->value, provider, key, std::string(value)); });
}

chorus_error chorus_request_clear_provider_option(chorus_request* req, const char* provider, const char* key) {
    if (!req || !provider || !key)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] {
        auto& options = req->value.options.provider_options;
        auto namespace_it = options.find(provider);
        if (namespace_it != options.end() && std::holds_alternative<Chorus::ProviderOptionMap>(namespace_it->second)) {
            auto& entries = std::get<Chorus::ProviderOptionMap>(namespace_it->second);
            entries.erase(key);
            if (entries.empty())
                options.erase(namespace_it);
        }
    });
}

chorus_error chorus_request_clear_provider_options(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.options.provider_options.clear();
    return CHORUS_OK;
}

chorus_error
chorus_request_add_inject(chorus_request* req, chorus_message_role role, const char* content, int32_t depth) {
    if (!req || !content)
        return CHORUS_ERR_INVALID_REQUEST;
    const auto cpp_role = to_cpp_role(role);
    if (!cpp_role)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] {
        req->value.inject.push_back({{*cpp_role, Chorus::MessageContent::text(content)}, depth});
    });
}

chorus_error chorus_request_set_chat_template(chorus_request* req, const char* chat_template) {
    if (!req || !chat_template)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { req->value.chat_template = chat_template; });
}

chorus_error chorus_request_clear_chat_template(chorus_request* req) {
    if (!req)
        return CHORUS_ERR_INVALID_REQUEST;
    req->value.chat_template.reset();
    return CHORUS_OK;
}

} // extern "C"

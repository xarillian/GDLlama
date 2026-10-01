#include "chorus/engine_factory.hpp"
#include "chorus_c/chorus_c.hpp"

#include <exception>
#include <string>
#include <utility>

struct chorus_options {
    Chorus::ProviderOptionMap values;
};

using namespace chorus_c;

namespace {

bool to_cpp_provider(chorus_provider provider, Chorus::Provider& out, const char*& provider_id) noexcept {
    switch (provider) {
    case CHORUS_PROVIDER_LLAMA:
        out = Chorus::Provider::Llama;
        provider_id = "llama";
        return true;
    case CHORUS_PROVIDER_ECHO:
        out = Chorus::Provider::Echo;
        provider_id = "echo";
        return true;
    }
    return false;
}

bool to_cpp_log_level(chorus_log_level level, Chorus::LogLevel& out) noexcept {
    switch (level) {
    case CHORUS_LOG_DEBUG:
        out = Chorus::LogLevel::Debug;
        return true;
    case CHORUS_LOG_INFO:
        out = Chorus::LogLevel::Info;
        return true;
    case CHORUS_LOG_WARN:
        out = Chorus::LogLevel::Warn;
        return true;
    case CHORUS_LOG_ERROR:
        out = Chorus::LogLevel::Error;
        return true;
    case CHORUS_LOG_FATAL:
        out = Chorus::LogLevel::Fatal;
        return true;
    case CHORUS_LOG_OFF:
        out = Chorus::LogLevel::Off;
        return true;
    }
    return false;
}

void initialize_load_result(chorus_load_result* result) noexcept {
    if (result)
        *result = {-1, CHORUS_ERR_INVALID_REQUEST, nullptr};
}

} // namespace

extern "C" {

chorus_options* chorus_options_new(void) {
    try {
        return new chorus_options;
    } catch (...) {
        return nullptr;
    }
}

void chorus_options_free(chorus_options* opts) {
    try {
        delete opts;
    } catch (...) {
    }
}

chorus_error chorus_options_set_int(chorus_options* opts, const char* key, int64_t value) {
    if (!opts || !key)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { opts->values[std::string(key)] = value; });
}

chorus_error chorus_options_set_float(chorus_options* opts, const char* key, double value) {
    if (!opts || !key)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { opts->values[std::string(key)] = value; });
}

chorus_error chorus_options_set_bool(chorus_options* opts, const char* key, bool value) {
    if (!opts || !key)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { opts->values[std::string(key)] = value; });
}

chorus_error chorus_options_set_string(chorus_options* opts, const char* key, const char* value) {
    if (!opts || !key || !value)
        return CHORUS_ERR_INVALID_REQUEST;
    return guard_builder([&] { opts->values[std::string(key)] = std::string(value); });
}

chorus_error chorus_load(
    chorus_runtime* rt,
    chorus_provider provider,
    const char* model_path,
    const chorus_options* options,
    chorus_log_level min_log_level,
    chorus_load_result* out_result
) {
    initialize_load_result(out_result);
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    clear_result_storage(rt);
    if (!out_result)
        return invalid_request(rt, "out_result is required.");

    Chorus::Provider cpp_provider = Chorus::Provider::Echo;
    const char* provider_id = nullptr;
    if (!to_cpp_provider(provider, cpp_provider, provider_id))
        return invalid_request(rt, "Unknown provider.");

    Chorus::LogLevel cpp_log_level = Chorus::LogLevel::Warn;
    if (!to_cpp_log_level(min_log_level, cpp_log_level))
        return invalid_request(rt, "Unknown log level.");
    if (provider == CHORUS_PROVIDER_LLAMA && !model_path)
        return invalid_request(rt, "model_path is required for the Llama provider.");

    try {
        Chorus::ChorusConfig config;
        config.log_level = cpp_log_level;
        const std::string model_source = model_path ? model_path : "";
        config.model = Chorus::make_initial_model_spec(cpp_provider, model_source, model_source);
        if (options && !options->values.empty())
            config.provider_options[provider_id] = options->values;

        rt->result_strings.emplace_back();
        auto result = rt->value.load_engine(Chorus::make_engine(cpp_provider), config);
        rt->result_strings.front() = std::move(result.message);
        *out_result = {
            result.load_id,
            to_c_error(result.error),
            rt->result_strings.front().empty() ? nullptr : rt->result_strings.front().c_str()
        };
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while loading the engine.");
    }
}

bool chorus_cancel_load(chorus_runtime* rt, chorus_load_id id) {
    if (!rt)
        return false;
    try {
        return rt->value.cancel_load(id);
    } catch (...) {
        return false;
    }
}

chorus_load_id chorus_active_load_id(const chorus_runtime* rt) {
    if (!rt)
        return -1;
    try {
        return rt->value.active_load_id().value_or(-1);
    } catch (...) {
        return -1;
    }
}

} // extern "C"

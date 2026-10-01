#include "chorus_c/chorus_c.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace chorus_c {

std::optional<Chorus::MessageRole> to_cpp_role(chorus_message_role role) noexcept {
    switch (role) {
    case CHORUS_ROLE_SYSTEM:
        return Chorus::MessageRole::System;
    case CHORUS_ROLE_USER:
        return Chorus::MessageRole::User;
    case CHORUS_ROLE_ASSISTANT:
        return Chorus::MessageRole::Assistant;
    }
    return std::nullopt;
}

const char* error_name(chorus_error error) noexcept {
    switch (error) {
    case CHORUS_OK:
        return "None";
    case CHORUS_ERR_MODEL_LOAD:
        return "ModelLoad";
    case CHORUS_ERR_CONTEXT_INIT:
        return "ContextInit";
    case CHORUS_ERR_DECODE:
        return "Decode";
    case CHORUS_ERR_TOKENIZE:
        return "Tokenize";
    case CHORUS_ERR_INVALID_REQUEST:
        return "InvalidRequest";
    case CHORUS_ERR_ENGINE_NOT_READY:
        return "EngineNotReady";
    case CHORUS_ERR_CANCELLED:
        return "Cancelled";
    case CHORUS_ERR_UNSUPPORTED_MODEL_FORMAT:
        return "UnsupportedModelFormat";
    case CHORUS_ERR_UNSUPPORTED_FEATURE:
        return "UnsupportedFeature";
    case CHORUS_ERR_UNSUPPORTED_OPTION:
        return "UnsupportedOption";
    case CHORUS_ERR_SESSION_BUSY:
        return "SessionBusy";
    case CHORUS_ERR_UNKNOWN:
        return "Unknown";
    }
    return "Unknown";
}

chorus_error to_c_error(Chorus::ChorusError error) noexcept {
    switch (error) {
    case Chorus::ChorusError::None:
        return CHORUS_OK;
    case Chorus::ChorusError::ModelLoad:
        return CHORUS_ERR_MODEL_LOAD;
    case Chorus::ChorusError::ContextInit:
        return CHORUS_ERR_CONTEXT_INIT;
    case Chorus::ChorusError::Decode:
        return CHORUS_ERR_DECODE;
    case Chorus::ChorusError::Tokenize:
        return CHORUS_ERR_TOKENIZE;
    case Chorus::ChorusError::InvalidRequest:
        return CHORUS_ERR_INVALID_REQUEST;
    case Chorus::ChorusError::EngineNotReady:
        return CHORUS_ERR_ENGINE_NOT_READY;
    case Chorus::ChorusError::Cancelled:
        return CHORUS_ERR_CANCELLED;
    case Chorus::ChorusError::UnsupportedModelFormat:
        return CHORUS_ERR_UNSUPPORTED_MODEL_FORMAT;
    case Chorus::ChorusError::UnsupportedFeature:
        return CHORUS_ERR_UNSUPPORTED_FEATURE;
    case Chorus::ChorusError::UnsupportedOption:
        return CHORUS_ERR_UNSUPPORTED_OPTION;
    case Chorus::ChorusError::SessionBusy:
        return CHORUS_ERR_SESSION_BUSY;
    case Chorus::ChorusError::Unknown:
        return CHORUS_ERR_UNKNOWN;
    }
    return CHORUS_ERR_UNKNOWN;
}

bool to_cpp_execution_mode(chorus_execution_mode mode, Chorus::ExecutionMode& out) noexcept {
    switch (mode) {
    case CHORUS_EXECUTION_SHARED:
        out = Chorus::ExecutionMode::Shared;
        return true;
    case CHORUS_EXECUTION_EXCLUSIVE:
        out = Chorus::ExecutionMode::Exclusive;
        return true;
    }
    return false;
}

void replace_last_error(const chorus_runtime* rt, std::string_view detail) noexcept {
    if (!rt)
        return;
    try {
        rt->last_error.assign(detail.data(), detail.size());
    } catch (...) {
        rt->last_error.clear();
    }
}

void clear_last_error(const chorus_runtime* rt) noexcept {
    if (rt)
        rt->last_error.clear();
}

chorus_error invalid_request(const chorus_runtime* rt, std::string_view detail) noexcept {
    replace_last_error(rt, detail);
    return CHORUS_ERR_INVALID_REQUEST;
}

chorus_error unknown_exception(const chorus_runtime* rt, const char* detail) noexcept {
    replace_last_error(rt, detail ? std::string_view(detail) : std::string_view("Unknown C ABI failure."));
    return CHORUS_ERR_UNKNOWN;
}

char* copy_owned_string(std::string_view value) noexcept {
    if (value.size() == std::numeric_limits<size_t>::max())
        return nullptr;
    auto* copy = static_cast<char*>(std::malloc(value.size() + 1));
    if (!copy)
        return nullptr;
    std::memcpy(copy, value.data(), value.size());
    copy[value.size()] = '\0';
    return copy;
}

void clear_result_storage(chorus_runtime* rt) noexcept {
    if (!rt)
        return;
    rt->result_strings.clear();
}

} // namespace chorus_c

using namespace chorus_c;

namespace {

constexpr uint32_t kAbiVersion = 9;
// Keeps the waiting thread's deadline arithmetic far from clock overflow.
constexpr uint64_t kLongestWaitMicroseconds = 86'400'000'000;

const chorus_event kEmptyEvent{};
const chorus_log_field kEmptyLogField{};
const chorus_log_record kEmptyLogRecord{};

chorus_log_level to_c_log_level(Chorus::LogLevel level) noexcept {
    switch (level) {
    case Chorus::LogLevel::Debug:
        return CHORUS_LOG_DEBUG;
    case Chorus::LogLevel::Info:
        return CHORUS_LOG_INFO;
    case Chorus::LogLevel::Warn:
        return CHORUS_LOG_WARN;
    case Chorus::LogLevel::Error:
        return CHORUS_LOG_ERROR;
    case Chorus::LogLevel::Fatal:
        return CHORUS_LOG_FATAL;
    case Chorus::LogLevel::Off:
        return CHORUS_LOG_OFF;
    }
    return CHORUS_LOG_OFF;
}

chorus_event_kind to_c_event_kind(Chorus::RuntimeEvent::Kind kind) noexcept {
    switch (kind) {
    case Chorus::RuntimeEvent::Kind::StreamedToken:
        return CHORUS_EVENT_TOKEN;
    case Chorus::RuntimeEvent::Kind::StreamedReasoningToken:
        return CHORUS_EVENT_REASONING_TOKEN;
    case Chorus::RuntimeEvent::Kind::Complete:
        return CHORUS_EVENT_COMPLETE;
    case Chorus::RuntimeEvent::Kind::Embedding:
        return CHORUS_EVENT_EMBEDDING;
    case Chorus::RuntimeEvent::Kind::Error:
        return CHORUS_EVENT_ERROR;
    case Chorus::RuntimeEvent::Kind::HistoryTruncated:
        return CHORUS_EVENT_HISTORY_TRUNCATED;
    case Chorus::RuntimeEvent::Kind::EngineFailed:
        return CHORUS_EVENT_ENGINE_FAILED;
    case Chorus::RuntimeEvent::Kind::PromptRendered:
        return CHORUS_EVENT_PROMPT_RENDERED;
    case Chorus::RuntimeEvent::Kind::MessageTokenCount:
        return CHORUS_EVENT_MESSAGE_TOKEN_COUNT;
    case Chorus::RuntimeEvent::Kind::ModelLoadProgress:
        return CHORUS_EVENT_MODEL_LOAD_PROGRESS;
    case Chorus::RuntimeEvent::Kind::ModelLoaded:
        return CHORUS_EVENT_MODEL_LOADED;
    case Chorus::RuntimeEvent::Kind::ModelLoadFailed:
        return CHORUS_EVENT_MODEL_LOAD_FAILED;
    }
    return CHORUS_EVENT_ERROR;
}

chorus_load_phase to_c_load_phase(Chorus::LoadPhase phase) noexcept {
    switch (phase) {
    case Chorus::LoadPhase::ReleasingEngine:
        return CHORUS_LOAD_RELEASING_ENGINE;
    case Chorus::LoadPhase::LoadingModel:
        return CHORUS_LOAD_LOADING_MODEL;
    case Chorus::LoadPhase::InitializingEngine:
        return CHORUS_LOAD_INITIALIZING_ENGINE;
    }
    return CHORUS_LOAD_RELEASING_ENGINE;
}

void initialize_submit_result(chorus_submit_result* result) noexcept {
    if (result)
        *result = {-1, -1, -1, CHORUS_ERR_INVALID_REQUEST, nullptr};
}

void store_submit_results(
    chorus_runtime* rt, const std::vector<Chorus::SubmitResult>& results, chorus_submit_result* output
) {
    clear_result_storage(rt);
    for (const auto& result : results)
        rt->result_strings.push_back(result.message);
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& source = results[i];
        output[i] = {
            source.request_id,
            source.request_message_id.value_or(-1),
            source.response_message_id.value_or(-1),
            to_c_error(source.error),
            rt->result_strings[i].empty() ? nullptr : rt->result_strings[i].c_str(),
        };
    }
}

std::optional<Chorus::EmbeddingRequest> to_cpp_embedding_request(const chorus_embedding_request& request) {
    if (!request.content)
        return std::nullopt;
    Chorus::ExecutionMode mode;
    if (!to_cpp_execution_mode(request.execution, mode))
        return std::nullopt;
    Chorus::EmbeddingRequest output;
    output.prompt = request.content;
    output.priority = request.priority;
    output.execution = mode;
    if (request.session)
        output.session_id = request.session;
    return output;
}

} // namespace

extern "C" {

uint32_t chorus_abi_version(void) {
    return kAbiVersion;
}

const char* chorus_error_name(chorus_error error) {
    return error_name(error);
}

void chorus_string_free(char* str) {
    std::free(str);
}

chorus_runtime* chorus_runtime_new(void) {
    try {
        return new chorus_runtime;
    } catch (...) {
        return nullptr;
    }
}

void chorus_runtime_free(chorus_runtime* rt) {
    if (!rt)
        return;
    try {
        rt->value.stop_all();
    } catch (...) {
    }
    try {
        delete rt;
    } catch (...) {
    }
}

const char* chorus_last_error_message(const chorus_runtime* rt) {
    return rt ? rt->last_error.c_str() : "";
}

bool chorus_is_loaded(const chorus_runtime* rt) {
    if (!rt)
        return false;
    try {
        return rt->value.is_loaded();
    } catch (...) {
        return false;
    }
}

bool chorus_get_capabilities(const chorus_runtime* rt, chorus_capabilities* out_capabilities) {
    if (!rt || !out_capabilities)
        return false;
    *out_capabilities = {};
    try {
        const auto capabilities = rt->value.capabilities();
        if (!capabilities)
            return false;
        out_capabilities->streaming = capabilities->streaming;
        out_capabilities->cancellation = capabilities->cancellation;
        out_capabilities->embeddings = capabilities->embeddings;
        out_capabilities->prompt_rendering = capabilities->prompt_rendering;
        out_capabilities->message_token_counting = capabilities->message_token_counting;
        return true;
    } catch (...) {
        return false;
    }
}

void chorus_stop_all(chorus_runtime* rt) {
    if (!rt)
        return;
    try {
        rt->value.stop_all();
        clear_last_error(rt);
    } catch (const std::exception& error) {
        unknown_exception(rt, error.what());
    } catch (...) {
        unknown_exception(rt, "Unknown exception while stopping the engine.");
    }
}

chorus_error chorus_generate(chorus_runtime* rt, const chorus_request* req, chorus_submit_result* out_result) {
    initialize_submit_result(out_result);
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    clear_result_storage(rt);
    if (!req || !out_result)
        return invalid_request(rt, "request and out_result are required.");
    try {
        store_submit_results(rt, {rt->value.submit(req->value)}, out_result);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while submitting a request.");
    }
}

chorus_error chorus_generate_batch(
    chorus_runtime* rt, const chorus_request* const* reqs, size_t count, chorus_submit_result* out_results
) {
    if (out_results)
        for (size_t i = 0; i < count; ++i)
            initialize_submit_result(&out_results[i]);
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    clear_result_storage(rt);
    if ((count != 0 && (!reqs || !out_results)))
        return invalid_request(rt, "requests and out_results are required for a nonempty batch.");
    try {
        std::vector<Chorus::SubmitResult> results;
        results.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (!reqs[i]) {
                results.push_back(
                    {-1, std::nullopt, std::nullopt, Chorus::ChorusError::InvalidRequest, "request is required."}
                );
                continue;
            }
            results.push_back(rt->value.submit(reqs[i]->value));
        }
        store_submit_results(rt, results, out_results);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while submitting a request batch.");
    }
}

chorus_error chorus_embed(chorus_runtime* rt, const chorus_embedding_request* req, chorus_submit_result* out_result) {
    initialize_submit_result(out_result);
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    clear_result_storage(rt);
    if (!req || !out_result)
        return invalid_request(rt, "request and out_result are required.");
    try {
        const auto request = to_cpp_embedding_request(*req);
        if (!request)
            return invalid_request(rt, "embedding content and execution are required.");
        store_submit_results(rt, {rt->value.submit(*request)}, out_result);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while submitting an embedding request.");
    }
}

chorus_error chorus_embed_batch(
    chorus_runtime* rt, const chorus_embedding_request* reqs, size_t count, chorus_submit_result* out_results
) {
    if (out_results)
        for (size_t i = 0; i < count; ++i)
            initialize_submit_result(&out_results[i]);
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    clear_result_storage(rt);
    if (count != 0 && (!reqs || !out_results))
        return invalid_request(rt, "requests and out_results are required for a nonempty batch.");
    try {
        std::vector<Chorus::SubmitResult> results;
        results.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto request = to_cpp_embedding_request(reqs[i]);
            if (!request) {
                results.push_back(
                    {-1,
                     std::nullopt,
                     std::nullopt,
                     Chorus::ChorusError::InvalidRequest,
                     "embedding content and execution are required."}
                );
                continue;
            }
            results.push_back(rt->value.submit(*request));
        }
        store_submit_results(rt, results, out_results);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while submitting an embedding request batch.");
    }
}

chorus_error chorus_regenerate(chorus_runtime* rt, const chorus_request* req, chorus_submit_result* out_result) {
    initialize_submit_result(out_result);
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    clear_result_storage(rt);
    if (!req || !out_result)
        return invalid_request(rt, "request and out_result are required.");
    try {
        store_submit_results(rt, {rt->value.regenerate(req->value)}, out_result);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while regenerating a request.");
    }
}

bool chorus_wait(chorus_runtime* rt, uint64_t timeout_us) {
    if (!rt)
        return false;
    try {
        return rt->value.wait_for_events(std::chrono::microseconds(std::min(timeout_us, kLongestWaitMicroseconds)));
    } catch (const std::exception& error) {
        unknown_exception(rt, error.what());
        return false;
    } catch (...) {
        unknown_exception(rt, "Unknown exception while waiting for events.");
        return false;
    }
}

bool chorus_cancel(chorus_runtime* rt, chorus_request_id request_id) {
    if (!rt)
        return false;
    try {
        return rt->value.cancel(request_id);
    } catch (const std::exception& error) {
        unknown_exception(rt, error.what());
        return false;
    } catch (...) {
        unknown_exception(rt, "Unknown exception while cancelling a request.");
        return false;
    }
}

bool chorus_is_request_active(const chorus_runtime* rt, chorus_request_id request_id) {
    if (!rt)
        return false;
    try {
        return rt->value.is_request_active(request_id);
    } catch (...) {
        return false;
    }
}

chorus_request_id chorus_active_request_for_session(const chorus_runtime* rt, const char* session) {
    if (!rt || !session) {
        if (rt)
            invalid_request(rt, "session is required.");
        return -1;
    }
    try {
        const auto request = rt->value.active_request_for_session(session);
        return request.value_or(-1);
    } catch (const std::exception& error) {
        unknown_exception(rt, error.what());
        return -1;
    } catch (...) {
        unknown_exception(rt, "Unknown exception while finding the active request.");
        return -1;
    }
}

const chorus_event* chorus_poll(chorus_runtime* rt, size_t* out_count) {
    if (out_count)
        *out_count = 0;
    if (!rt)
        return &kEmptyEvent;
    clear_result_storage(rt);
    if (!out_count) {
        invalid_request(rt, "out_count is required.");
        return &kEmptyEvent;
    }

    try {
        rt->event_source = rt->value.poll();
        rt->events.clear();
        rt->events.resize(rt->event_source.size());
        for (size_t i = 0; i < rt->event_source.size(); ++i) {
            const auto& source = rt->event_source[i];
            auto& event = rt->events[i];
            event.kind = to_c_event_kind(source.kind);
            event.request_id = source.request_id;
            event.session = source.session_id ? source.session_id->c_str() : nullptr;
            event.text = source.text.c_str();
            event.error = to_c_error(source.error);
            event.reasoning = source.kind == Chorus::RuntimeEvent::Kind::Complete ? source.reasoning.c_str() : nullptr;
            event.message_id = source.message_id.value_or(-1);
            event.omitted_message_ids =
                source.omitted_message_ids.empty() ? nullptr : source.omitted_message_ids.data();
            event.omitted_message_id_count = source.omitted_message_ids.size();
            event.embedding = source.kind == Chorus::RuntimeEvent::Kind::Embedding && !source.embedding.empty()
                                  ? source.embedding.data()
                                  : nullptr;
            event.embedding_count = source.kind == Chorus::RuntimeEvent::Kind::Embedding ? source.embedding.size() : 0;
            event.token_count = source.token_count;
            event.load_id = source.load_id.value_or(-1);
            event.model_id = source.load_id ? source.model_id.c_str() : nullptr;
            event.load_phase =
                source.load_progress ? to_c_load_phase(source.load_progress->phase) : CHORUS_LOAD_RELEASING_ENGINE;
            const auto fraction = source.load_progress ? source.load_progress->fraction : std::nullopt;
            event.has_load_fraction = fraction.has_value();
            event.load_fraction = fraction.value_or(0.0f);
        }
        *out_count = rt->events.size();
        clear_last_error(rt);
        return rt->events.empty() ? &kEmptyEvent : rt->events.data();
    } catch (const std::exception& error) {
        rt->events.clear();
        rt->event_source.clear();
        unknown_exception(rt, error.what());
        return &kEmptyEvent;
    } catch (...) {
        rt->events.clear();
        rt->event_source.clear();
        unknown_exception(rt, "Unknown exception while polling events.");
        return &kEmptyEvent;
    }
}

const chorus_log_record* chorus_poll_logs(chorus_runtime* rt, size_t* out_count) {
    if (out_count)
        *out_count = 0;
    if (!rt)
        return &kEmptyLogRecord;
    if (!out_count) {
        invalid_request(rt, "out_count is required.");
        return &kEmptyLogRecord;
    }

    try {
        rt->log_source = rt->value.poll_logs();
        rt->log_fields.clear();
        rt->log_fields.resize(rt->log_source.size());
        rt->logs.clear();
        rt->logs.resize(rt->log_source.size());

        for (size_t i = 0; i < rt->log_source.size(); ++i) {
            const auto& source = rt->log_source[i];
            auto& fields = rt->log_fields[i];
            fields.resize(source.fields.size());
            for (size_t field_index = 0; field_index < source.fields.size(); ++field_index) {
                const auto& source_field = source.fields[field_index];
                auto& field = fields[field_index];
                field.key = source_field.first.c_str();
                switch (source_field.second.index()) {
                case 0:
                    field.type = CHORUS_FIELD_INT;
                    field.value.int_value = std::get<int64_t>(source_field.second);
                    break;
                case 1:
                    field.type = CHORUS_FIELD_FLOAT;
                    field.value.float_value = std::get<double>(source_field.second);
                    break;
                case 2:
                    field.type = CHORUS_FIELD_BOOL;
                    field.value.bool_value = std::get<bool>(source_field.second);
                    break;
                case 3:
                    field.type = CHORUS_FIELD_STRING;
                    field.value.string_value = std::get<std::string>(source_field.second).c_str();
                    break;
                default:
                    field.type = CHORUS_FIELD_STRING;
                    field.value.string_value = "";
                    break;
                }
            }

            auto& record = rt->logs[i];
            record.level = to_c_log_level(source.level);
            record.message = source.message.c_str();
            record.fields = fields.empty() ? &kEmptyLogField : fields.data();
            record.field_count = fields.size();
            record.request_id = source.request_id.value_or(-1);
            record.session = source.session_id ? source.session_id->c_str() : nullptr;
            record.produced_at = std::chrono::duration<double>(source.timestamp.time_since_epoch()).count();
        }

        *out_count = rt->logs.size();
        clear_last_error(rt);
        return rt->logs.empty() ? &kEmptyLogRecord : rt->logs.data();
    } catch (const std::exception& error) {
        rt->logs.clear();
        rt->log_fields.clear();
        rt->log_source.clear();
        unknown_exception(rt, error.what());
        return &kEmptyLogRecord;
    } catch (...) {
        rt->logs.clear();
        rt->log_fields.clear();
        rt->log_source.clear();
        unknown_exception(rt, "Unknown exception while polling logs.");
        return &kEmptyLogRecord;
    }
}

chorus_error chorus_render_prompt(chorus_runtime* rt, const chorus_request* req, chorus_submit_result* out_result) {
    initialize_submit_result(out_result);
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    clear_result_storage(rt);
    if (!req || !out_result)
        return invalid_request(rt, "request and out_result are required.");
    try {
        store_submit_results(rt, {rt->value.render_prompt(req->value)}, out_result);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while rendering a prompt.");
    }
}

chorus_error chorus_count_message_tokens(chorus_runtime* rt, const char* text, chorus_submit_result* out_result) {
    initialize_submit_result(out_result);
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    clear_result_storage(rt);
    if (!text || !out_result)
        return invalid_request(rt, "text and out_result are required.");
    try {
        store_submit_results(rt, {rt->value.count_message_tokens(Chorus::MessageContent::text(text))}, out_result);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while counting message tokens.");
    }
}

} // extern "C"

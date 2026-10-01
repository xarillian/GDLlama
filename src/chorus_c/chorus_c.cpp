#include <chorus_c/chorus_c.h>

#include "../host_settings/generation_defaults_codec.hpp"
#include "chorus/core/common.hpp"
#include "chorus/engine_factory.hpp"
#include "chorus/runtime/runtime.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <system_error>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

struct chorus_options {
    Chorus::ProviderOptionMap values;
};

struct chorus_request {
    Chorus::GenerationRequest value;
};

struct chorus_runtime {
    Chorus::ChorusRuntime value;
    Chorus::GenerationDefaults generation_defaults;
    mutable std::string last_error;

    std::vector<Chorus::RuntimeEvent> event_source;
    std::vector<chorus_event> events;

    std::deque<std::string> result_strings;

    std::vector<Chorus::LogRecord> log_source;
    std::vector<std::vector<chorus_log_field>> log_fields;
    std::vector<chorus_log_record> logs;
};

namespace {

constexpr uint32_t kAbiVersion = 9;
// Keeps the waiting thread's deadline arithmetic far from clock overflow.
constexpr uint64_t kLongestWaitMicroseconds = 86'400'000'000;

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

chorus_message_role to_c_role(Chorus::MessageRole role) noexcept {
    switch (role) {
    case Chorus::MessageRole::System:
        return CHORUS_ROLE_SYSTEM;
    case Chorus::MessageRole::User:
        return CHORUS_ROLE_USER;
    case Chorus::MessageRole::Assistant:
        return CHORUS_ROLE_ASSISTANT;
    }
    return CHORUS_ROLE_USER;
}
const chorus_event kEmptyEvent{};
const chorus_log_field kEmptyLogField{};
const chorus_log_record kEmptyLogRecord{};

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

chorus_turn_outcome to_c_outcome(Chorus::TurnOutcome outcome) noexcept {
    switch (outcome) {
    case Chorus::TurnOutcome::None:
        return CHORUS_TURN_NONE;
    case Chorus::TurnOutcome::Completed:
        return CHORUS_TURN_COMPLETED;
    case Chorus::TurnOutcome::Cancelled:
        return CHORUS_TURN_CANCELLED;
    case Chorus::TurnOutcome::Errored:
        return CHORUS_TURN_ERRORED;
    }
    return CHORUS_TURN_NONE;
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

chorus_error runtime_result(const chorus_runtime* rt, Chorus::ChorusError error) noexcept {
    const chorus_error mapped = to_c_error(error);
    replace_last_error(rt, error_name(mapped));
    return mapped;
}

template <typename Action> chorus_error guard_builder(Action&& action) noexcept {
    try {
        action();
        return CHORUS_OK;
    } catch (const std::bad_alloc&) {
        return CHORUS_ERR_UNKNOWN;
    } catch (...) {
        return CHORUS_ERR_UNKNOWN;
    }
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

namespace fs = std::filesystem;

enum class FileState { Missing, Present, Error };

struct FileRead {
    FileState state = FileState::Error;
    std::string bytes;
    std::string error;
};

FileRead read_settings_file(const fs::path& path) {
    std::error_code ec;
    const auto status = fs::status(path, ec);
    if (ec == std::errc::no_such_file_or_directory || (!ec && !fs::exists(status)))
        return {FileState::Missing, {}, {}};
    if (ec)
        return {FileState::Error, {}, "Cannot inspect settings file: " + ec.message()};
    if (!fs::is_regular_file(status))
        return {FileState::Error, {}, "Settings destination is not a regular file."};
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return {FileState::Error, {}, "Cannot open settings file for reading."};
    const auto length = fs::file_size(path, ec);
    if (ec || length > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        return {FileState::Error, {}, "Cannot size settings file."};
    std::string bytes(static_cast<size_t>(length), '\0');
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size()) || input.peek() != std::char_traits<char>::eof() ||
        input.bad())
        return {FileState::Error, {}, "Cannot read stable settings file."};
    return {FileState::Present, std::move(bytes), {}};
}

int open_exclusive(const fs::path& path) {
#if defined(_WIN32)
    return _wopen(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    return ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
#endif
}

bool write_all(int fd, std::string_view bytes) {
    while (!bytes.empty()) {
#if defined(_WIN32)
        const auto written = _write(fd, bytes.data(), static_cast<unsigned>(std::min(bytes.size(), size_t{INT_MAX})));
#else
        const auto written = ::write(fd, bytes.data(), bytes.size());
#endif
        if (written < 0 && errno == EINTR)
            continue;
        if (written <= 0)
            return false;
        bytes.remove_prefix(static_cast<size_t>(written));
    }
    return true;
}

void close_file(int fd) {
#if defined(_WIN32)
    _close(fd);
#else
    ::close(fd);
#endif
}

#if defined(_WIN32)
bool same_windows_file(const fs::path& path, const BY_HANDLE_FILE_INFORMATION& owned) {
    const HANDLE current = CreateFileW(
        path.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT,
        nullptr
    );
    if (current == INVALID_HANDLE_VALUE)
        return false;
    BY_HANDLE_FILE_INFORMATION observed{};
    const bool matched =
        GetFileInformationByHandle(current, &observed) && owned.dwVolumeSerialNumber == observed.dwVolumeSerialNumber &&
        owned.nFileIndexHigh == observed.nFileIndexHigh && owned.nFileIndexLow == observed.nFileIndexLow;
    CloseHandle(current);
    return matched;
}
#endif

bool remove_owned_file(const fs::path& path, int fd) {
#if defined(_WIN32)
    BY_HANDLE_FILE_INFORMATION owned{};
    const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (!GetFileInformationByHandle(handle, &owned) || !same_windows_file(path, owned))
        return false;
#else
    struct stat owned{}, current{};
    if (::fstat(fd, &owned) != 0 || ::lstat(path.c_str(), &current) != 0 || owned.st_dev != current.st_dev ||
        owned.st_ino != current.st_ino)
        return false;
#endif
    std::error_code ec;
    return fs::remove(path, ec) && !ec;
}

bool finish_owned_file(int fd, const fs::path& path) {
#if defined(_WIN32)
    BY_HANDLE_FILE_INFORMATION owned{};
    const bool identified = GetFileInformationByHandle(reinterpret_cast<HANDLE>(_get_osfhandle(fd)), &owned);
    const bool synced = _commit(fd) == 0;
#else
    struct stat owned{}, current{};
    const bool identified = ::fstat(fd, &owned) == 0;
    const bool synced = ::fsync(fd) == 0;
#endif
    if (!synced)
        remove_owned_file(path, fd);
#if defined(_WIN32)
    const bool closed = _close(fd) == 0;
    if ((!closed || !synced) && identified && same_windows_file(path, owned)) {
        std::error_code ignored;
        fs::remove(path, ignored);
    }
#else
    const bool closed = ::close(fd) == 0;
    if (!closed && identified && ::lstat(path.c_str(), &current) == 0 && owned.st_dev == current.st_dev &&
        owned.st_ino == current.st_ino)
        ::unlink(path.c_str());
#endif
    return synced && closed;
}

void replace_file(const fs::path& from, const fs::path& to, std::error_code& ec) {
#if defined(_WIN32)
    if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        ec = std::error_code(static_cast<int>(GetLastError()), std::system_category());
#else
    fs::rename(from, to, ec);
#endif
}

bool make_parent(const fs::path& path, std::string& error) {
    std::error_code ec;
    const auto parent = path.parent_path();
    if (!parent.empty())
        fs::create_directories(parent, ec);
    if (ec) {
        error = "Cannot create settings directory: " + ec.message();
        return false;
    }
    return true;
}

chorus_error inject_defaults(chorus_runtime* rt, Chorus::GenerationDefaults defaults) {
    // Prepare the adapter copy before changing the runtime, so neither side can diverge on allocation failure.
    Chorus::GenerationDefaults next = defaults;
    rt->value.set_generation_defaults(std::move(defaults));
    rt->generation_defaults = std::move(next);
    clear_last_error(rt);
    return CHORUS_OK;
}

chorus_error parse_and_inject(chorus_runtime* rt, std::string_view bytes) {
    auto parsed = chorus_host_settings::parse_generation_defaults(bytes);
    if (!parsed.ok())
        return invalid_request(rt, parsed.path + ": " + parsed.error);
    return inject_defaults(rt, std::move(parsed.defaults));
}

int open_temporary_file(const fs::path& path, fs::path& temporary) {
    static std::atomic<uint64_t> counter{0};
    for (int attempt = 0; attempt < 16; ++attempt) {
        temporary = path;
        temporary += ".chorus-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                     std::to_string(counter.fetch_add(1));
        const int fd = open_exclusive(temporary);
        if (fd >= 0 || errno != EEXIST)
            return fd;
    }
    return -1;
}

chorus_error load_defaults_file(chorus_runtime* rt, const fs::path& path) {
    auto read = read_settings_file(path);
    if (read.state == FileState::Error)
        return unknown_exception(rt, read.error.c_str());
    if (read.state == FileState::Present)
        return parse_and_inject(rt, read.bytes);

    std::string error;
    if (!make_parent(path, error))
        return unknown_exception(rt, error.c_str());
    fs::path temporary;
    const int fd = open_temporary_file(path, temporary);
    if (fd < 0)
        return unknown_exception(rt, "Cannot create temporary settings file.");
    constexpr std::string_view empty = "{\"version\":1,\"generation\":{}}";
    if (!write_all(fd, empty)) {
        remove_owned_file(temporary, fd);
        close_file(fd);
        return unknown_exception(rt, "Cannot write empty settings file.");
    }
    if (!finish_owned_file(fd, temporary))
        return unknown_exception(rt, "Cannot finish empty settings file.");
    std::error_code ec;
    fs::create_hard_link(temporary, path, ec);
    std::error_code cleanup;
    fs::remove(temporary, cleanup);
    if (cleanup)
        return unknown_exception(rt, ("Cannot remove settings temporary file: " + cleanup.message()).c_str());
    if (ec) {
        if (ec == std::errc::file_exists) {
            read = read_settings_file(path);
            if (read.state == FileState::Present)
                return parse_and_inject(rt, read.bytes);
            if (read.state == FileState::Error)
                return unknown_exception(rt, read.error.c_str());
        }
        return unknown_exception(rt, ("Cannot exclusively create settings file: " + ec.message()).c_str());
    }
    return inject_defaults(rt, {});
}

chorus_error save_defaults_file(chorus_runtime* rt, const fs::path& path) {
    const auto encoded = chorus_host_settings::serialize_generation_defaults(rt->generation_defaults);
    if (!encoded.ok())
        return invalid_request(rt, encoded.path + ": " + encoded.error);
    auto before = read_settings_file(path);
    if (before.state == FileState::Error)
        return unknown_exception(rt, before.error.c_str());
    if (before.state == FileState::Present) {
        auto parsed = chorus_host_settings::parse_generation_defaults(before.bytes);
        if (!parsed.ok())
            return invalid_request(rt, parsed.path + ": " + parsed.error);
    }
    std::string error;
    if (!make_parent(path, error))
        return unknown_exception(rt, error.c_str());

    fs::path temporary;
    const int fd = open_temporary_file(path, temporary);
    if (fd < 0)
        return unknown_exception(rt, "Cannot create temporary settings file.");
    if (!write_all(fd, encoded.json)) {
        remove_owned_file(temporary, fd);
        close_file(fd);
        return unknown_exception(rt, "Cannot write temporary settings file.");
    }
    if (!finish_owned_file(fd, temporary))
        return unknown_exception(rt, "Cannot finish temporary settings file.");
    const auto current = read_settings_file(path);
    if (current.state != before.state || (current.state == FileState::Present && current.bytes != before.bytes)) {
        std::error_code ec;
        fs::remove(temporary, ec);
        return unknown_exception(rt, "Settings destination changed before save; reload it first.");
    }
    std::error_code ec;
    replace_file(temporary, path, ec);
    if (ec) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        return unknown_exception(rt, ("Cannot replace settings file: " + ec.message()).c_str());
    }
    clear_last_error(rt);
    return CHORUS_OK;
}

void free_conversation_messages(chorus_conversation_message* messages, size_t count) noexcept {
    if (!messages)
        return;
    for (size_t i = 0; i < count; ++i)
        std::free(const_cast<char*>(messages[i].content));
    std::free(messages);
}

void clear_result_storage(chorus_runtime* rt) noexcept {
    if (!rt)
        return;
    rt->result_strings.clear();
}

void initialize_load_result(chorus_load_result* result) noexcept {
    if (result)
        *result = {-1, CHORUS_ERR_INVALID_REQUEST, nullptr};
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

void free_string_list(char** strings, size_t count) noexcept {
    if (!strings)
        return;
    for (size_t i = 0; i < count; ++i)
        std::free(strings[i]);
    std::free(strings);
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

chorus_error chorus_generation_defaults_load_file(chorus_runtime* rt, const char* path) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!path || !*path)
        return invalid_request(rt, "A nonempty settings path is required.");
    try {
        return load_defaults_file(rt, fs::path(path));
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Cannot load settings file.");
    }
}

chorus_error chorus_generation_defaults_apply_json(chorus_runtime* rt, const char* json, size_t byte_count) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!json || !byte_count)
        return invalid_request(rt, "Nonempty JSON bytes are required.");
    try {
        return parse_and_inject(rt, std::string_view(json, byte_count));
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Cannot apply settings JSON.");
    }
}

chorus_error chorus_generation_defaults_export_json(const chorus_runtime* rt, char** out_json) {
    if (out_json)
        *out_json = nullptr;
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!out_json)
        return invalid_request(rt, "out_json is required.");
    try {
        auto encoded = chorus_host_settings::serialize_generation_defaults(rt->generation_defaults);
        if (!encoded.ok())
            return invalid_request(rt, encoded.path + ": " + encoded.error);
        *out_json = copy_owned_string(encoded.json);
        if (!*out_json)
            return unknown_exception(rt, "Cannot allocate exported JSON.");
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Cannot export settings JSON.");
    }
}

chorus_error chorus_generation_defaults_save_file(chorus_runtime* rt, const char* path) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!path || !*path)
        return invalid_request(rt, "A nonempty settings path is required.");
    try {
        return save_defaults_file(rt, fs::path(path));
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Cannot save settings file.");
    }
}

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
            event.has_load_fraction = source.load_progress && source.load_progress->fraction.has_value();
            event.load_fraction = event.has_load_fraction ? *source.load_progress->fraction : 0.0f;
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

chorus_error chorus_history_import(
    chorus_runtime* rt, const char* session, const chorus_conversation_message* history, size_t count
) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!session || (count != 0 && !history))
        return invalid_request(rt, "session and history storage are required.");

    try {
        std::vector<Chorus::ConversationMessage> copied;
        copied.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto role = to_cpp_role(history[i].role);
            if (!role || !history[i].content || history[i].id < 0)
                return invalid_request(rt, "Every history message requires a nonnegative ID, known role, and content.");
            copied.push_back({history[i].id, {*role, Chorus::MessageContent::text(history[i].content)}});
        }
        auto error = rt->value.import_conversation_history(session, std::move(copied));
        if (error)
            return runtime_result(rt, *error);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while importing history.");
    }
}

chorus_error chorus_history_export(
    const chorus_runtime* rt, const char* session, chorus_conversation_message** out_messages, size_t* out_count
) {
    if (out_messages)
        *out_messages = nullptr;
    if (out_count)
        *out_count = 0;
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!session || !out_messages || !out_count)
        return invalid_request(rt, "session, out_messages, and out_count are required.");

    try {
        const auto history = rt->value.export_conversation_history(session);
        bool known = !history.empty();
        if (!known) {
            for (const auto& candidate : rt->value.list_conversations()) {
                if (candidate == session) {
                    known = true;
                    break;
                }
            }
        }
        if (!known) {
            clear_last_error(rt);
            return CHORUS_OK;
        }

        const size_t allocation_count = history.empty() ? 1 : history.size();
        auto* output = static_cast<chorus_conversation_message*>(
            std::calloc(allocation_count, sizeof(chorus_conversation_message))
        );
        if (!output)
            return unknown_exception(rt, "Unable to allocate the history snapshot.");

        for (size_t i = 0; i < history.size(); ++i) {
            const auto role = Chorus::message_role_name(history[i].message.role);
            const auto content = Chorus::joined_text(history[i].message.content);
            output[i].id = history[i].id;
            output[i].role = to_c_role(history[i].message.role);
            output[i].content = content ? copy_owned_string(*content) : nullptr;
            if (!content || !output[i].content) {
                free_conversation_messages(output, history.size());
                return unknown_exception(rt, "Unable to allocate the history snapshot.");
            }
        }

        *out_messages = output;
        *out_count = history.size();
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while exporting history.");
    }
}

void chorus_conversation_messages_free(chorus_conversation_message* messages, size_t count) {
    free_conversation_messages(messages, count);
}

chorus_error chorus_history_clear(chorus_runtime* rt, const char* session) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!session)
        return invalid_request(rt, "session is required.");
    try {
        auto error = rt->value.clear_conversation_history(session);
        if (error)
            return runtime_result(rt, *error);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while clearing history.");
    }
}

chorus_error chorus_history_edit_message(
    chorus_runtime* rt, const char* session, chorus_message_id message_id, const char* content
) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!session || !content || message_id < 0)
        return invalid_request(rt, "session, nonnegative message_id, and content are required.");
    try {
        auto error = rt->value.edit_message(session, message_id, Chorus::MessageContent::text(content));
        if (error)
            return runtime_result(rt, *error);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while editing history.");
    }
}

chorus_error chorus_list_conversations(const chorus_runtime* rt, char*** out_sessions, size_t* out_count) {
    if (out_sessions)
        *out_sessions = nullptr;
    if (out_count)
        *out_count = 0;
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!out_sessions || !out_count)
        return invalid_request(rt, "out_sessions and out_count are required.");

    try {
        const auto conversations = rt->value.list_conversations();
        if (conversations.empty()) {
            clear_last_error(rt);
            return CHORUS_OK;
        }
        auto* output = static_cast<char**>(std::calloc(conversations.size(), sizeof(char*)));
        if (!output)
            return unknown_exception(rt, "Unable to allocate the conversation list.");
        for (size_t i = 0; i < conversations.size(); ++i) {
            output[i] = copy_owned_string(conversations[i]);
            if (!output[i]) {
                free_string_list(output, conversations.size());
                return unknown_exception(rt, "Unable to allocate the conversation list.");
            }
        }
        *out_sessions = output;
        *out_count = conversations.size();
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while listing conversations.");
    }
}

void chorus_string_list_free(char** strings, size_t count) {
    free_string_list(strings, count);
}

chorus_error chorus_reset_context(chorus_runtime* rt) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    try {
        auto error = rt->value.reset_context();
        if (error)
            return runtime_result(rt, *error);
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Unknown exception while resetting context.");
    }
}

chorus_turn_outcome chorus_last_turn_outcome(const chorus_runtime* rt, const char* session) {
    if (!rt || !session) {
        if (rt)
            invalid_request(rt, "session is required.");
        return CHORUS_TURN_NONE;
    }
    try {
        return to_c_outcome(rt->value.last_turn_outcome(session));
    } catch (const std::exception& error) {
        unknown_exception(rt, error.what());
        return CHORUS_TURN_NONE;
    } catch (...) {
        unknown_exception(rt, "Unknown exception while reading the turn outcome.");
        return CHORUS_TURN_NONE;
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

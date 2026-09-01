#include "chorus/providers/llama/llama_log_bridge.hpp"

#include "llama.h"

#include <map>
#include <mutex>
#include <string_view>
#include <utility>

namespace Chorus {
namespace {

std::optional<LogLevel> map_ggml_level(int ggml_level) {
    switch (ggml_level) {
    case GGML_LOG_LEVEL_DEBUG:
        return LogLevel::Debug;
    case GGML_LOG_LEVEL_INFO:
        return LogLevel::Info;
    case GGML_LOG_LEVEL_WARN:
        return LogLevel::Warn;
    case GGML_LOG_LEVEL_ERROR:
        return LogLevel::Error;
    default:
        // `GGML_LOG_LEVEL_NONE` is llama's "not for a reader" marker;
        // `GGML_LOG_LEVEL_CONT` never opens a line.
        return std::nullopt;
    }
}

void trim_trailing_newlines(std::string& text) {
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
        text.pop_back();
}

/*
 * The one owner of llama's process-global hook.
 *
 * Everything here is guarded by `Registry::mutex`: llama may log from its
 * own threads, and `llama_log_set` is documented as not thread-safe, so
 * registration and delivery are serialized against each other.
 */
struct Registry {
    std::mutex mutex;
    std::map<uint64_t, Logger> loggers;
    LlamaLogAssembler assembler;
    uint64_t next_id = 1;
    bool installed = false;
    ggml_log_callback previous_callback = nullptr;
    void* previous_user_data = nullptr;
};

Registry& registry() {
    // Function-local so the registry outlives any static engine and is built
    // on first use, never during another translation unit's static init.
    static Registry instance;
    return instance;
}

void deliver(ggml_log_level level, const char* text, void* /*user_data*/) noexcept {
    try {
        Registry& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        for (const auto& record : reg.assembler.feed(static_cast<int>(level), text)) {
            for (const auto& entry : reg.loggers)
                entry.second.log(record.level, record.message, record.fields);
        }
    } catch (...) {
        // Exceptions cannot cross the vendor's C callback boundary.
    }
}

} // namespace

std::vector<LogRecord> LlamaLogAssembler::feed(int ggml_level, const char* text) {
    if (text == nullptr || *text == '\0')
        return {};

    std::vector<LogRecord> completed;
    const bool is_continuation = ggml_level == GGML_LOG_LEVEL_CONT;
    if (!is_continuation) {
        const auto level = map_ggml_level(ggml_level);
        if (!level)
            return {};
        // Flush before replacing the level; otherwise an error can
        // disappear into a maskable `Chorus::LogLevel::Info` record.
        if (_has_pending) {
            if (auto stale = flush())
                completed.push_back(std::move(*stale));
        }
        _pending_level = *level;
        _has_pending = true;
    } else if (!_has_pending) {
        return {};
    }

    const LogLevel fragment_level = _pending_level;
    const std::string_view fragment(text);
    size_t start = 0;
    for (size_t newline = fragment.find('\n', start); newline != std::string_view::npos;
         newline = fragment.find('\n', start)) {
        if (!_has_pending) {
            _pending_level = fragment_level;
            _has_pending = true;
        }
        _pending.append(fragment.substr(start, newline - start));
        if (auto record = flush())
            completed.push_back(std::move(*record));
        start = newline + 1;
    }
    if (start < fragment.size()) {
        if (!_has_pending) {
            _pending_level = fragment_level;
            _has_pending = true;
        }
        _pending.append(fragment.substr(start));
    }
    return completed;
}

std::optional<LogRecord> LlamaLogAssembler::flush() {
    if (!_has_pending)
        return std::nullopt;

    std::string message = std::move(_pending);
    _pending.clear();
    _has_pending = false;

    // `LlamaLogAssembler::feed` removes `\n`; this trims the `\r` left by
    // CRLF framing.
    trim_trailing_newlines(message);
    if (message.empty())
        return std::nullopt;

    LogRecord record;
    record.level = _pending_level;
    record.message = std::move(message);
    return record;
}

std::shared_ptr<LlamaLogBridge> LlamaLogBridge::acquire(Logger logger) {
    return std::make_shared<LlamaLogBridge>(Registration{}, std::move(logger));
}

LlamaLogBridge::LlamaLogBridge(Registration, Logger logger) {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    _id = reg.next_id++;
    reg.loggers.emplace(_id, std::move(logger));
    if (!reg.installed) {
        llama_log_get(&reg.previous_callback, &reg.previous_user_data);
        llama_log_set(deliver, nullptr);
        reg.installed = true;
    }
}

LlamaLogBridge::~LlamaLogBridge() {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.loggers.erase(_id);
    if (!reg.loggers.empty() || !reg.installed)
        return;

    llama_log_set(reg.previous_callback, reg.previous_user_data);
    reg.installed = false;
    reg.previous_callback = nullptr;
    reg.previous_user_data = nullptr;
    // An unterminated line has no one left to reach; drop it rather than
    // prepend it to whatever the next engine logs.
    reg.assembler.flush();
}

} // namespace Chorus

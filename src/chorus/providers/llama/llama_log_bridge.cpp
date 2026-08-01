#include "chorus/providers/llama/llama_log_bridge.hpp"

#include "llama.h"

#include <iostream>
#include <map>
#include <mutex>
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
        // NONE is llama's "not for a reader" marker; CONT never opens a line.
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
 * Everything here is guarded by `mutex`: llama may log from its own threads,
 * and `llama_log_set` is documented as not thread-safe, so registration and
 * delivery are serialized against each other.
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

void deliver(ggml_log_level level, const char* text, void* /*user_data*/) {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    for (const auto& record : reg.assembler.feed(static_cast<int>(level), text)) {
        // A severe vendor record is owed one stderr line however many engines
        // are registered, so the fan-out below suppresses the per-logger echo
        // and it is written once here. Echoed only when some registration would
        // have kept it, which is what a lone logger would have done.
        if (record.level == LogLevel::Error || record.level == LogLevel::Fatal) {
            bool kept_by_anyone = false;
            for (const auto& entry : reg.loggers)
                kept_by_anyone = kept_by_anyone || entry.second.enabled(record.level);
            if (kept_by_anyone)
                std::cerr << format_log_record(record) << '\n';
        }
        // Multiplexed on purpose: llama's hook carries no per-engine context,
        // so with two live engines the choice is a duplicated line or a
        // missing one.
        for (const auto& entry : reg.loggers)
            entry.second.log(record.level, record.message, record.fields, StderrEcho::Suppress);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// LlamaLogAssembler
// ---------------------------------------------------------------------------

std::vector<LogRecord> LlamaLogAssembler::feed(int ggml_level, const char* text) {
    if (text == nullptr || *text == '\0')
        return {};

    std::vector<LogRecord> completed;
    const bool is_continuation = ggml_level == GGML_LOG_LEVEL_CONT;
    if (!is_continuation) {
        const auto level = map_ggml_level(ggml_level);
        if (!level)
            return {}; // NONE and anything llama adds later
        // A non-CONT fragment opens a new line. One arriving while a line is
        // still open means llama never terminated the old one; it goes out
        // as-is rather than absorbing this fragment and burying its level,
        // which could downgrade an error into a maskable Info line.
        if (_has_pending) {
            if (auto stale = flush())
                completed.push_back(std::move(*stale));
        }
        _pending_level = *level;
        _has_pending = true;
    } else if (!_has_pending) {
        return {}; // a continuation of nothing
    }

    _pending += text;
    if (!_pending.empty() && _pending.back() == '\n') {
        if (auto record = flush())
            completed.push_back(std::move(*record));
    }
    return completed;
}

std::optional<LogRecord> LlamaLogAssembler::flush() {
    if (!_has_pending)
        return std::nullopt;

    std::string message = std::move(_pending);
    _pending.clear();
    _has_pending = false;

    // llama formats its own '\n'; sinks add their own line breaks.
    trim_trailing_newlines(message);
    if (message.empty())
        return std::nullopt;

    LogRecord record;
    record.level = _pending_level;
    record.message = std::move(message);
    return record;
}

// ---------------------------------------------------------------------------
// LlamaLogBridge
// ---------------------------------------------------------------------------

LlamaLogBridge::LlamaLogBridge(Registration, Logger logger) {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    _id = reg.next_id++;
    reg.loggers.emplace(_id, logger.with_source(vendor_source));
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

std::shared_ptr<LlamaLogBridge> LlamaLogBridge::acquire(Logger logger) {
    return std::make_shared<LlamaLogBridge>(Registration{}, std::move(logger));
}

} // namespace Chorus

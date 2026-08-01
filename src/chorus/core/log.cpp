#include "chorus/core/log.hpp"

#include <iostream>
#include <utility>

namespace Chorus {
namespace {

void append_value(std::string& out, const LogValue& value) {
    if (const auto* integer = std::get_if<int64_t>(&value))
        out += std::to_string(*integer);
    else if (const auto* number = std::get_if<double>(&value))
        out += std::to_string(*number);
    else if (const auto* flag = std::get_if<bool>(&value))
        out += *flag ? "true" : "false";
    else
        out += std::get<std::string>(value);
}

} // namespace

const char* log_level_name(LogLevel level) {
    switch (level) {
    case LogLevel::Debug:
        return "DEBUG";
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Warn:
        return "WARNING";
    case LogLevel::Error:
        return "ERROR";
    case LogLevel::Fatal:
        return "FATAL";
    case LogLevel::Off:
        return "OFF"; // a threshold, never a record's own level
    }
    return "UNKNOWN"; // unreachable for a declared level; keeps a widened enum honest
}

std::string format_log_record(const LogRecord& record) {
    std::string line = "[Chorus] ";
    line += log_level_name(record.level);
    line += ": ";
    line += record.message;

    // Identity reads as a field like any other, so one glance finds it in a
    // line and one parse finds it in a log file.
    const bool has_identity = record.request_id.has_value() || record.session_id.has_value();
    if (!record.fields.empty() || has_identity) {
        line += " (";
        bool first = true;
        const auto separate = [&] {
            if (!first)
                line += ", ";
            first = false;
        };
        if (record.request_id) {
            separate();
            line += "request=" + std::to_string(*record.request_id);
        }
        if (record.session_id) {
            separate();
            line += "session=" + *record.session_id;
        }
        for (const auto& field : record.fields) {
            separate();
            line += field.first;
            line += '=';
            append_value(line, field.second);
        }
        line += ')';
    }
    return line;
}

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

Logger::Logger(LogSink sink, LogLevel minimum, std::string source)
    : _sink(std::move(sink)), _minimum(minimum), _source(std::move(source)) {}

bool Logger::enabled(LogLevel level) const {
    // LogLevel::Off sits above every real level, so a Logger left at it, or
    // handed it by a host, admits nothing.
    return level >= _minimum && level != LogLevel::Off;
}

void Logger::log(LogLevel level, std::string message, std::vector<LogField> fields, StderrEcho echo) const {
    if (!enabled(level))
        return;

    LogRecord record;
    record.level = level;
    record.message = std::move(message);
    record.fields = std::move(fields);
    record.request_id = _request_id;
    record.session_id = _session_id;
    record.source = _source;

    // Severe records take the synchronous path as well: a segfault mid-decode
    // still leaves evidence, at the price of a host seeing two failures a
    // session twice. '\n' rather than std::endl, since stderr is unbuffered.
    if (echo == StderrEcho::Severe && (level == LogLevel::Error || level == LogLevel::Fatal))
        std::cerr << format_log_record(record) << '\n';

    if (_sink)
        _sink(std::move(record));
}

void Logger::debug(std::string message, std::vector<LogField> fields) const {
    log(LogLevel::Debug, std::move(message), std::move(fields));
}

void Logger::info(std::string message, std::vector<LogField> fields) const {
    log(LogLevel::Info, std::move(message), std::move(fields));
}

void Logger::warn(std::string message, std::vector<LogField> fields) const {
    log(LogLevel::Warn, std::move(message), std::move(fields));
}

void Logger::error(std::string message, std::vector<LogField> fields) const {
    log(LogLevel::Error, std::move(message), std::move(fields));
}

void Logger::fatal(std::string message, std::vector<LogField> fields) const {
    log(LogLevel::Fatal, std::move(message), std::move(fields));
}

Logger Logger::for_request(RequestId id, std::optional<SessionId> session) const {
    Logger stamped = *this;
    stamped._request_id = id;
    stamped._session_id = std::move(session);
    return stamped;
}

Logger Logger::with_source(std::string source) const {
    Logger renamed = *this;
    renamed._source = std::move(source);
    return renamed;
}

// ---------------------------------------------------------------------------
// LogChannel
// ---------------------------------------------------------------------------

LogChannel::LogChannel(size_t capacity) : _capacity(capacity == 0 ? 1 : capacity) {}

void LogChannel::push(LogRecord record) {
    std::lock_guard<std::mutex> lock(_mutex);
    while (_records.size() >= _capacity) {
        _records.pop_front();
        ++_dropped;
    }
    _records.push_back(std::move(record));
}

std::vector<LogRecord> LogChannel::drain() {
    std::deque<LogRecord> taken;
    uint64_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        taken.swap(_records);
        dropped = std::exchange(_dropped, 0);
    }

    std::vector<LogRecord> records;
    records.reserve(taken.size() + (dropped > 0 ? 1 : 0));
    if (dropped > 0) {
        // Leads the batch: the loss happened before everything that survived.
        LogRecord report;
        report.level = LogLevel::Warn;
        report.message = "Log records dropped";
        report.fields = {{"count", static_cast<int64_t>(dropped)}};
        report.source = "runtime";
        records.push_back(std::move(report));
    }
    for (auto& record : taken)
        records.push_back(std::move(record));
    return records;
}

LogSink LogChannel::sink_for(std::shared_ptr<LogChannel> channel) {
    return [channel = std::move(channel)](LogRecord record) { channel->push(std::move(record)); };
}

} // namespace Chorus

#include "chorus/core/log.hpp"

#include <chrono>
#include <iostream>
#include <syncstream>
#include <utility>

namespace Chorus {
namespace {

void append_hex_escape(std::string& out, unsigned char byte) {
    static constexpr char digits[] = "0123456789ABCDEF";
    out += "\\x";
    out += digits[byte >> 4];
    out += digits[byte & 0x0F];
}

void append_text(std::string& out, const std::string& text, bool escape_field_delimiters) {
    for (const unsigned char byte : text) {
        switch (byte) {
        case '\\':
            out += "\\\\";
            continue;
        case '\n':
            out += "\\n";
            continue;
        case '\r':
            out += "\\r";
            continue;
        case '\t':
            out += "\\t";
            continue;
        default:
            break;
        }

        if (byte < 0x20 || byte == 0x7F) {
            append_hex_escape(out, byte);
            continue;
        }
        if (escape_field_delimiters && (byte == ',' || byte == '=' || byte == '(' || byte == ')'))
            out += '\\';
        out += static_cast<char>(byte);
    }
}

void append_value(std::string& out, const LogValue& value) {
    if (const auto* integer = std::get_if<int64_t>(&value))
        out += std::to_string(*integer);
    else if (const auto* number = std::get_if<double>(&value))
        out += std::to_string(*number);
    else if (const auto* flag = std::get_if<bool>(&value))
        out += *flag ? "true" : "false";
    else
        append_text(out, std::get<std::string>(value), true);
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
        return "OFF";
    }

    return "UNKNOWN"; // should be unreachable
}

std::string format_log_record(const LogRecord& record) {
    std::string line = "[Chorus] ";
    line += log_level_name(record.level);
    line += ": ";
    append_text(line, record.message, false);

    // Identity reads as a field like any other, so one glance finds it beside
    // the values that describe the event.
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
            line += "session=";
            append_text(line, *record.session_id, true);
        }
        for (const auto& field : record.fields) {
            separate();
            append_text(line, field.first, true);
            line += '=';
            append_value(line, field.second);
        }
        line += ')';
    }
    return line;
}

void write_log_record_to_stderr(const LogRecord& record) {
    std::osyncstream output(std::cerr);
    output << format_log_record(record) << '\n' << std::flush;
}

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

Logger::Logger(LogSink sink, LogLevel minimum) : _sink(std::move(sink)), _minimum(minimum) {}

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
    record.timestamp = std::chrono::system_clock::now();
    record.fields = std::move(fields);
    record.request_id = _request_id;
    record.session_id = _session_id;

    // Severe records take the synchronous path as well, so a crash mid-decode
    // still leaves evidence before the host drains the channel.
    if (echo == StderrEcho::Severe && level >= LogLevel::Error)
        write_log_record_to_stderr(record);

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

// ---------------------------------------------------------------------------
// LogChannel
// ---------------------------------------------------------------------------

LogChannel::LogChannel(size_t capacity) : _capacity(capacity == 0 ? 1 : capacity) {}

void LogChannel::push(LogRecord record) {
    std::lock_guard<std::mutex> lock(_mutex);
    while (_records.size() >= _capacity) {
        // A loss is an event like any other and is stamped where it happens.
        // Drains are what stop when a host stalls, so by the time one collects
        // the report its own clock reading can be seconds late.
        if (_dropped == 0)
            _first_drop = std::chrono::system_clock::now();

        _records.pop_front();
        ++_dropped;
    }

    _records.push_back(std::move(record));
}

std::vector<LogRecord> LogChannel::drain() {
    std::deque<LogRecord> taken;
    uint64_t dropped = 0;
    std::chrono::system_clock::time_point first_drop;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        taken.swap(_records);
        dropped = std::exchange(_dropped, 0);
        first_drop = _first_drop;
    }

    std::vector<LogRecord> records;
    records.reserve(taken.size() + (dropped > 0 ? 1 : 0));
    if (dropped > 0) {
        // The prefix describes the batch rather than taking part in the
        // surviving records' production order.
        LogRecord report;
        report.level = LogLevel::Warn;
        report.message = "Log records dropped";
        report.timestamp = first_drop;
        report.fields = {{"count", static_cast<int64_t>(dropped)}};
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

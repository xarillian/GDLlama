#include "chorus/core/log.hpp"

#include <chrono>
#include <utility>

namespace Chorus {

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

Logger::Logger(std::shared_ptr<LogChannel> records, LogLevel minimum)
    : _records(std::move(records)), _minimum(minimum) {}

bool Logger::enabled(LogLevel level) const {
    // LogLevel::Off sits above every real level, so a Logger left at it, or
    // handed it by a host, admits nothing.
    return _records && level >= _minimum && level != LogLevel::Off;
}

void Logger::log(LogLevel level, std::string message, std::vector<LogField> fields) const {
    if (!enabled(level))
        return;

    LogRecord record;
    record.level = level;
    record.message = std::move(message);
    record.timestamp = std::chrono::system_clock::now();
    record.fields = std::move(fields);
    record.request_id = _request_id;
    record.session_id = _session_id;

    _records->push(std::move(record));
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

} // namespace Chorus

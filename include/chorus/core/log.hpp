#pragma once

#include "chorus/core/identity.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace Chorus {

enum class LogLevel { Debug, Info, Warn, Error, Fatal, Off };

inline constexpr LogLevel log_level_default = LogLevel::Warn;

using LogValue = std::variant<int64_t, double, bool, std::string>;
using LogField = std::pair<std::string, LogValue>;

struct LogRecord {
    LogLevel level = LogLevel::Info;
    std::string message;
    std::chrono::system_clock::time_point timestamp;

    std::vector<LogField> fields;

    std::optional<RequestId> request_id;
    std::optional<SessionId> session_id;
};

class LogChannel;

/*
 * Produces structured records for a shared `LogChannel`.
 *
 * Each call at or above the configured threshold becomes one `LogRecord`.
 * A default-constructed `Logger` is disabled.
 */
class Logger {
  public:
    Logger() = default;
    Logger(std::shared_ptr<LogChannel> records, LogLevel minimum);

    /*
     * Whether a record at `level` would reach the channel.
     *
     * `Logger::log` already tests this before it touches its arguments, so
     * guard a call with it only when building the fields is itself expensive.
     */
    bool enabled(LogLevel level) const;

    void log(LogLevel level, std::string message, std::vector<LogField> fields = {}) const;

    void debug(std::string message, std::vector<LogField> fields = {}) const;
    void info(std::string message, std::vector<LogField> fields = {}) const;
    void warn(std::string message, std::vector<LogField> fields = {}) const;
    void error(std::string message, std::vector<LogField> fields = {}) const;
    void fatal(std::string message, std::vector<LogField> fields = {}) const;

    /// Returns a copy that stamps `id` and `session` on every record it produces.
    Logger for_request(RequestId id, std::optional<SessionId> session = std::nullopt) const;

  private:
    std::shared_ptr<LogChannel> _records;
    LogLevel _minimum = LogLevel::Off;
    std::optional<RequestId> _request_id;
    std::optional<SessionId> _session_id;
};

/*
 * The bounded buffer between producer threads and the host thread.
 *
 * Internally synchronized. A producer never waits for capacity: a full channel
 * drops its oldest record and counts the loss, and that count surfaces as batch
 * metadata on the next `LogChannel::drain`.
 */
class LogChannel {
  public:
    /// `capacity` counts records, and is raised to one if given as zero.
    explicit LogChannel(size_t capacity = 1024);

    void push(LogRecord record);

    /*
     * Takes everything buffered and leaves the channel empty.
     *
     * Returns:
     *  - `std::vector<Chorus::LogRecord>`: an optional loss-report prefix,
     *    followed by the surviving records in production order. The prefix is
     *    batch metadata; its position does not participate in record ordering.
     */
    std::vector<LogRecord> drain();

  private:
    mutable std::mutex _mutex;
    std::deque<LogRecord> _records;
    size_t _capacity;
    uint64_t _dropped = 0;
    std::chrono::system_clock::time_point _first_drop; // read only while _dropped is non-zero
};

} // namespace Chorus

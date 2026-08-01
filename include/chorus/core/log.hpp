#pragma once

#include "chorus/core/identity.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace Chorus {

/*
 * How severe one diagnostic is, ascending.
 *
 * Doubles as a verbosity: a `Logger` reports at or above the level it holds.
 * `LogLevel::Off` is a threshold only and never appears on a record.
 */
enum class LogLevel { Debug, Info, Warn, Error, Fatal, Off };

/// Static name for a level, e.g. "WARNING" for `LogLevel::Warn`. Never null.
const char* log_level_name(LogLevel level);

/// The threshold a host reports at unless it names another.
inline constexpr LogLevel log_level_default = LogLevel::Warn;

/// What a field on a `LogRecord` may hold.
using LogValue = std::variant<int64_t, double, bool, std::string>;
using LogField = std::pair<std::string, LogValue>;

struct LogRecord {
    LogLevel level = LogLevel::Info;

    std::string message;
    std::vector<LogField> fields;

    // Set when the record concerns one request.
    // Unset when it concerns the engine as a whole.
    std::optional<RequestId> request_id;
    std::optional<SessionId> session_id;

    // Who produced it: e.g.: "llama", "echo", "runtime"
    std::string source;
};

/// Receives a record on the thread that produced it; must be thread-safe.
using LogSink = std::function<void(LogRecord)>;

/*
 * Whether one call takes the synchronous stderr path alongside the sink.
 *
 * `StderrEcho::Suppress` is for a caller that writes the line itself, such as
 * one fanning a record out to several `Logger`s that stderr is owed once.
 */
enum class StderrEcho { Severe, Suppress };

/// One record as a line, e.g. "[Chorus] ERROR: Decode failed (code=-1, sequence=3)".
std::string format_log_record(const LogRecord& record);

/*
 * Reports that something happened.
 *
 * Each call at or above the threshold becomes a `LogRecord` and goes to a
 * `LogSink`. Copy a `Logger` freely and hold it across threads.
 *
 * `LogLevel::Error` and `LogLevel::Fatal` also reach stderr as they are built,
 * so a crash leaves evidence.
 */
class Logger {
  public:
    Logger() = default;
    Logger(LogSink sink, LogLevel minimum, std::string source);

    /*
     * Whether a record at `level` would reach the sink.
     *
     * `Logger::log` already tests this before it touches its arguments, so
     * guard a call with it only when building the fields is itself expensive.
     */
    bool enabled(LogLevel level) const;

    /// The general form behind the level-named helpers.
    void log(
        LogLevel level, std::string message, std::vector<LogField> fields = {}, StderrEcho echo = StderrEcho::Severe
    ) const;

    /// `Logger::log` at one fixed level.
    void debug(std::string message, std::vector<LogField> fields = {}) const;
    void info(std::string message, std::vector<LogField> fields = {}) const;
    void warn(std::string message, std::vector<LogField> fields = {}) const;
    void error(std::string message, std::vector<LogField> fields = {}) const;
    void fatal(std::string message, std::vector<LogField> fields = {}) const;

    /// Returns a copy that stamps `id` and `session` on every record it produces.
    Logger for_request(RequestId id, std::optional<SessionId> session = std::nullopt) const;

    /// Returns a copy that stamps `source` on every record it produces, as when
    /// relaying a vendor library's output.
    Logger with_source(std::string source) const;

  private:
    LogSink _sink;
    LogLevel _minimum = LogLevel::Off;
    std::string _source;
    std::optional<RequestId> _request_id;
    std::optional<SessionId> _session_id;
};

/*
 * The bounded buffer between producer threads and the host thread.
 *
 * Internally synchronized. A producer never blocks on a host: at capacity the
 * channel drops its oldest record and counts the loss, and that count surfaces
 * as its own record on the next `LogChannel::drain`.
 */
class LogChannel {
  public:
    /// `capacity` counts records, and is raised to one if given as zero.
    explicit LogChannel(size_t capacity = 1024);

    /// Buffers one record, dropping the oldest at capacity. Never blocks.
    void push(LogRecord record);

    /*
     * Takes everything buffered and leaves the channel empty.
     *
     * Returns:
     *  - `std::vector<Chorus::LogRecord>`: the buffered records in production
     *    order, led by one drop report when records were lost since the last
     *    drain.
     */
    std::vector<LogRecord> drain();

    /*
     * Returns:
     *  - `Chorus::LogSink`
     */
    static LogSink sink_for(std::shared_ptr<LogChannel> channel);

  private:
    mutable std::mutex _mutex;
    std::deque<LogRecord> _records;
    size_t _capacity;
    uint64_t _dropped = 0;
};

} // namespace Chorus

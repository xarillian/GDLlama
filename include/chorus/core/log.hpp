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

/*
 * One diagnostic, as the producer stated it.
 *
 * `LogRecord::message` is stable text: two occurrences of one failure carry the
 * same message and differ only in `LogRecord::fields`. Values go in the fields
 * and are never interpolated into the message.
 */
struct LogRecord {
    LogLevel level = LogLevel::Info;
    std::string message;
    std::vector<LogField> fields;

    // Set when the record concerns one request, unset when it concerns the
    // engine as a whole.
    std::optional<RequestId> request_id;
    std::optional<SessionId> session_id;

    std::string source; // who produced it: "llama", "echo", "runtime"
};

/*
 * Where a record goes once a producer has built it.
 *
 * Invoked on the thread that produced the record, so an implementation must be
 * thread-safe.
 */
using LogSink = std::function<void(LogRecord)>;

/// One record as a line: "[Chorus] ERROR: Decode failed (code=-1, sequence=3)".
std::string format_log_record(const LogRecord& record);

/*
 * A producer's handle on the log channel.
 *
 * Cheap to copy and safe to hold across threads. Carries the sink, the
 * threshold, and the producer's name. A default-constructed `Logger` holds
 * `LogLevel::Off` and discards everything, so a provider that was never given
 * one still runs.
 *
 * Records at `LogLevel::Error` and `LogLevel::Fatal` go to stderr as they are
 * produced, in addition to the sink; `Logger::without_stderr_echo` opts out.
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

    /// Builds a record and hands it to the sink. Below the threshold, does nothing.
    void log(LogLevel level, std::string message, std::vector<LogField> fields = {}) const;

    /// `Logger::log` at one fixed level.
    void debug(std::string message, std::vector<LogField> fields = {}) const;
    void info(std::string message, std::vector<LogField> fields = {}) const;
    void warn(std::string message, std::vector<LogField> fields = {}) const;
    void error(std::string message, std::vector<LogField> fields = {}) const;
    void fatal(std::string message, std::vector<LogField> fields = {}) const;

    /// Returns a copy that stamps `id` and `session` on every record it produces.
    Logger for_request(RequestId id, std::optional<SessionId> session = std::nullopt) const;

    /*
     * Returns a copy whose records carry `source` instead of this one's.
     *
     * For code that logs on behalf of something that is not itself, such as a
     * provider routing a vendor library's output. Sink, threshold, and request
     * identity carry over unchanged.
     */
    Logger with_source(std::string source) const;

    /*
     * Returns a copy that writes no record to stderr, leaving that to the
     * caller.
     *
     * Sink, threshold, source, and request identity carry over unchanged.
     */
    Logger without_stderr_echo() const;

  private:
    LogSink _sink;
    LogLevel _minimum = LogLevel::Off;
    std::string _source;
    std::optional<RequestId> _request_id;
    std::optional<SessionId> _session_id;
    bool _echo_severe = true;
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
     * Returns a `LogSink` that pushes into `channel`.
     *
     * The sink shares ownership of `channel`, so it stays writable for as long
     * as any `Logger` holds it.
     */
    static LogSink sink_for(std::shared_ptr<LogChannel> channel);

  private:
    mutable std::mutex _mutex;
    std::deque<LogRecord> _records;
    size_t _capacity;
    uint64_t _dropped = 0;
};

} // namespace Chorus

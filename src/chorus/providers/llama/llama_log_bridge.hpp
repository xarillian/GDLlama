#pragma once

#include "chorus/core/log.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Chorus {

/*
 * Turns llama.cpp's stream of log fragments into whole records.
 *
 * llama writes a line in pieces: an opening fragment carrying the level, then
 * `GGML_LOG_LEVEL_CONT` fragments continuing it, and its own trailing newline.
 * Our records are whole messages, so fragments accumulate here until each
 * newline, and emerge as one record per physical line at the level of the
 * fragment that opened it.
 *
 * Not thread-safe on its own: the bridge that owns one serializes access.
 */
class LlamaLogAssembler {
  public:
    /*
     * Feeds one fragment. `ggml_level` is a `ggml_log_level`, taken as an int
     * so this header stays free of vendor types.
     *
     * A non-`GGML_LOG_LEVEL_CONT` fragment opens a new line. If one arrives
     * before the previous line ends, that line flushes as its own record
     * first so the new fragment's level is never buried under it.
     *
     * Returns:
     *  - non-empty `std::vector<Chorus::LogRecord>`: the completed whole lines, oldest first, with framing newlines
     * removed.
     *  - empty `std::vector<Chorus::LogRecord>`: no line completed because the fragment was empty, continued an open
     * line, or had no reader-facing level.
     */
    std::vector<LogRecord> feed(int ggml_level, const char* text);

    /*
     * Emits whatever an unterminated line has accumulated, and forgets it.
     *
     * Returns:
     *  - `Chorus::LogRecord`: the accumulated line, with framing newlines removed.
     *  - `std::nullopt`: no line was pending, or the pending line contained only framing.
     */
    std::optional<LogRecord> flush();

  private:
    std::string _pending;
    LogLevel _pending_level = LogLevel::Info;
    bool _has_pending = false;
};

/*
 * Routes llama.cpp's global log into the Chorus channel.
 *
 * llama.cpp's hook is process-wide, forwards to ggml's, and its setters are
 * documented as not thread-safe, so exactly one bridge owns it. Engines
 * register their loggers with the live bridge through
 * `Chorus::LlamaLogBridge::acquire`; the hook is installed when the first
 * registers and llama's previous callback restored when the last departs.
 * Registration and delivery are serialized, so after a handle deregisters
 * its logger cannot receive another callback.
 *
 * llama's hook carries no per-engine context, so with two live engines both
 * hosts see both engines' vendor output. Multiplexing to every registered
 * logger is the deliberate choice: a duplicated line is diagnosable and a
 * missing one is not.
 */
class LlamaLogBridge {
    // Public only so `std::make_shared` can reach the constructor; callers
    // cannot name `Registration`, so
    // `Chorus::LlamaLogBridge::acquire` remains the only way in.
    struct Registration {};

  public:
    /*
     * Registers `logger` and installs llama's hook if this is the first
     * registration.
     *
     * Returns a handle whose destruction deregisters the logger and, when it
     * was the last, restores llama's previous callback.
     */
    static std::shared_ptr<LlamaLogBridge> acquire(Logger logger);
    LlamaLogBridge(Registration, Logger logger);
    ~LlamaLogBridge();

    LlamaLogBridge(const LlamaLogBridge&) = delete;
    LlamaLogBridge& operator=(const LlamaLogBridge&) = delete;

  private:
    uint64_t _id = 0;
};

} // namespace Chorus

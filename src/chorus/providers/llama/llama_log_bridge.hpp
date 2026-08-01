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
 * Our records are whole messages, so fragments accumulate here until one
 * arrives ending in a newline, and emerge as a single record at the level of
 * the fragment that opened the line.
 *
 * Not thread-safe on its own: the bridge that owns one serializes access.
 */
class LlamaLogAssembler {
  public:
    /*
     * Feeds one fragment. `ggml_level` is a `ggml_log_level`, taken as an int
     * so this header stays free of vendor types.
     *
     * A non-CONT fragment opens a new line. One arriving while a line is
     * still open flushes the open line as its own record first, so a
     * fragment's level is never buried under the line before it.
     *
     * Returns:
     *  - `std::vector<Chorus::LogRecord>`: the whole lines this fragment
     *    completed, oldest first, trailing newlines trimmed.
     *  - empty: the line is still open, or the fragment was empty or at a
     *    level llama does not mean for a reader.
     */
    std::vector<LogRecord> feed(int ggml_level, const char* text);

    /// Emits whatever an unterminated line has accumulated, and forgets it.
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
 * register their loggers with the live bridge through `acquire`; the hook is
 * installed when the first registers and llama's previous callback restored
 * when the last departs, which is what keeps the shutdown fence intact.
 *
 * llama's hook carries no per-engine context, so with two live engines both
 * hosts see both engines' vendor output. Multiplexing to every registered
 * logger is the deliberate choice: a duplicated line is diagnosable and a
 * missing one is not. The synchronous stderr echo for severe records is the
 * exception: the bridge writes it once per record itself, so it never
 * multiplies with the number of registrations.
 */
class LlamaLogBridge {
    // Public only so make_shared can reach the constructor; a caller cannot
    // name this type, so `acquire` stays the only way in.
    struct Registration {};

  public:
    LlamaLogBridge(Registration, Logger logger);
    ~LlamaLogBridge();

    LlamaLogBridge(const LlamaLogBridge&) = delete;
    LlamaLogBridge& operator=(const LlamaLogBridge&) = delete;

    /*
     * Registers `logger` and installs llama's hook if this is the first
     * registration.
     *
     * Returns a handle whose destruction deregisters the logger and, when it
     * was the last, restores llama's previous callback. Records are signed
     * `source = "llama.cpp"`, distinct from the provider's own "llama", which
     * is what lets a host mute vendor chatter and keep Chorus's diagnostics.
     */
    static std::shared_ptr<LlamaLogBridge> acquire(Logger logger);

    /// The source every record routed through the bridge carries.
    static constexpr const char* vendor_source = "llama.cpp";

  private:
    uint64_t _id = 0;
};

} // namespace Chorus

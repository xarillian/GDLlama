#pragma once

#include <functional>
#include <string>

namespace Chorus {

enum class LogLevel { Debug, Info, Warn, Error, Fatal };

/*
 * Where a provider's diagnostics go.
 *
 * Providers may log from their own threads, so a sink must be thread-safe.
 * A sink receives the message alone and decides its own prefix and severity;
 * a message that spells out either says it twice.
 */
using LogCallback = std::function<void(LogLevel, const std::string&)>;

/*
 * Sends one message to `callback`, or to stderr when no sink is installed.
 *
 * The fallback is there so a provider can report a failure before any host has
 * wired itself up, not as a logging facility in its own right.
 */
void chorus_log(const LogCallback& callback, LogLevel level, const std::string& message);

} // namespace Chorus

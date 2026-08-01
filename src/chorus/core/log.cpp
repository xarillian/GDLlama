#include "chorus/core/log.hpp"

#include <iostream>

namespace Chorus {
namespace {

const char* level_name(LogLevel level) {
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
    }
    return "UNKNOWN"; // unreachable for a declared level; keeps a widened enum honest
}

} // namespace

void chorus_log(const LogCallback& callback, LogLevel level, const std::string& message) {
    if (callback) {
        callback(level, message);
        return;
    }

    // '\n' rather than std::endl: providers log from their worker threads, and
    // stderr is already unbuffered, so the per-line flush buys nothing.
    std::cerr << "[Chorus] " << level_name(level) << ": " << message << '\n';
}

} // namespace Chorus

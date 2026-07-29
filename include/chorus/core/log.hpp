#pragma once

#include <functional>
#include <iostream>
#include <string>

namespace Chorus {

enum class LogLevel { Debug, Info, Warn, Error, Fatal };

using LogCallback = std::function<void(LogLevel, const std::string&)>;

inline void chorus_log(const LogCallback& callback, LogLevel level, const std::string& message) {
    if (callback) {
        callback(level, message);
        return;
    }

    // Fallback: stderr with level prefix
    const char* prefix = "[Chorus] ";

    switch (level) {
    case LogLevel::Debug:
        std::cerr << prefix << "DEBUG: " << message << std::endl;
        break;
    case LogLevel::Info:
        std::cerr << prefix << "INFO: " << message << std::endl;
        break;
    case LogLevel::Warn:
        std::cerr << prefix << "WARNING: " << message << std::endl;
        break;
    case LogLevel::Error:
        std::cerr << prefix << "ERROR: " << message << std::endl;
        break;
    case LogLevel::Fatal:
        std::cerr << prefix << "FATAL: " << message << std::endl;
        break;
    }
}

} // namespace Chorus

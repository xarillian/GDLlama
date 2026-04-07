#pragma once

#include <functional>
#include <iostream>
#include <string>

namespace Chorus {

enum class LogLevel { Debug, Info, Warn, Error, Fatal };

using LogCallback = std::function<void(LogLevel, const std::string&)>;

inline void chorus_log(const LogCallback& cb, LogLevel level, const std::string& msg) {
    if (cb) {
        cb(level, msg);
        return;
    }

    // Fallback: stderr with level prefix
    const char* prefix = "[Chorus] ";

    switch (level) {
    case LogLevel::Debug:
        std::cerr << prefix << "DEBUG: " << msg << std::endl;
        break;
    case LogLevel::Info:
        std::cerr << prefix << msg << std::endl;
        break;
    case LogLevel::Warn:
        std::cerr << prefix << "WARNING: " << msg << std::endl;
        break;
    case LogLevel::Error:
        std::cerr << prefix << "ERROR: " << msg << std::endl;
        break;
    case LogLevel::Fatal:
        std::cerr << prefix << "FATAL: " << msg << std::endl;
        break;
    }
}

} // namespace Chorus

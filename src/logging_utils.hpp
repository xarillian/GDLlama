#ifndef LOGGING_UTILS_HPP
#define LOGGING_UTILS_HPP

#include <string>
#include <functional>
#include <sstream>

#include <log.h>

void initialize_logging();

void log_to_godot_and_file(ggml_log_level level, const std::string& msg);

#ifndef LOG_DISABLE_LOGS
    #define GDLOG_DEBUG(msg) log_to_godot_and_file(GGML_LOG_LEVEL_DEBUG, std::string(__func__) + ": " + (msg))
    #define GDLOG_INFO(msg) log_to_godot_and_file(GGML_LOG_LEVEL_INFO, std::string(__func__) + ": " + (msg))
    #define GDLOG_WARN(msg) log_to_godot_and_file(GGML_LOG_LEVEL_WARN, std::string(__func__) + ": " + (msg))
    #define GDLOG_ERROR(msg) log_to_godot_and_file(GGML_LOG_LEVEL_ERROR, std::string(__func__) + ": " + (msg))
# else
    #define GDLOG_DEBUG(msg) do {} while(0)
    #define GDLOG_INFO(msg) do {} while(0)
    #define GDLOG_WARN(msg) do {} while(0)
    #define GDLOG_ERROR(msg) do {} while(0)
# endif

#endif // LOGGING_UTILS_HPP
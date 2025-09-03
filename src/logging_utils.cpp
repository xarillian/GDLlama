#include "logging_utils.hpp"
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <sstream>

// Helper function to convert std::string to Godot String
godot::String string_std_to_gd(const std::string& std_str) {
    return godot::String(std_str.c_str());
}

void log_to_godot_and_file(ggml_log_level level, const std::string& msg) {
    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            LOG_ERR("%s\n", msg.c_str());
            godot::UtilityFunctions::push_error(string_std_to_gd(msg));
            break;
        case GGML_LOG_LEVEL_WARN:
            LOG_WRN("%s\n", msg.c_str());
            godot::UtilityFunctions::push_warning(string_std_to_gd(msg));
            break;
        case GGML_LOG_LEVEL_INFO:
        default:
            LOG_INF("%s\n", msg.c_str());
            godot::UtilityFunctions::print(string_std_to_gd(msg));
            break;
        case GGML_LOG_LEVEL_DEBUG:
            LOG_DBG("%s\n", msg.c_str());
            // We don't print debug messages to the Godot console to avoid spam
            break;
    }
}
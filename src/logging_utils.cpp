#include "logging_utils.hpp"
#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <sstream>

void initialize_logging() {
    static bool is_initialized = false;
    if (is_initialized) {
        return;
    }

    godot::DirAccess::make_dir_recursive_absolute("res://logs");

    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);
    common_log_set_file(common_log_main(), "logs/gdllama.log");
    
    is_initialized = true;
    
    GDLOG_INFO("--- GDLlama logging initialized ---");
    GDLOG_INFO("Welcome to Loguetown!\n");
}

// Helper function to convert std::string to Godot String
godot::String string_std_to_gd(const std::string& std_str) {
    return godot::String(std_str.c_str());
}

// Log to both the configured file and to the Godot console
void log_to_godot_and_file(ggml_log_level level, const std::string& msg) {
    initialize_logging();

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
            // We don't print info messages to the Godot console to avoid spam 
            break;
        case GGML_LOG_LEVEL_DEBUG:
            LOG_DBG("%s\n", msg.c_str());
            godot::UtilityFunctions::print(string_std_to_gd(msg));
            break;
    }
}
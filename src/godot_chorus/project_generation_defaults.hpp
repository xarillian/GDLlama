#pragma once

#include <string>
#include <godot_cpp/variant/string.hpp>
#include "chorus/runtime/runtime.hpp"

namespace godot_chorus {

void initialize_project_generation_defaults();
void shutdown_project_generation_defaults();
bool project_generation_defaults(Chorus::GenerationDefaults& out, std::string& error);
bool reload_project_generation_defaults(std::string& error);

enum class SettingsStatus { Ok, Invalid, SourceFailed, Conflict, IoError };
struct SettingsResult {
    SettingsStatus status;
    std::string message;
};
SettingsResult sync_project_generation_defaults();
SettingsResult save_project_generation_defaults();

} // namespace godot_chorus

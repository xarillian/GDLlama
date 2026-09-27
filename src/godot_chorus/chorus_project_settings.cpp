#include "godot_chorus/chorus_project_settings.hpp"

#include "godot_chorus/project_generation_defaults.hpp"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/string.hpp>

using namespace godot;

namespace {
Dictionary result(godot_chorus::SettingsResult value) {
    Dictionary out;
    out["status"] = static_cast<int>(value.status);
    out["message"] = String::utf8(value.message.data(), static_cast<int>(value.message.size()));
    return out;
}
} // namespace

Dictionary ChorusProjectSettings::sync_generation_defaults() {
    return result(godot_chorus::sync_project_generation_defaults());
}

Dictionary ChorusProjectSettings::reload_generation_defaults() {
    std::string error;
    if (godot_chorus::reload_project_generation_defaults(error)) return result({godot_chorus::SettingsStatus::Ok, {}});
    return result({godot_chorus::SettingsStatus::SourceFailed, error});
}

Dictionary ChorusProjectSettings::save_generation_defaults() {
    return result(godot_chorus::save_project_generation_defaults());
}

void ChorusProjectSettings::_bind_methods() {
    ClassDB::bind_method(D_METHOD("sync_generation_defaults"), &ChorusProjectSettings::sync_generation_defaults);
    ClassDB::bind_method(D_METHOD("reload_generation_defaults"), &ChorusProjectSettings::reload_generation_defaults);
    ClassDB::bind_method(D_METHOD("save_generation_defaults"), &ChorusProjectSettings::save_generation_defaults);
    BIND_ENUM_CONSTANT(OK);
    BIND_ENUM_CONSTANT(INVALID);
    BIND_ENUM_CONSTANT(SOURCE_FAILED);
    BIND_ENUM_CONSTANT(CONFLICT);
    BIND_ENUM_CONSTANT(IO_ERROR);
}

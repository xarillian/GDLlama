#include "godot_chorus/editor_import_guard.hpp"

#include <godot_cpp/classes/display_server.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/script_editor.hpp>

namespace godot_chorus {

void ChorusEditorImportGuard::_enter_tree() {
    const auto version = godot::Engine::get_singleton()->get_version_info();
    if (int64_t(version["major"]) != 4 || int64_t(version["minor"]) < 5 ||
        godot::DisplayServer::get_singleton()->get_name() != "headless") {
        return;
    }

    // Godot #111645 can run deferred documentation work after its owner is destroyed.
    // Opening built-in help joins that worker before the editor starts processing imports.
    godot::EditorInterface::get_singleton()->get_script_editor()->goto_help("class_name:Object");
}

} // namespace godot_chorus

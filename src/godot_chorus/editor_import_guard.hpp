#pragma once

#include <godot_cpp/classes/editor_plugin.hpp>

namespace godot_chorus {

class ChorusEditorImportGuard : public godot::EditorPlugin {
    GDCLASS(ChorusEditorImportGuard, godot::EditorPlugin)

  protected:
    static void _bind_methods() {}

  public:
    void _enter_tree() override;
};

} // namespace godot_chorus

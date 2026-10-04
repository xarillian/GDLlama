#include "godot_chorus/register_types.hpp"
#include "godot_chorus/chorus_project_settings.hpp"
#include "godot_chorus/chorus_types.hpp"
#include "godot_chorus/editor_import_guard.hpp"
#include "godot_chorus/godot_chorus.hpp"
#include "godot_chorus/project_generation_defaults.hpp"

#include <gdextension_interface.h>
#include <godot_cpp/classes/editor_plugin_registration.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

void initialize_chorus_module(ModuleInitializationLevel p_level) {
    if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
        GDREGISTER_INTERNAL_CLASS(godot_chorus::ChorusEditorImportGuard);
        EditorPlugins::add_by_type<godot_chorus::ChorusEditorImportGuard>();
        return;
    }
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    GDREGISTER_ABSTRACT_CLASS(ChorusRole);
    GDREGISTER_ABSTRACT_CLASS(ChorusExecution);
    GDREGISTER_ABSTRACT_CLASS(ChorusConstraintFormat);
    GDREGISTER_ABSTRACT_CLASS(ChorusInferenceRequest);
    GDREGISTER_CLASS(ChorusInjectedMessage);
    GDREGISTER_CLASS(ChorusMessage);
    GDREGISTER_CLASS(ChorusRequest);
    GDREGISTER_CLASS(ChorusEmbeddingRequest);
    GDREGISTER_ABSTRACT_CLASS(ChorusLoadPhase);
    GDREGISTER_CLASS(ChorusLoadResult);
    GDREGISTER_CLASS(ChorusSubmitResult);
    GDREGISTER_CLASS(ChorusResult);
    GDREGISTER_CLASS(ChorusGenerationUsage);
    GDREGISTER_CLASS(GodotChorus);
    GDREGISTER_CLASS(ChorusProjectSettings);
    GodotChorus::register_project_settings();
    godot_chorus::initialize_project_generation_defaults();
}

void uninitialize_chorus_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    godot_chorus::shutdown_project_generation_defaults();
}

extern "C" {
GDExtensionBool GDE_EXPORT llm_library_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization* r_initialization
) {
    godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

    init_obj.register_initializer(initialize_chorus_module);
    init_obj.register_terminator(uninitialize_chorus_module);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}
}

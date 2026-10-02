#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>

class ChorusProjectSettings : public godot::RefCounted {
    GDCLASS(ChorusProjectSettings, godot::RefCounted)

  public:
    enum Status { OK = 0, INVALID = 1, SOURCE_FAILED = 2, CONFLICT = 3, IO_ERROR = 4 };

    godot::Dictionary sync_generation_defaults();
    godot::Dictionary reload_generation_defaults();
    godot::Dictionary save_generation_defaults();

  protected:
    static void _bind_methods();
};

VARIANT_ENUM_CAST(ChorusProjectSettings::Status);

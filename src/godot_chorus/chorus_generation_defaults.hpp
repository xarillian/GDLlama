#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include "chorus/core/generation_config.hpp"

// Reusable, shared generation defaults for one or more GodotChorus nodes. Every
// scalar is an override flag paired with a typed value; disabled overrides
// contribute nothing when merged onto a request (spec 3c-D overlay semantics).
// The Godot-facing state is backed directly by a Chorus::GenerationConfigPatch
// so to_patch() reuses Task 1's overlay vocabulary rather than re-deriving it.
class ChorusGenerationDefaults : public godot::Resource {
    GDCLASS(ChorusGenerationDefaults, godot::Resource);

  protected:
    static void _bind_methods();

  public:
    enum ConstraintFormat {
        CONSTRAINT_FORMAT_GBNF,
        CONSTRAINT_FORMAT_JSON_SCHEMA,
        CONSTRAINT_FORMAT_REGEX,
        CONSTRAINT_FORMAT_LARK,
    };

    ChorusGenerationDefaults();

    void set_override_max_tokens(bool enabled);
    bool get_override_max_tokens() const;
    void set_max_tokens(int32_t value);
    int32_t get_max_tokens() const;

    void set_override_temperature(bool enabled);
    bool get_override_temperature() const;
    void set_temperature(float value);
    float get_temperature() const;

    void set_override_top_k(bool enabled);
    bool get_override_top_k() const;
    void set_top_k(int32_t value);
    int32_t get_top_k() const;

    void set_override_top_p(bool enabled);
    bool get_override_top_p() const;
    void set_top_p(float value);
    float get_top_p() const;

    void set_override_seed(bool enabled);
    bool get_override_seed() const;
    void set_seed(int64_t value);
    int64_t get_seed() const;

    void set_override_frequency_penalty(bool enabled);
    bool get_override_frequency_penalty() const;
    void set_frequency_penalty(float value);
    float get_frequency_penalty() const;

    void set_override_presence_penalty(bool enabled);
    bool get_override_presence_penalty() const;
    void set_presence_penalty(float value);
    float get_presence_penalty() const;

    void set_override_stop(bool enabled);
    bool get_override_stop() const;
    void set_stop(const godot::PackedStringArray& value);
    godot::PackedStringArray get_stop() const;

    void set_override_constraint(bool enabled);
    bool get_override_constraint() const;
    void set_constraint_format(ConstraintFormat format);
    ConstraintFormat get_constraint_format() const;
    void set_constraint_source(const godot::String& source);
    godot::String get_constraint_source() const;

    void set_override_thinking(bool enabled);
    bool get_override_thinking() const;
    void set_thinking(bool value);
    bool get_thinking() const;

    void set_backend_options(const godot::Dictionary& options);
    godot::Dictionary get_backend_options() const;

    // Converts the current Inspector state into the host-neutral overlay
    // patch. Disabled overrides map to Inherit, enabled ones to Set (an
    // enabled, empty stop array is therefore an explicit empty replacement).
    Chorus::GenerationConfigPatch to_patch() const;

  private:
    Chorus::GenerationConfigPatch _patch;
    godot::Dictionary _backend_options;
};

VARIANT_ENUM_CAST(ChorusGenerationDefaults::ConstraintFormat);

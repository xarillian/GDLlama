#pragma once

#include <variant>
#include <vector>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include "chorus/core/generation_config.hpp"
#include "chorus/runtime/runtime.hpp"

// Converts a Godot request Dictionary into a host-neutral Chorus::GenerationRequest,
// applying it as a per-request overlay on top of an already-resolved defaults
// GenerationConfig (the engine's implicit defaults with the shared
// ChorusGenerationDefaults resource patch already applied). See GodotChorus::generate()'s
// doc comment for the request dictionary's key vocabulary and its absent/null/present
// overlay semantics.
namespace godot_chorus {

// Returns the assembled request on success, or a human-readable rejection reason.
std::variant<Chorus::GenerationRequest, godot::String>
normalize_generation_request(const godot::Dictionary& request, const Chorus::GenerationConfig& defaults);

// The same normalization minus the prompt requirement (prompt left empty when
// absent) -- the overlay vocabulary for regenerate(), whose turn text comes
// from session history rather than the dictionary.
std::variant<Chorus::GenerationRequest, godot::String>
normalize_generation_overrides(const godot::Dictionary& request, const Chorus::GenerationConfig& defaults);

// Parses an inject Array of {role, content, depth?} Dictionaries. Shared by
// the 'inject' request key and render_chat_prompt's inject argument, so the
// two entry points cannot drift.
std::variant<std::vector<Chorus::InjectedMessage>, godot::String> normalize_inject_array(const godot::Array& entries);

} // namespace godot_chorus

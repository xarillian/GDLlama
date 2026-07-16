#pragma once

#include <variant>

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

} // namespace godot_chorus

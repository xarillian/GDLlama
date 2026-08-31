#pragma once

#include <variant>
#include <vector>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include "chorus/core/generation_config.hpp"
#include "chorus/runtime/runtime.hpp"

namespace godot_chorus {

/*
 * Converts a Godot request dictionary into a host-neutral generation request.
 *
 * The resulting `Chorus::GenerationRequest::overrides` remains a patch so
 * `Chorus::ChorusRuntime` can distinguish values supplied by the caller from
 * inherited host defaults.
 *
 * Returns:
 *  - `Chorus::GenerationRequest`: the normalized request.
 *  - `godot::String`: a human-readable rejection reason.
 */
std::variant<Chorus::GenerationRequest, godot::String> normalize_generation_request(const godot::Dictionary& request);

/*
 * Normalizes generation input without requiring a prompt.
 *
 * This accepts the shared request vocabulary used by regeneration, whose
 * turn text comes from session history when the prompt is absent.
 *
 * Returns:
 *  - `Chorus::GenerationRequest`: the normalized input.
 *  - `godot::String`: a human-readable rejection reason.
 */
std::variant<Chorus::GenerationRequest, godot::String> normalize_generation_input(const godot::Dictionary& request);

/*
 * Converts Godot injection entries into host-neutral messages.
 *
 * This parser is shared by the generation request and rendered-prompt paths.
 *
 * Returns:
 *  - `std::vector<Chorus::InjectedMessage>`: the normalized messages.
 *  - `godot::String`: a human-readable rejection reason.
 */
std::variant<std::vector<Chorus::InjectedMessage>, godot::String> normalize_inject_array(const godot::Array& entries);

} // namespace godot_chorus

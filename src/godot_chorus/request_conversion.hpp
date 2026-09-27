#pragma once

#include <optional>
#include <string>
#include <variant>

#include "chorus/runtime/runtime.hpp"
#include "godot_chorus/chorus_types.hpp"

namespace godot_chorus {

std::optional<Chorus::GenerationConfig> generation_config_from_request(const ChorusRequest& request, std::string& error);
std::variant<Chorus::GenerationRequest, std::string> generation_request_from_resource(const godot::Ref<ChorusRequest>& request);
std::variant<Chorus::EmbeddingRequest, std::string> embedding_request_from_resource(const godot::Ref<ChorusEmbeddingRequest>& request);

} // namespace godot_chorus

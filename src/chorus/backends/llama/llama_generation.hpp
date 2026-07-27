#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"
#include "chorus/core/generation_config.hpp"
#include "common.h"
#include "sampling.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace Chorus {

struct ResolvedLlamaGeneration {
    int32_t max_tokens = -1;
    common_params_sampling sampling;
    std::vector<std::string> stop;
};

std::variant<ResolvedLlamaGeneration, RequestRejection> resolve_llama_generation(const GenerationConfig& config);
std::variant<common_sampler_ptr, RequestRejection>
make_llama_sampler(const llama_model* model, ResolvedLlamaGeneration resolved);
std::optional<RequestRejection> validate_llama_generation(const GenerationConfig& config);
std::optional<RequestRejection> validate_llama_request(const ChorusRequest& request);
const std::vector<std::string>& llama_portable_generation_option_names();
const std::vector<std::string>& llama_backend_generation_option_names();

} // namespace Chorus

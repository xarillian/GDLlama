#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/generation_config.hpp"
#include "common.h"
#include "sampling.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace Chorus {

/*
 * Stores generation settings resolved for llama.cpp.
 *
 * `Chorus::ResolvedLlamaGeneration::sampling` contains the provider-ready sampler
 * configuration. Output length and stop sequences remain under Chorus control.
 */
struct ResolvedLlamaGeneration {
    int32_t max_tokens = -1;
    common_params_sampling sampling;
    std::vector<std::string> stop;
};

/*
 * Validates one request against Llama-specific request and generation rules.
 *
 * Checks that require a loaded model are deferred to `Chorus::make_llama_sampler`.
 * A rejection is returned directly and is not signalled through request events.
 *
 * Returns:
 *  - `std::nullopt`: `Chorus::ChorusRequest` satisfies the model-independent rules.
 *  - `Chorus::RequestRejection`: The request violates a model-independent rule.
 *
 * Errors:
 *  - `Chorus::ChorusError::InvalidRequest`: A supplied constraint is invalid.
 *  - `Chorus::ChorusError::UnsupportedFeature`: The constraint format is unsupported.
 *  - `Chorus::ChorusError::UnsupportedOption`: A request control or generation option is invalid.
 */
std::optional<RequestRejection> validate_llama_request(const ChorusRequest& request);

/*
 * Validates model-independent generation configuration.
 *
 * Returns:
 *  - `std::nullopt`: `Chorus::GenerationConfig` can be resolved.
 *  - `Chorus::RequestRejection`: A common or Llama-specific option is invalid.
 *
 * Errors:
 *  - `Chorus::ChorusError::InvalidRequest`: A supplied constraint is invalid.
 *  - `Chorus::ChorusError::UnsupportedFeature`: The constraint format is unsupported.
 *  - `Chorus::ChorusError::UnsupportedOption`: A generation option is invalid.
 */
std::optional<RequestRejection> validate_llama_generation(const GenerationConfig& config);

/*
 * Resolves common and Llama-specific generation options into provider-ready state.
 *
 * Unset values retain llama.cpp defaults. Constraint conversion happens here, while
 * checks requiring a loaded model are deferred to `Chorus::make_llama_sampler`. Rejections
 * are returned directly.
 *
 * Returns:
 *  - `Chorus::ResolvedLlamaGeneration`: The resolved generation state.
 *  - `Chorus::RequestRejection`: The configuration cannot be resolved.
 *
 * Errors:
 *  - `Chorus::ChorusError::InvalidRequest`: A supplied constraint is invalid.
 *  - `Chorus::ChorusError::UnsupportedFeature`: The constraint format is unsupported.
 *  - `Chorus::ChorusError::UnsupportedOption`: An option is unknown, malformed, or incompatible.
 */
std::variant<ResolvedLlamaGeneration, RequestRejection> resolve_llama_generation(const GenerationConfig& config);

/*
 * Constructs a sampler and completes validation against the loaded model.
 *
 * This checks model-dependent state such as logit-bias token IDs against the
 * loaded vocabulary. Rejections are returned directly.
 *
 * Returns:
 *  - `::common_sampler_ptr`: The constructed llama.cpp sampler.
 *  - `Chorus::RequestRejection`: The sampler cannot be constructed.
 *
 * Errors:
 *  - `Chorus::ChorusError::InvalidRequest`: The model is absent or sampler configuration is invalid.
 *  - `Chorus::ChorusError::UnsupportedOption`: A token ID is outside the model vocabulary.
 */
std::variant<common_sampler_ptr, RequestRejection>
make_llama_sampler(const llama_model* model, ResolvedLlamaGeneration resolved);

} // namespace Chorus

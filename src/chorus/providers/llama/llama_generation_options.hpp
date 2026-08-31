#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/generation_config.hpp"
#include "sampling.h"

#include <optional>
#include <string>
#include <vector>

namespace Chorus {

/*
 * Applies the `llama` provider-option namespace to provider-ready sampling state.
 *
 * `::common_params_sampling` must already contain llama.cpp defaults and resolved
 * common options. Unknown namespaces, unknown keys, invalid values, and incompatible
 * Llama options are returned as rejections.
 *
 * Returns:
 *  - `std::nullopt`: Every provider option was applied.
 *  - `Chorus::RequestRejection`: A provider option could not be applied.
 *
 * Errors:
 *  - `Chorus::ChorusError::UnsupportedOption`: A provider option is unknown, malformed, or incompatible.
 */
std::optional<RequestRejection>
apply_llama_generation_options(common_params_sampling& sampling, const ProviderOptionMap& provider_options);

/*
 * Returns the common generation-option names accepted by the Llama resolver.
 *
 * The returned reference has static lifetime.
 */
const std::vector<std::string>& llama_common_generation_option_names();

/*
 * Returns the names accepted within the `llama` provider-option namespace.
 *
 * The returned reference has static lifetime and matches the resolver vocabulary.
 */
const std::vector<std::string>& llama_provider_generation_option_names();

} // namespace Chorus

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "chorus/core/provider_option_value.hpp"

namespace Chorus {

enum class ConstraintFormat {
    Gbnf,
    JsonSchema,
    Regex,
    Lark,
};

/*
 * Describes a constraint on generated output.
 *
 * `Chorus::OutputConstraint::source` is interpreted according to
 * `Chorus::OutputConstraint::format`; the provider samples only tokens
 * accepted by that constraint.
 */
struct OutputConstraint {
    ConstraintFormat format = ConstraintFormat::Gbnf;
    std::string source;
};

struct UnconstrainedOutput {};
using ConstraintChoice = std::variant<UnconstrainedOutput, OutputConstraint>;

/*
 * Generation options common to every provider, plus namespaced provider options such as
 * `Chorus::GenerationConfig::provider_options["llama"]["repeat_penalty"]`.
 *
 * An unset common option uses the provider's default for the loaded model. A set option is a
 * deliberate instruction that the provider must honor or reject, never silently ignore.
 */
struct GenerationConfig {
    // Cap on generated tokens, reasoning included. -1 removes the cap;
    // 0 completes immediately with empty output.
    std::optional<int32_t> max_tokens;
    std::optional<float> temperature;
    std::optional<int32_t> top_k;
    std::optional<float> top_p;
    // Providers reject seeds outside their supported range rather than truncating them.
    std::optional<uint64_t> seed;
    // Token-score adjustment proportional to prior occurrence count. Positive values discourage
    // repetition; negative values encourage it. 0 disables the adjustment.
    std::optional<float> frequency_penalty;
    // Token-score adjustment applied once to tokens that have already appeared. Positive values
    // discourage repetition; negative values encourage it. 0 disables the adjustment.
    std::optional<float> presence_penalty;
    // Sequences that cut generation short. When one appears in the output, generation stops there
    // and the marker is withheld from emitted text. Markers match only the content channel;
    // reasoning output never meets them, so a think block is bounded by
    // `Chorus::GenerationConfig::max_tokens` alone.
    std::optional<std::vector<std::string>> stop;
    std::optional<ConstraintChoice> constraint;
    std::optional<bool> show_thinking;
    // Provider-specific extensions, grouped under the provider's namespace.
    ProviderOptionMap provider_options;
};

} // namespace Chorus

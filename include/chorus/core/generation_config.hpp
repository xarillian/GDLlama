#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
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
    std::vector<std::string> stop;
    std::optional<OutputConstraint> constraint;
    std::optional<bool> show_thinking;
    // Provider-specific extensions, grouped under the provider's namespace.
    ProviderOptionMap provider_options;
};

enum class PatchAction { Inherit, Set, Clear };

/*
 * One layer's instruction for a single generation option.
 *
 * Generation config is layered: provider defaults beneath host defaults beneath per-request
 * overrides. `Chorus::PatchAction::Inherit` leaves the value below unchanged,
 * `Chorus::PatchAction::Set` replaces it, and `Chorus::PatchAction::Clear` resets it to the
 * bottom value. `std::optional` represents only two of those states, so
 * `Chorus::ConfigPatch<T>` carries the action separately. Clearing produces an unset optional
 * field or an empty stop list.
 */
template <typename T> struct ConfigPatch {
    PatchAction action = PatchAction::Inherit;
    T value{};

    static ConfigPatch set(T value) { return {PatchAction::Set, std::move(value)}; }
    static ConfigPatch clear() { return {PatchAction::Clear, {}}; }
};

/*
 * One layer of generation configuration overrides.
 *
 * Common options use `Chorus::ConfigPatch<T>`. Provider options merge recursively into inherited
 * options, then `Chorus::GenerationConfigPatch::provider_option_erasures` removes inherited or
 * newly set dotted paths.
 */
struct GenerationConfigPatch {
    ConfigPatch<int32_t> max_tokens;
    ConfigPatch<float> temperature;
    ConfigPatch<int32_t> top_k;
    ConfigPatch<float> top_p;
    ConfigPatch<uint64_t> seed;
    ConfigPatch<float> frequency_penalty;
    ConfigPatch<float> presence_penalty;
    ConfigPatch<std::vector<std::string>> stop;
    ConfigPatch<OutputConstraint> constraint;
    ConfigPatch<bool> show_thinking;

    ProviderOptionMap provider_options;

    std::vector<std::string> provider_option_erasures;
};

/*
 * Folds one config layer onto the config below it.
 *
 * Each option obeys its `Chorus::ConfigPatch<T>::action`.
 * `Chorus::GenerationConfigPatch::provider_options` merges with the base map, then
 * `Chorus::GenerationConfigPatch::provider_option_erasures` runs last so one patch can set some
 * options while removing inherited ones.
 */
GenerationConfig apply_generation_patch(const GenerationConfig& base, const GenerationConfigPatch& patch);

/*
 * Overlays one option map onto another.
 *
 * Keys absent from `overrides` survive. Map-on-map collisions merge recursively; any other
 * collision is replaced by `overrides`. Merging only adds or replaces values; removal belongs to
 * `Chorus::erase_option_path`.
 */
ProviderOptionMap merge_option_maps(const ProviderOptionMap& base, const ProviderOptionMap& overrides);

/*
 * Removes one dotted path such as `llama.repeat_penalty` from a namespaced option map.
 *
 * Nested maps left empty by removal remain because an empty namespace is the provider's
 * responsibility to accept or reject.
 *
 * A path is absent when a segment is missing or an intermediate segment holds a scalar where a
 * namespace was expected. Removing an absent path is a no-op.
 */
void erase_option_path(ProviderOptionMap& options, const std::string& path);

} // namespace Chorus

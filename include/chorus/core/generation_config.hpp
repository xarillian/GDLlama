#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "chorus/core/provider_option_value.hpp"

namespace Chorus {

/// Grammar schema an OutputConstraint must satisfy.
enum class ConstraintFormat {
    Gbnf,
    JsonSchema,
    Regex,
    Lark,
};

/*
 * Restricts output to a grammar.
 *
 * Generation is normally free text, and free text is sometimes useless. A caller that needs
 * parseable output supplies a grammar here, and the provider samples only tokens that obey it.
 */
struct OutputConstraint {
    ConstraintFormat format = ConstraintFormat::Gbnf;
    std::string source;
};

/*
 * Generation options every provider understands, plus options addressed to one
 * provider by namespace, e.g. provider_options["llama"]["repeat_penalty"].
 *
 * Unset fields use the provider's default for the loaded model; a set field is a
 * deliberate instruction the provider must honor or reject, never silently ignore.
 */
struct GenerationConfig {
    // Cap on generated tokens, reasoning included. -1 removes the cap;
    // 0 completes immediately with empty output.
    std::optional<int32_t> max_tokens;
    // Sampling randomness. 0 disables sampling, making the provider greedy.
    std::optional<float> temperature;
    // Sample only from the k most probable tokens. 0 disables the filter.
    std::optional<int32_t> top_k;
    // Nucleus sampling: sample from the smallest token set whose cumulative
    // probability reaches this value, in [0.0, 1.0]. 1.0 disables the filter.
    std::optional<float> top_p;
    // Sampler RNG seed.
    // A provider with a narrower seed range rejects values it cannot honor rather than truncating them.
    std::optional<uint64_t> seed;
    // Penalizes tokens by how often they have already appeared.
    std::optional<float> frequency_penalty;
    // Penalizes tokens that have appeared at all.
    std::optional<float> presence_penalty;
    // Sequences that cut generation short; "stop sequence". When one appears in the output,
    // generation stops there and the marker is withheld from the emitted
    // text. Markers match the content channel only; reasoning output never
    // meets them, so a think block is bounded by max_tokens alone.
    std::vector<std::string> stop;
    std::optional<OutputConstraint> constraint;
    // Reasoning-model thinking toggle.
    std::optional<bool> thinking;
    // Options only one provider understands, keyed by its namespace.
    ProviderOptionMap provider_options;
};

enum class PatchAction { Inherit, Set, Clear };

/*
 * One layer's instruction for a single generation option.
 *
 * Generation config is layered: provider defaults beneath host defaults beneath per-request
 * overrides. A layer says one of three things about an option: leave what is below alone (Inherit),
 * set it, or clear it back to the bottom. std::optional can say only two of those, so the third
 * state rides along explicitly. Clear resets to empty: unset for an optional field, the empty
 * list for stop.
 */
template <typename T> struct ConfigPatch {
    PatchAction action = PatchAction::Inherit;
    T value{};

    static ConfigPatch set(T value) { return {PatchAction::Set, std::move(value)}; }
    static ConfigPatch clear() { return {PatchAction::Clear, {}}; }
};

/// One layer's overlay on a full GenerationConfig.
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
    ConfigPatch<bool> thinking;

    ProviderOptionMap provider_options;

    std::vector<std::string> provider_option_erasures;
};

/*
 * Overlays one option map onto another.
 *
 * Keys absent from overrides survive; map-on-map collisions merge recursively;
 * any other collision the override replaces wholesale. A merge *only* adds or
 * replaces: removal is erase_option_path's job.
 */
ProviderOptionMap merge_option_maps(const ProviderOptionMap& base, const ProviderOptionMap& overrides);

/*
 * Removes one dotted path ("llama.repeat_penalty") from a namespaced option map.
 *
 * Nested maps left empty by the removal stay: an empty namespace is a
 * provider's own business to accept or reject.
 *
 * A path is absent when a segment is missing or when an intermediate segment
 * holds a scalar where a namespace was expected. Removing what is absent is a
 * no-op, so this raises nothing and reports nothing.
 */
void erase_option_path(ProviderOptionMap& options, const std::string& path);

/*
 * Folds one config layer onto the config below it.
 *
 * Each generation option obeys its ConfigPatch action; provider_options merge; the
 * erasures run last, so one patch can both set options and drop inherited
 * ones.
 */
GenerationConfig apply_generation_patch(const GenerationConfig& base, const GenerationConfigPatch& patch);

} // namespace Chorus

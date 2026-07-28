#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "chorus/core/options.hpp"

namespace Chorus {

enum class ConstraintFormat {
    Gbnf,
    JsonSchema,
    Regex,
    Lark,
};

struct OutputConstraint {
    ConstraintFormat format = ConstraintFormat::Gbnf;
    std::string source; // grammar text, schema JSON, pattern, ...
};

// Every field optional: unset means "backend default", set is a deliberate
// instruction the backend must honor or reject (spec 3c-D). No literal
// defaults here; resolution happens inside each backend.
struct PortableGenerationConfig {
    std::optional<int32_t> max_tokens;
    std::optional<float> temperature;
    std::optional<int32_t> top_k;
    std::optional<float> top_p;
    std::optional<uint64_t> seed;
    std::optional<float> frequency_penalty;
    std::optional<float> presence_penalty;
    // Empty = none requested. Stop sequences match the content channel only;
    // reasoning output never meets them, so max_tokens is what bounds a
    // think block.
    std::vector<std::string> stop;
    std::optional<OutputConstraint> constraint;
    // Reasoning-model thinking toggle; unset = template/backend default (on).
    std::optional<bool> thinking;
};

struct GenerationConfig {
    PortableGenerationConfig common;
    OptionMap backend_options; // e.g. backend_options["llama"]["repeat_penalty"]
};

enum class PatchAction { Inherit, Set, Clear };

template <typename T> struct OptionalPatch {
    PatchAction action = PatchAction::Inherit;
    T value{};
    static OptionalPatch set(T value) { return {PatchAction::Set, std::move(value)}; }
    static OptionalPatch clear() { return {PatchAction::Clear, {}}; }
};

template <typename T> using ValuePatch = OptionalPatch<T>;

struct GenerationConfigPatch {
    OptionalPatch<int32_t> max_tokens;
    OptionalPatch<float> temperature;
    OptionalPatch<int32_t> top_k;
    OptionalPatch<float> top_p;
    OptionalPatch<uint64_t> seed;
    OptionalPatch<float> frequency_penalty;
    OptionalPatch<float> presence_penalty;
    ValuePatch<std::vector<std::string>> stop;
    OptionalPatch<OutputConstraint> constraint;
    OptionalPatch<bool> thinking;
    OptionMap backend_options;
    // Dotted namespaced paths ("llama.repeat_penalty") to remove from the
    // inherited options, applied after the merge above. Merging alone can only
    // add or replace, so without this a patch cannot say what a host request
    // spells as an explicit null: drop what the layer below set. Erasing a path
    // that is not present is a no-op.
    std::vector<std::string> backend_option_erasures;
};

OptionMap merge_option_maps(const OptionMap& base, const OptionMap& overrides);

// Removes one dotted path from a namespaced option map. Intermediate segments
// must be maps; anything else means the path does not exist, which is not an
// error. Nested maps left empty by the removal stay -- an empty namespace is a
// backend's own business to accept or reject.
void erase_option_path(OptionMap& options, const std::string& path);

GenerationConfig apply_generation_patch(const GenerationConfig& base, const GenerationConfigPatch& patch);

} // namespace Chorus

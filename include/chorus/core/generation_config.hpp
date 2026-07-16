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
    std::vector<std::string> stop; // empty = none requested
    std::optional<OutputConstraint> constraint;
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
    OptionMap backend_options;
};

OptionMap merge_option_maps(const OptionMap& base, const OptionMap& overrides);
GenerationConfig apply_generation_patch(const GenerationConfig& base, const GenerationConfigPatch& patch);

} // namespace Chorus

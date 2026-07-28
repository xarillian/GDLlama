#include "chorus/core/generation_config.hpp"

namespace Chorus {
namespace {

template <typename T> void apply_optional_patch(std::optional<T>& target, const OptionalPatch<T>& patch) {
    switch (patch.action) {
    case PatchAction::Inherit:
        break;
    case PatchAction::Set:
        target = patch.value;
        break;
    case PatchAction::Clear:
        target.reset();
        break;
    }
}

template <typename T> void apply_value_patch(T& target, const ValuePatch<T>& patch) {
    switch (patch.action) {
    case PatchAction::Inherit:
        break;
    case PatchAction::Set:
        target = patch.value;
        break;
    case PatchAction::Clear:
        target = {};
        break;
    }
}

} // namespace

OptionMap merge_option_maps(const OptionMap& base, const OptionMap& overrides) {
    OptionMap merged = base;

    for (const auto& [key, override_value] : overrides) {
        const auto inherited = merged.find(key);
        const auto* override_map = std::get_if<OptionMap>(&override_value);
        if (inherited != merged.end() && override_map != nullptr) {
            if (const auto* inherited_map = std::get_if<OptionMap>(&inherited->second)) {
                inherited->second = merge_option_maps(*inherited_map, *override_map);
                continue;
            }
        }
        merged[key] = override_value;
    }

    return merged;
}

void erase_option_path(OptionMap& options, const std::string& path) {
    OptionMap* level = &options;
    size_t start = 0;
    while (true) {
        const size_t dot = path.find('.', start);
        if (dot == std::string::npos)
            break;
        const auto next = level->find(path.substr(start, dot - start));
        if (next == level->end())
            return;
        auto* nested = std::get_if<OptionMap>(&next->second);
        if (!nested)
            return; // a scalar where the path expects a namespace
        level = nested;
        start = dot + 1;
    }
    level->erase(path.substr(start));
}

GenerationConfig apply_generation_patch(const GenerationConfig& base, const GenerationConfigPatch& patch) {
    GenerationConfig merged = base;
    apply_optional_patch(merged.common.max_tokens, patch.max_tokens);
    apply_optional_patch(merged.common.temperature, patch.temperature);
    apply_optional_patch(merged.common.top_k, patch.top_k);
    apply_optional_patch(merged.common.top_p, patch.top_p);
    apply_optional_patch(merged.common.seed, patch.seed);
    apply_optional_patch(merged.common.frequency_penalty, patch.frequency_penalty);
    apply_optional_patch(merged.common.presence_penalty, patch.presence_penalty);
    apply_value_patch(merged.common.stop, patch.stop);
    apply_optional_patch(merged.common.constraint, patch.constraint);
    apply_optional_patch(merged.common.thinking, patch.thinking);
    merged.backend_options = merge_option_maps(base.backend_options, patch.backend_options);
    for (const auto& path : patch.backend_option_erasures)
        erase_option_path(merged.backend_options, path);
    return merged;
}

} // namespace Chorus

#include "chorus/core/generation_config.hpp"

namespace Chorus {
namespace {

// Clearing resets optional options to unset and the stop list to empty,
// allowing both field shapes to share the same patch logic.
template <typename Target, typename T> void apply_patch(Target& target, const ConfigPatch<T>& patch) {
    switch (patch.action) {
    case PatchAction::Inherit:
        break;
    case PatchAction::Set:
        target = patch.value;
        break;
    case PatchAction::Clear:
        target = Target{};
        break;
    }
}

} // namespace

ProviderOptionMap merge_option_maps(const ProviderOptionMap& base, const ProviderOptionMap& overrides) {
    ProviderOptionMap merged = base;

    for (const auto& [key, override_value] : overrides) {
        const auto inherited = merged.find(key);
        const auto* override_map = std::get_if<ProviderOptionMap>(&override_value);
        if (inherited != merged.end() && override_map != nullptr) {
            if (const auto* inherited_map = std::get_if<ProviderOptionMap>(&inherited->second)) {
                inherited->second = merge_option_maps(*inherited_map, *override_map);
                continue;
            }
        }
        merged[key] = override_value;
    }

    return merged;
}

void erase_option_path(ProviderOptionMap& options, const std::string& path) {
    ProviderOptionMap* level = &options;
    size_t start = 0;
    while (true) {
        const size_t dot = path.find('.', start);
        if (dot == std::string::npos)
            break;
        const auto next = level->find(path.substr(start, dot - start));
        if (next == level->end())
            return;
        auto* nested = std::get_if<ProviderOptionMap>(&next->second);
        if (!nested)
            return;
        level = nested;
        start = dot + 1;
    }
    level->erase(path.substr(start));
}

GenerationConfig apply_generation_patch(const GenerationConfig& base, const GenerationConfigPatch& patch) {
    GenerationConfig merged = base;
    // Both structs are destructured in full so this function refuses to
    // compile when a generation option is added without its apply line.
    auto& [max_tokens, temperature, top_k, top_p, seed, frequency_penalty, presence_penalty, stop, constraint, show_thinking, provider_options] =
        merged;
    const auto& [p_max_tokens, p_temperature, p_top_k, p_top_p, p_seed, p_frequency_penalty, p_presence_penalty, p_stop, p_constraint, p_show_thinking, p_provider_options, p_provider_option_erasures] =
        patch;
    apply_patch(max_tokens, p_max_tokens);
    apply_patch(temperature, p_temperature);
    apply_patch(top_k, p_top_k);
    apply_patch(top_p, p_top_p);
    apply_patch(seed, p_seed);
    apply_patch(frequency_penalty, p_frequency_penalty);
    apply_patch(presence_penalty, p_presence_penalty);
    apply_patch(stop, p_stop);
    apply_patch(constraint, p_constraint);
    apply_patch(show_thinking, p_show_thinking);
    provider_options = merge_option_maps(base.provider_options, p_provider_options);
    for (const auto& path : p_provider_option_erasures)
        erase_option_path(provider_options, path);
    return merged;
}

} // namespace Chorus

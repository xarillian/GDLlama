#pragma once

#include "chorus/core/generation_config.hpp"
#include "chorus/core/model_spec.hpp"
#include "chorus/core/provider_option_value.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Chorus {

enum class Modality {
    Text,
    Image,
    Audio,
};

enum class SchedulingAuthority {
    ChorusManaged,
    ProviderManaged,
    Hybrid,
};

struct ProviderOptionDescriptor {
    std::string key;
    std::string display_name;
    std::string description;

    ProviderOptionValue default_value;

    std::optional<double> minimum;
    std::optional<double> maximum;
    std::optional<double> step;
    std::optional<std::string> enabled_by;
};

using ProviderOptionDescriptors = std::vector<ProviderOptionDescriptor>;

struct EngineCapabilities {
    std::string provider_id;
    std::vector<ModelFormat> model_formats;
    std::vector<Modality> input_modalities;
    std::vector<Modality> output_modalities;
    std::vector<ConstraintFormat> constraint_formats;

    SchedulingAuthority scheduling = SchedulingAuthority::ProviderManaged;

    bool streaming = false;
    bool cancellation = false;
    bool native_sessions = false;
    bool embeddings = false;
    bool speculative_decoding = false;
    bool dynamic_adapters = false;
    bool prompt_rendering = false;

    std::vector<std::string> common_generation_options;
    std::vector<std::string> provider_generation_options;

    ProviderOptionDescriptors load_options;
};

const ProviderOptionDescriptor* find_option_descriptor(const ProviderOptionDescriptors& schema, const std::string& key);

bool option_is_enabled(
    const ProviderOptionDescriptors& schema, const ProviderOptionDescriptor& descriptor, const ProviderOptionMap& stored
);

ProviderOptionMap resolve_option_defaults(const ProviderOptionDescriptors& schema, const ProviderOptionMap& stored);

struct LoadedModelInfo {
    std::string model_id;
    std::string family;
    ModelFormat format = ModelFormat::Auto;
    std::string quantization;
    std::vector<Modality> input_modalities;
    std::vector<Modality> output_modalities;
    std::optional<uint32_t> maximum_context;
    std::optional<uint32_t> per_request_context;
    std::optional<uint64_t> model_bytes;
    bool has_speculative_assets = false;
};

} // namespace Chorus

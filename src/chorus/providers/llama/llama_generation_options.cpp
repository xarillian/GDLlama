#include "chorus/providers/llama/llama_generation_options.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

namespace Chorus {
namespace {

enum class ProviderOptionValueKind { Int64, Double, Bool, StringList, Map };

enum class RangePolicy {
    Boolean,
    StringList,
    NonnegativeInt32,
    SentinelInt32,
    Mirostat,
    Probability,
    FiniteFloat,
    NonnegativeFloat,
    DryBase,
    AdaptiveTarget,
    AdaptiveDecay,
};

enum class TargetMember {
    MinKeep,
    MinP,
    TypicalP,
    DynamicTemperatureRange,
    DynamicTemperatureExponent,
    PenaltyLastN,
    RepeatPenalty,
    IgnoreEos,
    Mirostat,
    MirostatTau,
    MirostatEta,
    XtcProbability,
    XtcThreshold,
    DryMultiplier,
    DryBase,
    DryAllowedLength,
    DryPenaltyLastN,
    DrySequenceBreakers,
    SamplerOrder,
    LogitBias,
    TopNSigma,
    AdaptiveTarget,
    AdaptiveDecay,
};

struct OptionDescriptor {
    const char* public_key;
    ProviderOptionValueKind value_kind;
    RangePolicy range_policy;
    TargetMember target_member;
};

constexpr std::array<const char*, 10> kCommonOptions{
    "max_tokens",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "constraint",
    "stop",
    "show_thinking", // honored at chat-render time, not in the sampler
};

constexpr std::array<OptionDescriptor, 23> kProviderOptions{{
    {"min_keep", ProviderOptionValueKind::Int64, RangePolicy::NonnegativeInt32, TargetMember::MinKeep},
    {"min_p", ProviderOptionValueKind::Double, RangePolicy::Probability, TargetMember::MinP},
    {"typical_p", ProviderOptionValueKind::Double, RangePolicy::Probability, TargetMember::TypicalP},
    {"dynamic_temperature_range",
     ProviderOptionValueKind::Double,
     RangePolicy::NonnegativeFloat,
     TargetMember::DynamicTemperatureRange},
    {"dynamic_temperature_exponent",
     ProviderOptionValueKind::Double,
     RangePolicy::FiniteFloat,
     TargetMember::DynamicTemperatureExponent},
    {"penalty_last_n", ProviderOptionValueKind::Int64, RangePolicy::SentinelInt32, TargetMember::PenaltyLastN},
    {"repeat_penalty", ProviderOptionValueKind::Double, RangePolicy::FiniteFloat, TargetMember::RepeatPenalty},
    {"ignore_eos", ProviderOptionValueKind::Bool, RangePolicy::Boolean, TargetMember::IgnoreEos},
    {"mirostat", ProviderOptionValueKind::Int64, RangePolicy::Mirostat, TargetMember::Mirostat},
    {"mirostat_tau", ProviderOptionValueKind::Double, RangePolicy::FiniteFloat, TargetMember::MirostatTau},
    {"mirostat_eta", ProviderOptionValueKind::Double, RangePolicy::FiniteFloat, TargetMember::MirostatEta},
    {"xtc_probability", ProviderOptionValueKind::Double, RangePolicy::Probability, TargetMember::XtcProbability},
    {"xtc_threshold", ProviderOptionValueKind::Double, RangePolicy::Probability, TargetMember::XtcThreshold},
    {"dry_multiplier", ProviderOptionValueKind::Double, RangePolicy::NonnegativeFloat, TargetMember::DryMultiplier},
    {"dry_base", ProviderOptionValueKind::Double, RangePolicy::DryBase, TargetMember::DryBase},
    {"dry_allowed_length",
     ProviderOptionValueKind::Int64,
     RangePolicy::NonnegativeInt32,
     TargetMember::DryAllowedLength},
    {"dry_penalty_last_n", ProviderOptionValueKind::Int64, RangePolicy::SentinelInt32, TargetMember::DryPenaltyLastN},
    {"dry_sequence_breakers",
     ProviderOptionValueKind::StringList,
     RangePolicy::StringList,
     TargetMember::DrySequenceBreakers},
    {"sampler_order", ProviderOptionValueKind::StringList, RangePolicy::StringList, TargetMember::SamplerOrder},
    {"logit_bias", ProviderOptionValueKind::Map, RangePolicy::FiniteFloat, TargetMember::LogitBias},
    {"top_n_sigma", ProviderOptionValueKind::Double, RangePolicy::FiniteFloat, TargetMember::TopNSigma},
    {"adaptive_target", ProviderOptionValueKind::Double, RangePolicy::AdaptiveTarget, TargetMember::AdaptiveTarget},
    {"adaptive_decay", ProviderOptionValueKind::Double, RangePolicy::AdaptiveDecay, TargetMember::AdaptiveDecay},
}};

const char* expected_type(ProviderOptionValueKind kind) {
    switch (kind) {
    case ProviderOptionValueKind::Int64:
        return "int64";
    case ProviderOptionValueKind::Double:
        return "double";
    case ProviderOptionValueKind::Bool:
        return "bool";
    case ProviderOptionValueKind::StringList:
        return "string list";
    case ProviderOptionValueKind::Map:
        return "map";
    }
    return "unknown";
}

std::string received_type(const ProviderOptionValue& value) {
    if (std::holds_alternative<bool>(value))
        return "bool";
    if (std::holds_alternative<int64_t>(value))
        return "int64";
    if (std::holds_alternative<double>(value))
        return "double";
    if (std::holds_alternative<std::string>(value))
        return "string";
    if (std::holds_alternative<ProviderOptionList>(value))
        return "list";
    return "map";
}

const char* allowed_range(RangePolicy policy) {
    switch (policy) {
    case RangePolicy::Boolean:
        return "{false, true}";
    case RangePolicy::StringList:
        return "list containing only strings";
    case RangePolicy::NonnegativeInt32:
        return "[0, 2147483647]";
    case RangePolicy::SentinelInt32:
        return "[-1, 2147483647]";
    case RangePolicy::Mirostat:
        return "{0, 1, 2}";
    case RangePolicy::Probability:
        return "[0.0, 1.0]";
    case RangePolicy::FiniteFloat:
        return "finite float range";
    case RangePolicy::NonnegativeFloat:
        return "[0.0, finite float maximum]";
    case RangePolicy::DryBase:
        return "[1.0, finite float maximum]";
    case RangePolicy::AdaptiveTarget:
        return "[finite float minimum, 1.0]";
    case RangePolicy::AdaptiveDecay:
        return "[0.0, 0.99]";
    }
    return "valid range";
}

RequestRejection option_rejection(
    const std::string& option_namespace,
    const std::string& key,
    const std::string& expected,
    const std::string& received,
    const std::string& range
) {
    return {
        ChorusError::UnsupportedOption,
        "Option namespace '" + option_namespace + "', key '" + key + "' expected " + expected + ", received " +
            received + ", allowed range " + range + ".",
    };
}

RequestRejection option_rejection(const OptionDescriptor& descriptor, const ProviderOptionValue& value) {
    return option_rejection(
        "llama",
        descriptor.public_key,
        expected_type(descriptor.value_kind),
        received_type(value),
        allowed_range(descriptor.range_policy)
    );
}

const OptionDescriptor* find_descriptor(const std::string& key) {
    for (const auto& descriptor : kProviderOptions) {
        if (key == descriptor.public_key)
            return &descriptor;
    }
    return nullptr;
}

bool matches_kind(const ProviderOptionValue& value, ProviderOptionValueKind kind) {
    switch (kind) {
    case ProviderOptionValueKind::Int64:
        return std::holds_alternative<int64_t>(value);
    case ProviderOptionValueKind::Double:
        return std::holds_alternative<double>(value);
    case ProviderOptionValueKind::Bool:
        return std::holds_alternative<bool>(value);
    case ProviderOptionValueKind::StringList: {
        const auto* list = std::get_if<ProviderOptionList>(&value);
        if (!list)
            return false;
        for (const auto& item : *list) {
            if (!std::holds_alternative<std::string>(item))
                return false;
        }
        return true;
    }
    case ProviderOptionValueKind::Map:
        return std::holds_alternative<ProviderOptionMap>(value);
    }
    return false;
}

std::optional<common_sampler_type> sampler_type_for_name(const std::string& name) {
    constexpr std::array<std::pair<const char*, common_sampler_type>, 10> sampler_types{{
        {"penalties", COMMON_SAMPLER_TYPE_PENALTIES},
        {"dry", COMMON_SAMPLER_TYPE_DRY},
        {"top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA},
        {"top_k", COMMON_SAMPLER_TYPE_TOP_K},
        {"typ_p", COMMON_SAMPLER_TYPE_TYPICAL_P},
        {"top_p", COMMON_SAMPLER_TYPE_TOP_P},
        {"min_p", COMMON_SAMPLER_TYPE_MIN_P},
        {"xtc", COMMON_SAMPLER_TYPE_XTC},
        {"temperature", COMMON_SAMPLER_TYPE_TEMPERATURE},
        {"adaptive_p", COMMON_SAMPLER_TYPE_ADAPTIVE_P},
    }};
    for (const auto& [canonical_name, sampler_type] : sampler_types) {
        if (name == canonical_name)
            return sampler_type;
    }
    return std::nullopt;
}

std::optional<RequestRejection> apply_sampler_order(common_params_sampling& sampling, const ProviderOptionList& order) {
    if (order.empty())
        return std::nullopt;

    std::vector<common_sampler_type> samplers;
    samplers.reserve(order.size());
    for (size_t index = 0; index < order.size(); ++index) {
        const auto& name = std::get<std::string>(order[index]);
        const auto sampler_type = sampler_type_for_name(name);
        if (!sampler_type) {
            return RequestRejection{
                ChorusError::UnsupportedOption,
                "Option namespace 'llama', key 'sampler_order' contains unknown canonical sampler '" + name + "'.",
            };
        }
        for (const auto existing : samplers) {
            if (existing == *sampler_type) {
                return RequestRejection{
                    ChorusError::UnsupportedOption,
                    "Option namespace 'llama', key 'sampler_order' contains duplicate sampler '" + name + "'.",
                };
            }
        }
        if (*sampler_type == COMMON_SAMPLER_TYPE_ADAPTIVE_P && index + 1 != order.size()) {
            return RequestRejection{
                ChorusError::UnsupportedOption,
                "Option namespace 'llama', key 'sampler_order' requires 'adaptive_p' to be last.",
            };
        }
        samplers.push_back(*sampler_type);
    }
    sampling.samplers = std::move(samplers);
    return std::nullopt;
}

std::optional<RequestRejection> apply_logit_bias(common_params_sampling& sampling, const ProviderOptionMap& values) {
    std::vector<llama_logit_bias> biases;
    biases.reserve(values.size());
    for (const auto& [token_text, value] : values) {
        const auto* number = std::get_if<double>(&value);
        if (!number) {
            return RequestRejection{
                ChorusError::UnsupportedOption,
                "Option namespace 'llama', key 'logit_bias', token '" + token_text + "' expected double, received " +
                    received_type(value) + ".",
            };
        }
        const double float_max = std::numeric_limits<float>::max();
        if (!std::isfinite(*number) || *number < -float_max || *number > float_max) {
            return RequestRejection{
                ChorusError::UnsupportedOption,
                "Option namespace 'llama', key 'logit_bias', token '" + token_text + "' expected finite float range.",
            };
        }

        llama_token token = 0;
        const char* begin = token_text.data();
        const char* end = begin + token_text.size();
        const auto parsed = std::from_chars(begin, end, token, 10);
        if (token_text.empty() || parsed.ec != std::errc{} || parsed.ptr != end || token < 0) {
            return RequestRejection{
                ChorusError::UnsupportedOption,
                "Option namespace 'llama', key 'logit_bias' has invalid decimal token id '" + token_text + "'.",
            };
        }
        for (const auto& bias : biases) {
            if (bias.token == token) {
                return RequestRejection{
                    ChorusError::UnsupportedOption,
                    "Option namespace 'llama', key 'logit_bias' has duplicate token id '" + token_text + "'.",
                };
            }
        }
        biases.push_back({token, static_cast<float>(*number)});
    }
    sampling.logit_bias = std::move(biases);
    return std::nullopt;
}

bool integer_in_range(int64_t value, RangePolicy policy) {
    const auto max = int64_t{std::numeric_limits<int32_t>::max()};
    switch (policy) {
    case RangePolicy::NonnegativeInt32:
        return value >= 0 && value <= max;
    case RangePolicy::SentinelInt32:
        return value >= -1 && value <= max;
    case RangePolicy::Mirostat:
        return value >= 0 && value <= 2;
    default:
        return false;
    }
}

bool double_in_range(double value, RangePolicy policy) {
    if (!std::isfinite(value))
        return false;
    const double float_max = std::numeric_limits<float>::max();
    if (value < -float_max || value > float_max)
        return false;
    switch (policy) {
    case RangePolicy::Probability:
        return value >= 0.0 && value <= 1.0;
    case RangePolicy::FiniteFloat:
        return true;
    case RangePolicy::NonnegativeFloat:
        return value >= 0.0;
    case RangePolicy::DryBase:
        return value >= 1.0;
    case RangePolicy::AdaptiveTarget:
        return value <= 1.0;
    case RangePolicy::AdaptiveDecay:
        return value >= 0.0 && value <= 0.99;
    default:
        return false;
    }
}

void assign_integer(common_params_sampling& sampling, TargetMember target, int32_t value) {
    switch (target) {
    case TargetMember::MinKeep:
        sampling.min_keep = value;
        return;
    case TargetMember::PenaltyLastN:
        sampling.penalty_last_n = value;
        return;
    case TargetMember::Mirostat:
        sampling.mirostat = value;
        return;
    case TargetMember::DryAllowedLength:
        sampling.dry_allowed_length = value;
        return;
    case TargetMember::DryPenaltyLastN:
        sampling.dry_penalty_last_n = value;
        return;
    default:
        return;
    }
}

void assign_double(common_params_sampling& sampling, TargetMember target, float value) {
    switch (target) {
    case TargetMember::MinP:
        sampling.min_p = value;
        return;
    case TargetMember::TypicalP:
        sampling.typ_p = value;
        return;
    case TargetMember::DynamicTemperatureRange:
        sampling.dynatemp_range = value;
        return;
    case TargetMember::DynamicTemperatureExponent:
        sampling.dynatemp_exponent = value;
        return;
    case TargetMember::RepeatPenalty:
        sampling.penalty_repeat = value;
        return;
    case TargetMember::MirostatTau:
        sampling.mirostat_tau = value;
        return;
    case TargetMember::MirostatEta:
        sampling.mirostat_eta = value;
        return;
    case TargetMember::XtcProbability:
        sampling.xtc_probability = value;
        return;
    case TargetMember::XtcThreshold:
        sampling.xtc_threshold = value;
        return;
    case TargetMember::DryMultiplier:
        sampling.dry_multiplier = value;
        return;
    case TargetMember::DryBase:
        sampling.dry_base = value;
        return;
    case TargetMember::TopNSigma:
        sampling.top_n_sigma = value;
        return;
    case TargetMember::AdaptiveTarget:
        sampling.adaptive_target = value;
        return;
    case TargetMember::AdaptiveDecay:
        sampling.adaptive_decay = value;
        return;
    default:
        return;
    }
}

void assign_bool(common_params_sampling& sampling, TargetMember target, bool value) {
    switch (target) {
    case TargetMember::IgnoreEos:
        sampling.ignore_eos = value;
        return;
    default:
        return;
    }
}

std::optional<RequestRejection> apply_provider_option(
    common_params_sampling& sampling, const OptionDescriptor& descriptor, const ProviderOptionValue& value
) {
    if (!matches_kind(value, descriptor.value_kind))
        return option_rejection(descriptor, value);

    if (descriptor.target_member == TargetMember::SamplerOrder)
        return apply_sampler_order(sampling, std::get<ProviderOptionList>(value));
    if (descriptor.target_member == TargetMember::LogitBias)
        return apply_logit_bias(sampling, std::get<ProviderOptionMap>(value));

    switch (descriptor.value_kind) {
    case ProviderOptionValueKind::Int64: {
        const auto integer = std::get<int64_t>(value);
        if (!integer_in_range(integer, descriptor.range_policy))
            return option_rejection(descriptor, value);
        assign_integer(sampling, descriptor.target_member, static_cast<int32_t>(integer));
        return std::nullopt;
    }
    case ProviderOptionValueKind::Double: {
        const auto number = std::get<double>(value);
        if (!double_in_range(number, descriptor.range_policy))
            return option_rejection(descriptor, value);
        assign_double(sampling, descriptor.target_member, static_cast<float>(number));
        return std::nullopt;
    }
    case ProviderOptionValueKind::Bool:
        assign_bool(sampling, descriptor.target_member, std::get<bool>(value));
        return std::nullopt;
    case ProviderOptionValueKind::StringList: {
        std::vector<std::string> breakers;
        for (const auto& item : std::get<ProviderOptionList>(value))
            breakers.push_back(std::get<std::string>(item));
        sampling.dry_sequence_breakers = std::move(breakers);
        return std::nullopt;
    }
    case ProviderOptionValueKind::Map:
        return std::nullopt;
    }
    return std::nullopt;
}

} // namespace

std::optional<RequestRejection>
apply_llama_generation_options(common_params_sampling& sampling, const ProviderOptionMap& provider_options) {
    bool has_custom_sampler_order = false;
    for (const auto& [option_namespace, namespace_value] : provider_options) {
        if (option_namespace != "llama") {
            return option_rejection(
                option_namespace,
                "<namespace>",
                "namespace 'llama'",
                received_type(namespace_value),
                "catalogued namespace"
            );
        }
        const auto* options = std::get_if<ProviderOptionMap>(&namespace_value);
        if (!options) {
            return option_rejection(
                "llama", "<namespace>", "map", received_type(namespace_value), "map of catalogued options"
            );
        }
        for (const auto& [key, value] : *options) {
            const auto* descriptor = find_descriptor(key);
            if (!descriptor) {
                return option_rejection(
                    "llama", key, "catalogued option key", received_type(value), "catalogued llama generation option"
                );
            }
            if (auto rejection = apply_provider_option(sampling, *descriptor, value))
                return *rejection;
            if (key == "sampler_order" && !std::get<ProviderOptionList>(value).empty())
                has_custom_sampler_order = true;
        }
    }
    if (has_custom_sampler_order && sampling.mirostat != 0) {
        return RequestRejection{
            ChorusError::UnsupportedOption,
            "Option namespace 'llama', key 'sampler_order' cannot be combined with nonzero Mirostat.",
        };
    }
    return std::nullopt;
}

const std::vector<std::string>& llama_common_generation_option_names() {
    static const std::vector<std::string> names(kCommonOptions.begin(), kCommonOptions.end());
    return names;
}

const std::vector<std::string>& llama_provider_generation_option_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> result;
        result.reserve(kProviderOptions.size());
        for (const auto& descriptor : kProviderOptions)
            result.emplace_back(descriptor.public_key);
        return result;
    }();
    return names;
}

} // namespace Chorus

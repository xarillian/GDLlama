#include "chorus/providers/llama/llama_generation_options.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <variant>

namespace Chorus {
namespace {

enum class IntegerRange {
    NonnegativeInt32,
    SentinelInt32,
    Mirostat,
};

enum class FloatRange {
    Probability,
    Finite,
    RepeatPenalty,
    Nonnegative,
    DryBase,
    AdaptiveTarget,
    AdaptiveDecay,
};

struct IntegerOption {
    IntegerRange range;
    int32_t common_params_sampling::*member;
};

struct FloatOption {
    FloatRange range;
    float common_params_sampling::*member;
};

struct BoolOption {
    bool common_params_sampling::*member;
};

struct DrySequenceBreakersOption {};
struct SamplerOrderOption {};
struct LogitBiasOption {};

using OptionRule = std::variant<
    IntegerOption,
    FloatOption,
    BoolOption,
    DrySequenceBreakersOption,
    SamplerOrderOption,
    LogitBiasOption>;

struct OptionDescriptor {
    const char* public_key;
    OptionRule rule;
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

const std::array<OptionDescriptor, 23> kProviderOptions{{
    {"min_keep", IntegerOption{IntegerRange::NonnegativeInt32, &common_params_sampling::min_keep}},
    {"min_p", FloatOption{FloatRange::Probability, &common_params_sampling::min_p}},
    {"typical_p", FloatOption{FloatRange::Probability, &common_params_sampling::typ_p}},
    {"dynamic_temperature_range",
     FloatOption{FloatRange::Nonnegative, &common_params_sampling::dynatemp_range}},
    {"dynamic_temperature_exponent",
     FloatOption{FloatRange::Finite, &common_params_sampling::dynatemp_exponent}},
    {"penalty_last_n", IntegerOption{IntegerRange::SentinelInt32, &common_params_sampling::penalty_last_n}},
    {"repeat_penalty", FloatOption{FloatRange::RepeatPenalty, &common_params_sampling::penalty_repeat}},
    {"ignore_eos", BoolOption{&common_params_sampling::ignore_eos}},
    {"mirostat", IntegerOption{IntegerRange::Mirostat, &common_params_sampling::mirostat}},
    {"mirostat_tau", FloatOption{FloatRange::Finite, &common_params_sampling::mirostat_tau}},
    {"mirostat_eta", FloatOption{FloatRange::Finite, &common_params_sampling::mirostat_eta}},
    {"xtc_probability", FloatOption{FloatRange::Probability, &common_params_sampling::xtc_probability}},
    {"xtc_threshold", FloatOption{FloatRange::Probability, &common_params_sampling::xtc_threshold}},
    {"dry_multiplier", FloatOption{FloatRange::Nonnegative, &common_params_sampling::dry_multiplier}},
    {"dry_base", FloatOption{FloatRange::DryBase, &common_params_sampling::dry_base}},
    {"dry_allowed_length",
     IntegerOption{IntegerRange::NonnegativeInt32, &common_params_sampling::dry_allowed_length}},
    {"dry_penalty_last_n",
     IntegerOption{IntegerRange::SentinelInt32, &common_params_sampling::dry_penalty_last_n}},
    {"dry_sequence_breakers", DrySequenceBreakersOption{}},
    {"sampler_order", SamplerOrderOption{}},
    {"logit_bias", LogitBiasOption{}},
    {"top_n_sigma", FloatOption{FloatRange::Finite, &common_params_sampling::top_n_sigma}},
    {"adaptive_target", FloatOption{FloatRange::AdaptiveTarget, &common_params_sampling::adaptive_target}},
    {"adaptive_decay", FloatOption{FloatRange::AdaptiveDecay, &common_params_sampling::adaptive_decay}},
}};

const char* expected_type(const IntegerOption&) {
    return "int64";
}

const char* expected_type(const FloatOption&) {
    return "double";
}

const char* expected_type(const BoolOption&) {
    return "bool";
}

const char* expected_type(const DrySequenceBreakersOption&) {
    return "string list";
}

const char* expected_type(const SamplerOrderOption&) {
    return "string list";
}

const char* expected_type(const LogitBiasOption&) {
    return "map";
}

const char* expected_type(const OptionRule& rule) {
    return std::visit([](const auto& typed_rule) { return expected_type(typed_rule); }, rule);
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

const char* allowed_range(IntegerRange range) {
    switch (range) {
    case IntegerRange::NonnegativeInt32:
        return "[0, 2147483647]";
    case IntegerRange::SentinelInt32:
        return "[-1, 2147483647]";
    case IntegerRange::Mirostat:
        return "{0, 1, 2}";
    }
    return "valid integer range";
}

const char* allowed_range(FloatRange range) {
    switch (range) {
    case FloatRange::Probability:
        return "[0.0, 1.0]";
    case FloatRange::Finite:
        return "finite float range";
    case FloatRange::RepeatPenalty:
        return "positive finite float with finite float reciprocal";
    case FloatRange::Nonnegative:
        return "[0.0, finite float maximum]";
    case FloatRange::DryBase:
        return "[1.0, finite float maximum]";
    case FloatRange::AdaptiveTarget:
        return "[finite float minimum, 1.0]";
    case FloatRange::AdaptiveDecay:
        return "[0.0, 0.99]";
    }
    return "valid float range";
}

const char* allowed_range(const IntegerOption& rule) {
    return allowed_range(rule.range);
}

const char* allowed_range(const FloatOption& rule) {
    return allowed_range(rule.range);
}

const char* allowed_range(const BoolOption&) {
    return "{false, true}";
}

const char* allowed_range(const DrySequenceBreakersOption&) {
    return "list containing only strings";
}

const char* allowed_range(const SamplerOrderOption&) {
    return "list containing only strings";
}

const char* allowed_range(const LogitBiasOption&) {
    return "finite float range";
}

const char* allowed_range(const OptionRule& rule) {
    return std::visit([](const auto& typed_rule) { return allowed_range(typed_rule); }, rule);
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
        expected_type(descriptor.rule),
        received_type(value),
        allowed_range(descriptor.rule)
    );
}

const OptionDescriptor* find_descriptor(const std::string& key) {
    for (const auto& descriptor : kProviderOptions) {
        if (key == descriptor.public_key)
            return &descriptor;
    }
    return nullptr;
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

bool integer_in_range(int64_t value, IntegerRange range) {
    const auto max = int64_t{std::numeric_limits<int32_t>::max()};
    switch (range) {
    case IntegerRange::NonnegativeInt32:
        return value >= 0 && value <= max;
    case IntegerRange::SentinelInt32:
        return value >= -1 && value <= max;
    case IntegerRange::Mirostat:
        return value >= 0 && value <= 2;
    }
    return false;
}

bool double_in_range(double value, FloatRange range) {
    if (!std::isfinite(value))
        return false;
    const double float_max = std::numeric_limits<float>::max();
    if (value < -float_max || value > float_max)
        return false;
    switch (range) {
    case FloatRange::Probability:
        return value >= 0.0 && value <= 1.0;
    case FloatRange::Finite:
        return true;
    case FloatRange::RepeatPenalty:
        return static_cast<float>(value) > 0.0f && std::isfinite(1.0f / static_cast<float>(value));
    case FloatRange::Nonnegative:
        return value >= 0.0;
    case FloatRange::DryBase:
        return value >= 1.0;
    case FloatRange::AdaptiveTarget:
        return value <= 1.0;
    case FloatRange::AdaptiveDecay:
        return value >= 0.0 && value <= 0.99;
    }
    return false;
}

std::optional<RequestRejection> apply_option_rule(
    common_params_sampling& sampling,
    const OptionDescriptor& descriptor,
    const IntegerOption& rule,
    const ProviderOptionValue& value
) {
    const auto* integer = std::get_if<int64_t>(&value);
    if (!integer || !integer_in_range(*integer, rule.range))
        return option_rejection(descriptor, value);
    sampling.*rule.member = static_cast<int32_t>(*integer);
    return std::nullopt;
}

std::optional<RequestRejection> apply_option_rule(
    common_params_sampling& sampling,
    const OptionDescriptor& descriptor,
    const FloatOption& rule,
    const ProviderOptionValue& value
) {
    const auto* number = std::get_if<double>(&value);
    if (!number || !double_in_range(*number, rule.range))
        return option_rejection(descriptor, value);
    sampling.*rule.member = static_cast<float>(*number);
    return std::nullopt;
}

std::optional<RequestRejection> apply_option_rule(
    common_params_sampling& sampling,
    const OptionDescriptor& descriptor,
    const BoolOption& rule,
    const ProviderOptionValue& value
) {
    const auto* boolean = std::get_if<bool>(&value);
    if (!boolean)
        return option_rejection(descriptor, value);
    sampling.*rule.member = *boolean;
    return std::nullopt;
}

std::optional<RequestRejection> apply_option_rule(
    common_params_sampling& sampling,
    const OptionDescriptor& descriptor,
    const DrySequenceBreakersOption&,
    const ProviderOptionValue& value
) {
    const auto* items = std::get_if<ProviderOptionList>(&value);
    if (!items)
        return option_rejection(descriptor, value);

    std::vector<std::string> breakers;
    breakers.reserve(items->size());
    for (const auto& item : *items) {
        const auto* breaker = std::get_if<std::string>(&item);
        if (!breaker)
            return option_rejection(descriptor, value);
        breakers.push_back(*breaker);
    }
    sampling.dry_sequence_breakers = std::move(breakers);
    return std::nullopt;
}

std::optional<RequestRejection> apply_option_rule(
    common_params_sampling& sampling,
    const OptionDescriptor& descriptor,
    const SamplerOrderOption&,
    const ProviderOptionValue& value
) {
    const auto* order = std::get_if<ProviderOptionList>(&value);
    if (!order)
        return option_rejection(descriptor, value);
    for (const auto& item : *order) {
        if (!std::holds_alternative<std::string>(item))
            return option_rejection(descriptor, value);
    }
    return apply_sampler_order(sampling, *order);
}

std::optional<RequestRejection> apply_option_rule(
    common_params_sampling& sampling,
    const OptionDescriptor& descriptor,
    const LogitBiasOption&,
    const ProviderOptionValue& value
) {
    const auto* biases = std::get_if<ProviderOptionMap>(&value);
    if (!biases)
        return option_rejection(descriptor, value);
    return apply_logit_bias(sampling, *biases);
}

std::optional<RequestRejection> apply_provider_option(
    common_params_sampling& sampling, const OptionDescriptor& descriptor, const ProviderOptionValue& value
) {
    return std::visit(
        [&](const auto& rule) { return apply_option_rule(sampling, descriptor, rule, value); }, descriptor.rule
    );
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
            if (std::holds_alternative<SamplerOrderOption>(descriptor->rule) &&
                !std::get<ProviderOptionList>(value).empty())
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

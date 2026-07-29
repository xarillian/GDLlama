#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/stop_sequence_filter.hpp"
#include "json-schema-to-grammar.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace Chorus {
namespace {

enum class OptionValueKind { Int64, Double, Bool, StringList, Map };

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
    OptionValueKind value_kind;
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
    "thinking", // honored at chat-render time (#5), not in the sampler
};

constexpr std::array<OptionDescriptor, 23> kProviderOptions{{
    {"min_keep", OptionValueKind::Int64, RangePolicy::NonnegativeInt32, TargetMember::MinKeep},
    {"min_p", OptionValueKind::Double, RangePolicy::Probability, TargetMember::MinP},
    {"typical_p", OptionValueKind::Double, RangePolicy::Probability, TargetMember::TypicalP},
    {"dynamic_temperature_range",
     OptionValueKind::Double,
     RangePolicy::NonnegativeFloat,
     TargetMember::DynamicTemperatureRange},
    {"dynamic_temperature_exponent",
     OptionValueKind::Double,
     RangePolicy::FiniteFloat,
     TargetMember::DynamicTemperatureExponent},
    {"penalty_last_n", OptionValueKind::Int64, RangePolicy::SentinelInt32, TargetMember::PenaltyLastN},
    {"repeat_penalty", OptionValueKind::Double, RangePolicy::FiniteFloat, TargetMember::RepeatPenalty},
    {"ignore_eos", OptionValueKind::Bool, RangePolicy::Boolean, TargetMember::IgnoreEos},
    {"mirostat", OptionValueKind::Int64, RangePolicy::Mirostat, TargetMember::Mirostat},
    {"mirostat_tau", OptionValueKind::Double, RangePolicy::FiniteFloat, TargetMember::MirostatTau},
    {"mirostat_eta", OptionValueKind::Double, RangePolicy::FiniteFloat, TargetMember::MirostatEta},
    {"xtc_probability", OptionValueKind::Double, RangePolicy::Probability, TargetMember::XtcProbability},
    {"xtc_threshold", OptionValueKind::Double, RangePolicy::Probability, TargetMember::XtcThreshold},
    {"dry_multiplier", OptionValueKind::Double, RangePolicy::NonnegativeFloat, TargetMember::DryMultiplier},
    {"dry_base", OptionValueKind::Double, RangePolicy::DryBase, TargetMember::DryBase},
    {"dry_allowed_length", OptionValueKind::Int64, RangePolicy::NonnegativeInt32, TargetMember::DryAllowedLength},
    {"dry_penalty_last_n", OptionValueKind::Int64, RangePolicy::SentinelInt32, TargetMember::DryPenaltyLastN},
    {"dry_sequence_breakers", OptionValueKind::StringList, RangePolicy::StringList, TargetMember::DrySequenceBreakers},
    {"sampler_order", OptionValueKind::StringList, RangePolicy::StringList, TargetMember::SamplerOrder},
    {"logit_bias", OptionValueKind::Map, RangePolicy::FiniteFloat, TargetMember::LogitBias},
    {"top_n_sigma", OptionValueKind::Double, RangePolicy::FiniteFloat, TargetMember::TopNSigma},
    {"adaptive_target", OptionValueKind::Double, RangePolicy::AdaptiveTarget, TargetMember::AdaptiveTarget},
    {"adaptive_decay", OptionValueKind::Double, RangePolicy::AdaptiveDecay, TargetMember::AdaptiveDecay},
}};

const char* expected_type(OptionValueKind kind) {
    switch (kind) {
    case OptionValueKind::Int64:
        return "int64";
    case OptionValueKind::Double:
        return "double";
    case OptionValueKind::Bool:
        return "bool";
    case OptionValueKind::StringList:
        return "string list";
    case OptionValueKind::Map:
        return "map";
    }
    return "unknown";
}

std::string received_type(const OptionValue& value) {
    if (std::holds_alternative<bool>(value))
        return "bool";
    if (std::holds_alternative<int64_t>(value))
        return "int64";
    if (std::holds_alternative<double>(value))
        return "double";
    if (std::holds_alternative<std::string>(value))
        return "string";
    if (std::holds_alternative<OptionList>(value))
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

const OptionDescriptor* find_descriptor(const std::string& key) {
    for (const auto& descriptor : kProviderOptions) {
        if (key == descriptor.public_key)
            return &descriptor;
    }
    return nullptr;
}

bool matches_kind(const OptionValue& value, OptionValueKind kind) {
    switch (kind) {
    case OptionValueKind::Int64:
        return std::holds_alternative<int64_t>(value);
    case OptionValueKind::Double:
        return std::holds_alternative<double>(value);
    case OptionValueKind::Bool:
        return std::holds_alternative<bool>(value);
    case OptionValueKind::StringList: {
        const auto* list = std::get_if<OptionList>(&value);
        if (!list)
            return false;
        for (const auto& item : *list) {
            if (!std::holds_alternative<std::string>(item))
                return false;
        }
        return true;
    }
    case OptionValueKind::Map:
        return std::holds_alternative<OptionMap>(value);
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

std::optional<RequestRejection> apply_sampler_order(common_params_sampling& sampling, const OptionList& order) {
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

std::optional<RequestRejection> apply_logit_bias(common_params_sampling& sampling, const OptionMap& values) {
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

std::optional<RequestRejection>
apply_provider_option(ResolvedLlamaGeneration& resolved, const OptionDescriptor& descriptor, const OptionValue& value) {
    if (!matches_kind(value, descriptor.value_kind)) {
        return option_rejection(
            "llama",
            descriptor.public_key,
            expected_type(descriptor.value_kind),
            received_type(value),
            allowed_range(descriptor.range_policy)
        );
    }

    if (descriptor.target_member == TargetMember::SamplerOrder)
        return apply_sampler_order(resolved.sampling, std::get<OptionList>(value));
    if (descriptor.target_member == TargetMember::LogitBias)
        return apply_logit_bias(resolved.sampling, std::get<OptionMap>(value));

    switch (descriptor.value_kind) {
    case OptionValueKind::Int64: {
        const auto integer = std::get<int64_t>(value);
        if (!integer_in_range(integer, descriptor.range_policy)) {
            return option_rejection(
                "llama",
                descriptor.public_key,
                expected_type(descriptor.value_kind),
                received_type(value),
                allowed_range(descriptor.range_policy)
            );
        }
        assign_integer(resolved.sampling, descriptor.target_member, static_cast<int32_t>(integer));
        return std::nullopt;
    }
    case OptionValueKind::Double: {
        const auto number = std::get<double>(value);
        if (!double_in_range(number, descriptor.range_policy)) {
            return option_rejection(
                "llama",
                descriptor.public_key,
                expected_type(descriptor.value_kind),
                received_type(value),
                allowed_range(descriptor.range_policy)
            );
        }
        assign_double(resolved.sampling, descriptor.target_member, static_cast<float>(number));
        return std::nullopt;
    }
    case OptionValueKind::Bool:
        assign_bool(resolved.sampling, descriptor.target_member, std::get<bool>(value));
        return std::nullopt;
    case OptionValueKind::StringList: {
        std::vector<std::string> breakers;
        for (const auto& item : std::get<OptionList>(value))
            breakers.push_back(std::get<std::string>(item));
        resolved.sampling.dry_sequence_breakers = std::move(breakers);
        return std::nullopt;
    }
    case OptionValueKind::Map:
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<RequestRejection> validate_common(const GenerationConfig& config) {
    if (config.max_tokens && *config.max_tokens < -1)
        return option_rejection("common", "max_tokens", "int32", "int32", "[-1, 2147483647]");
    if (config.temperature && (!std::isfinite(*config.temperature) || *config.temperature < 0.0f))
        return option_rejection("common", "temperature", "float", "float", "[0.0, finite float maximum]");
    if (config.top_k && *config.top_k < 0)
        return option_rejection("common", "top_k", "int32", "int32", "[0, 2147483647]");
    if (config.top_p && (!std::isfinite(*config.top_p) || *config.top_p < 0.0f || *config.top_p > 1.0f))
        return option_rejection("common", "top_p", "float", "float", "[0.0, 1.0]");
    if (config.seed && *config.seed > std::numeric_limits<uint32_t>::max())
        return option_rejection("common", "seed", "uint64", "uint64", "[0, 4294967295]");
    if (config.frequency_penalty && !std::isfinite(*config.frequency_penalty))
        return option_rejection("common", "frequency_penalty", "float", "float", "finite float range");
    if (config.presence_penalty && !std::isfinite(*config.presence_penalty))
        return option_rejection("common", "presence_penalty", "float", "float", "finite float range");
    if (auto rejection = validate_stop_sequences(config.stop))
        return rejection;
    return std::nullopt;
}

std::optional<RequestRejection>
resolve_constraint(common_params_sampling& sampling, const std::optional<OutputConstraint>& constraint) {
    if (!constraint)
        return std::nullopt;

    switch (constraint->format) {
    case ConstraintFormat::Gbnf:
        if (constraint->source.empty()) {
            return RequestRejection{ChorusError::InvalidRequest, "GBNF constraint source must not be empty."};
        }
        sampling.grammar = common_grammar{COMMON_GRAMMAR_TYPE_USER, constraint->source};
        return std::nullopt;
    case ConstraintFormat::JsonSchema:
        if (constraint->source.empty()) {
            return RequestRejection{ChorusError::InvalidRequest, "JSON Schema constraint source must not be empty."};
        }
        try {
            const auto schema = nlohmann::ordered_json::parse(constraint->source);
            sampling.grammar = common_grammar{COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(schema, true)};
        } catch (const std::exception& error) {
            return RequestRejection{
                ChorusError::InvalidRequest, "Invalid JSON Schema constraint: " + std::string(error.what())
            };
        }
        return std::nullopt;
    case ConstraintFormat::Regex:
        return RequestRejection{
            ChorusError::UnsupportedFeature, "Regex output constraints are not supported by Llama."
        };
    case ConstraintFormat::Lark:
        return RequestRejection{ChorusError::UnsupportedFeature, "Lark output constraints are not supported by Llama."};
    }
    return RequestRejection{ChorusError::UnsupportedFeature, "Unknown output constraint format."};
}

} // namespace

std::variant<ResolvedLlamaGeneration, RequestRejection> resolve_llama_generation(const GenerationConfig& config) {
    if (auto rejection = validate_common(config))
        return *rejection;

    ResolvedLlamaGeneration resolved;
    if (config.max_tokens)
        resolved.max_tokens = *config.max_tokens;
    if (config.temperature)
        resolved.sampling.temp = *config.temperature;
    if (config.top_k)
        resolved.sampling.top_k = *config.top_k;
    if (config.top_p)
        resolved.sampling.top_p = *config.top_p;
    if (config.seed)
        resolved.sampling.seed = static_cast<uint32_t>(*config.seed);
    if (config.frequency_penalty)
        resolved.sampling.penalty_freq = *config.frequency_penalty;
    if (config.presence_penalty)
        resolved.sampling.penalty_present = *config.presence_penalty;
    resolved.stop = config.stop;
    if (auto rejection = resolve_constraint(resolved.sampling, config.constraint))
        return *rejection;
    bool has_custom_sampler_order = false;

    for (const auto& [option_namespace, namespace_value] : config.provider_options) {
        if (option_namespace != "llama") {
            return option_rejection(
                option_namespace,
                "<namespace>",
                "namespace 'llama'",
                received_type(namespace_value),
                "catalogued namespace"
            );
        }
        const auto* options = std::get_if<OptionMap>(&namespace_value);
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
            if (auto rejection = apply_provider_option(resolved, *descriptor, value))
                return *rejection;
            if (key == "sampler_order" && !std::get<OptionList>(value).empty())
                has_custom_sampler_order = true;
        }
    }
    if (has_custom_sampler_order && resolved.sampling.mirostat != 0) {
        return RequestRejection{
            ChorusError::UnsupportedOption,
            "Option namespace 'llama', key 'sampler_order' cannot be combined with nonzero Mirostat.",
        };
    }
    return resolved;
}

std::variant<common_sampler_ptr, RequestRejection>
make_llama_sampler(const llama_model* model, ResolvedLlamaGeneration resolved) {
    if (!model) {
        return RequestRejection{ChorusError::InvalidRequest, "Cannot construct a Llama sampler without a model."};
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const llama_token vocabulary_size = llama_vocab_n_tokens(vocab);
    for (const auto& bias : resolved.sampling.logit_bias) {
        if (bias.token < 0 || bias.token >= vocabulary_size) {
            return RequestRejection{
                ChorusError::UnsupportedOption,
                "Option namespace 'llama', key 'logit_bias' token id '" + std::to_string(bias.token) +
                    "' is outside the model vocabulary [0, " + std::to_string(vocabulary_size) + ").",
            };
        }
    }

    if (resolved.sampling.ignore_eos) {
        auto& biases = resolved.sampling.logit_bias;
        biases.erase(
            std::remove_if(
                biases.begin(),
                biases.end(),
                [vocab](const llama_logit_bias& bias) { return llama_vocab_is_eog(vocab, bias.token); }
            ),
            biases.end()
        );
        for (llama_token token = 0; token < vocabulary_size; ++token) {
            if (llama_vocab_is_eog(vocab, token))
                biases.push_back({token, -INFINITY});
        }
    }

    try {
        return common_sampler_ptr(common_sampler_init(model, resolved.sampling));
    } catch (const std::runtime_error& error) {
        std::string constraint_name = "sampler configuration";
        if (resolved.sampling.grammar.type == COMMON_GRAMMAR_TYPE_USER)
            constraint_name = "GBNF constraint";
        else if (resolved.sampling.grammar.type == COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT)
            constraint_name = "JSON Schema constraint";
        return RequestRejection{
            ChorusError::InvalidRequest, "Invalid " + constraint_name + ": " + std::string(error.what())
        };
    }
}

std::optional<RequestRejection> validate_llama_generation(const GenerationConfig& config) {
    auto resolved = resolve_llama_generation(config);
    if (const auto* rejection = std::get_if<RequestRejection>(&resolved))
        return *rejection;
    return std::nullopt;
}

std::optional<RequestRejection> validate_llama_request(const ChorusRequest& request) {
    if (request.messages.empty()) {
        if (!request.chat_template.empty()) {
            return RequestRejection{
                ChorusError::UnsupportedOption,
                "Llama chat_template requires non-empty messages; unset it for raw-prompt generation.",
            };
        }
        if (request.gen_config.thinking.has_value()) {
            return RequestRejection{
                ChorusError::UnsupportedOption,
                "Llama thinking requires non-empty messages; unset it for raw-prompt generation.",
            };
        }
    }
    return validate_llama_generation(request.gen_config);
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

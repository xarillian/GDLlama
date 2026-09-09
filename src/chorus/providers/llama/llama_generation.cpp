#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_generation_options.hpp"
#include "chorus/providers/llama/stop_sequence_filter.hpp"
#include "json-schema-to-grammar.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace Chorus {
namespace {

RequestRejection common_option_rejection(
    const std::string& key, const std::string& expected, const std::string& received, const std::string& range
);
std::optional<RequestRejection> validate_common(const GenerationConfig& config);
std::optional<RequestRejection>
resolve_constraint(common_params_sampling& sampling, const std::optional<OutputConstraint>& constraint);
} // namespace

std::optional<RequestRejection> validate_llama_request(const ChorusRequest& request) {
    for (const auto& message : request.messages) {
        if (!message_role_name(message.role))
            return RequestRejection{ChorusError::InvalidRequest, "Chat message role is invalid."};
        if (!joined_text(message.content))
            return RequestRejection{ChorusError::UnsupportedFeature, "Llama accepts text-only chat content."};
    }
    if (request.messages.empty()) {
        if (!request.chat_template.empty()) {
            return RequestRejection{
                ChorusError::UnsupportedOption,
                "Llama chat_template requires non-empty messages; unset it for raw-prompt generation.",
            };
        }
        if (request.gen_config.show_thinking.has_value()) {
            return RequestRejection{
                ChorusError::UnsupportedOption,
                "Llama show_thinking requires non-empty messages; unset it for raw-prompt generation.",
            };
        }
    }
    return validate_llama_generation(request.gen_config);
}

std::optional<RequestRejection> validate_llama_generation(const GenerationConfig& config) {
    auto resolved = resolve_llama_generation(config);
    if (const auto* rejection = std::get_if<RequestRejection>(&resolved))
        return *rejection;
    return std::nullopt;
}

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
    if (auto rejection = apply_llama_generation_options(resolved.sampling, config.provider_options))
        return *rejection;
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
        // The `ignore_eos` option overrides explicit EOG biases, so replace
        // every existing EOG entry before banning all model-defined EOG tokens.
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
    } catch (const std::invalid_argument& error) {
        return RequestRejection{
            ChorusError::InvalidRequest, "Invalid sampler configuration: " + std::string(error.what())
        };
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

namespace {

RequestRejection common_option_rejection(
        const std::string& key, const std::string& expected, const std::string& received, const std::string& range
    ) {
        return {
            ChorusError::UnsupportedOption,
            "Option namespace 'common', key '" + key + "' expected " + expected + ", received " + received +
                ", allowed range " + range + ".",
        };
    }

    std::optional<RequestRejection> validate_common(const GenerationConfig& config) {
        if (config.max_tokens && *config.max_tokens < -1)
            return common_option_rejection("max_tokens", "int32", "int32", "[-1, 2147483647]");
        if (config.temperature && (!std::isfinite(*config.temperature) || *config.temperature < 0.0f))
            return common_option_rejection("temperature", "float", "float", "[0.0, finite float maximum]");
        if (config.top_k && *config.top_k < 0)
            return common_option_rejection("top_k", "int32", "int32", "[0, 2147483647]");
        if (config.top_p && (!std::isfinite(*config.top_p) || *config.top_p < 0.0f || *config.top_p > 1.0f))
            return common_option_rejection("top_p", "float", "float", "[0.0, 1.0]");
        if (config.seed && *config.seed > std::numeric_limits<uint32_t>::max())
            return common_option_rejection("seed", "uint64", "uint64", "[0, 4294967295]");
        if (config.frequency_penalty && !std::isfinite(*config.frequency_penalty))
            return common_option_rejection("frequency_penalty", "float", "float", "finite float range");
        if (config.presence_penalty && !std::isfinite(*config.presence_penalty))
            return common_option_rejection("presence_penalty", "float", "float", "finite float range");
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
                return RequestRejection{
                    ChorusError::InvalidRequest, "JSON Schema constraint source must not be empty."
                };
            }
            try {
                const auto schema = common_json::parse(constraint->source);
                sampling.grammar =
                    common_grammar{COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(schema, true)};
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
            return RequestRejection{
                ChorusError::UnsupportedFeature, "Lark output constraints are not supported by Llama."
            };
        }
        return RequestRejection{ChorusError::UnsupportedFeature, "Unknown output constraint format."};
    }

    } // namespace

} // namespace Chorus

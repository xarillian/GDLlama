#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_generation_options.hpp"
#include "silent_llama_log.hpp"
#include "gtest_utils.hpp"

class LlamaGenerationModelTest : public ChorusModelTest {};

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace {

constexpr const char* kModelPath = "tests/models/gemma-3-270m-it-F16.gguf";

struct LlamaModelFixture {
    // First member, so it outlives the free below and llama stays quiet
    // through teardown as well as load.
    SilentLlamaLog quiet;
    llama_model* model = nullptr;

    bool load() {
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        model = llama_model_load_from_file(kModelPath, params);
        return model != nullptr;
    }

    ~LlamaModelFixture() {
        if (model)
            llama_model_free(model);
    }
};

template <typename Configure, typename Check> void check_resolution(Configure configure, Check check) {
    Chorus::GenerationConfig config;
    configure(config);
    auto result = Chorus::resolve_llama_generation(config);
    ASSERT_TRUE(std::holds_alternative<Chorus::ResolvedLlamaGeneration>(result));
    check(std::get<Chorus::ResolvedLlamaGeneration>(result));
}

template <typename Check>
void check_provider_resolution(const std::string& key, Chorus::ProviderOptionValue value, Check check) {
    check_resolution(
        [&](Chorus::GenerationConfig& config) {
            config.provider_options["llama"] = Chorus::ProviderOptionMap{{key, std::move(value)}};
        },
        check
    );
}

Chorus::RequestRejection rejection_for(const std::string& key, Chorus::ProviderOptionValue value) {
    Chorus::GenerationConfig config;
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{key, std::move(value)}};
    auto result = Chorus::resolve_llama_generation(config);
    if (!std::holds_alternative<Chorus::RequestRejection>(result)) {
        ADD_FAILURE() << "Expected rejection for " << key << '.';
        return {};
    }
    return std::get<Chorus::RequestRejection>(result);
}

Chorus::RequestRejection constraint_rejection_for(Chorus::ConstraintFormat format, std::string source) {
    Chorus::GenerationConfig config;
    config.constraint = Chorus::OutputConstraint{format, std::move(source)};
    auto result = Chorus::resolve_llama_generation(config);
    if (!std::holds_alternative<Chorus::RequestRejection>(result)) {
        ADD_FAILURE() << "Expected constraint rejection.";
        return {};
    }
    return std::get<Chorus::RequestRejection>(result);
}

const llama_logit_bias* find_logit_bias(const common_params_sampling& sampling, llama_token token) {
    for (const auto& bias : sampling.logit_bias) {
        if (bias.token == token)
            return &bias;
    }
    return nullptr;
}

// One row of the capability conformance matrix: an advertised option name, a
// configure step that sets a valid value for it, and a verify step that proves
// the value reached a distinct resolved upstream field.
struct ConformanceCase {
    std::string name;
    std::function<void(Chorus::GenerationConfig&)> configure;
    std::function<void(const Chorus::ResolvedLlamaGeneration&)> verify;
};

} // namespace

TEST(LlamaGeneration, Llama_generation_preserves_upstream_defaults) {
    const common_params_sampling upstream;
    check_resolution(
        [](auto&) {},
        [&](const auto& value) {
            ASSERT_EQ(value.max_tokens, -1);
            ASSERT_TRUE(value.stop.empty());
            ASSERT_EQ(value.sampling.seed, upstream.seed);
            ASSERT_EQ(value.sampling.min_keep, upstream.min_keep);
            ASSERT_EQ(value.sampling.top_k, upstream.top_k);
            ASSERT_TRUE(value.sampling.top_p == upstream.top_p);
            ASSERT_TRUE(value.sampling.min_p == upstream.min_p);
            ASSERT_TRUE(value.sampling.temp == upstream.temp);
            ASSERT_TRUE(value.sampling.penalty_repeat == upstream.penalty_repeat);
            ASSERT_TRUE(value.sampling.dry_sequence_breakers == upstream.dry_sequence_breakers);
        }
    );
}

TEST(LlamaGeneration, Llama_generation_rejects_wrong_type_and_unknown_key) {
    auto wrong = rejection_for("min_keep", 1.0);
    ASSERT_TRUE(wrong.error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(wrong.message.find("llama") != std::string::npos);
    ASSERT_TRUE(wrong.message.find("min_keep") != std::string::npos);
    ASSERT_TRUE(wrong.message.find("int64") != std::string::npos);
    ASSERT_TRUE(wrong.message.find("double") != std::string::npos);
    ASSERT_TRUE(wrong.message.find("[0, 2147483647]") != std::string::npos);

    auto unknown = rejection_for("warp_factor", int64_t{3});
    ASSERT_TRUE(unknown.error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(unknown.message.find("llama") != std::string::npos);
    ASSERT_TRUE(unknown.message.find("warp_factor") != std::string::npos);

    Chorus::ProviderOptionList mixed{std::string("ok"), int64_t{2}};
    auto list = rejection_for("dry_sequence_breakers", mixed);
    ASSERT_TRUE(list.message.find("string list") != std::string::npos);
    ASSERT_TRUE(list.message.find("list containing only strings") != std::string::npos);

    auto boolean = rejection_for("ignore_eos", int64_t{1});
    ASSERT_TRUE(boolean.message.find("{false, true}") != std::string::npos);
}

TEST(LlamaGeneration, Llama_generation_rejects_namespace_errors) {
    Chorus::GenerationConfig config;
    config.provider_options["alpaca"] = Chorus::ProviderOptionMap{};
    auto unknown = Chorus::resolve_llama_generation(config);
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(unknown));
    ASSERT_TRUE(std::get<Chorus::RequestRejection>(unknown).message.find("alpaca") != std::string::npos);

    config.provider_options.clear();
    config.provider_options["llama"] = true;
    auto wrong = Chorus::resolve_llama_generation(config);
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(wrong));
    ASSERT_TRUE(std::get<Chorus::RequestRejection>(wrong).message.find("map") != std::string::npos);
}

TEST(LlamaGeneration, Llama_generation_rejects_integer_and_probability_ranges) {
    Chorus::GenerationConfig config;
    config.max_tokens = -2;
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(Chorus::resolve_llama_generation(config)));
    config.max_tokens.reset();
    config.top_k = -1;
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(Chorus::resolve_llama_generation(config)));
    config.top_k.reset();
    config.seed = uint64_t{std::numeric_limits<uint32_t>::max()} + 1;
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(Chorus::resolve_llama_generation(config)));

    ASSERT_TRUE(rejection_for("min_keep", int64_t{-1}).error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection_for("penalty_last_n", int64_t{-2}).error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection_for("dry_allowed_length", int64_t{-1}).error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection_for("dry_penalty_last_n", int64_t{-2}).error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection_for("mirostat", int64_t{3}).error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection_for("min_p", -0.01).error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection_for("typical_p", 1.01).error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection_for("xtc_probability", 1.01).error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection_for("xtc_threshold", -0.01).error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection_for("adaptive_decay", 1.0).error == Chorus::ChorusError::UnsupportedOption);
}

TEST(LlamaGeneration, Llama_generation_rejects_nonfinite_scalars) {
    Chorus::GenerationConfig config;
    config.temperature = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(Chorus::resolve_llama_generation(config)));
    config.temperature.reset();
    config.frequency_penalty = std::numeric_limits<float>::infinity();
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(Chorus::resolve_llama_generation(config)));

    ASSERT_TRUE(
        rejection_for("repeat_penalty", std::numeric_limits<double>::infinity()).error ==
        Chorus::ChorusError::UnsupportedOption
    );
    ASSERT_TRUE(
        rejection_for("mirostat_eta", std::numeric_limits<double>::quiet_NaN()).error ==
        Chorus::ChorusError::UnsupportedOption
    );
}

TEST(LlamaGeneration, Llama_sampler_order_resolves_canonical_stages) {
    Chorus::ProviderOptionList order{
        "penalties",
        "dry",
        "top_n_sigma",
        "top_k",
        "typ_p",
        "top_p",
        "min_p",
        "xtc",
        "temperature",
        "adaptive_p",
    };
    check_provider_resolution("sampler_order", order, [](const auto& value) {
        const std::vector<common_sampler_type> expected{
            COMMON_SAMPLER_TYPE_PENALTIES,
            COMMON_SAMPLER_TYPE_DRY,
            COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_TYPICAL_P,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_MIN_P,
            COMMON_SAMPLER_TYPE_XTC,
            COMMON_SAMPLER_TYPE_TEMPERATURE,
            COMMON_SAMPLER_TYPE_ADAPTIVE_P,
        };
        ASSERT_TRUE(value.sampling.samplers == expected);
    });
}

TEST(LlamaGeneration, Llama_sampler_order_preserves_defaults_when_absent_or_empty) {
    const common_params_sampling defaults;
    check_resolution(
        [](auto&) {}, [&](const auto& value) { ASSERT_TRUE(value.sampling.samplers == defaults.samplers); }
    );
    check_provider_resolution("sampler_order", Chorus::ProviderOptionList{}, [&](const auto& value) {
        ASSERT_TRUE(value.sampling.samplers == defaults.samplers);
    });
}

TEST(LlamaGeneration, Llama_sampler_order_rejects_invalid_configurations) {
    for (const std::string& name : {"top-k", "topk", "nucleus", "temp", "infill", "TOP_K", "warp"}) {
        auto rejection = rejection_for("sampler_order", Chorus::ProviderOptionList{name});
        ASSERT_TRUE(rejection.message.find(name) != std::string::npos);
    }

    auto duplicate = rejection_for(
        "sampler_order",
        Chorus::ProviderOptionList{std::string("top_k"), std::string("top_k"), std::string("temperature")}
    );
    ASSERT_TRUE(duplicate.message.find("duplicate") != std::string::npos);

    auto adaptive_not_last = rejection_for(
        "sampler_order", Chorus::ProviderOptionList{std::string("adaptive_p"), std::string("temperature")}
    );
    ASSERT_TRUE(adaptive_not_last.message.find("last") != std::string::npos);

    Chorus::GenerationConfig mirostat_config;
    mirostat_config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"mirostat", int64_t{1}},
        {"sampler_order", Chorus::ProviderOptionList{std::string("temperature")}},
    };
    auto mirostat = Chorus::resolve_llama_generation(mirostat_config);
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(mirostat));
    ASSERT_TRUE(std::get<Chorus::RequestRejection>(mirostat).message.find("Mirostat") != std::string::npos);
}

TEST(LlamaGeneration, Llama_logit_bias_resolves_decimal_token_ids) {
    Chorus::ProviderOptionMap biases{{"0", 1.25}, {"17", -2.5}, {"2147483647", 0.0}};
    check_provider_resolution("logit_bias", biases, [](const auto& value) {
        ASSERT_EQ(value.sampling.logit_bias.size(), size_t{3});
        const auto* zero = find_logit_bias(value.sampling, 0);
        const auto* seventeen = find_logit_bias(value.sampling, 17);
        const auto* maximum = find_logit_bias(value.sampling, std::numeric_limits<int32_t>::max());
        ASSERT_TRUE(zero != nullptr);
        ASSERT_TRUE(seventeen != nullptr);
        ASSERT_TRUE(maximum != nullptr);
        ASSERT_TRUE(zero->bias == 1.25f);
        ASSERT_TRUE(seventeen->bias == -2.5f);
        ASSERT_TRUE(maximum->bias == 0.0f);
    });
}

TEST(LlamaGeneration, Llama_logit_bias_rejects_invalid_entries) {
    for (const std::string& key : {"-1", "2147483648", "1.0", "1x", "+1", " 1", "0x10", ""}) {
        auto rejection = rejection_for("logit_bias", Chorus::ProviderOptionMap{{key, 0.5}});
        ASSERT_TRUE(rejection.message.find(key) != std::string::npos);
    }

    auto wrong_value = rejection_for("logit_bias", Chorus::ProviderOptionMap{{"1", int64_t{2}}});
    ASSERT_TRUE(wrong_value.message.find("double") != std::string::npos);

    auto nonfinite =
        rejection_for("logit_bias", Chorus::ProviderOptionMap{{"1", std::numeric_limits<double>::infinity()}});
    ASSERT_TRUE(nonfinite.message.find("finite") != std::string::npos);
}

TEST_F(LlamaGenerationModelTest, Llama_sampler_constructs_for_vocabulary_and_rejects_out_of_bounds_logit_bias) {
    LlamaModelFixture fixture;
    ASSERT_TRUE(fixture.load());

    Chorus::GenerationConfig valid_config;
    valid_config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"ignore_eos", true},
        {"logit_bias", Chorus::ProviderOptionMap{{"0", 1.0}}},
        {"sampler_order", Chorus::ProviderOptionList{std::string("temperature")}},
    };
    auto valid_resolution = Chorus::resolve_llama_generation(valid_config);
    const auto* valid_generation = std::get_if<Chorus::ResolvedLlamaGeneration>(&valid_resolution);
    ASSERT_TRUE(valid_generation != nullptr);
    if (!valid_generation)
        return;
    auto valid_sampler = Chorus::make_llama_sampler(fixture.model, std::move(*valid_generation));
    const auto* sampler = std::get_if<common_sampler_ptr>(&valid_sampler);
    ASSERT_TRUE(sampler != nullptr);
    if (sampler)
        ASSERT_TRUE(*sampler != nullptr);

    const auto* vocab = llama_model_get_vocab(fixture.model);
    const auto outside = llama_vocab_n_tokens(vocab);
    Chorus::GenerationConfig invalid_config;
    invalid_config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"logit_bias", Chorus::ProviderOptionMap{{std::to_string(outside), 1.0}}},
    };
    auto invalid_resolution = Chorus::resolve_llama_generation(invalid_config);
    const auto* invalid_generation = std::get_if<Chorus::ResolvedLlamaGeneration>(&invalid_resolution);
    ASSERT_TRUE(invalid_generation != nullptr);
    if (!invalid_generation)
        return;
    auto invalid_sampler = Chorus::make_llama_sampler(fixture.model, std::move(*invalid_generation));
    const auto* rejection = std::get_if<Chorus::RequestRejection>(&invalid_sampler);
    ASSERT_TRUE(rejection != nullptr);
    if (rejection) {
        ASSERT_EQ(rejection->error, Chorus::ChorusError::UnsupportedOption);
        ASSERT_TRUE(rejection->message.find(std::to_string(outside)) != std::string::npos);
        ASSERT_TRUE(rejection->message.find("vocabulary") != std::string::npos);
    }
}

TEST(LlamaGeneration, Llama_rejects_empty_constraint_sources) {
    const auto gbnf = constraint_rejection_for(Chorus::ConstraintFormat::Gbnf, "");
    ASSERT_TRUE(gbnf.error == Chorus::ChorusError::InvalidRequest);
    ASSERT_EQ(gbnf.message, std::string("GBNF constraint source must not be empty."));

    const auto schema = constraint_rejection_for(Chorus::ConstraintFormat::JsonSchema, "");
    ASSERT_TRUE(schema.error == Chorus::ChorusError::InvalidRequest);
    ASSERT_EQ(schema.message, std::string("JSON Schema constraint source must not be empty."));
}

TEST(LlamaGeneration, Llama_rejects_malformed_JSON_Schema_constraint) {
    const auto rejection = constraint_rejection_for(Chorus::ConstraintFormat::JsonSchema, R"({"type":})");
    ASSERT_TRUE(rejection.error == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(rejection.message.find("Invalid JSON Schema constraint: ") == 0);
    ASSERT_TRUE(rejection.message.find("parse error") != std::string::npos);
}

TEST(LlamaGeneration, Llama_resolves_GBNF_constraint) {
    Chorus::GenerationConfig config;
    config.constraint = Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, R"(root ::= "PINK_MOTH")"};
    auto result = Chorus::resolve_llama_generation(config);
    ASSERT_TRUE(std::holds_alternative<Chorus::ResolvedLlamaGeneration>(result));
    const auto& grammar = std::get<Chorus::ResolvedLlamaGeneration>(result).sampling.grammar;
    ASSERT_TRUE(grammar.type == COMMON_GRAMMAR_TYPE_USER);
    ASSERT_EQ(grammar.grammar, std::string(R"(root ::= "PINK_MOTH")"));
}

TEST(LlamaGeneration, Llama_converts_JSON_Schema_constraint) {
    Chorus::GenerationConfig config;
    config.constraint = Chorus::OutputConstraint{
        Chorus::ConstraintFormat::JsonSchema,
        R"({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]})",
    };
    auto result = Chorus::resolve_llama_generation(config);
    ASSERT_TRUE(std::holds_alternative<Chorus::ResolvedLlamaGeneration>(result));
    const auto& grammar = std::get<Chorus::ResolvedLlamaGeneration>(result).sampling.grammar;
    ASSERT_TRUE(grammar.type == COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT);
    ASSERT_TRUE(!grammar.grammar.empty());
}

TEST(LlamaGeneration, Llama_rejects_unsupported_constraint_formats) {
    const auto regex = constraint_rejection_for(Chorus::ConstraintFormat::Regex, "PINK_MOTH");
    ASSERT_TRUE(regex.error == Chorus::ChorusError::UnsupportedFeature);
    ASSERT_EQ(regex.message, std::string("Regex output constraints are not supported by Llama."));

    const auto lark = constraint_rejection_for(Chorus::ConstraintFormat::Lark, "start: WORD");
    ASSERT_TRUE(lark.error == Chorus::ChorusError::UnsupportedFeature);
    ASSERT_EQ(lark.message, std::string("Lark output constraints are not supported by Llama."));
}

// Capability conformance: iterate the advertised Llama catalog (the authoritative
// source, exposed by llama_common_generation_option_names /
// llama_provider_generation_option_names). Every advertised option gets a case that
// proves it resolves a distinct upstream field, and two completeness assertions pin
// the matrix to the catalog: (a) every advertised entry has a case, and (b) no case
// names an option the catalog does not advertise.
TEST(LlamaGeneration, Llama_generation_conformance_matrix_covers_the_catalog) {
    std::vector<ConformanceCase> cases;

    // Common options: each maps to a distinct resolved field.
    cases.push_back(
        {"max_tokens",
         [](Chorus::GenerationConfig& c) { c.max_tokens = 72; },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_EQ(r.max_tokens, 72); }}
    );
    cases.push_back(
        {"temperature",
         [](Chorus::GenerationConfig& c) { c.temperature = 0.35f; },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.temp == 0.35f); }}
    );
    cases.push_back(
        {"top_k",
         [](Chorus::GenerationConfig& c) { c.top_k = 17; },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_EQ(r.sampling.top_k, 17); }}
    );
    cases.push_back(
        {"top_p",
         [](Chorus::GenerationConfig& c) { c.top_p = 0.72f; },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.top_p == 0.72f); }}
    );
    cases.push_back(
        {"seed",
         [](Chorus::GenerationConfig& c) { c.seed = uint64_t{42}; },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_EQ(r.sampling.seed, uint32_t{42}); }}
    );
    cases.push_back(
        {"frequency_penalty",
         [](Chorus::GenerationConfig& c) { c.frequency_penalty = 0.4f; },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.penalty_freq == 0.4f); }}
    );
    cases.push_back(
        {"presence_penalty",
         [](Chorus::GenerationConfig& c) { c.presence_penalty = -0.2f; },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.penalty_present == -0.2f); }}
    );
    cases.push_back(
        {"constraint",
         [](Chorus::GenerationConfig& c) {
             c.constraint = Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, R"(root ::= "X")"};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) {
             ASSERT_TRUE(r.sampling.grammar.type == COMMON_GRAMMAR_TYPE_USER);
         }}
    );
    cases.push_back(
        {"stop",
         [](Chorus::GenerationConfig& c) { c.stop = {"HALT"}; },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.stop == std::vector<std::string>({"HALT"})); }}
    );

    // Provider options: each namespaced key maps to a distinct resolved field.
    cases.push_back(
        {"min_keep",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"min_keep", int64_t{3}}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_EQ(r.sampling.min_keep, 3); }}
    );
    cases.push_back(
        {"min_p",
         [](Chorus::GenerationConfig& c) { c.provider_options["llama"] = Chorus::ProviderOptionMap{{"min_p", 0.12}}; },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.min_p == 0.12f); }}
    );
    cases.push_back(
        {"typical_p",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"typical_p", 0.83}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.typ_p == 0.83f); }}
    );
    cases.push_back(
        {"dynamic_temperature_range",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"dynamic_temperature_range", 0.4}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.dynatemp_range == 0.4f); }}
    );
    cases.push_back(
        {"dynamic_temperature_exponent",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"dynamic_temperature_exponent", 1.6}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.dynatemp_exponent == 1.6f); }}
    );
    cases.push_back(
        {"penalty_last_n",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"penalty_last_n", int64_t{33}}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_EQ(r.sampling.penalty_last_n, 33); }}
    );
    cases.push_back(
        {"repeat_penalty",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"repeat_penalty", 1.15}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.penalty_repeat == 1.15f); }}
    );
    cases.push_back(
        {"ignore_eos",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.ignore_eos); }}
    );
    cases.push_back(
        {"mirostat",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"mirostat", int64_t{2}}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_EQ(r.sampling.mirostat, 2); }}
    );
    cases.push_back(
        {"mirostat_tau",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"mirostat_tau", 4.5}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.mirostat_tau == 4.5f); }}
    );
    cases.push_back(
        {"mirostat_eta",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"mirostat_eta", 0.2}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.mirostat_eta == 0.2f); }}
    );
    cases.push_back(
        {"xtc_probability",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"xtc_probability", 0.3}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.xtc_probability == 0.3f); }}
    );
    cases.push_back(
        {"xtc_threshold",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"xtc_threshold", 0.45}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.xtc_threshold == 0.45f); }}
    );
    cases.push_back(
        {"dry_multiplier",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"dry_multiplier", 0.7}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.dry_multiplier == 0.7f); }}
    );
    cases.push_back(
        {"dry_base",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"dry_base", 2.0}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.dry_base == 2.0f); }}
    );
    cases.push_back(
        {"dry_allowed_length",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"dry_allowed_length", int64_t{5}}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_EQ(r.sampling.dry_allowed_length, 5); }}
    );
    cases.push_back(
        {"dry_penalty_last_n",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"dry_penalty_last_n", int64_t{81}}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_EQ(r.sampling.dry_penalty_last_n, 81); }}
    );
    cases.push_back(
        {"dry_sequence_breakers",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] =
                 Chorus::ProviderOptionMap{{"dry_sequence_breakers", Chorus::ProviderOptionList{std::string("###")}}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) {
             ASSERT_TRUE(r.sampling.dry_sequence_breakers == std::vector<std::string>({"###"}));
         }}
    );
    cases.push_back(
        {"sampler_order",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] =
                 Chorus::ProviderOptionMap{{"sampler_order", Chorus::ProviderOptionList{std::string("temperature")}}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) {
             ASSERT_TRUE(r.sampling.samplers == std::vector<common_sampler_type>({COMMON_SAMPLER_TYPE_TEMPERATURE}));
         }}
    );
    cases.push_back(
        {"logit_bias",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] =
                 Chorus::ProviderOptionMap{{"logit_bias", Chorus::ProviderOptionMap{{"0", 1.25}}}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) {
             const auto* bias = find_logit_bias(r.sampling, 0);
             ASSERT_TRUE(bias != nullptr);
             ASSERT_TRUE(bias->bias == 1.25f);
         }}
    );
    cases.push_back(
        {"top_n_sigma",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"top_n_sigma", -1.5}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.top_n_sigma == -1.5f); }}
    );
    cases.push_back(
        {"adaptive_target",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"adaptive_target", 0.6}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.adaptive_target == 0.6f); }}
    );
    cases.push_back(
        {"adaptive_decay",
         [](Chorus::GenerationConfig& c) {
             c.provider_options["llama"] = Chorus::ProviderOptionMap{{"adaptive_decay", 0.95}};
         },
         [](const Chorus::ResolvedLlamaGeneration& r) { ASSERT_TRUE(r.sampling.adaptive_decay == 0.95f); }}
    );
    cases.push_back(
        {"show_thinking",
         [](Chorus::GenerationConfig& c) { c.show_thinking = false; },
         [](const Chorus::ResolvedLlamaGeneration& r) {
             // Honored at chat-render time, not in the sampler: the
             // resolver's whole contract for this option is "accepted".
             (void)r;
         }}
    );

    std::set<std::string> covered;
    for (const auto& conformance_case : cases) {
        covered.insert(conformance_case.name);
        Chorus::GenerationConfig config;
        conformance_case.configure(config);
        auto result = Chorus::resolve_llama_generation(config);
        if (!std::holds_alternative<Chorus::ResolvedLlamaGeneration>(result)) {
            ADD_FAILURE() << "Conformance case '" << conformance_case.name << "' did not resolve.";
            return;
        }
        conformance_case.verify(std::get<Chorus::ResolvedLlamaGeneration>(result));
    }

    // Build the advertised set from the authoritative catalog accessors.
    std::set<std::string> advertised;
    for (const auto& name : Chorus::llama_common_generation_option_names())
        advertised.insert(name);
    for (const auto& name : Chorus::llama_provider_generation_option_names())
        advertised.insert(name);

    // (a) Every advertised catalog entry has a conformance case.
    for (const auto& name : advertised)
        ASSERT_EQ(covered.count(name), 1) << "advertised option lacks a conformance case: " << name;
    // (b) No conformance case names an option the catalog does not advertise.
    for (const auto& name : covered)
        ASSERT_EQ(advertised.count(name), 1) << "conformance case names an unadvertised option: " << name;
    ASSERT_EQ(covered.size(), advertised.size());
}

TEST(LlamaGeneration, Llama_request_rejects_chat_controls_without_messages) {
    Chorus::ChorusRequest with_template;
    with_template.prompt = "raw";
    with_template.chat_template = "{{ messages }}";
    auto rejection = Chorus::validate_llama_request(with_template);
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection->message.find("chat_template") != std::string::npos);

    Chorus::ChorusRequest with_show_thinking;
    with_show_thinking.prompt = "raw";
    with_show_thinking.gen_config.show_thinking = false;
    rejection = Chorus::validate_llama_request(with_show_thinking);
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection->message.find("show_thinking") != std::string::npos);
}

TEST(LlamaGeneration, Llama_request_accepts_chat_controls_with_messages) {
    Chorus::ChorusRequest request;
    request.messages = {{Chorus::MessageRole::User, Chorus::MessageContent::text("hello")}};
    request.chat_template = "{{ messages }}";
    request.gen_config.show_thinking = false;
    ASSERT_TRUE(!Chorus::validate_llama_request(request).has_value());
}

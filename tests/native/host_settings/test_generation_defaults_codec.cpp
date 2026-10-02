#include "host_settings/generation_defaults_codec.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {
using namespace std::string_literals;
using chorus_host_settings::parse_generation_defaults;
using chorus_host_settings::serialize_generation_defaults;
using namespace Chorus;

std::string document(const std::string& fields) {
    return "{\"version\":1,\"generation\":{" + fields + "}}";
}

TEST(HostSettingsCodecTest, EmptyDocumentDoesNotSelectProviderDefaults) {
    const auto parsed = parse_generation_defaults("{\"version\":1,\"generation\":{}}");
    ASSERT_TRUE(parsed.ok()) << parsed.path << parsed.error;
    EXPECT_FALSE(parsed.defaults.options.max_tokens);
    EXPECT_FALSE(parsed.defaults.options.stop);
    EXPECT_FALSE(parsed.defaults.chat_template);
    const auto encoded = serialize_generation_defaults(parsed.defaults);
    ASSERT_TRUE(encoded.ok()) << encoded.path << encoded.error;
    EXPECT_EQ(encoded.json, document(""));
}

TEST(HostSettingsCodecTest, RawNulAfterValidDocumentIsNotEndOfInput) {
    const auto valid = document("\"chat_template\":\"\\u0000\"");
    const auto escaped = parse_generation_defaults(valid);
    ASSERT_TRUE(escaped.ok()) << escaped.error;
    EXPECT_EQ(escaped.defaults.chat_template, std::string(1, '\0'));
    for (const auto& suffix : {std::string(1, '\0'), std::string("\0garbage", 8)}) {
        const auto parsed = parse_generation_defaults(valid + suffix);
        EXPECT_FALSE(parsed.ok());
        EXPECT_FALSE(parsed.defaults.chat_template);
    }
}

TEST(HostSettingsCodecTest, SelectedZeroFalseEmptyAndAllCommonValuesSurvive) {
    const auto parsed = parse_generation_defaults(document(
        "\"max_tokens\":0,\"temperature\":-0.0,\"top_k\":-2147483648,\"top_p\":0.25,"
        "\"seed\":18446744073709551615,\"frequency_penalty\":-0.75,\"presence_penalty\":1.25,"
        "\"stop\":[],\"show_thinking\":false,\"chat_template\":\"\","
        "\"constraint\":{\"kind\":\"unconstrained\"}"
    ));
    ASSERT_TRUE(parsed.ok()) << parsed.path << parsed.error;
    const auto& o = parsed.defaults.options;
    EXPECT_EQ(o.max_tokens, 0);
    EXPECT_TRUE(std::signbit(*o.temperature));
    EXPECT_EQ(o.top_k, INT32_MIN);
    EXPECT_EQ(o.top_p, 0.25f);
    EXPECT_EQ(o.seed, UINT64_MAX);
    EXPECT_EQ(o.frequency_penalty, -0.75f);
    EXPECT_EQ(o.presence_penalty, 1.25f);
    EXPECT_TRUE(o.stop && o.stop->empty());
    EXPECT_EQ(o.show_thinking, false);
    EXPECT_EQ(parsed.defaults.chat_template, "");
    ASSERT_TRUE(o.constraint);
    EXPECT_TRUE(std::holds_alternative<UnconstrainedOutput>(*o.constraint));
    const auto encoded = serialize_generation_defaults(parsed.defaults);
    ASSERT_TRUE(encoded.ok()) << encoded.path << encoded.error;
    EXPECT_NE(encoded.json.find("18446744073709551615"), std::string::npos);
    const auto again = parse_generation_defaults(encoded.json);
    ASSERT_TRUE(again.ok()) << again.path << again.error;
    EXPECT_EQ(again.defaults.options.seed, UINT64_MAX);
    EXPECT_TRUE(std::signbit(*again.defaults.options.temperature));
    EXPECT_TRUE(again.defaults.options.stop && again.defaults.options.stop->empty());
    EXPECT_EQ(again.defaults.chat_template, "");
    EXPECT_TRUE(std::holds_alternative<UnconstrainedOutput>(*again.defaults.options.constraint));
}

TEST(HostSettingsCodecTest, EveryCompleteConstraintFormatAndUtf8RoundTrip) {
    for (const auto& kind : {"gbnf", "json_schema", "regex", "lark"}) {
        const std::string source = "é雪\0grammar"s;
        GenerationDefaults defaults;
        ConstraintFormat format = kind == std::string("gbnf")          ? ConstraintFormat::Gbnf
                                  : kind == std::string("json_schema") ? ConstraintFormat::JsonSchema
                                  : kind == std::string("regex")       ? ConstraintFormat::Regex
                                                                       : ConstraintFormat::Lark;
        defaults.options.constraint = OutputConstraint{format, source};
        defaults.options.stop = std::vector<std::string>{source};
        defaults.chat_template = source;
        const auto encoded = serialize_generation_defaults(defaults);
        ASSERT_TRUE(encoded.ok()) << encoded.path << encoded.error;
        const auto parsed = parse_generation_defaults(encoded.json);
        ASSERT_TRUE(parsed.ok()) << parsed.path << parsed.error;
        EXPECT_EQ(std::get<OutputConstraint>(*parsed.defaults.options.constraint).source, source);
        EXPECT_EQ(std::get<OutputConstraint>(*parsed.defaults.options.constraint).format, format);
        EXPECT_EQ((*parsed.defaults.options.stop)[0], source);
        EXPECT_EQ(parsed.defaults.chat_template, source);
    }
}

TEST(HostSettingsCodecTest, ProviderOptionTypesAndCompleteEmptyMapSurvive) {
    const auto parsed = parse_generation_defaults(document(
        "\"provider_options\":{\"empty_namespace\":{},\"llama\":{\"empty_map\":{},"
        "\"list\":[false,-9223372036854775808,9223372036854775807,1.0,\"\",[],"
        "{\"nested\":[0,2e0,-0.0]}]}}"
    ));
    ASSERT_TRUE(parsed.ok()) << parsed.path << parsed.error;
    const auto& providers = parsed.defaults.options.provider_options;
    EXPECT_EQ(providers.count("empty_namespace"), 0u);
    const auto& choices = std::get<ProviderOptionMap>(providers.at("llama"));
    EXPECT_TRUE(std::get<ProviderOptionMap>(choices.at("empty_map")).empty());
    const auto& list = std::get<ProviderOptionList>(choices.at("list"));
    EXPECT_TRUE(std::holds_alternative<int64_t>(list[1]));
    EXPECT_EQ(std::get<int64_t>(list[1]), INT64_MIN);
    EXPECT_EQ(std::get<int64_t>(list[2]), INT64_MAX);
    EXPECT_TRUE(std::holds_alternative<double>(list[3]));
    const auto encoded = serialize_generation_defaults(parsed.defaults);
    ASSERT_TRUE(encoded.ok()) << encoded.path << encoded.error;
    EXPECT_NE(encoded.json.find("1.0"), std::string::npos);
    const auto again = parse_generation_defaults(encoded.json);
    ASSERT_TRUE(again.ok()) << again.path << again.error;
    const auto& same = std::get<ProviderOptionList>(
        std::get<ProviderOptionMap>(again.defaults.options.provider_options.at("llama")).at("list")
    );
    EXPECT_TRUE(std::holds_alternative<double>(same[3]));
    EXPECT_TRUE(
        std::signbit(
            std::get<double>(std::get<ProviderOptionList>(std::get<ProviderOptionMap>(same[6]).at("nested"))[2])
        )
    );
}

TEST(HostSettingsCodecTest, SixtyFourContainerLevelsAcceptedSixtyFiveRejected) {
    auto nested = [](int count) {
        std::string value = "false";
        for (int i = 0; i < count; ++i)
            value = "[" + value + "]";
        return document("\"provider_options\":{\"p\":{\"v\":" + value + "}}");
    };
    const auto accepted = parse_generation_defaults(nested(64));
    ASSERT_TRUE(accepted.ok()) << accepted.path << accepted.error;
    EXPECT_TRUE(serialize_generation_defaults(accepted.defaults).ok());
    const auto rejected = parse_generation_defaults(nested(65));
    EXPECT_FALSE(rejected.ok());
    EXPECT_NE(rejected.path.find("/generation/provider_options/p/v"), std::string::npos);
    auto deep = accepted.defaults;
    auto* value = &std::get<ProviderOptionMap>(deep.options.provider_options.at("p")).at("v");
    for (int i = 0; i < 64; ++i)
        value = &std::get<ProviderOptionList>(*value).at(0);
    *value = ProviderOptionList{false};
    EXPECT_EQ(serialize_generation_defaults(deep).error, "option container nesting exceeds 64");
}

TEST(HostSettingsCodecTest, InvalidDocumentsReportErrorsWithoutPublishingChoices) {
    const std::vector<std::string> invalid = {
        "",
        "{}",
        "[]",
        "{",
        "{\"version\":2,\"generation\":{}}",
        "{\"version\":1.0,\"generation\":{}}",
        "{\"version\":1,\"generation\":{},\"extra\":0}",
        document("\"unknown\":true"),
        document("\"max_tokens\":null"),
        document("\"max_tokens\":2147483648"),
        document("\"top_k\":1.0"),
        document("\"temperature\":1e100"),
        document("\"temperature\":1e-100"),
        document("\"temperature\":1e-400"),
        document("\"temperature\":1e309"),
        document("\"provider_options\":{\"p\":{\"v\":1e-400}}"),
        document("\"seed\":-1"),
        document("\"seed\":18446744073709551616"),
        document("\"seed\":1.0"),
        document("\"max_tokens\":-2147483649"),
        document("\"show_thinking\":0"),
        document("\"stop\":[null]"),
        document("\"constraint\":{\"kind\":\"gbnf\"}"),
        document("\"constraint\":{\"kind\":\"not_a_format\",\"source\":\"\"}"),
        document("\"constraint\":{\"kind\":\"unconstrained\",\"source\":\"\"}"),
        document("\"constraint\":{\"kind\":\"gbnf\",\"source\":\"\",\"extra\":0}"),
        document("\"provider_options\":{\"p\":{\"v\":null}}"),
        document("\"provider_options\":{\"p\":{\"v\":18446744073709551615}}"),
        document("\"provider_options\":{\"p\":{\"v\":9223372036854775808}}"),
        document("\"provider_options\":{\"p\":{\"v\":-9223372036854775809}}"),
        document("\"provider_options\":{\"p\":{\"v\":18446744073709551616}}"),
        document("\"provider_options\":{\"p\":{\"v\":1e309}}"),
        document("\"provider_options\":{\"p\":false}"),
        document("\"chat_template\":1"),
        document("\"chat_template\":\"\xFF\""),
        document("\"provider_options\":{\"p\":{\"\xFF\":true}}")
    };
    for (const auto& text : invalid) {
        const auto result = parse_generation_defaults(text);
        EXPECT_FALSE(result.ok()) << text;
        EXPECT_FALSE(result.defaults.options.max_tokens) << text;
    }
}

TEST(HostSettingsCodecTest, DuplicateKeysAtEveryLevelAreRejected) {
    for (const auto& text : std::vector<std::string>{
             "{\"version\":1,\"version\":1,\"generation\":{}}",
             document("\"max_tokens\":1,\"max_tokens\":2"),
             document("\"constraint\":{\"kind\":\"gbnf\",\"kind\":\"regex\",\"source\":\"\"}"),
             document("\"provider_options\":{\"p\":{\"nested\":{\"x\":0,\"x\":1}}}"),
             document("\"provider_options\":{\"p\":{},\"p\":{}}")
         }) {
        const auto result = parse_generation_defaults(text);
        EXPECT_FALSE(result.ok()) << text;
        EXPECT_EQ(result.error, "duplicate key") << text;
        EXPECT_FALSE(result.path.empty()) << text;
        if (text.find("nested") != std::string::npos)
            EXPECT_EQ(result.path, "/generation/provider_options/p/nested/x");
    }
}

TEST(HostSettingsCodecTest, SerializationRejectsNonfiniteAndMalformedHostValues) {
    GenerationDefaults defaults;
    defaults.options.temperature = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(serialize_generation_defaults(defaults).ok());
    defaults.options.temperature.reset();
    defaults.options.provider_options["p"] = ProviderOptionMap{{"v", std::numeric_limits<double>::quiet_NaN()}};
    EXPECT_FALSE(serialize_generation_defaults(defaults).ok());
    defaults.options.provider_options["p"] = true;
    EXPECT_FALSE(serialize_generation_defaults(defaults).ok());
    defaults.options.provider_options.clear();
    defaults.chat_template = std::string(1, '\xFF');
    EXPECT_EQ(serialize_generation_defaults(defaults).path, "/generation/chat_template");
    defaults.chat_template.reset();
    defaults.options.provider_options["p"] = ProviderOptionMap{{std::string(1, '\xFF'), true}};
    EXPECT_EQ(serialize_generation_defaults(defaults).path, "/generation/provider_options/p/\xFF");
}

TEST(HostSettingsCodecTest, FloatPrecisionAndSignedZeroSurviveRepeatedEncoding) {
    GenerationDefaults defaults;
    defaults.options.temperature = std::numeric_limits<float>::denorm_min();
    defaults.options.top_p = std::numeric_limits<float>::max();
    defaults.options.provider_options["p"] = ProviderOptionMap{
        {"small", std::numeric_limits<double>::denorm_min()},
        {"large", std::numeric_limits<double>::max()},
        {"negative_zero", -0.0}
    };
    const auto encoded = serialize_generation_defaults(defaults);
    ASSERT_TRUE(encoded.ok()) << encoded.path << encoded.error;
    const auto parsed = parse_generation_defaults(encoded.json);
    ASSERT_TRUE(parsed.ok()) << parsed.path << parsed.error;
    EXPECT_EQ(parsed.defaults.options.temperature, defaults.options.temperature);
    EXPECT_EQ(parsed.defaults.options.top_p, defaults.options.top_p);
    const auto& values = std::get<ProviderOptionMap>(parsed.defaults.options.provider_options.at("p"));
    EXPECT_EQ(std::get<double>(values.at("small")), std::numeric_limits<double>::denorm_min());
    EXPECT_EQ(std::get<double>(values.at("large")), std::numeric_limits<double>::max());
    EXPECT_TRUE(std::signbit(std::get<double>(values.at("negative_zero"))));
    const auto zero_exponent = parse_generation_defaults(document("\"temperature\":0e-400"));
    ASSERT_TRUE(zero_exponent.ok()) << zero_exponent.error;
    EXPECT_EQ(zero_exponent.defaults.options.temperature, 0.0f);
    const auto escaped = parse_generation_defaults(document("\"chat_template\":\"a\\u0000b\""));
    ASSERT_TRUE(escaped.ok()) << escaped.error;
    EXPECT_EQ(escaped.defaults.chat_template, std::string("a\0b", 3));
}

} // namespace

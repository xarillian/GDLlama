#include "chorus/core/capabilities.hpp"

#include "gtest_utils.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <variant>

namespace {

Chorus::ProviderOptionDescriptors schema() {
    return {
        {"use_gpu", "Use GPU", "", true, std::nullopt, std::nullopt, std::nullopt, std::nullopt},
        {"threads", "Threads", "", int64_t{4}, 1, 32, 1, std::nullopt},
        {"gpu_index", "GPU Index", "", int64_t{0}, 0, 15, 1, "use_gpu"},
        {"orphan", "Orphan", "", int64_t{7}, std::nullopt, std::nullopt, std::nullopt, "no_such_option"},
    };
}

TEST(OptionSchema, Option_schema_preserves_provider_owned_string_choices) {
    const Chorus::ProviderOptionDescriptor descriptor{
        "pooling", "Pooling", "", std::string{"model"}, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        {"model", "none", "mean", "cls", "last"}
    };
    ASSERT_EQ(descriptor.choices.size(), size_t{5});
    ASSERT_EQ(descriptor.choices[0], std::string("model"));
    ASSERT_EQ(descriptor.choices[4], std::string("last"));
}

TEST(OptionSchema, Option_schema_finds_declared_keys) {
    const auto descriptors = schema();
    const auto* found = Chorus::find_option_descriptor(descriptors, "threads");
    ASSERT_TRUE(found != nullptr);
    ASSERT_EQ(std::get<int64_t>(found->default_value), int64_t{4});
    ASSERT_TRUE(Chorus::find_option_descriptor(descriptors, "nope") == nullptr);
}

TEST(OptionSchema, Option_schema_fills_declared_defaults) {
    const auto descriptors = schema();
    const auto resolved = Chorus::resolve_option_defaults(descriptors, {});

    ASSERT_EQ(std::get<int64_t>(resolved.at("threads")), int64_t{4});
    ASSERT_TRUE(std::get<bool>(resolved.at("use_gpu")));
    // use_gpu defaults true, so gpu_index resolves alongside it.
    ASSERT_EQ(std::get<int64_t>(resolved.at("gpu_index")), int64_t{0});
}

TEST(OptionSchema, Option_schema_stored_value_beats_default) {
    const auto descriptors = schema();
    Chorus::ProviderOptionMap stored{{"threads", int64_t{16}}};

    const auto resolved = Chorus::resolve_option_defaults(descriptors, stored);
    ASSERT_EQ(std::get<int64_t>(resolved.at("threads")), int64_t{16});
    ASSERT_EQ(std::get<int64_t>(resolved.at("gpu_index")), int64_t{0}); // untouched keys still resolve
}

TEST(OptionSchema, Option_schema_drops_options_whose_prerequisite_is_off) {
    const auto descriptors = schema();
    Chorus::ProviderOptionMap stored{{"use_gpu", false}, {"gpu_index", int64_t{3}}};

    const auto resolved = Chorus::resolve_option_defaults(descriptors, stored);
    ASSERT_TRUE(resolved.find("gpu_index") == resolved.end());
    // Only gpu_index leaves; use_gpu and its peers stay.
    ASSERT_TRUE(std::get<bool>(resolved.at("use_gpu")) == false);
    ASSERT_EQ(std::get<int64_t>(resolved.at("threads")), int64_t{4});
}

TEST(OptionSchema, Option_schema_prerequisite_naming_unknown_option_disables) {
    const auto descriptors = schema();
    const auto resolved = Chorus::resolve_option_defaults(descriptors, {});
    // orphan names no_such_option, which the schema never declares. A prerequisite nothing
    // can establish is never met, so dropping the option beats sending it.
    ASSERT_TRUE(resolved.find("orphan") == resolved.end());
}

TEST(OptionSchema, Option_schema_ignores_keys_outside_the_declaration) {
    const auto descriptors = schema();
    Chorus::ProviderOptionMap stored{{"threads", int64_t{8}}, {"leftover_from_another_provider", int64_t{1}}};

    const auto resolved = Chorus::resolve_option_defaults(descriptors, stored);
    ASSERT_EQ(std::get<int64_t>(resolved.at("threads")), int64_t{8});
    ASSERT_TRUE(resolved.find("leftover_from_another_provider") == resolved.end());
}

} // namespace

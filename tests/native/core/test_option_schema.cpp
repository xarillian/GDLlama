#include "chorus/core/capabilities.hpp"

#include "test_utils.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

namespace {

std::vector<Chorus::OptionDescriptor> schema() {
    return {
        {"use_gpu", "Use GPU", "", true, std::nullopt, std::nullopt, std::nullopt, std::nullopt},
        {"threads", "Threads", "", int64_t{4}, 1, 32, 1, std::nullopt},
        {"gpu_index", "GPU Index", "", int64_t{0}, 0, 15, 1, "use_gpu"},
        {"orphan", "Orphan", "", int64_t{7}, std::nullopt, std::nullopt, std::nullopt, "no_such_option"},
    };
}

void test_option_schema_finds_declared_keys() {
    const auto descriptors = schema();
    const auto* found = Chorus::find_option_descriptor(descriptors, "threads");
    ASSERT_TRUE(found != nullptr);
    ASSERT_EQ(std::get<int64_t>(found->default_value), int64_t{4});
    ASSERT_TRUE(Chorus::find_option_descriptor(descriptors, "nope") == nullptr);
}

void test_option_schema_fills_declared_defaults() {
    const auto descriptors = schema();
    const auto resolved = Chorus::resolve_option_defaults(descriptors, {});

    ASSERT_EQ(std::get<int64_t>(resolved.at("threads")), int64_t{4});
    ASSERT_TRUE(std::get<bool>(resolved.at("use_gpu")));
    // The gate defaults to true, so the option it governs is present.
    ASSERT_EQ(std::get<int64_t>(resolved.at("gpu_index")), int64_t{0});
}

void test_option_schema_stored_value_beats_default() {
    const auto descriptors = schema();
    Chorus::ProviderOptionMap stored{{"threads", int64_t{16}}};

    const auto resolved = Chorus::resolve_option_defaults(descriptors, stored);
    ASSERT_EQ(std::get<int64_t>(resolved.at("threads")), int64_t{16});
    ASSERT_EQ(std::get<int64_t>(resolved.at("gpu_index")), int64_t{0}); // untouched keys still resolve
}

void test_option_schema_drops_gated_off_options() {
    const auto descriptors = schema();
    Chorus::ProviderOptionMap stored{{"use_gpu", false}, {"gpu_index", int64_t{3}}};

    const auto resolved = Chorus::resolve_option_defaults(descriptors, stored);
    ASSERT_TRUE(resolved.find("gpu_index") == resolved.end());
    // Only the gated option leaves; the gate and its peers stay.
    ASSERT_TRUE(std::get<bool>(resolved.at("use_gpu")) == false);
    ASSERT_EQ(std::get<int64_t>(resolved.at("threads")), int64_t{4});
}

void test_option_schema_gate_naming_unknown_option_disables() {
    const auto descriptors = schema();
    const auto resolved = Chorus::resolve_option_defaults(descriptors, {});
    // A gate the provider never declared cannot be satisfied; dropping the
    // option beats sending one whose precondition nothing can establish.
    ASSERT_TRUE(resolved.find("orphan") == resolved.end());
}

void test_option_schema_ignores_keys_outside_the_declaration() {
    const auto descriptors = schema();
    Chorus::ProviderOptionMap stored{{"threads", int64_t{8}}, {"leftover_from_another_provider", int64_t{1}}};

    const auto resolved = Chorus::resolve_option_defaults(descriptors, stored);
    ASSERT_EQ(std::get<int64_t>(resolved.at("threads")), int64_t{8});
    ASSERT_TRUE(resolved.find("leftover_from_another_provider") == resolved.end());
}

} // namespace

int run_option_schema_tests() {
    std::cout << "\n--- Option Schema Tests ---\n";
    run_test("Option schema finds declared keys", test_option_schema_finds_declared_keys);
    run_test("Option schema fills declared defaults", test_option_schema_fills_declared_defaults);
    run_test("Option schema stored value beats default", test_option_schema_stored_value_beats_default);
    run_test("Option schema drops gated-off options", test_option_schema_drops_gated_off_options);
    run_test(
        "Option schema gate naming unknown option disables", test_option_schema_gate_naming_unknown_option_disables
    );
    run_test(
        "Option schema ignores keys outside the declaration", test_option_schema_ignores_keys_outside_the_declaration
    );
    return 0;
}

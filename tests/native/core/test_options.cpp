#include "chorus/core/options.hpp"
#include "test_utils.hpp"

#include <variant>

// No model, no skips: this suite must pass under CHORUS_SKIP_MODEL_TESTS=1.

void test_option_value_holds_each_scalar_type() {
    Chorus::OptionValue b = true;
    Chorus::OptionValue i = int64_t{42};
    Chorus::OptionValue d = 0.5;
    Chorus::OptionValue s = std::string("hello");
    ASSERT_TRUE(std::holds_alternative<bool>(b));
    ASSERT_TRUE(std::holds_alternative<int64_t>(i));
    ASSERT_TRUE(std::holds_alternative<double>(d));
    ASSERT_TRUE(std::holds_alternative<std::string>(s));
}

void test_option_value_string_literal_is_string_not_bool() {
    // Classic variant footgun: const char* prefers the bool conversion.
    Chorus::OptionValue v = "gemma";
    ASSERT_TRUE(std::holds_alternative<std::string>(v));
    ASSERT_EQ(std::get<std::string>(v), std::string("gemma"));
}

void test_option_map_nests_and_copies() {
    Chorus::OptionMap options;
    options["llama"] = Chorus::OptionMap{
        {"gpu_layers", int64_t{99}},
        {"use_gpu", true},
        {"breakers", Chorus::OptionList{"\n", "."}},
    };
    Chorus::OptionMap copy = options; // must be deep-copyable
    const auto& llama = std::get<Chorus::OptionMap>(copy.at("llama"));
    ASSERT_EQ(std::get<int64_t>(llama.at("gpu_layers")), 99);
    ASSERT_TRUE(std::get<bool>(llama.at("use_gpu")));
    ASSERT_EQ(std::get<Chorus::OptionList>(llama.at("breakers")).size(), (size_t)2);
}

int run_options_tests() {
    std::cout << "\n--- OptionMap Tests ---\n";
    run_test("Option value: scalar alternatives", test_option_value_holds_each_scalar_type);
    run_test("Option value: literal is string", test_option_value_string_literal_is_string_not_bool);
    run_test("Option map: nesting and copy", test_option_map_nests_and_copies);
    return g_tests_failed;
}

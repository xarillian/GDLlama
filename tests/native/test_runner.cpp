#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_filter.hpp"
#include "process_test.hpp"

int run_echo_engine_child_mode(std::string_view child_name);
int run_llama_reentry_child_mode(std::string_view child_name);

int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "__chorus_child") {
        const std::string_view child_name = argv[2];
        const int echo_result = run_echo_engine_child_mode(child_name);
        return echo_result == 64 ? run_llama_reentry_child_mode(child_name) : echo_result;
    }

    set_process_test_executable_path(argv[0]);

    std::string positional_filter;
    std::vector<char*> gtest_args;
    gtest_args.reserve(static_cast<std::size_t>(argc) + 1);
    gtest_args.push_back(argv[0]);

    bool has_explicit_gtest_filter = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (!argument.starts_with("-") && positional_filter.empty()) {
            positional_filter = argument;
            continue;
        }
        has_explicit_gtest_filter |= argument.starts_with("--gtest_filter");
        gtest_args.push_back(argv[i]);
    }
    gtest_args.push_back(nullptr);

    int gtest_argc = static_cast<int>(gtest_args.size() - 1);
    ::testing::InitGoogleTest(&gtest_argc, gtest_args.data());
    if (!positional_filter.empty() && !has_explicit_gtest_filter)
        GTEST_FLAG_SET(filter, make_gtest_substring_filter(positional_filter));

    return RUN_ALL_TESTS();
}

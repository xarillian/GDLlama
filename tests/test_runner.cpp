#include <cstdlib>
#include <iostream>

#include "test_utils.hpp"

int run_options_tests();
int run_generation_config_tests();
int run_contract_type_tests();
int run_core_mechanics_tests();
int run_echo_engine_tests();
int run_engine_factory_tests();
int run_runtime_tests();
int run_runtime_session_tests();
int run_llama_integration_tests();
int run_llama_generation_tests();
int run_llama_scheduler_tests();
int run_llama_utils_tests();
int run_stop_sequence_filter_tests();

int main(int argc, char** argv) {
    g_run_model_tests = !model_tests_disabled_by_env(std::getenv("CHORUS_SKIP_MODEL_TESTS"));
    if (argc > 1)
        g_test_filter = argv[1];

    std::cout << "======================================\n";
    std::cout << "      CHORUS UNIFIED TEST SUITE       \n";
    std::cout << "======================================\n";

    run_options_tests();
    run_generation_config_tests();
    run_contract_type_tests();
    run_core_mechanics_tests();
    run_echo_engine_tests();
    run_engine_factory_tests();
    run_runtime_tests();
    run_runtime_session_tests();
    run_llama_integration_tests();
    run_llama_generation_tests();
    run_llama_scheduler_tests();
    run_llama_utils_tests();
    run_stop_sequence_filter_tests();

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << "FINAL SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED.\n";
        return 1;
    }
    std::cout << "FINAL SUMMARY: ALL TESTS PASSED.\n";
    return 0;
}

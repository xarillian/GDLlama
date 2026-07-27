#include <cstdlib>
#include <iostream>
#include <string_view>

#include "test_utils.hpp"

int run_wlib_utf8_tests();
int run_wlib_utf8_chunker_tests();
int run_options_tests();
int run_generation_config_tests();
int run_contract_type_tests();
int run_chat_type_tests();
int run_core_mechanics_tests();
int run_godot_llama_load_options_tests();
int run_process_test_tests();
int run_echo_engine_tests();
int run_engine_factory_tests();
int run_runtime_tests();
int run_runtime_session_tests();
int run_chat_history_tests();
int run_llama_integration_tests();
int run_llama_generation_tests();
int run_llama_scheduler_tests();
int run_llama_utils_tests();
int run_llama_chat_tests();
int run_llama_reasoning_tests();
int run_stop_sequence_filter_tests();
int run_prompt_fitting_tests();
int run_llama_reentry_child_mode(std::string_view child_name);

int main(int argc, char** argv) {
    g_test_executable_path = argv[0];
    if (argc == 3 && std::string_view(argv[1]) == "__chorus_child")
        return run_llama_reentry_child_mode(argv[2]);

    g_run_model_tests = !model_tests_disabled_by_env(std::getenv("CHORUS_SKIP_MODEL_TESTS"));
    if (argc > 1)
        g_test_filter = argv[1];

    std::cout << "======================================\n";
    std::cout << "      CHORUS UNIFIED TEST SUITE       \n";
    std::cout << "======================================\n";

    run_wlib_utf8_tests();
    run_wlib_utf8_chunker_tests();
    run_options_tests();
    run_generation_config_tests();
    run_contract_type_tests();
    run_chat_type_tests();
    run_core_mechanics_tests();
    run_godot_llama_load_options_tests();
    run_process_test_tests();
    run_echo_engine_tests();
    run_engine_factory_tests();
    run_runtime_tests();
    run_runtime_session_tests();
    run_chat_history_tests();
    run_llama_integration_tests();
    run_llama_generation_tests();
    run_llama_scheduler_tests();
    run_llama_utils_tests();
    run_llama_chat_tests();
    run_llama_reasoning_tests();
    run_stop_sequence_filter_tests();
    run_prompt_fitting_tests();

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << "FINAL SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED.\n";
        return 1;
    }
    std::cout << "FINAL SUMMARY: ALL TESTS PASSED.\n";
    return 0;
}

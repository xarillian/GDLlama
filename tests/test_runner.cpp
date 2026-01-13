#include <iostream>

int run_core_mechanics_tests();
int run_llama_integration_tests();

extern int g_tests_passed;
extern int g_tests_failed;

int main(int argc, char** argv) {
    std::cout << "======================================\n";
    std::cout << "      CHORUS UNIFIED TEST SUITE       \n";
    std::cout << "======================================\n";

    run_core_mechanics_tests();
    run_llama_integration_tests();

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << "FINAL SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED.\n";
        return 1;
    }
    std::cout << "FINAL SUMMARY: ALL TESTS PASSED.\n";
    return 0;
}
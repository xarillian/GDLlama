#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <chrono>

// --- GTest-Lite Macros ---
#define RED "\033[31m"
#define GREEN "\033[32m"
#define RESET "\033[0m"

inline int g_tests_passed = 0;
inline int g_tests_failed = 0;

#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        std::cerr << RED << "[FAILED] " << #condition << " at " << __FILE__ << ":" << __LINE__ << RESET << std::endl; \
        g_tests_failed++; \
        return; \
    }

#define ASSERT_EQ(val1, val2) \
    if ((val1) != (val2)) { \
        std::cerr << RED << "[FAILED] Expected " << val1 << " == " << val2 << " at " << __FILE__ << ":" << __LINE__ << RESET << std::endl; \
        g_tests_failed++; \
        return; \
    }

// A helper to run test functions and print their status
inline void run_test(const std::string& name, std::function<void()> test_func) {
    int failures_before = g_tests_failed;
    std::cout << "[RUNNING] " << name << "..." << std::endl;
    test_func();
    if (g_tests_failed == failures_before) {
        std::cout << GREEN << "[PASSED] " << name << RESET << std::endl;
        g_tests_passed++;
    }
}
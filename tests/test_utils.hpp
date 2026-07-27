#include <functional>
#include <iostream>
#include <string>

// --- GTest-Lite Macros ---
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"

inline int g_tests_passed = 0;
inline int g_tests_failed = 0;
inline bool g_run_model_tests = true; // user-controlled; false skips model-dependent tests
inline std::string g_test_filter;     // empty = run every test; else a name substring filter
inline std::string g_test_executable_path;

inline bool test_name_matches(const std::string& name) {
    return g_test_filter.empty() || name.find(g_test_filter) != std::string::npos;
}

#define ASSERT_TRUE(condition)                                                                                         \
    if (!(condition)) {                                                                                                \
        std::cerr << RED << "[FAILED] " << #condition << " at " << __FILE__ << ":" << __LINE__ << RESET << std::endl;  \
        g_tests_failed++;                                                                                              \
        return;                                                                                                        \
    }

#define ASSERT_EQ(val1, val2)                                                                                          \
    if ((val1) != (val2)) {                                                                                            \
        std::cerr << RED << "[FAILED] Expected " << val1 << " == " << val2 << " at " << __FILE__ << ":" << __LINE__    \
                  << RESET << std::endl;                                                                               \
        g_tests_failed++;                                                                                              \
        return;                                                                                                        \
    }

// A helper to run test functions and print their status
inline void run_test(const std::string& name, std::function<void()> test_func) {
    if (!test_name_matches(name))
        return;
    int failures_before = g_tests_failed;
    std::cout << "[RUNNING] " << name << "..." << std::endl;
    test_func();
    if (g_tests_failed == failures_before) {
        std::cout << GREEN << "[PASSED] " << name << RESET << std::endl;
        g_tests_passed++;
    }
}

// Parses the CHORUS_SKIP_MODEL_TESTS env value. Model tests are disabled only when the
// value is exactly "1"; unset (nullptr), "0", or empty leaves them enabled. The gate is
// the caller's explicit choice -- we never probe the filesystem for the model.
inline bool model_tests_disabled_by_env(const char* env_value) {
    return env_value != nullptr && std::string(env_value) == "1";
}

// Skips (returns from) the current void test with a reported [SKIP] line when model tests
// are disabled by the user. When ENABLED but the model is missing, the test runs and its
// init assert fails loudly -- absence is never silently swallowed.
#define SKIP_IF_MODEL_TESTS_DISABLED()                                                                                 \
    if (!g_run_model_tests) {                                                                                          \
        std::cout << YELLOW << "[SKIP] model tests disabled (CHORUS_SKIP_MODEL_TESTS=1)" << RESET << std::endl;        \
        return;                                                                                                        \
    }

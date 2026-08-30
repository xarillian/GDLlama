#include "chorus/core/inference_engine.hpp"
#include "sync_mock_engine.hpp"
#include "test_utils.hpp"
#include <optional>

void test_initialize_returns_nullopt_on_success() {
    SyncMockEngine engine;
    Chorus::ChorusConfig config;

    ASSERT_TRUE(!engine.initialize(config, {}).has_value());
    ASSERT_TRUE(engine.is_initialized());

    engine.shutdown();
    ASSERT_TRUE(!engine.is_initialized());
}

void test_initialize_propagates_configured_error_and_leaves_engine_uninitialized() {
    SyncMockEngine engine;
    Chorus::ChorusConfig config;
    engine.fail_initialize_with = Chorus::ChorusError::InvalidRequest;

    auto err = engine.initialize(config, {});
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(err.value() == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(!engine.is_initialized());
}

void test_request_submission() {
    SyncMockEngine engine;
    Chorus::ChorusConfig config;
    engine.initialize(config, {});
    engine.tokens = {"Test", "Token"};

    bool completed = false;
    std::string content = "";

    Chorus::ChorusRequest req;
    req.id = 12345;
    req.prompt = "Hello World";

    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event)) {
            content += std::get<Chorus::ChorusSignal::Token>(sig.event).text;
        } else if (std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event)) {
            completed = true;
        }
    };

    engine.submit_request(req);

    ASSERT_TRUE(completed);
    ASSERT_EQ(content, "TestToken");
    ASSERT_EQ(engine.submitted_ids.size(), 1);
    ASSERT_EQ(engine.submitted_ids[0], 12345);
}

// ---------------------------------------------------------------------------
// Error code tests
// ---------------------------------------------------------------------------

void test_submit_to_uninitialized_engine_signals_EngineNotReady_error_code() {
    SyncMockEngine engine;
    engine.fail_submit_with = Chorus::ChorusError::EngineNotReady;

    Chorus::ChorusError received_code = Chorus::ChorusError::None;
    std::string received_text;

    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = "Hello";
    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)) {
            received_code = std::get<Chorus::ChorusSignal::Error>(sig.event).code;
            received_text = std::get<Chorus::ChorusSignal::Error>(sig.event).message;
        }
    };

    engine.submit_request(req);

    ASSERT_TRUE(received_code == Chorus::ChorusError::EngineNotReady);
    ASSERT_TRUE(!received_text.empty());
}

void test_submit_in_fail_mode_propagates_chosen_error_code_to_caller() {
    SyncMockEngine engine;
    Chorus::ChorusConfig config;
    engine.initialize(config, {});
    engine.fail_submit_with = Chorus::ChorusError::Decode;

    Chorus::ChorusError received = Chorus::ChorusError::None;

    Chorus::ChorusRequest req;
    req.id = 7;
    req.prompt = "Hello";
    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event))
            received = std::get<Chorus::ChorusSignal::Error>(sig.event).code;
    };

    engine.submit_request(req);

    ASSERT_TRUE(received == Chorus::ChorusError::Decode);
}

void test_submit_in_fail_mode_propagates_Tokenize_error_code() {
    SyncMockEngine engine;
    Chorus::ChorusConfig config;
    engine.initialize(config, {});
    engine.fail_submit_with = Chorus::ChorusError::Tokenize;

    Chorus::ChorusError received = Chorus::ChorusError::None;
    bool got_error_event = false;

    Chorus::ChorusRequest req;
    req.id = 9;
    req.prompt = "Hello";
    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)) {
            received = std::get<Chorus::ChorusSignal::Error>(sig.event).code;
            got_error_event = true;
        }
    };

    engine.submit_request(req);

    ASSERT_TRUE(got_error_event);
    ASSERT_TRUE(received == Chorus::ChorusError::Tokenize);
}

void test_model_tests_disabled_by_env_only_for_explicit_1() {
    ASSERT_TRUE(!model_tests_disabled_by_env(nullptr)); // unset  -> enabled
    ASSERT_TRUE(!model_tests_disabled_by_env("0"));     // "0"    -> enabled
    ASSERT_TRUE(!model_tests_disabled_by_env(""));      // empty  -> enabled
    ASSERT_TRUE(model_tests_disabled_by_env("1"));      // "1"    -> disabled
}

void test_name_filter_matches_substring_and_empty_runs_all() {
    g_test_filter = "";
    ASSERT_TRUE(test_name_matches("AnythingAtAll"));

    g_test_filter = "Decode";
    ASSERT_TRUE(test_name_matches("Transient_Decode_failure"));
    ASSERT_TRUE(!test_name_matches("Priority_ordering"));

    g_test_filter = ""; // restore so later tests are unaffected
}

// ---------------------------------------------------------------------------
// Suite entry point
// ---------------------------------------------------------------------------

int run_core_mechanics_tests() {
    std::cout << "\n--- CORE MECHANICS TEST SUITE ---\n";

    run_test("Initialize_returns_nullopt_on_success", test_initialize_returns_nullopt_on_success);
    run_test(
        "Initialize_propagates_configured_error_and_leaves_engine_uninitialized",
        test_initialize_propagates_configured_error_and_leaves_engine_uninitialized
    );
    run_test("Request_Lifecycle_Submission", test_request_submission);

    run_test(
        "Submit_to_uninitialized_engine_signals_EngineNotReady_error_code",
        test_submit_to_uninitialized_engine_signals_EngineNotReady_error_code
    );
    run_test(
        "Submit_in_fail_mode_propagates_chosen_error_code_to_caller",
        test_submit_in_fail_mode_propagates_chosen_error_code_to_caller
    );
    run_test(
        "Submit_in_fail_mode_propagates_Tokenize_error_code", test_submit_in_fail_mode_propagates_Tokenize_error_code
    );

    run_test("Model_tests_disabled_by_env_only_for_explicit_1", test_model_tests_disabled_by_env_only_for_explicit_1);
    run_test("Name_filter_matches_substring_and_empty_runs_all", test_name_filter_matches_substring_and_empty_runs_all);

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << RED << "SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED." << RESET
                  << "\n";
        return 1;
    } else {
        std::cout << GREEN << "SUMMARY: ALL TESTS PASSED." << RESET << "\n";
        return 0;
    }
}

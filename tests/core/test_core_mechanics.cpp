#include "chorus/core/inference_engine.hpp"
#include "chorus/core/log.hpp"
#include "sync_mock_engine.hpp"
#include "test_utils.hpp"
#include <optional>

void test_initialize_returns_nullopt_on_success() {
    SyncMockEngine engine;
    Chorus::ChorusConfig config;

    ASSERT_TRUE(!engine.initialize(config).has_value());
    ASSERT_TRUE(engine.is_initialized());

    engine.stop();
    ASSERT_TRUE(!engine.is_initialized());
}

void test_initialize_propagates_configured_error_and_leaves_engine_uninitialized() {
    SyncMockEngine engine;
    Chorus::ChorusConfig config;
    engine.fail_initialize_with = Chorus::ChorusError::InvalidRequest;

    auto err = engine.initialize(config);
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(err.value() == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(!engine.is_initialized());
}

void test_request_submission() {
    SyncMockEngine engine;
    Chorus::ChorusConfig config;
    engine.initialize(config);
    engine.tokens = {"Test", "Token"};

    bool completed = false;
    std::string content = "";

    Chorus::ChorusRequest req;
    req.id = 12345;
    req.prompt = "Hello World";

    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (sig.type == Chorus::EventType::Token) {
            content += sig.text;
        } else if (sig.type == Chorus::EventType::Stop) {
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
        if (sig.is_error()) {
            received_code = sig.error_code;
            received_text = sig.text;
        }
    };

    engine.submit_request(req);

    ASSERT_TRUE(received_code == Chorus::ChorusError::EngineNotReady);
    ASSERT_TRUE(!received_text.empty());
}

void test_new_ChorusSignal_has_error_code_None_by_default() {
    Chorus::ChorusSignal sig;
    sig.request_id = 1;
    sig.type = Chorus::EventType::Token;
    sig.text = "hello";

    ASSERT_TRUE(sig.error_code == Chorus::ChorusError::None);
    ASSERT_TRUE(!sig.is_error());
}

void test_ChorusSignal_preserves_error_code_and_request_id_after_assignment() {
    Chorus::ChorusSignal sig;
    sig.request_id = 42;
    sig.type = Chorus::EventType::Error;
    sig.error_code = Chorus::ChorusError::Decode;
    sig.text = "decode failed";

    ASSERT_TRUE(sig.is_error());
    ASSERT_TRUE(sig.error_code == Chorus::ChorusError::Decode);
    ASSERT_EQ(sig.request_id, 42);
}

// ---------------------------------------------------------------------------
// Logging tests
// ---------------------------------------------------------------------------

void test_chorus_log_forwards_level_and_message_to_callback() {
    Chorus::LogLevel received_level = Chorus::LogLevel::Debug;
    std::string received_msg;

    Chorus::LogCallback cb = [&](Chorus::LogLevel level, const std::string& msg) {
        received_level = level;
        received_msg = msg;
    };

    Chorus::chorus_log(cb, Chorus::LogLevel::Error, "something broke");

    ASSERT_TRUE(received_level == Chorus::LogLevel::Error);
    ASSERT_EQ(received_msg, "something broke");
}

void test_chorus_log_with_null_callback_falls_back_to_stderr_without_crashing() {
    Chorus::LogCallback empty_cb;
    Chorus::chorus_log(empty_cb, Chorus::LogLevel::Warn, "this should go to stderr");
    ASSERT_TRUE(true);
}

void test_LogCallback_on_ChorusConfig_is_invoked_by_chorus_log() {
    std::vector<std::pair<Chorus::LogLevel, std::string>> log_entries;

    Chorus::ChorusConfig config;
    config.log_callback = [&](Chorus::LogLevel level, const std::string& msg) { log_entries.push_back({level, msg}); };

    Chorus::LogCallback log = config.log_callback;
    Chorus::chorus_log(log, Chorus::LogLevel::Info, "engine starting");
    Chorus::chorus_log(log, Chorus::LogLevel::Error, "something failed");

    ASSERT_EQ(log_entries.size(), 2);
    ASSERT_TRUE(log_entries[0].first == Chorus::LogLevel::Info);
    ASSERT_TRUE(log_entries[1].first == Chorus::LogLevel::Error);
    ASSERT_EQ(log_entries[0].second, "engine starting");
    ASSERT_EQ(log_entries[1].second, "something failed");
}

void test_submit_in_fail_mode_propagates_chosen_error_code_to_caller() {
    SyncMockEngine engine;
    Chorus::ChorusConfig config;
    engine.initialize(config);
    engine.fail_submit_with = Chorus::ChorusError::Decode;

    Chorus::ChorusError received = Chorus::ChorusError::None;

    Chorus::ChorusRequest req;
    req.id = 7;
    req.prompt = "Hello";
    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (sig.is_error())
            received = sig.error_code;
    };

    engine.submit_request(req);

    ASSERT_TRUE(received == Chorus::ChorusError::Decode);
}

void test_submit_in_fail_mode_propagates_Tokenize_error_code() {
    SyncMockEngine engine;
    Chorus::ChorusConfig config;
    engine.initialize(config);
    engine.fail_submit_with = Chorus::ChorusError::Tokenize;

    Chorus::ChorusError received = Chorus::ChorusError::None;
    bool got_error_event = false;

    Chorus::ChorusRequest req;
    req.id = 9;
    req.prompt = "Hello";
    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (sig.is_error()) {
            received = sig.error_code;
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
    run_test("New_ChorusSignal_has_error_code_None_by_default", test_new_ChorusSignal_has_error_code_None_by_default);
    run_test(
        "ChorusSignal_preserves_error_code_and_request_id_after_assignment",
        test_ChorusSignal_preserves_error_code_and_request_id_after_assignment
    );
    run_test(
        "Submit_in_fail_mode_propagates_chosen_error_code_to_caller",
        test_submit_in_fail_mode_propagates_chosen_error_code_to_caller
    );
    run_test(
        "Submit_in_fail_mode_propagates_Tokenize_error_code", test_submit_in_fail_mode_propagates_Tokenize_error_code
    );

    run_test(
        "chorus_log_forwards_level_and_message_to_callback", test_chorus_log_forwards_level_and_message_to_callback
    );
    run_test(
        "chorus_log_with_null_callback_falls_back_to_stderr_without_crashing",
        test_chorus_log_with_null_callback_falls_back_to_stderr_without_crashing
    );
    run_test(
        "LogCallback_on_ChorusConfig_is_invoked_by_chorus_log",
        test_LogCallback_on_ChorusConfig_is_invoked_by_chorus_log
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

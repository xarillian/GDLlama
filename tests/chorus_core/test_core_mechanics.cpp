#include "../include/chorus_core/chorus_log.hpp"
#include "../include/chorus_core/inference_engine.hpp"
#include "../test_utils.hpp"
#include <thread>

// A "Dummy" LLM backend -- thank you Gemini!
class MockInferenceEngine : public Chorus::InferenceEngine {
  public:
    bool initialized = false;
    std::vector<int64_t> received_ids;

    bool initialize(const Chorus::ChorusConfig& config) override {
        if (config.model_path.empty())
            return false;
        initialized = true;
        return true;
    }

    void submit_request(const Chorus::ChorusRequest& req) override {
        // 1. Check Initialization
        if (!initialized) {
            if (req.on_event) {
                Chorus::ChorusSignal sig;
                sig.request_id = req.id;
                sig.type = Chorus::EventType::Error;
                sig.error_code = Chorus::ChorusError::EngineNotReady;
                sig.text = "Engine not initialized";
                req.on_event(sig);
            }
            return;
        }

        received_ids.push_back(req.id);

        // 2. Fire Async Events
        std::thread([req]() {
            // Emulate Token 1
            if (req.on_event) {
                Chorus::ChorusSignal sig;
                sig.request_id = req.id;
                sig.type = Chorus::EventType::Token;
                sig.text = "Test";
                req.on_event(sig);
            }

            // Emulate Token 2
            if (req.on_event) {
                Chorus::ChorusSignal sig;
                sig.request_id = req.id;
                sig.type = Chorus::EventType::Token;
                sig.text = "Token";
                req.on_event(sig);
            }

            // Emulate Stop
            if (req.on_event) {
                Chorus::ChorusSignal sig;
                sig.request_id = req.id;
                sig.type = Chorus::EventType::Stop;
                req.on_event(sig);
            }
        }).detach();
    }

    void stop() override { initialized = false; }
    bool is_initialized() const override { return initialized; }
};

void test_initialization() {
    // Happy Path
    MockInferenceEngine engine;
    Chorus::ChorusConfig config;

    config.model_path = "mock_model.bin";
    ASSERT_TRUE(engine.initialize(config) == true);
    ASSERT_TRUE(engine.is_initialized() == true);

    engine.stop();
    ASSERT_TRUE(engine.is_initialized() == false);
}

void test_initialization_failure() {
    MockInferenceEngine engine;
    Chorus::ChorusConfig config;
    config.model_path = ""; // Empty path should fail

    ASSERT_TRUE(engine.initialize(config) == false);
    ASSERT_TRUE(engine.is_initialized() == false);
}

void test_request_submission() {
    MockInferenceEngine engine;
    Chorus::ChorusConfig config;
    config.model_path = "mock.bin";
    engine.initialize(config);

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

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(completed);
    ASSERT_EQ(content, "TestToken");
    ASSERT_EQ(engine.received_ids.size(), 1);
    ASSERT_EQ(engine.received_ids[0], 12345);
}

// ---------------------------------------------------------------------------
// Error code tests
// ---------------------------------------------------------------------------

void test_submit_to_uninitialized_engine_signals_EngineNotReady_error_code() {
    MockInferenceEngine engine;

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
    config.model_path = "mock.bin";
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

// ---------------------------------------------------------------------------
// Suite entry point
// ---------------------------------------------------------------------------

int run_core_mechanics_tests() {
    std::cout << "\n--- CORE MECHANICS TEST SUITE ---\n";

    run_test("Initialization_HappyPath", test_initialization);
    run_test("Initialization_FailurePath", test_initialization_failure);
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

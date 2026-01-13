#include "../include/chorus_core/inference_engine.hpp"
#include "../test_utils.hpp"

// A "Dummy" LLM backend -- thank you Gemini!
class MockInferenceEngine : public Chorus::InferenceEngine {
public:
    bool initialized = false;
    std::vector<int64_t> received_ids;

    bool initialize(const Chorus::ChorusConfig& config) override {
        if (config.model_path.empty()) return false;
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

int run_core_mechanics_tests() {
    std::cout << "\n--- CORE MECHANICS TEST SUITE ---\n"; 

    run_test("Initialization_HappyPath", test_initialization);
    run_test("Initialization_FailurePath", test_initialization_failure);
    run_test("Request_Lifecycle_Submission", test_request_submission);

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << RED << "SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED." << RESET << "\n";
        return 1;
    } else {
        std::cout << GREEN << "SUMMARY: ALL TESTS PASSED." << RESET << "\n";
        return 0;
    }
}

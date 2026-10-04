#include "engine_contract_suite.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using Chorus::ChorusRequest;
using Chorus::ChorusSignal;
using Chorus::RequestId;

// Generous enough for a cold model on CPU; every wait is predicate-driven, so
// a healthy engine never spends it.
constexpr auto PATIENCE = std::chrono::seconds(20);

/*
 * Thread-safe signal collector.
 *
 * Callbacks may arrive on an engine worker, so every read and write of the
 * record goes through the lock, and waiters block on a predicate rather than
 * a sleep.
 */
class SignalLog {
  public:
    void record(const ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(_mutex);
        _signals.push_back(signal);
        _cv.notify_all();
    }

    template <typename Ready> bool wait_until(Ready ready) {
        std::unique_lock<std::mutex> lock(_mutex);
        return _cv.wait_for(lock, PATIENCE, [&] { return ready(_signals); });
    }

    std::vector<ChorusSignal> snapshot() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _signals;
    }

    size_t size() const { return snapshot().size(); }

    size_t terminals_for(RequestId id) const {
        size_t count = 0;
        for (const auto& signal : snapshot())
            count += static_cast<size_t>(signal.request_id == id && signal.is_terminal());
        return count;
    }

    std::optional<ChorusSignal> terminal_for(RequestId id) const {
        for (const auto& signal : snapshot())
            if (signal.request_id == id && signal.is_terminal())
                return signal;
        return std::nullopt;
    }

  private:
    mutable std::mutex _mutex;
    std::condition_variable _cv;
    std::vector<ChorusSignal> _signals;
};

/*
 * Holds an engine still by blocking inside its callback.
 *
 * Providers differ in how fast they stream, so the suite never races a clock:
 * the first Token parks the emitting thread until the test opens the gate,
 * which keeps the request in flight for exactly as long as a case needs.
 */
class TokenGate {
  public:
    void block_here() {
        std::unique_lock<std::mutex> lock(_mutex);
        if (_entered)
            return; // one parked thread is enough; later tokens pass through
        _entered = true;
        _cv.notify_all();
        _cv.wait(lock, [&] { return _open; });
    }

    bool wait_until_entered() {
        std::unique_lock<std::mutex> lock(_mutex);
        return _cv.wait_for(lock, PATIENCE, [&] { return _entered; });
    }

    void open() {
        std::lock_guard<std::mutex> lock(_mutex);
        _open = true;
        _cv.notify_all();
    }

  private:
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _entered = false;
    bool _open = false;
};

std::unique_ptr<Chorus::InferenceEngine> start_engine(const EngineUnderTest& subject) {
    auto engine = subject.make_engine();
    if (engine->initialize(subject.make_config(), {}, {}).has_value())
        return nullptr;
    return engine;
}

ChorusRequest long_request(const EngineUnderTest& subject, RequestId id) {
    ChorusRequest request;
    request.id = id;
    subject.shape_long_request(request);
    return request;
}

ChorusRequest short_request(const EngineUnderTest& subject, RequestId id) {
    ChorusRequest request;
    request.id = id;
    subject.shape_short_request(request);
    return request;
}

// --- cases ---

void case_pre_cancelled_initialization_can_recover(const EngineUnderTest& subject) {
    auto engine = subject.make_engine();
    std::stop_source stop;
    stop.request_stop();
    const auto failure = engine->initialize(subject.make_config(), {}, {stop.get_token(), {}});
    ASSERT_TRUE(failure.has_value());
    EXPECT_EQ(failure->error, Chorus::ChorusError::Cancelled);
    EXPECT_FALSE(engine->is_initialized());
    engine->shutdown();
    ASSERT_FALSE(engine->initialize(subject.make_config(), {}, {}).has_value());
    ASSERT_TRUE(engine->is_initialized());
    engine->shutdown();
}

/*
 * An engine that will not run the work says so, rather than dropping it.
 *
 * Refusal is a terminal like any other, which is what keeps the caller's
 * exactly-one-terminal accounting true for a request that never ran.
 */
void case_submit_before_initialize_is_refused(const EngineUnderTest& subject) {
    auto engine = subject.make_engine();
    SignalLog log;

    ChorusRequest request = short_request(subject, 1);
    request.on_event = [&log](ChorusSignal& signal) { log.record(signal); };
    engine->submit_request(request);

    // The refusal is synchronous: no wait, and none owed.
    ASSERT_EQ(log.size(), size_t{1});
    ASSERT_EQ(log.terminals_for(1), size_t{1});
    const auto terminal = log.terminal_for(1);
    ASSERT_TRUE(terminal.has_value());
    const auto* error = std::get_if<ChorusSignal::Error>(&terminal->event);
    ASSERT_TRUE(error != nullptr);
    if (error)
        ASSERT_TRUE(error->code == Chorus::ChorusError::EngineNotReady);
}

void case_submit_after_shutdown_is_refused(const EngineUnderTest& subject) {
    auto engine = start_engine(subject);
    ASSERT_TRUE(engine != nullptr);
    engine->shutdown();
    ASSERT_TRUE(!engine->is_initialized());

    SignalLog log;
    ChorusRequest request = short_request(subject, 2);
    request.on_event = [&log](ChorusSignal& signal) { log.record(signal); };
    engine->submit_request(request);

    ASSERT_EQ(log.size(), size_t{1});
    ASSERT_EQ(log.terminals_for(2), size_t{1});
    const auto terminal = log.terminal_for(2);
    ASSERT_TRUE(terminal.has_value());
    const auto* error = std::get_if<ChorusSignal::Error>(&terminal->event);
    ASSERT_TRUE(error != nullptr);
    if (error)
        ASSERT_TRUE(error->code == Chorus::ChorusError::EngineNotReady);
}

/// A completed generation ends on exactly one `Chorus::ChorusSignal::Completion`.
void case_completed_request_reaches_one_terminal(const EngineUnderTest& subject) {
    auto engine = start_engine(subject);
    ASSERT_TRUE(engine != nullptr);

    SignalLog log;
    ChorusRequest request = short_request(subject, 3);
    request.on_event = [&log](ChorusSignal& signal) { log.record(signal); };
    engine->submit_request(request);

    const bool finished = log.wait_until([](const std::vector<ChorusSignal>& signals) {
        return !signals.empty() && signals.back().is_terminal();
    });
    engine->shutdown();

    ASSERT_TRUE(finished);
    ASSERT_EQ(log.terminals_for(3), size_t{1});
    const auto terminal = log.terminal_for(3);
    ASSERT_TRUE(terminal.has_value());
    ASSERT_TRUE(std::holds_alternative<ChorusSignal::Completion>(terminal->event));
    const auto& usage = std::get<ChorusSignal::Completion>(terminal->event).usage;
    EXPECT_GE(usage.prompt_tokens, 0);
    EXPECT_GE(usage.cached_prompt_tokens, 0);
    EXPECT_LE(usage.cached_prompt_tokens, usage.prompt_tokens);
    EXPECT_GT(usage.generated_tokens, 0);
    const auto signals = log.snapshot();
    ASSERT_TRUE(std::holds_alternative<ChorusSignal::Completion>(signals.back().event));
    for (size_t index = 0; index + 1 < signals.size(); ++index)
        EXPECT_TRUE(std::holds_alternative<ChorusSignal::Token>(signals[index].event));
}

void case_zero_cap_completes_with_zero_usage_and_no_tokens(const EngineUnderTest& subject) {
    SignalLog log;
    auto engine = start_engine(subject);
    ASSERT_NE(engine, nullptr);
    auto request = short_request(subject, 8);
    request.gen_config.max_tokens = 0;
    request.on_event = [&log](ChorusSignal& signal) { log.record(signal); };
    engine->submit_request(request);
    const bool finished =
        log.wait_until([](const auto& signals) { return !signals.empty() && signals.back().is_terminal(); });
    engine->shutdown();

    ASSERT_TRUE(finished);
    const auto signals = log.snapshot();
    ASSERT_EQ(signals.size(), size_t{1});
    ASSERT_EQ(signals.front().request_id, request.id);
    const auto* completion = std::get_if<ChorusSignal::Completion>(&signals.front().event);
    ASSERT_NE(completion, nullptr);
    EXPECT_EQ(completion->usage.prompt_tokens, 0);
    EXPECT_EQ(completion->usage.cached_prompt_tokens, 0);
    EXPECT_EQ(completion->usage.generated_tokens, 0);
}

void case_embedding_is_one_vector_terminal(const EngineUnderTest& subject) {
    SignalLog log;
    auto engine = start_engine(subject);
    ASSERT_NE(engine, nullptr);
    if (!engine->capabilities().embeddings)
        GTEST_SKIP() << subject.label << " loaded without embedding support";

    ChorusRequest request;
    request.id = 9;
    request.type = Chorus::RequestType::Embedding;
    request.prompt = "A cat sleeps on the warm mat.";
    request.on_event = [&log](ChorusSignal& signal) { log.record(signal); };
    ASSERT_FALSE(engine->validate_request(request));
    engine->submit_request(request);
    const bool finished =
        log.wait_until([](const auto& signals) { return !signals.empty() && signals.back().is_terminal(); });
    engine->shutdown();

    ASSERT_TRUE(finished);
    const auto signals = log.snapshot();
    ASSERT_EQ(signals.size(), size_t{1});
    EXPECT_EQ(signals.front().request_id, request.id);
    const auto* embedding = std::get_if<ChorusSignal::Embedding>(&signals.front().event);
    ASSERT_NE(embedding, nullptr);
    ASSERT_FALSE(embedding->values.empty());
    double norm = 0.0;
    for (float value : embedding->values) {
        ASSERT_TRUE(std::isfinite(value));
        norm += static_cast<double>(value) * value;
    }
    EXPECT_NEAR(norm, 1.0, 1e-5);
}

/*
 * Cancelling twice, and cancelling a stranger, spends one terminal.
 *
 * The unknown id shares the case so its inertness is proved against a live
 * engine, where a provider that mishandled it would disturb the neighbor.
 * Providers that declare no cancellation are out of scope here; `stop` still
 * owes them its own case.
 */
void case_cancel_is_idempotent_and_terminal_once(const EngineUnderTest& subject) {
    auto engine = start_engine(subject);
    ASSERT_TRUE(engine != nullptr);

    if (!engine->capabilities().cancellation) {
        engine->shutdown();
        GTEST_SKIP() << subject.label << " declares no cancellation";
    }

    SignalLog log;
    TokenGate gate;
    ChorusRequest request = long_request(subject, 4);
    request.on_event = [&log, &gate](ChorusSignal& signal) {
        log.record(signal);
        if (std::holds_alternative<ChorusSignal::Token>(signal.event))
            gate.block_here();
    };
    engine->submit_request(request);

    if (!gate.wait_until_entered()) {
        gate.open();
        engine->shutdown();
        ASSERT_TRUE(false); // never streamed: the subject's long shape is wrong
    }

    engine->cancel_request(4);
    engine->cancel_request(4);
    engine->cancel_request(9999); // never submitted
    gate.open();

    const bool finished = log.wait_until([](const std::vector<ChorusSignal>& signals) {
        return !signals.empty() && signals.back().is_terminal();
    });
    engine->shutdown();

    ASSERT_TRUE(finished);
    ASSERT_EQ(log.terminals_for(4), size_t{1});
    const auto terminal = log.terminal_for(4);
    ASSERT_TRUE(terminal.has_value());
    const auto* error = std::get_if<ChorusSignal::Error>(&terminal->event);
    ASSERT_TRUE(error != nullptr);
    if (error)
        ASSERT_TRUE(error->code == Chorus::ChorusError::Cancelled);
    ASSERT_EQ(log.terminals_for(9999), size_t{0}); // the stranger got nothing
    const auto signals = log.snapshot();
    ASSERT_TRUE(std::holds_alternative<ChorusSignal::Error>(signals.back().event));
    for (size_t index = 0; index + 1 < signals.size(); ++index)
        EXPECT_TRUE(std::holds_alternative<ChorusSignal::Token>(signals[index].event));
}

/*
 * Stop closes the books, then bars the door.
 *
 * Work the engine still holds is terminated before stop returns, and nothing
 * the engine was handed is called afterwards; the second half is what lets a
 * caller destroy the objects those callbacks write into.
 */
void case_shutdown_terminates_in_flight_work_and_fences_callbacks(const EngineUnderTest& subject) {
    auto engine = start_engine(subject);
    ASSERT_TRUE(engine != nullptr);

    SignalLog log;
    TokenGate gate;
    std::atomic<bool> stop_returned{false};
    std::atomic<int> signals_after_stop{0};
    std::mutex stopper_mutex;
    std::condition_variable stopper_cv;
    bool stopper_entered = false;

    auto on_event = [&](ChorusSignal& signal) {
        if (stop_returned.load())
            ++signals_after_stop;
        log.record(signal);
        if (std::holds_alternative<ChorusSignal::Token>(signal.event))
            gate.block_here();
    };

    ChorusRequest held = long_request(subject, 5);
    held.on_event = on_event;
    engine->submit_request(held);

    if (!gate.wait_until_entered()) {
        gate.open();
        engine->shutdown();
        ASSERT_TRUE(false); // never streamed: the subject's long shape is wrong
    }

    // Submitted behind a parked engine, so it is still unfinished at stop.
    ChorusRequest trailing = short_request(subject, 6);
    trailing.on_event = on_event;
    engine->submit_request(trailing);

    std::thread stopper([&] {
        {
            std::lock_guard<std::mutex> lock(stopper_mutex);
            stopper_entered = true;
            stopper_cv.notify_all();
        }
        engine->shutdown();
        stop_returned.store(true);
    });
    bool stopper_is_waiting = false;
    {
        std::unique_lock<std::mutex> lock(stopper_mutex);
        stopper_is_waiting = stopper_cv.wait_for(lock, PATIENCE, [&] { return stopper_entered; });
    }
    const bool stop_began = wait_until_submissions_rejected(*engine, short_request(subject, 7));
    // The gate must open from outside: stop is entitled to wait on the
    // callback it is fencing, so releasing it after the join would deadlock.
    gate.open();
    stopper.join();

    const size_t settled = log.size();

    ASSERT_TRUE(stopper_is_waiting);
    ASSERT_TRUE(stop_began);
    ASSERT_TRUE(stop_returned.load());
    ASSERT_EQ(signals_after_stop.load(), 0);
    ASSERT_EQ(log.size(), settled); // the door stayed shut
    ASSERT_EQ(log.terminals_for(5), size_t{1});
    ASSERT_EQ(log.terminals_for(6), size_t{1});

    // The parked request could not have finished on its own.
    const auto terminal = log.terminal_for(5);
    ASSERT_TRUE(terminal.has_value());
    const auto* error = std::get_if<ChorusSignal::Error>(&terminal->event);
    ASSERT_TRUE(error != nullptr);
    if (error)
        ASSERT_TRUE(error->code == Chorus::ChorusError::Cancelled);
}

/// Stopping an idle engine, or a stopped one, is safe and silent.
void case_shutdown_is_idempotent(const EngineUnderTest& subject) {
    auto never_started = subject.make_engine();
    never_started->shutdown();
    ASSERT_TRUE(!never_started->is_initialized());

    auto engine = start_engine(subject);
    ASSERT_TRUE(engine != nullptr);

    SignalLog log;
    ChorusRequest request = short_request(subject, 7);
    request.on_event = [&log](ChorusSignal& signal) { log.record(signal); };
    engine->submit_request(request);
    ASSERT_TRUE(log.wait_until([](const std::vector<ChorusSignal>& signals) {
        return !signals.empty() && signals.back().is_terminal();
    }));

    engine->shutdown();
    const size_t after_first = log.size();
    engine->shutdown();
    ASSERT_EQ(log.size(), after_first);
    ASSERT_EQ(log.terminals_for(7), size_t{1});
}

} // namespace

bool wait_until_submissions_rejected(Chorus::InferenceEngine& engine, Chorus::ChorusRequest probe) {
    static std::atomic<Chorus::RequestId> next_probe_id{1'000'000};
    const auto rejected = std::make_shared<std::atomic<bool>>(false);
    probe.on_event = [rejected](Chorus::ChorusSignal& signal) {
        const auto* error = std::get_if<Chorus::ChorusSignal::Error>(&signal.event);
        if (error && error->code == Chorus::ChorusError::EngineNotReady)
            rejected->store(true);
    };
    const auto deadline = std::chrono::steady_clock::now() + PATIENCE;
    while (std::chrono::steady_clock::now() < deadline) {
        probe.id = next_probe_id++;
        engine.submit_request(probe);
        if (rejected->load())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

TEST_P(EngineContractTest, contract_pre_cancelled_initialization_can_recover) {
    case_pre_cancelled_initialization_can_recover(GetParam());
}

TEST_P(EngineContractTest, contract_submit_before_initialize_is_refused) {
    case_submit_before_initialize_is_refused(GetParam());
}

TEST_P(EngineContractTest, contract_submit_after_shutdown_is_refused) {
    case_submit_after_shutdown_is_refused(GetParam());
}

TEST_P(EngineContractTest, contract_completed_request_reaches_one_terminal) {
    case_completed_request_reaches_one_terminal(GetParam());
}

TEST_P(EngineContractTest, contract_zero_cap_completes_with_zero_usage_and_no_tokens) {
    case_zero_cap_completes_with_zero_usage_and_no_tokens(GetParam());
}

TEST_P(EngineContractTest, contract_embedding_is_one_vector_terminal) {
    case_embedding_is_one_vector_terminal(GetParam());
}

TEST_P(EngineContractTest, contract_cancel_is_idempotent_and_terminal_once) {
    case_cancel_is_idempotent_and_terminal_once(GetParam());
}

TEST_P(EngineContractTest, contract_shutdown_terminates_in_flight_work_and_fences_callbacks) {
    case_shutdown_terminates_in_flight_work_and_fences_callbacks(GetParam());
}

TEST_P(EngineContractTest, contract_shutdown_is_idempotent) {
    case_shutdown_is_idempotent(GetParam());
}

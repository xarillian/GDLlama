#include "engine_contract_suite.hpp"

#include "test_utils.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
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

bool is_terminal(const ChorusSignal& signal) {
    return std::holds_alternative<ChorusSignal::Stop>(signal.event) ||
           std::holds_alternative<ChorusSignal::Error>(signal.event);
}

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
            count += static_cast<size_t>(signal.request_id == id && is_terminal(signal));
        return count;
    }

    std::optional<ChorusSignal> terminal_for(RequestId id) const {
        for (const auto& signal : snapshot())
            if (signal.request_id == id && is_terminal(signal))
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

/// Reports a case the subject's declared capabilities put out of scope.
void skip(const std::string& reason) {
    std::cout << YELLOW << "[SKIP] " << reason << RESET << std::endl;
}

std::unique_ptr<Chorus::InferenceEngine> start_engine(const EngineUnderTest& subject) {
    auto engine = subject.make_engine();
    if (engine->initialize(subject.make_config(), {}).has_value())
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

    ASSERT_EQ(log.terminals_for(2), size_t{1});
    const auto terminal = log.terminal_for(2);
    ASSERT_TRUE(terminal.has_value());
    const auto* error = std::get_if<ChorusSignal::Error>(&terminal->event);
    ASSERT_TRUE(error != nullptr);
    if (error)
        ASSERT_TRUE(error->code == Chorus::ChorusError::EngineNotReady);
}

/// A request that runs to completion ends on exactly one Stop.
void case_completed_request_reaches_one_terminal(const EngineUnderTest& subject) {
    auto engine = start_engine(subject);
    ASSERT_TRUE(engine != nullptr);

    SignalLog log;
    ChorusRequest request = short_request(subject, 3);
    request.on_event = [&log](ChorusSignal& signal) { log.record(signal); };
    engine->submit_request(request);

    const bool finished = log.wait_until([](const std::vector<ChorusSignal>& signals) {
        return !signals.empty() && is_terminal(signals.back());
    });
    engine->shutdown();

    ASSERT_TRUE(finished);
    ASSERT_EQ(log.terminals_for(3), size_t{1});
    const auto terminal = log.terminal_for(3);
    ASSERT_TRUE(terminal.has_value());
    ASSERT_TRUE(std::holds_alternative<ChorusSignal::Stop>(terminal->event));
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
        skip(subject.label + " declares no cancellation");
        engine->shutdown();
        return;
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
        return !signals.empty() && is_terminal(signals.back());
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
        engine->shutdown();
        stop_returned.store(true);
    });
    // The gate must open from outside: stop is entitled to wait on the
    // callback it is fencing, so releasing it after the join would deadlock.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    gate.open();
    stopper.join();

    const size_t settled = log.size();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

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
        return !signals.empty() && is_terminal(signals.back());
    }));

    engine->shutdown();
    const size_t after_first = log.size();
    engine->shutdown();
    ASSERT_EQ(log.size(), after_first);
    ASSERT_EQ(log.terminals_for(7), size_t{1});
}

} // namespace

void run_engine_contract_suite(const EngineUnderTest& subject) {
    std::cout << "\n--- ENGINE CONTRACT SUITE: " << subject.label << " ---\n";

    const auto run_case = [&subject](const std::string& name, void (*body)(const EngineUnderTest&)) {
        run_test(subject.label + "_contract_" + name, [&subject, body] {
            if (subject.model_gated) {
                SKIP_IF_MODEL_TESTS_DISABLED();
            }
            body(subject);
        });
    };

    run_case("submit_before_initialize_is_refused", case_submit_before_initialize_is_refused);
    run_case("submit_after_shutdown_is_refused", case_submit_after_shutdown_is_refused);
    run_case("completed_request_reaches_one_terminal", case_completed_request_reaches_one_terminal);
    run_case("cancel_is_idempotent_and_terminal_once", case_cancel_is_idempotent_and_terminal_once);
    run_case(
        "shutdown_terminates_in_flight_work_and_fences_callbacks",
        case_shutdown_terminates_in_flight_work_and_fences_callbacks
    );
    run_case("shutdown_is_idempotent", case_shutdown_is_idempotent);
}

#pragma once

#include "chorus/runtime/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <map>

namespace Chorus {

inline constexpr size_t kPreparationCapacity = 256;
inline constexpr size_t kMessageCountCacheEntries = 4096;
inline constexpr size_t kContentCountCacheEntries = 4096;
inline constexpr size_t kContentCountCacheBytes = 4 * 1024 * 1024;

/*
 * How many threads prepare requests for one loaded runtime.
 *
 * Rendering and tokenizing a long history is CPU work, so a burst of turns queued behind one
 * worker waits for every render before it. A few workers remove most of that wait without
 * crowding the host's own threads and the provider's inference threads.
 */
inline size_t preparation_worker_count() {
    return std::clamp<size_t>(std::thread::hardware_concurrency() / 4, 1, 4);
}

/// A sticky wake flag: raised by producers of polled work, cleared when poll begins.
struct ChorusRuntime::Wakeup {
    std::mutex mutex;
    std::condition_variable cv;
    bool raised = false;

    void raise() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            raised = true;
        }
        cv.notify_all();
    }

    bool wait_for(std::chrono::nanoseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [this] { return raised; });
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        raised = false;
    }
};

struct ChorusRuntime::Control {
    RequestId id = -1;
    std::atomic<bool> cancelled{false};
    bool terminal = false;
    bool provider_active = false;
    bool preparation_finished = false;
    bool preparation_drained = false;
    bool preparation_outstanding = true;
};

struct ChorusRuntime::PreparationJob {
    Operation operation = Operation::Generate;
    ResolvedRequest resolved;
    ChorusRequest request;
    HistorySnapshot history;
    MessageNodePtr pending;
    MessageContent content;
    std::shared_ptr<Control> control = std::make_shared<Control>();
    std::vector<MessageId> omitted;
    std::string rendered;
    int64_t token_count = 0;
};

struct ChorusRuntime::PreparationState {
    using Output = std::variant<ChorusSignal, RuntimeEvent, std::unique_ptr<PreparationJob>>;
    explicit PreparationState(std::shared_ptr<Wakeup> wakeup) : wakeup(std::move(wakeup)) {}

    std::shared_ptr<Wakeup> wakeup;
    std::mutex mutex;
    std::condition_variable cv;
    bool closing = true;
    std::atomic<bool> failed{false};
    size_t outstanding = 0;
    std::deque<std::unique_ptr<PreparationJob>> jobs;
    std::vector<Output> output;
    std::unordered_map<RequestId, std::shared_ptr<Control>> controls;
    std::vector<std::thread> workers;
    std::shared_ptr<RequestPreparation> service;

    // Workers finish out of order, but outcomes publish in the order workers took their jobs,
    // which is admission order, so equal-priority requests still reach the engine as submitted.
    struct Outcome {
        std::unique_ptr<PreparationJob> job;
        std::optional<RequestRejection> failure;
        bool abandoned = false;
    };
    uint64_t next_ticket = 0;
    uint64_t next_publication = 0;
    std::map<uint64_t, Outcome> finished;

    EngineCapabilities capabilities;
    std::optional<LoadedModelInfo> model_info;

    // Guards the count caches below, which workers share.
    std::mutex cache_mutex;
    std::list<uint64_t> node_lru;
    struct NodeCount {
        int64_t count;
        std::list<uint64_t>::iterator position;
    };
    std::unordered_map<uint64_t, NodeCount> node_counts;
    struct ContentCount {
        size_t hash;
        std::string text;
        int64_t count;
    };
    std::list<ContentCount> content_counts;
    size_t content_bytes = 0;

    /// Queues output for poll; callers hold `mutex`.
    void publish(Output item) {
        output.push_back(std::move(item));
        wakeup->raise();
    }

    /// Queues a request signal unless the request already ended; callers hold `mutex`.
    void publish_signal(const ChorusSignal& signal, const std::shared_ptr<Control>& control);

    /// Records one prepared job and publishes every outcome now next in admission order; callers hold `mutex`.
    void finish(uint64_t ticket, Outcome outcome);

    int64_t count_text(const std::string& text);
    int64_t count_node(const MessageNodePtr& node);
    void clear_caches();
    void release_preparation(const std::shared_ptr<Control>& control) {
        if (control->preparation_outstanding && control->preparation_finished && control->preparation_drained) {
            control->preparation_outstanding = false;
            --outstanding;
        }
    }
};

struct ChorusRuntime::EngineLifetime {
    explicit EngineLifetime(std::shared_ptr<Wakeup> wakeup)
        : preparation(std::make_shared<PreparationState>(std::move(wakeup))) {}

    std::unique_ptr<InferenceEngine> engine;
    std::shared_ptr<PreparationState> preparation;
};

} // namespace Chorus

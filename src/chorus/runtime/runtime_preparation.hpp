#pragma once

#include "chorus/runtime/runtime.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>

namespace Chorus {

inline constexpr size_t kPreparationCapacity = 256;
inline constexpr size_t kMessageCountCacheEntries = 4096;
inline constexpr size_t kContentCountCacheEntries = 4096;
inline constexpr size_t kContentCountCacheBytes = 4 * 1024 * 1024;

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
    std::thread worker;
    std::shared_ptr<RequestPreparation> service;
    EngineCapabilities capabilities;
    std::optional<LoadedModelInfo> model_info;

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

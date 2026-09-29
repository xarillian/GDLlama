#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace wlib {

/*
 * Runs indexed work on the calling thread and a fixed set of helper threads.
 *
 * Each body call receives its index and a lane in `[0, wlib::ParallelFor::lanes())`.
 * Concurrent calls never share a lane, so callers may keep per-lane scratch storage.
 * Calls to `wlib::ParallelFor::run` must not overlap.
 */
class ParallelFor {
  public:
    using Body = std::function<void(std::size_t index, std::size_t lane)>;

    explicit ParallelFor(std::size_t lanes) {
        try {
            for (std::size_t lane = 1; lane < lanes; ++lane)
                _helpers.emplace_back(&ParallelFor::help, this, lane);
        } catch (...) {
            stop();
            throw;
        }
    }

    ~ParallelFor() { stop(); }

    ParallelFor(const ParallelFor&) = delete;
    ParallelFor& operator=(const ParallelFor&) = delete;

    std::size_t lanes() const noexcept { return _helpers.size() + 1; }

    /*
     * Calls `body` once for every index in `[0, count)` and returns when all calls finish.
     *
     * Raises:
     *  - The first exception thrown by `body`. Unstarted indices are skipped, and every
     *    started call returns before the exception propagates.
     */
    void run(std::size_t count, const Body& body) {
        if (count <= 1 || _helpers.empty()) {
            for (std::size_t index = 0; index < count; ++index)
                body(index, 0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _body = &body;
            _count = count;
            _next = 0;
            _active = _helpers.size();
            ++_generation;
        }
        _wake.notify_all();
        drain(0);
        std::exception_ptr failure;
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _done.wait(lock, [this] { return _active == 0; });
            _body = nullptr;
            failure = std::exchange(_failure, nullptr);
        }
        if (failure)
            std::rethrow_exception(failure);
    }

  private:
    void stop() {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopping = true;
        }
        _wake.notify_all();
        for (auto& helper : _helpers)
            helper.join();
    }

    void help(std::size_t lane) {
        std::uint64_t seen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _wake.wait(lock, [this, seen] { return _stopping || _generation != seen; });
                if (_stopping)
                    return;
                seen = _generation;
            }
            drain(lane);
            std::lock_guard<std::mutex> lock(_mutex);
            if (--_active == 0)
                _done.notify_one();
        }
    }

    void drain(std::size_t lane) {
        for (std::size_t index = _next++; index < _count; index = _next++) {
            try {
                (*_body)(index, lane);
            } catch (...) {
                std::lock_guard<std::mutex> lock(_mutex);
                if (!_failure)
                    _failure = std::current_exception();
                _next = _count;
            }
        }
    }

    std::vector<std::thread> _helpers;
    std::mutex _mutex;
    std::condition_variable _wake;
    std::condition_variable _done;
    const Body* _body = nullptr;
    std::size_t _count = 0;
    std::atomic<std::size_t> _next{0};
    std::size_t _active = 0;
    std::uint64_t _generation = 0;
    bool _stopping = false;
    std::exception_ptr _failure;
};

} // namespace wlib

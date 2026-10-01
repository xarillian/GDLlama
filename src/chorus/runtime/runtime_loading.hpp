#pragma once

#include "chorus/runtime/runtime_preparation.hpp"

#include <condition_variable>
#include <stop_token>

namespace Chorus {

struct ChorusRuntime::LoadAttempt {
    LoadId id = -1;
    ChorusConfig config;
    std::unique_ptr<InferenceEngine> engine;
    std::unique_ptr<EngineLifetime> retiring;
    std::unique_ptr<EngineLifetime> candidate;
    std::stop_source stop;
    std::optional<LoadProgress> progress;
    std::optional<LoadProgress> progress_high_water;
    size_t preparation_workers = preparation_worker_count();
#ifdef CHORUS_HOST_TEST
    bool test_hold_retirement = false;
    bool test_retirement_held = false;
    bool test_fail_preparation_worker_start = false;
#endif
    bool invalid_progress_reported = false;
    bool cancelled = false;
    bool cleanup_requested = false;
    bool parked = false;
    bool committed = false;
    bool finished = false;
    bool success = false;
    InitializationFailure failure{ChorusError::Unknown, "Engine loading failed unexpectedly."};
};

struct ChorusRuntime::LoadingState {
    std::mutex mutex;
    std::condition_variable cv;
    std::thread worker;
    std::shared_ptr<LoadAttempt> attempt;
    bool busy = false;
    bool stopping = false;
};

} // namespace Chorus

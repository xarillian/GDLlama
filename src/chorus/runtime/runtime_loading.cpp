#include "chorus/runtime/runtime_loading.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace Chorus {
namespace {

LoadSubmitResult load_rejection(ChorusError error, std::string message) {
    return {-1, error, std::move(message)};
}

RuntimeEvent load_event(RuntimeEvent::Kind kind, LoadId id, const std::string& model_id) {
    RuntimeEvent event{-1, std::nullopt, kind};
    event.load_id = id;
    event.model_id = model_id;
    return event;
}

} // namespace

LoadSubmitResult ChorusRuntime::load_engine(std::unique_ptr<InferenceEngine> engine, const ChorusConfig& config) {
    assert_host_thread();
    if (!engine)
        return load_rejection(ChorusError::InvalidRequest, "An inference engine is required.");
    if (auto active = active_load_id())
        return load_rejection(ChorusError::InvalidRequest, "Load " + std::to_string(*active) + " is still active.");
    if (_next_load_id == std::numeric_limits<LoadId>::max())
        return load_rejection(ChorusError::InvalidRequest, "Load identity capacity is exhausted.");

    try {
        auto attempt = std::make_shared<LoadAttempt>();
        attempt->config = config;
        attempt->engine = std::move(engine);
        attempt->id = _next_load_id;
#ifdef CHORUS_HOST_TEST
        attempt->test_hold_retirement = _test_hold_next_retirement && _lifetime != nullptr;
        attempt->test_fail_preparation_worker_start = _test_fail_next_preparation_worker_start;
        if (_test_preparation_workers)
            attempt->preparation_workers = _test_preparation_workers;
#endif
        if (_lifetime)
            _draining_states.reserve(_draining_states.size() + 1);
#ifdef CHORUS_HOST_TEST
        if (std::exchange(_test_fail_next_load_setup, false))
            throw std::runtime_error("Test load setup failure.");
#endif
        if (!_loading) {
            auto state = std::make_unique<LoadingState>();
            _loading = std::move(state);
            try {
#ifdef CHORUS_HOST_TEST
                if (std::exchange(_test_fail_next_worker_start, false))
                    throw std::runtime_error("Test lifecycle worker start failure.");
#endif
                _loading->worker = std::thread(&ChorusRuntime::lifecycle_loop, this);
            } catch (...) {
                _loading.reset();
                throw;
            }
        }
        {
            std::lock_guard lock(_loading->mutex);
            if (_lifetime) {
                _draining_states.push_back(_lifetime->preparation);
                attempt->retiring = std::move(_lifetime);
                auto& preparation = *attempt->retiring->preparation;
                {
                    std::lock_guard prep_lock(preparation.mutex);
                    preparation.closing = true;
                    for (const auto& [id, control] : preparation.controls)
                        control->cancelled = true;
                }
                preparation.cv.notify_all();
            }
#ifdef CHORUS_HOST_TEST
            _test_hold_next_retirement = false;
            _test_fail_next_preparation_worker_start = false;
#endif
            _engine_failure_reported = false;
            _loading->attempt = std::move(attempt);
            ++_next_load_id;
        }
        _loading->cv.notify_all();
        return {_next_load_id - 1, ChorusError::None, {}};
    } catch (const std::exception& error) {
        return load_rejection(ChorusError::Unknown, error.what());
    } catch (...) {
        return load_rejection(ChorusError::Unknown, "Unable to start engine loading.");
    }
}

bool ChorusRuntime::cancel_load(LoadId id) {
    assert_host_thread();
    if (!_loading)
        return false;
    std::lock_guard lock(_loading->mutex);
    const auto& attempt = _loading->attempt;
    if (!attempt || attempt->id != id || attempt->committed)
        return false;
    attempt->cancelled = true;
    attempt->stop.request_stop();
    _loading->cv.notify_all();
    return true;
}

std::optional<LoadId> ChorusRuntime::active_load_id() const {
    assert_host_thread();
    if (!_loading)
        return std::nullopt;
    std::lock_guard lock(_loading->mutex);
    return _loading->attempt ? std::optional<LoadId>{_loading->attempt->id} : std::nullopt;
}

#ifdef CHORUS_HOST_TEST
void ChorusRuntime::test_fail_next_load_setup() {
    assert_host_thread();
    _test_fail_next_load_setup = true;
}
void ChorusRuntime::test_fail_next_worker_start() {
    assert_host_thread();
    _test_fail_next_worker_start = true;
}
void ChorusRuntime::test_fail_next_preparation_worker_start() {
    assert_host_thread();
    _test_fail_next_preparation_worker_start = true;
}
void ChorusRuntime::test_use_preparation_workers(size_t count) {
    assert_host_thread();
    _test_preparation_workers = count;
}
bool ChorusRuntime::test_load_parked(LoadId id) const {
    assert_host_thread();
    if (!_loading)
        return false;
    std::lock_guard lock(_loading->mutex);
    const auto& attempt = _loading->attempt;
    return attempt && attempt->id == id && attempt->parked && !attempt->committed;
}
bool ChorusRuntime::test_load_committed(LoadId id) const {
    assert_host_thread();
    if (!_loading)
        return false;
    std::lock_guard lock(_loading->mutex);
    const auto& attempt = _loading->attempt;
    return attempt && attempt->id == id && attempt->committed;
}
void ChorusRuntime::test_exhaust_load_ids() {
    assert_host_thread();
    _next_load_id = std::numeric_limits<LoadId>::max();
}
void ChorusRuntime::test_hold_next_retirement() {
    assert_host_thread();
    _test_hold_next_retirement = true;
}
bool ChorusRuntime::test_retirement_held() const {
    assert_host_thread();
    if (!_loading)
        return false;
    std::lock_guard lock(_loading->mutex);
    return _loading->attempt && _loading->attempt->test_retirement_held;
}
void ChorusRuntime::test_release_retirement() {
    assert_host_thread();
    if (!_loading)
        return;
    {
        std::lock_guard lock(_loading->mutex);
        if (_loading->attempt)
            _loading->attempt->test_hold_retirement = false;
    }
    _loading->cv.notify_all();
}
#endif

void ChorusRuntime::retire_lifetime(EngineLifetime& lifetime) {
    close_preparation(*lifetime.preparation);
    fence_engine(lifetime);
}

void ChorusRuntime::lifecycle_loop() {
    auto& state = *_loading;
    for (;;) {
        std::unique_lock lock(state.mutex);
        state.cv.wait(lock, [&] {
            return state.stopping || (state.attempt && !state.busy && !state.attempt->committed);
        });
        if (state.stopping)
            return;
        auto attempt = state.attempt;
        state.busy = true;
        lock.unlock();

        InitializationFailure failure = std::move(attempt->failure);
        bool usable = false;
        try {
            if (attempt->retiring) {
                {
                    std::lock_guard guard(state.mutex);
                    attempt->progress = LoadProgress{LoadPhase::ReleasingEngine, std::nullopt};
                    attempt->progress_high_water = attempt->progress;
                }
                _wakeup->raise();
#ifdef CHORUS_HOST_TEST
                {
                    std::unique_lock guard(state.mutex);
                    if (attempt->test_hold_retirement) {
                        attempt->test_retirement_held = true;
                        state.cv.notify_all();
                        state.cv.wait(guard, [&] {
                            return !attempt->test_hold_retirement || attempt->stop.stop_requested();
                        });
                        attempt->test_retirement_held = false;
                    }
                }
#endif
                retire_lifetime(*attempt->retiring);
                attempt->retiring.reset();
            }
            if (!attempt->stop.stop_requested()) {
                auto lifetime = std::make_unique<EngineLifetime>(_wakeup);
                lifetime->engine = std::move(attempt->engine);
                attempt->candidate = std::move(lifetime);
                auto weak = std::weak_ptr<LoadAttempt>(attempt);
                auto progress = [this, weak, level = attempt->config.log_level](const LoadProgress& sample) noexcept {
                    try {
                        auto current = weak.lock();
                        if (!current)
                            return;
                        bool invalid = false;
                        {
                            std::lock_guard guard(_loading->mutex);
                            auto previous = current->progress_high_water;
                            invalid =
                                (sample.fraction &&
                                 (!std::isfinite(*sample.fraction) || *sample.fraction < 0 || *sample.fraction > 1)) ||
                                (previous && static_cast<int>(sample.phase) < static_cast<int>(previous->phase)) ||
                                (previous && sample.phase == previous->phase && previous->fraction && sample.fraction &&
                                 *sample.fraction < *previous->fraction);
                            if (!invalid && !current->committed) {
                                auto retained = sample;
                                if (previous && previous->phase == sample.phase && previous->fraction &&
                                    !sample.fraction)
                                    retained.fraction = previous->fraction;
                                current->progress_high_water = retained;
                                current->progress = retained;
                                _wakeup->raise();
                            }
                            if (invalid && !current->invalid_progress_reported)
                                current->invalid_progress_reported = true;
                            else
                                invalid = false;
                        }
                        if (invalid)
                            Logger(_log_channel, level).warn("Invalid load progress was discarded.");
                    } catch (...) {
                    }
                };
                InitializationControl control{attempt->stop.get_token(), progress};
                Logger logger(_log_channel, attempt->config.log_level);
                if (auto error = attempt->candidate->engine->initialize(attempt->config, std::move(logger), control)) {
                    failure = std::move(*error);
                } else if (!attempt->candidate->engine->is_initialized()) {
                    failure = {ChorusError::EngineNotReady, "The provider did not become ready."};
                } else if (auto service = attempt->candidate->engine->request_preparation()) {
                    auto& prep = *attempt->candidate->preparation;
                    prep.service = std::move(service);
                    prep.capabilities = attempt->candidate->engine->capabilities();
                    prep.model_info = attempt->candidate->engine->loaded_model_info();
                    prep.failed = false;
                    prep.closing = false;
#ifdef CHORUS_HOST_TEST
                    if (attempt->test_fail_preparation_worker_start)
                        throw std::runtime_error("Test preparation worker startup failure.");
#endif
                    for (size_t worker = 0; worker < attempt->preparation_workers; ++worker)
                        prep.workers.emplace_back(&ChorusRuntime::preparation_loop, attempt->candidate->preparation);
                    usable = true;
                } else {
                    failure = {ChorusError::EngineNotReady, "The provider has no preparation service."};
                }
            }
        } catch (const std::exception& error) {
            failure.error = ChorusError::Unknown;
            try {
                failure.message = error.what();
            } catch (...) {
            }
        } catch (...) {
            failure.error = ChorusError::Unknown;
        }

        lock.lock();
        if (usable && !attempt->cancelled) {
            attempt->parked = true;
            _wakeup->raise();
            state.cv.wait(lock, [&] {
                return attempt->cancelled || attempt->cleanup_requested || !attempt->candidate;
            });
            if (!attempt->candidate) {
                state.busy = false;
                continue;
            }
            if (attempt->cleanup_requested)
                failure = {ChorusError::EngineNotReady, "The provider failed before publication."};
        }
        attempt->parked = false;
        lock.unlock();
        if (attempt->candidate) {
            retire_lifetime(*attempt->candidate);
            attempt->candidate.reset();
        } else if (attempt->engine) {
            attempt->engine->shutdown();
            attempt->engine.reset();
        }
        lock.lock();
        if (attempt->cancelled)
            failure = {ChorusError::Cancelled, "Engine loading cancelled."};
        attempt->failure = std::move(failure);
        attempt->committed = true;
        attempt->finished = true;
        state.busy = false;
        lock.unlock();
        state.cv.notify_all();
        _wakeup->raise();
    }
}

void ChorusRuntime::append_load_events(std::vector<RuntimeEvent>& events, const std::function<void()>& drain_retiring) {
    if (!_loading)
        return;
    std::unique_lock lock(_loading->mutex);
    auto attempt = _loading->attempt;
    if (!attempt)
        return;
    if (attempt->parked && !attempt->cancelled && !attempt->committed) {
        auto& candidate = *attempt->candidate;
        if (candidate.preparation->failed || !candidate.engine->is_initialized()) {
            attempt->cleanup_requested = true;
            _loading->cv.notify_all();
        } else {
            _lifetime = std::move(attempt->candidate);
            _engine_failure_reported = false;
            attempt->success = true;
            attempt->committed = true;
            attempt->finished = true;
            _loading->cv.notify_all();
        }
    }
    if (attempt->committed)
        drain_retiring();
    if (attempt->progress) {
        auto event = load_event(RuntimeEvent::Kind::ModelLoadProgress, attempt->id, attempt->config.model.model_id);
        event.load_progress = std::exchange(attempt->progress, std::nullopt);
        events.push_back(std::move(event));
    }
    if (!attempt->committed)
        return;
    auto event = load_event(
        attempt->success ? RuntimeEvent::Kind::ModelLoaded : RuntimeEvent::Kind::ModelLoadFailed,
        attempt->id,
        attempt->config.model.model_id
    );
    if (!attempt->success) {
        event.error = attempt->failure.error;
        event.text = attempt->failure.message;
    }
    events.push_back(std::move(event));
    _loading->attempt.reset();
}

} // namespace Chorus

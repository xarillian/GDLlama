#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

namespace Chorus {
/*
 * Reference implementation of `Chorus::InferenceEngine`.
 *
 * Echoes space-delimited chunks as `Chorus::ChorusSignal::Token` signals, then
 * emits `Chorus::ChorusSignal::Completion` with usage measured in those chunks.
 * Prompt counts cover only the echoed text, which for chat is the last user
 * message, not the conversation. Uncapped output reconstructs that text exactly.
 *
 * A worker thread mirrors the asynchronous engine contract without a model or
 * external dependency, keeping the provider seam compiler-enforced and available
 * to CI and integrations. Echoed prompts do not represent the dialogue quality
 * or pacing of a real provider.
 */
class EchoEngine : public InferenceEngine {
  public:
    ~EchoEngine() override;

    std::optional<Chorus::InitializationFailure> initialize(
        const Chorus::ChorusConfig& config, Chorus::Logger logger, const InitializationControl& control
    ) override;
    bool is_initialized() const override;

    EngineCapabilities capabilities() const override;
    std::optional<LoadedModelInfo> loaded_model_info() const override;

    std::optional<RequestRejection> validate_request(const Chorus::ChorusRequest& request) const override;
    void submit_request(Chorus::ChorusRequest chorus_request) override;
    std::shared_ptr<RequestPreparation> request_preparation() const override;
    void cancel_request(RequestId id) override;
    void shutdown() override;

  private:
    void worker_loop();
    static std::string select_echo_text(const Chorus::ChorusRequest& request);
    int64_t emit_echo_tokens(const Chorus::ChorusRequest& request, const std::vector<std::string_view>& chunks);

    std::deque<Chorus::ChorusRequest> _queue;
    std::optional<Chorus::ChorusRequest> _active;
    std::unordered_set<RequestId> _cancelled_ids;
    size_t _queued_cancel_callbacks_in_flight = 0;
    std::mutex _queue_mutex;
    std::condition_variable _queue_cv;
    std::thread _worker;
    bool _running = false;
    bool _initialized = false;
    Chorus::Logger _log;
    struct Preparation;
    std::shared_ptr<Preparation> _preparation;
};
} // namespace Chorus

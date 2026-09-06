#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"

#include <memory>
#include <mutex>
#include <optional>

#ifdef TEST_BUILD
#include <functional>
#include <vector>
#endif

class LlamaScheduler;

namespace Chorus {

EngineCapabilities llama_provider_capabilities();

#ifdef TEST_BUILD
struct LlamaBatchRecord {
    RequestType type;
    int32_t token_count;
    std::vector<RequestId> request_ids;
    std::vector<int32_t> embedding_output_indices;
};
#endif

/*
 * llama.cpp-backed implementation of `Chorus::InferenceEngine`.
 *
 * Owns at most one `::LlamaScheduler` and synchronizes its publication and
 * replacement. The scheduler owns model resources, request scheduling, and
 * worker-thread callback delivery.
 */
class LlamaEngine : public InferenceEngine {
  public:
    ~LlamaEngine() override;

    std::optional<Chorus::ChorusError> initialize(const Chorus::ChorusConfig& config, Chorus::Logger logger) override;
    bool is_initialized() const override;

    EngineCapabilities capabilities() const override;
    std::optional<LoadedModelInfo> loaded_model_info() const override;
    std::optional<RenderedPrompt> render_chat_prompt(
        const std::vector<ChatMessage>& messages, const std::string& template_override, bool enable_thinking
    ) const override;

    std::optional<RequestRejection> validate_request(const Chorus::ChorusRequest& request) const override;
    void submit_request(const Chorus::ChorusRequest& chorus_request) override;
    void cancel_request(RequestId id) override;
    void shutdown() override;

#ifdef TEST_BUILD
    void set_batch_observer(std::function<void(const LlamaBatchRecord&)> observer);
#endif

  private:
    std::shared_ptr<LlamaScheduler> scheduler_snapshot() const;

    mutable std::mutex _lifecycle_mutex;
    std::shared_ptr<LlamaScheduler> _scheduler;
    Chorus::Logger _log;
};
} // namespace Chorus

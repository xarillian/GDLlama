#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"
#include <shared_mutex>
#include "chorus/providers/llama/llama_batch_planner.hpp"
#include "chorus/providers/llama/llama_batch_sampler.hpp"
#include "chorus/providers/llama/llama_chat.hpp"
#include "chorus/providers/llama/llama_chat_renderer.hpp"
#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_load_config.hpp"
#include "chorus/providers/llama/llama_recovery_planner.hpp"
#include "chorus/providers/llama/llama_sequence_id_pool.hpp"
#include "chorus/providers/llama/llama_session_cache.hpp"
#include "chorus/providers/llama/llama_log_bridge.hpp"
#include "chorus/providers/llama/llama_utils.hpp"
#include "chorus/providers/llama/stop_sequence_filter.hpp"
#include "wlib/utf8.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#ifdef TEST_BUILD
#include <functional>
#endif

struct llama_model;
struct llama_context;

namespace Chorus {
constexpr bool llama_embedding_architecture_supported(bool has_encoder, bool has_decoder) {
    return has_encoder != has_decoder;
}
#ifdef TEST_BUILD
struct LlamaBatchRecord;
#endif
} // namespace Chorus

class LlamaScheduler : public Chorus::RequestPreparation {
  public:
    ~LlamaScheduler();

    std::optional<Chorus::InitializationFailure> initialize(const Chorus::ChorusConfig& config, Chorus::Logger logger, const Chorus::InitializationControl& control);
    bool push_request(Chorus::ChorusRequest req);
    void cancel_request(Chorus::RequestId id);
    void shutdown();
    bool is_healthy() const;

    const std::optional<Chorus::LoadedModelInfo>& model_info() const { return _model_info; }
    Chorus::EngineCapabilities capabilities() const;
    std::optional<Chorus::RequestRejection> validate_embedding(const Chorus::ChorusRequest& request) const;
#ifdef TEST_BUILD
    void set_batch_observer(std::function<void(const Chorus::LlamaBatchRecord&)> observer);
    void set_admission_observer(std::function<void(Chorus::RequestId)> observer);
    uint64_t worker_iterations() const { return _worker_iterations.load(); }
#endif
    std::optional<Chorus::RequestRejection> validate_request(const Chorus::ChorusRequest& request) const override;
    std::variant<int64_t, Chorus::RequestRejection> count_message_tokens(const std::string& text) const override;
    std::variant<Chorus::RenderedPrompt, Chorus::RequestRejection> render_chat_prompt(
        const std::vector<Chorus::ChatMessage>& messages, const std::optional<std::string>& template_override, std::optional<bool> enable_thinking
    ) const;

  private:
    enum class GenerationPhase { Prefill, Decode };

    struct Sequence {
        int id = -1;
        Chorus::ChorusRequest request;
        uint64_t submission_sequence = 0;
        GenerationPhase phase = GenerationPhase::Prefill;
        int32_t n_past = 0;
        int32_t n_decoded = 0;
        int32_t max_tokens = -1;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> cached_tokens;
        size_t prompt_cursor = 0;
        int32_t pending_token = -1;
        common_sampler_ptr sampler;
        Chorus::LlamaSamplingPath sampling_path = Chorus::LlamaSamplingPath::Context;
        std::optional<Chorus::StopSequenceFilter> stop_filter;
        std::optional<Chorus::LlamaChatParseStream> parse_stream;
        wlib::Utf8Chunker reasoning_chunker;
        wlib::Utf8Chunker content_chunker;
    };

    struct PreparedRequest;
    struct PendingRequest {
        Chorus::ChorusRequest request;
        uint64_t submission_sequence = 0;
        std::optional<Chorus::ResolvedLlamaGeneration> resolved;
        std::shared_ptr<PreparedRequest> prepared;
    };
    using PendingRequestPtr = std::shared_ptr<PendingRequest>;
    struct PendingRequestCompare {
        bool operator()(const PendingRequestPtr& left, const PendingRequestPtr& right) const {
            if (left->request.priority != right->request.priority)
                return left->request.priority < right->request.priority;
            return left->submission_sequence > right->submission_sequence;
        }
    };
    struct PendingSignal {
        PendingSignal(Chorus::ChorusRequest request, Chorus::ChorusSignal::Event event)
            : request(std::move(request)), event(std::move(event)) {}
        Chorus::ChorusRequest request;
        Chorus::ChorusSignal::Event event;
    };
    struct PreparedRequest {
        std::vector<int32_t> tokens;
        common_sampler_ptr sampler;
        Chorus::LlamaSamplingPath sampling_path = Chorus::LlamaSamplingPath::Context;
        std::vector<std::string> stop_sequences;
        std::optional<Chorus::LlamaChatParseStream> parse_stream;
        int32_t max_tokens = -1;
        uint64_t submission_sequence = 0;
    };
    using PreparedRequestResult = std::variant<PreparedRequest, Chorus::RequestRejection>;
    struct BatchEntry {
        int sequence_id;
        int32_t token;
        int32_t position;
        bool logits;
    };
    struct SequenceDelta {
        int sequence_id;
        size_t prompt_advance = 0;
        int32_t kv_advance = 0;
        bool sampled = false;
        int32_t embedding_output_index = -1;
    };
    struct BatchPlan {
        Chorus::RequestType type = Chorus::RequestType::Generate;
        std::vector<BatchEntry> entries;
        std::vector<SequenceDelta> deltas;
        std::vector<int> participants;
        std::optional<Chorus::RequestType> contested_type;
        std::optional<uint64_t> decode_fairness;
    };

    bool load_model_from_file(const Chorus::LlamaLoadConfig& config, const Chorus::InitializationControl& control,
                              bool& callback_cancelled, bool& callback_failed);
    bool init_context(const Chorus::LlamaLoadConfig& config);
    void worker_loop();
    void run_worker();
    bool process_control_requests();
    bool take_cancellation(Chorus::RequestId id, bool& stopped);
    void admit_available();
    void compact_sequence_ids();
    void claim_sequence_id(Sequence& sequence);
    int32_t reuse_parked_prefix(int id, const std::vector<int32_t>& parked, const std::vector<int32_t>& prompt);
    int seat_for_new_sequence() const;
    bool is_short_session(int id) const;
    void drop_parked(int id);
    bool make_room(int id);
    void evict_for(int id);
    void move_kv(int from, int to);
    std::optional<PendingSignal> resolve_pending_request(PendingRequest& pending);
    PreparedRequestResult prepare_request(PendingRequest& pending);
    bool has_active_exclusive() const;
    std::vector<int> ordered_active_sequence_ids() const;
    std::vector<Sequence*> ordered_runnable() const;
    std::optional<BatchPlan> build_plan(int32_t generation_budget, int32_t embedding_budget) const;
    void populate_batch(const BatchPlan& plan);
    int run_inference(const BatchPlan& plan);
    void commit_plan(const BatchPlan& plan);
    void process_generation_plan(const BatchPlan& plan);
    void advance_sequence(Sequence& sequence, llama_token token);
    void process_embedding_plan(const BatchPlan& plan);
    bool recover_decode(const BatchPlan& failed);
    std::optional<BatchPlan> recovery_plan(const BatchPlan& failed, const Chorus::LlamaRecoverySelection& selection) const;
    std::string recovery_capacity_message() const;

    void retire_sequence(int sequence_id, Chorus::ChorusSignal::Event event, std::vector<PendingSignal>& signals);
    void emit_signal(PendingSignal pending);
    void emit_signals(std::vector<PendingSignal> pending);
    void emit_token(Sequence& sequence, std::string text, Chorus::TokenChannel channel = Chorus::TokenChannel::Content);
    void complete_sequence(int sequence_id, bool flush_pending_text);
    void fail_all(Chorus::ChorusError code);

    std::priority_queue<PendingRequestPtr, std::vector<PendingRequestPtr>, PendingRequestCompare> request_queue;
    struct TerminalDelivery {
        decltype(Chorus::ChorusRequest::on_event) on_event;
        Chorus::ChorusSignal failure;
        int priority;
        uint64_t submission_sequence;
    };
    // Delivery ownership outlives preparation and sequence retirement, including stack unwinding.
    std::map<Chorus::RequestId, TerminalDelivery> _terminal_deliveries;
    std::unordered_set<Chorus::RequestId> _cancel_requested;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> is_running{false};
    std::thread worker_thread;

    llama_model* model = nullptr;
    llama_context* context = nullptr;
    Chorus::LlamaUtils::Batch batch;
    std::optional<Chorus::LlamaBatchSampler> _batch_sampler;
    Chorus::LlamaOffloadDeviceList _no_offload_devices{};
    std::unordered_map<int, Sequence> _active_sequences;
    Chorus::LlamaSequenceIdPool _sequence_ids;
    int32_t _batch_capacity = 0;
    int32_t _micro_batch_capacity = 0;
    Chorus::RequestType _last_contested_type = Chorus::RequestType::Generate;
    bool _has_contested_type = false;
    uint64_t _decode_fairness_cursor = 0;
    enum llama_pooling_type _pooling = LLAMA_POOLING_TYPE_UNSPECIFIED;
    int32_t _embedding_dimensions = 0;
    bool _has_encoder = false;
    bool _has_decoder = false;
    uint64_t _next_submission_sequence = 0;
    uint32_t _max_concurrent_requests = 1;
    bool _compacts_sequence_ids = false;
    bool _parks_sessions = false;
    Chorus::LlamaSessionCache _session_cache;
    bool _serves_embeddings = false;

    Chorus::Logger _log;
    std::shared_ptr<Chorus::LlamaLogBridge> _llama_log_bridge;
    std::optional<Chorus::LoadedModelInfo> _model_info;
#ifdef TEST_BUILD
    std::mutex _batch_observer_mutex;
    std::function<void(const Chorus::LlamaBatchRecord&)> _batch_observer;
    std::function<void(Chorus::RequestId)> _admission_observer;
    std::atomic<uint64_t> _worker_iterations{0};
#endif
    std::optional<Chorus::LlamaChatRenderer> _chat_renderer;
    mutable std::shared_mutex _preparation_fence;
};

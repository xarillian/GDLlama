#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_chat.hpp"
#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_load_config.hpp"
#include "chorus/providers/llama/llama_log_bridge.hpp"
#include "chorus/providers/llama/llama_utils.hpp"
#include "chorus/providers/llama/stop_sequence_filter.hpp"
#include "wlib/utf8.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
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

/*
 * Schedules concurrent llama.cpp generation requests.
 *
 * Host-facing methods enqueue control work under `LlamaScheduler::queue_mutex`.
 * `LlamaScheduler::worker_loop` owns active slot mutation and invokes
 * `Chorus::ChorusRequest::on_event` only after releasing that mutex.
 * `LlamaScheduler::shutdown` joins the worker before releasing llama.cpp
 * resources, fencing every request callback before it returns.
 */
class LlamaScheduler {
  public:
    ~LlamaScheduler();

    std::optional<Chorus::ChorusError> initialize(const Chorus::ChorusConfig& config, Chorus::Logger logger);

    /*
     * Queues a request without invoking `Chorus::ChorusRequest::on_event` inline.
     *
     * Returns:
     *  - `true`: the scheduler accepted the request.
     *  - `false`: shutdown has begun and the request was not accepted.
     */
    bool push_request(const Chorus::ChorusRequest& req);
    void cancel_request(Chorus::RequestId id);
    void shutdown();
    bool is_healthy() const;

    const std::optional<Chorus::LoadedModelInfo>& model_info() const { return _model_info; }
    Chorus::EngineCapabilities capabilities() const;
    std::optional<Chorus::RequestRejection> validate_embedding(const Chorus::ChorusRequest& request) const;

#ifdef TEST_BUILD
    void set_batch_observer(std::function<void(const Chorus::LlamaBatchRecord&)> observer);
#endif

    /*
     * Renders the exact templated prompt and token count for the provided messages.
     *
     * The host may call this beside the worker. Every application of
     * `::common_chat_templates_apply` is serialized by
     * `LlamaScheduler::_template_mutex` because llama.cpp provides no
     * thread-safety guarantee for chat-template application.
     *
     * Returns:
     *  - `Chorus::RenderedPrompt`: rendering and tokenization succeeded.
     *  - `std::nullopt`: no model is loaded or the template rejected the request.
     */
    std::optional<Chorus::RenderedPrompt> render_chat_prompt(
        const std::vector<Chorus::ChatMessage>& messages, const std::string& template_override, bool enable_thinking
    ) const;

  private:
    struct Slot {
        int id = -1;
        bool is_busy = false;
        uint64_t submission_sequence = 0;
        bool included_in_batch = false;
        int32_t embedding_output_index = -1;

        Chorus::ChorusRequest current_request;
        int32_t n_past = 0;      // KV cache position; advanced only in `LlamaScheduler::prepare_next_batch`
        int32_t n_decoded = 0;   // sampled token count; advanced only in `LlamaScheduler::worker_loop`
        int32_t max_tokens = -1; // resolved at ingest; `-1` means unbounded until EOS or context exhaustion

        std::vector<int32_t> current_input_tokens;
        size_t input_cursor = 0;

        common_sampler_ptr sampler;
        std::optional<Chorus::StopSequenceFilter> stop_filter;
        // Present when rendering advertised thinking support. Token emission
        // uses it to separate reasoning and content channels.
        std::optional<Chorus::LlamaChatParseStream> parse_stream;
        // UTF-8 guards for emission paths not covered by
        // `Chorus::StopSequenceFilter`.
        wlib::Utf8Chunker reasoning_chunker;
        wlib::Utf8Chunker content_chunker;
    };

    struct PendingRequest {
        Chorus::ChorusRequest request;
        uint64_t submission_sequence = 0;
        std::optional<Chorus::ResolvedLlamaGeneration> resolved;
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
        std::vector<std::string> stop_sequences;
        std::optional<Chorus::LlamaChatParseStream> parse_stream;
        int32_t max_tokens = -1;
        uint64_t submission_sequence = 0;
    };

    using PreparedRequestResult = std::variant<PreparedRequest, Chorus::RequestRejection>;

    bool load_model_from_file(const Chorus::LlamaLoadConfig& config);
    bool init_context(const Chorus::LlamaLoadConfig& config);
    void init_slots(uint32_t count);

    void worker_loop();
    bool process_control_requests();
    void ingest_new_requests();
    std::optional<PendingSignal> take_cancellation_terminal_locked(const Chorus::ChorusRequest& request);
    std::optional<PendingSignal> resolve_pending_request(PendingRequest& pending);
    PreparedRequestResult prepare_request(PendingRequest& pending);
    std::optional<PendingSignal>
    admit_request(const Chorus::ChorusRequest& request, PreparedRequestResult prepared);

    bool prepare_next_batch(int32_t tokens_per_tick);
    int run_inference();
    void sample_batch();
    void extract_embedding_batch();

    void emit_signal(PendingSignal pending);
    void emit_signals(std::vector<PendingSignal> pending);
    void emit_token(Slot& slot, std::string text, Chorus::TokenChannel channel = Chorus::TokenChannel::Content);
    void complete_slot(Slot& slot, bool flush_pending_text);
    void fail_busy_slots(Chorus::ChorusError code);

    int find_free_slot();
    void release_slot(int slot_id);
    PendingSignal release_with_event(Slot& slot, Chorus::ChorusSignal::Event event);

    std::priority_queue<PendingRequestPtr, std::vector<PendingRequestPtr>, PendingRequestCompare> request_queue;
    std::unordered_set<Chorus::RequestId> _cancel_requested;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    std::atomic<bool> is_running{false};
    std::thread worker_thread;

    llama_model* model = nullptr;
    llama_context* context = nullptr;
    Chorus::LlamaUtils::Batch batch;
    Chorus::LlamaOffloadDeviceList _no_offload_devices{};

    std::vector<Slot> slots;
    int32_t _tokens_per_tick = 512;
    // Capacity of `LlamaScheduler::batch`; batching must never exceed
    // `LlamaScheduler::_batch_capacity`.
    int32_t _batch_capacity = 0;
    int32_t _micro_batch_capacity = 0;
    Chorus::RequestType _batch_type = Chorus::RequestType::Generate;
    enum llama_pooling_type _pooling = LLAMA_POOLING_TYPE_UNSPECIFIED;
    int32_t _embedding_dimensions = 0;
    bool _has_encoder = false;
    bool _has_decoder = false;
    uint64_t _next_submission_sequence = 0;

    Chorus::Logger _log;
    // Routes llama.cpp's process-global log into `LlamaScheduler::_log` and
    // remains acquired until `LlamaScheduler::shutdown` has released every
    // llama.cpp resource.
    std::shared_ptr<Chorus::LlamaLogBridge> _llama_log_bridge;
    std::optional<Chorus::LoadedModelInfo> _model_info;

#ifdef TEST_BUILD
    std::mutex _batch_observer_mutex;
    std::function<void(const Chorus::LlamaBatchRecord&)> _batch_observer;
#endif

    common_chat_templates_ptr _model_default_chat_templates;
    mutable std::mutex _template_mutex;
};

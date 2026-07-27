#pragma once

#include "chorus/backends/llama/llama_chat.hpp"
#include "chorus/backends/llama/llama_generation.hpp"
#include "chorus/backends/llama/llama_load_config.hpp"
#include "chorus/backends/llama/stop_sequence_filter.hpp"
#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"
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

struct llama_model;
struct llama_context;
struct llama_batch;

class LlamaScheduler {
  public:
    LlamaScheduler();
    ~LlamaScheduler();

    std::optional<Chorus::ChorusError> initialize(const Chorus::ChorusConfig& config);
    // Returns false after shutdown and never invokes the request callback inline.
    bool push_request(const Chorus::ChorusRequest& req);
    void cancel_request(Chorus::RequestId id);
    void stop();
    bool is_healthy() const;

    const std::optional<Chorus::LoadedModelInfo>& model_info() const { return _model_info; }

    // Host-thread render hook (#5): the exact templated prompt + token count
    // for `messages`. Safe beside the running worker; template application is
    // serialized via _template_mutex (no upstream thread-safety guarantee).
    std::optional<Chorus::RenderedPrompt> render_chat_prompt(
        const std::vector<Chorus::ChatMessage>& messages, const std::string& template_override, bool enable_thinking
    ) const;

  private:
    struct Slot;

    bool load_model_from_file(const Chorus::LlamaLoadConfig& config);
    bool init_context(const Chorus::LlamaLoadConfig& config);
    void init_slots(int count);

    struct TerminalEvent {
        Chorus::ChorusRequest request;
        Chorus::EventType type = Chorus::EventType::Error;
        Chorus::ChorusError error = Chorus::ChorusError::None;
        std::string text;
        Chorus::TokenChannel channel = Chorus::TokenChannel::Content; // meaningful on Token events
    };

    void ingest_new_requests();
    bool process_control_requests();
    void emit_terminal(TerminalEvent terminal);
    void emit_terminals(std::vector<TerminalEvent> terminals);
    TerminalEvent release_with_terminal(
        Slot& slot, Chorus::EventType type, Chorus::ChorusError error = Chorus::ChorusError::None, std::string text = {}
    );
    bool prepare_next_batch(int32_t tokens_per_tick);
    int run_inference();
    void fail_busy_slots(Chorus::ChorusError code);
    void emit_token(Slot& slot, std::string text, Chorus::TokenChannel channel = Chorus::TokenChannel::Content);
    void complete_slot(Slot& slot, bool flush_pending_text);

    void worker_loop();

    struct Slot {
        int id = -1; // KV Cache Sequence ID
        bool is_busy = false;

        Chorus::ChorusRequest current_request;
        int32_t n_past = 0;      // KV cache position; advanced only in prepare_next_batch
        int32_t n_decoded = 0;   // generated (sampled) tokens; advanced only in worker_loop
        int32_t max_tokens = -1; // resolved at ingest; -1 means unbounded (until EOS/context)

        // Input State
        std::vector<int32_t> current_input_tokens;
        size_t input_cursor = 0; // How many input tokens have we batched so far?

        common_sampler_ptr sampler;
        std::optional<Chorus::StopSequenceFilter> stop_filter;
        // #5: present when the render advertised thinking support; Task 10
        // wires it into token emission (reasoning/content channel split).
        std::optional<Chorus::LlamaChatParseStream> parse_stream;
        // UTF-8 boundary guards for the emission paths the stop filter does
        // not cover: reasoning deltas, and content when no filter is set.
        wlib::Utf8Chunker reasoning_chunker;
        wlib::Utf8Chunker content_chunker;
    };

    struct PendingRequest {
        Chorus::ChorusRequest request;
        std::optional<Chorus::ResolvedLlamaGeneration> resolved;
    };

    using PendingRequestPtr = std::shared_ptr<PendingRequest>;

    struct PendingRequestCompare {
        bool operator()(const PendingRequestPtr& left, const PendingRequestPtr& right) const {
            return left->request < right->request;
        }
    };

    int find_free_slot();
    void release_slot(int slot_id);

    std::priority_queue<PendingRequestPtr, std::vector<PendingRequestPtr>, PendingRequestCompare> request_queue;
    std::unordered_set<Chorus::RequestId> _cancel_requested;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    llama_model* model = nullptr;
    llama_context* context = nullptr;
    Chorus::LlamaOffloadDeviceList _no_offload_devices{};

    std::vector<Slot> slots;

    std::atomic<bool> is_running{false};
    std::thread worker_thread;

    int32_t _tokens_per_tick = 512;
    int32_t _batch_capacity = 0; // token capacity of `batch`; prepare_next_batch must never exceed it
    Chorus::LogCallback _log;
    std::optional<Chorus::LoadedModelInfo> _model_info;

    // #5 chat templates, initialized from the model at load. Reached from the
    // worker (ingest) and the host (render_chat_prompt); llama.cpp declares no
    // thread-safety for common_chat_templates_apply, so every application
    // takes _template_mutex first.
    common_chat_templates_ptr _chat_templates;
    mutable std::mutex _template_mutex;

    struct llama_batch* batch = nullptr;
};

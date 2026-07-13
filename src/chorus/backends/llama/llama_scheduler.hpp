#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <variant>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_batch;

class LlamaScheduler {
  public:
    LlamaScheduler();
    ~LlamaScheduler();

    std::optional<Chorus::ChorusError> initialize(const Chorus::ChorusConfig& config);
    void push_request(const Chorus::ChorusRequest& req);
    void stop();
    bool is_healthy() const;

    const std::optional<Chorus::LoadedModelInfo>& model_info() const { return _model_info; }

  private:
    struct LoadConfig {
        std::string weights_path;
        int32_t context_size = 2048;
        int32_t thread_count = 4;
        bool use_gpu = true;
        int32_t gpu_layers = 99;
        int32_t num_slots = 1;
        int32_t tokens_per_tick = 512;
    };
    // Parses config.model + config.backend_options["llama"]. Unknown keys, wrong
    // value types, missing weights asset, or non-empty model.backend_options
    // return an error: options are never silently dropped.
    static std::variant<LoadConfig, Chorus::RequestRejection> parse_load_config(const Chorus::ChorusConfig& config);

    bool load_model_from_file(const LoadConfig& config);
    bool init_context(const LoadConfig& config);
    void init_slots(int count);

    void ingest_new_requests();
    bool prepare_next_batch(int32_t tokens_per_tick);
    int run_inference();
    void fail_busy_slots(Chorus::ChorusError code);

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

        llama_sampler* sampler = nullptr;
    };

    int find_free_slot();
    void release_slot(int slot_id);

    std::priority_queue<Chorus::ChorusRequest> request_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    llama_model* model = nullptr;
    llama_context* context = nullptr;

    std::vector<Slot> slots;

    std::atomic<bool> is_running{false};
    std::thread worker_thread;

    int32_t _tokens_per_tick = 512;
    int32_t _batch_capacity = 0; // token capacity of `batch`; prepare_next_batch must never exceed it
    Chorus::LogCallback _log;
    std::optional<Chorus::LoadedModelInfo> _model_info;

    struct llama_batch* batch = nullptr;
};
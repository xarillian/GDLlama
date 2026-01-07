#pragma once

#include "chorus_core/chorus_common.hpp"

#include <queue>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <map>
#include <memory>


struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_batch;

class LlamaScheduler {
public:
    LlamaScheduler();
    ~LlamaScheduler();

    bool initialize(const Chorus::ChorusConfig& config);
    void push_request(const Chorus::ChorusRequest& req);
    void stop();

private:
    bool load_model_from_file(const Chorus::ChorusConfig& config);
    bool init_context(const Chorus::ChorusConfig& config);
    void init_slots(int count);

    void ingest_new_requests();
    bool prepare_next_batch(int32_t tokens_per_tick);
    bool run_inference();

    void worker_loop();

    struct Slot {
        int id = -1;  // KV Cache Sequence ID
        bool is_busy = false;

        Chorus::ChorusRequest current_request;
        int32_t tokens_generated = 0;  // n_past

        // Input State
        std::vector<int32_t> current_input_tokens;
        size_t input_cursor = 0;  // How many input tokens have we batched so far?

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

    struct llama_batch* batch = nullptr;
};
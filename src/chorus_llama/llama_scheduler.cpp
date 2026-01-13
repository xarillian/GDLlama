#include "chorus_llama/llama_scheduler.hpp"
#include "chorus_llama/llama_utils.hpp"
#include "chorus_core/chorus_common.hpp"

#include <iostream>
#include <algorithm>
#include <cassert>

// --------------------------------------------------------------------------
// LIFECYCLE
// --------------------------------------------------------------------------

LlamaScheduler::LlamaScheduler() {

}

LlamaScheduler::~LlamaScheduler() {
    stop();
}

bool LlamaScheduler::load_model_from_file(const Chorus::ChorusConfig& config) {
    if (model) return true;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.use_gpu ? config.gpu_layers : 0;
    
    // @todo Add progress callback here for Godot UI feedback
    
    model = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "[Chorus] Error: Failed to load model from " << config.model_path << std::endl;
        return false;
    }

    return true;
}

bool LlamaScheduler::init_context(const Chorus::ChorusConfig& config) {
    if (context) return true;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.context_size;
    ctx_params.n_threads = config.thread_count;
    ctx_params.n_threads_batch = config.thread_count;

    context = llama_init_from_model(model, ctx_params);
    if (!context) {
        std::cerr << "[Chorus] Error: Failed to create Llama context." << std::endl;
        return false;
    }

    return true;
}

void LlamaScheduler::init_slots(int count) {
    slots.clear();
    slots.resize(count);
    for (int i = 0; i < count; ++i) {
        slots[i].id = i;
        slots[i].is_busy = false;
        slots[i].tokens_generated = 0;
        slots[i].input_cursor = 0;
    }

}

bool LlamaScheduler::initialize(const Chorus::ChorusConfig& config) {
    if (!load_model_from_file(config)) return false;
    if (!init_context(config)) return false;

    init_slots(4);  // @todo we need to make this configurable, e.g. config.batch_size

    batch = new llama_batch(llama_batch_init(config.context_size, 0, 1));

    is_running = true;
    worker_thread = std::thread(&LlamaScheduler::worker_loop, this);

    return true;
}

void LlamaScheduler::stop() {
    if (!is_running) return;
    
    is_running = false;
    queue_cv.notify_all();

    if (worker_thread.joinable()) {
        worker_thread.join();
    }

    llama_batch_free(*batch);
    llama_free(context);
    context = nullptr;
    llama_model_free(model);
    model = nullptr;
}

void LlamaScheduler::push_request(const Chorus::ChorusRequest& req) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        request_queue.push(req);
    }
    queue_cv.notify_one();
}

int LlamaScheduler::find_free_slot() {
    for (int i = 0; i < slots.size(); ++i) {
        if (!slots[i].is_busy) return i;
    }
    return -1;
}

void LlamaScheduler::release_slot(int slot_id) {
    Slot& slot = slots[slot_id];
    
    if (slot.sampler) {
        // Clean up Llama Resources
        llama_sampler_free(slot.sampler);
        slot.sampler = nullptr;
    }

    // We do NOT clear the KV cache here necessarily. 
    // @todo if the next request uses the same system prompt, we could reuse it.

    slot.is_busy = false;
    slot.current_input_tokens.clear();
}

// --------------------------------------------------------------------------
// CORE PROCESSING
// --------------------------------------------------------------------------

void LlamaScheduler::ingest_new_requests() {
    std::lock_guard<std::mutex> lock(queue_mutex);

    while (!request_queue.empty()) {
        int slot_idx = find_free_slot();
        if (slot_idx == -1) break; 

        Chorus::ChorusRequest chorus_request = request_queue.top();
        request_queue.pop();

        Slot& slot = slots[slot_idx];
        slot.is_busy = true;
        slot.current_request = chorus_request;
        slot.tokens_generated = 0;
        slot.input_cursor = 0;

        slot.current_input_tokens = Chorus::LlamaUtils::tokenize(
            context, 
            chorus_request.prompt,
            true
        );

        slot.sampler = Chorus::LlamaUtils::build_sampler(chorus_request.gen_config);
        
        llama_memory_t mem = llama_get_memory(context);
        llama_memory_seq_rm(mem, slot.id, 0, -1);
    }
}

bool LlamaScheduler::prepare_next_batch(int32_t tokens_per_tick) {
    llama_batch& curr_batch = *batch;
    curr_batch.n_tokens = 0; // Reset for this tick

    for (auto& slot : slots) {
        if (!slot.is_busy) continue;

        if (slot.input_cursor < slot.current_input_tokens.size()) {
            
            size_t n_remaining = slot.current_input_tokens.size() - slot.input_cursor;
            size_t n_chunk = std::min(n_remaining, (size_t)tokens_per_tick); 

            for (size_t i = 0; i < n_chunk; ++i) {
                int32_t pos = slot.tokens_generated + i; 
                bool is_last_in_sequence = (slot.input_cursor + i == slot.current_input_tokens.size() - 1);
                
                Chorus::LlamaUtils::batch_add_seq(
                    curr_batch,
                    slot.current_input_tokens[slot.input_cursor + i],
                    slot.id,
                    pos,
                    is_last_in_sequence 
                );
            }

            slot.tokens_generated += n_chunk;
            slot.input_cursor += n_chunk;
        }
    }

    return curr_batch.n_tokens > 0;
}

bool LlamaScheduler::run_inference() {
    if (llama_decode(context, *batch) != 0) {
        std::cerr << "[Chorus] Critical Error: llama_decode failed." << std::endl;
        return false;
    }
    return true;
}

void LlamaScheduler::worker_loop() {
    int32_t tokens_per_tick = 512;

    while (is_running) {
        ingest_new_requests();

        bool has_work = prepare_next_batch(tokens_per_tick);

        if (!has_work) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;  // @todo can we do this without continue?
        }

        if (!run_inference()) {
            continue;  // @todo can we do this without continue?
        }

        const llama_vocab* vocab = llama_model_get_vocab(model);

        llama_batch& curr_batch = *batch;
        for (int i = 0; i < curr_batch.n_tokens; ++i) {
            if (!curr_batch.logits[i]) continue;

            int seq_id = curr_batch.seq_id[i][0];
            Slot& slot = slots[seq_id];

            llama_token new_token_id = llama_sampler_sample(slot.sampler, context, curr_batch.pos[i]);
            llama_sampler_accept(slot.sampler, new_token_id);
    
            Chorus::ChorusSignal chorus_signal;
            chorus_signal.request_id = slot.current_request.id;
            chorus_signal.type = Chorus::EventType::Token;
            chorus_signal.text = Chorus::LlamaUtils::token_to_piece(context, new_token_id);

            if (slot.current_request.on_event) {
                slot.current_request.on_event(chorus_signal);
            }

            bool is_eos = llama_vocab_is_eog(vocab, new_token_id);
            bool is_limit = (
                slot.current_request.gen_config.max_tokens > 0 && 
                slot.tokens_generated >= slot.current_request.gen_config.max_tokens
            );

            if (is_eos || is_limit) {
                Chorus::ChorusSignal stop_sig;
                stop_sig.request_id = slot.current_request.id;
                stop_sig.type = Chorus::EventType::Stop;
                if (slot.current_request.on_event) slot.current_request.on_event(stop_sig);

                release_slot(slot.id);
            } else {
                slot.current_input_tokens.push_back(new_token_id);
                slot.tokens_generated++;
            }
        }
    }
}

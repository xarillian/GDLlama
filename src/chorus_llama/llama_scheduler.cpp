#include "chorus_llama/llama_scheduler.hpp"
#include "chorus_core/chorus_common.hpp"
#include "chorus_llama/llama_utils.hpp"

#include <algorithm>
#include <cassert>

// --------------------------------------------------------------------------
// LIFECYCLE
// --------------------------------------------------------------------------

LlamaScheduler::LlamaScheduler() {}

LlamaScheduler::~LlamaScheduler() {
    stop();
}

bool LlamaScheduler::load_model_from_file(const Chorus::ChorusConfig& config) {
    if (model)
        return true;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.use_gpu ? config.gpu_layers : 0;

    // @todo Add a host-agnostic progress callback here (via ChorusConfig, like log_callback);
    //       the binding layer wires it to whatever UI the host uses. Core must not know the host.

    model = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model) {
        Chorus::chorus_log(_log, Chorus::LogLevel::Error, "Failed to load model from " + config.model_path);
        return false;
    }

    return true;
}

bool LlamaScheduler::init_context(const Chorus::ChorusConfig& config) {
    if (context)
        return true;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.context_size;
    ctx_params.n_seq_max = config.num_slots;
    ctx_params.n_threads = config.thread_count;
    ctx_params.n_threads_batch = config.thread_count;

    context = llama_init_from_model(model, ctx_params);
    if (!context) {
        Chorus::chorus_log(_log, Chorus::LogLevel::Error, "Failed to create Llama context.");
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
        slots[i].n_past = 0;
        slots[i].n_decoded = 0;
        slots[i].input_cursor = 0;
    }
}

std::optional<Chorus::ChorusError> LlamaScheduler::initialize(const Chorus::ChorusConfig& config) {
    if (!load_model_from_file(config))
        return Chorus::ChorusError::ModelLoad;
    if (!init_context(config))
        return Chorus::ChorusError::ContextInit;

    init_slots(config.num_slots);
    _tokens_per_tick = config.tokens_per_tick;
    _log = config.log_callback;

    batch = new llama_batch(llama_batch_init(config.context_size, 0, 1));

    is_running = true;
    worker_thread = std::thread(&LlamaScheduler::worker_loop, this);

    return std::nullopt;
}

void LlamaScheduler::stop() {
    is_running = false;
    queue_cv.notify_all();

    if (worker_thread.joinable()) {
        worker_thread.join();
    }

    if (batch) {
        llama_batch_free(*batch);
        delete batch;
        batch = nullptr;
    }
    if (context) {
        llama_free(context);
        context = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
}

bool LlamaScheduler::is_healthy() const {
    return is_running.load();
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
        if (!slots[i].is_busy)
            return i;
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

    // Reclaim this sequence's KV cache so freed capacity is available to other slots.
    if (context) {
        llama_memory_t mem = llama_get_memory(context);
        llama_memory_seq_rm(mem, slot.id, 0, -1);
    }

    slot.is_busy = false;
    slot.current_input_tokens.clear();
}

void LlamaScheduler::fail_busy_slots(Chorus::ChorusError code) {
    for (auto& slot : slots) {
        if (!slot.is_busy)
            continue;

        if (slot.current_request.on_event) {
            Chorus::ChorusSignal sig;
            sig.request_id = slot.current_request.id;
            sig.type = Chorus::EventType::Error;
            sig.error_code = code;
            sig.text = "Inference decode failed.";
            slot.current_request.on_event(sig);
        }
        release_slot(slot.id);
    }
}

// --------------------------------------------------------------------------
// CORE PROCESSING
// --------------------------------------------------------------------------

void LlamaScheduler::ingest_new_requests() {
    std::lock_guard<std::mutex> lock(queue_mutex);

    while (!request_queue.empty()) {
        int slot_idx = find_free_slot();
        if (slot_idx == -1)
            break;

        Chorus::ChorusRequest chorus_request = request_queue.top();
        request_queue.pop();

        std::vector<int32_t> tokens = Chorus::LlamaUtils::tokenize(context, chorus_request.prompt, true);
        if (tokens.empty()) {
            Chorus::chorus_log(_log, Chorus::LogLevel::Error, "Tokenization produced no tokens; dropping request.");
            if (chorus_request.on_event) {
                Chorus::ChorusSignal sig;
                sig.request_id = chorus_request.id;
                sig.type = Chorus::EventType::Error;
                sig.error_code = Chorus::ChorusError::Tokenize;
                sig.text = "Tokenization failed (empty result).";
                chorus_request.on_event(sig);
            }
            continue; // slot stays free
        }

        Slot& slot = slots[slot_idx];
        slot.is_busy = true;
        slot.current_request = chorus_request;
        slot.n_past = 0;
        slot.n_decoded = 0;
        slot.input_cursor = 0;
        slot.current_input_tokens = std::move(tokens);
        slot.sampler = Chorus::LlamaUtils::build_sampler(chorus_request.gen_config);
    }
}

bool LlamaScheduler::prepare_next_batch(int32_t tokens_per_tick) {
    llama_batch& curr_batch = *batch;
    curr_batch.n_tokens = 0; // Reset for this tick

    for (auto& slot : slots) {
        if (!slot.is_busy)
            continue;

        if (slot.input_cursor < slot.current_input_tokens.size()) {

            size_t n_remaining = slot.current_input_tokens.size() - slot.input_cursor;
            size_t n_chunk = std::min(n_remaining, (size_t)tokens_per_tick);

            for (size_t i = 0; i < n_chunk; ++i) {
                int32_t pos = slot.n_past + i;
                bool is_last_in_sequence = (slot.input_cursor + i == slot.current_input_tokens.size() - 1);

                Chorus::LlamaUtils::batch_add_seq(
                    curr_batch, slot.current_input_tokens[slot.input_cursor + i], slot.id, pos, is_last_in_sequence
                );
            }

            slot.n_past += n_chunk;
            slot.input_cursor += n_chunk;
        }
    }

    return curr_batch.n_tokens > 0;
}

int LlamaScheduler::run_inference() {
    int rc = llama_decode(context, *batch);
    if (rc != 0) {
        Chorus::chorus_log(_log, Chorus::LogLevel::Error, "llama_decode failed with code " + std::to_string(rc) + ".");
    }
    return rc;
}

void LlamaScheduler::worker_loop() {
    while (is_running) {
        ingest_new_requests();

        bool has_work = prepare_next_batch(_tokens_per_tick);

        if (!has_work) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue; // @todo can we do this without continue?
        }

        int decode_rc = run_inference();
        if (decode_rc != 0) {
            fail_busy_slots(Chorus::ChorusError::Decode);
            if (decode_rc < 0) {
                Chorus::chorus_log(_log, Chorus::LogLevel::Fatal, "Fatal decode error; stopping engine.");
                is_running = false;
            }
            continue;
        }

        const llama_vocab* vocab = llama_model_get_vocab(model);

        llama_batch& curr_batch = *batch;
        for (int i = 0; i < curr_batch.n_tokens; ++i) {
            if (!curr_batch.logits[i])
                continue;

            int seq_id = curr_batch.seq_id[i][0];
            Slot& slot = slots[seq_id];

            llama_token new_token_id = llama_sampler_sample(slot.sampler, context, i);
            llama_sampler_accept(slot.sampler, new_token_id);
            slot.n_decoded++;

            Chorus::ChorusSignal chorus_signal;
            chorus_signal.request_id = slot.current_request.id;
            chorus_signal.type = Chorus::EventType::Token;
            chorus_signal.text = Chorus::LlamaUtils::token_to_piece(context, new_token_id);

            if (slot.current_request.on_event) {
                slot.current_request.on_event(chorus_signal);
            }

            bool is_eos = llama_vocab_is_eog(vocab, new_token_id);
            bool is_limit =
                (slot.current_request.gen_config.max_tokens > 0 &&
                 slot.n_decoded >= slot.current_request.gen_config.max_tokens);

            if (is_eos || is_limit) {
                Chorus::ChorusSignal stop_sig;
                stop_sig.request_id = slot.current_request.id;
                stop_sig.type = Chorus::EventType::Stop;
                if (slot.current_request.on_event)
                    slot.current_request.on_event(stop_sig);

                release_slot(slot.id);
            } else {
                slot.current_input_tokens.push_back(new_token_id);
            }
        }
    }
}

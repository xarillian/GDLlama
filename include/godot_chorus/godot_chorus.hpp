#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

#include "chorus_core/chorus_common.hpp"
#include "chorus_llama/llama_engine.hpp"

using namespace godot;

class GodotChorus : public Node {
    GDCLASS(GodotChorus, Node);

  protected:
    static void _bind_methods();
    void _notification(int p_what);

  public:
    GodotChorus();
    ~GodotChorus();

    // --- Core API ---

    bool load_model();
    void stop_all();
    bool is_loaded() const;

    // Primary generation entry point.
    // Request dict keys:
    //   prompt: String       (required)
    //   stream: bool         (If true, token_generated fires per token. Default: false)
    //   priority: int        (Higher = earlier in queue. Default: 0)
    //   max_tokens: int
    //   temperature: float
    //   top_k: int
    //   top_p: float
    //   repeat_penalty: float
    //   seed: int
    //   grammar: String      (GBNF)  <- @todo implement
    // Returns the request ID (>= 0) on success, or -1 on failure.
    int64_t generate(const Dictionary& request);

    // Godot lifecycle
    void _process(double delta) override;

    // --- Properties (new API) ---

    void set_model_path(const String& path);
    String get_model_path() const;

    void set_context_size(int32_t size);
    int32_t get_context_size() const;

    void set_thread_count(int32_t count);
    int32_t get_thread_count() const;

    void set_use_gpu(bool use);
    bool get_use_gpu() const;

    void set_gpu_layers(int32_t layers);
    int32_t get_gpu_layers() const;

    void set_num_slots(int32_t count);
    int32_t get_num_slots() const;

    void set_tokens_per_tick(int32_t count);
    int32_t get_tokens_per_tick() const;

    // --- Deprecated: GDLlama backward compat ---
    // All methods below emit a deprecation warning. Some features (embeddings,
    // chat context, sync blocking) are not yet implemented in the new architecture.

    // Model management
    void unload_model();
    bool is_model_loaded() const;

    // Async generation. Forward to generate() with stream=true to preserve old signal behavior
    Error generate_text_async(const String& prompt, const String& grammar = "", const String& json = "");
    Error generate_chat_async(const String& prompt, const String& grammar = "", const String& json = "");

    // Sync generation. These block the game thread and are strongly discouraged.
    // @deprecated Provided only for compilation compat; migrate to generate() as soon as possible.
    String generate_text(const String& prompt, const String& grammar = "", const String& json = "");
    String generate_chat(const String& prompt, const String& grammar = "", const String& json = "");

    // Control
    void stop_generate_text();
    bool is_running() const;
    void reset_context();

    // Embeddings. Not yet implemented; stubs for compile compatibility.
    PackedFloat32Array compute_embedding(const String& prompt);
    Error compute_embedding_async(const String& prompt);

    // Utility
    float similarity_cos(PackedFloat32Array array1, PackedFloat32Array array2) const;

    // Deprecated properties with real backing
    void set_n_predict(int32_t n);
    int32_t get_n_predict() const;
    void set_temperature(float t);
    float get_temperature() const;
    void set_top_k(int32_t k);
    int32_t get_top_k() const;
    void set_top_p(float p);
    float get_top_p() const;
    void set_penalty_repeat(float r);
    float get_penalty_repeat() const;
    void set_seed(int32_t s);
    int32_t get_seed() const;
    void set_n_ctx(int32_t n);
    int32_t get_n_ctx() const;
    void set_n_gpu_layers(int32_t n);
    int32_t get_n_gpu_layers() const;

    // Deprecated properties without a new-API equivalent (warn + no-op setters)
    void set_ignore_eos(bool v);
    bool get_ignore_eos() const;
    void set_penalty_last_n(int32_t n);
    int32_t get_penalty_last_n() const;
    void set_chat_template(const String& s);
    String get_chat_template() const;
    void set_n_batch(int32_t n);
    int32_t get_n_batch() const;
    void set_main_gpu(int32_t n);
    int32_t get_main_gpu() const;

  private:
    Chorus::LlamaEngine _engine;
    Chorus::ChorusConfig _chorus_config;

    // Default generation config
    // Populated by deprecated property setters and used as the base config for any call that
    // doesn't supply explicit values.
    Chorus::GenerationConfig _default_gen_config;

    std::atomic<int64_t> _next_request_id{0};

    // Thread-safe signal queue (written by worker thread, drained by _process on main thread).
    std::mutex _pending_mutex;
    std::vector<Chorus::ChorusSignal> _pending_signals;

    // Per-request state (main-thread only)
    std::unordered_map<int64_t, bool> _request_streaming;       // request_id → stream flag
    std::unordered_map<int64_t, std::string> _text_accumulator; // request_id → accumulated text

    // Internal helpers
    int64_t _submit(const String& prompt, const Chorus::GenerationConfig& gen_config, bool streaming, int priority);
    String _generate_sync(const String& prompt, const String& grammar);
    void _queue_signal(const Chorus::ChorusSignal& signal);
    void _drain_signals();
};

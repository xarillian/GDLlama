#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

#include "chorus/core/common.hpp"
#include "chorus/runtime/runtime.hpp"

class GodotChorus : public godot::Node {
    GDCLASS(GodotChorus, godot::Node);

  protected:
    static void _bind_methods();
    void _notification(int p_what);

  public:
    enum ErrorCode {
        ERR_NONE,
        ERR_MODEL_LOAD,
        ERR_CONTEXT_INIT,
        ERR_DECODE,
        ERR_TOKENIZE,
        ERR_INVALID_REQUEST,
        ERR_ENGINE_NOT_READY,
        ERR_CANCELLED,
        ERR_UNSUPPORTED_MODEL_FORMAT,
        ERR_UNSUPPORTED_FEATURE,
        ERR_UNSUPPORTED_OPTION,
        ERR_SESSION_BUSY,
        ERR_UNKNOWN,
    };

    enum BackendChoice {
        BACKEND_LLAMA, // mirrors Chorus::Backend::Llama
        BACKEND_ECHO,  // mirrors Chorus::Backend::Echo
    };

    static int to_godot(Chorus::ChorusError e);

    // --- Core API ---

    // Constructs the selected backend via the factory and hands it to the
    // runtime. Always (re)loads: a second call replaces the engine (in-flight
    // requests get ERR_CANCELLED), and the current config fully applies.
    bool load_model();
    void stop_all();
    bool is_loaded() const;

    // Request dict keys:
    //   prompt: String       (required)
    //   stream: bool         (If true, token_generated fires per token. Default: false)
    //   priority: int        (Higher = earlier in queue. Default: 0)
    //   max_tokens: int
    //   temperature: float
    //   top_k: int
    //   top_p: float
    //   seed: int
    //   session: String      (stable continuity lane, e.g. "npc_42/dialogue"; one live request per
    //                          session; empty/omitted = stateless. A same-frame resubmit after
    //                          stop_all()/load_model() is SessionBusy until poll() drains the Cancelled
    //                          terminal; deliberate, to preserve per-session event ordering.)
    //   backend_options: Dictionary
    //                        (namespaced, e.g. {"llama": {"repeat_penalty": 1.1}}; unknown options
    //                          are rejected, never ignored)
    //   repeat_penalty: float (convenience for backend_options["llama"]["repeat_penalty"]; applied
    //                          after backend_options, so it wins if both are given)
    //   grammar: String      (GBNF)  <- rejected until sampler wiring lands (#4)
    // Returns the request ID (>= 0) on success, or -1 on failure.
    int64_t generate(const godot::Dictionary& request);

    void _process(double delta) override;

    // --- Properties ---

    void set_model_path(const godot::String& path);
    godot::String get_model_path() const;
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
    void set_backend(BackendChoice backend);
    BackendChoice get_backend() const;

    // --- Utility ---
    float similarity_cos(godot::PackedFloat32Array array1, godot::PackedFloat32Array array2) const;

  private:
    Chorus::ChorusRuntime _runtime;

    godot::String _model_path;
    int32_t _context_size = 2048;
    int32_t _thread_count = 4;
    bool _use_gpu = true;
    int32_t _gpu_layers = 99;
    int32_t _num_slots = 1;
    int32_t _tokens_per_tick = 512;

    BackendChoice _backend = BACKEND_LLAMA;
};

VARIANT_ENUM_CAST(GodotChorus::ErrorCode);
VARIANT_ENUM_CAST(GodotChorus::BackendChoice);

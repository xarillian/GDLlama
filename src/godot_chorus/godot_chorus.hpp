#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

#include "chorus/core/common.hpp"
#include "chorus/runtime/runtime.hpp"
#include "godot_chorus/chorus_generation_defaults.hpp"

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
    //   prompt: String       (required, must not be null)
    //   stream: bool         (If true, token_generated fires per token. Default: false)
    //   priority: int        (Higher = earlier in queue. Default: 0)
    //   session: String      (stable continuity lane, e.g. "npc_42/dialogue"; one live request per
    //                          session; empty/omitted = stateless. A same-frame resubmit after
    //                          stop_all()/load_model() is SessionBusy until poll() drains the Cancelled
    //                          terminal; deliberate, to preserve per-session event ordering.)
    //
    // Generation overlay keys layer onto generation_defaults (the assigned resource, else an
    // internal default carrying max_tokens = 128), which itself layers onto the engine's
    // defaults. Absent = inherit the layer below; an explicit null clears an inherited value;
    // a present non-null value replaces it:
    //   max_tokens, temperature, top_k, top_p, seed, frequency_penalty, presence_penalty
    //   stop: Array[String]  (null clears to the backend default; [] explicitly disables any
    //                          inherited stop sequences; a non-empty array replaces them)
    //   backend_options: Dictionary
    //                        (namespaced, e.g. {"llama": {"repeat_penalty": 1.1}}; deep-merges
    //                          onto the inherited backend options; a null leaf erases the
    //                          corresponding inherited key; unknown options are rejected)
    //   repeat_penalty: float (convenience for backend_options["llama"]["repeat_penalty"]; applied
    //                          after backend_options, so it wins if both are given; null erases it)
    //   constraint: Dictionary {"format": "gbnf"|"json_schema", "source": String}, or one of the
    //                          convenience spellings grammar: String (GBNF text) / json_schema:
    //                          String (schema text) / json: String (same as json_schema); at most
    //                          one spelling may be present; null clears an inherited constraint.
    // Returns the request ID (>= 0) on success, or -1 on failure.
    int64_t generate(const godot::Dictionary& request);

    // --- Runtime controls ---

    // Requests stay active until _process() drains their terminal event.
    bool cancel_request(int64_t request_id);
    bool is_request_active(int64_t request_id) const;
    // -1 if the session has no active request (including an unknown session).
    int64_t active_request_for_session(const godot::String& session) const;

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
    void set_n_batch(int32_t count);
    int32_t get_n_batch() const;
    void set_n_ubatch(int32_t count);
    int32_t get_n_ubatch() const;
    void set_main_gpu(int32_t index);
    int32_t get_main_gpu() const;
    void set_backend(BackendChoice backend);
    BackendChoice get_backend() const;
    void set_generation_defaults(const godot::Ref<ChorusGenerationDefaults>& defaults);
    godot::Ref<ChorusGenerationDefaults> get_generation_defaults() const;

    // --- Utility ---
    float similarity_cos(godot::PackedFloat32Array array1, godot::PackedFloat32Array array2) const;

  private:
    // The assigned resource when set, otherwise a lazily constructed internal
    // default instance (max_tokens = 128), so the middle layer always
    // contributes a generation default even with the property unset.
    godot::Ref<ChorusGenerationDefaults> effective_generation_defaults();

    Chorus::ChorusRuntime _runtime;

    godot::String _model_path;
    int32_t _context_size = 2048;
    int32_t _thread_count = 4;
    bool _use_gpu = true;
    int32_t _gpu_layers = 99;
    int32_t _num_slots = 1;
    int32_t _tokens_per_tick = 512;
    int32_t _n_batch = 2048;
    int32_t _n_ubatch = 512;
    int32_t _main_gpu = 0;

    BackendChoice _backend = BACKEND_LLAMA;

    // The bound property: null when the user has not assigned a resource, so
    // scene serialization stays clean. effective_generation_defaults() supplies
    // the internal fallback when unset.
    godot::Ref<ChorusGenerationDefaults> _generation_defaults;
    godot::Ref<ChorusGenerationDefaults> _fallback_generation_defaults;
};

VARIANT_ENUM_CAST(GodotChorus::ErrorCode);
VARIANT_ENUM_CAST(GodotChorus::BackendChoice);

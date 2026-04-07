#include "godot_chorus/godot_chorus.hpp"

#include <cmath>
#include <condition_variable>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

// ===========================================================================
// Lifecycle
// ===========================================================================

GodotChorus::GodotChorus() {}

GodotChorus::~GodotChorus() {
    _engine.stop();
}

void GodotChorus::_notification(int p_what) {
    if (p_what == NOTIFICATION_READY) {
        set_process(true);
    }
}

void GodotChorus::_process(double /*delta*/) {
    _drain_signals();
}

// ===========================================================================
// Core API
// ===========================================================================

bool GodotChorus::load_model() {
    if (_chorus_config.model_path.empty()) {
        UtilityFunctions::push_error("[Chorus] model_path is not set.");
        return false;
    }
    return _engine.initialize(_chorus_config);
}

void GodotChorus::stop_all() {
    _engine.stop();
}

bool GodotChorus::is_loaded() const {
    return _engine.is_initialized();
}

int64_t GodotChorus::generate(const Dictionary& request) {
    if (!_engine.is_initialized()) {
        UtilityFunctions::push_error("[Chorus] Cannot generate — model not loaded. Call load_model() first.");
        return -1;
    }
    if (!request.has("prompt")) {
        UtilityFunctions::push_error("[Chorus] generate() requires a 'prompt' key in the request dictionary.");
        return -1;
    }

    String prompt = request["prompt"];
    bool streaming = request.has("stream") ? (bool)request["stream"] : false;
    int priority = request.has("priority") ? (int)(int64_t)request["priority"] : 0;

    // Start from defaults set via deprecated properties, then override with request keys.
    Chorus::GenerationConfig cfg = _default_gen_config;
    if (request.has("max_tokens"))
        cfg.max_tokens = (int32_t)(int64_t)request["max_tokens"];
    if (request.has("temperature"))
        cfg.temperature = (float)request["temperature"];
    if (request.has("top_k"))
        cfg.top_k = (int32_t)(int64_t)request["top_k"];
    if (request.has("top_p"))
        cfg.top_p = (float)request["top_p"];
    if (request.has("repeat_penalty"))
        cfg.repeat_penalty = (float)request["repeat_penalty"];
    if (request.has("seed"))
        cfg.seed = (uint32_t)(int64_t)request["seed"];
    if (request.has("grammar"))
        cfg.grammar = ((String)request["grammar"]).utf8().get_data();

    return _submit(prompt, cfg, streaming, priority);
}

// ===========================================================================
// Internal helpers
// ===========================================================================

int64_t
GodotChorus::_submit(const String& prompt, const Chorus::GenerationConfig& gen_config, bool streaming, int priority) {
    int64_t id = _next_request_id.fetch_add(1);

    // Record streaming preference before submitting (read back on main thread in _drain_signals).
    _request_streaming[id] = streaming;

    Chorus::ChorusRequest req;
    req.id = id;
    req.priority = priority;
    req.prompt = prompt.utf8().get_data();
    req.gen_config = gen_config;
    req.on_event = [this](Chorus::ChorusSignal& sig) { _queue_signal(sig); };

    _engine.submit_request(req);
    return id;
}

void GodotChorus::_queue_signal(const Chorus::ChorusSignal& signal) {
    std::lock_guard<std::mutex> lock(_pending_mutex);
    _pending_signals.push_back(signal);
}

void GodotChorus::_drain_signals() {
    std::vector<Chorus::ChorusSignal> batch;
    {
        std::lock_guard<std::mutex> lock(_pending_mutex);
        batch.swap(_pending_signals);
    }

    for (const auto& sig : batch) {
        int64_t rid = sig.request_id;

        switch (sig.type) {
        case Chorus::EventType::Token: {
            _text_accumulator[rid] += sig.text;
            // Only forward individual tokens if the caller opted into streaming.
            auto it = _request_streaming.find(rid);
            if (it != _request_streaming.end() && it->second) {
                String token(sig.text.c_str());
                emit_signal("token_generated", rid, token);
                emit_signal("generate_text_updated", token); // @deprecated compat
            }
            break;
        }
        case Chorus::EventType::Stop: {
            String full_text(_text_accumulator[rid].c_str());
            emit_signal("generation_complete", rid, full_text);
            emit_signal("generate_text_finished", full_text); // @deprecated compat
            _text_accumulator.erase(rid);
            _request_streaming.erase(rid);
            break;
        }
        case Chorus::EventType::Error: {
            String msg(sig.text.c_str());
            emit_signal("generation_error", rid, msg);
            emit_signal("generate_text_error", msg); // @deprecated compat
            _text_accumulator.erase(rid);
            _request_streaming.erase(rid);
            break;
        }
        default:
            break;
        }
    }
}

// Blocks the calling thread until generation completes. Only used by deprecated sync methods.
String GodotChorus::_generate_sync(const String& prompt, const String& grammar) {
    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;
    std::string result;

    Chorus::GenerationConfig cfg = _default_gen_config;
    if (!grammar.is_empty())
        cfg.grammar = grammar.utf8().get_data();

    Chorus::ChorusRequest req;
    req.id = _next_request_id.fetch_add(1);
    req.prompt = prompt.utf8().get_data();
    req.gen_config = cfg;
    req.on_event = [&](Chorus::ChorusSignal& sig) {
        if (sig.type == Chorus::EventType::Token) {
            result += sig.text;
        } else if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error) {
            std::unique_lock<std::mutex> lock(done_mutex);
            done = true;
            done_cv.notify_one();
        }
    };

    _engine.submit_request(req);

    std::unique_lock<std::mutex> lock(done_mutex);
    done_cv.wait(lock, [&] { return done; });

    return String(result.c_str());
}

// ===========================================================================
// Properties
// ===========================================================================

void GodotChorus::set_model_path(const String& path) {
    _chorus_config.model_path = path.utf8().get_data();
}
String GodotChorus::get_model_path() const {
    return String(_chorus_config.model_path.c_str());
}

void GodotChorus::set_context_size(int32_t size) {
    _chorus_config.context_size = size;
}
int32_t GodotChorus::get_context_size() const {
    return _chorus_config.context_size;
}

void GodotChorus::set_thread_count(int32_t count) {
    _chorus_config.thread_count = count;
}
int32_t GodotChorus::get_thread_count() const {
    return _chorus_config.thread_count;
}

void GodotChorus::set_use_gpu(bool use) {
    _chorus_config.use_gpu = use;
}
bool GodotChorus::get_use_gpu() const {
    return _chorus_config.use_gpu;
}

void GodotChorus::set_gpu_layers(int32_t layers) {
    _chorus_config.gpu_layers = layers;
}
int32_t GodotChorus::get_gpu_layers() const {
    return _chorus_config.gpu_layers;
}

// ===========================================================================
// model management
// @deprecated
// ===========================================================================

void GodotChorus::unload_model() {
    UtilityFunctions::push_warning("[Chorus] unload_model() is deprecated. Use stop_all() instead.");
    stop_all();
}

bool GodotChorus::is_model_loaded() const {
    UtilityFunctions::push_warning("[Chorus] is_model_loaded() is deprecated. Use is_loaded() instead.");
    return is_loaded();
}

// ===========================================================================
// async generation
// @deprecated
// ===========================================================================

Error GodotChorus::generate_text_async(const String& prompt, const String& grammar, const String& json) {
    UtilityFunctions::push_warning(
        "[Chorus] generate_text_async() is deprecated. Use generate({prompt=..., stream=true}) instead."
    );
    if (!json.is_empty())
        UtilityFunctions::push_warning(
            "[Chorus] The 'json' parameter has no equivalent in the new API and will be ignored."
        );

    Chorus::GenerationConfig cfg = _default_gen_config;
    if (!grammar.is_empty())
        cfg.grammar = grammar.utf8().get_data();

    // stream=true preserves old behavior of emitting generate_text_updated per token.
    int64_t id = _submit(prompt, cfg, /*streaming=*/true, /*priority=*/0);
    return id >= 0 ? Error::OK : Error::FAILED;
}

Error GodotChorus::generate_chat_async(const String& prompt, const String& grammar, const String& json) {
    UtilityFunctions::push_warning(
        "[Chorus] generate_chat_async() is deprecated. Use generate({prompt=..., stream=true}) instead. Note: "
        "conversational context is not yet maintained in the new API."
    );
    if (!json.is_empty())
        UtilityFunctions::push_warning(
            "[Chorus] The 'json' parameter has no equivalent in the new API and will be ignored."
        );

    Chorus::GenerationConfig cfg = _default_gen_config;
    if (!grammar.is_empty())
        cfg.grammar = grammar.utf8().get_data();

    int64_t id = _submit(prompt, cfg, /*streaming=*/true, /*priority=*/0);
    return id >= 0 ? Error::OK : Error::FAILED;
}

// ===========================================================================
// sync generation (blocking)
// @deprecated
// ===========================================================================

String GodotChorus::generate_text(const String& prompt, const String& grammar, const String& /*json*/) {
    UtilityFunctions::push_warning(
        "[Chorus] generate_text() is deprecated and BLOCKS the game thread. Use generate({prompt=..., stream=false}) "
        "and await generation_complete instead."
    );
    if (!_engine.is_initialized()) {
        UtilityFunctions::push_error("[Chorus] Model not loaded.");
        return "";
    }
    return _generate_sync(prompt, grammar);
}

String GodotChorus::generate_chat(const String& prompt, const String& grammar, const String& /*json*/) {
    UtilityFunctions::push_warning(
        "[Chorus] generate_chat() is deprecated and BLOCKS the game thread. Use generate({prompt=..., stream=false}) "
        "and await generation_complete instead. Note: conversational context is not yet maintained in the new API."
    );
    if (!_engine.is_initialized()) {
        UtilityFunctions::push_error("[Chorus] Model not loaded.");
        return "";
    }
    return _generate_sync(prompt, grammar);
}

// ===========================================================================
// control
// @deprecated
// ===========================================================================

void GodotChorus::stop_generate_text() {
    UtilityFunctions::push_warning("[Chorus] stop_generate_text() is deprecated. Use stop_all() instead.");
    stop_all();
}

bool GodotChorus::is_running() const {
    UtilityFunctions::push_warning("[Chorus] is_running() is deprecated. Use is_loaded() instead.");
    return is_loaded();
}

void GodotChorus::reset_context() {
    UtilityFunctions::push_warning(
        "[Chorus] reset_context() is deprecated. Conversational context management is not yet implemented in the new "
        "API."
    );
}

// ===========================================================================
// embeddings (not yet implemented)
// @deprecated
// ===========================================================================

PackedFloat32Array GodotChorus::compute_embedding(const String& /*prompt*/) {
    UtilityFunctions::push_warning(
        "[Chorus] compute_embedding() is deprecated and embeddings are not yet implemented in the new API. Returning "
        "empty array."
    );
    return PackedFloat32Array();
}

Error GodotChorus::compute_embedding_async(const String& /*prompt*/) {
    UtilityFunctions::push_warning(
        "[Chorus] compute_embedding_async() is deprecated and embeddings are not yet implemented in the new API."
    );
    return Error::FAILED;
}

// ===========================================================================
// Utility Functions
// ===========================================================================

float GodotChorus::similarity_cos(PackedFloat32Array array1, PackedFloat32Array array2) const {
    if (array1.size() != array2.size() || array1.is_empty()) {
        UtilityFunctions::push_error("[Chorus] similarity_cos: arrays must be non-empty and the same size.");
        return 0.0f;
    }
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (int i = 0; i < array1.size(); ++i) {
        dot += (double)array1[i] * (double)array2[i];
        norm1 += (double)array1[i] * (double)array1[i];
        norm2 += (double)array2[i] * (double)array2[i];
    }
    if (norm1 == 0.0 || norm2 == 0.0)
        return 0.0f;
    return (float)(dot / (std::sqrt(norm1) * std::sqrt(norm2)));
}

// ===========================================================================
// Properties
// @deprecated
// ===========================================================================

void GodotChorus::set_n_predict(int32_t n) {
    UtilityFunctions::push_warning(
        "[Chorus] n_predict is deprecated. Pass max_tokens in the generate() request dict instead."
    );
    _default_gen_config.max_tokens = n;
}
int32_t GodotChorus::get_n_predict() const {
    return _default_gen_config.max_tokens;
}

void GodotChorus::set_temperature(float t) {
    UtilityFunctions::push_warning(
        "[Chorus] temperature is deprecated. Pass temperature in the generate() request dict instead."
    );
    _default_gen_config.temperature = t;
}
float GodotChorus::get_temperature() const {
    return _default_gen_config.temperature;
}

void GodotChorus::set_top_k(int32_t k) {
    UtilityFunctions::push_warning("[Chorus] top_k is deprecated. Pass top_k in the generate() request dict instead.");
    _default_gen_config.top_k = k;
}
int32_t GodotChorus::get_top_k() const {
    return _default_gen_config.top_k;
}

void GodotChorus::set_top_p(float p) {
    UtilityFunctions::push_warning("[Chorus] top_p is deprecated. Pass top_p in the generate() request dict instead.");
    _default_gen_config.top_p = p;
}
float GodotChorus::get_top_p() const {
    return _default_gen_config.top_p;
}

void GodotChorus::set_penalty_repeat(float r) {
    UtilityFunctions::push_warning(
        "[Chorus] penalty_repeat is deprecated. Pass repeat_penalty in the generate() request dict instead."
    );
    _default_gen_config.repeat_penalty = r;
}
float GodotChorus::get_penalty_repeat() const {
    return _default_gen_config.repeat_penalty;
}

void GodotChorus::set_seed(int32_t s) {
    UtilityFunctions::push_warning("[Chorus] seed is deprecated. Pass seed in the generate() request dict instead.");
    _default_gen_config.seed = (uint32_t)s;
}
int32_t GodotChorus::get_seed() const {
    return (int32_t)_default_gen_config.seed;
}

void GodotChorus::set_n_ctx(int32_t n) {
    UtilityFunctions::push_warning("[Chorus] n_ctx is deprecated. Use the context_size property instead.");
    _chorus_config.context_size = n;
}
int32_t GodotChorus::get_n_ctx() const {
    return _chorus_config.context_size;
}

void GodotChorus::set_n_gpu_layers(int32_t n) {
    UtilityFunctions::push_warning("[Chorus] n_gpu_layers is deprecated. Use the gpu_layers property instead.");
    _chorus_config.gpu_layers = n;
}
int32_t GodotChorus::get_n_gpu_layers() const {
    return _chorus_config.gpu_layers;
}

// ===========================================================================
// Deprecated properties.
// No equivalent in new API (warn + no-op setters)
// @deprecated
// ===========================================================================

void GodotChorus::set_ignore_eos(bool /*v*/) {
    UtilityFunctions::push_warning("[Chorus] ignore_eos has no equivalent in the new API and will be ignored.");
}
bool GodotChorus::get_ignore_eos() const {
    return false;
}

void GodotChorus::set_penalty_last_n(int32_t /*n*/) {
    UtilityFunctions::push_warning("[Chorus] penalty_last_n has no equivalent in the new API and will be ignored.");
}
int32_t GodotChorus::get_penalty_last_n() const {
    return 0;
}

void GodotChorus::set_chat_template(const String& /*s*/) {
    UtilityFunctions::push_warning("[Chorus] chat_template has no equivalent in the new API and will be ignored.");
}
String GodotChorus::get_chat_template() const {
    return "";
}

void GodotChorus::set_n_batch(int32_t /*n*/) {
    UtilityFunctions::push_warning("[Chorus] n_batch has no equivalent in the new API and will be ignored.");
}
int32_t GodotChorus::get_n_batch() const {
    return 0;
}

void GodotChorus::set_main_gpu(int32_t /*n*/) {
    UtilityFunctions::push_warning("[Chorus] main_gpu has no equivalent in the new API and will be ignored.");
}
int32_t GodotChorus::get_main_gpu() const {
    return 0;
}

// ===========================================================================
// Bindings
// ===========================================================================

void GodotChorus::_bind_methods() {
    // --- Signals ---
    ADD_SIGNAL(
        MethodInfo("token_generated", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::STRING, "token"))
    );
    ADD_SIGNAL(MethodInfo(
        "generation_complete", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::STRING, "full_text")
    ));
    ADD_SIGNAL(MethodInfo(
        "generation_error", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::STRING, "message")
    ));

    // --- Signals (deprecated compat) ---
    // @deprecated
    ADD_SIGNAL(MethodInfo("generate_text_updated", PropertyInfo(Variant::STRING, "new_text")));
    ADD_SIGNAL(MethodInfo("generate_text_finished", PropertyInfo(Variant::STRING, "full_text")));
    ADD_SIGNAL(MethodInfo("generate_text_error", PropertyInfo(Variant::STRING, "error_text")));
    ADD_SIGNAL(MethodInfo("embedding_computed", PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "embedding")));
    ADD_SIGNAL(MethodInfo("embedding_failed", PropertyInfo(Variant::STRING, "error_message")));

    // --- Core methods ---
    ClassDB::bind_method(D_METHOD("load_model"), &GodotChorus::load_model);
    ClassDB::bind_method(D_METHOD("stop_all"), &GodotChorus::stop_all);
    ClassDB::bind_method(D_METHOD("is_loaded"), &GodotChorus::is_loaded);
    ClassDB::bind_method(D_METHOD("generate", "request"), &GodotChorus::generate);

    // --- Properties ---
    ClassDB::bind_method(D_METHOD("set_model_path", "path"), &GodotChorus::set_model_path);
    ClassDB::bind_method(D_METHOD("get_model_path"), &GodotChorus::get_model_path);
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "*.gguf"), "set_model_path", "get_model_path"
    );

    ClassDB::bind_method(D_METHOD("set_context_size", "size"), &GodotChorus::set_context_size);
    ClassDB::bind_method(D_METHOD("get_context_size"), &GodotChorus::get_context_size);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "context_size", PROPERTY_HINT_RANGE, "128,65536,128"),
        "set_context_size",
        "get_context_size"
    );

    ClassDB::bind_method(D_METHOD("set_thread_count", "count"), &GodotChorus::set_thread_count);
    ClassDB::bind_method(D_METHOD("get_thread_count"), &GodotChorus::get_thread_count);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "thread_count", PROPERTY_HINT_RANGE, "1,32,1"),
        "set_thread_count",
        "get_thread_count"
    );

    ClassDB::bind_method(D_METHOD("set_use_gpu", "use"), &GodotChorus::set_use_gpu);
    ClassDB::bind_method(D_METHOD("get_use_gpu"), &GodotChorus::get_use_gpu);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_gpu"), "set_use_gpu", "get_use_gpu");

    ClassDB::bind_method(D_METHOD("set_gpu_layers", "layers"), &GodotChorus::set_gpu_layers);
    ClassDB::bind_method(D_METHOD("get_gpu_layers"), &GodotChorus::get_gpu_layers);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "gpu_layers", PROPERTY_HINT_RANGE, "0,999,1"), "set_gpu_layers", "get_gpu_layers"
    );

    // --- Deprecated methods ---
    // @deprecated
    ClassDB::bind_method(D_METHOD("unload_model"), &GodotChorus::unload_model);
    ClassDB::bind_method(D_METHOD("is_model_loaded"), &GodotChorus::is_model_loaded);
    ClassDB::bind_method(D_METHOD("stop_generate_text"), &GodotChorus::stop_generate_text);
    ClassDB::bind_method(D_METHOD("is_running"), &GodotChorus::is_running);
    ClassDB::bind_method(D_METHOD("reset_context"), &GodotChorus::reset_context);

    ClassDB::bind_method(
        D_METHOD("generate_text_async", "prompt", "grammar", "json"),
        &GodotChorus::generate_text_async,
        DEFVAL(""),
        DEFVAL("")
    );
    ClassDB::bind_method(
        D_METHOD("generate_chat_async", "prompt", "grammar", "json"),
        &GodotChorus::generate_chat_async,
        DEFVAL(""),
        DEFVAL("")
    );
    ClassDB::bind_method(
        D_METHOD("generate_text", "prompt", "grammar", "json"), &GodotChorus::generate_text, DEFVAL(""), DEFVAL("")
    );
    ClassDB::bind_method(
        D_METHOD("generate_chat", "prompt", "grammar", "json"), &GodotChorus::generate_chat, DEFVAL(""), DEFVAL("")
    );

    ClassDB::bind_method(D_METHOD("compute_embedding", "prompt"), &GodotChorus::compute_embedding);
    ClassDB::bind_method(D_METHOD("compute_embedding_async", "prompt"), &GodotChorus::compute_embedding_async);
    ClassDB::bind_method(D_METHOD("similarity_cos", "array1", "array2"), &GodotChorus::similarity_cos);

// --- Deprecated properties (with real backing) ---
#define BIND_DEPRECATED_PROP(m_name, m_type)                                                                           \
    ClassDB::bind_method(D_METHOD("set_" #m_name, #m_name), &GodotChorus::set_##m_name);                               \
    ClassDB::bind_method(D_METHOD("get_" #m_name), &GodotChorus::get_##m_name);                                        \
    ADD_PROPERTY(PropertyInfo(m_type, #m_name), "set_" #m_name, "get_" #m_name);

    BIND_DEPRECATED_PROP(n_predict, Variant::INT)
    BIND_DEPRECATED_PROP(temperature, Variant::FLOAT)
    BIND_DEPRECATED_PROP(top_k, Variant::INT)
    BIND_DEPRECATED_PROP(top_p, Variant::FLOAT)
    BIND_DEPRECATED_PROP(penalty_repeat, Variant::FLOAT)
    BIND_DEPRECATED_PROP(seed, Variant::INT)
    BIND_DEPRECATED_PROP(n_ctx, Variant::INT)
    BIND_DEPRECATED_PROP(n_gpu_layers, Variant::INT)

    // --- Deprecated properties (warn + no-op) ---
    BIND_DEPRECATED_PROP(ignore_eos, Variant::BOOL)
    BIND_DEPRECATED_PROP(penalty_last_n, Variant::INT)
    BIND_DEPRECATED_PROP(chat_template, Variant::STRING)
    BIND_DEPRECATED_PROP(n_batch, Variant::INT)
    BIND_DEPRECATED_PROP(main_gpu, Variant::INT)

#undef BIND_DEPRECATED_PROP
}

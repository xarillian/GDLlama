#include "gdllama.hpp"
#include "conversion.hpp"
#include "logging_utils.hpp"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/mutex_lock.hpp>
#include <godot_cpp/variant/callable.hpp>

// TODO: All of the 'region' blocks indicate we can split this file up.

namespace godot {
    GDLlama::GDLlama() {
        controller = std::make_unique<LlamaController>();
        
        params = common_params{};
        params.embedding = true;

        generate_text_thread.instantiate();
        generation_mutex.instantiate();
    }

    GDLlama::~GDLlama() {
        controller->unload_model();
    }

    void GDLlama::_exit_tree() {
        if (is_running()) {
            stop_generate_text();
            generate_text_thread->wait_to_finish();
        }
        unload_model();
    }

    Error GDLlama::load_model() {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return controller->load_model(params);
    }

    void GDLlama::unload_model() {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        controller->unload_model();
    }

    bool GDLlama::is_model_loaded() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return controller->is_model_loaded();
    }

    String GDLlama::_generate(
        String prompt,
        String grammar,
        String json,
        bool is_conversational,
        std::string* error_msg
    ) {
        auto on_update = [this](const std::string& text_chunk) {
            this->_on_generate_text_update(text_chunk);
        };

        std::string s_prompt = string_gd_to_std(prompt);
        std::string s_grammar = string_gd_to_std(grammar);
        std::string s_json = string_gd_to_std(json);
        std::string internal_error; 

        std::string result = controller->start_generation(
            params,
            s_prompt, 
            s_grammar,
            s_json,
            is_conversational,
            on_update,
            &internal_error
        );

        if (!internal_error.empty()) {
            GDLOG_ERROR("Generation failed: " + internal_error);
            if (error_msg) *error_msg = internal_error;
            return "";
        }

        return string_std_to_gd(result);
    }

    PackedFloat32Array GDLlama::_compute_embedding(
        godot::String prompt,
        std::string* error_msg
    ) {
        PackedFloat32Array result;
        std::string internal_error;
        
        std::vector<float> embd_vec = controller->generate_embedding(
            params, 
            string_gd_to_std(prompt),
            &internal_error
        );

        if (!internal_error.empty()) {
            GDLOG_ERROR("_compute_embedding failed: " + internal_error);
            if (error_msg) *error_msg = internal_error;
            return {};
        }

        result.resize(embd_vec.size());
        for (size_t i = 0; i < embd_vec.size(); ++i) {
            result.set(i, embd_vec[i]);
        }

        return result;
    }

    // region: Asynchronous Task Workers & Launchers

    Error GDLlama::_call_async_process(Callable callable) {
        if (is_thread_busy) {
            GDLOG_ERROR("An async process is already in progress.");
            return FAILED;
        }

        if (generate_text_thread.is_valid() && generate_text_thread->is_started()) {
            generate_text_thread->wait_to_finish();
        }

        _mark_thread_busy();
        return generate_text_thread->start(callable);
    }

    void GDLlama::_generation_task(String prompt, String grammar, String json, bool is_conversational) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        std::string error_msg;
        String result = _generate(prompt, grammar, json, is_conversational, &error_msg);
    
        if (!error_msg.empty()) {
            callable_mp(this, &GDLlama::_on_generate_text_error).call_deferred(string_std_to_gd(error_msg));
        } else {
            callable_mp(this, &GDLlama::_async_generation_completed).call_deferred(result);
        }
    }

    void GDLlama::_embedding_task(String prompt) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        
        std::string error_msg;
        PackedFloat32Array result = _compute_embedding(prompt, &error_msg);

        if (!error_msg.empty()) {
            callable_mp(this, &GDLlama::_on_embedding_failed)
                .call_deferred(string_std_to_gd(error_msg));
        } else {
            callable_mp(this, &GDLlama::_async_embedding_completed).call_deferred(result);
        }
    }
    // endregion: Asynchronous Task Workers & Launchers

    // region: Threading & State
    
    void GDLlama::_mark_thread_idle() {
        is_thread_busy = false;
    }

    void GDLlama::_mark_thread_busy() {
        is_thread_busy = true;
    }

    // endregion: Threading & State

    // region: Godot Bindings
    void GDLlama::_bind_methods() {
        _bind_model_methods();
        _bind_generation_methods();
        _bind_embedding_methods();
        _bind_properties();
        _bind_signals();
    }

    void GDLlama::_bind_model_methods() {
        ClassDB::bind_method(D_METHOD("load_model"), &GDLlama::load_model);
        ClassDB::bind_method(D_METHOD("unload_model"), &GDLlama::unload_model);
        ClassDB::bind_method(D_METHOD("is_model_loaded"), &GDLlama::is_model_loaded);
        ClassDB::bind_method(D_METHOD("set_model_path", "p_model_path"), &GDLlama::set_model_path);
        ClassDB::bind_method(D_METHOD("get_model_path"), &GDLlama::get_model_path);
    
        ClassDB::add_property("GDLlama", PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE), "set_model_path", "get_model_path");
    }

    void GDLlama::_bind_generation_methods() {
        ClassDB::bind_method(D_METHOD("generate_text", "prompt", "grammar", "json"), &GDLlama::generate_text, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("generate_text_async", "prompt", "grammar", "json"), &GDLlama::generate_text_async, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("generate_chat", "prompt", "grammar", "json"), &GDLlama::generate_chat, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("generate_chat_async", "prompt", "grammar", "json"), &GDLlama::generate_chat_async, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("reset_context"), &GDLlama::reset_context);
        ClassDB::bind_method(D_METHOD("stop_generate_text"), &GDLlama::stop_generate_text);

        ClassDB::bind_method(D_METHOD("is_running"), &GDLlama::is_running);
    }

    void GDLlama::_bind_embedding_methods() {
        ClassDB::bind_method(D_METHOD("compute_embedding", "prompt"), &GDLlama::compute_embedding);
        ClassDB::bind_method(D_METHOD("compute_embedding_async", "prompt"), &GDLlama::compute_embedding_async);
        ClassDB::bind_method(D_METHOD("similarity_cos", "array1", "array2"), &GDLlama::similarity_cos);
    }

    void GDLlama::_bind_properties() {
        #define BIND_GDL_PROPERTY(m_name, m_type) \
            ClassDB::bind_method(D_METHOD("set_" #m_name, #m_name), &GDLlama::set_##m_name); \
            ClassDB::bind_method(D_METHOD("get_" #m_name), &GDLlama::get_##m_name); \
            ClassDB::add_property("GDLlama", PropertyInfo(m_type, #m_name), "set_" #m_name, "get_" #m_name);

        #define BIND_GDL_PROPERTY_HINT(m_name, m_type, m_hint) \
            ClassDB::bind_method(D_METHOD("set_" #m_name, "p_" #m_name), &GDLlama::set_##m_name); \
            ClassDB::bind_method(D_METHOD("get_" #m_name), &GDLlama::get_##m_name); \
            ClassDB::add_property("GDLlama", PropertyInfo(m_type, #m_name, m_hint), "set_" #m_name, "get_" #m_name);

        BIND_GDL_PROPERTY(n_predict, Variant::INT);
        BIND_GDL_PROPERTY(temperature, Variant::FLOAT);
        BIND_GDL_PROPERTY(top_k, Variant::INT);
        BIND_GDL_PROPERTY(top_p, Variant::FLOAT);
        BIND_GDL_PROPERTY(ignore_eos, Variant::BOOL);
        BIND_GDL_PROPERTY(penalty_repeat, Variant::FLOAT);
        BIND_GDL_PROPERTY(penalty_last_n, Variant::INT);
        BIND_GDL_PROPERTY(chat_template, Variant::STRING);
        BIND_GDL_PROPERTY(n_batch, Variant::INT);
        BIND_GDL_PROPERTY(n_gpu_layers, Variant::INT);
        BIND_GDL_PROPERTY(n_ctx, Variant::INT);
        BIND_GDL_PROPERTY(main_gpu, Variant::INT);
        BIND_GDL_PROPERTY(seed, Variant::INT);

        #undef BIND_GDL_PROPERTY
        #undef BIND_GDL_PROPERTY_HINT
    }

    void GDLlama::_bind_signals() {
        ADD_SIGNAL(MethodInfo("generate_text_updated", PropertyInfo(Variant::STRING, "new_text")));
        ADD_SIGNAL(MethodInfo("generate_text_finished", PropertyInfo(Variant::STRING, "full_text")));
        ADD_SIGNAL(MethodInfo("generate_text_error", PropertyInfo(Variant::STRING, "error_text")));
        ADD_SIGNAL(MethodInfo("embedding_computed", PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "embedding")));
        ADD_SIGNAL(MethodInfo("embedding_failed", PropertyInfo(Variant::STRING, "error_message")));
    }

    // endregion: Godot Bindings

    // region: signals

    void GDLlama::_on_generate_text_update(std::string text_chunk) {
        call_deferred(
            "emit_signal",
            "generate_text_updated",
            string_std_to_gd(text_chunk)
        );
    }

    void GDLlama::_on_generate_text_error(String error_msg) {
        _mark_thread_idle();
        emit_signal("generate_text_error", error_msg);
    }

    void GDLlama::_async_generation_completed(String result) {
        _mark_thread_idle();
        emit_signal("generate_text_finished", result);
        GDLOG_DEBUG("Async Signal emitted.");
    }

    void GDLlama::_async_embedding_completed(PackedFloat32Array result) {
        if (result.is_empty()) {
            _on_embedding_failed(string_std_to_gd("Embedding result is empty."));
        } else {
            _mark_thread_idle();
            emit_signal("embedding_computed", result);
        }
        GDLOG_DEBUG("Async embedding signal emitted.");
    }

    void GDLlama::_on_embedding_failed(String error_msg) {
        _mark_thread_idle();
        emit_signal("embedding_failed", error_msg);
    }

    // endregion: signals

    String GDLlama::generate_text(String prompt, String grammar, String json) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return _generate(prompt, grammar, json, false);
    }

    Error GDLlama::generate_text_async(String prompt, String grammar, String json) {
        Callable c = callable_mp(
            this,
            &GDLlama::_generation_task
        ).bind(prompt, grammar, json, false);
        return _call_async_process(c);
    }

    String GDLlama::generate_chat(String prompt, String grammar, String json) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return _generate(prompt, grammar, json, true);
    }

    Error GDLlama::generate_chat_async(String prompt, String grammar, String json) {
        Callable c = callable_mp(
            this,
            &GDLlama::_generation_task
        ).bind(prompt, grammar, json, true);
        return _call_async_process(c);
    }

    bool GDLlama::is_running() const {
        return generate_text_thread->is_alive();
    }

    void GDLlama::reset_context() {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        controller->reset_context();
    }

    void GDLlama::stop_generate_text() {
        if (is_running()) {
            controller->stop_generation();
            GDLOG_INFO("Stop signal sent to generation thread.");
        }
    }

    PackedFloat32Array GDLlama::compute_embedding(godot::String prompt) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return _compute_embedding(prompt);
    }

    Error GDLlama::compute_embedding_async(String prompt) {
        Callable c = callable_mp(this, &GDLlama::_embedding_task).bind(prompt);
        return _call_async_process(c);
    }

    float GDLlama::similarity_cos(PackedFloat32Array array1, PackedFloat32Array array2) {
        godot::MutexLock lock(*(generation_mutex.ptr()));

        if (array1.size() != array2.size() || array1.is_empty()) {
            GDLOG_ERROR("Cannot compute similarity: arrays have different sizes or are empty.");
            return 0.0f;
        }
        double dot_product = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        for (int i = 0; i < array1.size(); ++i) {
            dot_product += array1[i] * array2[i];
            norm1 += array1[i] * array1[i];
            norm2 += array2[i] * array2[i];
        }
        if (norm1 == 0.0 || norm2 == 0.0) return 0.0f;
        return dot_product / (sqrt(norm1) * sqrt(norm2));
    }

    // region: Properties

    void GDLlama::set_model_path(const godot::String p_model_path) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting model_path to " + string_gd_to_std(p_model_path));
        params.model.path = string_gd_to_std(p_model_path.trim_prefix("res://"));
    }

    godot::String GDLlama::get_model_path() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return string_std_to_gd(params.model.path);
    }

    void GDLlama::set_n_predict(int n_predict) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting n_predict to " + std::to_string(n_predict));
        params.n_predict = n_predict;
    }

    int GDLlama::get_n_predict() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.n_predict;
    }

    void GDLlama::set_temperature(float temperature) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting temperature to " + std::to_string(temperature));
        params.sampling.temp = temperature;
    }

    float GDLlama::get_temperature() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.sampling.temp;
    }

    void GDLlama::set_top_k(int top_k) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting top_k to " + std::to_string(top_k));
        params.sampling.top_k = top_k;
    }

    int GDLlama::get_top_k() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.sampling.top_k;
    }

    void GDLlama::set_top_p(float top_p) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting top_p to " + std::to_string(top_p));
        params.sampling.top_p = top_p;
    }

    float GDLlama::get_top_p() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.sampling.top_p;
    }

    void GDLlama::set_ignore_eos(bool p_ignore_eos) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting ignore_eos to " + std::to_string(p_ignore_eos));
        params.sampling.ignore_eos = p_ignore_eos;
    }

    bool GDLlama::get_ignore_eos() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.sampling.ignore_eos;
    }

    void GDLlama::set_penalty_repeat(float p_penalty_repeat) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        params.sampling.penalty_repeat = p_penalty_repeat;
    }

    float GDLlama::get_penalty_repeat() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.sampling.penalty_repeat;
    }

    void GDLlama::set_penalty_last_n(int p_penalty_last_n) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        params.sampling.penalty_last_n = p_penalty_last_n;
    }

    int GDLlama::get_penalty_last_n() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.sampling.penalty_last_n;
    }

    void GDLlama::set_chat_template(const String &p_chat_template) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        params.chat_template = string_gd_to_std(p_chat_template);
    }

    String GDLlama::get_chat_template() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return string_std_to_gd(params.chat_template);
    }

    void GDLlama::set_n_batch(int p_n_batch) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting n_batch to " + std::to_string(p_n_batch));
        params.n_batch = p_n_batch;
    }

    int GDLlama::get_n_batch() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.n_batch;
    }

    void GDLlama::set_n_gpu_layers(int p_n_gpu_layers) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting n_gpu_layers to " + std::to_string(p_n_gpu_layers));
        params.n_gpu_layers = p_n_gpu_layers;
    }

    int GDLlama::get_n_gpu_layers() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.n_gpu_layers;
    }

    void GDLlama::set_n_ctx(int p_n_ctx) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting n_ctx to " + std::to_string(p_n_ctx));
        params.n_ctx = p_n_ctx;
    }

    int GDLlama::get_n_ctx() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.n_ctx;
    }

    void GDLlama::set_main_gpu(int p_main_gpu) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting main_gpu to " + std::to_string(p_main_gpu));
        params.main_gpu = p_main_gpu;
    }

    int GDLlama::get_main_gpu() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.main_gpu;
    }

    void GDLlama::set_seed(int p_seed) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        GDLOG_DEBUG("Setting seed to " + std::to_string(p_seed));
        params.sampling.seed = p_seed;
    }

    int GDLlama::get_seed() const {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        return params.sampling.seed;
    }

    // endregion: Properties
}  // namespace godot
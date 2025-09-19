#include "gdllama.hpp"
#include "conversion.hpp"
#include "logging_utils.hpp"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/mutex_lock.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {
    void GDLlama::_bind_methods() {
        #define BIND_GDL_PROPERTY(m_name, m_type) \
            ClassDB::bind_method(D_METHOD("set_" #m_name, #m_name), &GDLlama::set_##m_name); \
            ClassDB::bind_method(D_METHOD("get_" #m_name), &GDLlama::get_##m_name); \
            ClassDB::add_property("GDLlama", PropertyInfo(m_type, #m_name), "set_" #m_name, "get_" #m_name);

        #define BIND_GDL_PROPERTY_HINT(m_name, m_type, m_hint) \
            ClassDB::bind_method(D_METHOD("set_" #m_name, "p_" #m_name), &GDLlama::set_##m_name); \
            ClassDB::bind_method(D_METHOD("get_" #m_name), &GDLlama::get_##m_name); \
            ClassDB::add_property("GDLlama", PropertyInfo(m_type, #m_name, m_hint), "set_" #m_name, "get_" #m_name);

        // ## Model Management ##
        ClassDB::bind_method(D_METHOD("load_model"), &GDLlama::load_model);
        ClassDB::bind_method(D_METHOD("unload_model"), &GDLlama::unload_model);
        ClassDB::bind_method(D_METHOD("is_model_loaded"), &GDLlama::is_model_loaded);
        ClassDB::bind_method(D_METHOD("set_model_path", "p_model_path"), &GDLlama::set_model_path);
        ClassDB::bind_method(D_METHOD("get_model_path"), &GDLlama::get_model_path);
    
        ClassDB::add_property("GDLlama", PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE), "set_model_path", "get_model_path");

        // ## Text Generation Methods ##
        ClassDB::bind_method(D_METHOD("generate_text", "prompt", "grammar", "json"), &GDLlama::generate_text, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("generate_text_async", "prompt", "grammar", "json"), &GDLlama::generate_text_async, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("generate_chat", "prompt", "grammar", "json"), &GDLlama::generate_chat, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("generate_chat_async", "prompt", "grammar", "json"), &GDLlama::generate_chat_async, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("reset_context"), &GDLlama::reset_context);
        ClassDB::bind_method(D_METHOD("stop_generate_text"), &GDLlama::stop_generate_text);

        // ## Generation Parameters ##
        BIND_GDL_PROPERTY(n_predict, Variant::INT);
        BIND_GDL_PROPERTY(temperature, Variant::FLOAT);
        BIND_GDL_PROPERTY(top_k, Variant::INT);
        BIND_GDL_PROPERTY(top_p, Variant::FLOAT);
        BIND_GDL_PROPERTY(ignore_eos, Variant::BOOL);
        BIND_GDL_PROPERTY(penalty_repeat, Variant::FLOAT);
        BIND_GDL_PROPERTY(penalty_last_n, Variant::INT);
        BIND_GDL_PROPERTY(chat_template, Variant::STRING);

        // ## State Checking ##
        ClassDB::bind_method(D_METHOD("is_running"), &GDLlama::is_running);

        // ## Signals ##
        ADD_SIGNAL(MethodInfo("generate_text_updated", PropertyInfo(Variant::STRING, "new_text")));
        ADD_SIGNAL(MethodInfo("generate_text_finished", PropertyInfo(Variant::STRING, "full_text")));
        ADD_SIGNAL(MethodInfo("generate_text_error", PropertyInfo(Variant::STRING, "error_text")));
    
        #undef BIND_GDL_PROPERTY
        #undef BIND_GDL_PROPERTY_HINT
    }

    GDLlama::GDLlama() {
        controller = std::make_unique<LlamaController>();
        
        params = common_params{};

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
        bool is_continuous
    ) {

        auto on_update = [this](std::string text_chunk) {
            call_deferred(
                "emit_signal",
                "generate_text_updated",
                string_std_to_gd(text_chunk)
            );
        };

        std::string s_prompt = string_gd_to_std(prompt);
        std::string s_grammar = string_gd_to_std(grammar);
        std::string s_json = string_gd_to_std(json);
        std::string result = controller->start_generation(
            params,
            s_prompt, 
            s_grammar,
            s_json,
            is_continuous,
            on_update
        );

        return string_std_to_gd(result);
    }

    Error GDLlama::_generate_async(Callable callable) {
        if (is_thread_busy) {
            GDLOG_ERROR("An async generation is already in progress.");
            return FAILED;
        }

        if (generate_text_thread.is_valid() && generate_text_thread->is_started()) {
            generate_text_thread->wait_to_finish();
        }

        _mark_thread_busy();
        return generate_text_thread->start(callable);
    }

    void GDLlama::_generation_task(String prompt, String grammar, String json, bool is_continuous) {
        godot::MutexLock lock(*(generation_mutex.ptr()));

        try {
            String result = _generate(prompt, grammar, json, is_continuous);
            Callable cleanup_callable = callable_mp(this, &GDLlama::_async_generation_completed);
            cleanup_callable.call_deferred(result);
        } catch (const std::runtime_error& e) {
            String error_msg = string_std_to_gd(e.what());
            call_deferred("emit_signal", "generate_text_error", error_msg);
            callable_mp(this, &GDLlama::_mark_thread_idle).call_deferred();
            GDLOG_ERROR("Async generation failed: " + std::string(e.what()));
        }
    }

    void GDLlama::_async_generation_completed(String result) {
        _mark_thread_idle();
        emit_signal("generate_text_finished", result);
        GDLOG_DEBUG("Async Signal emitted.");
    }

    void GDLlama::_mark_thread_idle() {
        is_thread_busy = false;
    }

    void GDLlama::_mark_thread_busy() {
        is_thread_busy = true;
    }

    String GDLlama::generate_text(String prompt, String grammar, String json) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        try {
            String result = _generate(prompt, grammar, json, false);
            return result;
        } catch (const std::runtime_error& e) {
            GDLOG_ERROR("generate_text failed: " + std::string(e.what()));
            return "";
        }
    }

    Error GDLlama::generate_text_async(String prompt, String grammar, String json) {
        Callable c = callable_mp(
            this,
            &GDLlama::_generation_task
        ).bind(prompt, grammar, json, false);
        return _generate_async(c);
    }

    String GDLlama::generate_chat(String prompt, String grammar, String json) {
        godot::MutexLock lock(*(generation_mutex.ptr()));
        try {
            String result = _generate(prompt, grammar, json, true);
            return result;
        } catch (const std::runtime_error& e) {
            GDLOG_ERROR("generate_chat failed: " + std::string(e.what()));
            return "";
        }
    }

    Error GDLlama::generate_chat_async(String prompt, String grammar, String json) {
        Callable c = callable_mp(
            this,
            &GDLlama::_generation_task
        ).bind(prompt, grammar, json, true);
        return _generate_async(c);
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
}  // namespace godot
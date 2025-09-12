#include "gdllama.hpp"
#include "conversion.hpp"
#include "logging_utils.hpp"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <mutex_lock.hpp>

namespace godot {
    void GDLlama::_bind_methods() {
        // Model Management
        ClassDB::bind_method(D_METHOD("load_model"), &GDLlama::load_model);
        ClassDB::bind_method(D_METHOD("unload_model"), &GDLlama::unload_model);
        ClassDB::bind_method(D_METHOD("is_model_loaded"), &GDLlama::is_model_loaded);
        ClassDB::bind_method(D_METHOD("set_model_path", "p_model_path"), &GDLlama::set_model_path);
        ClassDB::bind_method(D_METHOD("get_model_path"), &GDLlama::get_model_path);
        ClassDB::add_property("GDLlama", PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE), "set_model_path", "get_model_path");

        // Generation Methods
        ClassDB::bind_method(D_METHOD("generate_text", "prompt", "grammar", "json"), &GDLlama::generate_text, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("generate_text_async", "prompt", "grammar", "json"), &GDLlama::generate_text_async, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("generate_chat", "prompt", "grammar", "json"), &GDLlama::generate_chat, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("generate_chat_async", "prompt", "grammar", "json"), &GDLlama::generate_chat_async, DEFVAL(""), DEFVAL(""));
        ClassDB::bind_method(D_METHOD("reset_conversation"), &GDLlama::reset_conversation);
        ClassDB::bind_method(D_METHOD("stop_generate_text"), &GDLlama::stop_generate_text);

        // Generation Parameters
        ClassDB::bind_method(D_METHOD("set_n_predict", "n_predict"), &GDLlama::set_n_predict);
        ClassDB::bind_method(D_METHOD("get_n_predict"), &GDLlama::get_n_predict);
        ClassDB::add_property("GDLlama", PropertyInfo(Variant::INT, "n_predict"), "set_n_predict", "get_n_predict");

        ClassDB::bind_method(D_METHOD("set_temperature", "temp"), &GDLlama::set_temperature);
        ClassDB::bind_method(D_METHOD("get_temperature"), &GDLlama::get_temperature);
        ClassDB::add_property("GDLlama", PropertyInfo(Variant::FLOAT, "temperature"), "set_temperature", "get_temperature");

        ClassDB::bind_method(D_METHOD("set_top_k", "top_k"), &GDLlama::set_top_k);
        ClassDB::bind_method(D_METHOD("get_top_k"), &GDLlama::get_top_k);
        ClassDB::add_property("GDLlama", PropertyInfo(Variant::INT, "top_k"), "set_top_k", "get_top_k");

        ClassDB::bind_method(D_METHOD("set_top_p", "top_p"), &GDLlama::set_top_p);
        ClassDB::bind_method(D_METHOD("get_top_p"), &GDLlama::get_top_p);
        ClassDB::add_property("GDLlama", PropertyInfo(Variant::FLOAT, "top_p"), "set_top_p", "get_top_p");

        // State Checking
        ClassDB::bind_method(D_METHOD("is_running"), &GDLlama::is_running);

        // Signals
        ADD_SIGNAL(MethodInfo("generate_text_updated", PropertyInfo(Variant::STRING, "new_text")));
        ADD_SIGNAL(MethodInfo("generate_text_finished", PropertyInfo(Variant::STRING, "full_text")));
    }

    GDLlama::GDLlama() {
        llama_state = std::make_unique<LlamaState>();
        controller = std::make_unique<LlamaController>();

        generation_mutex.instantiate();
        generate_text_thread.instantiate();

        params = common_params{};
    }

    GDLlama::~GDLlama() {
        llama_state->unload();
    }

    void GDLlama::_exit_tree() {
        if (is_running()) {
            stop_generate_text();
            generate_text_thread->wait_to_finish();
        }
        unload_model();
    }

    /*
     * @todo something is fucked up with our threads
     * An instance of `GDLlama` enters an unrecoverable state after an async generation
     * task completes. Any subsequent operation on that same instance (another async call,
     * a synchronous call like `generate_chat`, or even `unload_model`) fails,
     * typically with a "Llama failed to decode" error from the llama.cpp backend.
     * 
     * The issue appears to be a subtle thread-safety or resource lifecycle problem
     * that is not a simple data race or deadlock at our application level.
     * 
     * @note Next Steps & Hypothesis
     *
     * - Research the thread-safety guarantees of `llama.cpp`. Can a
     * single `llama_context` be passed between and used by different OS threads?
     *
     * - As an experiment, try creating a *separate, temporary `llama_context`* just for the
     * background thread inside `_generation_task`. This would be slow, but if it *fixes the 
     * crash*, it will prove that context sharing is the root cause of the problem.
     * 
     * - Look into how Godot threading works. There may be subtleties here that we are not
     * considering
     */

    Error GDLlama::load_model() {
        if (params.model.path.empty()) {
            GDLOG_ERROR("Cannot load model: model_path is not set.");
            return FAILED;
        }
        bool success = llama_state->load(params);
        return success ? OK : FAILED;
    }

    void GDLlama::unload_model() {
        llama_state->unload();
    }

    bool GDLlama::is_model_loaded() const {
        return llama_state->is_loaded();
    }

    void GDLlama::set_model_path(const String p_model_path) {
        params.model.path = string_gd_to_std(p_model_path.trim_prefix("res://"));
    }

    String GDLlama::get_model_path() const {
        return string_std_to_gd(params.model.path);
    }

    String GDLlama::_generate(
        String prompt,
        String grammar,
        String json,
        bool is_continuous,
        bool should_emit_finish_signal
    ) {
        if (!is_model_loaded()) {
            std::string err_msg = "Cannot generate text: Model is not loaded.";
            GDLOG_ERROR(err_msg);
            throw std::runtime_error(err_msg);  // @todo not this
        }

        if (!is_continuous) { reset_conversation(); }

        auto on_update = [this](std::string text_chunk) {
            call_deferred(
                "emit_signal",
                "generate_text_updated",
                string_std_to_gd(text_chunk)
            );
        };

        std::function<void(std::string)> on_finish = nullptr;
        if (should_emit_finish_signal) {
            on_finish = [this](std::string full_text) {
                call_deferred(
                    "emit_signal",
                    "generate_text_finished",
                    string_std_to_gd(full_text)
                );
            };
        }

        std::string s_prompt = string_gd_to_std(prompt);
        std::string s_grammar = string_gd_to_std(grammar);
        std::string s_json = string_gd_to_std(json);
        llama_context* ctx = llama_state->get_context();
        llama_model* model = llama_state->get_model();

        std::string result = controller->start_generation(
            model, ctx, params, s_prompt, s_grammar, s_json, on_update, on_finish
        );

        return string_std_to_gd(result);
    }

    void GDLlama::_generation_task(String prompt, String grammar, String json, bool is_continuous) {
        generation_mutex->lock();

        common_params temp_params = params;
        common_init_result temp_init = common_init_from_params(temp_params);
        llama_context* temp_ctx = temp_init.context.get();
        llama_model* temp_model = temp_init.model.get();

        _generate(prompt, grammar, json, is_continuous, true);

        generation_mutex->unlock();
    }

    Error GDLlama::_generate_async(Callable callable) {
        if (is_running()) {
            GDLOG_WARN("An async generation is already in progress.");
            return FAILED;
        }

        if (generate_text_thread->is_started()) {
            generate_text_thread->wait_to_finish();
        }

        generate_text_thread.instantiate();
        return generate_text_thread->start(callable);
    }

    String GDLlama::generate_text(String prompt, String grammar, String json) {
        generation_mutex->lock();
        String result = _generate(prompt, grammar, json, false, false);
        generation_mutex->unlock();
        return result;
    }

    Error GDLlama::generate_text_async(String prompt, String grammar, String json) {
        Callable c = callable_mp(
            this,
            &GDLlama::_generation_task
        ).bind(prompt, grammar, json, false);
        return _generate_async(c);
    }

    String GDLlama::generate_chat(String prompt, String grammar, String json) {
        generation_mutex->lock();
        String result = _generate(prompt, grammar, json, true, false);
        generation_mutex->unlock();
        return result;
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

    void GDLlama::reset_conversation() {
        if (is_model_loaded()) {
            llama_memory_clear(llama_get_memory(llama_state->get_context()), true);
            GDLOG_DEBUG("LLM context (KV cache) cleared.");
        }
    }

    void GDLlama::stop_generate_text() {
        if (is_running()) {
            controller->stop_generation();
            GDLOG_INFO("Stop signal sent to generation thread.");
        }
    }

    void GDLlama::set_n_predict(int n_predict) {
        params.n_predict = n_predict;
    }

    int GDLlama::get_n_predict() const {
        return params.n_predict;
    }

    void GDLlama::set_temperature(float temp) {
        params.sampling.temp = temp;
    }

    float GDLlama::get_temperature() const {
        return params.sampling.temp;
    }

    void GDLlama::set_top_k(int top_k) {
        params.sampling.top_k = top_k;
    }

    int GDLlama::get_top_k() const {
        return params.sampling.top_k;
    }

    void GDLlama::set_top_p(float top_p) {
        params.sampling.top_p = top_p;
    }

    float GDLlama::get_top_p() const {
        return params.sampling.top_p;
    }
}  // namespace godot
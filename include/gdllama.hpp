#ifndef GDLLAMA_HPP
#define GDLLAMA_HPP

#include "llama_controller.hpp"
#include "llama_state.hpp"
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <common/common.h>
#include <memory>

namespace godot {

class GDLlama : public Node {
    GDCLASS(GDLlama, Node)

    public:
        GDLlama();
        ~GDLlama();

        void _exit_tree() override;

        // Model Management
        /**
         * @brief Loads the model from the specified path.
         * @return OK if the model was loaded successfully.
         */
        Error load_model();
        void unload_model();
        bool is_model_loaded() const;
        String get_model_path() const;
        void set_model_path(const String p_model_path);

        // Generation Methods
        String generate_text(String prompt, String grammar = "", String json = "");
        Error generate_text_async(String prompt, String grammar = "", String json = "");
        String generate_chat(String prompt, String grammar = "", String json = "");
        Error generate_chat_async(String prompt, String grammar = "", String json = "");
        void reset_context();
        void stop_generate_text();

        // Embedding Methods
        PackedFloat32Array compute_embedding(String prompt);
        Error compute_embedding_async(String prompt);
        float similarity_cos(PackedFloat32Array array1, PackedFloat32Array array2);

        // State Checking
        bool is_running() const;

        // Generation Parameters
        void set_n_predict(int n_predict);
        int get_n_predict() const;

        void set_temperature(float temperature);
        float get_temperature() const;

        void set_top_k(int top_k);
        int get_top_k() const;

        void set_top_p(float top_p);
        float get_top_p() const;

        void set_ignore_eos(bool p_ignore_eos);
        bool get_ignore_eos() const;

        void set_penalty_repeat(float p_penalty_repeat);
        float get_penalty_repeat() const;

        void set_penalty_last_n(int p_penalty_last_n);
        int get_penalty_last_n() const;

        void set_chat_template(const String &p_chat_template);
        String get_chat_template() const;

        void set_n_batch(int p_n_batch);
        int get_n_batch() const;

        void set_n_gpu_layers(int p_n_gpu_layer);
        int get_n_gpu_layers() const;

        void set_n_ctx(int p_n_ctx);
        int get_n_ctx() const;

        void set_main_gpu(int p_main_gpu);
        int get_main_gpu() const;

        void set_seed(int  p_seed);
        int  get_seed() const;

    protected:
        static void _bind_methods();

    private:
        // Core
        std::unique_ptr<LlamaController> controller;
        common_params params;
        std::string text_generation_buffer;

        godot::String _generate(
            godot::String prompt,
            godot::String grammar,
            godot::String json,
            bool is_conversational,
            std::string* error_msg = nullptr
        );

        PackedFloat32Array _compute_embedding(godot::String prompt, std::string* error_msg = nullptr);

        // Threading & State
        godot::Ref<godot::Thread> generate_text_thread;
        mutable godot::Ref<godot::Mutex> generation_mutex;
        std::atomic<bool> is_thread_busy = false;

        void _mark_thread_idle();
        void _mark_thread_busy();

        // Asynchronous Task Workers & Launchers
        godot::Error _call_async_process(godot::Callable callable);
        void _generation_task(
            godot::String prompt,
            godot::String grammar,
            godot::String json,
            bool is_conversational
        );
        void _embedding_task(String prompt);

        // Asynchronous Callbacks & Signals
        void _async_generation_completed(godot::String result);
        void _on_generate_text_update(std::string text_chunk);
        void _on_generate_text_error(godot::String error_msg);
        void _async_embedding_completed(PackedFloat32Array result);
        void _on_embedding_failed(godot::String error_msg);

        // Godot Bindings
        static void _bind_model_methods();
        static void _bind_generation_methods();
        static void _bind_embedding_methods();
        static void _bind_properties();
        static void _bind_signals();
    };
} // namespace godot

#endif // GDLLAMA_HPP

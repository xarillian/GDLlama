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
        void reset_conversation();
        void stop_generate_text();

        // State Checking
        bool is_running() const;

        // Generation Parameters
        void set_n_predict(int n_predict);
        int get_n_predict() const;

        void set_temperature(float temp);
        float get_temperature() const;

        void set_top_k(int top_k);
        int get_top_k() const;

        void set_top_p(float top_p);
        float get_top_p() const;

    protected:
        static void _bind_methods();

    private:
        String _generate(
            String prompt,
            String grammar,
            String json,
            bool is_continuous,
            bool should_emit_finish_signal
        );
        void _generation_task(
            String prompt,
            String grammar,
            String json,
            bool is_continuous
        );
        Error _generate_async(Callable callable);

        std::unique_ptr<LlamaState> llama_state;
        std::unique_ptr<LlamaController> controller;
        common_params params;

        godot::Ref<Thread> generate_text_thread;
        godot::Ref<Mutex> generation_mutex;
        std::string text_generation_buffer;
    };
} // namespace godot

#endif GDLLAMA_HPP

#ifndef LLAMA_RUNNER_HPP
#define LLAMA_RUNNER_HPP

#include <common.h>
#include <functional>
#include <string>

class LlamaRunner {
    private:
        bool should_stop_generation;
        bool is_waiting_input;
        bool should_output_prompt;
        std::string input;
        std::function<void(ggml_log_level, const std::string&)> log_callback;

    private:
        /**
         * @brief Pre-flight check to validate parameters prior to model initialization.
         * @param params The model parameters to validate.
         * @return An error message. If there is no message, then there is no error.
         */
        std::string validate_params_for_initialization(const common_params &params);

    public:
        LlamaRunner(
            bool should_output_prompt = true,
            std::function<void(ggml_log_level, const std::string&)> log_callback
        );
        ~LlamaRunner();
        static bool file_exists(const std::string &path);
        static bool file_is_empty(const std::string &path);
        std::string llama_generate_text(
            std::string prompt,
            common_params params,
            std::function<void(std::string)> on_generate_text_updated,
            std::function<void()> on_input_wait_started,
            std::function<void(std::string)> on_generate_text_finished
        );
        void llama_stop_generate_text();
        void set_input(std::string input);
        bool get_is_waiting_input();
};

#endif //LLAMA_RUNNER_H

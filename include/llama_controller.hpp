#ifndef LLAMA_CONTROLLER_HPP
#define LLAMA_CONTROLLER_HPP

#include "llama_runner.hpp"
#include "logging_utils.hpp"
#include <mutex>
#include <functional>

class LlamaController {
    private:
        std::unique_ptr<LlamaRunner> llama_runner;
        std::mutex generate_text_mutex;
        common_params params;
        std::string reverse_prompt;

    public:
        LlamaController();
        void set_llama_runner(std::unique_ptr<LlamaRunner> runner);
        void set_reverse_prompt(const std::string& p_reverse_prompt);
        std::string generate_text_locked(
            const common_params& model_params,
            const std::string& prompt,
            const std::string& grammar,
            const std::string& json,
            std::function<void(std::string)> on_update,
            std::function<void()> on_wait_start,
            std::function<void(std::string)> on_finish
        );
};

#endif // LLAMA_CONTROLLER_HPP
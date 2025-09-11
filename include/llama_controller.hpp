#ifndef LLAMA_CONTROLLER_HPP
#define LLAMA_CONTROLLER_HPP

#include "llama_runner.hpp"
#include <mutex>
#include <functional>

class LlamaController {
    public:
        LlamaController();

        /**
         * @brief Generates text using a provided context and parameters.
         * @param model @todo
         * @param context An active llama_context from a loaded LlamaState.
         * @param params Parameters for the generation (sampling, prompt, etc.).
         * @param prompt The user's input prompt.
         * @param grammar Optional BNF grammar string to constrain generation. Empty string for no grammar.         
         * @param json Optional JSON schema to constrain generation. Will be converted to grammar
    *  *               internally. If both grammar and JSON are provided, grammar takes precedence.
         * @param on_update Callback for streaming text chunks.
         * @param on_finish Callback for when generation is complete.
         * @return The complete generated text.
         */
        std::string start_generation(
            llama_model* model,
            llama_context* context,
            common_params& params,
            const std::string& prompt,
            const std::string& grammar,
            const std::string& json,
            std::function<void(std::string)> on_update,
            std::function<void(std::string)> on_finish
        );

        /**
         * @brief Signals the LlamaRunner to stop the current generation.
         */
        void stop_generation();

    private:
        std::unique_ptr<LlamaRunner> llama_runner;
};


#endif // LLAMA_CONTROLLER_HPP
#ifndef LLAMA_RUNNER_HPP
#define LLAMA_RUNNER_HPP

#include <common.h>
#include <atomic>
#include <functional>
#include <string>

/**
 * @class LlamaRunner
 * @brief Executes the core prediction loop on a pre-loaded model context.
 */
class LlamaRunner {
    public:

        LlamaRunner();

        virtual ~LlamaRunner();

        /**
         * @brief Runs the prediction loop using an existing model and context.
         * @param model A pointer to the loaded llama_model.
         * @param ctx A pointer to the active llama_context.
         * @param params The common_params struct for this generation task.
         * @param on_generate_text_updated Callback for streaming text chunks.
         * @return The complete generated text string.
         */
        virtual std::string run_prediction(
            llama_model* model,
            llama_context* ctx,
            common_params& params,
            std::function<void(std::string)> on_generate_text_updated
        );

        void stop_generation();

        /**
         * @brief Generates an embedding vector from the provided prompt.
         * @param model A pointer to the loaded llama_model.
         * @param ctx A pointer to the active llama_context.
         * @param params The common_params struct for this embedding task.
         * @return A vector of floats representing the embedding.
         */
        virtual std::vector<float> run_embedding(
            llama_model* model,
            llama_context* ctx,
            common_params& params
        );


    private:
        std::atomic<bool> should_stop_generation;
        bool is_waiting_input;
        std::string user_input;
};

#endif //LLAMA_RUNNER_HPP

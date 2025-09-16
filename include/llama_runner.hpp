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

        /** @brief Default constructor. */
        LlamaRunner();

        /** @brief Default deconstructor. */
        virtual ~LlamaRunner();

        /**
         * @brief Runs the prediction loop using an existing model and context.
         * @param model A pointer to the loaded llama_model.
         * @param ctx A pointer to the active llama_context.
         * @param params The common_params struct for this generation task.
         * @param on_generate_text_updated Callback for streaming text chunks.
         * @param on_generate_text_finished Callback for when generation is complete.
         * @return The complete generated text string.
         */
        virtual std::string run_prediction(
            llama_model* model,
            llama_context* ctx,
            common_params& params,
            std::function<void(std::string)> on_generate_text_updated
        );

        void stop_generation();

    private:
        std::atomic<bool> should_stop_generation;
        bool is_waiting_input;
        std::string user_input;
};

#endif //LLAMA_RUNNER_HPP

#ifndef LLAMA_RUNNER_HPP
#define LLAMA_RUNNER_HPP

#include <common.h>
#include <atomic>
#include <functional>
#include <string>
#include "common_types.hpp"

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
         * @param conversation_history A pointer to the conversation history vector.
         * @param on_generate_text_updated Callback for streaming text chunks.
         * @param error_msg Optional output parameter for error messages.
         * @return The complete generated text string.
         */
        virtual std::string run_prediction(
            llama_model* model,
            llama_context* ctx,
            common_params& params,
            const std::vector<ChatMessage>* conversation_history,
            std::function<void(std::string)> on_generate_text_updated,
            std::string* error_msg = nullptr
        );

        void stop_generation();

        /**
         * @brief Generates an embedding vector from the provided prompt.
         * @param model A pointer to the loaded llama_model.
         * @param ctx A pointer to the active llama_context.
         * @param params The common_params struct for this embedding task.
         * @param error_msg Optional output parameter for error messages.
         * @return A vector of floats representing the embedding.
         */
        virtual std::vector<float> run_embedding(
            llama_model* model,
            llama_context* ctx,
            common_params& params,
            std::string* error_msg = nullptr
        );


    private:
        std::atomic<bool> should_stop_generation;
        bool is_waiting_input;
        std::string user_input;

        bool decode_with_error_handling(
            llama_context* ctx,
            llama_batch& batch,
            bool free_batch_on_failure,
            std::string* error_msg = nullptr
        );

};

#endif //LLAMA_RUNNER_HPP

#ifndef LLAMA_STATE_HPP
#define LLAMA_STATE_HPP

#include <common.h>
#include <llama.h>
#include <memory>

/**
 * @class LlamaState
 * @brief Manages the lifecycle of llama.cpp model and context resources.
 */
class LlamaState {
    public:
        LlamaState();
        ~LlamaState();

        LlamaState(const LlamaState&) = delete;
        LlamaState& operator=(const LlamaState&) = delete;

        /**
         * @brief Loads a model and creates a context based on provided parameters.
         * @param params The common_params struct containing model path and other settings.
         * @return True if the model was loaded successfully, false otherwise.
         */
        bool load(common_params & params);

        /** @brief Unloads the model and frees all associated resources. */
        void unload();

        bool is_loaded() const;

        llama_context* get_context();
 
        llama_model * get_model();

    private:
        common_init_result llama_init_result;
        llama_model* model;
        llama_context* model_context;
        bool is_model_loaded;
};

#endif // LLAMA_STATE_HPP
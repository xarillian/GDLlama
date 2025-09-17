#ifndef LLAMA_CONTROLLER_HPP
#define LLAMA_CONTROLLER_HPP

#include "llama_state.hpp"
#include "llama_runner.hpp"
#include "logging_utils.hpp"
#include <mutex>
#include <functional>
#include <common/common.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/global_constants.hpp>

class LlamaController {
    public:
        LlamaController();

        /**
         * @brief Generates text using a provided context and parameters. Isolated from Godot.
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
            common_params& params,
            const std::string& prompt,
            const std::string& grammar,
            const std::string& json,
            bool is_continuous,
            std::function<void(std::string)> on_update
        );


        void reset_context();
        void stop_generation();
        bool is_model_loaded() const;
        godot::Error load_model(common_params&);
        void unload_model();

    private:
        std::unique_ptr<LlamaState> llama_state;
        std::unique_ptr<LlamaRunner> llama_runner;

};


#endif // LLAMA_CONTROLLER_HPP
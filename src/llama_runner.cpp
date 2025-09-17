#include "llama_runner.hpp"
#include "logging_utils.hpp"
#include "llama.h"
#include "llama-sampling.h"
#include <common.h>

LlamaRunner::LlamaRunner() :
    should_stop_generation(false),
    is_waiting_input(false),
    user_input("")
{}

LlamaRunner::~LlamaRunner() {}

void LlamaRunner::stop_generation() {
    should_stop_generation = true;
}

std::string LlamaRunner::run_prediction(
    llama_model* model,
    llama_context* ctx,
    common_params& params,
    std::function<void(std::string)> on_generate_text_updated
){
    should_stop_generation = false;
    if (ctx == nullptr) {
        std::string err_msg = "Invalid context.";
        GDLOG_ERROR(err_msg);
        throw std::runtime_error(err_msg);
    }

    if (model == nullptr) {
        std::string err_msg = "Invalid model";
        GDLOG_ERROR(err_msg);
        throw std::runtime_error(err_msg);
    }

    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    std::vector<llama_token> prompt_tokens = ::common_tokenize(ctx, params.prompt, add_bos, true);
    const int n_ctx = llama_n_ctx(ctx);

    if ((int)prompt_tokens.size() > n_ctx - 4) {
        std::string err_msg = "Prompt is too long for the context size.";
        GDLOG_ERROR(err_msg);
        throw std::runtime_error(err_msg);
    }

    auto * sampler_chain = llama_sampler_chain_init(llama_sampler_chain_default_params());

    llama_sampler_chain_add(
        sampler_chain,
        llama_sampler_init_penalties(
            params.sampling.penalty_last_n,
            params.sampling.penalty_repeat,
            params.sampling.penalty_freq,
            params.sampling.penalty_present
        )
    );

    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_k(params.sampling.top_k));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_p(params.sampling.top_p, 1));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_min_p(params.sampling.min_p, 1));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_typical(params.sampling.typ_p, 1));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_temp(params.sampling.temp));

    // The chain must end with a sampler that selects a token.
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_dist(params.sampling.seed));

    std::string generated_text = "";
    int n_remain = params.n_predict;
    std::vector<llama_token> embd;
    const llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    int n_past = (max_pos == -1) ? 0 : max_pos + 1;
    const auto * vocab = llama_model_get_vocab(model);
    
    // int n_past = llama_vocab_n_tokens(vocab); <- prob not this, but keep it around for now
    
    const llama_token EOD_TOKEN = llama_vocab_eos(vocab);

    GDLOG_DEBUG("Starting Generation Loop.");
    GDLOG_DEBUG("n_remain: " + std::to_string(n_remain));
    while (n_remain > 0 && !should_stop_generation) {
        if (n_past < prompt_tokens.size()) {
            // If we haven't processed the whole prompt yet, let's do that first.
            embd.clear();
            int n_eval = (int)prompt_tokens.size() - n_past;
            if (n_eval > params.n_batch) n_eval = params.n_batch;
            for (int i = 0; i < n_eval; i++) {
                embd.push_back(prompt_tokens[n_past + i]);
            }
        }
        
        if (!embd.empty()) {
            if (n_past + (int)embd.size() > n_ctx) {
                GDLOG_WARN("Context window is full, stopping generation.");
                break;
            }
            
            llama_batch batch = llama_batch_get_one(embd.data(), embd.size());

            std::vector<llama_pos> positions;
            positions.reserve(embd.size());
            for(size_t i = 0; i < embd.size(); ++i) {
                positions.push_back(n_past + i);
            }
            batch.pos = positions.data();

            if (llama_decode(ctx, batch) != 0) {
                std::string err_msg = "Llama failed to decode.";
                GDLOG_ERROR(err_msg);
                throw std::runtime_error(err_msg);
            }
            n_past += embd.size();
        }

        if (n_past >= prompt_tokens.size()) {
            llama_token new_token_id = llama_sampler_sample(sampler_chain, ctx, -1);
            llama_sampler_accept(sampler_chain, new_token_id);

            if (new_token_id == EOD_TOKEN && !params.sampling.ignore_eos) {
                GDLOG_DEBUG("End of generation token found.");
                break;
            }

            const std::string token_str = common_token_to_piece(ctx, new_token_id);
            on_generate_text_updated(token_str);
            generated_text.append(token_str);

            GDLOG_DEBUG("Token string chunk generated: " + token_str);
            n_remain--;

            embd.clear();
            embd.push_back(new_token_id);
        }
    }

    llama_sampler_free(sampler_chain);

    GDLOG_DEBUG(("Generated Text: \"\"\"\n" + generated_text + "\n\"\"\"").c_str());

    GDLOG_INFO("Prediction finished.");
    return generated_text;
}

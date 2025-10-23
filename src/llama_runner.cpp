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

bool LlamaRunner::decode_with_error_handling(
    llama_context* ctx,
    llama_batch& batch,
    bool free_batch_on_failure,
    std::string* error_msg
) {
    // @todo not a fan of this function
    if (llama_decode(ctx, batch) != 0) {
        if (free_batch_on_failure) {
            llama_batch_free(batch);
        }
        std::string err = "Llama failed to decode.";
        GDLOG_ERROR(err);
        if (error_msg) *error_msg = err;
        return false;
    }
    return true;
}

std::string apply_chat_template(
    llama_context* ctx,
    const std::vector<ChatMessage>& conversation_history,
    const std::string& chat_template,
    std::string* error_msg
) {
    std::vector<llama_chat_message> messages_for_api;
    for (const auto& msg : conversation_history) {
        messages_for_api.push_back({msg.role.c_str(), msg.content.c_str()});
    }

    const int32_t buffer_size = llama_n_ctx(ctx);
    std::vector<char> buffer(buffer_size);

    int32_t formatted_size = llama_chat_apply_template(
        chat_template.empty() ? nullptr : chat_template.c_str(),
        messages_for_api.data(),
        messages_for_api.size(),
        true, // add_assistant_prefix
        buffer.data(),
        buffer.size()
    );
    
    if (formatted_size < 0) {
        std::string err = "Failed to apply chat template.";
        GDLOG_ERROR(err);
        if (error_msg) *error_msg = err;
        return "";
    }

    if (static_cast<int32_t>(buffer.size()) <= formatted_size) {
        std::string err = "Formatted chat prompt exceeds the buffer size.";
        GDLOG_ERROR(err);
        if (error_msg) *error_msg = err;
        return "";
    }

    return std::string(buffer.data());
}

std::string LlamaRunner::run_prediction(
    llama_model* model,
    llama_context* ctx,
    common_params& params,
    const std::vector<ChatMessage>* conversation_history,
    std::function<void(std::string)> on_generate_text_updated,
    std::string* error_msg
){
    should_stop_generation = false;
    if (ctx == nullptr) {
        std::string err_msg = "Invalid context.";
        GDLOG_ERROR(err_msg);
        if (error_msg) *error_msg = err_msg;
        return "";
    }

    if (model == nullptr) {
        std::string err_msg = "Invalid model";
        GDLOG_ERROR(err_msg);
        if (error_msg) *error_msg = err_msg;
        return "";
    }

    if (conversation_history != nullptr) {
        std::string formatted_prompt = apply_chat_template(
            ctx,
            *conversation_history,
            params.chat_template,
            error_msg
        );

        params.prompt = formatted_prompt;
    }

    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    std::vector<llama_token> prompt_tokens = ::common_tokenize(ctx, params.prompt, add_bos, true);
    const int n_ctx = llama_n_ctx(ctx);

    if ((int)prompt_tokens.size() > n_ctx - 4) {
        std::string err_msg = "Prompt is too long for the context size.";
        GDLOG_ERROR(err_msg);
        if (error_msg) *error_msg = err_msg;
        return "";
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
 
    const llama_token EOD_TOKEN = llama_vocab_eos(vocab);

    GDLOG_DEBUG("Starting Generation Loop.");
    GDLOG_DEBUG("n_remain: " + std::to_string(n_remain));
    while ((params.n_predict == -1 || n_remain > 0) && !should_stop_generation) {
        if (n_past < prompt_tokens.size()) {
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

            if (!decode_with_error_handling(ctx, batch, false, error_msg)) {
                return "";  // Error already set in decode_with_error_handling
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

            if (params.n_predict != -1) {
                n_remain--;
            }

            embd.clear();
            embd.push_back(new_token_id);
        }
    }

    llama_sampler_free(sampler_chain);

    GDLOG_DEBUG(("Generated Text: \"\"\"\n" + generated_text + "\n\"\"\"").c_str());

    GDLOG_INFO("Prediction finished.");
    return generated_text;
}

std::vector<float> LlamaRunner::run_embedding(
    llama_model* model,
    llama_context* ctx,
    common_params& params,
    std::string* error_msg
) {
    GDLOG_DEBUG("Starting embedding generation for prompt: " + params.prompt);

    if (ctx == nullptr) {
        std::string err_msg = "Invalid context.";
        GDLOG_ERROR(err_msg);
        if (error_msg) *error_msg = err_msg;
        return {};
    }

    if (model == nullptr) {
        std::string err_msg = "Invalid model";
        GDLOG_ERROR(err_msg);
        if (error_msg) *error_msg = err_msg;
        return {};
    }

    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    std::vector<llama_token> prompt_tokens = ::common_tokenize(ctx, params.prompt, add_bos, false);

    if (prompt_tokens.empty()) {
        GDLOG_WARN("Prompt is empty, cannot generate embedding.");
        return {};
    }

    const int n_tokens = prompt_tokens.size();
    GDLOG_DEBUG("Prompt tokenized into " + std::to_string(n_tokens) + " tokens.");
    if (n_tokens > llama_n_ctx(ctx)) {
        std::string err_msg = "Prompt is too long for the context size.";
        GDLOG_ERROR(err_msg);
        if (error_msg) *error_msg = err_msg;
        return {};
    }

    const int n_embd = llama_model_n_embd(model);
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    const float* embd_ptr = nullptr;
    std::vector<float> embedding(n_embd);

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        GDLOG_DEBUG("Using last-token pooling strategy for embedding.");

        for (int32_t i = 0; i < n_tokens; i++) {
            batch.token[i]       = prompt_tokens[i];
            batch.pos[i]         = i;
            batch.n_seq_id[i]    = 1;
            batch.seq_id[i][0]   = 0;
            // Only compute logits for the last token
            batch.logits[i]      = (i == n_tokens - 1);
        }
        batch.n_tokens = n_tokens;

        if (!decode_with_error_handling(ctx, batch, true, error_msg)) {
            return {};  // Error already set in decode_with_error_handling
        }

        embd_ptr = llama_get_embeddings(ctx);

    } else if (pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED) {
        GDLOG_DEBUG("Using built-in pooling strategy for embedding: " + std::to_string(pooling_type));

        for (int32_t i = 0; i < n_tokens; i++) {
            batch.token[i]       = prompt_tokens[i];
            batch.pos[i]         = i;
            batch.n_seq_id[i]    = 1;
            batch.seq_id[i][0]   = 0;
            batch.logits[i]      = true;
        }
        batch.n_tokens = n_tokens;

        if (!decode_with_error_handling(ctx, batch, true, error_msg)) {
            return {};  // Error already set in decode_with_error_handling
        }

        embd_ptr = llama_get_embeddings_seq(ctx, 0);

    } else {
        GDLOG_WARN("Model uses an unspecified pooling strategy. Embeddings will not be generated.");
        llama_batch_free(batch);
        return {};
    }

    
    if (embd_ptr == nullptr) {
        llama_batch_free(batch);
        std::string err_msg = "Failed to get sequence embeddings.";
        GDLOG_ERROR(err_msg);
        if (error_msg) *error_msg = err_msg;
        return {};
    }

    common_embd_normalize(embd_ptr, embedding.data(), n_embd, params.embd_normalize);
    llama_batch_free(batch);
    return embedding;
}

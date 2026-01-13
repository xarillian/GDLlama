#pragma once

#include "llama.h"
#include "chorus_core/chorus_common.hpp"

#include <vector>
#include <string>
#include <cstring> // for memcpy

namespace Chorus {
namespace LlamaUtils {
    inline void batch_add_seq(
        llama_batch& batch,
        llama_token token,
        int seq_id,
        int pos,
        bool logits
    ) {
        batch.token[batch.n_tokens] = token;
        batch.pos[batch.n_tokens] = pos;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = seq_id;
        batch.logits[batch.n_tokens] = logits;
        batch.n_tokens++;
    }

    inline std::vector<llama_token> tokenize(
        llama_context* ctx,
        const std::string& text,
        bool add_special
    ) {
        const llama_model* model = llama_get_model(ctx);
        const llama_vocab* vocab = llama_model_get_vocab(model);

        int n_tokens_max = text.length() + 2; 
        std::vector<llama_token> result(n_tokens_max);

        int n_tokens = llama_tokenize(
            vocab,
            text.c_str(),
            text.length(),
            result.data(),
            result.size(),
            add_special,
            false
        );

        if (n_tokens < 0) {
            // Buffer was too small. In a real app, verify and resize.
            // For now, we assume text.length() + 2 is sufficient for standard text.
            // @todo we should not just assume this!
            result.resize(0);
        } else {
            result.resize(n_tokens);
        }
        
        return result;
    }

    inline std::string token_to_piece(llama_context* ctx, llama_token token) {
        const llama_model* model = llama_get_model(ctx);
        const llama_vocab* vocab = llama_model_get_vocab(model);

        char buf[256];

        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);

        if (n < 0) {
            return ""; // Error or buffer too small
        }
        return std::string(buf, n);
    }

    inline llama_sampler* build_sampler(const Chorus::GenerationConfig& gen_config) {
        llama_sampler_chain_params params = llama_sampler_chain_default_params();
        llama_sampler* chain = llama_sampler_chain_init(params);

        // Penalties
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
            -1,             // last_n
            gen_config.repeat_penalty, 
            0.0f,           // freq_penalty
            0.0f            // present_penalty
        ));

        // Sampling strategies
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(gen_config.top_k));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(gen_config.top_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(gen_config.temperature));

        // RNG
        llama_sampler_chain_add(chain, llama_sampler_init_dist(gen_config.seed));

        return chain;
    }
}
}
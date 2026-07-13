#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/options.hpp"
#include "llama.h"

#include <string>
#include <vector>

namespace Chorus {
namespace LlamaUtils {
inline void batch_add_seq(llama_batch& batch, llama_token token, int seq_id, int pos, bool logits) {
    batch.token[batch.n_tokens] = token;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = seq_id;
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

inline std::vector<llama_token> tokenize(llama_context* ctx, const std::string& text, bool add_special) {
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> result(text.length() + 2);

    int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), result.data(), result.size(), add_special, false);

    if (n_tokens < 0) {
        // llama.cpp returns -(required_size) on overflow; resize and retry.
        result.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), result.data(), result.size(), add_special, false);
    }

    if (n_tokens < 0) {
        return {};
    }

    result.resize(n_tokens);
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

// Unset portable fields resolve to llama.cpp's own defaults (verified against
// the pinned revision, b9934). This is the one place those numbers live.
struct ResolvedSampling {
    int32_t max_tokens = -1; // upstream n_predict default: until EOS/context
    float temperature = 0.80f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    uint32_t seed = LLAMA_DEFAULT_SEED; // sentinel: random
    float repeat_penalty = 1.0f;        // upstream default: disabled
};

inline ResolvedSampling resolve_sampling(const Chorus::GenerationConfig& config) {
    ResolvedSampling r;
    const auto& c = config.common;
    if (c.max_tokens)
        r.max_tokens = *c.max_tokens;
    if (c.temperature)
        r.temperature = *c.temperature;
    if (c.top_k)
        r.top_k = *c.top_k;
    if (c.top_p)
        r.top_p = *c.top_p;
    if (c.seed)
        r.seed = static_cast<uint32_t>(*c.seed); // llama.cpp seeds are 32-bit
    auto ns = config.backend_options.find("llama");
    if (ns != config.backend_options.end()) {
        if (const auto* llama_opts = std::get_if<Chorus::OptionMap>(&ns->second)) {
            auto rp = llama_opts->find("repeat_penalty");
            if (rp != llama_opts->end())
                if (const auto* v = std::get_if<double>(&rp->second))
                    r.repeat_penalty = static_cast<float>(*v);
        }
    }
    return r;
}

inline llama_sampler* build_sampler(const ResolvedSampling& s) {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(params);
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(-1, s.repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(s.top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(s.top_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(s.temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(s.seed));
    return chain;
}
} // namespace LlamaUtils
} // namespace Chorus
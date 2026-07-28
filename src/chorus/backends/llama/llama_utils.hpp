#pragma once

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

// parse_special: raw user prompts keep special-token text literal (false);
// rendered chat prompts carry real control tokens ("<start_of_turn>") that
// must map to their token ids (true).
inline std::vector<llama_token>
tokenize(llama_context* ctx, const std::string& text, bool add_special, bool parse_special = false) {
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> result(text.length() + 2);

    int n_tokens =
        llama_tokenize(vocab, text.c_str(), text.length(), result.data(), result.size(), add_special, parse_special);

    if (n_tokens < 0) {
        // llama.cpp returns -(required_size) on overflow; resize and retry.
        result.resize(-n_tokens);
        n_tokens = llama_tokenize(
            vocab, text.c_str(), text.length(), result.data(), result.size(), add_special, parse_special
        );
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

} // namespace LlamaUtils
} // namespace Chorus

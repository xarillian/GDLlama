#pragma once

#include "llama.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <optional>
#include <limits>
#include <vector>

namespace Chorus {
namespace LlamaUtils {
namespace detail {
template <typename Reader>
inline std::string read_string(Reader&& reader) {
    std::array<char, 256> buffer{};
    int32_t length = reader(buffer.data(), buffer.size());
    if (length < 0)
        return {};
    if (static_cast<std::size_t>(length) < buffer.size())
        return std::string(buffer.data(), static_cast<std::size_t>(length));

    std::string result(static_cast<std::size_t>(length) + 1, '\0');
    length = reader(result.data(), result.size());
    if (length < 0 || static_cast<std::size_t>(length) >= result.size())
        return {};
    result.resize(static_cast<std::size_t>(length));
    return result;
}
} // namespace detail

class Batch {
  public:
    Batch() = default;
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    ~Batch() {
        reset();
    }

    void initialize(int32_t token_capacity, int32_t embedding_size, int32_t max_sequences) {
        reset();
        _batch = llama_batch_init(token_capacity, embedding_size, max_sequences);
        _initialized = true;
    }

    void reset() {
        if (!_initialized)
            return;
        llama_batch_free(_batch);
        _batch = {};
        _initialized = false;
    }

    llama_batch& get() {
        return _batch;
    }

  private:
    llama_batch _batch{};
    bool _initialized = false;
};

inline std::string model_metadata(const llama_model* model, const char* key) {
    return detail::read_string(
        [model, key](char* buffer, std::size_t size) { return llama_model_meta_val_str(model, key, buffer, size); }
    );
}

inline std::string model_description(const llama_model* model) {
    return detail::read_string(
        [model](char* buffer, std::size_t size) { return llama_model_desc(model, buffer, size); }
    );
}

inline void batch_add_seq(llama_batch& batch, llama_token token, int seq_id, int pos, bool logits) {
    batch.token[batch.n_tokens] = token;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = seq_id;
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

/*
 * Tokenizes provider input with optional control-token parsing.
 *
 * `parse_special` remains `false` for raw user prompts so special-token text
 * stays literal. Rendered chat prompts contain control tokens such as
 * `<start_of_turn>`, so callers pass `true` to map them to token IDs.
 */
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

inline std::optional<std::vector<llama_token>> tokenize_vocabulary(
    const llama_vocab* vocab, const std::string& text, bool add_special, bool parse_special = false
) {
    if (!vocab || text.size() > static_cast<size_t>(INT32_MAX))
        return std::nullopt;
    if (text.empty() && !add_special)
        return std::vector<llama_token>{};
    const auto length = static_cast<int32_t>(text.size());
    int32_t needed = llama_tokenize(vocab, text.data(), length, nullptr, 0, add_special, parse_special);
    if (needed == INT32_MIN || needed > 0)
        return std::nullopt;
    if (needed == 0)
        return std::vector<llama_token>{};
    std::vector<llama_token> tokens(static_cast<size_t>(-needed));
    const int32_t count = llama_tokenize(vocab, text.data(), length, tokens.data(), -needed, add_special, parse_special);
    if (count < 0 || count > -needed)
        return std::nullopt;
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

inline std::string token_to_piece(llama_context* ctx, llama_token token) {
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n >= 0)
        return std::string(buf, n);

    // A negative result reports the required buffer size.
    std::string piece(static_cast<size_t>(-n), '\0');
    n = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    if (n < 0)
        return "";
    piece.resize(static_cast<size_t>(n));
    return piece;
}

} // namespace LlamaUtils
} // namespace Chorus

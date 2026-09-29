#pragma once

#include "sampling.h"
#include "wlib/parallel_for.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

struct llama_context;

namespace Chorus {

enum class LlamaSamplingPath {
    Context,      // Samples through `llama_context` on the calling thread.
    Chain,        // Applies the sampler chain to one logits row.
    FirstMaximum, // The sampler chain reduces to `Chorus::llama_first_maximum`.
};

struct LlamaSamplingSlot {
    common_sampler* sampler;
    int32_t output_index;
    LlamaSamplingPath path;
};

/// Reports whether a sampler built from `sampling` can select tokens from a logits row alone.
bool llama_sampling_is_context_free(const common_params_sampling& sampling);

/// Reports whether a context-free sampler's chain is a zero-temperature greedy selection.
bool llama_sampler_selects_first_maximum(const common_sampler* sampler);

/*
 * Returns the token a zero-temperature chain ending in greedy selection picks from `logits`.
 *
 * The first strictly greater logit wins and NaN never does, except at index zero where the
 * chain's scan starts. A row holding only -inf or NaN selects zero. When `parallel` is
 * given, the row is split across its lanes.
 */
llama_token llama_first_maximum(std::span<const float> logits, wlib::ParallelFor* parallel);

/*
 * Samples and accepts one token for each output of a decoded batch.
 *
 * `llama_context` reads are not thread-safe, so `Chorus::LlamaSamplingPath::Context` samplers
 * run on the calling thread and every logits row is resolved there. Other samplers then run
 * across lanes; with fewer samplers than lanes, each row is split across lanes instead.
 * Tokens are returned in slot order.
 */
class LlamaBatchSampler {
  public:
    LlamaBatchSampler(std::size_t lanes, int32_t vocabulary_size);

    std::vector<llama_token> sample(llama_context* context, std::span<const LlamaSamplingSlot> slots);

  private:
    llama_token select(const LlamaSamplingSlot& slot, const float* logits, std::size_t lane, bool across_lanes);
    llama_token apply_chain(common_sampler* sampler, const float* logits, std::size_t lane, bool across_lanes);

    wlib::ParallelFor _parallel;
    std::size_t _vocabulary_size;
    std::vector<std::vector<llama_token_data>> _candidates;
};

} // namespace Chorus

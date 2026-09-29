#include "chorus/providers/llama/llama_batch_sampler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace Chorus {
namespace {

struct Maximum {
    float value = -std::numeric_limits<float>::infinity();
    std::size_t index = std::numeric_limits<std::size_t>::max();
};

// Independent accumulators let the compiler vectorize the maximum, which a single running
// maximum's dependency chain prevents; the first index holding it is found afterwards.
Maximum scan_maximum(std::span<const float> logits, std::size_t begin, std::size_t end) {
    constexpr std::size_t kAccumulators = 16;
    constexpr float kLowest = -std::numeric_limits<float>::infinity();
    std::array<float, kAccumulators> accumulators;
    accumulators.fill(kLowest);
    std::size_t index = begin;
    for (; index + kAccumulators <= end; index += kAccumulators) {
        for (std::size_t lane = 0; lane < kAccumulators; ++lane)
            accumulators[lane] = logits[index + lane] > accumulators[lane] ? logits[index + lane] : accumulators[lane];
    }
    float maximum = kLowest;
    for (; index < end; ++index)
        maximum = logits[index] > maximum ? logits[index] : maximum;
    for (const float accumulator : accumulators)
        maximum = accumulator > maximum ? accumulator : maximum;
    if (maximum == kLowest)
        return {};
    index = begin;
    while (!(logits[index] == maximum))
        ++index;
    return {maximum, index};
}

void for_each_part(
    wlib::ParallelFor& parallel, std::size_t size,
    const std::function<void(std::size_t part, std::size_t begin, std::size_t end)>& body
) {
    const std::size_t chunk = (size + parallel.lanes() - 1) / parallel.lanes();
    parallel.run(parallel.lanes(), [&](std::size_t part, std::size_t) {
        body(part, std::min(size, part * chunk), std::min(size, (part + 1) * chunk));
    });
}

std::string_view sampler_name(llama_sampler* chain, int32_t index) {
    return llama_sampler_name(llama_sampler_chain_get(chain, index));
}

} // namespace

// These are the only features for which `common_sampler_sample` does more than apply
// the sampler chain to one logits row.
bool llama_sampling_is_context_free(const common_params_sampling& sampling) {
    return common_grammar_value(sampling.grammar).empty() && sampling.reasoning_budget_start.empty() &&
           !sampling.backend_sampling;
}

// `Chorus::make_llama_sampler` ends a chain in greedy selection only after zero temperature
// has collapsed it, so this exact pair is an argmax over the raw logits.
bool llama_sampler_selects_first_maximum(const common_sampler* sampler) {
    llama_sampler* chain = common_sampler_get(sampler);
    return llama_sampler_chain_n(chain) == 2 && sampler_name(chain, 0) == "temp-ext" &&
           sampler_name(chain, 1) == "greedy";
}

llama_token llama_first_maximum(std::span<const float> logits, wlib::ParallelFor* parallel) {
    if (logits.empty() || std::isnan(logits[0]))
        return 0;
    Maximum best;
    if (parallel) {
        std::vector<Maximum> parts(parallel->lanes());
        for_each_part(*parallel, logits.size(), [&](std::size_t part, std::size_t begin, std::size_t end) {
            parts[part] = scan_maximum(logits, begin, end);
        });
        for (const auto& part : parts) {
            if (part.value > best.value)
                best = part;
        }
    } else {
        best = scan_maximum(logits, 0, logits.size());
    }
    return best.index == Maximum{}.index ? 0 : static_cast<llama_token>(best.index);
}

LlamaBatchSampler::LlamaBatchSampler(std::size_t lanes, int32_t vocabulary_size)
    : _parallel(lanes), _vocabulary_size(static_cast<std::size_t>(vocabulary_size)), _candidates(_parallel.lanes()) {}

std::vector<llama_token> LlamaBatchSampler::sample(llama_context* context, std::span<const LlamaSamplingSlot> slots) {
    std::vector<llama_token> tokens(slots.size(), LLAMA_TOKEN_NULL);
    std::vector<const float*> rows(slots.size(), nullptr);
    std::vector<std::size_t> row_slots;
    for (std::size_t index = 0; index < slots.size(); ++index) {
        const auto& slot = slots[index];
        if (slot.path == LlamaSamplingPath::Context) {
            tokens[index] = common_sampler_sample(slot.sampler, context, slot.output_index);
            common_sampler_accept(slot.sampler, tokens[index], true);
            continue;
        }
        rows[index] = llama_get_logits_ith(context, slot.output_index);
        if (!rows[index])
            throw std::logic_error("decoded batch has no logits for a sampled output");
        row_slots.push_back(index);
    }

    if (row_slots.size() >= _parallel.lanes()) {
        _parallel.run(row_slots.size(), [&](std::size_t item, std::size_t lane) {
            const std::size_t index = row_slots[item];
            tokens[index] = select(slots[index], rows[index], lane, false);
        });
    } else {
        for (std::size_t index : row_slots)
            tokens[index] = select(slots[index], rows[index], 0, true);
    }
    return tokens;
}

llama_token LlamaBatchSampler::select(const LlamaSamplingSlot& slot, const float* logits, std::size_t lane, bool across_lanes) {
    const llama_token token = slot.path == LlamaSamplingPath::FirstMaximum
                                  ? llama_first_maximum({logits, _vocabulary_size}, across_lanes ? &_parallel : nullptr)
                                  : apply_chain(slot.sampler, logits, lane, across_lanes);
    common_sampler_accept(slot.sampler, token, true);
    return token;
}

llama_token LlamaBatchSampler::apply_chain(common_sampler* sampler, const float* logits, std::size_t lane, bool across_lanes) {
    auto& candidates = _candidates[lane];
    candidates.resize(_vocabulary_size);
    const auto fill = [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t token = begin; token < end; ++token)
            candidates[token] = {static_cast<llama_token>(token), logits[token], 0.0f};
    };
    if (across_lanes)
        for_each_part(_parallel, _vocabulary_size, fill);
    else
        fill(0, 0, _vocabulary_size);
    llama_token_data_array array{candidates.data(), candidates.size(), -1, false};
    llama_sampler_apply(common_sampler_get(sampler), &array);
    return array.data[array.selected].id;
}

} // namespace Chorus

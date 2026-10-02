#pragma once

#include "chorus/core/common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <vector>

namespace Chorus {

enum class LlamaPlannerPhase { Prefill, Decode };

struct LlamaPlannerSequence {
    int sequence_id;
    RequestType type;
    int priority;
    uint64_t submission_sequence;
    LlamaPlannerPhase phase;
    size_t prompt_cursor;
    size_t prompt_size;
    bool exclusive;
};

struct LlamaPlannerEntry {
    int sequence_id;
    size_t prompt_offset;
    bool decode;
    bool logits;
};

struct LlamaPlannedBatch {
    RequestType type;
    std::vector<LlamaPlannerEntry> entries;
    std::vector<int> participants;
    std::optional<RequestType> contested_type;
    std::optional<uint64_t> decode_fairness;
};

inline std::optional<LlamaPlannedBatch> llama_plan_batch(
    std::vector<LlamaPlannerSequence> sequences,
    int32_t generation_budget,
    int32_t micro_batch_budget,
    uint64_t decode_fairness_cursor,
    std::optional<RequestType> last_contested_type
) {
    if (generation_budget <= 0 || micro_batch_budget <= 0)
        return std::nullopt;
    if (std::ranges::any_of(sequences, &LlamaPlannerSequence::exclusive))
        std::erase_if(sequences, [](const LlamaPlannerSequence& sequence) { return !sequence.exclusive; });
    std::ranges::sort(sequences, [](const LlamaPlannerSequence& left, const LlamaPlannerSequence& right) {
        if (left.priority != right.priority)
            return left.priority > right.priority;
        return left.submission_sequence < right.submission_sequence;
    });
    if (sequences.empty())
        return std::nullopt;
    const int priority = sequences.front().priority;
    std::erase_if(sequences, [priority](const LlamaPlannerSequence& sequence) {
        return sequence.priority != priority;
    });
    const bool has_generation = std::ranges::any_of(sequences, [](const LlamaPlannerSequence& sequence) {
        return sequence.type == RequestType::Generate;
    });
    const bool has_embedding = std::ranges::any_of(sequences, [](const LlamaPlannerSequence& sequence) {
        return sequence.type == RequestType::Embedding;
    });
    const RequestType type =
        has_generation && has_embedding
            ? (last_contested_type
                   ? (*last_contested_type == RequestType::Generate ? RequestType::Embedding : RequestType::Generate)
                   : sequences.front().type)
            : (has_generation ? RequestType::Generate : RequestType::Embedding);
    std::erase_if(sequences, [type](const LlamaPlannerSequence& sequence) { return sequence.type != type; });

    LlamaPlannedBatch plan{type};
    if (has_generation && has_embedding)
        plan.contested_type = type;
    auto add = [&plan](const LlamaPlannerEntry& entry) {
        if (std::ranges::find(plan.participants, entry.sequence_id) == plan.participants.end())
            plan.participants.push_back(entry.sequence_id);
        plan.entries.push_back(entry);
    };
    if (type == RequestType::Embedding) {
        int32_t remaining = micro_batch_budget;
        for (const auto& sequence : sequences) {
            const int32_t count = static_cast<int32_t>(sequence.prompt_size);
            if (count > remaining)
                break;
            for (size_t offset = 0; offset < sequence.prompt_size; ++offset)
                add({sequence.sequence_id, offset, false, offset + 1 == sequence.prompt_size});
            remaining -= count;
        }
        return plan.entries.empty() ? std::nullopt : std::optional<LlamaPlannedBatch>{std::move(plan)};
    }

    int32_t remaining = generation_budget;
    std::vector<LlamaPlannerSequence> decoders;
    for (const auto& sequence : sequences) {
        if (sequence.phase == LlamaPlannerPhase::Decode)
            decoders.push_back(sequence);
    }
    std::ranges::sort(
        decoders, [decode_fairness_cursor](const LlamaPlannerSequence& left, const LlamaPlannerSequence& right) {
            const bool left_after = left.submission_sequence >= decode_fairness_cursor;
            const bool right_after = right.submission_sequence >= decode_fairness_cursor;
            if (left_after != right_after)
                return left_after;
            return left.submission_sequence < right.submission_sequence;
        }
    );
    for (const auto& sequence : decoders) {
        if (remaining == 0)
            break;
        add({sequence.sequence_id, 0, true, true});
        plan.decode_fairness = sequence.submission_sequence + 1;
        --remaining;
    }
    // A step waits for all of its prefill, so capping it keeps streaming text arriving between micro-batches.
    if (!decoders.empty())
        remaining = std::min(remaining, micro_batch_budget);
    for (const auto& sequence : sequences) {
        if (remaining == 0 || sequence.phase != LlamaPlannerPhase::Prefill)
            continue;
        const size_t count = std::min(sequence.prompt_size - sequence.prompt_cursor, static_cast<size_t>(remaining));
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t token = sequence.prompt_cursor + offset;
            add({sequence.sequence_id, token, false, token + 1 == sequence.prompt_size});
        }
        remaining -= static_cast<int32_t>(count);
    }
    return plan.entries.empty() ? std::nullopt : std::optional<LlamaPlannedBatch>{std::move(plan)};
}

} // namespace Chorus

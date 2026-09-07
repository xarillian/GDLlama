#pragma once

#include <cstddef>
#include <vector>

namespace Chorus {

struct LlamaRecoveryContribution {
    int sequence_id;
    size_t entries;
    size_t minimum_entries;
};

struct LlamaRecoverySelection {
    std::vector<int> sequence_ids;
    std::vector<size_t> entry_limits;
};

inline std::vector<LlamaRecoverySelection>
llama_recovery_reductions(const std::vector<LlamaRecoveryContribution>& contributors) {
    std::vector<LlamaRecoverySelection> reductions;
    if (contributors.empty())
        return reductions;

    LlamaRecoverySelection current;
    for (const auto& contributor : contributors) {
        if (contributor.minimum_entries == 0 || contributor.minimum_entries > contributor.entries)
            return {};
        current.sequence_ids.push_back(contributor.sequence_id);
        current.entry_limits.push_back(contributor.entries);
    }

    for (size_t index = current.entry_limits.size(); index > 0; --index) {
        const size_t contributor = index - 1;
        while (current.entry_limits[contributor] > contributors[contributor].minimum_entries) {
            --current.entry_limits[contributor];
            reductions.push_back(current);
        }
    }

    for (size_t count = contributors.size(); count > 1; --count) {
        current.sequence_ids.resize(count - 1);
        current.entry_limits.resize(count - 1);
        for (size_t index = 0; index < current.entry_limits.size(); ++index)
            current.entry_limits[index] = contributors[index].minimum_entries;
        reductions.push_back(current);
    }

    return reductions;
}

} // namespace Chorus

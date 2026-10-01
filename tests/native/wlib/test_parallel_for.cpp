#include "gtest_utils.hpp"
#include "wlib/parallel_for.hpp"

#include <atomic>
#include <stdexcept>
#include <vector>

namespace {

TEST(ParallelFor, Every_index_runs_once_and_concurrent_calls_never_share_a_lane) {
    wlib::ParallelFor parallel(4);
    std::vector<std::atomic<int>> runs(257);
    std::vector<std::atomic<bool>> busy(parallel.lanes());
    std::atomic<bool> shared_lane{false};

    for (int round = 0; round < 20; ++round) {
        parallel.run(runs.size(), [&](std::size_t index, std::size_t lane) {
            if (busy[lane].exchange(true))
                shared_lane = true;
            ++runs[index];
            busy[lane] = false;
        });
    }

    for (const auto& count : runs)
        ASSERT_EQ(count.load(), 20);
    ASSERT_FALSE(shared_lane.load());
}

TEST(ParallelFor, A_failing_body_reaches_the_caller_and_the_pool_keeps_working) {
    wlib::ParallelFor parallel(4);

    ASSERT_THROW(
        parallel.run(
            64,
            [](std::size_t index, std::size_t) {
                if (index == 7)
                    throw std::runtime_error("sensor offline");
            }
        ),
        std::runtime_error
    );

    std::atomic<int> runs{0};
    parallel.run(64, [&](std::size_t, std::size_t) { ++runs; });
    ASSERT_EQ(runs.load(), 64);
}

} // namespace

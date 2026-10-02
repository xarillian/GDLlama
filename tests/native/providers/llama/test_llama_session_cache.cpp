#include "chorus/providers/llama/llama_engine.hpp"
#include "chorus/providers/llama/llama_session_cache.hpp"
#include "gtest_utils.hpp"

class LlamaSessionCacheModelTest : public ChorusModelTest {};

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr const char* kModelPath = "tests/models/gemma-3-270m-it-F16.gguf";

std::string user_turn(const std::string& text) {
    return "<start_of_turn>user\n" + text + "<end_of_turn>\n<start_of_turn>model\n";
}

std::string history_of(int entries) {
    std::string text;
    for (int entry = 1; entry <= entries; ++entry)
        text += "Day " + std::to_string(entry) + ": the caravan crossed the dunes and the guards traded stories. ";
    return user_turn("Here is my travel log. " + text + "Summarize it in one sentence.");
}

// One engine; turns run one at a time so each batch belongs to a single request.
class Conversations {
  public:
    struct Turn {
        std::string text;
        int32_t processed = 0;
        bool completed = false;
        std::vector<float> first_logits;
    };

    explicit Conversations(int64_t slots) {
        Chorus::ChorusConfig config;
        config.model.model_id = "test-model";
        config.model.format = Chorus::ModelFormat::Gguf;
        config.model.assets.push_back({Chorus::AssetRole::Weights, kModelPath});
        config.provider_options["llama"] = Chorus::ProviderOptionMap{
            {"use_gpu", false}, {"max_concurrent_requests", slots}, {"context_size", 2048 * slots}
        };
        loaded = !_engine.initialize(config, {}, {}).has_value();
        _engine.set_batch_observer([this](const Chorus::LlamaBatchRecord& record) {
            std::lock_guard<std::mutex> lock(_mutex);
            for (const auto id : record.request_ids)
                _processed[id] += record.token_count;
        });
        _engine.set_logits_observer([this](Chorus::RequestId id, std::span<const float> logits) {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_turns[id].first_logits.empty())
                _turns[id].first_logits.assign(logits.begin(), logits.end());
        });
    }

    ~Conversations() { _engine.shutdown(); }

    struct Line {
        std::optional<std::string> session;
        std::string prompt;
        int max_tokens = 8;
    };

    Turn say(const std::optional<std::string>& session, const std::string& prompt, int max_tokens = 8) {
        return say_together({{session, prompt, max_tokens}}).front();
    }

    // Submits every line at once, so the requests share batches and finish in reply-length order.
    std::vector<Turn> say_together(const std::vector<Line>& lines) {
        std::vector<Chorus::RequestId> ids;
        for (const auto& line : lines)
            ids.push_back(submit(line.session, line.prompt, line.max_tokens));
        std::vector<Turn> turns;
        std::unique_lock<std::mutex> lock(_mutex);
        for (const auto id : ids) {
            _cv.wait_for(lock, std::chrono::seconds(60), [&] { return _finished[id]; });
            Turn turn = _turns[id];
            turn.processed = _processed[id];
            turns.push_back(turn);
        }
        return turns;
    }

    bool loaded = false;

  private:
    Chorus::RequestId submit(const std::optional<std::string>& session, const std::string& prompt, int max_tokens) {
        Chorus::ChorusRequest request;
        request.id = ++_next_id;
        request.session_id = session;
        request.prompt = prompt;
        request.gen_config.max_tokens = max_tokens;
        request.gen_config.temperature = 0.0f;
        request.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
        request.on_event = [this, id = request.id](const Chorus::ChorusSignal& signal) {
            std::lock_guard<std::mutex> lock(_mutex);
            if (const auto* token = std::get_if<Chorus::ChorusSignal::Token>(&signal.event))
                _turns[id].text += token->text;
            else if (std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event))
                _turns[id].completed = _finished[id] = true;
            else if (std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
                _finished[id] = true;
            _cv.notify_all();
        };
        _engine.submit_request(request);
        return request.id;
    }

    std::mutex _mutex;
    std::condition_variable _cv;
    std::map<Chorus::RequestId, Turn> _turns;
    std::map<Chorus::RequestId, bool> _finished;
    std::map<Chorus::RequestId, int32_t> _processed;
    Chorus::RequestId _next_id = 0;
    Chorus::LlamaEngine _engine; // last, so its worker stops before the state its callbacks use
};

Conversations::Turn fresh(const std::string& prompt, int max_tokens = 8) {
    Conversations engine(1);
    return engine.say(std::nullopt, prompt, max_tokens);
}

std::vector<double> log_probabilities(const std::vector<float>& logits) {
    const double peak = *std::ranges::max_element(logits);
    double total = 0.0;
    for (const float logit : logits)
        total += std::exp(logit - peak);
    const double normalizer = peak + std::log(total);
    std::vector<double> result;
    result.reserve(logits.size());
    for (const float logit : logits)
        result.push_back(logit - normalizer);
    return result;
}

/*
 * Checks that a turn served from the cache predicts its first token as reprocessing does.
 *
 * Text comparison is too strict: a different batch shape rounds differently, and that flips the
 * choice between near-tied tokens. On x86 that rounding moved these log probabilities by up to
 * 0.04, while changing one early word of the history moved them by 0.1.
 */
testing::AssertionResult predicts_alike(const Conversations::Turn& cached, const Conversations::Turn& reprocessed) {
    constexpr size_t kLikelyTokens = 10;
    constexpr double kTolerance = 0.07;
    if (cached.first_logits.size() != reprocessed.first_logits.size() || cached.first_logits.empty())
        return testing::AssertionFailure() << "first-token logits missing";
    const auto expected = log_probabilities(reprocessed.first_logits);
    const auto actual = log_probabilities(cached.first_logits);
    std::vector<size_t> likely(expected.size());
    for (size_t token = 0; token < likely.size(); ++token)
        likely[token] = token;
    std::ranges::partial_sort(likely, likely.begin() + kLikelyTokens, [&](size_t a, size_t b) {
        return expected[a] > expected[b];
    });
    double divergence = 0.0;
    for (size_t rank = 0; rank < kLikelyTokens; ++rank)
        divergence = std::max(divergence, std::abs(actual[likely[rank]] - expected[likely[rank]]));
    if (divergence < kTolerance)
        return testing::AssertionSuccess();
    return testing::AssertionFailure() << "log probabilities of likely first tokens differ by " << divergence;
}

} // namespace

TEST(LlamaSessionCache, A_session_parks_in_one_slot_and_its_older_copy_is_handed_back) {
    Chorus::LlamaSessionCache cache;

    ASSERT_EQ(cache.park(0, "guard", {1, 2}), std::nullopt);
    ASSERT_EQ(cache.park(1, "smith", {3}), std::nullopt);
    ASSERT_EQ(cache.park(2, "guard", {1, 2, 4}), std::optional<int>{0});

    ASSERT_EQ(cache.slot_for("guard"), std::optional<int>{2});
    ASSERT_FALSE(cache.holds(0));
    ASSERT_EQ(cache.token_count(2), size_t{3});
    cache.move(1, 0);
    ASSERT_EQ(cache.slot_for("smith"), std::optional<int>{0});
    ASSERT_EQ(cache.take(0).tokens, (std::vector<int32_t>{3}));
}

TEST_F(LlamaSessionCacheModelTest, A_conversations_next_turn_processes_only_its_new_tokens) {
    Conversations npc(1);
    ASSERT_TRUE(npc.loaded);
    const std::string first = history_of(12);
    const auto reply = npc.say("guard", first);
    const std::string second = first + reply.text + "<end_of_turn>\n" + user_turn("And what happened at the keep?");

    const auto continued = npc.say("guard", second);
    const auto reprocessed = fresh(second);

    ASSERT_TRUE(reply.completed && continued.completed);
    ASSERT_LT(continued.processed * 3, reprocessed.processed);
    ASSERT_TRUE(predicts_alike(continued, reprocessed));
}

TEST_F(LlamaSessionCacheModelTest, An_edit_early_in_the_history_reprocesses_from_the_edit) {
    Conversations npc(1);
    ASSERT_TRUE(npc.loaded);
    const std::string original = history_of(12);
    npc.say("guard", original);
    std::string edited = original;
    edited.replace(edited.find("Day 6: the caravan"), 18, "Day 6: the airship");

    const auto continued = npc.say("guard", edited);
    const auto reprocessed = fresh(edited);

    ASSERT_TRUE(continued.completed);
    ASSERT_LT(continued.processed, reprocessed.processed);
    ASSERT_GT(continued.processed * 3, reprocessed.processed);
    ASSERT_TRUE(predicts_alike(continued, reprocessed));
}

TEST_F(LlamaSessionCacheModelTest, A_long_history_past_the_attention_window_still_continues_from_its_cache) {
    Conversations npc(1);
    ASSERT_TRUE(npc.loaded);
    const std::string first = history_of(70);
    const auto reply = npc.say("guard", first, 4);
    const std::string second = first + reply.text + "<end_of_turn>\n" + user_turn("Who told the best story?");

    const auto continued = npc.say("guard", second);
    const auto reprocessed = fresh(second);

    ASSERT_GT(reprocessed.processed, 1100); // past Gemma's 512-token window plus one micro-batch
    ASSERT_LT(continued.processed * 10, reprocessed.processed);
    ASSERT_TRUE(predicts_alike(continued, reprocessed));
}

TEST_F(LlamaSessionCacheModelTest, A_divergence_older_than_the_attention_window_reprocesses_everything) {
    Conversations npc(1);
    ASSERT_TRUE(npc.loaded);
    const std::string original = history_of(70);
    npc.say("guard", original, 4);
    std::string edited = original;
    edited.replace(edited.find("Day 2: the caravan"), 18, "Day 2: the airship");

    const auto continued = npc.say("guard", edited);
    const auto reprocessed = fresh(edited);

    ASSERT_EQ(continued.processed, reprocessed.processed);
    ASSERT_TRUE(predicts_alike(continued, reprocessed));
}

TEST_F(LlamaSessionCacheModelTest, A_conversation_whose_slot_was_needed_starts_over) {
    Conversations npcs(1);
    ASSERT_TRUE(npcs.loaded);
    const std::string guard = history_of(8);
    npcs.say("guard", guard);
    npcs.say("smith", history_of(9));
    const std::string follow_up = guard + user_turn("Anything else?");

    const auto returning = npcs.say("guard", follow_up);
    const auto reprocessed = fresh(follow_up);

    ASSERT_EQ(returning.processed, reprocessed.processed);
    ASSERT_TRUE(predicts_alike(returning, reprocessed));
}

TEST_F(LlamaSessionCacheModelTest, A_new_request_takes_an_empty_slot_and_leaves_parked_conversations_cached) {
    Conversations npcs(2);
    ASSERT_TRUE(npcs.loaded);
    const std::string guard = history_of(12);
    npcs.say("guard", guard);
    npcs.say(std::nullopt, history_of(3));
    const std::string follow_up = guard + user_turn("Anything else?");

    const auto returning = npcs.say("guard", follow_up);
    const auto reprocessed = fresh(follow_up);

    ASSERT_LT(returning.processed * 3, reprocessed.processed);
    ASSERT_TRUE(predicts_alike(returning, reprocessed));
}

TEST_F(LlamaSessionCacheModelTest, With_every_idle_slot_parked_a_short_conversation_gives_way_before_a_long_one) {
    Conversations npcs(2);
    ASSERT_TRUE(npcs.loaded);
    const std::string guard = history_of(40); // longer than one 512-token micro-batch
    const std::string smith = history_of(3);
    npcs.say("guard", guard);
    npcs.say("smith", smith);
    npcs.say(std::nullopt, history_of(4));
    const std::string guard_follow_up = guard + user_turn("Anything else?");
    const std::string smith_follow_up = smith + user_turn("Anything else?");

    const auto guard_returning = npcs.say("guard", guard_follow_up);
    const auto smith_returning = npcs.say("smith", smith_follow_up);

    ASSERT_LT(guard_returning.processed * 10, fresh(guard_follow_up).processed);
    ASSERT_EQ(smith_returning.processed, fresh(smith_follow_up).processed);
}

// With three slots the hole left by the middle conversation cannot be filled without dropping it, so a
// gap stays; with four, the parked conversation moves to the empty slot and the hole closes.
class LlamaSessionCacheSlotsTest : public ChorusModelTest, public ::testing::WithParamInterface<int64_t> {};

TEST_P(LlamaSessionCacheSlotsTest, Conversations_finishing_out_of_order_all_continue_from_their_caches) {
    Conversations npcs(GetParam());
    ASSERT_TRUE(npcs.loaded);
    const std::vector<std::string> sessions{"guard", "smith", "healer"};
    std::vector<std::string> histories;
    for (int index = 0; index < 3; ++index)
        histories.push_back(history_of(40 + index)); // each longer than one 512-token micro-batch
    // The middle conversation replies shortest, leaving a hole while the others still run.
    const auto first = npcs.say_together(
        {{sessions[0], histories[0], 24}, {sessions[1], histories[1], 4}, {sessions[2], histories[2], 16}}
    );

    std::vector<Conversations::Line> follow_ups;
    for (int index = 0; index < 3; ++index)
        follow_ups.push_back(
            {sessions[index], histories[index] + first[index].text + "<end_of_turn>\n" + user_turn("Anything else?")}
        );
    const auto second = npcs.say_together(follow_ups);

    for (int index = 0; index < 3; ++index) {
        const auto reprocessed = fresh(follow_ups[index].prompt);
        EXPECT_LT(second[index].processed * 3, reprocessed.processed) << sessions[index];
        EXPECT_TRUE(predicts_alike(second[index], reprocessed)) << sessions[index];
    }
}

INSTANTIATE_TEST_SUITE_P(SlotCounts, LlamaSessionCacheSlotsTest, ::testing::Values(int64_t{3}, int64_t{4}));

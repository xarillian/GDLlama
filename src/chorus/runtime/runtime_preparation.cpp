#include "chorus/runtime/runtime_preparation.hpp"
#include "chorus/runtime/prompt_fitting.hpp"

#include <algorithm>
#include <limits>

namespace Chorus {
namespace {
void check_cancelled(const std::atomic<bool>& cancelled) {
    if (cancelled)
        throw RequestRejection{ChorusError::Cancelled, "Request cancelled."};
}
} // namespace

int64_t ChorusRuntime::PreparationState::count_text(const std::string& text) {
    const size_t hash = std::hash<std::string>{}(text);
    const auto find = [&] {
        return std::find_if(content_counts.begin(), content_counts.end(), [&](const auto& entry) {
            return entry.hash == hash && entry.text == text;
        });
    };
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (auto found = find(); found != content_counts.end()) {
            content_counts.splice(content_counts.begin(), content_counts, found);
            return found->count;
        }
    }
    auto result = service->count_message_tokens(text);
    if (auto* failure = std::get_if<RequestRejection>(&result))
        throw *failure;
    const auto count = std::get<int64_t>(result);
    if (count < 0)
        throw RequestRejection{ChorusError::Tokenize, "Provider returned a negative message token count."};
    if (text.size() <= kContentCountCacheBytes) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (find() == content_counts.end()) {
            while (!content_counts.empty() && (content_counts.size() >= kContentCountCacheEntries ||
                                               content_bytes > kContentCountCacheBytes - text.size())) {
                content_bytes -= content_counts.back().text.size();
                content_counts.pop_back();
            }
            content_counts.push_front({hash, text, count});
            content_bytes += text.size();
        }
    }
    return count;
}

int64_t ChorusRuntime::PreparationState::count_node(const MessageNodePtr& node) {
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (auto found = node_counts.find(node->identity); found != node_counts.end()) {
            node_lru.splice(node_lru.begin(), node_lru, found->second.position);
            return found->second.count;
        }
    }
    auto text = joined_text(node->value.message.content);
    if (!text)
        throw RequestRejection{ChorusError::UnsupportedFeature, "Message content must be text."};
    // History entries retain only identities; arbitrary-text caching must not keep old history alive.
    auto result = service->count_message_tokens(*text);
    if (auto* failure = std::get_if<RequestRejection>(&result))
        throw *failure;
    const auto count = std::get<int64_t>(result);
    if (count < 0)
        throw RequestRejection{ChorusError::Tokenize, "Provider returned a negative message token count."};
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (!node_counts.contains(node->identity)) {
        if (node_counts.size() >= kMessageCountCacheEntries) {
            node_counts.erase(node_lru.back());
            node_lru.pop_back();
        }
        node_lru.push_front(node->identity);
        node_counts.emplace(node->identity, NodeCount{count, node_lru.begin()});
    }
    return count;
}

void ChorusRuntime::PreparationState::clear_caches() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    node_counts.clear();
    node_lru.clear();
    content_counts.clear();
    content_bytes = 0;
}

void ChorusRuntime::PreparationState::publish_signal(
    const ChorusSignal& signal, const std::shared_ptr<Control>& control
) {
    if (control->terminal || signal.request_id != control->id)
        return;
    publish(signal);
    if (std::holds_alternative<ChorusSignal::Stop>(signal.event) ||
        std::holds_alternative<ChorusSignal::Error>(signal.event))
        control->terminal = true;
}

void ChorusRuntime::PreparationState::finish(uint64_t ticket, Outcome outcome) {
    finished.emplace(ticket, std::move(outcome));
    for (auto next = finished.begin(); next != finished.end() && next->first == next_publication;
         next = finished.begin()) {
        Outcome ready = std::move(next->second);
        finished.erase(next);
        ++next_publication;
        auto& job = *ready.job;
        const auto control = job.control;
        if (ready.failure && !ready.abandoned)
            publish_signal(
                {job.request.id, ChorusSignal::Error{ready.failure->error, ready.failure->message}}, control
            );
        control->preparation_finished = true;
        if (ready.abandoned || ready.failure || control->terminal || control->cancelled) {
            release_preparation(control);
        } else if (job.operation == Operation::Count || job.operation == Operation::Preview) {
            RuntimeEvent event{
                job.request.id,
                job.request.session_id,
                job.operation == Operation::Count ? RuntimeEvent::Kind::MessageTokenCount
                                                  : RuntimeEvent::Kind::PromptRendered
            };
            event.text = std::move(job.rendered);
            event.token_count = job.token_count;
            event.omitted_message_ids = std::move(job.omitted);
            publish(std::move(event));
            control->terminal = true;
        } else {
            publish(std::move(ready.job));
        }
    }
}

void ChorusRuntime::fit_turn_messages(PreparationState& state, PreparationJob& job) {
    HistoryNodes nodes = *job.history;
    if (job.pending)
        nodes.push_back(job.pending);
    size_t pin = 0;
    while (pin < nodes.size() && nodes[pin]->value.message.role == MessageRole::System)
        ++pin;
    std::vector<size_t> boundaries{pin};
    if (pin < nodes.size()) {
        for (size_t start = pin + 1; start < nodes.size(); ++start)
            if (nodes[start]->value.message.role == MessageRole::User || start == nodes.size() - 1)
                boundaries.push_back(start);
    }
    const auto materialize = [&](size_t start) {
        check_cancelled(job.control->cancelled);
        std::vector<ChatMessage> messages;
        messages.reserve(pin + nodes.size() - start);
        for (size_t i = 0; i < pin; ++i)
            messages.push_back(nodes[i]->value.message);
        for (size_t i = start; i < nodes.size(); ++i)
            messages.push_back(nodes[i]->value.message);
        return place_injections(std::move(messages), job.resolved.request.inject);
    };
    if (!state.capabilities.prompt_rendering) {
        job.request.messages = materialize(pin);
        return;
    }
    std::optional<int64_t> budget;
    if (state.model_info && state.model_info->per_request_context) {
        const int64_t reservation = job.resolved.config.max_tokens && *job.resolved.config.max_tokens >= 0
                                        ? *job.resolved.config.max_tokens
                                        : 512;
        budget = static_cast<int64_t>(*state.model_info->per_request_context) - reservation;
        if (*budget <= 0)
            throw RequestRejection{
                ChorusError::InvalidRequest, "Response reservation leaves no prompt room in the per-request context."
            };
        job.request.exact_prompt_budget = budget;
    }
    size_t selected = 0;
    if (budget && state.capabilities.message_token_counting) {
        int64_t estimate = 0;
        const auto add = [&](int64_t count) {
            estimate = estimate > *budget || count > *budget - estimate ? *budget + 1 : estimate + count;
        };
        for (size_t i = 0; i < pin; ++i) {
            check_cancelled(job.control->cancelled);
            add(state.count_node(nodes[i]));
        }
        for (const auto& injection : job.resolved.request.inject) {
            check_cancelled(job.control->cancelled);
            auto text = joined_text(injection.message.content);
            if (!text)
                throw RequestRejection{ChorusError::UnsupportedFeature, "Injected content must be text."};
            add(state.count_text(*text));
        }
        if (pin < nodes.size()) {
            check_cancelled(job.control->cancelled);
            if (job.pending)
                add(state.count_text(joined_text(job.pending->value.message.content).value()));
            else
                add(state.count_node(nodes.back()));
        }
        selected = boundaries.size() - 1;
        while (selected > 0 && estimate <= *budget) {
            int64_t previous = estimate;
            for (size_t i = boundaries[selected]; i > boundaries[selected - 1];) {
                check_cancelled(job.control->cancelled);
                add(state.count_node(nodes[--i]));
                if (estimate > *budget)
                    break;
            }
            if (estimate > *budget) {
                estimate = previous;
                break;
            }
            --selected;
        }
    }
    const auto probe = [&](size_t index) {
        auto messages = materialize(boundaries[index]);
        check_cancelled(job.control->cancelled);
        auto result =
            state.service->render_chat_prompt(messages, job.resolved.chat_template, job.resolved.config.show_thinking);
        check_cancelled(job.control->cancelled);
        if (auto* failure = std::get_if<RequestRejection>(&result))
            throw *failure;
        auto rendered = std::get<RenderedPrompt>(std::move(result));
        if (rendered.token_count < 0)
            throw RequestRejection{ChorusError::Tokenize, "Provider returned a negative rendered token count."};
        if (budget && rendered.token_count > *budget)
            return false;
        job.request.messages = std::move(messages);
        job.rendered = std::move(rendered.text);
        for (size_t i = pin; i < boundaries[index]; ++i)
            if (nodes[i]->value.id >= 0)
                job.omitted.push_back(nodes[i]->value.id);
        return true;
    };
    for (size_t i = selected; i < boundaries.size(); ++i)
        if (probe(i))
            return;
    for (size_t i = 0; i < selected; ++i)
        if (probe(i))
            return;
    throw RequestRejection{
        ChorusError::InvalidRequest, "Conversation does not fit the context window even after truncation."
    };
}

void ChorusRuntime::prepare(PreparationState& state, PreparationJob& job) {
    check_cancelled(job.control->cancelled);
    if (job.operation == Operation::Count) {
        auto text = joined_text(job.content);
        if (!text)
            throw RequestRejection{ChorusError::UnsupportedFeature, "Message content must be text."};
        job.token_count = state.count_text(*text);
    } else {
        if (job.operation != Operation::Embed) {
            if (job.history)
                fit_turn_messages(state, job);
            else
                job.request.prompt = std::move(job.resolved.request.prompt);
            job.request.gen_config = std::move(job.resolved.config);
            job.request.chat_template = std::move(job.resolved.chat_template);
        }
        check_cancelled(job.control->cancelled);
        if (auto failure = state.service->validate_request(job.request))
            throw *failure;
        if (job.operation == Operation::Preview && !job.history)
            job.rendered = std::move(job.request.prompt);
    }
    check_cancelled(job.control->cancelled);
}

void ChorusRuntime::preparation_loop(const std::shared_ptr<PreparationState>& owned_state) {
    auto& state = *owned_state;
    std::unique_ptr<PreparationJob> job;
    uint64_t ticket = 0;
    try {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(state.mutex);
                state.cv.wait(lock, [&] { return state.closing || state.failed || !state.jobs.empty(); });
                if (state.closing || state.failed)
                    break;
                job = std::move(state.jobs.front());
                state.jobs.pop_front();
                ticket = state.next_ticket++;
            }
            std::optional<RequestRejection> failure;
            try {
                prepare(state, *job);
            } catch (const RequestRejection& rejection) {
                failure = rejection;
            }
            std::lock_guard<std::mutex> lock(state.mutex);
            state.finish(ticket, {std::move(job), std::move(failure)});
        }
    } catch (...) {
        fail_preparation(state);
        if (job) {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.finish(ticket, {std::move(job), std::nullopt, true});
        }
    }
    std::deque<std::unique_ptr<PreparationJob>> discarded;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        discarded.swap(state.jobs);
        for (const auto& queued : discarded) {
            queued->control->preparation_finished = true;
            state.release_preparation(queued->control);
        }
    }
}

void ChorusRuntime::fail_preparation(PreparationState& state) {
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.failed = true;
        for (const auto& [id, control] : state.controls) {
            control->cancelled = true;
            if (!control->provider_active && !control->terminal) {
                state.publish(
                    ChorusSignal{
                        id, ChorusSignal::Error{ChorusError::Unknown, "Preparation worker or provider failed."}
                    }
                );
                control->terminal = true;
            }
        }
    }
    state.cv.notify_all();
}

} // namespace Chorus

#include "llama_runner.hpp"
#include "logging_utils.hpp"
#include "llama.h"
#include "llama-sampling.h"
#include <chrono>
#include <common.h>
#include <log.h>
#include <fstream>
#include <filesystem>
#include <cstddef>
#include <functional>
#include <string>
#include <thread>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <log.h>
#include <random>

namespace {
    void log_tokenization_details(
        llama_context* ctx,
        const std::string& title,
        const std::string& text,
        bool add_bos
    ) {
        if (text.empty()) return;

        auto tokens = ::common_tokenize(ctx, text, add_bos, true);
        std::ostringstream oss;
        oss << title << " tokenized (" << tokens.size() << " tokens):\n";
        for (const auto& token : tokens) {
            oss << "  " << std::setw(6) << token << " -> '" << common_token_to_piece(ctx, token) << "'\n";
        }
        GDLOG_INFO(oss.str());
    }
}

LlamaRunner::LlamaRunner(
    bool should_output_prompt
) :
    should_stop_generation{false},
    is_waiting_input{false},
    input{""},
    should_output_prompt{should_output_prompt}
{ }

LlamaRunner::~LlamaRunner() {}

void LlamaRunner::llama_stop_generate_text() {
    should_stop_generation = true;
}

void LlamaRunner::set_input(std::string input) {
    this->input = input;
    is_waiting_input = false;
}

bool LlamaRunner::get_is_waiting_input() {
    return is_waiting_input;
}

std::string LlamaRunner::validate_params_for_initialization(const common_params &params) {
    if (params.ppl_output_type != 0) {
        return "Please use the 'perplexity' tool for perplexity calculations.";
    }

    if (params.embedding) {
        return "Please use the 'embedding' tool for embedding calculations.";
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        return "Context size must be 0 (default) or at least 8.";
    }

    if (params.rope_freq_base != 0.0) {
        std::string msg = "RoPE frequency base is being changed to " + std::to_string(params.rope_freq_base) + ". This may affect output quality.";
        GDLOG_WARN(std::string(__func__) + ": " + msg);
    }

    if (params.rope_freq_scale != 0.0) {
        std::string msg = "RoPE frequency is being scaled by " + std::to_string(params.rope_freq_scale) + ". This may affect output quality.";
        GDLOG_WARN(std::string(__func__) + ": " + msg);
    }

    return "";
}

std::string LlamaRunner::llama_generate_text(
    std::string prompt,
    common_params params,
    std::function<void(std::string)> on_generate_text_updated,
    std::function<void()> on_input_wait_started,
    std::function<void(std::string)> on_generate_text_finished
){
    GDLOG_DEBUG("Start");
    GDLOG_INFO("User Prompt: " + prompt);

    std::string generated_text = "";
    bool is_interacting = false;

    params.prompt = prompt;

    // --- 1. Pre-Flight Validation

    const std::string validation_err = validate_params_for_initialization(params);
    if (!validation_err.empty()) {
        GDLOG_ERROR(validation_err);
        on_generate_text_finished(validation_err);
        return validation_err;
    }

    GDLOG_DEBUG("Pre-flight validation successful");

    // --- 2. Model and Context Initialization

    if (params.sampling.seed == LLAMA_DEFAULT_SEED) {
        params.sampling.seed = time(NULL);
        std::mt19937 rng(params.sampling.seed);
    }
    GDLOG_INFO("Using seed: " + std::to_string(params.sampling.seed));

    GDLOG_DEBUG("Calling llama_backend_init()");
    llama_backend_init();
    GDLOG_DEBUG("llama_backend_init() completed successfully");
    GDLOG_DEBUG("Calling llama_numa_init() with strategy: " + std::to_string(params.numa));
    llama_numa_init(params.numa);
    GDLOG_DEBUG("llama_numa_init() completed successfully");

    GDLOG_DEBUG("Model Params: \n" + std::to_string(params));

    GDLOG_DEBUG("Calling common_init_from_params()");
    common_init_result result = common_init_from_params(params);
    GDLOG_DEBUG("common_init_from_params() completed");
    llama_model * model = result.model.get();
    llama_context * ctx = result.context.get();

    if (model == NULL) {
        std::string msg = "Failed to load model from path: " + params.model.path;
        GDLOG_ERROR(msg);
        on_generate_text_finished(msg);
        return msg;
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        std::string msg = "Model was trained on only " + std::to_string(n_ctx_train) + 
                          " context tokens (" + std::to_string(n_ctx) + " specified).";
        GDLOG_WARN(msg);
    }

    GDLOG_INFO(common_params_get_system_info(params).c_str());
    GDLOG_DEBUG("Model initialized");

    // --- 3. Session File Loading

    std::string session_path = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!session_path.empty()) {
        GDLOG_INFO("Attempting to load session from '" + session_path + "'");
        if (!std::filesystem::exists((session_path))) {
            GDLOG_WARN("Session file does not exist, a new one will be created.");
        } else if (std::filesystem::is_empty((session_path))) {
            GDLOG_WARN("Session file is empty, a new session will be initialized.");
        } else {
            // The session file exists and is not empty.
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, session_path.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                std::string msg = "Failed to load session file: " + session_path;
                GDLOG_ERROR(std::string(__func__) + ": " + msg);
                on_generate_text_finished(msg);
                return msg;
            }

            session_tokens.resize(n_token_count_out);
            GDLOG_INFO("Loaded session with " + std::to_string(session_tokens.size()) + " tokens.");
        }
    }

    GDLOG_DEBUG("Session Path Loaded");

    // --- 4. Prompt Tokenization and Validation

    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    GGML_ASSERT(llama_vocab_get_add_eos(llama_model_get_vocab(model)) != 1);
    GDLOG_DEBUG("add_bos token setting: " + std::to_string(add_bos));

    std::vector<llama_token> prompt_tokens;  // Embedding Input

    if (params.interactive_first || !params.prompt.empty() || session_tokens.empty()) {
        GDLOG_INFO("Tokenizing the prompt.");
        prompt_tokens = ::common_tokenize(ctx, params.prompt, true, true);
    } else {
        GDLOG_INFO("Using session tokens as prompt.");
        prompt_tokens = session_tokens;
    }

    GDLOG_INFO("Tokenized User Prompt: " + string_from(ctx, prompt_tokens));

    if (prompt_tokens.empty()) {
        // We should not run without any tokens. An empty prompt is an invalid state.
        std::string msg = "Prompt is empty. Generation requires at least one token.";
        GDLOG_ERROR(std::string(__func__) + ": " + msg);
        on_generate_text_finished(msg);
        return msg;
    }

    const int max_prompt_tokens = n_ctx - 4; // Reserve 4 tokens for generation
    const bool prompt_is_too_long = (int)prompt_tokens.size() > max_prompt_tokens;
    if (prompt_is_too_long) {
        std::string msg = "Prompt is longer than parameters allow. Max allowable tokens: " + 
                          std::to_string(max_prompt_tokens) + 
                          ",  prompt has " + std::to_string(prompt_tokens.size()) + " tokens.";
        GDLOG_ERROR(std::string(__func__) + ": " + msg);
        on_generate_text_finished(msg);
        return msg;
    }

    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= prompt_tokens.size() || id != prompt_tokens[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }

        if (params.prompt.empty() && n_matching_session_tokens == prompt_tokens.size()) {
            GDLOG_INFO("Using full prompt from session file.");
        } else if (n_matching_session_tokens >= prompt_tokens.size()) {
            GDLOG_INFO("Session file has an exact match for the prompt.");
        } else if (n_matching_session_tokens < (prompt_tokens.size() / 2)) {
            GDLOG_WARN(
                std::string(__func__) + ": Session file has low similarity to prompt (" + 
                std::to_string(n_matching_session_tokens) + " / " + 
                std::to_string(prompt_tokens.size()) + " tokens); will mostly be reevaluated."
            );
        } else {
            GDLOG_INFO(
                "Session file matches " + std::to_string(n_matching_session_tokens) + " / " + 
                std::to_string(prompt_tokens.size()) + " tokens of prompt."
            );
        }

        // Remove any "future" tokens that we might have inherited from the previous session.
        if (llama_get_memory(ctx)) {
            llama_pos p0 = n_matching_session_tokens;
            llama_pos p1 = -1;  // until end of sequence
            llama_memory_seq_rm(llama_get_memory(ctx), -1, p0, p1);
        }
    }

    // When a prompt is a prefix of a longer cached session, the KV cache already contains the
    // processed state of the prompt. However, the logits for the *next* token are stale. To fix 
    // this, we re-evaluate the final token of the prompt to generate fresh logits for the
    // prediction loop.
    if (n_matching_session_tokens == prompt_tokens.size() && session_tokens.size() > prompt_tokens.size()) {
        GDLOG_WARN("Forcing re-evaluation of last token to recalculate cached logits.");
        session_tokens.resize(prompt_tokens.size() - 1);
    }

    // The number of tokens that should be permanently kept in the conext during context 
    // shifting (or, "infinite generation")
    if (params.n_keep < 0 || params.n_keep > (int) prompt_tokens.size()) {
        params.n_keep = (int)prompt_tokens.size();
    } else {
        params.n_keep += add_bos;
    }

    // TODO I have no idea if conversation modes are working: https://github.com/xarillian/GDLlama/issues/11
    switch (params.conversation_mode) {
        case COMMON_CONVERSATION_MODE_ENABLED:
            params.interactive_first = true;
            GDLOG_INFO("Conversation mode: Enabled (interactive_first=true)");
            break;
        case COMMON_CONVERSATION_MODE_AUTO:
            GDLOG_INFO("Conversation mode: Auto");
            break;
        case COMMON_CONVERSATION_MODE_DISABLED:
        default:
            GDLOG_INFO("Conversation mode: Disabled");
            break;
    }

    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        GDLOG_INFO("Verbose prompt enabled.");
        
        // This log is inefficient, but it is likely a user would expect slower output when
        //increasing verbosity anyway.
        log_tokenization_details(ctx, "Initial prompt", params.prompt, true);

        if (params.n_keep > add_bos) {
            std::string static_prompt;
            for (int i = 0; i < params.n_keep; i++) {
                static_prompt += common_token_to_piece(ctx, prompt_tokens[i]);
            }
            GDLOG_INFO("Static prompt based on n_keep: '" + static_prompt + "'");
        }
    }

    if (params.interactive) {
        GDLOG_INFO("Interactive mode on.");

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                GDLOG_INFO("Reverse prompt: '" + antiprompt + "'");
                if (params.verbose_prompt) {
                    log_tokenization_details(ctx, " Antiprompt details: ", antiprompt, false);
                }
            }
        }
        if (params.input_prefix_bos) {
            GDLOG_INFO("Input prefix with BOS was used.");
        }
        if (!params.input_prefix.empty()) {
            GDLOG_INFO("Input prefix: '" + params.input_prefix + "'");
            if (params.verbose_prompt) {
                log_tokenization_details(
                    ctx,
                    " Input prefix details: ",
                    params.input_prefix,
                    true
                );
            }
        }
        if (!params.input_suffix.empty()) {
            GDLOG_INFO("Input suffix: '" + params.input_suffix + "'");
        }
        if (!params.input_suffix.empty()) {
            GDLOG_INFO("Input suffix: '" + params.input_suffix + "'");
            if (params.verbose_prompt) {
                log_tokenization_details(ctx, "  Input suffix details: ", params.input_suffix, false);
            }
        }
    }

    GDLOG_INFO("Sampling parameters: " + params.sampling.print());
    GDLOG_INFO(
        "Generation parameters: \n"
        "n_ctx = " + std::to_string(n_ctx) + ",\n"
        "n_batch = " + std::to_string(params.n_batch) + ",\n" 
        "n_predict = " + std::to_string(params.n_predict) + ",\n" 
        "n_keep = " + std::to_string(params.n_keep)
    );

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;
    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;


    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0 && "grp_attn_n must be positive");
        GGML_ASSERT(ga_w % ga_n == 0 && "grp_attn_w must be a multiple of grp_attn_n");
        GDLOG_INFO(
            "Self-extend enabled: n_ctx_train = " + std::to_string(n_ctx_train) + 
            ", grp_attn_n = " + std::to_string(ga_n) + ", grp_attn_w = " + std::to_string(ga_w)
        );
    }

    if (params.interactive) {
        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    bool input_echo           = should_output_prompt;
    bool display              = true;
    bool need_to_save_session = !session_path.empty() && n_matching_session_tokens < prompt_tokens.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;
    int n_past_guidance    = 0;

    std::vector<int>   input_tokens;
    std::vector<int>   output_tokens;
    std::ostringstream output_ss;

    display = params.display_prompt;

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance;

    // tokenized antiprompts
    std::vector<std::vector<llama_token>> antiprompt_ids;

    antiprompt_ids.reserve(params.antiprompt.size());
    for (const std::string & antiprompt : params.antiprompt) {
        antiprompt_ids.emplace_back(::common_tokenize(ctx, antiprompt, false, true));
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * ctx_sampling = llama_sampler_chain_init(sparams);

    if (params.sampling.top_k > 0) {
        llama_sampler_chain_add(ctx_sampling, llama_sampler_init_top_k(params.sampling.top_k));
    }
    if (params.sampling.top_p < 1.0f) {
        llama_sampler_chain_add(ctx_sampling, llama_sampler_init_top_p(params.sampling.top_p, 1));
    }
    if (params.sampling.temp > 0.0f) {
        llama_sampler_chain_add(ctx_sampling, llama_sampler_init_temp(params.sampling.temp));
    }

    llama_sampler_chain_add(ctx_sampling, llama_sampler_init_dist(params.sampling.seed));

    std::vector<llama_token> generated_tokens_history;
    const int n_prev = 32;

    GDLOG_INFO("Starting generation loop.");

    while (!should_stop_generation && ((n_remain != 0 && !is_antiprompt) || params.interactive)) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                embd.resize(max_embd_size);
            }

            if (ga_n == 1) {
                // Infinite Text Generation via Context Shifting
                // If we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
                if (n_past + (int)embd.size() >= n_ctx) {
                    if (params.n_predict == -2) {
                        // LOG("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left = n_past - params.n_keep;
                    const int n_discard = n_left / 2;

                    GDLOG_INFO( 
                        "Context is full, swapping tokens to make space. "
                        "n_past=" + std::to_string(n_past) + 
                        ", n_left=" + std::to_string(n_left) + 
                        ", n_ctx=" + std::to_string(n_ctx) + 
                        ", n_keep=" + std::to_string(params.n_keep) + 
                        ", n_discard=" + std::to_string(n_discard)
                    );

                    llama_memory_t mem = llama_get_memory(ctx);
                    llama_memory_seq_rm(mem, 0, params.n_keep, params.n_keep + n_discard);
                    llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    GDLOG_INFO(
                        "After swap: "
                        "n_past = " + std::to_string(n_past) +
                        ", n_past_guidance = " + std::to_string(n_past_guidance)
                    );
                    session_path.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    GDLOG_INFO("Performing self-extend context shift.");

                    llama_memory_t mem = llama_get_memory(ctx);
                    if (mem) {
                        llama_memory_seq_add(mem, 0, ga_i, n_past, ib*bd);
                        llama_memory_seq_div(mem, 0, ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n);
                        llama_memory_seq_add(mem, 0, ga_i + ib*bd + ga_w, n_past + ib*bd, dd);
                    }

                    n_past -= bd;
                    ga_i += ga_w / ga_n;
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    GDLOG_INFO("Reusing " + std::to_string(i) + " tokens from session cache.");
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                std::vector<llama_pos> pos_array(n_eval);
                for (int j = 0; j < n_eval; j++) {
                    pos_array[j] = n_past + j;
                }

                llama_batch batch = llama_batch_get_one(&embd[i], n_eval);
                batch.pos = pos_array.data();

                const int32_t decode_result = llama_decode(ctx, batch);
                if (decode_result != 0) {
                    std::string error_msg;
                    bool is_fatal = true;

                    switch (decode_result) {
                        case 1:
                            // This is a recoverable warning: the context is simply full.
                            // We won't recover, however. The user should be responsible for 
                            // managing context size.
                            error_msg = "Could not find a KV cache slot for the batch. The context window is likely full.";
                            break;
                        case -1:
                            error_msg = "Failed to initialize the token batch for decoding. This can be caused by an invalid prompt (e.g., one with unusual characters) or incorrect generation parameters. Please check your input.";
                            break;
                        case -2:
                            error_msg = "Memory allocation failed during decoding. The model may be too large for your system memory/VRAM.";
                            break;
                        case -3:
                            error_msg = "A fatal computation error occurred during decoding. This may be caused by a corrupted model file, incompatible hardware, or outdated drivers. Try re-downloading the model file and updating your graphics drivers, and may God be with you.";
                            break;
                        default:
                            error_msg = "An unknown error occurred during decoding. Code: " + std::to_string(decode_result);
                            break;
                    }

                    GDLOG_ERROR(std::string(__func__) + ": " + error_msg);
                    on_generate_text_finished(error_msg);
                    return error_msg;
                }

                n_past += n_eval;

                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    GDLOG_INFO("Tokens consumed so far = " + std::to_string(n_past) + " / " + std::to_string(n_ctx));
                }
            }

            if (!embd.empty() && !session_path.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();
        embd_guidance.clear();

        if ((int) prompt_tokens.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!session_path.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, session_path.c_str(), session_tokens.data(), session_tokens.size());

                GDLOG_INFO("Saved session to " + session_path);
            }


            const llama_token token_id = llama_sampler_sample(ctx_sampling, ctx, -1);
            llama_sampler_accept(ctx_sampling, token_id);

            generated_tokens_history.push_back(token_id);
            if (generated_tokens_history.size() > static_cast<size_t>(n_prev)) {
                generated_tokens_history.erase(generated_tokens_history.begin());
            }

            embd.push_back(token_id);

            input_echo = true;
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            // LOG("prompt_tokens.size(): %d, n_consumed: %d\n", (int) prompt_tokens.size(), n_consumed);
            while ((int) prompt_tokens.size() > n_consumed) {
                embd.push_back(prompt_tokens[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                const llama_token token_id = llama_sampler_sample(ctx_sampling, ctx, -1);
                llama_sampler_accept(ctx_sampling, token_id);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                generated_text.append(token_str);
                on_generate_text_updated(token_str);

                // Console/Stream Output
                fprintf(stdout, "%s", token_str.c_str());

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(id);
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }

                fflush(stdout);
            }
        }

        // reset color to default if there is no pending user input
        if (input_echo && (int) prompt_tokens.size() == n_consumed) {
            display = true;
        }

        // if not currently processing queued inputs;
        if ((int) prompt_tokens.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                std::string last_output = "";
                for (llama_token token : generated_tokens_history) {
                    last_output += common_token_to_piece(ctx, token);
                }

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                if (!generated_tokens_history.empty()) {
                    llama_token last_token = generated_tokens_history.back();
                    for (std::vector<llama_token> ids : antiprompt_ids) {
                        if (ids.size() == 1 && last_token == ids[0]) {
                            if (params.interactive) {
                                is_interacting = true;
                            }
                            is_antiprompt = true;
                            break;
                        }
                    }
                }
                if (is_antiprompt) {
                    GDLOG_INFO("Found antiprompt: " + last_output);
                }
            }

            // deal with end of generation tokens in interactive mode
            if (!generated_tokens_history.empty() && llama_vocab_is_eog(llama_model_get_vocab(model), generated_tokens_history.back())) {
                GDLOG_INFO("Found EOG token.");
                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        const auto first_antiprompt = ::common_tokenize(
                            ctx, 
                            params.antiprompt.front(),
                            false,
                            true
                        );

                        prompt_tokens.insert(
                            prompt_tokens.end(),
                            first_antiprompt.begin(),
                            first_antiprompt.end()
                        );

                        is_antiprompt = true;
                    }
                    is_interacting = true;
                    printf("\n");
                }
            }

            if (n_past > 0 && is_interacting) {
                if (params.conversation_mode == COMMON_CONVERSATION_MODE_ENABLED) { printf("\n> "); }
                if (params.input_prefix_bos) { prompt_tokens.push_back(llama_token_bos(llama_model_get_vocab(model))); }

                std::string buffer;
                if (!params.input_prefix.empty() && params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
                    printf("%s", params.input_prefix.c_str());
                }

                is_waiting_input = true;
                on_input_wait_started();

                GDLOG_INFO("Waiting for user input...");
                while(is_waiting_input && !should_stop_generation) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                GDLOG_INFO("User input received.");

                buffer = input;

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty() && params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
                        printf("%s", params.input_suffix.c_str());
                    }
    
                    const size_t original_size = prompt_tokens.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    const auto line_pfx = ::common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = ::common_tokenize(ctx, buffer,              false, false);
                    const auto line_sfx = ::common_tokenize(ctx, params.input_suffix, false, true);

                    GDLOG_INFO("Tokenized user input: " + string_from(ctx, line_inp));

                    prompt_tokens.insert(prompt_tokens.end(), line_pfx.begin(), line_pfx.end());
                    prompt_tokens.insert(prompt_tokens.end(), line_inp.begin(), line_inp.end());
                    prompt_tokens.insert(prompt_tokens.end(), line_sfx.begin(), line_sfx.end());

                    for (size_t i = original_size; i < prompt_tokens.size(); ++i) {
                        const llama_token token = prompt_tokens[i];
                        output_tokens.push_back(token);
                        output_ss << common_token_to_piece(ctx, token);
                    }

                    n_remain -= line_inp.size();
                } else {
                    GDLOG_WARN("Empty line received, passing control back.");
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                if (is_interacting) {
                    llama_sampler_reset(ctx_sampling);
                }
                is_interacting = false;
            }
        }

        // End of Generation

        if (!embd.empty() && llama_vocab_is_eog(llama_model_get_vocab(model), embd.back()) && !(params.interactive)) {
            GDLOG_INFO("End of text reached.");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if (!session_path.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        GDLOG_INFO("Saving final output to session file '" + session_path + "'");
        llama_state_save_file(ctx, session_path.c_str(), session_tokens.data(), session_tokens.size());
    }

    llama_free(ctx);
    llama_model_free(model);

    llama_sampler_free(ctx_sampling);
    llama_backend_free();

    GDLOG_INFO("Generated Text: " + std::string(generated_text));

    on_generate_text_finished(generated_text);

    GDLOG_DEBUG("Done");
    return generated_text;
}

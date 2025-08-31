#include "llama_runner.hpp"
#include "llama.h"
#include "llama-sampling.h"
#include <chrono>
#include <common.h>
#include <log.h>
#include <fstream>
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

LlamaRunner::LlamaRunner(
    bool should_output_prompt,
    std::function<void(std::string)> glog
) : should_stop_generation {false},
    is_waiting_input {false},
    input {""},
    should_output_prompt {should_output_prompt},
    glog {glog}
{
    common_log_set_file(common_log_main(), "llama.log");
}

LlamaRunner::~LlamaRunner() { }

bool LlamaRunner::file_exists(const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
}

bool LlamaRunner::file_is_empty(const std::string &path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

void LlamaRunner::llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    LOG("%s", text);
}

std::string LlamaRunner::llama_generate_text(
    std::string prompt, common_params params,
    std::function<void(std::string)> on_generate_text_updated,
    std::function<void()> on_input_wait_started,
    std::function<void(std::string)> on_generate_text_finished
){
    std::string generated_text = "";

    params.prompt = prompt;
    bool is_interacting = false;

    #ifndef LOG_DISABLE_LOGS
        LOG("Log start\n");
    #endif

    if (params.ppl_output_type != 0) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        std::string msg = std::string(__func__) + ": please use the 'perplexity' tool for perplexity calculations";
        glog(msg);
        on_generate_text_finished(msg);
        return msg;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        std::string msg = std::string(__func__) + ": please use the 'embedding' tool for embedding calculations";
        glog(msg);
        on_generate_text_finished(msg);
        return msg;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG("%s: build = %d (%s)\n",      __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    if (params.sampling.seed == LLAMA_DEFAULT_SEED) {
        params.sampling.seed = time(NULL);
    }

    LOG("%s: seed  = %u\n", __func__, params.sampling.seed);

    std::mt19937 rng(params.sampling.seed);

    LOG("%s: llama backend init\n", __func__);
    llama_backend_init();
    llama_numa_init(params.numa);

    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result result = common_init_from_params(params);
    llama_model * model = result.model.get();
    llama_context * ctx = result.context.get();
    
    if (model == NULL) {
        LOG("%s: error: unable to load model\n", __func__);
        std::string msg = std::string(__func__) + ": error: unable to load model";
        glog(msg);
        on_generate_text_finished(msg);
        return msg;
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train) {
        LOG("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    LOG("\n");
    LOG("%s\n", common_params_get_system_info(params).c_str());

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG("%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(path_session)) {
            LOG("%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG("%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                std::string msg = std::string(__func__) + ": error: failed to load session file " + path_session;
                glog(msg);
                on_generate_text_finished(msg);
                return msg;
            }
            session_tokens.resize(n_token_count_out);
            LOG("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    GGML_ASSERT(llama_vocab_get_add_eos(llama_model_get_vocab(model)) != 1);
    LOG("add_bos: %d\n", add_bos);

    std::vector<llama_token> embd_inp;

    if (params.interactive_first || !params.prompt.empty() || session_tokens.empty()) {
        LOG("tokenize the prompt\n");
        embd_inp = ::common_tokenize(ctx, params.prompt, true, true);
    } else {
        LOG("use session tokens\n");
        embd_inp = session_tokens;
    }

    LOG("prompt: \"%s\"\n", params.prompt.c_str());
    LOG("tokens: %s\n", string_from(ctx, embd_inp).c_str());

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_vocab_bos(llama_model_get_vocab(model)));
        LOG("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
    }

    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        std::string msg = std::string(__func__) + ": error: prompt is too long (" + std::to_string((int) embd_inp.size()) + " tokens, max " + std::to_string(n_ctx - 4) + ")";
        glog(msg);
        on_generate_text_finished(msg);
        return msg;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG("%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        if (llama_get_memory(ctx)) {
            // p0 = start position
            // p1 = end position
            llama_pos p0 = n_matching_session_tokens;
            llama_pos p1 = -1;  // until end of sequence

            llama_memory_seq_rm(llama_get_memory(ctx), -1, p0, p1);
        }
    }

    LOG_DBG(
        "recalculate the cached logits (check): embd_inp.empty() %s, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu, embd_inp.size() %zu",
        embd_inp.empty() ? "true" : "false", n_matching_session_tokens, embd_inp.size(), session_tokens.size(), embd_inp.size()
    );

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    switch (params.conversation_mode) {
        case COMMON_CONVERSATION_MODE_ENABLED:
            params.interactive_first = true;
            break;
        case COMMON_CONVERSATION_MODE_AUTO:
            // Let llama.cpp handle this automatically
            break;
        case COMMON_CONVERSATION_MODE_DISABLED:
        default:
            // Explicitly don't enable conversation features
            break;
    }

    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG("\n");
        LOG("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            LOG("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG("'\n");
        }
        LOG("\n");
    }

    if (params.interactive) {
        LOG("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = ::common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        LOG("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG("Input prefix: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = ::common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG("Input suffix: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = ::common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    LOG("sampling: \n%s\n", params.sampling.print().c_str());
    LOG("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
      //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
      //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG("\n\n");

    if (params.interactive) {
        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    bool input_echo           = should_output_prompt;
    bool display              = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

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
                        LOG("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left = n_past - params.n_keep;
                    const int n_discard = n_left / 2;

                    LOG(
                        "context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                        n_past, n_left, n_ctx, params.n_keep, n_discard
                    );

                    llama_memory_t mem = llama_get_memory(ctx);
                    llama_memory_seq_rm(mem, 0, params.n_keep, params.n_keep + n_discard);
                    llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    LOG("after swap: n_past = %d, n_past_guidance = %d\n", n_past, n_past_guidance);
                    LOG("embd: %s\n", string_from(ctx, embd).c_str());
                    LOG("clear session path\n");
                    path_session.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG("\n");
                    LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_memory_t mem = llama_get_memory(ctx);
                    if (mem) {
                        llama_memory_seq_add(mem, 0, ga_i, n_past, ib*bd);
                        llama_memory_seq_div(mem, 0, ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n);
                        llama_memory_seq_add(mem, 0, ga_i + ib*bd + ga_w, n_past + ib*bd, dd);
                    }

                    n_past -= bd;
                    ga_i += ga_w/ga_n;

                    LOG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
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
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG("eval: %s\n", string_from(ctx, embd).c_str());

                std::vector<llama_pos> pos_array(n_eval);
                for (int j = 0; j < n_eval; j++) {
                    pos_array[j] = n_past + j;
                }

                llama_batch batch = llama_batch_get_one(&embd[i], n_eval);
                batch.pos = pos_array.data();

                if (llama_decode(ctx, batch)) {
                    LOG("%s : failed to eval\n", __func__);
                    std::string msg = std::string(__func__) + ": failed to eval";
                    glog(msg);
                    on_generate_text_finished(msg);
                    return msg;
                }

                n_past += n_eval;

                LOG("n_past = %d\n", n_past);
                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();
        embd_guidance.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG("saved session to %s\n", path_session.c_str());
            }


            const llama_token token_id = llama_sampler_sample(ctx_sampling, ctx, -1);
            llama_sampler_accept(ctx_sampling, token_id);

            generated_tokens_history.push_back(token_id);
            if (generated_tokens_history.size() > static_cast<size_t>(n_prev)) {
                generated_tokens_history.erase(generated_tokens_history.begin());
            }

            embd.push_back(token_id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

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
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            //console::set_display(console::reset);
            display = true;
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
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
                    LOG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of generation tokens in interactive mode
            if (
                !generated_tokens_history.empty() &&
                llama_vocab_is_eog(llama_model_get_vocab(model), generated_tokens_history.back())
            ) {
                LOG("found an EOG token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = ::common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    is_interacting = true;
                    printf("\n");
                }
            }

            if (n_past > 0 && is_interacting) {
                if (params.conversation_mode == COMMON_CONVERSATION_MODE_ENABLED) {
                    printf("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_token_bos(llama_model_get_vocab(model)));
                }

                std::string buffer;
                if (!params.input_prefix.empty() && params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
                    LOG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    printf("%s", params.input_prefix.c_str());
                }

                is_waiting_input = true;
                on_input_wait_started();

                LOG("Waiting for user input\n");
                while(is_waiting_input && !should_stop_generation) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                LOG("Ending user input\n");

                buffer = input;

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty() && params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
                        LOG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        printf("%s", params.input_suffix.c_str());
                    }

                    LOG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    const auto line_pfx = ::common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = ::common_tokenize(ctx, buffer,              false, false);
                    const auto line_sfx = ::common_tokenize(ctx, params.input_suffix, false, true);

                    LOG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << common_token_to_piece(ctx, token);
                    }

                    n_remain -= line_inp.size();
                    LOG("n_remain: %d\n", n_remain);
                } else {
                    LOG("empty line, passing control back\n");
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

        // end of generation
        if (
            !embd.empty() &&
            llama_vocab_is_eog(llama_model_get_vocab(model), embd.back()) &&
            !(params.interactive)) 
        {
            LOG(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    llama_free(ctx);
    llama_model_free(model);

    llama_sampler_free(ctx_sampling);
    llama_backend_free();

#ifndef LOG_DISABLE_LOGS
    LOG("Log end\n");
#endif // LOG_DISABLE_LOGS

    on_generate_text_finished(generated_text);

    return generated_text;
}

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

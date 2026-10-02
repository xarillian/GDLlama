#pragma once

#include "llama.h"

/*
 * Silences llama.cpp's log for as long as it is alive.
 *
 * Fixtures in this directory load a model directly rather than through
 * `LlamaScheduler`, so `Chorus::LlamaLogBridge` never installs and llama's
 * default callback sprays a model's whole boot log across the suite's output.
 * Hold one of these beside the model to keep a test's output its own.
 *
 * Passing a null callback would not do it: `llama_log_set` reads null as
 * "restore the default", so an explicit no-op is the only way to say nothing.
 * The previous callback is restored on destruction, because llama's hook is
 * process-global and the bridge's own suite asserts on what it finds there.
 */
class SilentLlamaLog {
  public:
    SilentLlamaLog() {
        llama_log_get(&_previous_callback, &_previous_user_data);
        llama_log_set(discard, nullptr);
    }

    ~SilentLlamaLog() { llama_log_set(_previous_callback, _previous_user_data); }

    SilentLlamaLog(const SilentLlamaLog&) = delete;
    SilentLlamaLog& operator=(const SilentLlamaLog&) = delete;

  private:
    static void discard(ggml_log_level, const char*, void*) {}

    ggml_log_callback _previous_callback = nullptr;
    void* _previous_user_data = nullptr;
};

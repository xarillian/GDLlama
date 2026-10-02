#include "ggml-backend.h"
#include "llama-batch.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <unordered_set>

static std::unordered_set<void*> buffers;
static std::unordered_set<void*> events;
static std::unordered_set<void*> synchronized_events;
static std::unordered_set<void*> backends;
static int allocations, event_allocations, backend_allocations, records, synchronizations, backend_synchronizations,
    buffer_frees, event_frees, backend_frees;
static int injected_failure, failure_count, frees_before_sync;
static bool measuring, throw_next_allocation;
static bool measuring_batch;
static int batch_allocation_attempts, batch_fail_at, batch_allocations, batch_frees;
static void* batch_handles[4096];

void* operator new(std::size_t size) {
    if (throw_next_allocation) {
        throw_next_allocation = false;
        throw std::bad_alloc();
    }
    if (measuring_batch && ++batch_allocation_attempts == batch_fail_at)
        throw std::bad_alloc();
    void* p = std::malloc(size ? size : 1);
    if (!p)
        throw std::bad_alloc();
    if (measuring_batch) {
        if (batch_allocations == 4096)
            std::abort();
        batch_handles[batch_allocations++] = p;
    }
    return p;
}
void operator delete(void* p) noexcept {
    for (int i = 0; i < batch_allocations; ++i) {
        if (batch_handles[i] == p) {
            batch_handles[i] = nullptr;
            ++batch_frees;
            break;
        }
    }
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    ::operator delete(p);
}

extern "C" ggml_backend_buffer_t __real_ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t, size_t);
extern "C" ggml_backend_buffer_t __wrap_ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t type, size_t size) {
    if (measuring && size == 1024 * 1024 && injected_failure == 1 && ++failure_count == 2)
        return nullptr;
    auto* p = __real_ggml_backend_buft_alloc_buffer(type, size);
    if (measuring && p && size == 1024 * 1024) {
        buffers.insert(p);
        ++allocations;
        if (injected_failure == 4 && allocations == 1)
            throw_next_allocation = true;
    }
    return p;
}
extern "C" void __real_ggml_backend_buffer_free(ggml_backend_buffer_t);
extern "C" void __wrap_ggml_backend_buffer_free(ggml_backend_buffer_t p) {
    if (measuring && buffers.erase(p)) {
        ++buffer_frees;
        if (records && !backend_synchronizations)
            ++frees_before_sync;
    }
    __real_ggml_backend_buffer_free(p);
}
extern "C" ggml_backend_event_t __real_ggml_backend_event_new(ggml_backend_dev_t);
extern "C" ggml_backend_event_t __wrap_ggml_backend_event_new(ggml_backend_dev_t dev) {
    if (measuring && injected_failure == 2 && ++failure_count == 2)
        return nullptr;
    auto* p = __real_ggml_backend_event_new(dev);
    if (measuring && p) {
        events.insert(p);
        ++event_allocations;
    }
    return p;
}
extern "C" void __real_ggml_backend_event_synchronize(ggml_backend_event_t);
extern "C" void __wrap_ggml_backend_event_synchronize(ggml_backend_event_t p) {
    if (measuring && events.count(p)) {
        ++synchronizations;
        synchronized_events.insert(p);
    }
    __real_ggml_backend_event_synchronize(p);
}
extern "C" void __real_ggml_backend_event_free(ggml_backend_event_t);
extern "C" void __wrap_ggml_backend_event_free(ggml_backend_event_t p) {
    if (measuring && events.erase(p)) {
        ++event_frees;
        if (!synchronized_events.erase(p) || (records && !backend_synchronizations))
            ++frees_before_sync;
    }
    __real_ggml_backend_event_free(p);
}
extern "C" void __real_ggml_backend_event_record(ggml_backend_event_t, ggml_backend_t);
extern "C" void __wrap_ggml_backend_event_record(ggml_backend_event_t e, ggml_backend_t b) {
    if (measuring && events.count(e))
        ++records;
    __real_ggml_backend_event_record(e, b);
}
extern "C" ggml_backend_t __real_ggml_backend_dev_init(ggml_backend_dev_t, const char*);
extern "C" ggml_backend_t __wrap_ggml_backend_dev_init(ggml_backend_dev_t d, const char* p) {
    if (measuring && injected_failure == 3 && event_allocations == 4)
        return nullptr;
    auto* b = __real_ggml_backend_dev_init(d, p);
    if (measuring && b) {
        backends.insert(b);
        ++backend_allocations;
    }
    return b;
}
extern "C" void __real_ggml_backend_synchronize(ggml_backend_t);
extern "C" void __wrap_ggml_backend_synchronize(ggml_backend_t p) {
    if (measuring && backends.count(p))
        ++backend_synchronizations;
    __real_ggml_backend_synchronize(p);
}
extern "C" void __real_ggml_backend_free(ggml_backend_t);
extern "C" void __wrap_ggml_backend_free(ggml_backend_t p) {
    if (measuring && backends.erase(p)) {
        ++backend_frees;
        if (records && !backend_synchronizations)
            ++frees_before_sync;
    }
    __real_ggml_backend_free(p);
}

struct Attempt {
    const char* name;
    bool cancel_after_upload;
    bool cancel_final;
    int callbacks;
    bool cancelled;
    float cancel_fraction;
};
static bool callback(float progress, void* data) {
    auto& a = *static_cast<Attempt*>(data);
    ++a.callbacks;
    if ((a.cancel_after_upload && records > 0 && progress < 1.0f) || (a.cancel_final && progress == 1.0f)) {
        a.cancelled = true;
        a.cancel_fraction = progress;
        return false;
    }
    return true;
}
static void report(const Attempt& a, bool loaded) {
    std::printf(
        "PROBE name=%s loaded=%d callbacks=%d cancelled=%d fraction=%.6f allocations=%d buffer_frees=%d "
        "buffers_live=%zu events=%d event_frees=%d events_live=%zu records=%d synchronizations=%d "
        "backend_synchronizations=%d backend_allocations=%d backend_frees=%d backends_live=%zu frees_before_sync=%d\n",
        a.name,
        loaded,
        a.callbacks,
        a.cancelled,
        a.cancel_fraction,
        allocations,
        buffer_frees,
        buffers.size(),
        event_allocations,
        event_frees,
        events.size(),
        records,
        synchronizations,
        backend_synchronizations,
        backend_allocations,
        backend_frees,
        backends.size(),
        frees_before_sync
    );
    std::fflush(stdout);
}
static int check_batch_conversion(const char* model_path) {
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(model_path, params);
    if (!model)
        return 3;
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 32;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    auto* ctx = llama_init_from_model(model, context_params);
    if (!ctx) {
        llama_model_free(model);
        return 3;
    }
    llama_token tokens[8];
    for (auto& token : tokens)
        token = llama_vocab_bos(llama_model_get_vocab(model));
    auto batch = llama_batch_get_one(tokens, 8);
    int failures = 0;
    for (int fail_at : {2, 8, 16}) {
        batch_allocation_attempts = batch_allocations = batch_frees = 0;
        batch_fail_at = fail_at;
        measuring_batch = true;
        bool threw = false;
        try {
            llama_batch_compat conversion(ctx, batch);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        measuring_batch = false;
        std::printf(
            "BATCH fault=%d threw=%d allocations=%d frees=%d live=%d\n",
            fail_at,
            threw,
            batch_allocations,
            batch_frees,
            batch_allocations - batch_frees
        );
        std::fflush(stdout);
        if (!threw || batch_allocations == 0 || batch_allocations != batch_frees)
            ++failures;
    }
    const bool decoded =
        llama_decode(ctx, llama_batch_get_one(tokens, 1)) == 0 && llama_get_logits_ith(ctx, 0) != nullptr;
    std::printf("BATCH decode=%d\n", decoded);
    llama_free(ctx);
    llama_model_free(model);
    return failures || !decoded ? 3 : 0;
}

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    const bool batch_fault = std::strcmp(argv[2], "batch-fault") == 0;
    if (batch_fault) {
        llama_backend_init();
        const int result = check_batch_conversion(argv[1]);
        llama_backend_free();
        return result;
    }
    const bool early = std::strcmp(argv[2], "early") == 0;
    const bool final = std::strcmp(argv[2], "final") == 0;
    const bool success = std::strcmp(argv[2], "success") == 0;
    const bool setup = std::strcmp(argv[2], "buffer-fail") == 0 || std::strcmp(argv[2], "event-fail") == 0 ||
                       std::strcmp(argv[2], "backend-fail") == 0;
    const bool exception = std::strcmp(argv[2], "allocation-throw") == 0;
    const bool mixed = std::strcmp(argv[2], "mixed") == 0;
    if (!early && !final && !success && !setup && !exception && !mixed)
        return 2;
    injected_failure = std::strcmp(argv[2], "buffer-fail") == 0    ? 1
                       : std::strcmp(argv[2], "event-fail") == 0   ? 2
                       : std::strcmp(argv[2], "backend-fail") == 0 ? 3
                       : exception                                 ? 4
                                                                   : 0;
    llama_backend_init();
    for (int i = 0; i < (mixed ? 5 : early ? 3 : 1); ++i) {
        buffers.clear();
        events.clear();
        synchronized_events.clear();
        backends.clear();
        allocations = event_allocations = backend_allocations = records = synchronizations = backend_synchronizations =
            buffer_frees = event_frees = backend_frees = failure_count = frees_before_sync = 0;
        const bool attempt_early = early || (mixed && i < 3);
        const bool attempt_final = final || (mixed && i == 3);
        const bool attempt_success = success || (mixed && i == 4);
        Attempt a{
            attempt_early     ? "early"
            : attempt_final   ? "final"
            : attempt_success ? "success"
                              : argv[2],
            attempt_early,
            attempt_final,
            0,
            false,
            -1.0f
        };
        auto params = llama_model_default_params();
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.n_gpu_layers = -1;
        params.progress_callback = callback;
        params.progress_callback_user_data = &a;
        measuring = true;
        auto* model = llama_model_load_from_file(argv[1], params);
        measuring = false;
        bool used = false;
        if (model && attempt_success) {
            auto cp = llama_context_default_params();
            cp.n_ctx = 128;
            cp.n_batch = 32;
            cp.n_threads = 2;
            cp.n_threads_batch = 2;
            auto* ctx = llama_init_from_model(model, cp);
            if (ctx) {
                auto token = llama_vocab_bos(llama_model_get_vocab(model));
                if (token >= 0)
                    used = llama_decode(ctx, llama_batch_get_one(&token, 1)) == 0 &&
                           llama_get_logits_ith(ctx, 0) != nullptr;
                llama_free(ctx);
            }
        }
        const bool loaded = model != nullptr;
        report(a, loaded);
        if (model)
            llama_model_free(model);
        std::printf("USE name=%s decode=%d\n", a.name, used);
        std::fflush(stdout);
        if (buffers.size() || events.size() || backends.size() || buffer_frees != allocations ||
            event_frees != event_allocations || backend_frees != backend_allocations || frees_before_sync ||
            (setup ? !loaded
             : exception
                 ? (loaded || allocations != 1)
                 : (allocations != 4 || event_allocations != 4 || backend_allocations == 0 || records == 0 ||
                    !backend_synchronizations || a.cancelled == attempt_success || (attempt_success && !used)))) {
            std::fprintf(stderr, "PROBE_INCOMPLETE name=%s\n", a.name);
            return 3;
        }
    }
    llama_backend_free();
    return 0;
}

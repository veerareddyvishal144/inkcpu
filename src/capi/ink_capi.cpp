// C ABI (include/ink/ink_abi.h) on top of ink::Engine. Owner: agent C1; speculative-decoding additions (ink_spec_*, ink_sample) by agent S2. All calls must come from ONE thread.
#include "ink/ink_abi.h"

#include <cstring>
#include <memory>
#include <string>

#include "ink/engine.h"
#ifdef INK_HAVE_SPEC
#include "ink/spec.h"
#endif

struct ink_engine {
    std::unique_ptr<ink::Engine> e;
    double total_us = 0, kernel_us = 0;
#ifdef INK_HAVE_SPEC
    std::unique_ptr<ink::SpecDecoder> spec;   // created lazily (plain steps) or by ink_spec_load (with the drafter)
    int spec_mode = 1;                        // default mode of requests with mode = -1 (env INK_SPEC overrides at decoder creation)
#endif
};

namespace {
std::string g_err;
void set_err(const std::string& s) { g_err = s; }
}  // namespace

extern "C" {

ink_engine* ink_load(const char* gguf_path, int n_threads, int n_slots, int max_ctx) {
    if (!gguf_path) {
        set_err("null path");
        return nullptr;
    }
    ink::EngineOptions o;
    o.n_threads = n_threads > 0 ? n_threads : 12;
    o.n_slots = n_slots > 0 ? n_slots : 1;
    o.max_ctx = max_ctx > 0 ? max_ctx : 4096;
    std::string err;
    std::unique_ptr<ink::Engine> e;
    try {
        e = ink::Engine::create(gguf_path, o, &err);
    } catch (const std::exception& ex) {
        err = std::string("exception: ") + ex.what();
    }
    if (!e) {
        set_err(err.empty() ? "load failed" : err);
        return nullptr;
    }
    auto* h = new ink_engine();
    h->e = std::move(e);
    return h;
}

void ink_free(ink_engine* h) {
    if (h && h->e && h->e->profiling()) h->e->profile_report(stderr);
#ifdef INK_HAVE_SPEC
    if (h) h->spec.reset();   // before the engine (the decoder removes the tap hook and frees the drafter)
#endif
    delete h;
}

const char* ink_last_error(void) { return g_err.c_str(); }
const char* ink_version(void) {
#ifdef INK_HAVE_SPEC
    return "inkcpu 0.2 (decode executor C1, batched prefill P, speculative decoding S2; AVX-512 kernels, spin-barrier pool)";
#else
    return "inkcpu 0.1 (decode executor C1; AVX-512 kernels, spin-barrier pool)";
#endif
}

int ink_n_vocab(const ink_engine* h) { return h && h->e ? h->e->n_vocab() : 0; }

int ink_slot_new(ink_engine* h) { return h && h->e ? h->e->slot_new() : -1; }
void ink_slot_free(ink_engine* h, int slot) {
    if (h && h->e) h->e->slot_free(slot);
}
void ink_slot_reset(ink_engine* h, int slot) {
    if (h && h->e) h->e->slot_reset(slot);
}
int ink_slot_pos(const ink_engine* h, int slot) { return h && h->e ? (int)h->e->slot_pos(slot) : -1; }

int ink_forward(ink_engine* h, int slot, const int32_t* tokens, int n, float* logits_out) {
    if (!h || !h->e || !tokens) {
        set_err("bad arguments");
        return -10;
    }
    int rc;
    try {
        rc = h->e->forward(slot, tokens, n, logits_out);
    } catch (const std::exception& ex) {
        set_err(std::string("exception: ") + ex.what());
        return -11;
    }
    if (rc) set_err("ink_forward failed: code " + std::to_string(rc) + " (-1 bad slot, -2 context full, -3 bad token, -4 n < 1)");
    h->total_us = h->e->last_forward_us;   // all tokens of the call; every microsecond is spent inside the executor
    h->kernel_us = h->e->last_forward_us;
    return rc;
}

void ink_last_timing(const ink_engine* h, double* total_us, double* kernel_us) {
    if (total_us) *total_us = h ? h->total_us : 0;
    if (kernel_us) *kernel_us = h ? h->kernel_us : 0;
}

// ======================================================================= slot checkpoints and rewind (agent S3, docs/REWIND.md)
int ink_slot_checkpoint(ink_engine* h, int slot) { return h && h->e ? h->e->slot_checkpoint(slot, false) : -10; }
int ink_slot_checkpoint_deep(ink_engine* h, int slot) { return h && h->e ? h->e->slot_checkpoint(slot, true) : -10; }
int64_t ink_slot_rewind(ink_engine* h, int slot, int64_t pos) { return h && h->e ? h->e->slot_rewind(slot, pos) : -10; }
int ink_slot_checkpoints(const ink_engine* h, int slot, int64_t* out, int max) { return h && h->e ? h->e->slot_checkpoints(slot, out, max) : 0; }

// ======================================================================= speculative decoding
#ifdef INK_HAVE_SPEC
namespace {
ink::SamplingParams to_sp(const ink_sampling_params* p) {
    ink::SamplingParams sp;
    if (p) {
        sp.temperature = p->temperature;
        sp.top_k = p->top_k;
        sp.top_p = p->top_p;
        sp.min_p = p->min_p;
    }
    return sp;
}
ink::SpecDecoder& ensure_spec(ink_engine* h, std::unique_ptr<ink::Drafter> drafter) {
    ink::SpecOptions o;
    o.mode = h->spec_mode;
    h->spec.reset();   // releases the previous drafter / tap hook first
    h->spec = std::make_unique<ink::SpecDecoder>(*h->e, std::move(drafter), o);
    h->spec_mode = h->spec->options().mode;   // env INK_SPEC applied
    return *h->spec;
}
}  // namespace

int ink_spec_load(ink_engine* h, const char* drafter_gguf) {
    if (!h || !h->e || !drafter_gguf) {
        set_err("bad arguments");
        return -10;
    }
#ifdef INK_HAVE_DRAFTER
    std::string err;
    std::unique_ptr<ink::Drafter> d;
    try {
        d = ink::Drafter::create(*h->e, drafter_gguf, &err);
    } catch (const std::exception& ex) {
        err = std::string("exception: ") + ex.what();
    }
    if (!d) {
        set_err(err.empty() ? "drafter load failed" : err);
        return -12;
    }
    ensure_spec(h, std::move(d));
    return 0;
#else
    set_err("libink was built without the drafter module (src/drafter)");
    return -13;
#endif
}

void ink_spec_free(ink_engine* h) {
    if (h && h->e) ensure_spec(h, nullptr);
}

int ink_spec_enable(ink_engine* h, int mode) {
    if (!h || !h->e || mode < 0 || mode > 8) {
        set_err("bad arguments");
        return -10;
    }
    h->spec_mode = mode;
    if (h->spec) h->spec->options().mode = mode;
    return 0;
}

int ink_spec_mode(const ink_engine* h) { return h && h->e ? (h->spec ? h->spec->options().mode : ink::spec_mode_from_env(h->spec_mode)) : -1; }

int ink_spec_synced(const ink_engine* h, int slot) { return h && h->e && h->spec && h->spec->synced(slot) ? 1 : 0; }

int ink_spec_step(ink_engine* h, int slot, const ink_spec_request* rq, ink_spec_result* out) {
    if (!h || !h->e || !rq || !out) {
        set_err("bad arguments");
        return -10;
    }
    std::memset(out, 0, sizeof(*out));
    if (!h->spec) ensure_spec(h, nullptr);
    ink::SpecRequest q;
    q.anchor = rq->anchor;
    q.sp = to_sp(&rq->sp);
    q.seed = rq->seed;
    q.stop_ids = rq->n_stop > 0 ? rq->stop_ids : nullptr;
    q.n_stop = rq->n_stop > 0 && rq->stop_ids ? rq->n_stop : 0;
    q.max_emit = rq->max_emit;
    q.mode = rq->mode;
    ink::SpecResult r;
    int rc;
    try {
        rc = h->spec->step(slot, q, &r);
    } catch (const std::exception& ex) {
        set_err(std::string("exception: ") + ex.what());
        return out->rc = -11;
    }
    out->rc = rc;
    if (rc) {
        set_err("ink_spec_step failed: code " + std::to_string(rc) + " (-1 bad slot, -2 context full, -3 bad token, -4 bad request)");
        return rc;
    }
    out->n_tokens = r.n_tokens;
    for (int i = 0; i < r.n_tokens; ++i) out->tokens[i] = r.tokens[i];
    out->n_fed = r.n_tokens;   // rows processed: the anchor and every emitted token except the last one (the next anchor or the stop id)
    out->fed[0] = rq->anchor;
    for (int i = 1; i < r.n_tokens; ++i) out->fed[i] = r.tokens[i - 1];
    out->stopped = r.stopped;
    out->T = r.T;
    out->drafted = r.drafted;
    out->accepted = r.accepted;
    out->draft_us = r.draft_us;
    out->verify_us = r.verify_us;
    out->accept_us = r.accept_us;
    out->total_us = r.total_us;
    out->next_seed = r.next_seed;
    h->total_us = r.total_us;
    h->kernel_us = r.verify_us + r.draft_us;
    return 0;
}

int32_t ink_sample(const ink_engine* h, const float* logits, const ink_sampling_params* p, uint64_t seed, uint64_t* next_seed) {
    if (!h || !h->e || !logits) {
        set_err("bad arguments");
        return -1;
    }
    static thread_local std::unique_ptr<ink::Sampler> smp;
    if (!smp || smp->n_vocab() != h->e->n_vocab()) smp = std::make_unique<ink::Sampler>(h->e->n_vocab());
    ink::SpecRng rng(seed);
    const int32_t t = smp->sample(logits, to_sp(p), rng);
    if (next_seed) *next_seed = rng.next_seed();
    return t;
}
#else
int ink_spec_load(ink_engine*, const char*) {
    set_err("libink was built without the spec module");
    return -13;
}
void ink_spec_free(ink_engine*) {}
int ink_spec_enable(ink_engine*, int) { return -13; }
int ink_spec_mode(const ink_engine*) { return -1; }
int ink_spec_synced(const ink_engine*, int) { return 0; }
int ink_spec_step(ink_engine*, int, const ink_spec_request*, ink_spec_result* out) {
    if (out) {
        std::memset(out, 0, sizeof(*out));
        out->rc = -13;
    }
    set_err("libink was built without the spec module");
    return -13;
}
int32_t ink_sample(const ink_engine*, const float*, const ink_sampling_params*, uint64_t, uint64_t*) { return -1; }
#endif

}  // extern "C"

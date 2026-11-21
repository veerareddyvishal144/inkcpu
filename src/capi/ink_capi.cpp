// C ABI (include/ink/ink_abi.h) on top of ink::Engine. Owner: agent C1. All calls must come from ONE thread.
#include "ink/ink_abi.h"

#include <cstring>
#include <memory>
#include <string>

#include "ink/engine.h"

struct ink_engine {
    std::unique_ptr<ink::Engine> e;
    double total_us = 0, kernel_us = 0;
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
    delete h;
}

const char* ink_last_error(void) { return g_err.c_str(); }
const char* ink_version(void) { return "inkcpu 0.1 (decode executor C1; AVX-512 kernels, spin-barrier pool)"; }

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
    h->total_us = h->e->last_total_us;
    h->kernel_us = h->e->last_total_us;
    return rc;
}

void ink_last_timing(const ink_engine* h, double* total_us, double* kernel_us) {
    if (total_us) *total_us = h ? h->total_us : 0;
    if (kernel_us) *kernel_us = h ? h->kernel_us : 0;
}

}  // extern "C"

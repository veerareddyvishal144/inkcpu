// Shared helpers of the spec tests (agent S2): engine creation on the S2 CPU set, prompt files, bit comparison, plain greedy reference, test-double drafters.
#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ink/drafter.h"
#include "ink/engine.h"
#include "ink/engine_prefill.h"
#include "ink/spec.h"

#ifndef INK_SOURCE_DIR
#define INK_SOURCE_DIR "."
#endif

namespace sut {

using namespace ink;

inline int g_fail = 0;
#define SUT_CHECK(cond, ...)                                  \
    do {                                                      \
        if (!(cond)) {                                        \
            std::printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                         \
            std::printf("\n");                                \
            std::fflush(stdout);                              \
            ++sut::g_fail;                                    \
        }                                                     \
    } while (0)

inline std::vector<int32_t> read_i32(const std::string& p) {
    std::vector<int32_t> v;
    if (FILE* f = std::fopen(p.c_str(), "rb")) {
        int32_t x;
        while (std::fread(&x, 4, 1, f) == 1) v.push_back(x);
        std::fclose(f);
    }
    return v;
}
inline void write_i32(const std::string& p, const std::vector<int32_t>& v) {
    if (FILE* f = std::fopen(p.c_str(), "wb")) {
        std::fwrite(v.data(), 4, v.size(), f);
        std::fclose(f);
    }
}
inline const std::vector<std::string>& prompt_names() {
    // the first 8 are the DSpark prompts (tools/dspark/prompts); the lookup_* prompts (agent L: copy-heavy editor / tool-call turns, tests/spec/prompts/lookup_*.txt) follow,
    // so that SPEC_PROMPTS <= 8 and the pooled bench numbers over the first 8 are unchanged
    static const std::vector<std::string> n = {"code_completion", "tool_call", "chat_code", "reasoning", "chat_answer", "prose_continuation", "chat_email", "long_context_qa",
                                               "lookup_docstring", "lookup_toolcall", "lookup_typo"};
    return n;
}
inline std::vector<int32_t> prompt(const std::string& name) { return read_i32(std::string(INK_SOURCE_DIR) + "/tests/spec/prompts/" + name + ".i32"); }
inline std::string data_dir() {
    const char* d = std::getenv("SPEC_DATA");
    return d ? d : std::string(INK_SOURCE_DIR) + "/build-s2-b/data";
}
inline int env_int(const char* n, int d) {
    const char* v = std::getenv(n);
    return v && *v ? std::atoi(v) : d;
}

inline std::unique_ptr<Engine> make_engine(int n_slots, int max_ctx = 4096, int threads = 6) {
    EngineOptions o;
    o.n_threads = env_int("INK_THREADS", threads);
    o.n_slots = n_slots;
    o.max_ctx = max_ctx;
    std::string err;
    auto e = Engine::create("/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf", o, &err);
    if (!e) {
        std::fprintf(stderr, "engine load failed: %s\n", err.c_str());
        std::exit(1);
    }
    return e;
}
inline bool bits_equal(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }
inline double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}
inline int argmax(const float* x, int n) { return Sampler::argmax(x, n); }
inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline std::string load1() {
    FILE* f = std::fopen("/proc/loadavg", "r");
    char b[64] = {0};
    if (f) {
        if (!std::fgets(b, sizeof b, f)) b[0] = 0;
        std::fclose(f);
    }
    char* sp = std::strchr(b, ' ');
    if (sp) *sp = 0;
    return b;
}

// Plain greedy reference: prefill `prompt` into `slot` (Engine::forward) and decode greedily; gen[0] = argmax of the prefill logits (the first anchor),
// gen[i] = the token after feeding gen[i-1]. The slot ends with the prompt and gen[0..n-2] processed (gen[n-1] pending). Cached in SPEC_DATA/ref_<name>_<n>.i32 (tokens only).
inline std::vector<int32_t> plain_greedy(Engine& E, int slot, const std::vector<int32_t>& prompt, int n) {
    std::vector<float> lg((size_t)E.n_vocab());
    std::vector<int32_t> gen;
    E.forward(slot, prompt.data(), (int)prompt.size(), lg.data());
    gen.push_back(argmax(lg.data(), E.n_vocab()));
    for (int i = 1; i < n; ++i) {
        E.decode(slot, gen.back(), lg.data());
        gen.push_back(argmax(lg.data(), E.n_vocab()));
    }
    return gen;
}

inline bool kv_cells_equal(const KvLayer& a, const KvLayer& b, int64_t p0, int64_t p1, int* first_bad_g = nullptr) {
    const int64_t ring = a.ring ? (int64_t)a.n_blocks * kKvBlock : 0;
    for (int64_t pos = p0; pos < p1; ++pos) {
        const int64_t slot = ring ? pos % ring : pos;
        const size_t blk = (size_t)(slot / kKvBlock), kib = (size_t)(slot % kKvBlock);
        for (int g = 0; g < kNKvHeads; ++g)
            for (int d = 0; d < kHeadDim; ++d) {
                const size_t ik = blk * kKvBlockElems + ((size_t)g * kHeadDim + d) * kKvBlock + kib;
                const size_t iv = blk * kKvBlockElems + ((size_t)g * kKvBlock + kib) * kHeadDim + d;
                if (a.k[ik] != b.k[ik] || a.v[iv] != b.v[iv]) {
                    if (first_bad_g) *first_bad_g = (int)pos;
                    return false;
                }
            }
    }
    return true;
}

inline bool slots_equal(Engine& E, int sa, int sb, std::string* why) {
    const SlotState& a = E.slots[sa];
    const SlotState& b = E.slots[sb];
    if (a.pos != b.pos) {
        *why = "pos " + std::to_string(a.pos) + " vs " + std::to_string(b.pos);
        return false;
    }
    if (std::memcmp(a.hist, b.hist, (size_t)kVisits * kHistPerVisit * sizeof(float)) != 0) {
        *why = "conv history differs";
        return false;
    }
    for (int v = 0; v < kVisits; ++v) {
        const int64_t lo = a.kv[v].ring ? std::max<int64_t>(0, a.pos - (kSwaWindow - 1)) : 0;
        int bad = -1;
        if (!kv_cells_equal(a.kv[v], b.kv[v], lo, a.pos, &bad)) {
            *why = "KV of visit " + std::to_string(v) + (a.kv[v].ring ? " (ring)" : " (global)") + " differs at position " + std::to_string(bad);
            return false;
        }
    }
    return true;
}


// ---------------------------------------------------------------------------------------------------- test-double drafters (need no S1 code)
// A sequence-bound double: `truth` = prompt ++ plain greedy continuation, indexed by position. The anchor sits at position state_pos(st); draft i (1-based) is the token at
// position pos + i. ORACLE returns exactly that (full accept), WRONG returns tokens that differ from it (all reject), NOISY replaces each draft by a random wrong token with
// probability `err` (so the accepted prefix length varies over 0..n). Sample mode returns dense q rows: 0.6 on the proposed token, 0.4 spread over 8 fixed tokens.
class DoubleDrafter : public Drafter {
public:
    enum Kind { ORACLE, WRONG, NOISY };
    DoubleDrafter(Kind k, int n_vocab, double err = 0.3, uint64_t seed = 1) : kind(k), V(n_vocab), err_(err), rng_(seed) {}
    Kind kind;
    int V;
    std::vector<int32_t> truth;
    int n_commits = 0;
    int64_t rows_committed = 0;
    int state_new() override {
        pos_.push_back(0);
        return (int)pos_.size() - 1;
    }
    void state_free(int) override {}
    void state_reset(int st) override { pos_[(size_t)st] = 0; }
    int64_t state_pos(int st) const override { return pos_[(size_t)st]; }
    int commit(int st, const float* taps, int rows) override {
        (void)taps;
        pos_[(size_t)st] += rows;
        ++n_commits;
        rows_committed += rows;
        return 0;
    }
    int draft(int st, const DraftRequest& rq, int32_t* out, float* q_out) override {
        const int64_t p = pos_[(size_t)st];
        uint64_t s = rq.seed ^ 0x9E3779B97F4A7C15ull;
        auto rnd = [&]() {
            s = spec_mix64(s);
            return (double)(s >> 11) * (1.0 / 9007199254740992.0);
        };
        for (int i = 0; i < rq.n; ++i) {
            const size_t at = (size_t)(p + 1 + i);
            int32_t t = at < truth.size() ? truth[at] : 7;
            if (kind == WRONG || (kind == NOISY && next_u() < err_)) t = wrong_of(t);
            if (rq.sample && q_out) {   // dense q: 0.6 on t, 0.4 over 8 fixed tokens; draw d_i from it
                float* q = q_out + (size_t)i * V;
                std::memset(q, 0, (size_t)V * sizeof(float));
                q[t] += 0.6f;
                for (int j = 0; j < 8; ++j) q[(size_t)((j * 7919 + 13) % V)] += 0.05f;
                const double u = rnd();
                double c = 0;
                int32_t d = t;
                for (int j = 0; j < V; ++j)
                    if (q[j] > 0) {
                        c += q[j];
                        d = j;
                        if (c > u) break;
                    }
                out[i] = d;
            } else {
                out[i] = t;
            }
        }
        return 0;
    }

private:
    double next_u() {
        rng_ = spec_mix64(rng_);
        return (double)(rng_ >> 11) * (1.0 / 9007199254740992.0);
    }
    int32_t wrong_of(int32_t t) {
        rng_ = spec_mix64(rng_);
        int32_t w = (int32_t)(rng_ % (uint64_t)V);
        return w == t ? (int32_t)((w + 1) % V) : w;
    }
public:
    void set_pos(int st, int64_t p) { pos_[(size_t)st] = p; }
private:
    std::vector<int64_t> pos_;
    double err_;
    uint64_t rng_;
};

}  // namespace sut

// Batched prefill executor: ONE ThreadPool::run() per chunk of up to 1024 tokens, phases separated by pool.barrier(). Owner: agent P.
// Same visit structure as the decode executor (engine_decode.cpp) but every phase works on n tokens: token-parallel mixes / norms /
// quantisation, GEMM items (matrix x row chunk) over all tokens for the dense projections, per KV group conv/clip/append, attention per token
// range, grouped MoE GEMMs (expert x row chunk) with the SiTU + Q8_0 epilogue fused into the gate_up items, and a per-channel-range combine of
// the 8 expert outputs in top-k order. See docs/PREFILL_DESIGN.md. Control flow is data independent so the dynamic-scheduling counters (one per
// phase, numbered identically on every thread) line up, exactly as in the decode executor.
#include <pthread.h>
#include <sched.h>
#include <x86intrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <map>
#include <unordered_map>

#include "ink/engine_prefill.h"
#include "ink/kernels_gemm.h"
#include "ink/kernels_misc_simd.h"

namespace ink {

namespace {

constexpr float kMixScale = 1.0f / 48.0f;   // 1/sqrtf(2304), exact
constexpr float kSituBeta = 4.0f, kSituLin = 25.0f;
constexpr int kMaxPh = 16384;               // dynamic-scheduling counters per chunk
constexpr int kMaxTh = 64;
constexpr int kQB = kD / 32;                // Q8_0 blocks per 2304 row (72)
constexpr int kQWB = kQW / 32;              // 96
constexpr int kFB = kFF / 32;               // 36
constexpr int kGuW = 64;                    // max gate rows per MoE item (gate + up tile = 2 * kGuW floats per pair)

struct alignas(64) PfCounter {
    std::atomic<int64_t> v;
};

enum PfPhase {
    PF_EMBED = 0, PF_COPY, PF_MIX, PF_CTRL, PF_PROJ, PF_CONV, PF_ATTN, PF_WO, PF_FFNMIX, PF_ROUTER, PF_ROUTE, PF_GATEUP, PF_PACKDN, PF_DOWN,
    PF_COMBINE, PF_FINAL, PF_LMHEAD, PF_COUNT
};
const char* const kPfNames[PF_COUNT] = {"embed",
                                        "prelude/bank copies",
                                        "attn-in mix+norm+quant",
                                        "loop controllers (batched)",
                                        "proj q,k,v,gate,rel,sink GEMM",
                                        "conv/clip/kv_append",
                                        "attention + quant",
                                        "wo GEMM + post-conv + resid",
                                        "ffn mix+norm+quant",
                                        "router GEMM + top-8",
                                        "route build (serial)",
                                        "MoE gate_up + SiTU + q8",
                                        "MoE pack down act",
                                        "MoE down GEMM",
                                        "combine + post-conv + resid",
                                        "final mix/norm/quant",
                                        "lm_head + softcap"};

struct PfProfile {
    uint64_t ticks[PF_COUNT] = {}, calls[PF_COUNT] = {}, macs[PF_COUNT] = {}, bytes[PF_COUNT] = {};
    double cpu_ns[PF_COUNT] = {};                         // thread CPU time spent in the phase body, summed over threads (excludes barrier waits)
    uint64_t tokens = 0, chunks = 0, total_ticks = 0;
};

struct PItem {
    int m, r0, r1;
};

struct PfScratch {
    int cap = 0;
    void* arena = nullptr;
    // residual stream and attention-residual state, [cap][kD]
    float *cur[2] = {}, *blk[kMaxBlocks] = {}, *prelude = nullptr, *bank_b = nullptr, *bank_out = nullptr;
    float *msgs[4] = {}, *mix[3] = {}, *cin[4] = {};
    block_q8_0* inq[3] = {};
    float* inbf[3] = {};
    void* act_in[3] = {};
    float *ctrl_in[3] = {}, *ctrl_g[3] = {}, *ctrl_gain[3] = {}, *ctrl_forget[3] = {};
    float *q = nullptr, *kp = nullptr, *vp = nullptr, *rel = nullptr, *gate = nullptr, *sink = nullptr, *kdyn = nullptr, *vdyn = nullptr,
          *kc = nullptr, *vc = nullptr, *tau = nullptr;
    float* ao = nullptr;
    block_q8_0* aoq = nullptr;
    void* act_wo = nullptr;
    float *wo_out = nullptr, *attn_out = nullptr, *fmix = nullptr, *xffn = nullptr;
    block_q8_0* xq = nullptr;
    void* act_xe = nullptr;
    float* rlogits = nullptr;
    int32_t* ids = nullptr;
    float* rw = nullptr;
    int32_t* inv = nullptr;
    block_q8_0* actq = nullptr;
    void* act_dn = nullptr;
    float *y_dn = nullptr, *moe_out = nullptr, *moe_conv = nullptr;
    float *fin_mix = nullptr, *fin_x = nullptr;
    block_q8_0* fin_q = nullptr;
    void* act_fin = nullptr;
    float* gu_tmp = nullptr;
    size_t gu_tmp_stride = 0;                 // floats per thread
    PfCounter* ctr = nullptr;
    int used_phases = 0;
    struct alignas(64) CpuAcc {
        double last = 0;
        double ns[PF_COUNT] = {};
    };
    CpuAcc cpu[kMaxTh];
    int n_used_experts = 0;                   // experts with >= 1 pair in the current visit (profile bytes)
    int kind = 0;                             // 0 = prefill scratch (chunk rows), 1 = verify scratch (<= kVerifyMaxRows rows)
    float* taps_mem = nullptr;                // prefill tap-sink buffer [taps_rows][kEngineTapFloats] (allocated on demand when a sink is set)
    size_t taps_rows = 0;
    float* rec_mem = nullptr;                 // verify: conv-input record [48 visits][rows + 3][6656] (allocated on first verify)
    size_t rec_floats = 0;
    int pend_slot = -1, pend_T = 0;           // verify awaiting engine_verify_commit
    int64_t pend_pos0 = 0;
    MoeRoute route;
    std::vector<MoeItem> items_gu, items_dn;
    std::vector<PItem> items_proj[2];         // [0] SWA layers, [1] global layers
    PfProfile prof;
    ~PfScratch() {
        if (arena) free_weight_buffer(arena);
        std::free(ctr);
        std::free(taps_mem);
        std::free(rec_mem);
    }
};

struct Carver {
    uintptr_t base;
    size_t off = 0;
    void* take(size_t bytes) {
        off = (off + 63) & ~(size_t)63;
        void* p = (void*)(base + off);
        off += bytes;
        return p;
    }
};

size_t max_act_bytes(int nt, int k) { return std::max(gemm_act_bytes(GEMM_FMT_Q8, nt, k), gemm_act_bytes(GEMM_FMT_NVFP4, nt, k)); }

void carve(PfScratch& X, Carver& A, int cap) {
    const size_t C = (size_t)cap;
    auto F = [&](size_t n) { return (float*)A.take(n * 4); };
    for (int i = 0; i < 2; ++i) X.cur[i] = F(C * kD);
    for (int i = 0; i < kMaxBlocks; ++i) X.blk[i] = F(C * kD);
    X.prelude = F(C * kD);
    X.bank_b = F(C * kD);
    X.bank_out = F(C * kD);
    for (int i = 0; i < 4; ++i) X.msgs[i] = F(C * kD);
    for (int i = 0; i < 3; ++i) X.mix[i] = F(C * kD);
    for (int i = 0; i < 4; ++i) X.cin[i] = F(C * kD);
    for (int i = 0; i < 3; ++i) {
        X.inq[i] = (block_q8_0*)A.take(C * kQB * sizeof(block_q8_0));
        X.inbf[i] = F(C * kD);
        X.act_in[i] = A.take(gemm_act_bytes(GEMM_FMT_Q8, cap, kD));
        X.ctrl_in[i] = F(C * 2 * kD);
        X.ctrl_g[i] = F(C * kCtrlRank);
        X.ctrl_gain[i] = F(C * kD);
        X.ctrl_forget[i] = F(C * kD);
    }
    X.q = F(C * kQW);
    X.gate = F(C * kQW);
    X.ao = F(C * kQW);
    X.kp = F(C * kKVW);
    X.vp = F(C * kKVW);
    X.kc = F(C * kKVW);
    X.vc = F(C * kKVW);
    X.rel = F(C * kRelW);
    X.sink = F(C * kNQHeads);
    X.kdyn = F(C * 32);
    X.vdyn = F(C * 32);
    X.tau = F(C);
    X.aoq = (block_q8_0*)A.take(C * kQWB * sizeof(block_q8_0));
    X.act_wo = A.take(gemm_act_bytes(GEMM_FMT_Q8, cap, kQW));
    X.wo_out = F(C * kD);
    X.attn_out = F(C * kD);
    X.fmix = F(C * kD);
    X.xffn = F(C * kD);
    X.xq = (block_q8_0*)A.take(C * kQB * sizeof(block_q8_0));
    X.act_xe = A.take(max_act_bytes(cap, kD));
    X.rlogits = F(C * kNExp);
    X.ids = (int32_t*)A.take(C * kTopK * 4);
    X.rw = F(C * kTopK);
    X.inv = (int32_t*)A.take(C * kTopK * 4);
    X.actq = (block_q8_0*)A.take(C * kTopK * kFB * sizeof(block_q8_0));
    X.act_dn = A.take(max_act_bytes(cap * kTopK, kFF));
    X.y_dn = F(C * kTopK * kD);
    X.moe_out = F(C * kD);
    X.moe_conv = F(C * kD);
    X.fin_mix = F(C * kD);
    X.fin_x = F(C * kD);
    X.fin_q = (block_q8_0*)A.take(C * kQB * sizeof(block_q8_0));
    X.act_fin = A.take(gemm_act_bytes(GEMM_FMT_Q8, cap, kD));
    X.gu_tmp_stride = C * 2 * kGuW;
    X.gu_tmp = F(X.gu_tmp_stride * kMaxTh);
}

// ------------------------------------------------------------------------------------------------ registry / tunables
std::mutex g_mu;
std::map<std::pair<const Engine*, int>, std::unique_ptr<PfScratch>> g_scr;

int env_int(const char* n, int d) {
    const char* v = std::getenv(n);
    return v && *v ? std::atoi(v) : d;
}

PrefillTune make_tune() {
    PrefillTune t;
    t.chunk = env_int("INK_PF_CHUNK", t.chunk);
    t.chunk_proj = env_int("INK_PF_PROJ", t.chunk_proj);
    t.chunk_wo = env_int("INK_PF_WO", t.chunk_wo);
    t.chunk_gu = env_int("INK_PF_GU", t.chunk_gu);
    t.chunk_dn = env_int("INK_PF_DN", t.chunk_dn);
    t.tok_claim = env_int("INK_PF_TOK", t.tok_claim);
    t.attn_tq = env_int("INK_PF_ATTN_TQ", t.attn_tq);
    t.gemm_group = env_int("INK_PF_GROUP", t.gemm_group);
    if (const char* a = std::getenv("INK_PF_ATTN")) t.attn_decode_kernels = std::strcmp(a, "decode") == 0 ? 1 : 0;
    return t;
}

PfScratch* find_scratch(const Engine& e, int kind = 0) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_scr.find({&e, kind});
    return it == g_scr.end() ? nullptr : it->second.get();
}

PfScratch& get_scratch(const Engine& e, int cap, int kind = 0) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto& up = g_scr[{&e, kind}];
    if (!up) up.reset(new PfScratch());
    PfScratch& X = *up;
    X.kind = kind;
    if (X.cap < cap || !X.arena) {
        if (X.arena) free_weight_buffer(X.arena);
        Carver probe{0};
        carve(X, probe, cap);
        X.arena = alloc_weight_buffer(probe.off + 4096, true);
        if (!X.arena) {
            std::fprintf(stderr, "engine_prefill: cannot allocate %zu MiB of scratch\n", probe.off >> 20);
            std::abort();
        }
        Carver real{(uintptr_t)X.arena};
        carve(X, real, cap);
        X.cap = cap;
        if (!X.ctr) {
            X.ctr = (PfCounter*)std::aligned_alloc(64, sizeof(PfCounter) * kMaxPh);
            std::memset((void*)X.ctr, 0, sizeof(PfCounter) * kMaxPh);
        }
    }
    return X;
}

// ------------------------------------------------------------------------------------------------ chunk context
struct PCtx {
    Engine* E;
    PfScratch* X;
    SlotState* S;
    ThreadPool* pool;
    PrefillTune tu;
    int64_t pos0;
    int n;
    const int32_t* tokens;
    float* logits;       // logits of the last token (n_vocab floats) or nullptr
    float* logits_all;   // [n][n_vocab] or nullptr
    float* taps;         // [n][kEngineTapFloats] DSpark taps of the chunk rows, or nullptr
    float* rec;          // verify only: conv-input record of this call ([visit][rows + 3][6656]), or nullptr
    size_t rec_visit;    // floats per visit in `rec`
    bool verify;         // small-n verify mode: attention items (kv group x token) and a separate quantise phase
    bool prof;
    uint64_t last;
};
struct PTh {
    int tid, nth, phase;
};

inline double thread_cpu_ns() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
inline void pcpu_before(PCtx& c, const PTh& t, int ph) {   // busy CPU time of this thread since the last barrier, charged to `ph`
    if (!c.prof) return;
    PfScratch::CpuAcc& a = c.X->cpu[t.tid];
    const double now = thread_cpu_ns();
    a.ns[ph] += now - a.last;
}
inline void pmark(PCtx& c, const PTh& t, int ph, uint64_t macs, uint64_t bytes) {
    if (!c.prof) return;
    c.X->cpu[t.tid].last = thread_cpu_ns();
    if (t.tid != 0) return;
    const uint64_t now = __rdtsc();
    PfProfile& P = c.X->prof;
    P.ticks[ph] += now - c.last;
    P.calls[ph]++;
    P.macs[ph] += macs;
    P.bytes[ph] += bytes;
    c.last = now;
}
#define PBAR(ph, macs, by)                      \
    do {                                        \
        pcpu_before(c, t, (ph));                \
        pool.barrier();                         \
        pmark(c, t, (ph), (macs), (by));        \
    } while (0)

// dynamic scheduling: every thread claims consecutive items / ranges from one shared counter per phase (zeroed before every run)
template <class F>
inline void dyn_items(PCtx& c, PTh& t, int64_t nitems, F&& f) {
    std::atomic<int64_t>& a = c.X->ctr[t.phase++].v;
    for (;;) {
        const int64_t i = a.fetch_add(1, std::memory_order_relaxed);
        if (i >= nitems) break;
        f(i);
    }
}
template <class F>
inline void dyn_range(PCtx& c, PTh& t, int64_t total, int64_t chunk, F&& f) {
    std::atomic<int64_t>& a = c.X->ctr[t.phase++].v;
    for (;;) {
        const int64_t b = a.fetch_add(1, std::memory_order_relaxed) * chunk;
        if (b >= total) break;
        f(b, std::min<int64_t>(b + chunk, total));
    }
}
template <class F>
inline void dyn_guided(PCtx& c, PTh& t, int64_t total, int64_t chunk, int64_t fine, F&& f) {
    std::atomic<int64_t>& a = c.X->ctr[t.phase++].v;
    const int64_t nbig = (total * 3 / 4) / chunk, big_rows = nbig * chunk;
    for (;;) {
        const int64_t i = a.fetch_add(1, std::memory_order_relaxed);
        int64_t b, e;
        if (i < nbig) {
            b = i * chunk;
            e = b + chunk;
        } else {
            b = big_rows + (i - nbig) * fine;
            if (b >= total) break;
            e = std::min<int64_t>(b + fine, total);
        }
        f(b, e);
    }
}

inline void bf16_round(const float* x, float* y, int n) {   // n % 16 == 0
    for (int i = 0; i < n; i += 16)
        _mm512_storeu_ps(y + i, simd::bf16x16_to_f32(simd::f32x16_to_bf16(_mm512_loadu_ps(x + i))));
}

inline float tau_of(const ModelConfig& cfg, int64_t pos) {   // float operation order of the ggml graph (identical to the decode executor)
    float tt = (float)pos + 1.0f;
    tt = tt * (1.0f / (float)cfg.log_scale_floor);
    tt = std::min(INFINITY, std::max(1.0f, tt));
    return std::log(tt) * cfg.log_scale_alpha + 1.0f;
}

// BF16 / F32 GEMM over n tokens in token groups sized so that the activation group (~256 KB) stays in L2 while all rows of the item stream over it
// (the kernel re-reads the activations once per 2-row panel; with all 256 tokens at once that traffic comes from L3 and the kernel runs at ~2 GMAC/s).
inline void gemm_lp_tg(int type, const void* w, size_t rs, const float* x, size_t ldx, int n, int k, float* y, size_t ldy, int nrows) {
    const int tg = std::max(6, (65536 / k) / 6 * 6);
    for (int t0 = 0; t0 < n; t0 += tg) {
        const int nt = std::min(tg, n - t0);
        if (type == T_BF16)
            gemm_bf16(w, rs, x + (size_t)t0 * ldx, ldx, nt, k, y + (size_t)t0 * ldy, ldy, nrows);
        else
            gemm_f32(w, rs, x + (size_t)t0 * ldx, ldx, nt, k, y + (size_t)t0 * ldy, ldy, nrows);
    }
}

inline int tap_index(int visit_index) {   // 0..2 for the visits whose entering stream is a DSpark tap, else -1
    for (int k = 0; k < 3; ++k)
        if (visit_index == kTapVisit[k]) return k;
    return -1;
}

// verify: record the inputs of one short conv for the channel range [c0, c1): rec = [rows + 3][C]; rows 0..2 = the history BEFORE this call, rows 3.. = the
// n inputs (so the history after k rows is rec rows [k, k + 3), contiguous = the SoA hist layout). Must run before sconv_seq updates the history.
inline void rec_conv(float* rec, const float* hist, const float* x, size_t n, size_t C, size_t c0, size_t c1) {
    for (size_t i = 0; i < 3; ++i) std::memcpy(rec + i * C + c0, hist + i * C + c0, (c1 - c0) * sizeof(float));
    for (size_t t = 0; t < n; ++t) std::memcpy(rec + (3 + t) * C + c0, x + t * C + c0, (c1 - c0) * sizeof(float));
}

inline float* off_floats(float* base, int64_t nf) { return (float*)((uintptr_t)base + (uintptr_t)(nf * (int64_t)sizeof(float))); }

// ------------------------------------------------------------------------------------------------ loop controllers (batched)
// nj controllers: X.ctrl_in[j] ([n][2*kD] = [rms(cur), rms(msg)]) was filled by the token phase before the barrier before this call.
void controllers(PCtx& c, PTh& t, const LayerPacked& L, int nj, const int* cidx, const float* const* cur_in, const float* const* msg,
                 float* const* out) {
    PfScratch& X = *c.X;
    ThreadPool& pool = *c.pool;
    const int n = c.n;
    const int ty = L.ctrl_down[cidx[0]].type;
    auto gemm_lp = [&](const Mat& m, int64_t r0, const float* x, size_t ldx, int k, float* y, size_t ldy, int nr) {
        gemm_lp_tg(ty, (const char*)m.w + (size_t)r0 * m.rs, m.rs, x, ldx, n, k, y, ldy, nr);
    };
    uint64_t bytes = 0;
    for (int j = 0; j < nj; ++j) bytes += L.ctrl_down[cidx[j]].bytes() + 2 * L.ctrl_gain[cidx[j]].bytes();
    const int crd = 4, ndn = kCtrlRank / crd;
    dyn_items(c, t, (int64_t)nj * ndn, [&](int64_t i) {
        const int j = (int)(i / ndn), r0 = (int)(i % ndn) * crd;
        gemm_lp(L.ctrl_down[cidx[j]], r0, X.ctrl_in[j], 2 * kD, 2 * kD, X.ctrl_g[j] + r0, kCtrlRank, crd);
    });
    PBAR(PF_CTRL, (uint64_t)nj * kCtrlRank * 2 * kD * n, bytes);
    const int gch = 32, ng = kD / gch;
    dyn_items(c, t, (int64_t)nj * 2 * ng, [&](int64_t i) {
        const int j = (int)(i / (2 * ng)), rem = (int)(i % (2 * ng)), which = rem / ng, r0 = (rem % ng) * gch;
        const Mat& m = which ? L.ctrl_forget[cidx[j]] : L.ctrl_gain[cidx[j]];
        float* y = (which ? X.ctrl_forget[j] : X.ctrl_gain[j]) + r0;
        gemm_lp(m, r0, X.ctrl_g[j], kCtrlRank, kCtrlRank, y, kD, gch);
    });
    PBAR(PF_CTRL, (uint64_t)nj * 2 * kD * kCtrlRank * n, 0);
    dyn_range(c, t, n, c.tu.tok_claim, [&](int64_t b, int64_t e) {
        for (int64_t tt = b; tt < e; ++tt)
            for (int j = 0; j < nj; ++j)
                ctrl_apply(cur_in[j] + tt * kD, msg[j] + tt * kD, X.ctrl_gain[j] + tt * kD, X.ctrl_forget[j] + tt * kD, out[j] + tt * kD, kD);
    });
    PBAR(PF_CTRL, 0, 0);
}

// ------------------------------------------------------------------------------------------------ one layer visit over n tokens
void visit(PCtx& c, PTh& t, const Visit& vis, float*& cur, float*& nxt, int& nb) {
    Engine& E = *c.E;
    PfScratch& X = *c.X;
    ThreadPool& pool = *c.pool;
    SlotState& S = *c.S;
    const ModelConfig& cfg = E.cfg;
    const LayerPacked& L = E.layers[vis.phys];
    const int n = c.n;
    const bool swa = L.swa, msgs = vis.has_message;
    const float eps = cfg.rms_eps, ceps = cfg.hard_coded.ctrl_rms_eps;
    const bool block_start = (vis.phys % cfg.attn_res_block_size) == 0;
    const int tok = std::max(1, c.tu.tok_claim);
    float* hist = S.hist + (size_t)vis.index * kHistPerVisit;
    float* h_k = hist;
    float* h_v = hist + 3 * kKVW;
    float* h_a = hist + 6 * kKVW;
    float* h_f = h_a + 3 * kD;
    KvLayer& kv = S.kv[vis.index];
    const bool nv = L.expert_type == T_NVFP4;
    const int efmt = nv ? GEMM_FMT_NVFP4 : GEMM_FMT_Q8;
    const int tapk = c.taps ? tap_index(vis.index) : -1;
    float *rec_k = nullptr, *rec_v = nullptr, *rec_a = nullptr, *rec_f = nullptr;   // verify: conv-input records of this visit
    if (c.rec) {
        const size_t R = (size_t)n + 3;
        rec_k = c.rec + (size_t)vis.index * c.rec_visit;
        rec_v = rec_k + R * kKVW;
        rec_a = rec_v + R * kKVW;
        rec_f = rec_a + R * kD;
    }

    // ---------------------------------------------------------------- A: attention-input mix, [gate input of the controllers | norms + Q8_0 + bf16 + pack]
    const float* in_raw[3] = {cur, cur, cur};
    if (L.attn_res_query)
        for (int j = 0; j < 3; ++j) in_raw[j] = X.mix[swa ? 0 : j];
    const bool shared = swa && !msgs;   // q, k, v read the same normalised input
    auto norm_quant = [&](const float* x, const float* w, int j, int64_t tt, bool bf) {
        alignas(64) float inx[kD];
        rmsnorm(x, w, inx, kD, eps);
        quantize_q8_0(inx, X.inq[j] + (size_t)tt * kQB, kD, Q8Mode::X86);
        if (bf) bf16_round(inx, X.inbf[j] + (size_t)tt * kD, kD);
    };
    auto norms_tok = [&](const float* const* src, int64_t tt) {
        if (shared) {
            norm_quant(src[0] + tt * kD, L.attn_norm, 0, tt, true);
        } else if (swa) {
            for (int j = 0; j < 3; ++j) norm_quant(src[j] + tt * kD, L.attn_norm, j, tt, j == 0);
        } else {
            for (int j = 0; j < 3; ++j) norm_quant(src[j] + tt * kD, L.inp_norm[j], j, tt, true);
        }
    };
    auto pack_in = [&](int64_t b, int64_t e) {
        const int nj = shared ? 1 : 3;
        for (int j = 0; j < nj; ++j) gemm_pack_act(GEMM_FMT_Q8, X.inq[j], (int)b, (int)e, n, kD, X.act_in[j]);
    };
    dyn_range(c, t, n, tok, [&](int64_t b, int64_t e) {
        for (int64_t tt = b; tt < e; ++tt) {
            if (tapk >= 0) std::memcpy(c.taps + (size_t)tt * kEngineTapFloats + (size_t)tapk * kD, cur + tt * kD, kD * sizeof(float));   // DSpark tap (cur is read-only here)
            if (msgs) {
                const float* ms[2] = {X.bank_b + tt * kD, X.bank_out + tt * kD};
                float* mo[4] = {X.msgs[0] + tt * kD, X.msgs[1] + tt * kD, X.msgs[2] + tt * kD, X.msgs[3] + tt * kD};
                depth_mix(ms, 2, L.msg_query, 4, mo, kD, kMixScale);
            }
            if (L.attn_res_query) {
                const float* st[8];
                int ns = 0;
                for (int bb = nb - 1; bb >= 0; --bb) st[ns++] = X.blk[bb] + tt * kD;
                st[ns++] = cur + tt * kD;
                float* mo[3] = {X.mix[0] + tt * kD, X.mix[1] + tt * kD, X.mix[2] + tt * kD};
                depth_mix(st, ns, L.attn_res_query, swa ? 1 : 3, mo, kD, kMixScale);
            }
            if (msgs) {
                for (int j = 0; j < 3; ++j) ctrl_gate_input(in_raw[j] + tt * kD, X.msgs[j] + tt * kD, X.ctrl_in[j] + tt * 2 * kD, kD, ceps);
            } else {
                norms_tok(in_raw, tt);
            }
        }
        if (!msgs) pack_in(b, e);
    });
    PBAR(PF_MIX, 0, 0);
    if (msgs) {
        const int cidx[3] = {0, 1, 2};
        const float* cin[3] = {in_raw[0], in_raw[1], in_raw[2]};
        const float* mm[3] = {X.msgs[0], X.msgs[1], X.msgs[2]};
        float* outs[3] = {X.cin[0], X.cin[1], X.cin[2]};
        controllers(c, t, L, 3, cidx, cin, mm, outs);
        for (int j = 0; j < 3; ++j) in_raw[j] = X.cin[j];
        dyn_range(c, t, n, tok, [&](int64_t b, int64_t e) {
            for (int64_t tt = b; tt < e; ++tt) norms_tok(in_raw, tt);
            pack_in(b, e);
        });
        PBAR(PF_MIX, 0, 0);
    }

    // ---------------------------------------------------------------- B: projections (GEMM items = matrix x row chunk) + BF16 sink / dyn taps
    {
        const GemmAct a0 = gemm_act_view(GEMM_FMT_Q8, X.act_in[0], n, kD);
        const GemmAct a1 = shared ? a0 : gemm_act_view(GEMM_FMT_Q8, X.act_in[1], n, kD);
        const GemmAct a2 = shared ? a0 : gemm_act_view(GEMM_FMT_Q8, X.act_in[2], n, kD);
        const std::vector<PItem>& items = X.items_proj[swa ? 0 : 1];
        dyn_items(c, t, (int64_t)items.size(), [&](int64_t i) {
            const PItem& it = items[(size_t)i];
            const int nr = it.r1 - it.r0;
            auto q8 = [&](const Mat& m, const GemmAct& a, float* y, size_t ldy) {
                gemm_q8_0((const char*)m.w + (size_t)it.r0 * m.rs, m.rs, a, 0, n, kD, y + it.r0, ldy, nr);
            };
            auto bf = [&](const Mat& m, const float* x, float* y, size_t ldy) {
                gemm_lp_tg(T_BF16, (const char*)m.w + (size_t)it.r0 * m.rs, m.rs, x, kD, n, kD, y + it.r0, ldy, nr);
            };
            switch (it.m) {
                case 0: q8(L.wq, a0, X.q, kQW); break;
                case 1: q8(L.wgate, a0, X.gate, kQW); break;
                case 2: q8(L.wk, a1, X.kp, kKVW); break;
                case 3: q8(L.wv, a2, X.vp, kKVW); break;
                case 4: q8(L.wrel, a0, X.rel, kRelW); break;
                case 5: bf(L.sink, X.inbf[0], X.sink, kNQHeads); break;
                case 6: bf(L.kproj, X.inbf[1], X.kdyn, 32); break;
                default: bf(L.vproj, X.inbf[2], X.vdyn, 32); break;
            }
        });
        uint64_t rows = (uint64_t)L.wq.rows + L.wk.rows + L.wv.rows + L.wgate.rows + L.wrel.rows;
        uint64_t by = L.wq.bytes() + L.wk.bytes() + L.wv.bytes() + L.wgate.bytes() + L.wrel.bytes() + L.sink.bytes();
        uint64_t brows = L.sink.rows;
        if (!swa) {
            by += L.kproj.bytes() + L.vproj.bytes();
            brows += L.kproj.rows + L.vproj.rows;
        }
        PBAR(PF_PROJ, (uint64_t)n * kD * (rows + brows), by);
    }

    // ---------------------------------------------------------------- C1: per KV group: short conv K,V (all n tokens), clips, V norm, kv_append
    {
        dyn_items(c, t, kNKvHeads, [&](int64_t g) {
            const size_t c0 = (size_t)g * kHeadDim, c1 = c0 + kHeadDim;
            if (rec_k) {
                rec_conv(rec_k, h_k, X.kp, (size_t)n, kKVW, c0, c1);
                rec_conv(rec_v, h_v, X.vp, (size_t)n, kKVW, c0, c1);
            }
            sconv_seq(X.kp, h_k, L.kconv, X.kc, (size_t)n, kKVW, c0, c1, swa ? nullptr : X.kdyn, kHeadDim);
            sconv_seq(X.vp, h_v, L.vconv, X.vc, (size_t)n, kKVW, c0, c1, swa ? nullptr : X.vdyn, kHeadDim);
            for (int tt = 0; tt < n; ++tt) {
                float* kr = X.kc + (size_t)tt * kKVW;
                float* vr = X.vc + (size_t)tt * kKVW;
                l2clip_heads(kr + c0, nullptr, 1, kHeadDim, cfg.qk_clip, 1.0f);
                vnorm_heads(vr + c0, vr + c0, 1, kHeadDim, eps);
                l2clip_heads(X.q + (size_t)tt * kQW + (size_t)g * (kGqa * kHeadDim), L.q_clip + g * (kGqa * kHeadDim), kGqa, kHeadDim, cfg.qk_clip, 1.0f);
                l2clip_heads(X.rel + (size_t)tt * kRelW + (size_t)g * (kGqa * kRelFeat), L.rel_clip + g * (kGqa * kRelFeat), kGqa, kRelFeat,
                             cfg.rel_clip, 1.0f);
                kv_append(kv, c.pos0 + tt, kr, vr, (int)g, (int)g + 1);
            }
        });
        PBAR(PF_CONV, 0, 0);
    }

    // ---------------------------------------------------------------- C2: attention per token range (all 8 groups), Q8_0 quantise + pack of the output
    if (c.verify) {
        // small n (<= 8 rows): one item per (kv group [, row]) so that all threads work; quantise + pack in a token-parallel phase afterwards
        const bool dk = c.tu.attn_decode_kernels != 0;
        const AttnPrefill P{X.q, X.rel, X.sink, X.gate, swa ? nullptr : X.tau, c.pos0, n};
        const int split_keys = std::max(1, E.tune.split_keys);
        const int per_g = dk ? n : 1;
        dyn_items(c, t, (int64_t)kNKvHeads * per_g, [&](int64_t i) {
            const int g = (int)(i / per_g);
            if (!dk) {
                attn_prefill_group(kv, L.rel_table, P, g, 0, n, X.ao);
            } else {
                const int tt = (int)(i % per_g);
                const int64_t pos = c.pos0 + tt;
                const int64_t n_keys = swa ? std::min<int64_t>(pos + 1, kSwaWindow) : pos + 1;
                const int nsplit = (int)std::min<int64_t>(kMaxSplit, std::max<int64_t>(1, n_keys / split_keys));
                const AttnQuery aq{X.q + (size_t)tt * kQW, X.rel + (size_t)tt * kRelW, X.sink + (size_t)tt * kNQHeads, X.gate + (size_t)tt * kQW,
                                   pos, swa ? 1.0f : X.tau[tt]};
                AttnPartial parts[kMaxSplit];
                for (int s2 = 0; s2 < nsplit; ++s2) attn_decode_task(kv, L.rel_table, aq, g, s2, nsplit, &parts[s2]);
                attn_decode_finish(aq, g, parts, nsplit, X.ao + (size_t)tt * kQW);
            }
        });
        PBAR(PF_ATTN, 0, 0);
        dyn_range(c, t, n, 1, [&](int64_t b, int64_t e) {
            for (int64_t tt = b; tt < e; ++tt) quantize_q8_0(X.ao + (size_t)tt * kQW, X.aoq + (size_t)tt * kQWB, kQW, Q8Mode::X86);
            gemm_pack_act(GEMM_FMT_Q8, X.aoq, (int)b, (int)e, n, kQW, X.act_wo);
        });
        PBAR(PF_ATTN, 0, 0);
    } else {
        const int TQ = std::max(1, c.tu.attn_tq);
        const int nitems = (n + TQ - 1) / TQ;
        const AttnPrefill P{X.q, X.rel, X.sink, X.gate, swa ? nullptr : X.tau, c.pos0, n};
        const int split_keys = std::max(1, E.tune.split_keys);
        const bool dk = c.tu.attn_decode_kernels != 0;
        dyn_items(c, t, nitems, [&](int64_t i) {   // late tokens (most keys) first
            const int t1 = n - (int)i * TQ, t0 = std::max(0, t1 - TQ);
            if (!dk) {
                for (int g = 0; g < kNKvHeads; ++g) attn_prefill_group(kv, L.rel_table, P, g, t0, t1, X.ao);
            } else {
                for (int tt = t0; tt < t1; ++tt) {
                    const int64_t pos = c.pos0 + tt;
                    const int64_t n_keys = swa ? std::min<int64_t>(pos + 1, kSwaWindow) : pos + 1;
                    const int nsplit = (int)std::min<int64_t>(kMaxSplit, std::max<int64_t>(1, n_keys / split_keys));
                    const AttnQuery aq{X.q + (size_t)tt * kQW, X.rel + (size_t)tt * kRelW, X.sink + (size_t)tt * kNQHeads, X.gate + (size_t)tt * kQW,
                                       pos, swa ? 1.0f : X.tau[tt]};
                    AttnPartial parts[kMaxSplit];
                    for (int g = 0; g < kNKvHeads; ++g) {
                        for (int s = 0; s < nsplit; ++s) attn_decode_task(kv, L.rel_table, aq, g, s, nsplit, &parts[s]);
                        attn_decode_finish(aq, g, parts, nsplit, X.ao + (size_t)tt * kQW);
                    }
                }
            }
            for (int tt = t0; tt < t1; ++tt) quantize_q8_0(X.ao + (size_t)tt * kQW, X.aoq + (size_t)tt * kQWB, kQW, Q8Mode::X86);
            gemm_pack_act(GEMM_FMT_Q8, X.aoq, t0, t1, n, kQW, X.act_wo);
        });
        PBAR(PF_ATTN, 0, 0);
    }

    // ---------------------------------------------------------------- E: wo GEMM per row chunk + post-conv + residual (the thread that has the rows has all tokens)
    {
        const GemmAct awo = gemm_act_view(GEMM_FMT_Q8, X.act_wo, n, kQW);
        const int64_t chunk = std::max<int64_t>(16, (c.tu.chunk_wo / 16) * 16);
        dyn_range(c, t, kD, chunk, [&](int64_t b, int64_t e) {
            gemm_q8_0((const char*)L.wo.w + (size_t)b * L.wo.rs, L.wo.rs, awo, 0, n, kQW, X.wo_out + b, kD, (int)(e - b));
            float* y = block_start ? nxt : X.attn_out;
            if (rec_a) rec_conv(rec_a, h_a, X.wo_out, (size_t)n, kD, (size_t)b, (size_t)e);
            sconv_seq(X.wo_out, h_a, L.aconv, y, (size_t)n, kD, (size_t)b, (size_t)e);
            if (!block_start)
                for (int tt = 0; tt < n; ++tt) vadd(cur + (size_t)tt * kD + b, X.attn_out + (size_t)tt * kD + b, nxt + (size_t)tt * kD + b, (size_t)(e - b));
        });
        PBAR(PF_WO, (uint64_t)n * kD * kQW, L.wo.bytes());
    }

    // ---------------------------------------------------------------- F1: push inpSA (block starts), ffn depth mix, [controller], norm + Q8_0 + pack
    auto ffn_norm_quant = [&](const float* m, int64_t tt) {
        rmsnorm(m, L.ffn_norm, X.xffn + tt * kD, kD, eps);
        quantize_q8_0(X.xffn + tt * kD, X.xq + (size_t)tt * kQB, kD, Q8Mode::X86);
    };
    dyn_range(c, t, n, tok, [&](int64_t b, int64_t e) {
        for (int64_t tt = b; tt < e; ++tt) {
            if (block_start) std::memcpy(X.blk[nb] + tt * kD, cur + tt * kD, kD * sizeof(float));
            const int nbk = nb + (block_start ? 1 : 0);
            const float* st[8];
            int ns = 0;
            for (int bb = nbk - 1; bb >= 0; --bb) st[ns++] = X.blk[bb] + tt * kD;
            st[ns++] = nxt + tt * kD;
            float* mo[1] = {X.fmix + tt * kD};
            depth_mix(st, ns, L.ffn_res_query, 1, mo, kD, kMixScale);
            if (msgs)
                ctrl_gate_input(X.fmix + tt * kD, X.msgs[3] + tt * kD, X.ctrl_in[0] + tt * 2 * kD, kD, ceps);
            else
                ffn_norm_quant(X.fmix + tt * kD, tt);
        }
        if (!msgs) gemm_pack_act(efmt, X.xq, (int)b, (int)e, n, kD, X.act_xe);
    });
    if (block_start) ++nb;
    PBAR(PF_FFNMIX, 0, 0);
    if (msgs) {
        const int cidx[1] = {3};
        const float* cin[1] = {X.fmix};
        const float* mm[1] = {X.msgs[3]};
        float* outs[1] = {X.cin[3]};
        controllers(c, t, L, 1, cidx, cin, mm, outs);
        dyn_range(c, t, n, tok, [&](int64_t b, int64_t e) {
            for (int64_t tt = b; tt < e; ++tt) ffn_norm_quant(X.cin[3] + tt * kD, tt);
            gemm_pack_act(efmt, X.xq, (int)b, (int)e, n, kD, X.act_xe);
        });
        PBAR(PF_FFNMIX, 0, 0);
    }

    // ---------------------------------------------------------------- F2: router GEMM, top-8 per token, routing plan (serial)
    {
        const Mat& R = L.router;
        dyn_range(c, t, kNExp, 4, [&](int64_t b, int64_t e) {
            gemm_lp_tg(R.type, (const char*)R.w + (size_t)b * R.rs, R.rs, X.xffn, kD, n, kD, X.rlogits + b, kNExp, (int)(e - b));
        });
        PBAR(PF_ROUTER, (uint64_t)n * kNExp * kD, R.bytes());
        dyn_range(c, t, n, 8, [&](int64_t b, int64_t e) {
            for (int64_t tt = b; tt < e; ++tt)
                router_topk(X.rlogits + tt * kNExp, L.exp_bias, kNExp, kTopK, cfg.expert_weights_scale, cfg.hard_coded.expert_weights_norm_eps,
                            X.ids + tt * kTopK, X.rw + tt * kTopK);
        });
        PBAR(PF_ROUTER, 0, 0);
        if (t.tid == 0) {
            moe_build_route(X.ids, n, kTopK, kNExp, X.route);
            moe_work_items(X.route, kFF, c.tu.chunk_gu, X.items_gu);
            moe_work_items(X.route, kD, c.tu.chunk_dn, X.items_dn);
            const int np = n * kTopK;
            for (int p = 0; p < np; ++p) X.inv[X.route.pair_slot[p]] = p;
            int nu = 0;
            for (int e = 0; e < kNExp; ++e) nu += X.route.offset[e + 1] > X.route.offset[e];
            X.n_used_experts = nu;
        }
        PBAR(PF_ROUTE, 0, 0);
    }
    const int np_all = n * kTopK;
    const uint64_t used_exp = (uint64_t)X.n_used_experts;

    // ---------------------------------------------------------------- G: expert gate_up (gate chunk + matching up chunk) + scale + SiTU + Q8_0
    {
        const GemmAct axe = gemm_act_view(efmt, X.act_xe, n, kD);
        float* tmp = X.gu_tmp + (size_t)t.tid * X.gu_tmp_stride;
        dyn_items(c, t, (int64_t)X.items_gu.size(), [&](int64_t i) {
            const MoeItem& it = X.items_gu[(size_t)i];
            const int p0 = X.route.offset[it.expert], p1 = X.route.offset[it.expert + 1], np = p1 - p0;
            const int w = it.row1 - it.row0, ldt = 2 * w;   // tile per pair: [w gate | w up]
            float* yg = off_floats(tmp, -((int64_t)p0 * ldt + it.row0));
            float* yu = off_floats(tmp, -((int64_t)p0 * ldt + (kFF + it.row0)) + w);
            const MoeItem ig{it.expert, it.row0, it.row1}, iu{it.expert, kFF + it.row0, kFF + it.row1};
            if (nv) {
                gemm_moe_nvfp4(L.gu, L.gu_eb, L.gu_rs, axe, X.route.pair_token.data(), X.route, ig, kD, yg, (size_t)ldt);
                gemm_moe_nvfp4(L.gu, L.gu_eb, L.gu_rs, axe, X.route.pair_token.data(), X.route, iu, kD, yu, (size_t)ldt);
            } else {
                gemm_moe_q8_0(L.gu, L.gu_eb, L.gu_rs, axe, X.route.pair_token.data(), X.route, ig, kD, yg, (size_t)ldt);
                gemm_moe_q8_0(L.gu, L.gu_eb, L.gu_rs, axe, X.route.pair_token.data(), X.route, iu, kD, yu, (size_t)ldt);
            }
            const float s = nv ? L.gu_scale[it.expert] : 1.0f;
            for (int q = 0; q < np; ++q) {
                float* g = tmp + (size_t)q * ldt;
                float* u = g + w;
                for (int sb = 0; sb < w / 32; ++sb) {
                    block_q8_0* dst = &X.actq[(size_t)(p0 + q) * kFB + (it.row0 / 32 + sb)];
                    if (nv) {
                        alignas(64) float gg[32], uu[32];
                        vscale(g + 32 * sb, s, gg, 32);
                        vscale(u + 32 * sb, s, uu, 32);
                        vsitu_q8_0(gg, uu, dst, 32, kSituBeta, kSituLin, Q8Mode::X86);
                    } else {
                        vsitu_q8_0(g + 32 * sb, u + 32 * sb, dst, 32, kSituBeta, kSituLin, Q8Mode::X86);
                    }
                }
            }
        });
        PBAR(PF_GATEUP, (uint64_t)np_all * 2 * kFF * kD, used_exp * (uint64_t)2 * kFF * L.gu_rs);
    }

    // ---------------------------------------------------------------- pack the per-pair activations of the down projection
    dyn_range(c, t, np_all, 32, [&](int64_t b, int64_t e) { gemm_pack_act(efmt, X.actq, (int)b, (int)e, np_all, kFF, X.act_dn); });
    PBAR(PF_PACKDN, 0, 0);

    // ---------------------------------------------------------------- H: expert down GEMM (pair-indexed activations) -> y_dn[pair][kD]
    {
        const GemmAct adn = gemm_act_view(efmt, X.act_dn, np_all, kFF);
        dyn_items(c, t, (int64_t)X.items_dn.size(), [&](int64_t i) {
            const MoeItem& it = X.items_dn[(size_t)i];
            if (nv)
                gemm_moe_nvfp4(L.dn, L.dn_eb, L.dn_rs, adn, nullptr, X.route, it, kFF, X.y_dn, kD);
            else
                gemm_moe_q8_0(L.dn, L.dn_eb, L.dn_rs, adn, nullptr, X.route, it, kFF, X.y_dn, kD);
        });
        PBAR(PF_DOWN, (uint64_t)np_all * kD * kFF, used_exp * (uint64_t)kD * L.dn_rs);
    }

    // ---------------------------------------------------------------- I: combine the 8 expert outputs per token (top-k order, sequential adds), post-conv, residual
    {
        dyn_range(c, t, kD, 32, [&](int64_t b, int64_t e) {
            for (int tt = 0; tt < n; ++tt) {
                const int32_t* idt = X.ids + (size_t)tt * kTopK;
                const float* wt = X.rw + (size_t)tt * kTopK;
                const int32_t* pv = X.inv + (size_t)tt * kTopK;
                float sc[kTopK];
                for (int j = 0; j < kTopK; ++j) sc[j] = nv ? L.dn_scale[idt[j]] : 1.0f;
                float* mo = X.moe_out + (size_t)tt * kD;
                for (int64_t c0 = b; c0 < e; c0 += 16) {   // b, e multiples of 16 (kD = 144 * 16)
                    __m512 acc = _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(X.y_dn + (size_t)pv[0] * kD + c0), _mm512_set1_ps(sc[0])), _mm512_set1_ps(wt[0]));
                    for (int j = 1; j < kTopK; ++j)
                        acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(X.y_dn + (size_t)pv[j] * kD + c0), _mm512_set1_ps(sc[j])),
                                                               _mm512_set1_ps(wt[j])));
                    _mm512_storeu_ps(mo + c0, acc);
                }
            }
            if (rec_f) rec_conv(rec_f, h_f, X.moe_out, (size_t)n, kD, (size_t)b, (size_t)e);
            sconv_seq(X.moe_out, h_f, L.fconv, X.moe_conv, (size_t)n, kD, (size_t)b, (size_t)e);
            for (int tt = 0; tt < n; ++tt)
                vadd(nxt + (size_t)tt * kD + b, X.moe_conv + (size_t)tt * kD + b, nxt + (size_t)tt * kD + b, (size_t)(e - b));
        });
        PBAR(PF_COMBINE, 0, 0);
    }
    std::swap(cur, nxt);
}

// ------------------------------------------------------------------------------------------------ whole chunk
void chunk_body(PCtx& c, PTh& t) {
    Engine& E = *c.E;
    PfScratch& X = *c.X;
    ThreadPool& pool = *c.pool;
    const ModelConfig& cfg = E.cfg;
    const int n = c.n;
    const int tok = std::max(1, c.tu.tok_claim);
    if (t.tid == 0) c.last = __rdtsc();
    if (c.prof) X.cpu[t.tid].last = thread_cpu_ns();
    dyn_range(c, t, n, tok, [&](int64_t b, int64_t e) {
        for (int64_t tt = b; tt < e; ++tt) {
            alignas(64) float emb[kD];
            const block_q8_0* row = (const block_q8_0*)(E.tok_embd + (size_t)c.tokens[tt] * E.tok_embd_rs);
            for (int bk = 0; bk < kD / 32; ++bk) {
                const float d = fp16_to_f32(row[bk].d);
                for (int j = 0; j < 32; ++j) emb[bk * 32 + j] = d * (float)row[bk].qs[j];
            }
            rmsnorm(emb, E.tok_norm, X.cur[0] + (size_t)tt * kD, kD, cfg.rms_eps);
        }
    });
    PBAR(PF_EMBED, 0, 0);
    float* cur = X.cur[0];
    float* nxt = X.cur[1];
    int nb = 0, prelude_nb = 0;
    for (int idx = 0; idx < kVisits; ++idx) {
        const Visit& vis = E.plan[idx];
        if (idx == cfg.loop_start) {   // end of the prelude: remember its output and blocks
            dyn_range(c, t, n, tok, [&](int64_t b, int64_t e) { std::memcpy(X.prelude + b * kD, cur + b * kD, (size_t)(e - b) * kD * sizeof(float)); });
            PBAR(PF_COPY, 0, 0);
            prelude_nb = nb;
        }
        if (vis.in_core && vis.pass > 0 && vis.phys == cfg.loop_start) {   // start of an extra pass
            dyn_range(c, t, n, tok, [&](int64_t b, int64_t e) {
                const size_t o = (size_t)b * kD, nbytes = (size_t)(e - b) * kD * sizeof(float);
                std::memcpy(X.bank_out + o, cur + o, nbytes);
                std::memcpy(X.bank_b + o, X.blk[prelude_nb + 1] + o, nbytes);
                std::memcpy(cur + o, X.prelude + o, nbytes);
            });
            PBAR(PF_COPY, 0, 0);
            nb = prelude_nb;
        }
        visit(c, t, vis, cur, nxt, nb);
    }
    const bool all = c.logits_all != nullptr;
    const bool want_taps = c.taps != nullptr;
    if (!c.logits && !all && !want_taps) return;
    const int nv = cfg.n_vocab;
    const Mat& H = E.lm_head;
    const int t_first = (all || want_taps) ? 0 : n - 1;   // rows that need the final normalised hidden state: [t_first, n)
    dyn_range(c, t, n - t_first, 1, [&](int64_t b, int64_t) {
        const int64_t tt = t_first + b;
        const float* st[8];
        int ns = 0;
        for (int bb = nb - 1; bb >= 0; --bb) st[ns++] = X.blk[bb] + tt * kD;
        st[ns++] = cur + tt * kD;
        float* mo[1] = {X.fin_mix + tt * kD};
        depth_mix(st, ns, E.out_res_query, 1, mo, kD, kMixScale);
        rmsnorm(X.fin_mix + tt * kD, E.out_norm, X.fin_x + tt * kD, kD, cfg.rms_eps);
        if (want_taps) std::memcpy(c.taps + (size_t)tt * kEngineTapFloats + 3 * (size_t)kD, X.fin_x + tt * kD, kD * sizeof(float));
        if (all || (c.logits && tt == n - 1)) {
            quantize_q8_0(X.fin_x + tt * kD, X.fin_q + (size_t)tt * kQB, kD, Q8Mode::X86);
            if (all) gemm_pack_act(GEMM_FMT_Q8, X.fin_q, (int)tt, (int)tt + 1, n, kD, X.act_fin);
        }
    });
    PBAR(PF_FINAL, 0, 0);
    if (!all) {
        if (!c.logits) return;   // taps only (non-final prefill chunk): no lm head
        dyn_guided(c, t, H.rows, 512, 64, [&](int64_t b, int64_t e) {
            gemv_q8_0((const char*)H.w + (size_t)b * H.rs, H.rs, X.fin_q + (size_t)(n - 1) * kQB, kD, c.logits + b, (int)(e - b));
            vsoftcap(c.logits + b, c.logits + b, (size_t)(e - b), cfg.final_logit_softcap);
        });
        PBAR(PF_LMHEAD, (uint64_t)H.rows * kD, H.bytes());
    } else {
        const GemmAct afin = gemm_act_view(GEMM_FMT_Q8, X.act_fin, n, kD);
        dyn_range(c, t, H.rows, 32, [&](int64_t b, int64_t e) {
            gemm_q8_0((const char*)H.w + (size_t)b * H.rs, H.rs, afin, 0, n, kD, c.logits_all + b, (size_t)nv, (int)(e - b));
            for (int tt = 0; tt < n; ++tt) vsoftcap(c.logits_all + (size_t)tt * nv + b, c.logits_all + (size_t)tt * nv + b, (size_t)(e - b), cfg.final_logit_softcap);
        });
        PBAR(PF_LMHEAD, (uint64_t)n * H.rows * kD, H.bytes());
    }
}

void build_proj_items(const Engine& E, const PrefillTune& tu, std::vector<PItem>* out) {
    for (int g = 0; g < 2; ++g) {
        std::vector<PItem>& v = out[g];
        v.clear();
        const LayerPacked* L = nullptr;
        for (const LayerPacked& l : E.layers)
            if (l.swa == (g == 0)) {
                L = &l;
                break;
            }
        if (!L) continue;
        auto add = [&](int m, int rows, int ch) {
            for (int r = 0; r < rows; r += ch) v.push_back({m, r, std::min(rows, r + ch)});
        };
        const int ch = std::max(4, tu.chunk_proj & ~3);
        add(5, L->sink.rows, 8);   // BF16 items are the slowest per row: first
        if (g == 1) {
            add(6, L->kproj.rows, 8);
            add(7, L->vproj.rows, 8);
        }
        add(0, L->wq.rows, ch);
        add(1, L->wgate.rows, ch);
        add(2, L->wk.rows, ch);
        add(3, L->wv.rows, ch);
        add(4, L->wrel.rows, ch);
    }
}

// SWA rings must hold 1023 + chunk positions (a chunk appends all its K/V before attending), or the chunk must not wrap the ring.
int chunk_limit(const Engine& E, const SlotState& S, int want) {
    int64_t ring = 0;
    for (int v = 0; v < kVisits; ++v)
        if (S.kv[v].ring) {
            ring = (int64_t)S.kv[v].n_blocks * kKvBlock;
            break;
        }
    if (ring <= 0) return want;
    if (S.pos + want <= ring) return want;                       // no position of the chunk wraps the ring
    return (int)std::max<int64_t>(1, std::min<int64_t>(want, ring - (kSwaWindow - 1)));
}

void pin_calling_thread(const Engine& E) {   // the calling thread acts as tid 0: pin it
    const std::vector<int> cpus = E.opt.cpus.empty() ? engine_default_cpus(E.opt.n_threads) : E.opt.cpus;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus[0], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

int prefill_impl(Engine& E, int slot, const int32_t* tokens, int n, float* logits_last, float* logits_all) {
    if (n < 1) return -4;
    if (slot < 0 || slot >= (int)E.slots.size() || !E.slots[slot].used) return -1;
    SlotState& S = E.slots[slot];
    if (S.pos + n > E.opt.max_ctx) return -2;
    for (int i = 0; i < n; ++i)
        if (tokens[i] < 0 || tokens[i] >= E.cfg.n_vocab_padded) return -3;
    if (E.pool->size() > kMaxTh) return -5;
    PrefillTune& gt = prefill_tune();
    const int want_chunk = std::max(1, std::min(gt.chunk, 1024));
    PfScratch& X = get_scratch(E, want_chunk);
    if (gt.gemm_group > 0) g_gemm_group = gt.gemm_group;
    build_proj_items(E, gt, X.items_proj);
    pin_calling_thread(E);
    float* taps_buf = nullptr;   // tap sink: rows of the current chunk
    if (E.tap_sink) {
        if (X.taps_rows < (size_t)X.cap) {
            std::free(X.taps_mem);
            X.taps_mem = (float*)std::aligned_alloc(64, ((size_t)X.cap * kEngineTapFloats * sizeof(float) + 63) / 64 * 64);
            X.taps_rows = (size_t)X.cap;
        }
        taps_buf = X.taps_mem;
    }
    const int nv = E.cfg.n_vocab;
    for (int off = 0; off < n;) {
        const int m = std::min(chunk_limit(E, S, std::min(want_chunk, n - off)), n - off);
        const bool last = off + m == n;
        PCtx c;
        c.E = &E;
        c.X = &X;
        c.S = &S;
        c.pool = E.pool.get();
        c.tu = gt;
        c.pos0 = S.pos;
        c.n = m;
        c.tokens = tokens + off;
        c.logits = (last && !logits_all) ? logits_last : nullptr;
        c.logits_all = logits_all ? logits_all + (size_t)off * nv : nullptr;
        c.taps = taps_buf;
        c.rec = nullptr;
        c.rec_visit = 0;
        c.verify = false;
        c.prof = E.prof_on || env_int("INK_PF_PROFILE", 0) != 0;
        c.last = 0;
        for (int v = 0; v < kVisits; ++v)
            if (!S.kv[v].ring) S.kv[v].n_blocks = std::max<uint32_t>(S.kv[v].n_blocks, (uint32_t)((S.pos + m - 1) / kKvBlock + 1));
        for (int tt = 0; tt < m; ++tt) X.tau[tt] = tau_of(E.cfg, S.pos + tt);
        const auto t0 = std::chrono::steady_clock::now();
        E.pool->run([&](int tid, int nth) {
            PTh t{tid, nth, 0};
            chunk_body(c, t);
            if (tid == 0) X.used_phases = t.phase;
        });
        if (X.used_phases >= kMaxPh) std::abort();
        for (int i = 0; i < X.used_phases; ++i) X.ctr[i].v.store(0, std::memory_order_relaxed);
        if (c.prof) {
            for (int th = 0; th < E.pool->size(); ++th)
                for (int p = 0; p < PF_COUNT; ++p) {
                    X.prof.cpu_ns[p] += X.cpu[th].ns[p];
                    X.cpu[th].ns[p] = 0;
                }
            X.prof.tokens += (uint64_t)m;
            X.prof.chunks++;
            X.prof.total_ticks += (uint64_t)(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() * E.prof.tsc_per_us);
        }
        if (E.tap_sink) E.tap_sink(slot, S.pos, m, taps_buf);   // before S.pos advances: first_pos = S.pos
        S.pos += m;
        if (S.hwm < S.pos) S.hwm = S.pos;
        off += m;
        E.slot_checkpoint(slot);   // S3: automatic shallow checkpoint at every chunk end (the last one = the prompt boundary)
    }
    return 0;
}

bool verify_exact_default() {
    static const bool v = env_int("INK_SPEC_EXACT", 0) != 0;
    return v;
}

void profile_report_impl(Engine& e, FILE* f, int kind);

}  // namespace

PrefillTune& prefill_tune() {
    static PrefillTune t = make_tune();
    return t;
}

int engine_prefill(Engine& e, int slot, const int32_t* tokens, int n, float* logits_last) {
    return prefill_impl(e, slot, tokens, n, logits_last, nullptr);
}

int engine_prefill_all(Engine& e, int slot, const int32_t* tokens, int n, float* logits_all) {
    return prefill_impl(e, slot, tokens, n, nullptr, logits_all);
}

// ------------------------------------------------------------------------------------------------ speculative verify
// T rows [anchor, d_1 .. d_{T-1}] at positions p .. p+T-1 through the batched executor with the verify scratch (8 rows, a few MB), logits of EVERY row and (optionally) the
// 4 taps of every row. The slot is left at p + T with the conv-input record kept; engine_verify_commit(keep) then moves it to exactly "the first `keep` rows were processed".
int engine_verify(Engine& E, int slot, const int32_t* tokens, int T, float* logits_all, float* taps, int exact) {
    if (T < 1 || T > kVerifyMaxRows) return -4;
    if (slot < 0 || slot >= (int)E.slots.size() || !E.slots[slot].used) return -1;
    SlotState& S = E.slots[slot];
    if (S.pos + T > E.opt.max_ctx) return -2;
    for (int i = 0; i < T; ++i)
        if (tokens[i] < 0 || tokens[i] >= E.cfg.n_vocab_padded) return -3;
    if (E.pool->size() > kMaxTh) return -5;
    if (!logits_all) return -9;
    if (chunk_limit(E, S, T) < T) return -6;   // SWA ring too small for an atomic T-row step (only with INK_SWA_BLOCKS < 16)
    PfScratch& X = get_scratch(E, kVerifyMaxRows, 1);
    if (X.pend_slot >= 0) return -8;           // previous verify not committed
    PrefillTune gt = prefill_tune();
    gt.attn_decode_kernels = (exact < 0 ? verify_exact_default() : exact != 0) ? 1 : 0;
    build_proj_items(E, gt, X.items_proj);
    pin_calling_thread(E);
    const size_t rec_visit = (size_t)(T + 3) * (2 * kKVW + 2 * kD);
    if (X.rec_floats < (size_t)kVisits * (kVerifyMaxRows + 3) * (2 * kKVW + 2 * kD)) {
        std::free(X.rec_mem);
        X.rec_floats = (size_t)kVisits * (kVerifyMaxRows + 3) * (2 * kKVW + 2 * kD);
        X.rec_mem = (float*)std::aligned_alloc(64, (X.rec_floats * sizeof(float) + 63) / 64 * 64);
    }
    PCtx c;
    c.E = &E;
    c.X = &X;
    c.S = &S;
    c.pool = E.pool.get();
    c.tu = gt;
    c.pos0 = S.pos;
    c.n = T;
    c.tokens = tokens;
    c.logits = nullptr;
    c.logits_all = logits_all;
    c.taps = taps;
    c.rec = X.rec_mem;
    c.rec_visit = rec_visit;
    c.verify = true;
    c.prof = E.prof_on || env_int("INK_PF_PROFILE", 0) != 0;
    c.last = 0;
    for (int v = 0; v < kVisits; ++v)
        if (!S.kv[v].ring) S.kv[v].n_blocks = std::max<uint32_t>(S.kv[v].n_blocks, (uint32_t)((S.pos + T - 1) / kKvBlock + 1));
    for (int tt = 0; tt < T; ++tt) X.tau[tt] = tau_of(E.cfg, S.pos + tt);
    const auto t0 = std::chrono::steady_clock::now();
    E.pool->run([&](int tid, int nth) {
        PTh t{tid, nth, 0};
        chunk_body(c, t);
        if (tid == 0) X.used_phases = t.phase;
    });
    if (X.used_phases >= kMaxPh) std::abort();
    for (int i = 0; i < X.used_phases; ++i) X.ctr[i].v.store(0, std::memory_order_relaxed);
    if (c.prof) {
        for (int th = 0; th < E.pool->size(); ++th)
            for (int p = 0; p < PF_COUNT; ++p) {
                X.prof.cpu_ns[p] += X.cpu[th].ns[p];
                X.cpu[th].ns[p] = 0;
            }
        X.prof.tokens += (uint64_t)T;
        X.prof.chunks++;
        X.prof.total_ticks += (uint64_t)(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() * E.prof.tsc_per_us);
    }
    X.pend_slot = slot;
    X.pend_T = T;
    X.pend_pos0 = S.pos;
    S.pos += T;
    if (S.hwm < S.pos) S.hwm = S.pos;   // the rejected rows wrote ring cells too (S3: shallow checkpoints are judged against hwm)
    return 0;
}

int engine_verify_commit(Engine& E, int slot, int keep) {
    PfScratch* X = find_scratch(E, 1);
    if (!X || X->pend_slot < 0 || X->pend_slot != slot) return -7;
    const int T = X->pend_T;
    if (keep < 1 || keep > T) return -4;
    SlotState& S = E.slots[slot];
    if (keep < T) {   // history after `keep` rows = rows [keep, keep + 3) of each record (rows 0..2 = the history before the verify)
        const size_t R = (size_t)T + 3, rec_visit = R * (2 * kKVW + 2 * kD);
        for (int v = 0; v < kVisits; ++v) {
            float* hist = S.hist + (size_t)v * kHistPerVisit;
            const float* rk = X->rec_mem + (size_t)v * rec_visit;
            const float* rv = rk + R * kKVW;
            const float* ra = rv + R * kKVW;
            const float* rf = ra + R * kD;
            std::memcpy(hist, rk + (size_t)keep * kKVW, 3 * kKVW * sizeof(float));
            std::memcpy(hist + 3 * kKVW, rv + (size_t)keep * kKVW, 3 * kKVW * sizeof(float));
            std::memcpy(hist + 6 * kKVW, ra + (size_t)keep * kD, 3 * kD * sizeof(float));
            std::memcpy(hist + 6 * kKVW + 3 * kD, rf + (size_t)keep * kD, 3 * kD * sizeof(float));
        }
    }
    S.pos = X->pend_pos0 + keep;   // KV rows beyond are junk: global rows > pos are never read, SWA ring cells they overwrote were out of every later window
    X->pend_slot = -1;
    return 0;
}

namespace {
void profile_report_impl(Engine& e, FILE* f, int kind) {
    PfScratch* X = find_scratch(e, kind);
    if (!X || !X->prof.tokens) {
        std::fprintf(f, "prefill profile: nothing recorded (set INK_PF_PROFILE=1)\n");
        return;
    }
    const PfProfile& P = X->prof;
    const double tpu = e.prof.tsc_per_us;
    double sum = 0;
    for (int p = 0; p < PF_COUNT; ++p) sum += (double)P.ticks[p];
    std::fprintf(f, "%s profile over %llu tokens in %llu chunks (thread 0 rdtsc, phase = time between barrier completions; includes straggler wait)\n",
                 kind ? "verify" : "prefill", (unsigned long long)P.tokens, (unsigned long long)P.chunks);
    const int nth = e.pool->size();
    double cpu_sum = 0;
    for (int p = 0; p < PF_COUNT; ++p) cpu_sum += P.cpu_ns[p];
    std::fprintf(f, "%-32s %9s %6s %8s %9s %9s %9s %8s | %9s %9s %9s\n", "phase", "ms/token", "%", "calls", "GMAC/tok", "GMAC/s", "MB/tok", "GB/s", "busy ms/t", "busy %", "GMAC/s/thr");
    const double nt = (double)P.tokens;
    for (int p = 0; p < PF_COUNT; ++p) {
        if (!P.calls[p]) continue;
        const double ms = (double)P.ticks[p] / tpu / 1000.0;   // total
        const double gm = (double)P.macs[p] / 1e9, mb = (double)P.bytes[p] / 1e6;
        const double busy_ms = P.cpu_ns[p] / 1e6 / nth;   // mean busy time per thread
        std::fprintf(f, "%-32s %9.3f %5.1f%% %8.0f %9.3f %9.1f %9.1f %8.1f | %9.3f %8.1f%% %9.2f\n", kPfNames[p], ms / nt, 100.0 * (double)P.ticks[p] / std::max(sum, 1.0),
                     (double)P.calls[p] / (double)P.chunks, gm / nt, ms > 0 ? gm / (ms / 1000.0) : 0.0, mb / nt, ms > 0 ? (mb / 1000.0) / (ms / 1000.0) : 0.0,
                     busy_ms / nt, 100.0 * P.cpu_ns[p] / std::max(cpu_sum, 1.0), P.cpu_ns[p] > 0 ? (gm * 1e9 / P.cpu_ns[p]) : 0.0);
    }
    std::fprintf(f, "%-32s %9.3f ms/token = %.1f tok/s (sum of phases %.3f; wall incl. launch %.3f)\n", "TOTAL", sum / tpu / 1000.0 / nt,
                 1e6 / (sum / tpu / nt), sum / tpu / 1000.0 / nt, (double)P.total_ticks / tpu / 1000.0 / nt);
    std::fprintf(f, "busy-time bound: mean busy CPU time per thread %.3f ms/token over %d threads = %.1f tok/s if no thread were ever preempted/waiting (barrier cost excluded)\n",
                 cpu_sum / 1e6 / nth / nt, nth, 1e9 / (cpu_sum / nth / nt));
}
}  // namespace

void engine_prefill_profile_reset(Engine& e) {
    if (PfScratch* X = find_scratch(e, 0)) X->prof = PfProfile();
}
void engine_prefill_profile_report(Engine& e, FILE* f) { profile_report_impl(e, f, 0); }
void engine_verify_profile_reset(Engine& e) {
    if (PfScratch* X = find_scratch(e, 1)) X->prof = PfProfile();
}
void engine_verify_profile_report(Engine& e, FILE* f) { profile_report_impl(e, f, 1); }

void engine_prefill_release(Engine& e) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_scr.erase({&e, 0});
    g_scr.erase({&e, 1});
}

}  // namespace ink

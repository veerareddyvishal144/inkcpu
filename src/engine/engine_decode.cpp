// Single-sequence decode executor: ONE ThreadPool::run() per token, phases separated by pool.barrier(). Owner: agent C1.
// Phase structure per visit (docs/ENGINE_DESIGN.md): A serial mix+norm | B projections | C conv/clip/kv/attention | wo+post-conv
// | F1 serial ffn mix+norm | F2 router | G expert gate_up+SiTU | H expert down+post-conv+residual. Control flow is identical on
// every thread (data-dependent values only index memory), so the dynamic-scheduling phase counters line up.
#include <pthread.h>
#include <sched.h>
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "engine_priv.h"
#include "ink/kernels_misc_simd.h"
#ifdef INK_HAVE_PREFILL
#include <cstdlib>

#include "ink/engine_prefill.h"
#endif

namespace ink {

namespace {

constexpr float kMixScale = 1.0f / 48.0f;   // 1/sqrtf(2304), exact
constexpr float kSituBeta = 4.0f, kSituLin = 25.0f;

struct Ctx {
    Engine* E;
    Scratch* X;
    SlotState* S;
    PhaseCounter* ctr;
    int64_t pos;
    float tau;
    int32_t token;
    float* logits;
    float* taps;        // [kEngineTapFloats] or nullptr: DSpark taps of this token (decode executor: written by tid 0)
    bool prof;
    uint64_t last;
};
struct Th {
    int tid, nth, phase;
};

inline void mark(Ctx& c, const Th& t, int ph, uint64_t bytes) {
    if (!c.prof || t.tid != 0) return;
    const uint64_t now = __rdtsc();
    EngineProfile& P = c.E->prof;
    P.ticks[ph] += now - c.last;
    P.calls[ph]++;
    P.bytes[ph] += bytes;
    c.last = now;
}
#define BAR(ph, by)             \
    do {                        \
        pool.barrier();         \
        mark(c, t, (ph), (by)); \
    } while (0)

// dynamic row scheduling inside a phase: every thread claims chunks from a shared counter (one counter per phase, zeroed per token).
// Guided: the first 3/4 of the rows go out in chunks of `chunk`, the last quarter in chunks of `fine` (shorter tail at the barrier).
// Static-first (E1, wave 1): chunk indices tid, tid + nth, ... (`g_static_rounds` of them) belong to thread tid, so a thread that is idle before the
// barrier (serial phases, half the threads at short-context attention, early finishers) can PREFETCH its first chunk(s) of the next phase into L2.
// Rows are grouped exactly as before, only the thread that runs a chunk changes, so every result is bit-identical. env INK_PF_ROUNDS: default 0 = OFF
// (measured 2026-11-23, docs/status/E1.md: even with the stop epoch below, idle-thread weight prefetch + static rounds lost ~5 % at ctx 16 against the
// plain dynamic scheduling, because software prefetches drain through the L1 fill buffers at ~3 GB/s per core and still delayed the barriers); 1-2 = on.
int g_static_rounds = -1;
inline int static_rounds() {
    if (g_static_rounds < 0) {
        const char* e = std::getenv("INK_PF_ROUNDS");
        g_static_rounds = e && *e ? std::max(0, std::atoi(e)) : 0;
    }
    return g_static_rounds;
}
struct DynGeom {
    int64_t total, chunk, fine, nbig, big_rows, nchunks;
    bool guided;
    DynGeom(int64_t total_, int64_t chunk_, int64_t fine_) : total(total_), chunk(chunk_), fine(fine_) {
        guided = !(fine >= chunk || total <= 4 * chunk);
        nbig = guided ? (total * 3 / 4) / chunk : 0;
        big_rows = nbig * chunk;
        nchunks = guided ? nbig + (total - big_rows + fine - 1) / fine : (total + chunk - 1) / chunk;
    }
    void range(int64_t i, int64_t& b, int64_t& e) const {
        if (!guided || i < nbig) {
            b = i * chunk;
            e = std::min<int64_t>(b + chunk, total);
        } else {
            b = big_rows + (i - nbig) * fine;
            e = std::min<int64_t>(b + fine, total);
        }
    }
    // the static chunk of thread t in `round` (false: none)
    bool static_range(const Th& t, int round, int64_t& b, int64_t& e) const {
        const int64_t i = t.tid + (int64_t)round * t.nth;
        if (round >= static_rounds() || i >= nchunks) return false;
        range(i, b, e);
        return true;
    }
};
// Returns how many threads had already left the phase when this one left (0 = first out, nth-1 = last out). The last one bumps the prefetch
// stop epoch so that early finishers stop prefetching and reach the barrier.
template <class F>
inline int dyn(Ctx& c, Th& t, int64_t total, int64_t chunk, int64_t fine, F&& f) {
    std::atomic<int64_t>& a = c.ctr[t.phase++].v;
    const DynGeom G(total, chunk, fine);
    const int64_t base = std::min<int64_t>(G.nchunks, (int64_t)static_rounds() * t.nth);
    int64_t b, e;
    for (int64_t i = t.tid; i < base; i += t.nth) {
        G.range(i, b, e);
        f(b, e);
    }
    for (;;) {
        const int64_t v = a.fetch_add(1, std::memory_order_relaxed);
        const int64_t i = base + v;
        if (i >= G.nchunks) {
            const int before = (int)(v - (G.nchunks - base));
            if (before == t.nth - 1) c.X->pf_stop.fetch_add(1, std::memory_order_relaxed);
            return before;
        }
        G.range(i, b, e);
        f(b, e);
    }
}
template <class F>
inline int dyn(Ctx& c, Th& t, int64_t total, int64_t chunk, F&& f) {
    return dyn(c, t, total, chunk, chunk, std::forward<F>(f));
}
// a thread that left a dynamic phase among the first half has >= half a chunk of idle time: enough to issue its next-phase prefetch
inline bool early(int before, const Th& t) { return static_rounds() > 0 && before < t.nth / 2; }

// ---- next-phase weight prefetch (into L2; the same thread consumes the rows right after the barrier).
// Software prefetches go through the L1 fill buffers (~3 GB/s per core on this host), so an idle thread can only pull ~40-50 KB during a 16 us serial
// segment: the loops stop as soon as the stop epoch moves (the thread everyone waits for bumps it right before entering the barrier) so that the
// prefetch never delays the barrier, and a byte budget caps the rest.
constexpr size_t kPfMaxBytes = (size_t)96 << 10;
struct Pf {
    const std::atomic<uint32_t>* stop;
    uint32_t e0;
    size_t budget;
    explicit Pf(const Scratch& X, size_t budget_ = kPfMaxBytes) : stop(&X.pf_stop), e0(X.pf_stop.load(std::memory_order_relaxed)), budget(budget_) {}
    bool stopped() const { return stop->load(std::memory_order_relaxed) != e0; }
    void bytes(const void* p, size_t n) {
        const char* q = (const char*)p;
        n = std::min(n, budget);
        for (size_t o = 0; o < n; o += 64) {
            _mm_prefetch(q + o, _MM_HINT_T1);
            if ((o & 1023) == 960 && stopped()) {
                budget = 0;
                return;
            }
        }
        budget -= n;
    }
    void rows(const void* w, size_t rs, int64_t r0, int64_t r1) {
        if (r1 > r0 && budget) bytes((const char*)w + (size_t)r0 * rs, (size_t)(r1 - r0) * rs);
    }
    // virtual rows [vr0, vr1) of a GemvMat list (the row space of gemv_multi_*)
    void multi(const GemvMat* m, int nm, int64_t vr0, int64_t vr1) {
        int64_t off = 0;
        for (int i = 0; i < nm && budget; ++i) {
            const int64_t lo = std::max(vr0, off), hi = std::min(vr1, off + m[i].nrows);
            if (lo < hi) rows(m[i].w, m[i].row_stride, lo - off, hi - off);
            off += m[i].nrows;
        }
    }
    // this thread's static chunk of round 0 of a coming gemv_multi phase with geometry (total, chunk, fine)
    void static_multi(const Th& t, const GemvMat* m, int nm, int64_t total, int64_t chunk, int64_t fine) {
        const DynGeom G(total, chunk, fine);
        int64_t b, e;
        if (G.static_range(t, 0, b, e)) multi(m, nm, b, e);
    }
};
inline void pf_signal(Scratch& X) { X.pf_stop.fetch_add(1, std::memory_order_relaxed); }

// ---- E1 debug (env INK_FINE=1): where the short phases spend their time, thread 0 only
struct FineProf {
    bool on = false;
    uint64_t a_work = 0, a_bar = 0, a_n = 0;                             // phase A: serial work, barrier (completion - arrival)
    uint64_t a_t1_early = 0, a_t1_late = 0;                              // thread 1 arrived this much before / after thread 0 at the A barrier
    uint64_t c_pre = 0, c_task = 0, c_fin = 0, c_bar = 0, c_n = 0;      // phase C (nsplit 1): group 0 pre / task / finish, barrier
    uint64_t s_pre = 0, s_bar1 = 0, s_items = 0, s_bar2 = 0, s_n = 0;   // phase C (nsplit > 1): group 0 pre, barrier, thread 0's items, final barrier
    uint64_t f_work = 0, f_bar = 0, f_n = 0;                             // phase F1
};
FineProf g_fine;
inline bool fine_on() {
    static const int v = [] {
        const char* e = std::getenv("INK_FINE");
        return e && *e ? std::atoi(e) : 0;
    }();
    return v != 0;
}

inline void gemv_multi_lp(int type, const GemvMat* m, int nm, int k, int64_t a, int64_t b) {
    if (type == T_BF16)
        gemv_multi_bf16(m, nm, k, a, b);
    else
        gemv_multi_f32(m, nm, k, a, b);
}

inline void bf16_round(const float* x, float* y, int n) {   // n % 16 == 0
    for (int i = 0; i < n; i += 16)
        _mm512_storeu_ps(y + i, simd::bf16x16_to_f32(simd::f32x16_to_bf16(_mm512_loadu_ps(x + i))));
}

// ------------------------------------------------------------------------------------------------ loop controllers
// nj controllers (index cidx[j] into L.ctrl_*): X.ctrl_in[j] (= [rms(cur), rms(msg)]) was filled by tid 0 before a barrier.
void controllers(Ctx& c, Th& t, const LayerPacked& L, int nj, const int* cidx, const float* const* cur_in, const float* const* msg,
                 float* const* out) {
    Scratch& X = *c.X;
    Engine& E = *c.E;
    ThreadPool& pool = *E.pool;
    uint64_t bytes = 0;
    GemvMat dm[4];
    for (int j = 0; j < nj; ++j) {
        const Mat& m = L.ctrl_down[cidx[j]];
        dm[j] = GemvMat{m.w, m.rs, X.ctrl_in[j], X.ctrl_g[j], m.rows};
        const Mat& g = L.ctrl_gain[cidx[j]];
        bytes += m.bytes() + 2 * g.bytes();
    }
    const int ty = L.ctrl_down[cidx[0]].type;
    const int before = dyn(c, t, (int64_t)nj * kCtrlRank, 4, [&](int64_t b, int64_t e) { gemv_multi_lp(ty, dm, nj, 2 * kD, b, e); });
    // fused second phase: item = (controller j, 64-channel block): the block's gain + forget rows (same rows, same kernel as the whole-matrix
    // matvec in 64-row chunks: bit-identical) and ctrl_apply of the block right away -> 2 barriers per call instead of 3
    constexpr int64_t kBlk = 64, nblk = kD / kBlk;
    auto gf_mats = [&](int j, int64_t b, GemvMat* gm) {
        const Mat& g = L.ctrl_gain[cidx[j]];
        const Mat& f = L.ctrl_forget[cidx[j]];
        gm[0] = GemvMat{(const char*)g.w + (size_t)b * g.rs, g.rs, X.ctrl_g[j], X.ctrl_gain[j] + b, (int)kBlk};
        gm[1] = GemvMat{(const char*)f.w + (size_t)b * f.rs, f.rs, X.ctrl_g[j], X.ctrl_forget[j] + b, (int)kBlk};
    };
    if (early(before, t)) {   // prefetch my static item of the fused phase while the others finish the down matvec (stops when the last one leaves)
        const DynGeom G((int64_t)nj * nblk, 1, 1);
        int64_t u0, u1;
        if (G.static_range(t, 0, u0, u1)) {
            GemvMat gm[2];
            gf_mats((int)(u0 / nblk), (u0 % nblk) * kBlk, gm);
            Pf pf(X);
            pf.multi(gm, 2, 0, 2 * kBlk);
        }
    }
    BAR(PH_CTRL, bytes);
    dyn(c, t, (int64_t)nj * nblk, 1, [&](int64_t u, int64_t) {
        const int j = (int)(u / nblk);
        const int64_t b = (u % nblk) * kBlk;
        GemvMat gm[2];
        gf_mats(j, b, gm);
        gemv_multi_lp(ty, gm, 2, kCtrlRank, 0, 2 * kBlk);
        ctrl_apply(cur_in[j] + b, msg[j] + b, X.ctrl_gain[j] + b, X.ctrl_forget[j] + b, out[j] + b, (size_t)kBlk);
    });
    BAR(PH_CTRL, 0);
}

// ------------------------------------------------------------------------------------------------ one layer visit
void visit(Ctx& c, Th& t, const Visit& vis, int& ci, int& nb) {
    Engine& E = *c.E;
    Scratch& X = *c.X;
    ThreadPool& pool = *E.pool;
    SlotState& S = *c.S;
    const ModelConfig& cfg = E.cfg;
    const LayerPacked& L = E.layers[vis.phys];
    const int tid = t.tid, nth = t.nth;
    const bool swa = L.swa, msgs = vis.has_message;
    const int64_t pos = c.pos;
    const float eps = cfg.rms_eps;
    float* cur = X.cur[ci];
    float* nxt = X.cur[1 - ci];
    float* hist = S.hist + (size_t)vis.index * kHistPerVisit;
    float* h_k = hist;
    float* h_v = hist + 3 * kKVW;
    float* h_a = hist + 6 * kKVW;
    float* h_f = h_a + 3 * kD;
    const bool block_start = (vis.phys % cfg.attn_res_block_size) == 0;

    // ---------------------------------------------------------------- A: attention-input mix, norms (serial on tid 0)
    if (c.taps && tid == 0)   // DSpark tap: the stream entering visit 5 / 23 / 41 (read-only here; `cur` is not written before phase E)
        for (int k = 0; k < 3; ++k)
            if (vis.index == kTapVisit[k]) std::memcpy(c.taps + (size_t)k * kD, cur, kD * sizeof(float));
    const float* in_raw[3] = {cur, cur, cur};
    if (L.attn_res_query)
        for (int j = 0; j < 3; ++j) in_raw[j] = X.mix[swa ? 0 : j];
    const block_q8_0* xq[3];
    const float* xb[3];
    if (swa && !msgs) {
        for (int j = 0; j < 3; ++j) {
            xq[j] = X.inq[0];
            xb[j] = X.inbf[0];
        }
    } else {
        for (int j = 0; j < 3; ++j) {
            xq[j] = X.inq[j];
            xb[j] = X.inbf[j];
        }
    }
    auto norm_quant = [&](const float* x, const float* w, int j, bool bf) {
        rmsnorm(x, w, X.inx[j], kD, eps);
        quantize_q8_0(X.inx[j], X.inq[j], kD, Q8Mode::X86);
        if (bf) bf16_round(X.inx[j], X.inbf[j], kD);
    };
    auto norms_all = [&]() {
        if (swa && !msgs) {
            norm_quant(in_raw[0], L.attn_norm, 0, true);
        } else if (swa) {
            for (int j = 0; j < 3; ++j) norm_quant(in_raw[j], L.attn_norm, j, j == 0);
        } else {
            for (int j = 0; j < 3; ++j) norm_quant(in_raw[j], L.inp_norm[j], j, true);
        }
    };
    // phase B descriptors are built here so that threads idle during the serial segments can prefetch their first chunks of it
    GemvMat pm[8] = {{L.wq.w, L.wq.rs, xq[0], X.q, L.wq.rows},       {L.wk.w, L.wk.rs, xq[1], X.kp, L.wk.rows},
                     {L.wv.w, L.wv.rs, xq[2], X.vp, L.wv.rows},       {L.wgate.w, L.wgate.rs, xq[0], X.gate, L.wgate.rows},
                     {L.wrel.w, L.wrel.rs, xq[0], X.rel, L.wrel.rows}, {L.sink.w, L.sink.rs, xb[0], X.sink, L.sink.rows},
                     {L.kproj.w, L.kproj.rs, xb[1], X.kdyn, L.kproj.rows}, {L.vproj.w, L.vproj.rs, xb[2], X.vdyn, L.vproj.rows}};
    const int nsm = swa ? 1 : 3;
    int64_t nqr = 0, nsr = 0;
    for (int i = 0; i < 5; ++i) nqr += pm[i].nrows;
    for (int i = 0; i < nsm; ++i) nsr += pm[5 + i].nrows;
    const int64_t proj_chunk = E.tune.chunk_proj, proj_fine = std::max(4, E.tune.chunk_proj / 4);
    auto pf_proj = [&]() { Pf(X).static_multi(t, pm, 5 + nsm, nqr + nsr, proj_chunk, proj_fine); };
    auto pf_ctrl_down = [&](int j0, int nj) {   // first phase of controllers(): nj*64 virtual rows in chunks of 4
        GemvMat dm[4];
        for (int j = 0; j < nj; ++j) dm[j] = GemvMat{L.ctrl_down[j0 + j].w, L.ctrl_down[j0 + j].rs, nullptr, nullptr, L.ctrl_down[j0 + j].rows};
        Pf(X).static_multi(t, dm, nj, (int64_t)nj * kCtrlRank, 4, 4);
    };
    auto pf_router = [&]() {
        GemvMat rm{L.router.w, L.router.rs, nullptr, nullptr, L.router.rows};
        Pf(X).static_multi(t, &rm, 1, kNExp, E.tune.chunk_router, E.tune.chunk_router);
    };
    const int64_t wo_chunk = std::max<int64_t>(16, (E.tune.chunk_wo / 16) * 16);
    auto pf_wo = [&]() {
        GemvMat wm{L.wo.w, L.wo.rs, nullptr, nullptr, L.wo.rows};
        Pf(X).static_multi(t, &wm, 1, kD, wo_chunk, 16);
    };
    const bool pf_on = static_rounds() > 0;
    const bool fine = tid == 0 && fine_on();
    uint64_t ft0 = fine ? __rdtsc() : 0;

    if (tid == 0) {
        if (msgs) {
            const float* ms[2] = {X.bank_b, X.bank_out};
            float* mo[4] = {X.msgs[0], X.msgs[1], X.msgs[2], X.msgs[3]};
            depth_mix(ms, 2, L.msg_query, 4, mo, kD, kMixScale);
        }
        if (L.attn_res_query) {
            const float* st[8];
            int ns = 0;
            for (int b = nb - 1; b >= 0; --b) st[ns++] = X.blocks[b];
            st[ns++] = cur;
            float* mo[3] = {X.mix[0], X.mix[1], X.mix[2]};
            depth_mix(st, ns, L.attn_res_query, swa ? 1 : 3, mo, kD, kMixScale);
        }
        if (msgs) {
            for (int j = 0; j < 3; ++j) ctrl_gate_input(in_raw[j], X.msgs[j], X.ctrl_in[j], kD, cfg.hard_coded.ctrl_rms_eps);
        } else {
            norms_all();
        }
        pf_signal(X);   // serial work done: idle prefetchers stop and come to the barrier
    } else if (pf_on) {
        if (msgs) pf_ctrl_down(0, 3); else pf_proj();
    }
    uint64_t ft1 = fine ? __rdtsc() : 0;
    if (tid == 1 && fine_on()) X.fine_t1_arrive.store(__rdtsc(), std::memory_order_relaxed);   // waiter side: when did thread 1 reach this barrier
    BAR(PH_MIX, 0);
    if (fine) {
        const uint64_t ft2 = __rdtsc();
        g_fine.a_work += ft1 - ft0;
        g_fine.a_bar += ft2 - ft1;
        g_fine.a_n++;
        const uint64_t t1 = X.fine_t1_arrive.load(std::memory_order_relaxed);
        if (t1 && t1 < ft1) g_fine.a_t1_early += ft1 - t1; else if (t1) g_fine.a_t1_late += t1 - ft1;
    }
    if (msgs) {
        const int cidx[3] = {0, 1, 2};
        const float* cin[3] = {in_raw[0], in_raw[1], in_raw[2]};
        const float* mm[3] = {X.msgs[0], X.msgs[1], X.msgs[2]};
        float* outs[3] = {X.cin[0], X.cin[1], X.cin[2]};
        controllers(c, t, L, 3, cidx, cin, mm, outs);
        for (int j = 0; j < 3; ++j) in_raw[j] = X.cin[j];
        if (tid == 0) {
            norms_all();
            pf_signal(X);
        } else if (pf_on) {
            pf_proj();
        }
        BAR(PH_MIX, 0);
    }

    // ---------------------------------------------------------------- B: projections q|k|v|gate|rel (+ sink, dyn taps)
    {
        dyn(c, t, nqr + nsr, proj_chunk, proj_fine, [&](int64_t b, int64_t e) {
            if (b < nqr) gemv_multi_q8_0(pm, 5, kD, b, std::min(e, nqr));
            if (e > nqr) gemv_multi_bf16(pm + 5, nsm, kD, std::max(b, nqr) - nqr, e - nqr);
        });
        uint64_t by = L.wq.bytes() + L.wk.bytes() + L.wv.bytes() + L.wgate.bytes() + L.wrel.bytes() + L.sink.bytes();
        if (!swa) by += L.kproj.bytes() + L.vproj.bytes();
        BAR(PH_PROJ, by);
    }

    // ---------------------------------------------------------------- C: conv, clips, kv append, attention
    {
        KvLayer& kv = S.kv[vis.index];
        AttnQuery aq{X.q, X.rel, X.sink, X.gate, pos, swa ? 1.0f : c.tau};
        const int64_t n_keys = swa ? std::min<int64_t>(pos + 1, kSwaWindow) : pos + 1;
        const int nsplit = attn_decode_nsplit(n_keys, E.tune.split_keys, kMaxSplit);   // shared rule (kernels_misc.h): keys only, never threads
        // E1 experiment (env INK_PF_ATTN_IN=1): before the group's work, prefetch everything it touches that the weight stream has evicted since the
        // previous token: the conv histories and taps of its channels, the clip scales, the rel-bias rows of the query's block, the other cores' projection
        // outputs, the K column lines kv_append will write (RFO) and, at short context, the K block the task reads. Pure hints: numerics unchanged.
        auto pf_group_inputs = [&](int g) {
            auto pf = [](const void* p, size_t bytes, int hint) {
                const char* q = (const char*)p;
                for (size_t o = 0; o < bytes; o += 64)
                    if (hint) _mm_prefetch(q + o, _MM_HINT_T0); else _mm_prefetch(q + o, _MM_HINT_T1);
            };
            const size_t c0 = (size_t)g * kHeadDim;
            for (int r = 0; r < 3; ++r) {
                pf(h_k + (size_t)r * kKVW + c0, kHeadDim * 4, 1);
                pf(h_v + (size_t)r * kKVW + c0, kHeadDim * 4, 1);
            }
            for (int tp = 0; tp < kConvTaps; ++tp) {
                pf(L.kconv + (size_t)tp * kKVW + c0, kHeadDim * 4, 1);
                pf(L.vconv + (size_t)tp * kKVW + c0, kHeadDim * 4, 1);
            }
            pf(L.q_clip + g * (kGqa * kHeadDim), kGqa * kHeadDim * 4, 1);
            pf(L.rel_clip + g * (kGqa * kRelFeat), kGqa * kRelFeat * 4, 1);
            pf(X.kp + c0, kHeadDim * 4, 1);
            pf(X.vp + c0, kHeadDim * 4, 1);
            pf(X.q + g * (kGqa * kHeadDim), kGqa * kHeadDim * 4, 1);
            pf(X.rel + g * (kGqa * kRelFeat), kGqa * kRelFeat * 4, 1);
            pf(X.gate + g * (kGqa * kHeadDim), kGqa * kHeadDim * 4, 0);
            {   // rel-bias rows for the block holding the query position: table[f][64 + e_base .. +64)
                const int64_t jb = (pos / kKvBlock) * kKvBlock, e_base = jb - pos + (kRelExtent - 1);
                for (int f = 0; f < kRelFeat; ++f) pf(L.rel_table + (size_t)f * kRelTableLen + 64 + e_base, kKvBlock * 4, 0);
            }
            {   // the K column of this position (128 lines, written by kv_append) and, if the visible range is one block, the whole K block of the group
                const size_t slot = kv.ring ? (size_t)(pos % ((int64_t)kv.n_blocks * kKvBlock)) : (size_t)pos;
                const uint16_t* kb = kv.k + (slot / kKvBlock) * kKvBlockElems + (size_t)g * (kKvBlock * kHeadDim);
                if (n_keys <= kKvBlock) {
                    pf(kb, (size_t)kHeadDim * kKvBlock * 2, 0);
                } else {
                    const char* col = (const char*)(kb + slot % kKvBlock);
                    for (int d = 0; d < kHeadDim; ++d) _mm_prefetch(col + (size_t)d * kKvBlock * 2, _MM_HINT_T0);
                }
            }
        };
        static const bool pf_attn_in = [] {   // default ON (same-window A/B 2026-11-23: +4 % tok/s at ctx 16 and 1000); INK_PF_ATTN_IN=0 disables
            const char* e = std::getenv("INK_PF_ATTN_IN");
            return !(e && *e && std::atoi(e) == 0);
        }();
        auto group_pre = [&](int g) {
            if (pf_attn_in) pf_group_inputs(g);
            sconv_step(X.kp, h_k, L.kconv, X.kp, kKVW, (size_t)g * kHeadDim, (size_t)(g + 1) * kHeadDim, swa ? nullptr : X.kdyn, kHeadDim);
            sconv_step(X.vp, h_v, L.vconv, X.vp, kKVW, (size_t)g * kHeadDim, (size_t)(g + 1) * kHeadDim, swa ? nullptr : X.vdyn, kHeadDim);
            l2clip_heads(X.kp + g * kHeadDim, nullptr, 1, kHeadDim, cfg.qk_clip, 1.0f);
            vnorm_heads(X.vp + g * kHeadDim, X.vp + g * kHeadDim, 1, kHeadDim, eps);
            l2clip_heads(X.q + g * (kGqa * kHeadDim), L.q_clip + g * (kGqa * kHeadDim), kGqa, kHeadDim, cfg.qk_clip, 1.0f);
            l2clip_heads(X.rel + g * (kGqa * kRelFeat), L.rel_clip + g * (kGqa * kRelFeat), kGqa, kRelFeat, cfg.rel_clip, 1.0f);
            kv_append(kv, pos, X.kp, X.vp, g, g + 1);
        };
        auto group_fin = [&](int g) {
            attn_decode_finish(aq, g, &X.parts[g * nsplit], nsplit, X.ao);
            quantize_q8_0(X.ao + g * (kGqa * kHeadDim), X.aoq + g * (kGqa * kHeadDim / 32), kGqa * kHeadDim, Q8Mode::X86);
        };
        const uint64_t kvb = (uint64_t)n_keys * kKVW * 4;   // K+V fp16 bytes read
        if (nsplit == 1) {
            uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int g = tid; g < kNKvHeads; g += nth) {
                if (fine) c0 = __rdtsc();
                group_pre(g);
                if (fine) c1 = __rdtsc();
                attn_decode_task(kv, L.rel_table, aq, g, 0, 1, &X.parts[g]);
                if (fine) c2 = __rdtsc();
                group_fin(g);
                if (fine) c3 = __rdtsc();
            }
            if (tid < kNKvHeads) pf_signal(X);                 // the first group to finish stops the idle prefetchers
            else if (pf_on) pf_wo();                            // threads without a group: fetch their first wo chunk meanwhile
            BAR(PH_ATTN, kvb);
            if (fine) {
                const uint64_t c4 = __rdtsc();
                g_fine.c_pre += c1 - c0;
                g_fine.c_task += c2 - c1;
                g_fine.c_fin += c3 - c2;
                g_fine.c_bar += c4 - c3;
                g_fine.c_n++;
            }
        } else {
            const uint64_t s0 = fine ? __rdtsc() : 0;
            for (int g = tid; g < kNKvHeads; g += nth) group_pre(g);
            const uint64_t s1 = fine ? __rdtsc() : 0;
            BAR(PH_ATTN, 0);
            const uint64_t s2 = fine ? __rdtsc() : 0;
            // items (g, s) dynamically; the thread completing the LAST item of a group merges that group (per-group arrival counters, zeroed per
            // token like the phase counters): no barrier between the items and the finish, and the merge order is fixed so the result is the same
            const int gbase = t.phase;
            t.phase += kNKvHeads;
            const int before = dyn(c, t, (int64_t)kNKvHeads * nsplit, 1, [&](int64_t b, int64_t) {
                const int g = (int)(b / nsplit), s = (int)(b % nsplit);
                attn_decode_task(kv, L.rel_table, aq, g, s, nsplit, &X.parts[g * nsplit + s]);
                if (c.ctr[gbase + g].v.fetch_add(1, std::memory_order_acq_rel) + 1 == nsplit) group_fin(g);
            });
            const uint64_t s3 = fine ? __rdtsc() : 0;
            (void)before;   // no wo prefetch here: the attention items saturate DRAM and the issue time delayed the barrier (fine profile 19:11)
            BAR(PH_ATTN, kvb);
            if (fine) {
                const uint64_t s4 = __rdtsc();
                g_fine.s_pre += s1 - s0;
                g_fine.s_bar1 += s2 - s1;
                g_fine.s_items += s3 - s2;
                g_fine.s_bar2 += s4 - s3;
                g_fine.s_n++;
            }
        }
        if (E.trace && tid == 0) {
            E.trace("Qcur", vis.index, X.q, kQW);
            E.trace("Kcur", vis.index, X.kp, kKVW);
            E.trace("Vcur", vis.index, X.vp, kKVW);
            E.trace("Rcur", vis.index, X.rel, kRelW);
        }
    }

    // ---------------------------------------------------------------- E: wo + post-conv + residual (per channel range)
    {
        const int64_t chunk = wo_chunk;
        dyn(c, t, kD, chunk, 16, [&](int64_t b, int64_t e) {
            gemv_q8_0((const char*)L.wo.w + (size_t)b * L.wo.rs, L.wo.rs, X.aoq, kQW, X.wo_out + b, (int)(e - b));
            float* yc = block_start ? nxt : X.attn_out;
            sconv_step(X.wo_out, h_a, L.aconv, yc, kD, (size_t)b, (size_t)e);
            if (!block_start) vadd(cur + b, X.attn_out + b, nxt + b, (size_t)(e - b));
        });
        BAR(PH_WO, L.wo.bytes());
    }

    // ---------------------------------------------------------------- F1: ffn depth mix + norm (serial)
    auto ffn_norm_quant = [&](const float* m) {
        rmsnorm(m, L.ffn_norm, X.xffn, kD, eps);
        quantize_q8_0(X.xffn, X.xq, kD, Q8Mode::X86);
    };
    const uint64_t ff0 = fine ? __rdtsc() : 0;
    if (tid == 0) {
        if (E.trace) {
            E.trace("attn_out", vis.index, block_start ? nxt : X.attn_out, kD);
            E.trace("ffn_inp", vis.index, nxt, kD);
        }
        if (block_start) std::memcpy(X.blocks[nb], cur, kD * sizeof(float));   // push inpSA
        const int nbk = nb + (block_start ? 1 : 0);
        const float* st[8];
        int ns = 0;
        for (int b = nbk - 1; b >= 0; --b) st[ns++] = X.blocks[b];
        st[ns++] = nxt;
        float* mo[1] = {X.fmix};
        depth_mix(st, ns, L.ffn_res_query, 1, mo, kD, kMixScale);
        if (msgs)
            ctrl_gate_input(X.fmix, X.msgs[3], X.ctrl_in[0], kD, cfg.hard_coded.ctrl_rms_eps);
        else
            ffn_norm_quant(X.fmix);
        pf_signal(X);
    } else if (pf_on) {
        if (msgs) pf_ctrl_down(3, 1); else pf_router();
    }
    if (block_start) ++nb;
    const uint64_t ff1 = fine ? __rdtsc() : 0;
    BAR(PH_FFNMIX, 0);
    if (fine) {
        const uint64_t ff2 = __rdtsc();
        g_fine.f_work += ff1 - ff0;
        g_fine.f_bar += ff2 - ff1;
        g_fine.f_n++;
    }
    if (msgs) {
        const int cidx[1] = {3};
        const float* cin[1] = {X.fmix};
        const float* mm[1] = {X.msgs[3]};
        float* outs[1] = {X.cin[3]};
        controllers(c, t, L, 1, cidx, cin, mm, outs);
        if (tid == 0) {
            ffn_norm_quant(X.cin[3]);
            pf_signal(X);
        } else if (pf_on) {
            pf_router();
        }
        BAR(PH_FFNMIX, 0);
    }

    // ---------------------------------------------------------------- F2: router (parallel matvec, top-8 computed redundantly)
    {
        const Mat& R = L.router;
        GemvMat rm[1] = {{R.w, R.rs, X.xffn, X.rlogits, R.rows}};
        dyn(c, t, kNExp, E.tune.chunk_router, [&](int64_t b, int64_t e) { gemv_multi_lp(R.type, rm, 1, kD, b, e); });
        BAR(PH_ROUTER, R.bytes());
    }
    int ids[kTopK];
    float rw[kTopK];
    router_topk(X.rlogits, L.exp_bias, kNExp, kTopK, cfg.expert_weights_scale, cfg.hard_coded.expert_weights_norm_eps, ids, rw);

    // ---------------------------------------------------------------- G: expert gate_up + scale + SiTU + Q8_0 (one block per unit)
    const bool nv = L.expert_type == T_NVFP4;
    dyn(c, t, (int64_t)kTopK * (kFF / 32), 1, [&](int64_t u, int64_t) {
        const int i = (int)(u / (kFF / 32)), cb = (int)(u % (kFF / 32));
        const int ex = ids[i];
        const char* base = (const char*)L.gu + (size_t)ex * L.gu_eb;
        const char* wg = base + (size_t)(cb * 32) * L.gu_rs;
        const char* wu = base + (size_t)(kFF + cb * 32) * L.gu_rs;
        alignas(64) float g[32], up[32];
        if (nv) {
            if (L.nvfp4r) {   // K1: arena scales re-encoded at load (INK_NVFP4R=1); bit-identical results
                gemv_nvfp4r(wg, L.gu_rs, X.xq, kD, g, 32);
                gemv_nvfp4r(wu, L.gu_rs, X.xq, kD, up, 32);
            } else {
                gemv_nvfp4(wg, L.gu_rs, X.xq, kD, g, 32);
                gemv_nvfp4(wu, L.gu_rs, X.xq, kD, up, 32);
            }
            const float s = L.gu_scale[ex];
            vscale(g, s, g, 32);
            vscale(up, s, up, 32);
        } else {
            gemv_q8_0(wg, L.gu_rs, X.xq, kD, g, 32);
            gemv_q8_0(wu, L.gu_rs, X.xq, kD, up, 32);
        }
        vsitu_q8_0(g, up, &X.actq[i][cb], 32, kSituBeta, kSituLin, Q8Mode::X86);
    });
    BAR(PH_GATEUP, (uint64_t)kTopK * 2 * kFF * L.gu_rs);

    // ---------------------------------------------------------------- H: expert down (8 experts summed in top-k order) + post-conv + residual
    {
        float sc[kTopK];
        for (int i = 0; i < kTopK; ++i) sc[i] = nv ? L.dn_scale[ids[i]] : 1.0f;
        const int64_t chunk = std::max<int64_t>(16, (E.tune.chunk_down / 16) * 16);
        dyn(c, t, kD, chunk, 16, [&](int64_t b, int64_t e) {
            alignas(64) float tmp[kTopK][16];
            for (int64_t r0 = b; r0 < e; r0 += 16) {
                const int nr = (int)std::min<int64_t>(16, e - r0);
                for (int i = 0; i < kTopK; ++i) {
                    const char* wd = (const char*)L.dn + (size_t)ids[i] * L.dn_eb + (size_t)r0 * L.dn_rs;
                    if (nv)
                        (L.nvfp4r ? gemv_nvfp4r : gemv_nvfp4)(wd, L.dn_rs, X.actq[i], kFF, tmp[i], nr);   // K1: see phase G
                    else
                        gemv_q8_0(wd, L.dn_rs, X.actq[i], kFF, tmp[i], nr);
                }
                // d_i = (t_i * down_scale[e_i]) * w_i ; out = d_0 + d_1 + ... + d_7 (sequential, top-k order)
                if (nr == 16) {
                    __m512 acc = _mm512_mul_ps(_mm512_mul_ps(_mm512_load_ps(tmp[0]), _mm512_set1_ps(sc[0])), _mm512_set1_ps(rw[0]));
                    for (int i = 1; i < kTopK; ++i)
                        acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_mul_ps(_mm512_load_ps(tmp[i]), _mm512_set1_ps(sc[i])), _mm512_set1_ps(rw[i])));
                    _mm512_storeu_ps(X.moe_out + r0, acc);
                } else {
                    for (int r = 0; r < nr; ++r) {
                        float s = (tmp[0][r] * sc[0]) * rw[0];
                        for (int i = 1; i < kTopK; ++i) s = s + (tmp[i][r] * sc[i]) * rw[i];
                        X.moe_out[r0 + r] = s;
                    }
                }
            }
            sconv_step(X.moe_out, h_f, L.fconv, X.moe_out, kD, (size_t)b, (size_t)e);
            vadd(nxt + b, X.moe_out + b, nxt + b, (size_t)(e - b));
        });
        BAR(PH_DOWN, (uint64_t)kTopK * kD * L.dn_rs);
    }
    ci = 1 - ci;
    if (E.trace && tid == 0) E.trace("l_out", vis.index, X.cur[ci], kD);
}

// ------------------------------------------------------------------------------------------------ whole token
void token_body(Ctx& c, Th& t) {
    Engine& E = *c.E;
    Scratch& X = *c.X;
    ThreadPool& pool = *E.pool;
    const ModelConfig& cfg = E.cfg;
    const int tid = t.tid;
    if (tid == 0) {
        c.last = __rdtsc();
        alignas(64) float emb[kD];
        const block_q8_0* row = (const block_q8_0*)(E.tok_embd + (size_t)c.token * E.tok_embd_rs);
        for (int b = 0; b < kD / 32; ++b) {
            const float d = fp16_to_f32(row[b].d);
            for (int j = 0; j < 32; ++j) emb[b * 32 + j] = d * (float)row[b].qs[j];
        }
        rmsnorm(emb, E.tok_norm, X.cur[0], kD, cfg.rms_eps);
    }
    BAR(PH_EMBED, 0);
    int ci = 0, nb = 0, prelude_nb = 0;
    for (int idx = 0; idx < kVisits; ++idx) {
        const Visit& vis = E.plan[idx];
        if (idx == cfg.loop_start) {   // end of the prelude: remember its output and blocks
            if (tid == 0) std::memcpy(X.prelude, X.cur[ci], kD * sizeof(float));
            prelude_nb = nb;
        }
        if (vis.in_core && vis.pass > 0 && vis.phys == cfg.loop_start) {   // start of an extra pass
            if (tid == 0) {
                std::memcpy(X.bank_out, X.cur[ci], kD * sizeof(float));
                std::memcpy(X.bank_b, X.blocks[prelude_nb + 1], kD * sizeof(float));
                std::memcpy(X.cur[ci], X.prelude, kD * sizeof(float));
            }
            nb = prelude_nb;
        }
        visit(c, t, vis, ci, nb);
    }
    if (!c.logits && !c.taps) return;
    {
        float* cur = X.cur[ci];
        if (tid == 0) {
            const float* st[8];
            int ns = 0;
            for (int b = nb - 1; b >= 0; --b) st[ns++] = X.blocks[b];
            st[ns++] = cur;
            float* mo[1] = {X.fin_mix};
            depth_mix(st, ns, E.out_res_query, 1, mo, kD, kMixScale);
            rmsnorm(X.fin_mix, E.out_norm, X.fin_x, kD, cfg.rms_eps);
            if (c.taps) std::memcpy(c.taps + 3 * (size_t)kD, X.fin_x, kD * sizeof(float));
            if (c.logits) quantize_q8_0(X.fin_x, X.fin_q, kD, Q8Mode::X86);
        }
        if (!c.logits) return;   // taps only: the lm head is skipped (uniform over threads)
        BAR(PH_FINAL, 0);
        const Mat& H = E.lm_head;
        dyn(c, t, H.rows, E.tune.chunk_lm, std::max(16, E.tune.chunk_lm / 8), [&](int64_t b, int64_t e) {
            gemv_q8_0((const char*)H.w + (size_t)b * H.rs, H.rs, X.fin_q, kD, c.logits + b, (int)(e - b));
            vsoftcap(c.logits + b, c.logits + b, (size_t)(e - b), cfg.final_logit_softcap);
        });
        if (c.prof) BAR(PH_LMHEAD, H.bytes());
    }
}

}  // namespace

void engine_fine_report(FILE* f, double tsc_per_us) {
    if (!fine_on() || !g_fine.a_n) return;
    const double tpu = tsc_per_us > 0 ? tsc_per_us : 1000.0;   // < 0 or 0: print kcycles
    auto us = [&](uint64_t ticks, uint64_t n) { return n ? (double)ticks / (double)n / tpu : 0.0; };
    std::fprintf(f, "fine (thread 0, us per visit): A work %.1f + barrier %.1f (thread 1 arrived %.1f us before / %.1f after thread 0) (n=%llu) | C nsplit=1: pre %.1f, task %.1f, finish %.1f, barrier %.1f (n=%llu)"
                 " | C split: pre %.1f, barrier %.1f, items %.1f, barrier %.1f (n=%llu) | F1 work %.1f + barrier %.1f\n",
                 us(g_fine.a_work, g_fine.a_n), us(g_fine.a_bar, g_fine.a_n), us(g_fine.a_t1_early, g_fine.a_n), us(g_fine.a_t1_late, g_fine.a_n),
                 (unsigned long long)g_fine.a_n, us(g_fine.c_pre, g_fine.c_n),
                 us(g_fine.c_task, g_fine.c_n), us(g_fine.c_fin, g_fine.c_n), us(g_fine.c_bar, g_fine.c_n), (unsigned long long)g_fine.c_n,
                 us(g_fine.s_pre, g_fine.s_n), us(g_fine.s_bar1, g_fine.s_n), us(g_fine.s_items, g_fine.s_n), us(g_fine.s_bar2, g_fine.s_n),
                 (unsigned long long)g_fine.s_n, us(g_fine.f_work, g_fine.f_n), us(g_fine.f_bar, g_fine.f_n));
    g_fine = FineProf();
}

void Engine::run_token(SlotState& S, int32_t token, float* logits) {
    const auto t0 = std::chrono::steady_clock::now();
    // the calling thread acts as tid 0: pin it once per thread
    if (opt.n_threads > 0 && std::this_thread::get_id() != pinned_thread_) {
        const std::vector<int> cpus = opt.cpus.empty() ? engine_default_cpus(opt.n_threads) : opt.cpus;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[0], &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        pinned_thread_ = std::this_thread::get_id();
    }
    Ctx c;
    c.E = this;
    c.X = scr_;
    c.S = &S;
    c.ctr = ctr_;
    c.pos = S.pos;
    c.token = token;
    c.logits = logits;
    c.taps = tap_sink ? scr_->taps : nullptr;
    c.prof = prof_on;
    c.last = 0;
    {   // tau = 1 + alpha * log(max(1, (pos+1)/floor)) with the float operation order of the ggml graph
        float tt = (float)S.pos + 1.0f;
        tt = tt * (1.0f / (float)cfg.log_scale_floor);
        tt = std::min(INFINITY, std::max(1.0f, tt));
        c.tau = std::log(tt) * cfg.log_scale_alpha + 1.0f;
    }
    for (int v = 0; v < kVisits; ++v)
        if (!S.kv[v].ring) S.kv[v].n_blocks = std::max<uint32_t>(S.kv[v].n_blocks, (uint32_t)(S.pos / kKvBlock + 1));   // grow-only: junk rows of a rolled-back verify stay tracked
    const int used = scr_->used_phases > 0 ? std::min(scr_->used_phases, kMaxPhases) : kMaxPhases;
    for (int i = 0; i < used; ++i) ctr_[i].v.store(0, std::memory_order_relaxed);
    pool->run([&](int tid, int nth) {
        Th t{tid, nth, 0};
        token_body(c, t);
        if (tid == 0) scr_->used_phases = t.phase;
    });
    if (prof_on) prof.tokens++;
    last_total_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

int Engine::decode(int slot, int32_t token, float* logits) {
    if (slot < 0 || slot >= (int)slots.size() || !slots[slot].used) return -1;
    SlotState& S = slots[slot];
    if (S.pos >= opt.max_ctx) return -2;
    if (token < 0 || token >= cfg.n_vocab_padded) return -3;
    run_token(S, token, logits);
    if (tap_sink) tap_sink(slot, S.pos, 1, scr_->taps);
    S.pos++;
    if (S.hwm < S.pos) S.hwm = S.pos;
    return 0;
}

const float* Engine::last_taps() const { return scr_->taps; }

int Engine::forward(int slot, const int32_t* tokens, int n, float* logits_last) {
    if (n < 1) return -4;
    if (slot < 0 || slot >= (int)slots.size() || !slots[slot].used) return -1;
    if (slots[slot].pos + n > opt.max_ctx) return -2;
#ifdef INK_HAVE_PREFILL
    {   // batched executor (agent P): bit-identical to the loop below up to 767 visible keys, ~3.5x the token rate on prompts.
        // INK_PREFILL_MIN = smallest n routed to it (default 3; 0 = never, i.e. always the decode loop).
        static const int pf_min = [] {
            const char* s = std::getenv("INK_PREFILL_MIN");
            return s ? std::atoi(s) : 3;
        }();
        if (pf_min > 0 && n >= pf_min) {
            const auto t0 = std::chrono::steady_clock::now();
            const int rc = engine_prefill(*this, slot, tokens, n, logits_last);
            last_forward_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            return rc;
        }
    }
#endif
    double us = 0;
    for (int i = 0; i < n; ++i) {
        const int rc = decode(slot, tokens[i], i + 1 == n ? logits_last : nullptr);
        if (rc) return rc;
        us += last_total_us;
    }
    last_forward_us = us;
    if (n >= 2) slot_checkpoint(slot);   // prompt boundary (the prefill path checkpoints every chunk); single decode steps never checkpoint
    return 0;
}

int Engine::prefill(int slot, const int32_t* tokens, int n, float* logits_last) {
    if (n < 1) return -4;
    if (slot < 0 || slot >= (int)slots.size() || !slots[slot].used) return -1;
    if (slots[slot].pos + n > opt.max_ctx) return -2;
#ifdef INK_HAVE_PREFILL
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = engine_prefill(*this, slot, tokens, n, logits_last);
    last_forward_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return rc;
#else
    return forward(slot, tokens, n, logits_last);
#endif
}

}  // namespace ink

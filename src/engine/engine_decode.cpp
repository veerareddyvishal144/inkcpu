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
template <class F>
inline void dyn(Ctx& c, Th& t, int64_t total, int64_t chunk, int64_t fine, F&& f) {
    std::atomic<int64_t>& a = c.ctr[t.phase++].v;
    if (fine >= chunk || total <= 4 * chunk) {
        for (;;) {
            const int64_t b = a.fetch_add(1, std::memory_order_relaxed) * chunk;
            if (b >= total) break;
            f(b, std::min<int64_t>(b + chunk, total));
        }
        return;
    }
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
template <class F>
inline void dyn(Ctx& c, Th& t, int64_t total, int64_t chunk, F&& f) {
    dyn(c, t, total, chunk, chunk, std::forward<F>(f));
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
    dyn(c, t, (int64_t)nj * kCtrlRank, 4, [&](int64_t b, int64_t e) { gemv_multi_lp(ty, dm, nj, 2 * kD, b, e); });
    BAR(PH_CTRL, bytes);
    GemvMat gm[8];
    for (int j = 0; j < nj; ++j) {
        const Mat& g = L.ctrl_gain[cidx[j]];
        const Mat& f = L.ctrl_forget[cidx[j]];
        gm[2 * j] = GemvMat{g.w, g.rs, X.ctrl_g[j], X.ctrl_gain[j], g.rows};
        gm[2 * j + 1] = GemvMat{f.w, f.rs, X.ctrl_g[j], X.ctrl_forget[j], f.rows};
    }
    dyn(c, t, (int64_t)nj * 2 * kD, E.tune.chunk_ctrl, [&](int64_t b, int64_t e) { gemv_multi_lp(ty, gm, 2 * nj, kCtrlRank, b, e); });
    BAR(PH_CTRL, 0);
    dyn(c, t, kD, 64, [&](int64_t b, int64_t e) {
        for (int j = 0; j < nj; ++j)
            ctrl_apply(cur_in[j] + b, msg[j] + b, X.ctrl_gain[j] + b, X.ctrl_forget[j] + b, out[j] + b, (size_t)(e - b));
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
    }
    BAR(PH_MIX, 0);
    if (msgs) {
        const int cidx[3] = {0, 1, 2};
        const float* cin[3] = {in_raw[0], in_raw[1], in_raw[2]};
        const float* mm[3] = {X.msgs[0], X.msgs[1], X.msgs[2]};
        float* outs[3] = {X.cin[0], X.cin[1], X.cin[2]};
        controllers(c, t, L, 3, cidx, cin, mm, outs);
        for (int j = 0; j < 3; ++j) in_raw[j] = X.cin[j];
        if (tid == 0) norms_all();
        BAR(PH_MIX, 0);
    }

    // ---------------------------------------------------------------- B: projections q|k|v|gate|rel (+ sink, dyn taps)
    {
        GemvMat qm[5] = {{L.wq.w, L.wq.rs, xq[0], X.q, L.wq.rows},       {L.wk.w, L.wk.rs, xq[1], X.kp, L.wk.rows},
                         {L.wv.w, L.wv.rs, xq[2], X.vp, L.wv.rows},       {L.wgate.w, L.wgate.rs, xq[0], X.gate, L.wgate.rows},
                         {L.wrel.w, L.wrel.rs, xq[0], X.rel, L.wrel.rows}};
        GemvMat sm[3] = {{L.sink.w, L.sink.rs, xb[0], X.sink, L.sink.rows},
                         {L.kproj.w, L.kproj.rs, xb[1], X.kdyn, L.kproj.rows},
                         {L.vproj.w, L.vproj.rs, xb[2], X.vdyn, L.vproj.rows}};
        const int nsm = swa ? 1 : 3;
        int64_t nqr = 0, nsr = 0;
        for (int i = 0; i < 5; ++i) nqr += qm[i].nrows;
        for (int i = 0; i < nsm; ++i) nsr += sm[i].nrows;
        dyn(c, t, nqr + nsr, E.tune.chunk_proj, std::max(4, E.tune.chunk_proj / 4), [&](int64_t b, int64_t e) {
            if (b < nqr) gemv_multi_q8_0(qm, 5, kD, b, std::min(e, nqr));
            if (e > nqr) gemv_multi_bf16(sm, nsm, kD, std::max(b, nqr) - nqr, e - nqr);
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
        const int nsplit = (int)std::min<int64_t>(kMaxSplit, std::max<int64_t>(1, n_keys / std::max(1, E.tune.split_keys)));
        auto group_pre = [&](int g) {
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
            for (int g = tid; g < kNKvHeads; g += nth) {
                group_pre(g);
                attn_decode_task(kv, L.rel_table, aq, g, 0, 1, &X.parts[g]);
                group_fin(g);
            }
            BAR(PH_ATTN, kvb);
        } else {
            for (int g = tid; g < kNKvHeads; g += nth) group_pre(g);
            BAR(PH_ATTN, 0);
            dyn(c, t, (int64_t)kNKvHeads * nsplit, 1, [&](int64_t b, int64_t) {
                const int g = (int)(b / nsplit), s = (int)(b % nsplit);
                attn_decode_task(kv, L.rel_table, aq, g, s, nsplit, &X.parts[g * nsplit + s]);
            });
            BAR(PH_ATTN, kvb);
            for (int g = tid; g < kNKvHeads; g += nth) group_fin(g);
            BAR(PH_ATTN, 0);
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
        const int64_t chunk = std::max<int64_t>(16, (E.tune.chunk_wo / 16) * 16);
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
    }
    if (block_start) ++nb;
    BAR(PH_FFNMIX, 0);
    if (msgs) {
        const int cidx[1] = {3};
        const float* cin[1] = {X.fmix};
        const float* mm[1] = {X.msgs[3]};
        float* outs[1] = {X.cin[3]};
        controllers(c, t, L, 1, cidx, cin, mm, outs);
        if (tid == 0) ffn_norm_quant(X.cin[3]);
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
            gemv_nvfp4(wg, L.gu_rs, X.xq, kD, g, 32);
            gemv_nvfp4(wu, L.gu_rs, X.xq, kD, up, 32);
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
                        gemv_nvfp4(wd, L.dn_rs, X.actq[i], kFF, tmp[i], nr);
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

}  // namespace ink

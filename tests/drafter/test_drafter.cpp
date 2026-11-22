// Tests of the DSpark drafter (agent S1). `test_drafter [unit|model|all] [--threads N] [--target gguf] [--drafter gguf] [--pbig P]`
//   unit : synthetic data, no model files: RoPE, attention kernel (cache blocks + splits + block keys) vs double reference, fold kernel vs double reference
//          (odd row counts, Q8_0 and bf16), content hash independent of the thread count, sampling selection statistics (20000 draws).
//   model: real drafter GGUF + real lm_head/embedding of the target GGUF (light target, no 17 GB engine): fused ctx K/V, block pass layer by layer, chain
//          (fold vs unfolded reference, teacher forced), lazy-commit equivalence, truncation property, sampled mode (q rows sum to 1, determinism).
// The reference (ref_drafter.cpp) is scalar, written from docs/DSPARK.md; honours INK_CPUS.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "light_target.h"
#include "ref_drafter.h"

using namespace ink;
using namespace ink::dr;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            ++g_fail;                                                      \
            std::printf("  FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond);  \
            std::printf(__VA_ARGS__);                                      \
            std::printf("\n");                                             \
        }                                                                  \
    } while (0)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 12345) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double u() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    float normal() {
        const double a = std::max(u(), 1e-12), b = u();
        return (float)(std::sqrt(-2 * std::log(a)) * std::cos(6.283185307179586 * b));
    }
};

static double rel_l2(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { num += (double)(a[i] - b[i]) * (a[i] - b[i]); den += (double)b[i] * b[i]; }
    return std::sqrt(num / std::max(den, 1e-30));
}
static double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}
static double max_abs(const float* a, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(a[i]));
    return m;
}
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ------------------------------------------------------------------------------------------------------------------------------ unit tests
static void test_rope() {
    std::printf("[unit] rope\n");
    Rng r(1);
    for (int64_t pos : {0, 1, 17, 1000, 30000, 250000}) {
        float x[128], y[128], cs[128];
        for (auto& v : x) v = r.normal();
        std::memcpy(y, x, sizeof(x));
        rope_table(pos, cs);
        rope_head(y, cs);
        double me = 0;
        for (int i = 0; i < 64; ++i) {
            const double a = (double)pos * std::pow(1.0e6, -(double)i / 64.0), c = std::cos(a), s = std::sin(a);
            me = std::max(me, std::fabs(x[i] * c - x[i + 64] * s - y[i]));
            me = std::max(me, std::fabs(x[i + 64] * c + x[i] * s - y[i + 64]));
        }
        CHECK(me < 2e-6, "pos %lld err %g", (long long)pos, me);
    }
}

static void test_attn() {
    std::printf("[unit] attention kernel vs double reference\n");
    Rng r(2);
    const int maxpos = 2600;
    for (int ring = 0; ring < 2; ++ring) {
        const uint32_t nb = ring ? 16 : 45;
        const size_t bytes = (size_t)nb * kKvBlockElems * 2;
        KvLayer L;
        L.k = (uint16_t*)std::calloc(bytes, 1);
        L.v = (uint16_t*)std::calloc(bytes, 1);
        L.n_blocks = nb;
        L.ring = ring;
        std::vector<float> K((size_t)maxpos * KVW), V((size_t)maxpos * KVW);
        int appended = 0;
        auto append_to = [&](int n) {   // positions appended so far = the committed context of a query anchored at p = n
            for (; appended < n; ++appended) {
                const int t = appended;
                std::vector<float> k(KVW), v(KVW);
                for (int i = 0; i < KVW; ++i) { k[i] = r.normal() * 1.5f; v[i] = r.normal() * 2.0f; }
                kv_append(L, t, k.data(), v.data());
                for (int i = 0; i < KVW; ++i) { K[(size_t)t * KVW + i] = fp16_to_f32(f32_to_fp16(k[i])); V[(size_t)t * KVW + i] = fp16_to_f32(f32_to_fp16(v[i])); }
            }
        };
        for (int p : {1, 15, 64, 65, 333, 1023, 1024, 1100, 2599}) {
            if (!ring && p > 64 * (int)nb) continue;
            append_to(p);
            for (int nq : {3, 9, 24}) {
                const int64_t nvalid = ring ? std::min(p, 1024) : p;
                const int64_t lo = ring ? std::max(0, p - 1024) : 0;   // visible positions lo..p-1
                for (int nsp : {1, 2, 3}) {
                    const int g = (int)(r.next() % 8);
                    std::vector<float> q((size_t)nq * HD), kb(8 * HD), vb(8 * HD);
                    for (auto& x : q) x = r.normal() * 0.3f;
                    for (auto& x : kb) x = r.normal();
                    for (auto& x : vb) x = r.normal();
                    const int nblk = (int)((nvalid + 63) / 64);
                    std::vector<AttnPartial2> parts(nsp);
                    for (int sp = 0; sp < nsp; ++sp) attn_ctx_part(L, g, nblk * sp / nsp, nblk * (sp + 1) / nsp, nvalid, q.data(), nq, &parts[sp]);
                    double worst = 0;
                    for (int i = 0; i < nq; ++i) {
                        float out[128];
                        attn_finish_query(parts.data(), nsp, i, q.data() + (size_t)i * HD, kb.data(), vb.data(), HD, 8, out);
                        // reference
                        std::vector<double> sc;
                        for (int64_t t = lo; t < p; ++t) {
                            double d = 0;
                            for (int e = 0; e < HD; ++e) d += (double)q[(size_t)i * HD + e] * K[(size_t)t * KVW + g * HD + e];
                            sc.push_back(d);
                        }
                        for (int t = 0; t < 8; ++t) {
                            double d = 0;
                            for (int e = 0; e < HD; ++e) d += (double)q[(size_t)i * HD + e] * kb[(size_t)t * HD + e];
                            sc.push_back(d);
                        }
                        const double mx = *std::max_element(sc.begin(), sc.end());
                        double Z = 0;
                        for (auto& x : sc) { x = std::exp(x - mx); Z += x; }
                        for (int e = 0; e < HD; ++e) {
                            double a = 0;
                            for (int64_t t = lo; t < p; ++t) a += sc[(size_t)(t - lo)] * V[(size_t)t * KVW + g * HD + e];
                            for (int t = 0; t < 8; ++t) a += sc[(size_t)(p - lo) + t] * vb[(size_t)t * HD + e];
                            worst = std::max(worst, std::fabs(a / Z - out[e]));
                        }
                    }
                    CHECK(worst < 3e-5, "ring %d nq %d p %d nsplit %d err %g", ring, nq, p, nsp, worst);
                }
            }
        }
        std::free(L.k);
        std::free(L.v);
    }
}

static void test_fold_small(ThreadPool& pool) {
    std::printf("[unit] fold kernel vs double reference, hash\n");
    Rng r(3);
    for (int nrows : {1003, 48, 96, 7}) {
        const size_t rs = row_bytes_q8_0(kD);
        std::vector<char> lm((size_t)nrows * rs);
        std::vector<float> lmf((size_t)nrows * kD);
        for (int v = 0; v < nrows; ++v) {
            std::vector<float> row(kD);
            for (auto& x : row) x = r.normal() * 0.05f;
            quantize_row_q8_0_ref(row.data(), (block_q8_0*)(lm.data() + (size_t)v * rs), kD);
            const block_q8_0* b = (const block_q8_0*)(lm.data() + (size_t)v * rs);
            for (int i = 0; i < kD; ++i) lmf[(size_t)v * kD + i] = fp16_to_f32(b[i / 32].d) * (float)b[i / 32].qs[i % 32];
        }
        std::vector<uint16_t> hw((size_t)kD * RANK);
        std::vector<float> hb(kD);
        for (auto& x : hw) x = f32_to_bf16(r.normal() * 0.04f);
        for (auto& x : hb) x = r.normal() * 0.1f;
        Mat m;
        m.w = lm.data(); m.type = T_Q8_0; m.rows = nrows; m.k = kD; m.rs = rs;
        for (int q8 = 1; q8 >= 0; --q8) {
            const size_t mrs = q8 ? row_bytes_q8_0(RANK) : row_bytes_bf16(RANK);
            std::vector<char> M((size_t)nrows * mrs);
            std::vector<float> m0(nrows);
            fold_compute(pool, m, nrows, hw.data(), hb.data(), q8, M.data(), mrs, m0.data());
            double worst = 0, worst0 = 0;
            for (int v = 0; v < nrows; ++v) {
                std::vector<double> ref(RANK, 0.0);
                double r0 = 0;
                for (int k = 0; k < kD; ++k) {
                    const double a = lmf[(size_t)v * kD + k];
                    r0 += a * hb[k];
                    for (int i = 0; i < RANK; ++i) ref[i] += a * bf16_to_f32(hw[(size_t)k * RANK + i]);
                }
                worst0 = std::max(worst0, std::fabs(r0 - m0[v]) / (std::fabs(r0) + 1e-3));
                for (int b = 0; b < RANK / 32; ++b) {
                    double amax = 0;
                    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(ref[b * 32 + i]));
                    for (int i = 0; i < 32; ++i) {
                        float got;
                        if (q8) {
                            const block_q8_0* bl = (const block_q8_0*)(M.data() + (size_t)v * mrs) + b;
                            got = fp16_to_f32(bl->d) * (float)bl->qs[i];
                            worst = std::max(worst, std::fabs(got - ref[b * 32 + i]) / (amax / 127.0 * 0.5 + 1e-6));   // in units of half a quantum
                        } else {
                            got = bf16_to_f32(((const uint16_t*)(M.data() + (size_t)v * mrs))[b * 32 + i]);
                            worst = std::max(worst, std::fabs(got - ref[b * 32 + i]) / (std::fabs(ref[b * 32 + i]) * (1.0 / 256) + 1e-5));
                        }
                    }
                }
            }
            CHECK(worst < 1.15, "rows %d q8 %d worst error %g (units: half quantum / bf16 half ulp)", nrows, q8, worst);
            CHECK(worst0 < 1e-4, "rows %d m0 rel err %g", nrows, worst0);
        }
    }
    {   // hash independent of the thread count and sensitive to a flipped bit
        std::vector<uint8_t> buf(3000001);
        Rng rr(9);
        for (auto& b : buf) b = (uint8_t)rr.next();
        const uint64_t h1 = hash_bytes_parallel(pool, buf.data(), buf.size());
        ThreadPool p1(1, {pool.size() ? engine_default_cpus(1)[0] : 2});
        const uint64_t h2 = hash_bytes_parallel(p1, buf.data(), buf.size());
        CHECK(h1 == h2, "hash differs between %d and 1 threads", pool.size());
        buf[123456] ^= 1;
        CHECK(hash_bytes_parallel(pool, buf.data(), buf.size()) != h1, "hash insensitive");
    }
}

static void test_sample_select() {
    std::printf("[unit] sampling selection (20000 draws of a toy distribution)\n");
    const int nv = 3000, nch = (nv + CH_ROWS - 1) / CH_ROWS;
    Rng r(4);
    std::vector<float> logit(nv), e(nv);
    std::vector<ChunkRes> cres(nch);
    for (auto& x : logit) x = r.normal() * 2.0f;
    logit[17] += 4.0f; logit[2999] += 3.0f; logit[512] += 3.5f;   // heavy tokens incl. chunk edges
    for (int ch = 0; ch < nch; ++ch) {
        const int r0 = ch * CH_ROWS, rows = std::min(CH_ROWS, nv - r0);
        float mx = logit[r0];
        for (int k = 0; k < rows; ++k) mx = std::max(mx, logit[r0 + k]);
        float sum = 0;
        for (int k = 0; k < rows; ++k) { e[r0 + k] = std::exp(logit[r0 + k] - mx); sum += e[r0 + k]; }
        cres[ch] = {mx, r0, sum, 0};
    }
    double M, Z;
    sample_select(cres.data(), nch, e.data(), nv, 1, 1, &M, &Z);
    std::vector<double> q(nv);
    for (int ch = 0; ch < nch; ++ch)
        for (int k = 0; k < std::min(CH_ROWS, nv - ch * CH_ROWS); ++k) q[ch * CH_ROWS + k] = std::exp((double)cres[ch].val - M) * e[ch * CH_ROWS + k] / Z;
    double qs = 0;
    for (double x : q) qs += x;
    CHECK(std::fabs(qs - 1.0) < 1e-5, "q sums to %g", qs);
    const int N = 20000;
    std::vector<int> cnt(nv, 0);
    for (int i = 0; i < N; ++i) {
        double M2, Z2;
        const int d = sample_select(cres.data(), nch, e.data(), nv, 1000 + (uint64_t)i * 7919, 1, &M2, &Z2);
        CHECK(d >= 0 && d < nv && e[d] > 0, "bad draw %d", d);
        cnt[d]++;
    }
    double worst = 0;
    int checked = 0;
    for (int v = 0; v < nv; ++v)
        if (q[v] > 0.004) {
            const double sd = std::sqrt(q[v] * (1 - q[v]) / N);
            worst = std::max(worst, std::fabs((double)cnt[v] / N - q[v]) / sd);
            ++checked;
        }
    std::printf("    %d tokens with q > 0.004, worst deviation %.2f sigma\n", checked, worst);
    CHECK(worst < 4.5, "empirical frequency off by %g sigma", worst);
    double tailq = 0, tailc = 0;
    for (int v = 0; v < nv; ++v)
        if (q[v] <= 0.004) { tailq += q[v]; tailc += cnt[v]; }
    const double sdt = std::sqrt(tailq * (1 - tailq) / N);
    CHECK(std::fabs(tailc / N - tailq) < 4.5 * sdt + 1e-9, "tail mass %g vs %g", tailc / N, tailq);
    // determinism and seed sensitivity
    double m1, z1;
    const int a = sample_select(cres.data(), nch, e.data(), nv, 77, 3, &m1, &z1), b = sample_select(cres.data(), nch, e.data(), nv, 77, 3, &m1, &z1);
    CHECK(a == b, "not deterministic");
}

// ------------------------------------------------------------------------------------------------------------------------------ model tests
struct Env {
    LightTarget lt;
    std::unique_ptr<DrafterImpl> D;
};

static void gen_taps(std::vector<float>& taps, int rows, uint64_t seed) {
    Rng r(seed);
    taps.resize((size_t)rows * kTapFloats);
    const float sc[4] = {3.0f, 6.0f, 9.0f, 1.0f};
    for (int t = 0; t < rows; ++t)
        for (int k = 0; k < 4; ++k)
            for (int i = 0; i < kD; ++i) taps[(size_t)t * kTapFloats + (size_t)k * kD + i] = r.normal() * sc[k] + 0.1f * (float)(i % 7);
}

static void read_ctx(StateData& S, int l, int64_t pos, float* K, float* V) {
    const KvLayer& L = S.kv[l];
    const size_t slot = L.ring ? (size_t)(pos % ((int64_t)L.n_blocks * 64)) : (size_t)pos;
    const size_t b = slot / 64, o = slot % 64;
    for (int g = 0; g < NKV; ++g)
        for (int d = 0; d < HD; ++d) {
            K[g * HD + d] = fp16_to_f32(L.k[b * kKvBlockElems + (size_t)g * 8192 + (size_t)d * 64 + o]);
            V[g * HD + d] = fp16_to_f32(L.v[b * kKvBlockElems + (size_t)g * 8192 + o * HD + d]);
        }
}

static int argmax_row(const float* x, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return b;
}

// scenario: commit P random rows, compare ctx K/V, block pass, chain
static void scenario(Env& E, int P, int nn, uint64_t seed, bool chunked_commit) {
    DrafterImpl& D = *E.D;
    std::printf("[model] scenario P=%d n=%d seed=%llu (%s commits)\n", P, nn, (unsigned long long)seed, chunked_commit ? "small" : "one");
    const int st = D.state_new();
    CHECK(st >= 0, "state_new");
    std::vector<float> taps;
    gen_taps(taps, P, seed);
    const double t0 = now_s();
    if (chunked_commit) {
        int off = 0;
        Rng r(seed);
        while (off < P) {
            const int c = std::min<int>(P - off, 1 + (int)(r.next() % 9));
            CHECK(D.commit(st, taps.data() + (size_t)off * kTapFloats, c) == 0, "commit");
            off += c;
        }
    } else {
        CHECK(D.commit(st, taps.data(), P) == 0, "commit");
    }
    CHECK(D.state_pos(st) == P, "pos");
    CHECK(D.flush(st) == 0, "flush");
    CHECK(D.state_pending(st) == 0, "pending after flush");
    std::printf("    commit+flush %.3f s\n", now_s() - t0);
    StateData& S = D.state(st);
    // f_rnn
    std::vector<float> frnn;
    rd::fuse(D, taps.data() + (size_t)(P - 1) * kTapFloats, 3, frnn);
    CHECK(rel_l2(S.f_rnn, frnn.data(), kD) < 1e-5, "f_rnn rel err %g", rel_l2(S.f_rnn, frnn.data(), kD));
    // ctx K/V of all positions (reference) vs cache (window only for the SWA layers)
    std::vector<float> ck[NL], cv[NL];
    for (int l = 0; l < NL; ++l) { ck[l].assign((size_t)P * KVW, 0.0f); cv[l].assign((size_t)P * KVW, 0.0f); }
    double worstK = 0, worstV = 0;
    const double tr0 = now_s();
    for (int t = 0; t < P; ++t) {
        for (int l = 0; l < NL; ++l) {
            if (l != 1 && t < P - 1024) continue;   // the cache keeps only the window; the reference never needs these either
            std::vector<float> f, K, V, kc(KVW), vc(KVW);
            rd::fuse(D, taps.data() + (size_t)t * kTapFloats, l, f);
            rd::ctx_kv(D, l, f.data(), t, K, V);
            std::memcpy(ck[l].data() + (size_t)t * KVW, K.data(), KVW * 4);
            std::memcpy(cv[l].data() + (size_t)t * KVW, V.data(), KVW * 4);
            read_ctx(S, l, t, kc.data(), vc.data());
            worstK = std::max(worstK, rel_l2(kc.data(), K.data(), KVW));
            worstV = std::max(worstV, rel_l2(vc.data(), V.data(), KVW));
        }
    }
    std::printf("    ctx K/V vs reference over %d positions: worst rel L2 K %.2e V %.2e (reference %.1f s)\n", P, worstK, worstV, now_s() - tr0);
    CHECK(worstK < 2e-3 && worstV < 2e-3, "ctx K/V mismatch K %g V %g", worstK, worstV);

    // block pass + chain, greedy; then teacher forced
    const int32_t anchor = (int32_t)(Rng(seed + 5).next() % 150000);
    DraftRequest rq;
    rq.anchor = anchor; rq.n = nn;
    int32_t tok[8] = {}, tok2[8] = {};
    D.dbg.on = true;
    double td = now_s();
    CHECK(D.draft(st, rq, tok, nullptr) == 0, "draft");
    td = now_s() - td;
    rd::BlockOut bo;
    const double tr1 = now_s();
    rd::block_pass(D, ck, cv, P, anchor, &bo);
    std::printf("    block pass reference %.1f s, drafter draft(n=%d) %.1f ms\n", now_s() - tr1, nn, td * 1e3);
    for (int l = 0; l < NL; ++l) {
        double worst = 0;
        const int s_lo = (l == NL - 1) ? 1 : 0, s_hi = (l == NL - 1) ? nn : 7;
        for (int s = s_lo; s <= s_hi; ++s) worst = std::max(worst, rel_l2(D.dbg.x[l].data() + (size_t)s * kD, bo.X[l].data() + (size_t)s * kD, kD));
        std::printf("    layer %d output vs reference: worst per-slot rel L2 %.2e\n", l, worst);
        CHECK(worst < 5e-2, "layer %d mismatch %g", l, worst);   // chaos floor of Q8_0 activations: a 1e-6 float difference flips single quanta and saturates near 1e-2
    }
    double worst_h = 0;
    for (int s = 1; s <= nn; ++s) worst_h = std::max(worst_h, rel_l2(D.dbg.hid.data() + (size_t)s * kD, bo.hid.data() + (size_t)s * kD, kD));
    std::printf("    hid vs reference: worst rel L2 %.2e\n", worst_h);
    CHECK(worst_h < 5e-2, "hid mismatch %g", worst_h);

    // chain: unfolded reference on the drafter's own hid (isolates the chain/fold from the block pass), free-running
    rd::ChainOut co;
    const double tr2 = now_s();
    rd::chain(D, frnn.data(), D.dbg.hid.data(), anchor, nn, nullptr, &co);
    std::printf("    chain reference (unfolded, 2 lm_head products per step) %.1f s\n", now_s() - tr2);
    int agree = 0;
    for (int j = 0; j < nn; ++j) agree += (tok[j] == co.tokens[j]);
    std::printf("    free-running tokens: drafter vs reference agree on %d/%d positions (first mismatch cuts the comparison)\n", agree, nn);
    // teacher forced logits comparison
    D.dbg.force_on = true;
    for (int j = 0; j < nn; ++j) D.dbg.force[j] = co.tokens[j];
    CHECK(D.draft(st, rq, tok2, nullptr) == 0, "draft forced");
    D.dbg.force_on = false;
    for (int j = 0; j < nn; ++j) {
        const float* got = D.dbg.logits.data() + (size_t)j * D.tgt.n_vocab;
        const float* ref_l = co.logits[j].data();
        const double md = max_abs_diff(got, ref_l, D.tgt.n_vocab);
        const int a = argmax_row(got, D.tgt.n_vocab), b = argmax_row(ref_l, D.tgt.n_vocab);
        // top-2 gap of the reference (is a mismatch a near tie?)
        float best = -1e30f, second = -1e30f;
        for (int v = 0; v < D.tgt.n_vocab; ++v) { if (ref_l[v] > best) { second = best; best = ref_l[v]; } else if (ref_l[v] > second) second = ref_l[v]; }
        std::printf("    step %d: max|dlogit| %.4f  rel L2(lat) %.2e  top1 drafter %d reference %d (ref top1-top2 gap %.4f)%s\n", j + 1, md,
                    rel_l2(D.dbg.lat.data() + (size_t)j * RANK, co.lat[j].data(), RANK), a, b, best - second, a == b ? "" : "  <-- differs");
        CHECK(md < 0.25, "step %d logits differ by %g", j + 1, md);
        CHECK(a == b || (best - second) < 0.1, "step %d argmax differs with a clear margin", j + 1);
    }
    D.dbg.on = false;
    D.state_free(st);
}

static void test_semantics(Env& E) {
    std::printf("[model] semantics: lazy commit equivalence, truncation, determinism, sampled mode\n");
    DrafterImpl& D = *E.D;
    const int P = 150;
    std::vector<float> taps;
    gen_taps(taps, P, 777);
    const int s1 = D.state_new(), s2 = D.state_new();
    CHECK(s1 >= 0 && s2 >= 0, "two states needed (opt.max_states)");
    CHECK(D.commit(s1, taps.data(), P) == 0, "commit one");
    for (int t = 0; t < P; ++t) CHECK(D.commit(s2, taps.data() + (size_t)t * kTapFloats, 1) == 0, "commit single");
    DraftRequest rq;
    rq.anchor = 4242; rq.n = 7;
    int32_t a[8], b[8], c[8];
    D.dbg.on = true;
    CHECK(D.draft(s1, rq, a, nullptr) == 0, "draft s1");
    const std::vector<float> lg1 = D.dbg.logits;
    CHECK(D.draft(s2, rq, b, nullptr) == 0, "draft s2");
    const std::vector<float> lg2 = D.dbg.logits;
    CHECK(std::memcmp(a, b, 7 * 4) == 0, "one big commit vs 150 single commits give different tokens");
    CHECK(lg1 == lg2, "logits differ between commit schedules (must be bit-identical)");
    CHECK(D.draft(s1, rq, c, nullptr) == 0, "draft again");
    CHECK(std::memcmp(a, c, 7 * 4) == 0 && lg1 == D.dbg.logits, "draft changed the committed state (second draft differs)");
    DraftRequest r3 = rq;
    r3.n = 3;
    CHECK(D.draft(s1, r3, c, nullptr) == 0, "draft n=3");
    CHECK(std::memcmp(a, c, 3 * 4) == 0, "n=3 is not the prefix of n=7: %d %d %d vs %d %d %d", c[0], c[1], c[2], a[0], a[1], a[2]);
    D.dbg.on = false;
    // sampled mode
    const int nv = D.tgt.n_vocab;
    std::vector<float> q((size_t)3 * nv), q2((size_t)3 * nv);
    DraftRequest rs = rq;
    rs.n = 3; rs.sample = true; rs.temperature = 1.0f; rs.seed = 99;
    int32_t t1[8], t2[8], t3[8];
    CHECK(D.draft(s1, rs, t1, q.data()) == 0, "sampled draft");
    CHECK(D.draft(s1, rs, t2, q2.data()) == 0, "sampled draft again");
    CHECK(std::memcmp(t1, t2, 12) == 0 && q == q2, "sampled mode not deterministic for a fixed seed");
    for (int j = 0; j < 3; ++j) {
        double sum = 0;
        float mn = 1;
        for (int v = 0; v < nv; ++v) { sum += q[(size_t)j * nv + v]; mn = std::min(mn, q[(size_t)j * nv + v]); }
        CHECK(std::fabs(sum - 1.0) < 2e-4 && mn >= 0.0f, "row %d sum %.7f min %g", j, sum, mn);
        CHECK(t1[j] >= 0 && t1[j] < nv && q[(size_t)j * nv + t1[j]] > 0.0f, "drawn token has zero probability");
    }
    // q row 0 = softmax(logits step 1 / T): compare with the greedy-mode logits (same prev = anchor)
    {
        D.dbg.on = true;
        CHECK(D.draft(s1, rq, c, nullptr) == 0, "greedy for logits");
        D.dbg.on = false;
        const float* lg = D.dbg.logits.data();
        double mx = lg[0];
        for (int v = 1; v < nv; ++v) mx = std::max<double>(mx, lg[v]);
        double Z = 0;
        for (int v = 0; v < nv; ++v) Z += std::exp(lg[v] - mx);
        double worst = 0;
        for (int v = 0; v < nv; ++v) worst = std::max(worst, std::fabs(std::exp(lg[v] - mx) / Z - q[v]));
        CHECK(worst < 1e-5, "q row 0 vs softmax(logits): max abs diff %g", worst);
        std::printf("    q row 0 vs softmax(greedy logits): max abs diff %.2e; max q %.4f\n", worst, std::exp(0.0) / Z);
    }
    rs.seed = 100;
    CHECK(D.draft(s1, rs, t3, q2.data()) == 0, "seed 100");
    int diff = 0;
    for (int seed = 0; seed < 8; ++seed) {
        rs.seed = 5000 + seed;
        D.draft(s1, rs, t3, q2.data());
        diff += (std::memcmp(t1, t3, 12) != 0);
    }
    CHECK(diff > 0, "different seeds never changed the draw");
    // temperature scaling: a very low temperature picks the argmax chain
    DraftRequest cold = rq;
    cold.n = 1; cold.sample = true; cold.temperature = 0.02f; cold.seed = 3;
    int32_t tc[8];
    D.draft(s1, cold, tc, q.data());
    CHECK(tc[0] == a[0], "T=0.02 draw %d vs greedy %d", tc[0], a[0]);
    // error paths
    CHECK(D.draft(s1, rq, c, nullptr) == 0, "valid");
    DraftRequest bad = rq;
    bad.n = 8;
    CHECK(D.draft(s1, bad, c, nullptr) < 0, "n=8 must fail");
    bad = rq; bad.anchor = -1;
    CHECK(D.draft(s1, bad, c, nullptr) < 0, "anchor -1 must fail");
    const int s3 = D.state_new();
    if (s3 >= 0) {
        CHECK(D.draft(s3, rq, c, nullptr) == -3, "draft without context must return -3");
        D.state_free(s3);
    }
    D.state_reset(s1);
    CHECK(D.state_pos(s1) == 0, "reset");
    D.state_free(s1);
    D.state_free(s2);
}

int main(int argc, char** argv) {
    std::string mode = "all", target = "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf", drafter = "/data/inkling-mini/models/dspark-Inkling-2-Mini-Q8_0.gguf";
    int threads = 4, pbig = 1100;
    bool bf16 = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "unit" || a == "model" || a == "all") mode = a;
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--target" && i + 1 < argc) target = argv[++i];
        else if (a == "--drafter" && i + 1 < argc) drafter = argv[++i];
        else if (a == "--pbig" && i + 1 < argc) pbig = std::atoi(argv[++i]);
        else if (a == "--bf16") bf16 = true;
    }
    std::printf("test_drafter mode=%s threads=%d loadavg ", mode.c_str(), threads);
    if (FILE* f = std::fopen("/proc/loadavg", "r")) { char b[64] = {}; if (std::fgets(b, sizeof b, f)) std::printf("%s", b); std::fclose(f); }
    if (mode == "unit" || mode == "all") {
        ThreadPool pool(threads, engine_default_cpus(threads));
        test_rope();
        test_attn();
        test_fold_small(pool);
        test_sample_select();
    }
    if (mode == "model" || mode == "all") {
        Env E;
        std::string err;
        double t0 = now_s();
        if (!E.lt.init(target, threads, std::max(4096, pbig + 64), &err)) { std::printf("FAIL light target: %s\n", err.c_str()); return 2; }
        std::printf("[model] light target ready in %.1f s\n", now_s() - t0);
        DrafterOptions opt;
        opt.fold_q8 = !bf16;
        opt.max_states = 3;
        t0 = now_s();
        E.D = DrafterImpl::make(E.lt.t, drafter, opt, &err);
        if (!E.D) { std::printf("FAIL create: %s\n", err.c_str()); return 2; }
        std::printf("[model] drafter created in %.1f s (fold %s, %.1f s, cache %s, weights+fold %.0f MB)\n", now_s() - t0, E.D->fold_from_cache ? "loaded" : "computed",
                    E.D->fold_seconds, E.D->fold_path.c_str(), E.D->weight_bytes() / 1e6);
        scenario(E, 37, 7, 11, false);
        scenario(E, 90, 3, 12, true);
        scenario(E, pbig, 7, 13, false);
        test_semantics(E);
    }
    std::printf("\n%s: %d checks, %d failures\n", g_fail ? "FAILED" : "PASSED", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

#include "ref_drafter.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ink {
namespace dr {
namespace rd {

namespace {

// y = W x with the engine's activation quantisation (Q8_0, X86 rounding); W rows [rows] x K, single thread
void lin(const Mat& W, const float* x, float* y) {
    std::vector<block_q8_0> xq(W.k / 32);
    ink::ref::quantize_q8_0(x, xq.data(), (size_t)W.k, Q8Mode::X86);
    gemv_q8_0(W.w, W.rs, xq.data(), W.k, y, W.rows);
}
// same, rows split over the pool (lm_head products)
void lin_pool(ThreadPool& pool, const Mat& W, int rows, const float* x, float* y) {
    std::vector<block_q8_0> xq(W.k / 32);
    ink::ref::quantize_q8_0(x, xq.data(), (size_t)W.k, Q8Mode::X86);
    pool.run_chunks(rows, 512, [&](int, int64_t b, int64_t e) { gemv_q8_0((const char*)W.w + (size_t)b * W.rs, W.rs, xq.data(), W.k, y + b, (int)(e - b)); });
}

void rms(const float* x, const float* w, float* y, int n, double eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const double sc = 1.0 / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) y[i] = (float)((double)x[i] * sc * (w ? (double)w[i] : 1.0));
}

void rope(float* x, int64_t pos) {   // NEOX, pairs (i, i+64), theta 1e6, double angles
    for (int i = 0; i < 64; ++i) {
        const double a = (double)pos * std::pow(1.0e6, -(double)i / 64.0);
        const double c = std::cos(a), s = std::sin(a);
        const double x0 = x[i], x1 = x[i + 64];
        x[i] = (float)(x0 * c - x1 * s);
        x[i + 64] = (float)(x1 * c + x0 * s);
    }
}

inline float f16r(float v) { return fp16_to_f32(f32_to_fp16(v)); }

}  // namespace

void fuse(const DrafterImpl& D, const float* taps, int c, std::vector<float>& f) {
    // fuse weights are re-read from the GGUF-derived alpha in the drafter (softmax in double was done at load); recompute from alpha for the mix
    std::vector<float> mix(kD);
    for (int i = 0; i < kD; ++i) {
        double s = 0;
        for (int k = 0; k < 4; ++k) s += (double)D.alpha[c][k] * taps[(size_t)k * kD + i];
        mix[i] = (float)s;
    }
    f.resize(kD);
    rms(mix.data(), c < 3 ? D.L[c].dflare_norm : D.dspark_norm, f.data(), kD, D.tgt.rms_eps);
}

void ctx_kv(const DrafterImpl& D, int l, const float* f, int64_t pos, std::vector<float>& K, std::vector<float>& V) {
    K.resize(KVW);
    V.resize(KVW);
    lin(D.L[l].wk_ctx, f, K.data());
    lin(D.L[l].wv_ctx, f, V.data());
    for (int g = 0; g < NKV; ++g) {
        rms(K.data() + g * HD, D.L[l].k_ctx_norm, K.data() + g * HD, HD, D.tgt.rms_eps);
        rope(K.data() + g * HD, pos);
    }
    for (auto& v : K) v = f16r(v);
    for (auto& v : V) v = f16r(v);
}

void block_pass(const DrafterImpl& D, const std::vector<float> ck[NL], const std::vector<float> cv[NL], int64_t p, int32_t anchor, BlockOut* out) {
    const double eps = D.tgt.rms_eps;
    std::vector<float> X((size_t)BS * kD);
    {   // slot 0: embed_norm(embed(anchor)); slots 1..7: mask_embd
        std::vector<float> emb(kD);
        const block_q8_0* row = (const block_q8_0*)(D.tgt.tok_embd + (size_t)anchor * D.tgt.tok_embd_rs);
        for (int b = 0; b < kD / 32; ++b)
            for (int j = 0; j < 32; ++j) emb[b * 32 + j] = fp16_to_f32(row[b].d) * (float)row[b].qs[j];
        rms(emb.data(), D.tgt.tok_norm, X.data(), kD, eps);
        for (int s = 1; s < BS; ++s) std::memcpy(X.data() + (size_t)s * kD, D.mask_embd, kD * 4);
    }
    for (int l = 0; l < NL; ++l) {
        const LayerW& W = D.L[l];
        std::vector<float> q((size_t)BS * QW), k((size_t)BS * KVW), v((size_t)BS * KVW), h(kD);
        for (int s = 0; s < BS; ++s) {
            rms(X.data() + (size_t)s * kD, W.attn_norm, h.data(), kD, eps);
            lin(W.wq, h.data(), q.data() + (size_t)s * QW);
            lin(W.wk, h.data(), k.data() + (size_t)s * KVW);
            lin(W.wv, h.data(), v.data() + (size_t)s * KVW);
            for (int hh = 0; hh < NQH; ++hh) {
                rms(q.data() + (size_t)s * QW + hh * HD, W.q_norm, q.data() + (size_t)s * QW + hh * HD, HD, eps);
                rope(q.data() + (size_t)s * QW + hh * HD, p + s);
            }
            for (int g = 0; g < NKV; ++g) {
                rms(k.data() + (size_t)s * KVW + g * HD, W.k_norm, k.data() + (size_t)s * KVW + g * HD, HD, eps);
                rope(k.data() + (size_t)s * KVW + g * HD, p + s);
            }
        }
        const int64_t lo = (l == 1) ? 0 : std::max<int64_t>(0, p - 1024);
        for (int s = 0; s < BS; ++s) {
            std::vector<float> attn(QW);
            for (int hh = 0; hh < NQH; ++hh) {
                const int g = hh / 3;
                const float* qh = q.data() + (size_t)s * QW + hh * HD;
                std::vector<double> sc;
                for (int64_t t = lo; t < p; ++t) {
                    double d = 0;
                    for (int i = 0; i < HD; ++i) d += (double)qh[i] * ck[l][(size_t)t * KVW + g * HD + i];
                    sc.push_back(d / std::sqrt((double)HD));
                }
                for (int t = 0; t < BS; ++t) {
                    double d = 0;
                    for (int i = 0; i < HD; ++i) d += (double)qh[i] * k[(size_t)t * KVW + g * HD + i];
                    sc.push_back(d / std::sqrt((double)HD));
                }
                const double mx = *std::max_element(sc.begin(), sc.end());
                double Z = 0;
                for (auto& x : sc) { x = std::exp(x - mx); Z += x; }
                for (int i = 0; i < HD; ++i) {
                    double a = 0;
                    for (int64_t t = lo; t < p; ++t) a += sc[(size_t)(t - lo)] * cv[l][(size_t)t * KVW + g * HD + i];
                    for (int t = 0; t < BS; ++t) a += sc[(size_t)(p - lo) + t] * v[(size_t)t * KVW + g * HD + i];
                    attn[hh * HD + i] = (float)(a / Z);
                }
            }
            std::vector<float> o(kD);
            lin(W.wo, attn.data(), o.data());
            for (int i = 0; i < kD; ++i) X[(size_t)s * kD + i] += o[i];
            std::vector<float> h2(kD), g(FF), u(FF), a(FF), dn(kD);
            rms(X.data() + (size_t)s * kD, W.ffn_norm, h2.data(), kD, eps);
            lin(W.wgate, h2.data(), g.data());
            lin(W.wup, h2.data(), u.data());
            for (int i = 0; i < FF; ++i) a[i] = (float)((double)g[i] / (1.0 + std::exp(-(double)g[i])) * u[i]);
            lin(W.wdown, a.data(), dn.data());
            for (int i = 0; i < kD; ++i) X[(size_t)s * kD + i] += dn[i];
        }
        out->X[l] = X;
    }
    out->hid.assign((size_t)BS * kD, 0.0f);
    for (int s = 1; s < BS; ++s) rms(X.data() + (size_t)s * kD, D.out_norm, out->hid.data() + (size_t)s * kD, kD, eps);
}

void chain(const DrafterImpl& D, const float* f_rnn, const float* hid, int32_t anchor, int n, const int32_t* force, ChainOut* out) {
    const int nv = D.tgt.n_vocab;
    std::vector<double> s(RANK, 0.0);
    int32_t prev = anchor;
    out->lat.clear();
    out->logits.clear();
    out->tokens.clear();
    std::vector<float> base(nv), bias(nv), corr(kD);
    for (int j = 1; j <= n; ++j) {
        std::vector<float> z(JW);
        for (int i = 0; i < RANK; ++i) z[i] = (float)s[i];
        for (int i = 0; i < RANK; ++i) z[RANK + i] = bf16_to_f32(D.markov_w1[(size_t)prev * RANK + i]);
        for (int i = 0; i < kD; ++i) z[2 * RANK + i] = hid[(size_t)j * kD + i];
        for (int i = 0; i < kD; ++i) z[2 * RANK + kD + i] = f_rnn[i];
        std::vector<double> u(JO);
        for (int r = 0; r < JO; ++r) {
            double a = D.joint_b[r];
            for (int i = 0; i < JW; ++i) a += (double)bf16_to_f32(D.joint_w[(size_t)r * JW + i]) * z[i];
            u[r] = a;
        }
        std::vector<float> lat(RANK);
        for (int i = 0; i < RANK; ++i) {
            const double g = 1.0 / (1.0 + std::exp(-u[i])), c = std::tanh(u[RANK + i]);
            s[i] = g * s[i] + (1.0 - g) * c;
            lat[i] = (float)std::tanh(u[2 * RANK + i]);
        }
        for (int k = 0; k < kD; ++k) {
            double a = D.hout_b[k];
            for (int i = 0; i < RANK; ++i) a += (double)bf16_to_f32(D.hout_w[(size_t)k * RANK + i]) * lat[i];
            corr[k] = (float)a;
        }
        lin_pool(*D.tgt.pool, D.tgt.lm_head, nv, hid + (size_t)j * kD, base.data());
        lin_pool(*D.tgt.pool, D.tgt.lm_head, nv, corr.data(), bias.data());
        std::vector<float> logit(nv);
        int best = 0;
        for (int v = 0; v < nv; ++v) {
            logit[v] = (float)(30.0 * std::tanh(((double)base[v] + bias[v]) / 30.0));
            if (logit[v] > logit[best]) best = v;
        }
        prev = force ? force[j - 1] : best;
        out->lat.push_back(lat);
        out->logits.push_back(std::move(logit));
        out->tokens.push_back(prev);
    }
}

}  // namespace rd
}  // namespace dr
}  // namespace ink

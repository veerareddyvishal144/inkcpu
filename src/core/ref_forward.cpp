#include "ref_forward.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "ink/ggml_types.h"

namespace ink {

namespace {

// ------------------------------------------------------------------ rounding helpers
inline float f16r(float x) { return fp16_to_f32(f32_to_fp16(x)); }

struct Tabs {
    float ue4m3[256];
    Tabs() { for (int i = 0; i < 256; ++i) ue4m3[i] = ue4m3_to_f32((uint8_t)i); }
};
const Tabs& tabs() { static Tabs t; return t; }

// Q8_0 quantisation exactly like ggml's x86 quantize_row_q8_0 (AVX2 path): d = amax/127 stored as fp16, but the
// multiplier is id = 127/amax (not 1/d) and rounding is round-half-to-even. (quantize_row_q8_0_ref uses roundf(x/d).)
void quantize_q8_0(const float* x, int64_t k, block_q8_0* y) {
    const int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * QK8_0;
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) amax = std::max(amax, std::fabs(xb[j]));
        y[i].d = f32_to_fp16(amax / 127.0f);
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            float q = std::nearbyintf(xb[j] * id);
            q = std::min(127.0f, std::max(-128.0f, q));
            y[i].qs[j] = (int8_t)q;
        }
    }
}

// An activation vector with lazily built ggml-style quantised forms.
struct Act {
    const float* x = nullptr;
    int64_t K = 0;
    std::vector<block_q8_0> q8;
    std::vector<uint16_t> bf;
    void build_q8() { if (q8.empty()) { q8.resize(K / QK8_0); quantize_q8_0(x, K, q8.data()); } }
    void build_bf() { if (bf.empty()) { bf.resize(K); for (int64_t i = 0; i < K; ++i) bf[i] = f32_to_bf16(x[i]); } }
};

// F32 dot with the accumulation order of ggml's llamafile tinyBLAS AVX512 kernel (k % 16 == 0): ONE 16-lane accumulator
// chained with FMA over k/16 steps, then _mm512_reduce_add_ps (GCC tree: hi256+lo256, hi128+lo128, then 2+2, then 1+1).
// Other lengths fall back to a plain float loop.
inline float dotf(const float* a, const float* b, int64_t n) {
    if (n % 16 == 0) {
        float acc[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int64_t i = 0; i < n; i += 16)
            for (int l = 0; l < 16; ++l) acc[l] = std::fmaf(a[i + l], b[i + l], acc[l]);
        float r8[8], r4[4];
        for (int i = 0; i < 8; ++i) r8[i] = acc[i + 8] + acc[i];
        for (int i = 0; i < 4; ++i) r4[i] = r8[i + 4] + r8[i];
        const float r2a = r4[2] + r4[0], r2b = r4[3] + r4[1];
        return r2a + r2b;
    }
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// ggml_vec_dot_f32, AVX512 branch: four 16-lane FMA accumulators over 64-element steps, reduced 0+2, 1+3, then 0+1, then
// _mm512_reduce_add_ps, leftovers (n % 64) added in double.
inline float reduce16(const float v[16]);
inline float dot_vecdot(const float* x, const float* y, int64_t n) {
    float sum[4][16];
    for (auto& r : sum) for (float& e : r) e = 0.0f;
    const int64_t np = n & ~(int64_t)63;
    for (int64_t i = 0; i < np; i += 64)
        for (int j = 0; j < 4; ++j)
            for (int l = 0; l < 16; ++l) sum[j][l] = std::fmaf(x[i + 16 * j + l], y[i + 16 * j + l], sum[j][l]);
    for (int l = 0; l < 16; ++l) { sum[0][l] += sum[2][l]; sum[1][l] += sum[3][l]; }
    for (int l = 0; l < 16; ++l) sum[0][l] += sum[1][l];
    double sumf = np ? (double)reduce16(sum[0]) : 0.0;
    for (int64_t i = np; i < n; ++i) sumf += (double)(x[i] * y[i]);
    return (float)sumf;
}

// ---- bit-exact emulation of ggml's x86 AVX2 kernels (arch/x86/quants.c), so that the float results (and therefore the
// downstream discrete decisions: Q8_0 re-quantisation of activations, top-k expert choice, F16 rounding) agree with llama.cpp.
// 8 accumulator lanes, lane l holds the int sum of elements 4l..4l+3 (maddubs + madd), one FMA per block, hsum_float_8 at the end.
inline float hsum8(const float a[8]) {
    const float s0 = a[4] + a[0], s1 = a[5] + a[1], s2 = a[6] + a[2], s3 = a[7] + a[3];
    return (s0 + s2) + (s1 + s3);
}

// ggml_vec_dot_bf16, AVX512F branch (no AVX512_BF16 on this CPU): two 16-lane accumulators over 32-element steps, products
// and sums NOT fused, each accumulator reduced with _mm512_reduce_add_ps and added in double, leftovers in double, cast to float.
inline float reduce16(const float v[16]) {
    float r8[8], r4[4];
    for (int i = 0; i < 8; ++i) r8[i] = v[i + 8] + v[i];
    for (int i = 0; i < 4; ++i) r4[i] = r8[i + 4] + r8[i];
    return (r4[2] + r4[0]) + (r4[3] + r4[1]);
}
inline float dot_bf16_ggml(const uint16_t* w, const uint16_t* a, int64_t n) {
    float c1[16] = {0}, c2[16] = {0};
    int64_t i = 0;
    for (; i + 32 <= n; i += 32)
        for (int l = 0; l < 16; ++l) {
            c1[l] = (bf16_to_f32(w[i + l]) * bf16_to_f32(a[i + l])) + c1[l];
            c2[l] = (bf16_to_f32(w[i + 16 + l]) * bf16_to_f32(a[i + 16 + l])) + c2[l];
        }
    double sumf = 0.0;
    sumf += (double)reduce16(c1);
    sumf += (double)reduce16(c2);
    for (; i < n; ++i) sumf += (double)(bf16_to_f32(w[i]) * bf16_to_f32(a[i]));
    return (float)sumf;
}

inline float dot_q8_q8(const block_q8_0* w, const block_q8_0* a, int64_t nb) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t b = 0; b < nb; ++b) {
        const float d = fp16_to_f32(w[b].d) * fp16_to_f32(a[b].d);
        for (int l = 0; l < 8; ++l) {
            int32_t t = 0;
            for (int i = 0; i < 4; ++i) t += (int32_t)w[b].qs[4 * l + i] * (int32_t)a[b].qs[4 * l + i];
            acc[l] = std::fmaf(d, (float)t, acc[l]);
        }
    }
    return hsum8(acc);
}

// ggml_vec_dot_nvfp4_q8_0, AVX2 branch: per 64-block two FMAs (sub-blocks 0,1 with q8 block 2b, then 2,3 with q8 block 2b+1);
// lanes 0-3 = first sub-block of the pair, lanes 4-7 = second.
inline float dot_nvfp4_q8(const block_nvfp4* w, const block_q8_0* a, int64_t nb64) {
    const float* tab = tabs().ue4m3;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t b = 0; b < nb64; ++b) {
        for (int half = 0; half < 2; ++half) {
            const block_q8_0& y = a[2 * b + half];
            const float dy = fp16_to_f32(y.d);
            const float sc[2] = {tab[w[b].d[2 * half]] * dy, tab[w[b].d[2 * half + 1]] * dy};
            for (int l = 0; l < 8; ++l) {
                const int sub = l >> 2, k = l & 3;                   // sub-block within the pair, 4-element group within it
                const uint8_t* q = w[b].qs + (2 * half + sub) * 8;
                int32_t t = 0;
                for (int i = 0; i < 4; ++i) {
                    const int e = 4 * k + i;                         // element within the 16-element sub-block
                    const int nib = e < 8 ? (q[e] & 15) : (q[e - 8] >> 4);
                    t += (int32_t)kvalues_fp4[nib] * (int32_t)y.qs[sub * 16 + e];
                }
                acc[l] = std::fmaf(sc[sub], (float)t, acc[l]);
            }
        }
    }
    return hsum8(acc);
}

// dequantise one row of `k` elements to float
void dequant_row(int type, const uint8_t* row, int64_t k, float* out) {
    switch (type) {
        case T_F32: std::memcpy(out, row, (size_t)k * 4); break;
        case T_F16: for (int64_t i = 0; i < k; ++i) { uint16_t h; std::memcpy(&h, row + 2 * i, 2); out[i] = fp16_to_f32(h); } break;
        case T_BF16: for (int64_t i = 0; i < k; ++i) { uint16_t h; std::memcpy(&h, row + 2 * i, 2); out[i] = bf16_to_f32(h); } break;
        case T_Q8_0: {
            const block_q8_0* b = (const block_q8_0*)row;
            for (int64_t i = 0; i < k / QK8_0; ++i) {
                const float d = fp16_to_f32(b[i].d);
                for (int j = 0; j < QK8_0; ++j) out[i * QK8_0 + j] = d * (float)b[i].qs[j];
            }
        } break;
        case T_NVFP4: {
            const block_nvfp4* b = (const block_nvfp4*)row;
            const float* tab = tabs().ue4m3;
            for (int64_t i = 0; i < k / QK_NVFP4; ++i)
                for (int sb = 0; sb < 4; ++sb) {
                    const float d = tab[b[i].d[sb]];
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t q = b[i].qs[sb * 8 + j];
                        out[i * 64 + sb * 16 + j] = (float)kvalues_fp4[q & 15] * d;
                        out[i * 64 + sb * 16 + 8 + j] = (float)kvalues_fp4[q >> 4] * d;
                    }
                }
        } break;
        default: throw std::runtime_error("dequant_row: unsupported ggml type " + std::to_string(type));
    }
}

// dot of one weight row with the activation. aq = quantise activation like ggml, else float activation.
inline float row_dot(int type, const uint8_t* row, int64_t K, Act& a, bool aq, float* tmp) {
    switch (type) {
        case T_F32: return dotf((const float*)row, a.x, K);
        case T_BF16:
            if (aq) return dot_bf16_ggml((const uint16_t*)row, a.bf.data(), K);
            // fallthrough
        case T_F16:
            dequant_row(type, row, K, tmp);
            return dotf(tmp, a.x, K);
        case T_Q8_0:
            if (aq) return dot_q8_q8((const block_q8_0*)row, a.q8.data(), K / QK8_0);
            dequant_row(type, row, K, tmp);
            return dotf(tmp, a.x, K);
        case T_NVFP4:
            if (aq) return dot_nvfp4_q8((const block_nvfp4*)row, a.q8.data(), K / QK_NVFP4);
            dequant_row(type, row, K, tmp);
            return dotf(tmp, a.x, K);
        default: throw std::runtime_error("row_dot: unsupported ggml type " + std::to_string(type));
    }
}

void prep_act(const TensorRef& W, Act& a, bool aq) {
    if (!aq) return;
    if (W.type == T_Q8_0 || W.type == T_NVFP4) a.build_q8();
    else if (W.type == T_BF16) a.build_bf();
}

// out[s*nr + r] = dot(W.row(r0 + r, experts[s]), acts[shared ? 0 : s])
void gemv(const TensorRef& W, const int* experts, int nsel, int64_t r0, int64_t nr, std::vector<Act>& acts, bool shared, float* out, bool aq, int nth) {
    if (!W.valid()) throw std::runtime_error("gemv: missing tensor");
    const int64_t K = W.ne[0];
    for (size_t i = 0; i < acts.size(); ++i) {
        if (acts[i].K != K) throw std::runtime_error("gemv: activation length != K");
        prep_act(W, acts[i], aq);
        if (shared) break;
    }
    const int64_t total = (int64_t)nsel * nr;
    const int type = W.type;
    auto body = [&](int64_t i, float* tmp) {
        const int s = (int)(i / nr);
        const int64_t r = i % nr;
        out[i] = row_dot(type, W.row(r0 + r, experts[s]), K, acts[shared ? 0 : s], aq, tmp);
    };
    if (total < 64 || nth <= 1) {
        std::vector<float> tmp(aq && (type == T_F32 || type == T_Q8_0 || type == T_NVFP4 || type == T_BF16) ? 1 : (size_t)K);
        if (!aq) tmp.resize((size_t)K);
        for (int64_t i = 0; i < total; ++i) body(i, tmp.data());
        return;
    }
#pragma omp parallel num_threads(nth)
    {
        std::vector<float> tmp(aq ? 1 : (size_t)K);
#pragma omp for schedule(static)
        for (int64_t i = 0; i < total; ++i) body(i, tmp.data());
    }
}

// single-matrix helper: out[nr] = W (rows r0..r0+nr-1) * x
void matvec(const TensorRef& W, const float* x, float* out, bool aq, int nth, int64_t r0 = 0, int64_t nr = -1) {
    std::vector<Act> a(1);
    a[0].x = x;
    a[0].K = W.ne[0];
    const int e = 0;
    gemv(W, &e, 1, r0, nr < 0 ? W.ne[1] : nr, a, true, out, aq, nth);
}

// ------------------------------------------------------------------ elementwise pieces
void rms_norm(const float* x, float* y, int n, float eps, const float* w = nullptr) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)(x[i] * x[i]);
    const float mean = (float)(sum / n);
    const float scale = 1.0f / std::sqrt(mean + eps);
    if (w) for (int i = 0; i < n; ++i) y[i] = (x[i] * scale) * w[i];
    else   for (int i = 0; i < n; ++i) y[i] = x[i] * scale;
}

// clip(): per row of `dim`: optional per-channel scale, then scale the row down to L2 norm <= thr (norm + eps)
void clip_rows(float* x, int nrows, int dim, const float* scale, float thr, float eps) {
    for (int r = 0; r < nrows; ++r) {
        float* p = x + (size_t)r * dim;
        if (scale) for (int d = 0; d < dim; ++d) p[d] *= scale[(size_t)r * dim + d];
        double s = 0.0;
        for (int d = 0; d < dim; ++d) s += (double)(p[d] * p[d]);
        float norm = std::sqrt((float)s);
        norm = norm * 1.0f + eps;
        float factor = thr / norm;
        factor = std::min(1.0f, std::max(0.0f, factor));
        for (int d = 0; d < dim; ++d) p[d] *= factor;
    }
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// depth mix: states (raw, un-normed) -> out[q] = sum_s softmax_s(rms(state_s) . query_q / sqrt(D)) * rms(state_s)
void depth_mix(const std::vector<const float*>& st, int D, const float* queries, int nq, float eps, float* out) {
    const int S = (int)st.size();
    std::vector<float> nrm((size_t)S * D);
    for (int s = 0; s < S; ++s) rms_norm(st[s], nrm.data() + (size_t)s * D, D, eps);
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> sc(S);
    for (int q = 0; q < nq; ++q) {
        float mx = -INFINITY;
        for (int s = 0; s < S; ++s) {
            static const int md = std::getenv("INK_MIXDOT") ? std::atoi(std::getenv("INK_MIXDOT")) : 1;   // default 1: measured bit-exact (ggml_vec_dot_f32 order)
            sc[s] = (md == 1 ? dot_vecdot(nrm.data() + (size_t)s * D, queries + (size_t)q * D, D) : dotf(nrm.data() + (size_t)s * D, queries + (size_t)q * D, D)) * scale;
            mx = std::max(mx, sc[s]);
        }
        double sum = 0.0;
        for (int s = 0; s < S; ++s) { sc[s] = std::exp(sc[s] - mx); sum += sc[s]; }
        const float inv = (float)(1.0 / sum);
        float* o = out + (size_t)q * D;
        std::vector<float> w(S);
        for (int s = 0; s < S; ++s) w[s] = sc[s] * inv;
        // ggml: mul_mat over the (tiny) state axis (k = number of states): measured to match a plain float FMA chain
        static const int mv = std::getenv("INK_MIX_VARIANT") ? std::atoi(std::getenv("INK_MIX_VARIANT")) : 0;
        for (int d = 0; d < D; ++d) {
            if (mv == 1) {        // exact products accumulated in double, one final rounding
                double acc = 0.0;
                for (int s = 0; s < S; ++s) acc += (double)nrm[(size_t)s * D + d] * (double)w[s];
                o[d] = (float)acc;
            } else if (mv == 2) { // float products, float adds (no FMA)
                float acc = 0.0f;
                for (int s = 0; s < S; ++s) acc = acc + nrm[(size_t)s * D + d] * w[s];
                o[d] = acc;
            } else if (mv == 3) { // FMA chain in reverse state order
                float acc = 0.0f;
                for (int s = S - 1; s >= 0; --s) acc = std::fmaf(nrm[(size_t)s * D + d], w[s], acc);
                o[d] = acc;
            } else {
                float acc = 0.0f;
                for (int s = 0; s < S; ++s) acc = std::fmaf(nrm[(size_t)s * D + d], w[s], acc);
                o[d] = acc;
            }
        }
    }
}

// loop controller: (cur - forget*cur) + gain*message with gates from the rank-r bottleneck of [rms(cur), rms(message)]
void control(const float* cur, const float* msg, const LoopCtrl& c, int D, float eps, bool aq, int nth, float* out) {
    std::vector<float> gate_in(2 * (size_t)D);
    rms_norm(cur, gate_in.data(), D, eps);
    rms_norm(msg, gate_in.data() + D, D, eps);
    const int R = (int)c.down.ne[1];
    std::vector<float> g(R), gain(D), forget(D);
    matvec(c.down, gate_in.data(), g.data(), aq, nth);
    matvec(c.gain, g.data(), gain.data(), aq, nth);
    matvec(c.forget, g.data(), forget.data(), aq, nth);
    for (int i = 0; i < D; ++i) {
        const float ga = std::tanh(gain[i]);
        const float fo = std::tanh(forget[i]);
        out[i] = (cur[i] - fo * cur[i]) + ga * msg[i];
    }
}

// causal depthwise short conv with residual (ggml `convolve`). hist = last (nc-1) inputs, oldest first. dyn: [heads][nc] taps.
// Arithmetic measured bit-exact against llama.cpp: ssm_conv is a plain float loop WITHOUT FMA contraction
// (sumf = ((0 + x0*w0) + x1*w1) + ...), then ggml_add(cur, out), then ggml_scale. This file is built with -ffp-contract=off.
void convolve(const float* in, const float* w, const float* dyn, int heads, int ch, int nc, std::vector<float>& hist, float* out) {
    const int hs = nc - 1;
    const float scale = 1.0f / std::sqrt((float)((dyn ? 2 : 1) * nc + 1));
    const int cph = dyn ? ch / heads : 1;
    for (int c = 0; c < ch; ++c) {
        float xs[16];
        for (int j = 0; j < nc; ++j) xs[j] = j < hs ? hist[(size_t)j * ch + c] : in[c];
        float s = 0.0f;
        for (int j = 0; j < nc; ++j) s += xs[j] * w[(size_t)c * nc + j];       // tap j multiplies the input (nc-1-j) tokens ago
        if (dyn) {
            const int h = c / cph;
            for (int lag = 0; lag < nc; ++lag) s = s + xs[nc - 1 - lag] * dyn[(size_t)h * nc + lag];   // lag 0 = current input
        }
        out[c] = (in[c] + s) * scale;
    }
    for (int j = 0; j + 1 < hs; ++j) std::memcpy(&hist[(size_t)j * ch], &hist[(size_t)(j + 1) * ch], (size_t)ch * sizeof(float));
    std::memcpy(&hist[(size_t)(hs - 1) * ch], in, (size_t)ch * sizeof(float));
}

}  // namespace

// ------------------------------------------------------------------ state
size_t RefState::bytes() const {
    size_t b = 0;
    for (const auto& v : visits)
        b += (v.k.capacity() + v.v.capacity() + v.hist_k.capacity() + v.hist_v.capacity() + v.hist_attn.capacity() + v.hist_ffn.capacity()) * 4 +
             v.cell_pos.capacity() * 4;
    return b;
}

RefModel::RefModel(const GgufFile& g, const RefOptions& opt) : opt_(opt) {
    std::string err;
    if (!ModelConfig::from_gguf(g, &cfg_, &err)) throw std::runtime_error("config: " + err);
    if (!load_weights(g, cfg_, &w_, &err)) throw std::runtime_error("weights: " + err);
    plan_ = visit_plan(cfg_);
    window_ = opt.sliding_window > 0 ? opt.sliding_window : cfg_.sliding_window;
    auto tri = [&](int v) { return v < 0 ? opt.act_quant_like_ggml : v != 0; };
    aq_ = opt.act_quant_like_ggml;
    kv16_ = tri(opt.kv_f16);
    q16_ = opt.q_f16 < 0 ? false : opt.q_f16 != 0;   // llama.cpp keeps Q in F32 (measured), see docs/MODEL.md
    bias16_ = tri(opt.bias_f16);
    if (cfg_.n_head != 24 && cfg_.head_dim % 2) throw std::runtime_error("unexpected head geometry");
}

RefState RefModel::new_state() const {
    RefState st;
    reset(st);
    return st;
}

void RefModel::reset(RefState& st) const {
    st.pos = 0;
    st.visits.assign(plan_.size(), RefVisitState());
    const size_t hs = (size_t)cfg_.n_conv - 1;
    for (auto& v : st.visits) {
        v.hist_k.assign(hs * cfg_.kv_width(), 0.0f);
        v.hist_v.assign(hs * cfg_.kv_width(), 0.0f);
        v.hist_attn.assign(hs * cfg_.n_embd, 0.0f);
        v.hist_ffn.assign(hs * cfg_.n_embd, 0.0f);
    }
}

// ------------------------------------------------------------------ MoE (build_moe_ffn with LLM_FFN_SITU, sigmoid gating)
void RefModel::moe(const LayerWeights& L, const float* x, float* out, int il) const {
    const ModelConfig& c = cfg_;
    const int D = c.n_embd, E = c.n_expert, U = c.n_expert_used, F = c.n_ff_exp;
    const int nth = opt_.n_threads;

    // router: F32 logits (router weight is F32 in the GGUF; a bf16 router would go through the same matvec)
    std::vector<float> logits(E), probs(E), sel(E);
    matvec(L.ffn_gate_inp, x, logits.data(), aq_, nth);
    trace("ffn_moe_logits", il, logits.data(), logits.size());
    for (int e = 0; e < E; ++e) {
        probs[e] = sigmoidf(logits[e]);
        sel[e] = L.exp_probs_b.valid() ? probs[e] + L.exp_probs_b.f32()[e] : probs[e];
    }
    std::vector<int> idx(E);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sel[a] > sel[b]; });
    std::vector<int> ex(idx.begin(), idx.begin() + U);
    if (trace_) { std::vector<float> idf(ex.begin(), ex.end()); trace("ffn_moe_topk", il, idf.data(), idf.size()); }
    std::vector<float> wt(U);
    double wsum_d = 0.0;   // ggml sum_rows: double accumulation, cast to float
    for (int i = 0; i < U; ++i) { wt[i] = probs[ex[i]]; wsum_d += (double)wt[i]; }
    float wsum = (float)wsum_d;
    wsum = wsum * 1.0f + c.hard_coded.expert_weights_norm_eps;   // ggml_scale_bias(sum, 1, eps)
    for (int i = 0; i < U; ++i) { wt[i] = wt[i] / wsum; wt[i] = wt[i] * c.expert_weights_scale; }
    trace("ffn_moe_weights_scaled", il, wt.data(), wt.size());

    // gate_up for the selected experts (activation shared by all of them)
    std::vector<Act> a0(1);
    a0[0].x = x;
    a0[0].K = D;
    std::vector<float> gu((size_t)U * 2 * F);
    gemv(L.ffn_gate_up_exps, ex.data(), U, 0, 2 * F, a0, true, gu.data(), aq_, nth);
    if (L.ffn_gate_up_scale.valid())
        for (int i = 0; i < U; ++i) {
            const float s = L.ffn_gate_up_scale.f32()[ex[i]];
            for (int r = 0; r < 2 * F; ++r) gu[(size_t)i * 2 * F + r] *= s;
        }
    trace("ffn_moe_gate_up_scaled", il, gu.data(), gu.size());
    // SiTU: beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)
    const float beta = c.situ_beta, lb = c.situ_linear_beta;
    std::vector<float> act((size_t)U * F);
    for (int i = 0; i < U; ++i) {
        const float* g = gu.data() + (size_t)i * 2 * F;
        const float* u = g + F;
        float* a = act.data() + (size_t)i * F;
        for (int j = 0; j < F; ++j) {
            float t = std::tanh(g[j] * (1.0f / beta)) * beta;
            t = t * sigmoidf(g[j]);
            float uu = u[j];
            if (lb > 0.0f) uu = std::tanh(uu * (1.0f / lb)) * lb;
            a[j] = t * uu;
        }
    }
    trace("ffn_moe_situ", il, act.data(), act.size());
    // down projection: each slot has its own activation row
    std::vector<Act> a1(U);
    for (int i = 0; i < U; ++i) { a1[i].x = act.data() + (size_t)i * F; a1[i].K = F; }
    std::vector<float> dn((size_t)U * D);
    gemv(L.ffn_down_exps, ex.data(), U, 0, D, a1, false, dn.data(), aq_, nth);
    if (L.ffn_down_scale.valid())
        for (int i = 0; i < U; ++i) {
            const float s = L.ffn_down_scale.f32()[ex[i]];
            for (int r = 0; r < D; ++r) dn[(size_t)i * D + r] *= s;
        }
    trace("ffn_moe_down_scaled", il, dn.data(), dn.size());
    for (int i = 0; i < U; ++i)
        for (int r = 0; r < D; ++r) dn[(size_t)i * D + r] *= wt[i];
    for (int r = 0; r < D; ++r) {
        float s = dn[r];
        for (int i = 1; i < U; ++i) s = s + dn[(size_t)i * D + r];   // sequential adds in slot order (ggml_add chain)
        out[r] = s;
    }
    trace("ffn_moe_out", il, out, D);
}

// ------------------------------------------------------------------ one layer visit (build_layer)
void RefModel::visit(RefState& st, const Visit& vis, std::vector<float>& cur, std::vector<std::vector<float>>& blocks, const float* msgs,
                     int pos, float tau) const {
    const ModelConfig& c = cfg_;
    const LayerWeights& L = w_.layers[vis.phys];
    RefVisitState& vs = st.visits[vis.index];
    const int D = c.n_embd, H = c.n_head, G = c.n_head_kv, HD = c.head_dim, KVW = c.kv_width(), QW = c.q_width();
    const int NR = c.n_rel, NC = c.n_conv;
    const int il = vis.llama_slot;
    const bool swa = L.is_swa;
    const int nth = opt_.n_threads;
    const float eps = c.rms_eps;
    const std::vector<float> inpSA = cur;

    // ---- attention inputs: depth mix of the block states and cur (1 query on SWA layers, 3 on global layers)
    std::vector<float> in[3];
    if (L.attn_res_query.valid()) {
        const int nq = swa ? 1 : 3;
        std::vector<const float*> sts;
        for (size_t bi = blocks.size(); bi-- > 0;) sts.push_back(blocks[bi].data());
        sts.push_back(cur.data());
        std::vector<float> mixed((size_t)nq * D);
        depth_mix(sts, D, L.attn_res_query.f32(), nq, c.hard_coded.mix_rms_eps, mixed.data());
        for (int j = 0; j < 3; ++j) in[j].assign(mixed.begin() + (size_t)(swa ? 0 : j) * D, mixed.begin() + (size_t)((swa ? 0 : j) + 1) * D);
    } else {
        for (int j = 0; j < 3; ++j) in[j] = cur;
    }
    if (msgs)
        for (int j = 0; j < 3; ++j) {
            std::vector<float> o(D);
            control(in[j].data(), msgs + (size_t)j * D, L.ctrl[j], D, c.hard_coded.ctrl_rms_eps, aq_, nth, o.data());
            in[j] = std::move(o);
        }
    if (swa && msgs) {
        for (int j = 0; j < 3; ++j) rms_norm(in[j].data(), in[j].data(), D, eps, L.attn_norm.f32());
    } else if (swa) {
        rms_norm(in[0].data(), in[0].data(), D, eps, L.attn_norm.f32());
        in[1] = in[0];
        in[2] = in[0];
    } else {
        rms_norm(in[0].data(), in[0].data(), D, eps, L.attn_q_inp_norm.f32());
        rms_norm(in[1].data(), in[1].data(), D, eps, L.attn_k_inp_norm.f32());
        rms_norm(in[2].data(), in[2].data(), D, eps, L.attn_v_inp_norm.f32());
    }

    // ---- projections
    std::vector<float> q(QW), kpre(KVW), vpre(KVW), r((size_t)NR * H), sink(H), kdyn, vdyn;
    matvec(L.wq, in[0].data(), q.data(), aq_, nth);
    matvec(L.wk, in[1].data(), kpre.data(), aq_, nth);
    matvec(L.wv, in[2].data(), vpre.data(), aq_, nth);
    matvec(L.attn_rel, in[0].data(), r.data(), aq_, nth);
    matvec(L.attn_sink_proj, in[0].data(), sink.data(), aq_, nth);
    if (!swa) {
        kdyn.resize((size_t)NC * G);
        vdyn.resize((size_t)NC * G);
        matvec(L.attn_k_conv_proj, in[1].data(), kdyn.data(), aq_, nth);
        matvec(L.attn_v_conv_proj, in[2].data(), vdyn.data(), aq_, nth);
    }
    std::vector<float> k(KVW), v(KVW);
    convolve(kpre.data(), L.attn_k_conv.f32(), swa ? nullptr : kdyn.data(), G, KVW, NC, vs.hist_k, k.data());
    convolve(vpre.data(), L.attn_v_conv.f32(), swa ? nullptr : vdyn.data(), G, KVW, NC, vs.hist_v, v.data());

    clip_rows(q.data(), H, HD, L.attn_q_clip.f32(), c.qk_clip, c.hard_coded.clip_norm_eps);
    clip_rows(k.data(), G, HD, nullptr, c.qk_clip, c.hard_coded.clip_norm_eps);
    for (int g = 0; g < G; ++g) rms_norm(v.data() + (size_t)g * HD, v.data() + (size_t)g * HD, HD, eps);
    clip_rows(r.data(), H, NR, L.attn_rel_clip.f32(), c.rel_clip, c.hard_coded.clip_norm_eps);
    if (!swa) {
        for (auto& x : q) x *= tau;
        for (auto& x : r) x *= tau;
    }
    trace("Qcur", il, q.data(), q.size());
    trace("Kcur", il, k.data(), k.size());
    trace("Vcur", il, v.data(), v.size());
    trace("Rcur", il, r.data(), r.size());

    // ---- cache insert (ring for SWA visits, append for global visits)
    int cell;
    if (swa && vs.n_cells >= window_) {
        cell = vs.ring_next;
        vs.ring_next = (vs.ring_next + 1) % window_;
    } else {
        cell = vs.n_cells++;
        vs.k.resize((size_t)vs.n_cells * KVW);
        vs.v.resize((size_t)vs.n_cells * KVW);
        vs.cell_pos.resize(vs.n_cells);
    }
    for (int i = 0; i < KVW; ++i) {
        vs.k[(size_t)cell * KVW + i] = kv16_ ? f16r(k[i]) : k[i];
        vs.v[(size_t)cell * KVW + i] = kv16_ ? f16r(v[i]) : v[i];
    }
    vs.cell_pos[cell] = pos;

    // ---- attention (scores = q.k/sqrt(d) + rel_bias[distance] - sink_logit; implicit zero-logit sink in the softmax)
    std::vector<float> ao(QW);
    {
        const float scale = 1.0f / std::sqrt((float)HD);
        const int rows = (int)L.attn_rel_bias.ne[1];           // bias table rows (extent); index `rows` = zero row
        const float* relb = L.attn_rel_bias.f32();
        const int rep = H / G;
        std::vector<float> s(vs.n_cells), qh(HD), tb(rows);
        for (int h = 0; h < H; ++h) {
            const int g = h / rep;
            for (int d = 0; d < HD; ++d) qh[d] = q16_ ? f16r(q[(size_t)h * HD + d]) : q[(size_t)h * HD + d];
            // table[dist] = rel_bias[:, dist] . R_h  (only the distances that occur are needed)
            float mx = 0.0f;   // the sink contributes a logit of 0
            for (int j = 0; j < vs.n_cells; ++j) {
                const int dist = pos - vs.cell_pos[j];
                if (dist < 0 || (swa && dist >= window_)) { s[j] = -INFINITY; continue; }
                const int bi = dist < rows ? dist : rows;
                float b = 0.0f;
                if (bi < rows) b = dotf(relb + (size_t)bi * NR, r.data() + (size_t)h * NR, NR);   // tinyBLAS order (k = 16)
                b = b - sink[h];
                if (bias16_) b = f16r(b);
                float dot = dotf(qh.data(), vs.k.data() + (size_t)j * KVW + (size_t)g * HD, HD);
                s[j] = dot * scale + b;
                mx = std::max(mx, s[j]);
            }
            double den = std::exp(0.0f - mx);
            for (int j = 0; j < vs.n_cells; ++j) {
                if (s[j] == -INFINITY) { s[j] = 0.0f; continue; }
                s[j] = std::exp(s[j] - mx);
                den += s[j];
            }
            float* o = ao.data() + (size_t)h * HD;
            for (int d = 0; d < HD; ++d) o[d] = 0.0f;
            for (int j = 0; j < vs.n_cells; ++j) {
                if (s[j] == 0.0f) continue;
                const float p = s[j];
                const float* vv = vs.v.data() + (size_t)j * KVW + (size_t)g * HD;
                for (int d = 0; d < HD; ++d) o[d] += p * vv[d];
            }
            const float inv = (float)(1.0 / den);
            for (int d = 0; d < HD; ++d) o[d] *= inv;
        }
    }

    // ---- output gate, wo, post conv
    std::vector<float> gate(QW);
    matvec(L.attn_gate, in[0].data(), gate.data(), aq_, nth);
    for (int i = 0; i < QW; ++i) ao[i] *= sigmoidf(gate[i]);
    std::vector<float> wo_out(D), attn_out(D);
    matvec(L.wo, ao.data(), wo_out.data(), aq_, nth);
    convolve(wo_out.data(), L.attn_post_conv.f32(), nullptr, 1, D, NC, vs.hist_attn, attn_out.data());
    trace("attn_out", il, attn_out.data(), attn_out.size());

    // ---- residual: the first layer of an attention-residual block starts a new stream
    if (vis.phys % c.attn_res_block_size == 0) {
        blocks.push_back(inpSA);
        cur = attn_out;
    } else {
        cur.resize(D);
        for (int i = 0; i < D; ++i) cur[i] = inpSA[i] + attn_out[i];
    }
    const std::vector<float> ffn_inp = cur;
    trace("ffn_inp", il, ffn_inp.data(), ffn_inp.size());

    // ---- MoE
    std::vector<float> mixed(D);
    {
        std::vector<const float*> sts;
        for (size_t bi = blocks.size(); bi-- > 0;) sts.push_back(blocks[bi].data());
        sts.push_back(ffn_inp.data());
        depth_mix(sts, D, L.ffn_res_query.f32(), 1, c.hard_coded.mix_rms_eps, mixed.data());
    }
    if (msgs) {
        std::vector<float> o(D);
        control(mixed.data(), msgs + (size_t)3 * D, L.ctrl[3], D, c.hard_coded.ctrl_rms_eps, aq_, nth, o.data());
        mixed = std::move(o);
    }
    {
        std::vector<float> nn(D);
        rms_norm(mixed.data(), nn.data(), D, eps);
        trace("norm", il, nn.data(), nn.size());
        const float* fw = L.ffn_norm.f32();
        for (int i = 0; i < D; ++i) mixed[i] = nn[i] * fw[i];
    }
    std::vector<float> moe_out(D), ffn_out(D);
    moe(L, mixed.data(), moe_out.data(), il);
    convolve(moe_out.data(), L.ffn_post_conv.f32(), nullptr, 1, D, NC, vs.hist_ffn, ffn_out.data());
    trace("ffn_out", il, ffn_out.data(), ffn_out.size());
    for (int i = 0; i < D; ++i) cur[i] = ffn_inp[i] + ffn_out[i];
    trace("l_out", il, cur.data(), cur.size());
}

// ------------------------------------------------------------------ full token
void RefModel::forward(RefState& st, int32_t token, float* logits) const {
    const ModelConfig& c = cfg_;
    const int D = c.n_embd;
    if (token < 0 || token >= c.n_vocab_padded) throw std::out_of_range("token id out of range");
    if (st.visits.size() != plan_.size()) reset(st);
    const int pos = st.pos;

    // tau = 1 + alpha*log(max(1, (pos+1)/floor))   (same float operation order as the ggml graph)
    float tau;
    {
        float t = (float)pos + 1.0f;
        t = t * (1.0f / (float)c.log_scale_floor);
        t = std::min(INFINITY, std::max(1.0f, t));
        tau = std::log(t) * c.log_scale_alpha + 1.0f;
    }

    std::vector<float> cur(D);
    dequant_row(w_.tok_embd.type, w_.tok_embd.row(token), D, cur.data());
    rms_norm(cur.data(), cur.data(), D, c.rms_eps, w_.tok_norm.f32());
    if (opt_.debug_embed_noise != 0.0f)
        for (int i = 0; i < D; ++i) {
            uint32_t h = (uint32_t)(i * 2654435761u) ^ (uint32_t)(pos * 40503u + 12345u);
            h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
            cur[i] *= 1.0f + opt_.debug_embed_noise * ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
        }

    std::vector<std::vector<float>> blocks;
    int idx = 0;
    for (; idx < c.loop_start; ++idx) visit(st, plan_[idx], cur, blocks, nullptr, pos, tau);

    const std::vector<float> prelude = cur;
    const std::vector<std::vector<float>> prelude_blocks = blocks;
    std::vector<std::vector<float>> bank;
    std::vector<float> bank_out;
    for (int p = 0; p < c.loop_count; ++p) {
        if (p > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());   // blocks after the first core block
            bank_out = cur;                                                           // output of the previous pass
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ip = c.loop_start; ip < c.loop_end; ++ip, ++idx) {
            std::vector<float> msgs;
            if (p > 0) {
                std::vector<const float*> sts;
                for (size_t bi = bank.size(); bi-- > 0;) sts.push_back(bank[bi].data());
                sts.push_back(bank_out.data());
                msgs.resize((size_t)4 * D);
                depth_mix(sts, D, w_.layers[ip].loop_msg_query.f32(), 4, c.hard_coded.mix_rms_eps, msgs.data());
            }
            visit(st, plan_[idx], cur, blocks, p > 0 ? msgs.data() : nullptr, pos, tau);
        }
    }
    for (; idx < (int)plan_.size(); ++idx) visit(st, plan_[idx], cur, blocks, nullptr, pos, tau);
    st.pos = pos + 1;

    if (!logits) return;
    std::vector<float> mixed(D), xn(D);
    {
        std::vector<const float*> sts;
        for (size_t bi = blocks.size(); bi-- > 0;) sts.push_back(blocks[bi].data());
        sts.push_back(cur.data());
        depth_mix(sts, D, w_.output_res_query.f32(), 1, c.hard_coded.mix_rms_eps, mixed.data());
    }
    rms_norm(mixed.data(), xn.data(), D, c.rms_eps, w_.output_norm.f32());
    trace("result_norm", -1, xn.data(), xn.size());
    matvec(w_.output, xn.data(), logits, aq_, opt_.n_threads, 0, c.n_vocab);   // padded rows dropped
    if (c.final_logit_softcap != 0.0f) {
        const float cap = c.final_logit_softcap;
        for (int i = 0; i < c.n_vocab; ++i) logits[i] = std::tanh(logits[i] * (1.0f / cap)) * cap;
    }
}

std::vector<float> RefModel::run(RefState& st, const std::vector<int32_t>& tokens) const {
    std::vector<float> logits(cfg_.n_vocab);
    for (size_t i = 0; i < tokens.size(); ++i) forward(st, tokens[i], i + 1 == tokens.size() ? logits.data() : nullptr);
    return logits;
}

}  // namespace ink

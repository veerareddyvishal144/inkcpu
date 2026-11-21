// Scalar reference implementations (ink::ref::*): literal ggml semantics, double accumulation where it matters.  Owner: agent B2.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>
#include "ink/kernels_misc.h"

namespace ink {
namespace ref {

void vexp(const float* x, float* y, size_t n) { for (size_t i = 0; i < n; ++i) y[i] = (float)std::exp((double)x[i]); }
void vtanh(const float* x, float* y, size_t n) { for (size_t i = 0; i < n; ++i) y[i] = (float)std::tanh((double)x[i]); }
void vsigmoid(const float* x, float* y, size_t n) { for (size_t i = 0; i < n; ++i) y[i] = (float)(1.0 / (1.0 + std::exp(-(double)x[i]))); }
void vsitu(const float* g, const float* u, float* y, size_t n, float beta, float lin) {
    for (size_t i = 0; i < n; ++i) {
        const double gd = g[i], ud = u[i];
        const double act = (double)beta * std::tanh(gd / (double)beta) * (1.0 / (1.0 + std::exp(-gd)));
        const double upv = lin > 0.0f ? (double)lin * std::tanh(ud / (double)lin) : ud;
        y[i] = (float)(act * upv);
    }
}
void vsoftcap(const float* x, float* y, size_t n, float cap) { for (size_t i = 0; i < n; ++i) y[i] = (float)((double)cap * std::tanh((double)x[i] / (double)cap)); }

void rmsnorm(const float* x, const float* w, float* y, size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const double s = 1.0 / std::sqrt(ss / (double)n + (double)eps);
    for (size_t i = 0; i < n; ++i) y[i] = (float)((double)x[i] * s * (w ? (double)w[i] : 1.0));
}

void quantize_q8_0(const float* x, block_q8_0* out, size_t n, Q8Mode mode) {
    for (size_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32 * b + j]));
        const float d = amax / 127.0f;
        out[b].d = f32_to_fp16(d);
        if (mode == Q8Mode::Ref) {
            const float id = d ? 1.0f / d : 0.0f;
            for (int j = 0; j < 32; ++j) out[b].qs[j] = (int8_t)roundf(x[32 * b + j] * id);
        } else {
            const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
            for (int j = 0; j < 32; ++j) out[b].qs[j] = (int8_t)nearbyintf(x[32 * b + j] * id);
        }
    }
}

void l2clip_heads(float* x, const float* chan_scale, int n_heads, int dim, float threshold, float post_scale) {
    for (int h = 0; h < n_heads; ++h) {
        float* xh = x + (size_t)h * dim;
        double ss = 0.0;
        for (int i = 0; i < dim; ++i) {
            if (chan_scale) xh[i] = xh[i] * chan_scale[(size_t)h * dim + i];
            ss += (double)xh[i] * xh[i];
        }
        const float norm = sqrtf((float)ss) + 1e-6f;
        const float f = std::min(std::max(threshold / norm, 0.0f), 1.0f);
        for (int i = 0; i < dim; ++i) xh[i] = (xh[i] * f) * post_scale;
    }
}

void vnorm_heads(const float* x, float* y, int n_heads, int dim, float eps) {
    for (int h = 0; h < n_heads; ++h) rmsnorm(x + (size_t)h * dim, nullptr, y + (size_t)h * dim, (size_t)dim, eps);
}

void sconv_seq(const float* x, float* hist, const float* w, float* y, size_t n, size_t C, const float* dyn, int head_dim) {
    const double scale = 1.0 / std::sqrt((double)((dyn ? 2 : 1) * kConvTaps + 1));
    const size_t nh = dyn ? C / (size_t)head_dim : 0;
    std::vector<double> ext((n + 3) * C);  // ext[(t+3)*C + c]; first 3 rows = history (index 0 = oldest)
    for (size_t c = 0; c < C; ++c)
        for (int i = 0; i < 3; ++i) ext[(size_t)i * C + c] = hist[c * 3 + i];
    for (size_t t = 0; t < n; ++t)
        for (size_t c = 0; c < C; ++c) ext[(t + 3) * C + c] = x[t * C + c];
    for (size_t t = 0; t < n; ++t)
        for (size_t c = 0; c < C; ++c) {
            double s = 0.0;
            for (int i = 0; i < kConvTaps; ++i) s += ext[(t + i) * C + c] * (double)w[c * kConvTaps + i];
            if (dyn)
                for (int lag = 0; lag < kConvTaps; ++lag)
                    s += ext[(t + 3 - lag) * C + c] * (double)dyn[t * nh * kConvTaps + (c / (size_t)head_dim) * kConvTaps + lag];
            y[t * C + c] = (float)((ext[(t + 3) * C + c] + s) * scale);
        }
    for (size_t c = 0; c < C; ++c)
        for (int i = 0; i < 3; ++i) hist[c * 3 + i] = (float)ext[(n + i) * C + c];
}

void attn_decode(const float* K, const float* V, int64_t pos, bool swa, const float* rel_bias, const float* q, const float* r,
                 const float* sink, const float* gate, float tau, float* out) {
    const int64_t lo = swa ? std::max<int64_t>(0, pos - (kSwaWindow - 1)) : 0;
    const size_t nk = (size_t)(pos - lo + 1);
    std::vector<double> s(nk);
    const double kq = 1.0 / std::sqrt((double)kHeadDim);
    for (int h = 0; h < kNQHeads; ++h) {
        const int g = h / kGqa;
        for (size_t jj = 0; jj < nk; ++jj) {
            const int64_t j = lo + (int64_t)jj;
            const float* kr = K + (size_t)j * kNKvHeads * kHeadDim + (size_t)g * kHeadDim;
            double dot = 0.0;
            for (int d = 0; d < kHeadDim; ++d) dot += (double)q[h * kHeadDim + d] * (double)tau * (double)kr[d];
            double sc = dot * kq;
            const int64_t dist = pos - j;
            if (dist < kRelExtent) {
                double b = 0.0;
                for (int f = 0; f < kRelFeat; ++f) b += (double)rel_bias[(size_t)dist * kRelFeat + f] * (double)r[h * kRelFeat + f] * (double)tau;
                sc += b;
            }
            s[jj] = sc;
        }
        double m = sink[h];
        for (size_t jj = 0; jj < nk; ++jj) m = std::max(m, s[jj]);
        double den = std::exp((double)sink[h] - m);
        std::vector<double> acc(kHeadDim, 0.0);
        for (size_t jj = 0; jj < nk; ++jj) {
            const double p = std::exp(s[jj] - m);
            den += p;
            const float* vr = V + (size_t)(lo + (int64_t)jj) * kNKvHeads * kHeadDim + (size_t)g * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d) acc[d] += p * (double)vr[d];
        }
        for (int d = 0; d < kHeadDim; ++d) {
            double v = acc[d] / den;
            if (gate) v *= 1.0 / (1.0 + std::exp(-(double)gate[h * kHeadDim + d]));
            out[h * kHeadDim + d] = (float)v;
        }
    }
}

void router_select(const float* score, int n, int k, int* ids) {
    std::vector<int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return score[a] > score[b]; });  // ggml cmp_argsort<DESC>
    for (int i = 0; i < k; ++i) ids[i] = idx[i];
}

void router_topk(const float* logits, const float* bias, int n_expert, int k, float route_scale, float eps, int* ids, float* weights) {
    std::vector<float> p(n_expert), sel(n_expert);
    for (int i = 0; i < n_expert; ++i) {
        p[i] = 1.0f / (1.0f + expf(-logits[i]));  // ggml_vec_sigmoid_f32
        sel[i] = bias ? p[i] + bias[i] : p[i];
    }
    router_select(sel.data(), n_expert, k, ids);
    double s = 0.0;
    for (int i = 0; i < k; ++i) s += (double)p[ids[i]];
    const float denom = (float)s + eps;
    for (int i = 0; i < k; ++i) weights[i] = (p[ids[i]] / denom) * route_scale;
}

void depth_mix(const float* const* states, int ns, const float* queries, int nq, float* const* outs, size_t n, float scale, float eps) {
    std::vector<std::vector<double>> nrm(ns, std::vector<double>(n));
    for (int s = 0; s < ns; ++s) {
        double ss = 0.0;
        for (size_t i = 0; i < n; ++i) ss += (double)states[s][i] * states[s][i];
        const double r = 1.0 / std::sqrt(ss / (double)n + (double)eps);
        for (size_t i = 0; i < n; ++i) nrm[s][i] = (double)states[s][i] * r;
    }
    for (int q = 0; q < nq; ++q) {
        std::vector<double> sc(ns);
        double mx = -1e300;
        for (int s = 0; s < ns; ++s) {
            double d = 0.0;
            for (size_t i = 0; i < n; ++i) d += nrm[s][i] * (double)queries[(size_t)q * n + i];
            sc[s] = d * (double)scale;
            mx = std::max(mx, sc[s]);
        }
        double sum = 0.0;
        for (int s = 0; s < ns; ++s) { sc[s] = std::exp(sc[s] - mx); sum += sc[s]; }
        for (size_t i = 0; i < n; ++i) {
            double o = 0.0;
            for (int s = 0; s < ns; ++s) o += sc[s] / sum * nrm[s][i];
            outs[q][i] = (float)o;
        }
    }
}

void ctrl_gate_input(const float* cur, const float* msg, float* gate_in, size_t n, float eps) {
    rmsnorm(cur, nullptr, gate_in, n, eps);
    rmsnorm(msg, nullptr, gate_in + n, n, eps);
}

void ctrl_apply(const float* cur, const float* msg, const float* gain_pre, const float* forget_pre, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const double g = std::tanh((double)gain_pre[i]), f = std::tanh((double)forget_pre[i]);
        out[i] = (float)(((double)cur[i] - f * (double)cur[i]) + g * (double)msg[i]);
    }
}

}  // namespace ref
}  // namespace ink

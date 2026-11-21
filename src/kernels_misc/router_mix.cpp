// Router top-k, attention-residual depth mix, loop controller vector ops.  Owner: agent B2.
#include <cmath>
#include <cstring>
#include "ink/kernels_misc.h"
#include "ink/kernels_misc_simd.h"

namespace ink {
using namespace simd;

namespace {
inline float exp_s(float x) { return _mm_cvtss_f32(_mm512_castps512_ps128(exp16(_mm512_set1_ps(x)))); }

// fixed-order dot product: 4 accumulator vectors over 64-element strides, then 16-element strides, then a masked tail
inline float dot4(const float* a, const float* b, size_t n) {
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), a3);
    }
    for (; i + 16 <= n; i += 16) a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), a0);
    if (i < n) {
        const __mmask16 m = tail_mask(n - i);
        a1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i), a1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
}
}  // namespace

// ---------------------------------------------------------------- router
void router_select(const float* score, int n, int k, int* ids) {
    constexpr float kNegInf = -__builtin_inff();
    alignas(64) float buf[128];
    for (int i = 0; i < 128; ++i) buf[i] = i < n ? score[i] : kNegInf;
    __m512 v[8];
    for (int j = 0; j < 8; ++j) v[j] = _mm512_load_ps(buf + 16 * j);
    for (int it = 0; it < k; ++it) {
        __m512 mx = _mm512_max_ps(_mm512_max_ps(_mm512_max_ps(v[0], v[1]), _mm512_max_ps(v[2], v[3])),
                                  _mm512_max_ps(_mm512_max_ps(v[4], v[5]), _mm512_max_ps(v[6], v[7])));
        const __m512 vm = _mm512_set1_ps(_mm512_reduce_max_ps(mx));
        for (int j = 0; j < 8; ++j) {
            const __mmask16 e = _mm512_cmp_ps_mask(v[j], vm, _CMP_EQ_OQ);
            if (e) {
                const int lane = __builtin_ctz((unsigned)e);
                ids[it] = 16 * j + lane;
                v[j] = _mm512_mask_mov_ps(v[j], (__mmask16)(1u << lane), _mm512_set1_ps(kNegInf));
                break;
            }
        }
    }
}

void router_topk(const float* logits, const float* bias, int n_expert, int k, float route_scale, float eps, int* ids,
                 float* weights) {
    alignas(64) float p[128], sel[128];
    vsigmoid(logits, p, (size_t)n_expert);
    if (bias)
        vadd(p, bias, sel, (size_t)n_expert);
    else
        std::memcpy(sel, p, sizeof(float) * (size_t)n_expert);
    router_select(sel, n_expert, k, ids);
    double s = 0.0;
    for (int i = 0; i < k; ++i) s += (double)p[ids[i]];
    const float denom = (float)s + eps;
    for (int i = 0; i < k; ++i) weights[i] = (p[ids[i]] / denom) * route_scale;
}

// ---------------------------------------------------------------- depth mix
void depth_mix(const float* const* states, int ns, const float* queries, int nq, float* const* outs, size_t n, float scale,
               float eps) {
    constexpr int kMaxS = 8, kMaxQ = 4;
    if (ns <= 0 || nq <= 0 || ns > kMaxS || nq > kMaxQ || n == 0) return;
    float r[kMaxS];
    float score[kMaxQ][kMaxS];
    // pass 1: per state sum of squares and dots with every query (4 independent accumulators per reduction, fixed order)
    for (int s = 0; s < ns; ++s) {
        const float* x = states[s];
        r[s] = 1.0f / sqrtf(dot4(x, x, n) / (float)n + eps);
        for (int q = 0; q < nq; ++q) score[q][s] = (dot4(x, queries + (size_t)q * n, n) * r[s]) * scale;
    }
    // softmax over states, coefficient on the RAW state = w_s * r_s
    float coef[kMaxQ][kMaxS];
    for (int q = 0; q < nq; ++q) {
        float mx = score[q][0];
        for (int s = 1; s < ns; ++s) mx = fmaxf(mx, score[q][s]);
        float e[kMaxS], sum = 0.0f;
        for (int s = 0; s < ns; ++s) {
            e[s] = exp_s(score[q][s] - mx);
            sum += e[s];
        }
        for (int s = 0; s < ns; ++s) coef[q][s] = (e[s] / sum) * r[s];
    }
    // pass 2: out_q = sum_s coef[q][s] * state_s, states summed in ascending order (states stay L1/L2 resident)
    for (int q = 0; q < nq; ++q) {
        __m512 cv[kMaxS];
        for (int s = 0; s < ns; ++s) cv[s] = _mm512_set1_ps(coef[q][s]);
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 acc = _mm512_mul_ps(cv[0], _mm512_loadu_ps(states[0] + i));
            for (int s = 1; s < ns; ++s) acc = _mm512_fmadd_ps(cv[s], _mm512_loadu_ps(states[s] + i), acc);
            _mm512_storeu_ps(outs[q] + i, acc);
        }
        if (i < n) {
            const __mmask16 m = tail_mask(n - i);
            __m512 acc = _mm512_mul_ps(cv[0], _mm512_maskz_loadu_ps(m, states[0] + i));
            for (int s = 1; s < ns; ++s) acc = _mm512_fmadd_ps(cv[s], _mm512_maskz_loadu_ps(m, states[s] + i), acc);
            _mm512_mask_storeu_ps(outs[q] + i, m, acc);
        }
    }
}

// ---------------------------------------------------------------- loop controller
void ctrl_gate_input(const float* cur, const float* msg, float* gate_in, size_t n, float eps) {
    rmsnorm(cur, nullptr, gate_in, n, eps);
    rmsnorm(msg, nullptr, gate_in + n, n, eps);
}

void ctrl_apply(const float* cur, const float* msg, const float* gain_pre, const float* forget_pre, float* out, size_t n) {
    for (size_t i = 0; i < n; i += 16) {
        const __mmask16 m = (i + 16 <= n) ? (__mmask16)0xFFFF : tail_mask(n - i);
        const __m512 c = _mm512_maskz_loadu_ps(m, cur + i);
        const __m512 g = tanh16(_mm512_maskz_loadu_ps(m, gain_pre + i));
        const __m512 f = tanh16(_mm512_maskz_loadu_ps(m, forget_pre + i));
        const __m512 t = _mm512_sub_ps(c, _mm512_mul_ps(f, c));
        _mm512_mask_storeu_ps(out + i, m, _mm512_add_ps(t, _mm512_mul_ps(g, _mm512_maskz_loadu_ps(m, msg + i))));
    }
}

}  // namespace ink

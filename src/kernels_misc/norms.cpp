// RMSNorm (+Q8_0 fused), L2 clip, V norm, converters and fused elementwise helpers.  Owner: agent B2.
#include <cmath>
#include "ink/kernels_misc.h"
#include "ink/kernels_misc_simd.h"

namespace ink {
using namespace simd;

namespace {
// fixed-order sum of squares: 4 accumulator vectors over 64-element strides, then 16-element strides, then a masked tail
inline float sumsq_f32(const float* x, size_t n) {
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        const __m512 v0 = _mm512_loadu_ps(x + i), v1 = _mm512_loadu_ps(x + i + 16);
        const __m512 v2 = _mm512_loadu_ps(x + i + 32), v3 = _mm512_loadu_ps(x + i + 48);
        a0 = _mm512_fmadd_ps(v0, v0, a0);
        a1 = _mm512_fmadd_ps(v1, v1, a1);
        a2 = _mm512_fmadd_ps(v2, v2, a2);
        a3 = _mm512_fmadd_ps(v3, v3, a3);
    }
    for (; i + 16 <= n; i += 16) {
        const __m512 v = _mm512_loadu_ps(x + i);
        a0 = _mm512_fmadd_ps(v, v, a0);
    }
    if (i < n) {
        const __m512 v = _mm512_maskz_loadu_ps(tail_mask(n - i), x + i);
        a1 = _mm512_fmadd_ps(v, v, a1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
}

inline void q8_block(__m512 y0, __m512 y1, block_q8_0* o, Q8Mode mode) {
    const float amax = _mm512_reduce_max_ps(_mm512_max_ps(abs16(y0), abs16(y1)));
    const float d = amax / 127.0f;
    o->d = (uint16_t)_cvtss_sh(d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512i q0, q1;
    if (mode == Q8Mode::Ref) {
        const float id = d ? 1.0f / d : 0.0f;
        const __m512 vid = _mm512_set1_ps(id);
        q0 = round_half_away_i32(_mm512_mul_ps(y0, vid));
        q1 = round_half_away_i32(_mm512_mul_ps(y1, vid));
    } else {
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        const __m512 vid = _mm512_set1_ps(id);
        q0 = _mm512_cvtps_epi32(_mm512_mul_ps(y0, vid));  // default MXCSR rounding = nearest even
        q1 = _mm512_cvtps_epi32(_mm512_mul_ps(y1, vid));
    }
    _mm_storeu_si128((__m128i*)o->qs, _mm512_cvtepi32_epi8(q0));
    _mm_storeu_si128((__m128i*)(o->qs + 16), _mm512_cvtepi32_epi8(q1));
}
}  // namespace

float rms_scale(const float* x, size_t n, float eps) {
    if (n == 0) return 0.0f;
    return 1.0f / sqrtf(sumsq_f32(x, n) / (float)n + eps);
}

void rmsnorm_apply(const float* x, const float* w, float s, float* y, size_t i0, size_t i1) {
    const __m512 vs = _mm512_set1_ps(s);
    size_t i = i0;
    if (w) {
        for (; i + 16 <= i1; i += 16)
            _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + i), vs), _mm512_loadu_ps(w + i)));
        if (i < i1) {
            const __mmask16 m = tail_mask(i1 - i);
            _mm512_mask_storeu_ps(y + i, m, _mm512_mul_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(m, x + i), vs), _mm512_maskz_loadu_ps(m, w + i)));
        }
    } else {
        for (; i + 16 <= i1; i += 16) _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), vs));
        if (i < i1) {
            const __mmask16 m = tail_mask(i1 - i);
            _mm512_mask_storeu_ps(y + i, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, x + i), vs));
        }
    }
}

void rmsnorm(const float* x, const float* w, float* y, size_t n, float eps) {
    if (n == 0) return;
    rmsnorm_apply(x, w, rms_scale(x, n, eps), y, 0, n);
}

void rmsnorm_q8_0(const float* x, const float* w, block_q8_0* out, size_t n, float eps, Q8Mode mode) {
    if (n == 0) return;
    rmsnorm_apply_q8_0(x, w, rms_scale(x, n, eps), out, 0, n / 32, mode);
}

void rmsnorm_apply_q8_0(const float* x, const float* w, float s, block_q8_0* out, size_t b0, size_t b1, Q8Mode mode) {
    const __m512 vs = _mm512_set1_ps(s);
    for (size_t b = b0; b < b1; ++b) {
        __m512 y0 = _mm512_mul_ps(_mm512_loadu_ps(x + 32 * b), vs);
        __m512 y1 = _mm512_mul_ps(_mm512_loadu_ps(x + 32 * b + 16), vs);
        if (w) {
            y0 = _mm512_mul_ps(y0, _mm512_loadu_ps(w + 32 * b));
            y1 = _mm512_mul_ps(y1, _mm512_loadu_ps(w + 32 * b + 16));
        }
        q8_block(y0, y1, out + b, mode);
    }
}

void vsitu_q8_0(const float* gate, const float* up, block_q8_0* out, size_t n, float beta, float lin, Q8Mode mode) {
    const __m512 b = _mm512_set1_ps(beta), ib = _mm512_set1_ps(1.0f / beta);
    const __m512 l = _mm512_set1_ps(lin), il = _mm512_set1_ps(1.0f / (lin > 0.0f ? lin : 1.0f));
    for (size_t k = 0; k < n / 32; ++k) {
        __m512 y[2];
        for (int h = 0; h < 2; ++h) {
            const __m512 g = _mm512_loadu_ps(gate + 32 * k + 16 * h), u = _mm512_loadu_ps(up + 32 * k + 16 * h);
            if (lin > 0.0f) {
                y[h] = situ16(g, u, b, ib, l, il);
            } else {
                const __m512 act = _mm512_mul_ps(_mm512_mul_ps(b, tanh16(_mm512_mul_ps(g, ib))), sigmoid16(g));
                y[h] = _mm512_mul_ps(act, u);
            }
        }
        q8_block(y[0], y[1], out + k, mode);
    }
}

void quantize_q8_0(const float* x, block_q8_0* out, size_t n, Q8Mode mode) {
    for (size_t b = 0; b < n / 32; ++b) q8_block(_mm512_loadu_ps(x + 32 * b), _mm512_loadu_ps(x + 32 * b + 16), out + b, mode);
}

void l2clip_heads(float* x, const float* chan_scale, int n_heads, int dim, float threshold, float post_scale) {
    for (int h = 0; h < n_heads; ++h) {
        float* xh = x + (size_t)h * dim;
        const float* sh = chan_scale ? chan_scale + (size_t)h * dim : nullptr;
        __m512 acc = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= (size_t)dim; i += 16) {
            __m512 v = _mm512_loadu_ps(xh + i);
            if (sh) {
                v = _mm512_mul_ps(v, _mm512_loadu_ps(sh + i));
                _mm512_storeu_ps(xh + i, v);
            }
            acc = _mm512_fmadd_ps(v, v, acc);
        }
        if (i < (size_t)dim) {
            const __mmask16 m = tail_mask(dim - i);
            __m512 v = _mm512_maskz_loadu_ps(m, xh + i);
            if (sh) {
                v = _mm512_mul_ps(v, _mm512_maskz_loadu_ps(m, sh + i));
                _mm512_mask_storeu_ps(xh + i, m, v);
            }
            acc = _mm512_fmadd_ps(v, v, acc);
        }
        const float norm = sqrtf(_mm512_reduce_add_ps(acc)) + 1e-6f;
        float f = threshold / norm;
        f = fminf(fmaxf(f, 0.0f), 1.0f);
        const __m512 vf = _mm512_set1_ps(f), vp = _mm512_set1_ps(post_scale);
        const bool post = post_scale != 1.0f;
        for (i = 0; i + 16 <= (size_t)dim; i += 16) {
            __m512 v = _mm512_mul_ps(_mm512_loadu_ps(xh + i), vf);
            if (post) v = _mm512_mul_ps(v, vp);
            _mm512_storeu_ps(xh + i, v);
        }
        if (i < (size_t)dim) {
            const __mmask16 m = tail_mask(dim - i);
            __m512 v = _mm512_mul_ps(_mm512_maskz_loadu_ps(m, xh + i), vf);
            if (post) v = _mm512_mul_ps(v, vp);
            _mm512_mask_storeu_ps(xh + i, m, v);
        }
    }
}

void vnorm_heads(const float* x, float* y, int n_heads, int dim, float eps) {
    for (int h = 0; h < n_heads; ++h) rmsnorm(x + (size_t)h * dim, nullptr, y + (size_t)h * dim, (size_t)dim, eps);
}

// ------------------------------------------------------------------ converters / fused helpers
namespace {
template <class F>
inline __attribute__((always_inline)) void loop_n(size_t n, F f) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) f(i, (__mmask16)0xFFFF);
    if (i < n) f(i, tail_mask(n - i));
}
}  // namespace

void bf16_to_f32(const uint16_t* x, float* y, size_t n) {
    loop_n(n, [&](size_t i, __mmask16 m) {
        _mm512_mask_storeu_ps(y + i, m, bf16x16_to_f32(_mm256_maskz_loadu_epi16(m, x + i)));
    });
}
void f16_to_f32(const uint16_t* x, float* y, size_t n) {
    loop_n(n, [&](size_t i, __mmask16 m) {
        _mm512_mask_storeu_ps(y + i, m, _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(m, x + i)));
    });
}
void f32_to_f16(const float* x, uint16_t* y, size_t n) {
    loop_n(n, [&](size_t i, __mmask16 m) {
        _mm256_mask_storeu_epi16(y + i, m, f32x16_to_f16(_mm512_maskz_loadu_ps(m, x + i)));
    });
}
void f32_to_bf16(const float* x, uint16_t* y, size_t n) {
    loop_n(n, [&](size_t i, __mmask16 m) {
        _mm256_mask_storeu_epi16(y + i, m, f32x16_to_bf16(_mm512_maskz_loadu_ps(m, x + i)));
    });
}
void vadd(const float* a, const float* b, float* y, size_t n) {
    loop_n(n, [&](size_t i, __mmask16 m) {
        _mm512_mask_storeu_ps(y + i, m, _mm512_add_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i)));
    });
}
void vmul(const float* a, const float* b, float* y, size_t n) {
    loop_n(n, [&](size_t i, __mmask16 m) {
        _mm512_mask_storeu_ps(y + i, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i)));
    });
}
void vscale(const float* x, float s, float* y, size_t n) {
    const __m512 vs = _mm512_set1_ps(s);
    loop_n(n, [&](size_t i, __mmask16 m) { _mm512_mask_storeu_ps(y + i, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, x + i), vs)); });
}
void vaxpy(float* y, float a, const float* x, size_t n) {
    const __m512 va = _mm512_set1_ps(a);
    loop_n(n, [&](size_t i, __mmask16 m) {
        _mm512_mask_storeu_ps(y + i, m, _mm512_fmadd_ps(va, _mm512_maskz_loadu_ps(m, x + i), _mm512_maskz_loadu_ps(m, y + i)));
    });
}
void vadd_scale(const float* a, const float* b, float s, float* y, size_t n) {
    const __m512 vs = _mm512_set1_ps(s);
    loop_n(n, [&](size_t i, __mmask16 m) {
        _mm512_mask_storeu_ps(y + i, m, _mm512_mul_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i)), vs));
    });
}
void vmul_sigmoid(const float* x, const float* g, float* y, size_t n) {
    loop_n(n, [&](size_t i, __mmask16 m) {
        _mm512_mask_storeu_ps(y + i, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, x + i), sigmoid16(_mm512_maskz_loadu_ps(m, g + i))));
    });
}

}  // namespace ink

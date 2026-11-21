// Short causal depthwise convolution (kernel 4) with residual scaling, static and input-dependent taps.  Owner: agent B2.
#include <cmath>
#include "ink/kernels_misc.h"
#include "ink/kernels_misc_simd.h"

namespace ink {
using namespace simd;

void sconv_repack_static(const float* w_ggml, float* w_soa, size_t C) {
    for (size_t c = 0; c < C; ++c)
        for (int i = 0; i < kConvTaps; ++i) w_soa[(size_t)i * C + c] = w_ggml[c * kConvTaps + i];
}

namespace {
// one 16-channel vector: inputs p0..p2 = x[t-3], x[t-2], x[t-1], x0 = x[t]; returns (x0 + conv) * scale
inline __attribute__((always_inline)) __m512 conv16(__m512 p0, __m512 p1, __m512 p2, __m512 x0, const float* w_soa, size_t C,
                                                    size_t c, __mmask16 m, const float* dyn, int head_dim, float scale) {
    const __m512 w0 = _mm512_maskz_loadu_ps(m, w_soa + c);
    const __m512 w1 = _mm512_maskz_loadu_ps(m, w_soa + C + c);
    const __m512 w2 = _mm512_maskz_loadu_ps(m, w_soa + 2 * C + c);
    const __m512 w3 = _mm512_maskz_loadu_ps(m, w_soa + 3 * C + c);
    __m512 s = _mm512_mul_ps(p0, w0);  // ssm_conv order: oldest tap first
    s = _mm512_fmadd_ps(p1, w1, s);
    s = _mm512_fmadd_ps(p2, w2, s);
    s = _mm512_fmadd_ps(x0, w3, s);
    if (dyn) {
        const float* d = dyn + (c / (size_t)head_dim) * kConvTaps;  // 16 channels never straddle a head (head_dim % 16 == 0)
        s = _mm512_fmadd_ps(x0, _mm512_set1_ps(d[0]), s);
        s = _mm512_fmadd_ps(p2, _mm512_set1_ps(d[1]), s);
        s = _mm512_fmadd_ps(p1, _mm512_set1_ps(d[2]), s);
        s = _mm512_fmadd_ps(p0, _mm512_set1_ps(d[3]), s);
    }
    return _mm512_mul_ps(_mm512_add_ps(x0, s), _mm512_set1_ps(scale));
}
inline float conv_scale(bool dynamic) { return 1.0f / sqrtf((float)((dynamic ? 2 : 1) * kConvTaps + 1)); }
}  // namespace

void sconv_step(const float* x, float* hist, const float* w_soa, float* y, size_t C, size_t c0, size_t c1, const float* dyn,
                int head_dim) {
    const float scale = conv_scale(dyn != nullptr);
    for (size_t c = c0; c < c1; c += 16) {
        const __mmask16 m = (c + 16 <= c1) ? (__mmask16)0xFFFF : tail_mask(c1 - c);
        const __m512 p0 = _mm512_maskz_loadu_ps(m, hist + c);
        const __m512 p1 = _mm512_maskz_loadu_ps(m, hist + C + c);
        const __m512 p2 = _mm512_maskz_loadu_ps(m, hist + 2 * C + c);
        const __m512 x0 = _mm512_maskz_loadu_ps(m, x + c);
        const __m512 out = conv16(p0, p1, p2, x0, w_soa, C, c, m, dyn, head_dim, scale);
        _mm512_mask_storeu_ps(y + c, m, out);
        _mm512_mask_storeu_ps(hist + c, m, p1);
        _mm512_mask_storeu_ps(hist + C + c, m, p2);
        _mm512_mask_storeu_ps(hist + 2 * C + c, m, x0);
    }
}

void sconv_seq(const float* x, float* hist, const float* w_soa, float* y, size_t n, size_t C, size_t c0, size_t c1,
               const float* dyn, int head_dim) {
    if (n == 0) return;
    const float scale = conv_scale(dyn != nullptr);
    const size_t nh = dyn ? C / (size_t)head_dim : 0;
    for (size_t c = c0; c < c1; c += 16) {
        const __mmask16 m = (c + 16 <= c1) ? (__mmask16)0xFFFF : tail_mask(c1 - c);
        // sliding registers: p0,p1,p2 = inputs t-3,t-2,t-1
        __m512 p0 = _mm512_maskz_loadu_ps(m, hist + c);
        __m512 p1 = _mm512_maskz_loadu_ps(m, hist + C + c);
        __m512 p2 = _mm512_maskz_loadu_ps(m, hist + 2 * C + c);
        for (size_t t = 0; t < n; ++t) {
            const __m512 x0 = _mm512_maskz_loadu_ps(m, x + t * C + c);
            const float* dt = dyn ? dyn + t * nh * kConvTaps : nullptr;
            _mm512_mask_storeu_ps(y + t * C + c, m, conv16(p0, p1, p2, x0, w_soa, C, c, m, dt, head_dim, scale));
            p0 = p1;
            p1 = p2;
            p2 = x0;
        }
        _mm512_mask_storeu_ps(hist + c, m, p0);
        _mm512_mask_storeu_ps(hist + C + c, m, p1);
        _mm512_mask_storeu_ps(hist + 2 * C + c, m, p2);
    }
}

}  // namespace ink

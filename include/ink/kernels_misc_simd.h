// AVX-512 (F/BW/DQ/VL + FMA + F16C only) lane-level building blocks for fused phases.  Owner: agent B2.
// Everything here is header-only and always_inline so the executor can fuse several of these into one loop.
// Accuracy (vs double, measured by tests/kernels_misc/test_kernels_misc.cpp, see docs/KERNELS_MISC.md):
//   exp16     x in [-87.3, 88.7]            max rel err ~2.5e-7   (inputs outside are clamped: exp(-inf) -> 1.2e-38, exp(+big) -> 3e38)
//   tanh16    all finite x                  max rel err ~3e-7     (saturates to +-1 exactly for |x| > ~9)
//   sigmoid16 x >= -87.3 (any larger x ok)  max rel err ~3e-7     (below -87.3: abs err < 1.2e-38)
// NaN inputs are NOT supported (they map to a finite clamp value, never trap).
#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

#define INK_AI static inline __attribute__((always_inline))

namespace ink {
namespace simd {

INK_AI __mmask16 tail_mask(size_t rem) { return (__mmask16)((1u << rem) - 1u); }  // rem in [1,15]

INK_AI __m512 abs16(__m512 x) {
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0x7fffffff)));
}

// exp(x), Cephes-style: x = n*ln2 + r, |r| <= ln2/2, degree-6 polynomial, scale by 2^n with vscalefps.
INK_AI __m512 exp16(__m512 x) {
    x = _mm512_min_ps(x, _mm512_set1_ps(88.7f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.3f));
    const __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                           _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), r);
    const __m512 z = _mm512_mul_ps(r, r);
    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201E-1f));
    y = _mm512_fmadd_ps(y, z, r);
    y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(y, fx);
}

// 1/d with ~1 ulp: vrcp14ps + one Newton step. d must be finite and not tiny/denormal; result for |d| > 2^126 is 0.
INK_AI __m512 rcp16(__m512 d) {
    const __m512 r = _mm512_rcp14_ps(d);
    const __m512 e = _mm512_fnmadd_ps(d, r, _mm512_set1_ps(1.0f));
    return _mm512_fmadd_ps(r, e, r);
}

// tanh: odd minimax polynomial for |x| < 0.625 (rel err 4.4e-9), (1-e)/(1+e), e = exp(-2|x|) otherwise.
INK_AI __m512 tanh16(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 ax = abs16(x);
    // large branch
    const __m512 axc = _mm512_min_ps(ax, _mm512_set1_ps(20.0f));
    const __m512 e = exp16(_mm512_mul_ps(axc, _mm512_set1_ps(-2.0f)));
    __m512 t = _mm512_mul_ps(_mm512_sub_ps(one, e), rcp16(_mm512_add_ps(one, e)));
    t = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(t),
                                            _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32((int)0x80000000))));
    // small branch: x + x*z*P(z), z = x^2
    const __m512 z = _mm512_mul_ps(x, x);
    __m512 p = _mm512_set1_ps(-0.0057047391310334206f);
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(0.020638886839151382f));
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(-0.05373966693878174f));
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(0.13331441581249237f));
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(-0.3333328068256378f));
    p = _mm512_fmadd_ps(_mm512_mul_ps(p, z), x, x);
    const __mmask16 small = _mm512_cmp_ps_mask(ax, _mm512_set1_ps(0.625f), _CMP_LT_OQ);
    return _mm512_mask_blend_ps(small, t, p);
}

INK_AI __m512 sigmoid16(__m512 x) {
    const __m512 e = exp16(_mm512_sub_ps(_mm512_setzero_ps(), x));
    return rcp16(_mm512_add_ps(_mm512_set1_ps(1.0f), e));
}

// cap * tanh(x / cap); inv_cap = 1.0f/cap computed by the caller (same op order as ggml_scale/ggml_tanh/ggml_scale)
INK_AI __m512 softcap16(__m512 x, __m512 cap, __m512 inv_cap) {
    return _mm512_mul_ps(cap, tanh16(_mm512_mul_ps(x, inv_cap)));
}

// SiTU as llama.cpp defines it: beta*tanh(g/beta)*sigmoid(g) * lin*tanh(u/lin)  (lin <= 0 means: up passes through)
INK_AI __m512 situ16(__m512 g, __m512 u, __m512 beta, __m512 inv_beta, __m512 lin, __m512 inv_lin) {
    const __m512 act = _mm512_mul_ps(_mm512_mul_ps(beta, tanh16(_mm512_mul_ps(g, inv_beta))), sigmoid16(g));
    const __m512 up = _mm512_mul_ps(lin, tanh16(_mm512_mul_ps(u, inv_lin)));
    return _mm512_mul_ps(act, up);
}

INK_AI __m512 bf16x16_to_f32(__m256i v) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
}
// round-to-nearest-even f32 -> bf16 for 16 lanes (NaN handling: payload preserved as quiet NaN)
INK_AI __m256i f32x16_to_bf16(__m512 v) {
    const __m512i u = _mm512_castps_si512(v);
    const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
    const __m512i r = _mm512_srli_epi32(_mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7fff)), lsb), 16);
    const __mmask16 isnan = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
    const __m512i q = _mm512_or_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(0x40));
    return _mm512_cvtepi32_epi16(_mm512_mask_blend_epi32(isnan, r, q));
}
INK_AI __m256i f32x16_to_f16(__m512 v) { return _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }

// round-half-away-from-zero to int32 (== roundf), valid for |v| < 2^22
INK_AI __m512i round_half_away_i32(__m512 v) {
    const __m512i sign = _mm512_and_si512(_mm512_castps_si512(v), _mm512_set1_epi32((int)0x80000000));
    const __m512 h = _mm512_castsi512_ps(_mm512_or_si512(sign, _mm512_set1_epi32(0x3EFFFFFF)));  // +-0.49999997f
    return _mm512_cvttps_epi32(_mm512_add_ps(v, h));
}

}  // namespace simd
}  // namespace ink

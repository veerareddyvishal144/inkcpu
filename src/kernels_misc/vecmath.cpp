// Pointwise vector math: exp, tanh, sigmoid, SiTU, softcap.  Owner: agent B2.
#include "ink/kernels_misc.h"
#include "ink/kernels_misc_simd.h"

namespace ink {
using namespace simd;

namespace {
template <class F>
inline __attribute__((always_inline)) void map1(const float* x, float* y, size_t n, F f) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512 a = _mm512_loadu_ps(x + i);
        __m512 b = _mm512_loadu_ps(x + i + 16);
        a = f(a);
        b = f(b);
        _mm512_storeu_ps(y + i, a);
        _mm512_storeu_ps(y + i + 16, b);
    }
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(y + i, f(_mm512_loadu_ps(x + i)));
    if (i < n) {
        const __mmask16 m = tail_mask(n - i);
        _mm512_mask_storeu_ps(y + i, m, f(_mm512_maskz_loadu_ps(m, x + i)));
    }
}
template <class F>
inline __attribute__((always_inline)) void map2(const float* a, const float* b, float* y, size_t n, F f) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(y + i, f(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    if (i < n) {
        const __mmask16 m = tail_mask(n - i);
        _mm512_mask_storeu_ps(y + i, m, f(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i)));
    }
}
}  // namespace

void vexp(const float* x, float* y, size_t n) { map1(x, y, n, [](__m512 v) { return exp16(v); }); }
void vtanh(const float* x, float* y, size_t n) { map1(x, y, n, [](__m512 v) { return tanh16(v); }); }
void vsigmoid(const float* x, float* y, size_t n) { map1(x, y, n, [](__m512 v) { return sigmoid16(v); }); }

void vsoftcap(const float* x, float* y, size_t n, float cap) {
    const __m512 c = _mm512_set1_ps(cap);
    const __m512 ic = _mm512_set1_ps(1.0f / cap);
    map1(x, y, n, [&](__m512 v) { return softcap16(v, c, ic); });
}

void vsitu(const float* gate, const float* up, float* y, size_t n, float beta, float lin) {
    const __m512 b = _mm512_set1_ps(beta), ib = _mm512_set1_ps(1.0f / beta);
    if (lin > 0.0f) {
        const __m512 l = _mm512_set1_ps(lin), il = _mm512_set1_ps(1.0f / lin);
        map2(gate, up, y, n, [&](__m512 g, __m512 u) { return situ16(g, u, b, ib, l, il); });
    } else {
        map2(gate, up, y, n, [&](__m512 g, __m512 u) {
            const __m512 act = _mm512_mul_ps(_mm512_mul_ps(b, tanh16(_mm512_mul_ps(g, ib))), sigmoid16(g));
            return _mm512_mul_ps(act, u);
        });
    }
}

void vsitu_fused(const float* gu, float* y, size_t n, float beta, float lin) { vsitu(gu, gu + n, y, n, beta, lin); }

}  // namespace ink

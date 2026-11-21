// BF16 x F32 and F32 x F32 batched small GEMMs, bit-identical to ink::gemv_bf16 / gemv_f32 per token. Owner: agent G.
// Same lane structure as B1: two 16-lane accumulators (a0: 32-steps first half and the 16-tail, a1: second half), fmadd per step,
// y = hsum16(a0 + a1). Register blocking: RB weight rows x TB tokens (2 x 6 = 24 accumulator pairs... 2*RB*TB zmm).
#include <immintrin.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "ink/kernels_gemm.h"

namespace ink {
namespace {

inline float hsum16(__m512 v) {
    const __m256 t = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
    const __m128 u = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
    __m128 w = _mm_add_ps(u, _mm_movehl_ps(u, u));
    w = _mm_add_ss(w, _mm_shuffle_ps(w, w, 1));
    return _mm_cvtss_f32(w);
}

template <bool BF>
inline __m512 ldw(const char* p) {
    if constexpr (BF)
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
    else
        return _mm512_loadu_ps((const float*)p);
}

template <bool BF, int RB, int TB>
void tile(const char* w, size_t rs, const float* x, size_t ldx, int k, float* y, size_t ldy) {
    constexpr size_t E = BF ? 2 : 4;
    __m512 a0[RB][TB], a1[RB][TB];
    const float* xt[TB];
    for (int t = 0; t < TB; ++t) xt[t] = x + (size_t)t * ldx;
    for (int r = 0; r < RB; ++r)
        for (int t = 0; t < TB; ++t) a0[r][t] = a1[r][t] = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= k; i += 32) {
        for (int r = 0; r < RB; ++r) {
            const char* wp = w + (size_t)r * rs + (size_t)i * E;
            const __m512 w0 = ldw<BF>(wp), w1 = ldw<BF>(wp + 16 * E);
            for (int t = 0; t < TB; ++t) {
                a0[r][t] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(xt[t] + i), a0[r][t]);
                a1[r][t] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(xt[t] + i + 16), a1[r][t]);
            }
        }
    }
    if (i < k) {
        for (int r = 0; r < RB; ++r) {
            const __m512 w0 = ldw<BF>(w + (size_t)r * rs + (size_t)i * E);
            for (int t = 0; t < TB; ++t) a0[r][t] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(xt[t] + i), a0[r][t]);
        }
    }
    for (int r = 0; r < RB; ++r)
        for (int t = 0; t < TB; ++t) y[(size_t)t * ldy + r] = hsum16(_mm512_add_ps(a0[r][t], a1[r][t]));
}

template <bool BF, int RB>
inline void tile_tb(int tb, const char* w, size_t rs, const float* x, size_t ldx, int k, float* y, size_t ldy) {
    switch (tb) {
        case 1: tile<BF, RB, 1>(w, rs, x, ldx, k, y, ldy); break;
        case 2: tile<BF, RB, 2>(w, rs, x, ldx, k, y, ldy); break;
        case 3: tile<BF, RB, 3>(w, rs, x, ldx, k, y, ldy); break;
        case 4: tile<BF, RB, 4>(w, rs, x, ldx, k, y, ldy); break;
        case 5: tile<BF, RB, 5>(w, rs, x, ldx, k, y, ldy); break;
        default: tile<BF, RB, 6>(w, rs, x, ldx, k, y, ldy); break;
    }
}

template <bool BF>
void gemm_small(const void* w, size_t rs, const float* x, size_t ldx, int nt, int k, float* y, size_t ldy, int nrows) {
    if (nrows <= 0 || nt <= 0) return;
    if (k <= 0 || k % 16) {
        std::fprintf(stderr, "ink::gemm_%s: unsupported K=%d\n", BF ? "bf16" : "f32", k);
        std::abort();
    }
    for (int r0 = 0; r0 < nrows; r0 += 2) {
        const int rb = std::min(2, nrows - r0);
        const char* wp = (const char*)w + (size_t)r0 * rs;
        for (int t0 = 0; t0 < nt; t0 += 6) {
            const int tb = std::min(6, nt - t0);
            const float* xp = x + (size_t)t0 * ldx;
            float* yp = y + (size_t)t0 * ldy + r0;
            if (rb == 2)
                tile_tb<BF, 2>(tb, wp, rs, xp, ldx, k, yp, ldy);
            else
                tile_tb<BF, 1>(tb, wp, rs, xp, ldx, k, yp, ldy);
        }
    }
}

}  // namespace

void gemm_bf16(const void* w, size_t rs, const float* x, size_t ldx, int nt, int k, float* y, size_t ldy, int nrows) {
    gemm_small<true>(w, rs, x, ldx, nt, k, y, ldy, nrows);
}
void gemm_f32(const void* w, size_t rs, const float* x, size_t ldx, int nt, int k, float* y, size_t ldy, int nrows) {
    gemm_small<false>(w, rs, x, ldx, nt, k, y, ldy, nrows);
}

}  // namespace ink

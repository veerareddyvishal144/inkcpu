// Attention of the DSpark block pass (agent S1). 24 queries (8 slots x 3 heads of one kv group) against the cached ctx keys of one layer
// (B2 KvLayer layout: K transposed [kv head][d 128][key 64] fp16, V [kv head][key 64][d 128] fp16) + the 8 block keys (fp32, outside the cache).
// No mask except the live-key count, no bias / sink / gate / softcap (docs/DSPARK.md 4.2). Online softmax per 64-key block; key splits give
// independent partial states (m, l, acc) that attn_finish_query merges (deterministic: the split count depends only on the number of blocks).
#include <algorithm>
#include <cmath>
#include <cstring>

#include "drafter_impl.h"

namespace ink {
namespace dr {

namespace {

constexpr float kNeg = -1e30f;

template <int NQ>
INK_AI void qk_tile(const uint16_t* kT, const float* q, float* sc, int col) {
    __m512 a[NQ];
#pragma GCC unroll 24
    for (int i = 0; i < NQ; ++i) a[i] = _mm512_setzero_ps();
    for (int d = 0; d < HD; ++d) {
        const __m512 kv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(kT + (size_t)d * 64)));
#pragma GCC unroll 24
        for (int i = 0; i < NQ; ++i) a[i] = _mm512_fmadd_ps(kv, _mm512_set1_ps(q[i * HD + d]), a[i]);
    }
#pragma GCC unroll 24
    for (int i = 0; i < NQ; ++i) _mm512_storeu_ps(sc + i * 64 + col, a[i]);
}

INK_AI __mmask16 lane_mask(int valid, int v) {   // lanes of 16-key vector v that are live
    const int r = valid - v * 16;
    return r >= 16 ? (__mmask16)0xFFFF : (r <= 0 ? (__mmask16)0 : (__mmask16)((1u << r) - 1u));
}

template <int NQ>
void blocks_t(const KvLayer& L, int g, int b0, int b1, int64_t nvalid, const float* q, AttnPartial2* out) {
    alignas(64) float sc[NQ * 64];
    alignas(64) float vf[64 * HD];
    alignas(64) float acc[NQ * HD];
    float m[NQ], l[NQ];
    for (int i = 0; i < NQ; ++i) m[i] = kNeg, l[i] = 0.0f;
    std::memset(acc, 0, sizeof(acc));
    for (int b = b0; b < b1; ++b) {
        const int64_t base = (int64_t)b * 64;
        const int valid = (int)std::min<int64_t>(64, nvalid - base);
        if (valid <= 0) break;
        const uint16_t* kT = L.k + (size_t)b * kKvBlockElems + (size_t)g * 8192;
        const uint16_t* vb = L.v + (size_t)b * kKvBlockElems + (size_t)g * 8192;
        const int nt = (valid + 15) / 16;
        for (int t = 0; t < nt; ++t) qk_tile<NQ>(kT + t * 16, q, sc, t * 16);
        // online softmax update
        for (int i = 0; i < NQ; ++i) {
            __m512 s[4];
            __m512 mx = _mm512_set1_ps(kNeg);
            for (int v = 0; v < nt; ++v) {
                s[v] = _mm512_loadu_ps(sc + i * 64 + v * 16);
                mx = _mm512_max_ps(mx, _mm512_mask_blend_ps(lane_mask(valid, v), _mm512_set1_ps(kNeg), s[v]));
            }
            const float mnew = std::max(m[i], _mm512_reduce_max_ps(mx));
            const float alpha = std::exp(m[i] - mnew);
            const __m512 vm = _mm512_set1_ps(mnew);
            __m512 sum = _mm512_setzero_ps();
            for (int v = 0; v < nt; ++v) {
                const __m512 p = _mm512_maskz_mov_ps(lane_mask(valid, v), simd::exp16(_mm512_sub_ps(s[v], vm)));
                _mm512_storeu_ps(sc + i * 64 + v * 16, p);
                sum = _mm512_add_ps(sum, p);
            }
            l[i] = l[i] * alpha + _mm512_reduce_add_ps(sum);
            if (alpha != 1.0f) {
                const __m512 va = _mm512_set1_ps(alpha);
                for (int c = 0; c < 8; ++c) _mm512_storeu_ps(acc + i * HD + c * 16, _mm512_mul_ps(_mm512_loadu_ps(acc + i * HD + c * 16), va));
            }
            m[i] = mnew;
        }
        // V block -> fp32
        for (int j = 0; j < valid; ++j)
            for (int c = 0; c < 8; ++c) _mm512_storeu_ps(vf + j * HD + c * 16, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vb + (size_t)j * HD + c * 16))));
        // acc += P . V, 3 queries x 8 vectors per tile
        for (int qt = 0; qt < NQ / 3; ++qt) {
            __m512 a0[8], a1[8], a2[8];
            float* ap = acc + qt * 3 * HD;
            for (int c = 0; c < 8; ++c) {
                a0[c] = _mm512_loadu_ps(ap + c * 16);
                a1[c] = _mm512_loadu_ps(ap + HD + c * 16);
                a2[c] = _mm512_loadu_ps(ap + 2 * HD + c * 16);
            }
            const float* p0 = sc + (qt * 3) * 64;
            const float* p1 = p0 + 64;
            const float* p2 = p1 + 64;
            for (int j = 0; j < valid; ++j) {
                const __m512 b0v = _mm512_set1_ps(p0[j]), b1v = _mm512_set1_ps(p1[j]), b2v = _mm512_set1_ps(p2[j]);
#pragma GCC unroll 8
                for (int c = 0; c < 8; ++c) {
                    const __m512 v = _mm512_loadu_ps(vf + j * HD + c * 16);
                    a0[c] = _mm512_fmadd_ps(b0v, v, a0[c]);
                    a1[c] = _mm512_fmadd_ps(b1v, v, a1[c]);
                    a2[c] = _mm512_fmadd_ps(b2v, v, a2[c]);
                }
            }
            for (int c = 0; c < 8; ++c) {
                _mm512_storeu_ps(ap + c * 16, a0[c]);
                _mm512_storeu_ps(ap + HD + c * 16, a1[c]);
                _mm512_storeu_ps(ap + 2 * HD + c * 16, a2[c]);
            }
        }
    }
    for (int i = 0; i < 24; ++i) {
        out->m[i] = i < NQ ? m[i] : kNeg;
        out->l[i] = i < NQ ? l[i] : 0.0f;
    }
    std::memcpy(out->acc, acc, sizeof(acc));
}

INK_AI float dot128(const float* a, const float* b) {
    __m512 s = _mm512_mul_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
    for (int c = 1; c < 8; ++c) s = _mm512_fmadd_ps(_mm512_loadu_ps(a + c * 16), _mm512_loadu_ps(b + c * 16), s);
    return _mm512_reduce_add_ps(s);
}

}  // namespace

void attn_ctx_part(const KvLayer& L, int g, int b0, int b1, int64_t nvalid, const float* q, int nq, AttnPartial2* out) {
    switch (nq) {
        case 3: blocks_t<3>(L, g, b0, b1, nvalid, q, out); break;
        case 6: blocks_t<6>(L, g, b0, b1, nvalid, q, out); break;
        case 9: blocks_t<9>(L, g, b0, b1, nvalid, q, out); break;
        case 12: blocks_t<12>(L, g, b0, b1, nvalid, q, out); break;
        case 15: blocks_t<15>(L, g, b0, b1, nvalid, q, out); break;
        case 18: blocks_t<18>(L, g, b0, b1, nvalid, q, out); break;
        case 21: blocks_t<21>(L, g, b0, b1, nvalid, q, out); break;
        default: blocks_t<24>(L, g, b0, b1, nvalid, q, out); break;
    }
}

void attn_finish_query(const AttnPartial2* parts, int nparts, int i, const float* q, const float* kb, const float* vb, size_t kstride, int nkeys, float* out) {
    float sc[8], mb = kNeg;
    for (int k = 0; k < nkeys; ++k) {
        sc[k] = dot128(q, kb + k * kstride);
        mb = std::max(mb, sc[k]);
    }
    float M = mb;
    for (int sp = 0; sp < nparts; ++sp)
        if (parts[sp].l[i] > 0.0f) M = std::max(M, parts[sp].m[i]);
    float Lsum = 0.0f;
    __m512 acc[8];
    for (int c = 0; c < 8; ++c) acc[c] = _mm512_setzero_ps();
    for (int sp = 0; sp < nparts; ++sp) {
        if (!(parts[sp].l[i] > 0.0f)) continue;
        const float w = std::exp(parts[sp].m[i] - M);
        Lsum += w * parts[sp].l[i];
        const __m512 vw = _mm512_set1_ps(w);
        for (int c = 0; c < 8; ++c) acc[c] = _mm512_fmadd_ps(vw, _mm512_loadu_ps(parts[sp].acc + i * HD + c * 16), acc[c]);
    }
    const float wb = std::exp(mb - M);
    for (int k = 0; k < nkeys; ++k) {
        const float p = std::exp(sc[k] - mb) * wb;
        Lsum += p;
        const __m512 vp = _mm512_set1_ps(p);
        for (int c = 0; c < 8; ++c) acc[c] = _mm512_fmadd_ps(vp, _mm512_loadu_ps(vb + k * kstride + c * 16), acc[c]);
    }
    const __m512 inv = _mm512_set1_ps(1.0f / Lsum);
    for (int c = 0; c < 8; ++c) _mm512_storeu_ps(out + c * 16, _mm512_mul_ps(acc[c], inv));
}

// ---------------------------------------------------------------------------------------------------------------- RoPE (NEOX, theta 1e6, pairs (i, i+64))
void rope_table(int64_t pos, float* cs) {
    static const std::vector<double> inv = [] {
        std::vector<double> v(64);
        for (int i = 0; i < 64; ++i) v[i] = std::pow(1.0e6, -(double)i / 64.0);
        return v;
    }();
    for (int i = 0; i < 64; ++i) {
        const double a = (double)pos * inv[i];
        cs[i] = (float)std::cos(a);
        cs[64 + i] = (float)std::sin(a);
    }
}

void rope_head(float* x, const float* cs) {
    for (int c = 0; c < 4; ++c) {
        const __m512 a = _mm512_loadu_ps(x + c * 16), b = _mm512_loadu_ps(x + 64 + c * 16);
        const __m512 co = _mm512_loadu_ps(cs + c * 16), si = _mm512_loadu_ps(cs + 64 + c * 16);
        _mm512_storeu_ps(x + c * 16, _mm512_sub_ps(_mm512_mul_ps(a, co), _mm512_mul_ps(b, si)));
        _mm512_storeu_ps(x + 64 + c * 16, _mm512_add_ps(_mm512_mul_ps(b, co), _mm512_mul_ps(a, si)));
    }
}

}  // namespace dr
}  // namespace ink

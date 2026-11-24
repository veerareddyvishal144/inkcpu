// AVX-512 (F/BW/DQ only; NO VNNI/VBMI/BF16) GEMV kernels and Q8_0 activation quantiser. Owner: agent B1.
// Design notes are in docs/KERNELS_GEMV.md. Scalar references with identical accumulation order: gemv_ref.cpp.
#include <immintrin.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ink/kernels_gemv.h"
#include "ink/nvfp4_repack.h"

namespace ink {

int g_gemv_prefetch_bytes = 512;
int g_gemv_prefetch_hint = 0;
int g_gemv_nvfp4r_group = [] {   // rows per group of the nvfp4r kernel (4 or 8); env INK_NVFP4R_GROUP for A/B experiments
    const char* s = std::getenv("INK_NVFP4R_GROUP");
    const int v = s && *s ? std::atoi(s) : 8;
    return v == 4 ? 4 : 8;
}();
int g_gemv_nvfp4r_pre = [] {     // 1 (default): per-call precompute of the shared activation vectors (XP, C, DX) per super-block; env INK_NVFP4R_PRE=0 off
    const char* s = std::getenv("INK_NVFP4R_PRE");
    return s && *s ? (std::atoi(s) != 0) : 1;
}();

namespace {

alignas(64) float g_ue4m3_tab[256];  // exact copy of ue4m3_to_f32 for every byte value
struct Ue4m3Init {
    Ue4m3Init() {
        for (int i = 0; i < 256; ++i) g_ue4m3_tab[i] = ue4m3_to_f32((uint8_t)i);
    }
} g_ue4m3_init;

[[noreturn]] void bad_k(const char* fn, int k) {
    std::fprintf(stderr, "ink::%s: unsupported K=%d\n", fn, k);
    std::abort();
}

// fixed reduction tree (mirrored by hsum16_ref): lanes j+8, j+4, j+2, j+1
inline float hsum16(__m512 v) {
    const __m256 t = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
    const __m128 u = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
    __m128 w = _mm_add_ps(u, _mm_movehl_ps(u, u));
    w = _mm_add_ss(w, _mm_shuffle_ps(w, w, 1));
    return _mm_cvtss_f32(w);
}

template <int H>
inline void pf1(const void* p) {
    if constexpr (H == 0)
        _mm_prefetch((const char*)p, _MM_HINT_T0);
    else if constexpr (H == 1)
        _mm_prefetch((const char*)p, _MM_HINT_T1);
    else if constexpr (H == 2)
        _mm_prefetch((const char*)p, _MM_HINT_NTA);
}
template <int H>
inline void pf_head(const char* p, int bytes) {
    for (int o = 0; o < bytes; o += 64) pf1<H>(p + o);
}

// two 32-byte loads (unaligned) -> one zmm
inline __m512i ld2(const void* a, const void* b) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)a)),
                              _mm256_loadu_si256((const __m256i*)b), 1);
}

// ------------------------------------------------------------------ Q8_0 x Q8_0
// zmm = 2 blocks (64 weights). |w| (unsigned) x (x with the sign of w applied): maddubs -> int16 pair sums
// (|pair| <= 2*128*127 < 32768, no saturation), madd -> int32 lanes (lanes 0-7 = block 0, 8-15 = block 1),
// then acc[lane] = fma((float)I, dw*dx, acc[lane]).
inline __m512 q8_step(__m512 acc, __m512i W, __m512i X, __m128 s, __m512i ones16, __m512i idx01) {
    const __m512i AW = _mm512_abs_epi8(W);
    const __mmask64 neg = _mm512_movepi8_mask(W);
    const __m512i XS = _mm512_mask_sub_epi8(X, neg, _mm512_setzero_si512(), X);
    const __m512i I = _mm512_madd_epi16(_mm512_maddubs_epi16(AW, XS), ones16);
    const __m512 S = _mm512_permutexvar_ps(idx01, _mm512_castps128_ps512(s));
    return _mm512_fmadd_ps(_mm512_cvtepi32_ps(I), S, acc);
}

template <int R, int H>
void q8_rows(const char* w, size_t rs, const void* xv, int nb, float* y, int pf) {
    const block_q8_0* x = static_cast<const block_q8_0*>(xv);
    const __m512i ones16 = _mm512_set1_epi16(1);
    const __m512i idx01 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    __m512 acc[R];
    const char* wr[R];
    for (int r = 0; r < R; ++r) {
        acc[r] = _mm512_setzero_ps();
        wr[r] = w + (size_t)r * rs;
    }
    const int np = nb >> 1;
    for (int p = 0; p < np; ++p) {
        const block_q8_0* xb = x + 2 * p;
        const __m512i X = ld2(xb[0].qs, xb[1].qs);
        const __m128 dxv = _mm_setr_ps(_cvtsh_ss(xb[0].d), _cvtsh_ss(xb[1].d), 0.f, 0.f);
        const size_t off = (size_t)p * (2 * sizeof(block_q8_0));
        for (int r = 0; r < R; ++r) {
            const char* wp = wr[r] + off;
            pf1<H>(wp + pf);
            pf1<H>(wp + pf + 64);
            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(wp);
            const __m512i W = ld2(wb[0].qs, wb[1].qs);
            const __m128 dw = _mm_cvtph_ps(_mm_cvtsi32_si128((int)((uint32_t)wb[0].d | ((uint32_t)wb[1].d << 16))));
            acc[r] = q8_step(acc[r], W, X, _mm_mul_ps(dw, dxv), ones16, idx01);
        }
    }
    if (nb & 1) {  // odd block count: last block alone in the low half
        const block_q8_0* xb = x + (nb - 1);
        const __m512i X = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)xb->qs));
        const __m128 dxv = _mm_setr_ps(_cvtsh_ss(xb->d), 0.f, 0.f, 0.f);
        const size_t off = (size_t)(nb - 1) * sizeof(block_q8_0);
        for (int r = 0; r < R; ++r) {
            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(wr[r] + off);
            const __m512i W = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)wb->qs));
            const __m128 dw = _mm_cvtph_ps(_mm_cvtsi32_si128((int)(uint32_t)wb->d));
            acc[r] = q8_step(acc[r], W, X, _mm_mul_ps(dw, dxv), ones16, idx01);
        }
    }
    for (int r = 0; r < R; ++r) y[r] = hsum16(acc[r]);
}

// ------------------------------------------------------------------ NVFP4 x Q8_0
// One 36-byte super-block = 64 weights = 2 activation blocks. The 32 nibble bytes are broadcast to both 256-bit halves,
// the upper half is shifted right by 4 and everything masked with 0x0F, so the 64 nibbles sit in this byte order:
//   [lo(sb0) lo(sb1) lo(sb2) lo(sb3) hi(sb0) hi(sb1) hi(sb2) hi(sb3)]  (8 bytes each, sb = 16-weight sub-block)
// The (row independent) activation is permuted once per super-block to the same order. pshufb maps nibble -> 12+kvalue
// (unsigned, 0..24), maddubs/madd give sum((kv+12)*x) per 4-byte lane; subtracting C = sum(12*x) (row independent)
// leaves the exact integer sum(kv*x). Lane l carries sub-block (l&7)>>1, scaled by ue4m3(d[sb]) * dx(block sb>>1).
template <int R, int H>
void nvfp4_rows(const char* w, size_t rs, const void* xv, int nsb, float* y, int pf) {
    const block_q8_0* x = static_cast<const block_q8_0*>(xv);
    const __m512i TBL = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m512i M0F = _mm512_set1_epi8(0x0F);
    const __m512i SH = _mm512_setr_epi64(0, 0, 0, 0, 4, 4, 4, 4);
    const __m512i IDXQ = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    const __m512i ONES16 = _mm512_set1_epi16(1);
    const __m512i U12 = _mm512_set1_epi8(12);
    const __m512i IDXS = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3);
    __m512 acc[R];
    const char* wr[R];
    for (int r = 0; r < R; ++r) {
        acc[r] = _mm512_setzero_ps();
        wr[r] = w + (size_t)r * rs;
    }
    for (int sb = 0; sb < nsb; ++sb) {
        const block_q8_0* xb = x + 2 * sb;
        const __m512i XP = _mm512_permutexvar_epi64(IDXQ, ld2(xb[0].qs, xb[1].qs));
        const __m512i C = _mm512_madd_epi16(_mm512_maddubs_epi16(U12, XP), ONES16);
        const float dx0 = _cvtsh_ss(xb[0].d), dx1 = _cvtsh_ss(xb[1].d);
        const __m128 dxv = _mm_setr_ps(dx0, dx0, dx1, dx1);
        const size_t off = (size_t)sb * sizeof(block_nvfp4);
        for (int r = 0; r < R; ++r) {
            const char* wp = wr[r] + off;
            pf1<H>(wp + pf);
            const block_nvfp4* wb = reinterpret_cast<const block_nvfp4*>(wp);
            const __m512i Q = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)wb->qs));
            const __m512i N = _mm512_and_si512(_mm512_srlv_epi64(Q, SH), M0F);
            const __m512i Wu = _mm512_shuffle_epi8(TBL, N);
            const __m512i I = _mm512_sub_epi32(_mm512_madd_epi16(_mm512_maddubs_epi16(Wu, XP), ONES16), C);
            const __m128 sc = _mm_mul_ps(_mm_setr_ps(g_ue4m3_tab[wb->d[0]], g_ue4m3_tab[wb->d[1]], g_ue4m3_tab[wb->d[2]],
                                                     g_ue4m3_tab[wb->d[3]]),
                                         dxv);
            const __m512 S = _mm512_permutexvar_ps(IDXS, _mm512_castps128_ps512(sc));
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(I), S, acc[r]);
        }
    }
    for (int r = 0; r < R; ++r) y[r] = hsum16(acc[r]);
}

// ------------------------------------------------------------------ NVFP4 with RE-ENCODED scales ("nvfp4r") x Q8_0
// Same nibble / integer path as nvfp4_rows. The sub-block scale bytes are the re-encoded codes of include/ink/nvfp4_repack.h
// (engine arena after the load-time repack): for every group of up to 4 rows the 4 x 4 scale bytes of the current super-block are
// gathered into one xmm (vmovd + 3 vpinsrd, 4 loads), decoded with 3 vector uops (vpmovzxbd, vpslld 20, vmulps 2^116), multiplied
// ONCE by the activation scales [dx0 dx0 dx1 dx1] x 4 and expanded per row with ONE vpermps (lane l of row j <- element
// 4j + ((l & 7) >> 1)). That replaces 8 scalar loads + 3 vinsertps + vmulps + vpermps per row super-block.
// Bit-identity: code_f * 2^116 is exact (= ue4m3_to_f32 of the original byte, see the header), so S_l = fl(ue4m3 * dx) is the same
// float as in nvfp4_rows and gemv_nvfp4_ref; the integer sums, the fma order and the reduction tree are unchanged.
template <int R, int H>
void nvfp4r_rows(const char* w, size_t rs, const void* xv, int nsb, float* y, int pf) {
    static_assert(R == 1 || R == 2 || R == 4 || R == 8, "row group");
    const block_q8_0* x = static_cast<const block_q8_0*>(xv);
    const __m512i TBL = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m512i M0F = _mm512_set1_epi8(0x0F);
    const __m512i SH = _mm512_setr_epi64(0, 0, 0, 0, 4, 4, 4, 4);
    const __m512i IDXQ = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    const __m512i ONES16 = _mm512_set1_epi16(1);
    const __m512i U12 = _mm512_set1_epi8(12);
    const __m512i IDXDX = _mm512_setr_epi32(0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1);   // (dx0, dx1, ., .) -> [dx0 dx0 dx1 dx1] x 4
    const __m512i IDXR0 = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3);   // row j of a group: IDXR0 + 4j
    const __m512i IDXR1 = _mm512_add_epi32(IDXR0, _mm512_set1_epi32(4));
    const __m512i IDXR2 = _mm512_add_epi32(IDXR0, _mm512_set1_epi32(8));
    const __m512i IDXR3 = _mm512_add_epi32(IDXR0, _mm512_set1_epi32(12));
    constexpr int NG = (R + 3) / 4;
    __m512 acc[R];
    const char* wr[R];
    for (int r = 0; r < R; ++r) {
        acc[r] = _mm512_setzero_ps();
        wr[r] = w + (size_t)r * rs;
    }
    auto ld32 = [](const char* p) {
        int32_t v;
        std::memcpy(&v, p, 4);
        return v;
    };
    for (int sb = 0; sb < nsb; ++sb) {
        const block_q8_0* xb = x + 2 * sb;
        const __m512i XP = _mm512_permutexvar_epi64(IDXQ, ld2(xb[0].qs, xb[1].qs));
        const __m512i C = _mm512_madd_epi16(_mm512_maddubs_epi16(U12, XP), ONES16);
        const __m128 dx2 = _mm_cvtph_ps(_mm_cvtsi32_si128((int)((uint32_t)xb[0].d | ((uint32_t)xb[1].d << 16))));
        const __m512 DX = _mm512_permutexvar_ps(IDXDX, _mm512_castps128_ps512(dx2));
        const size_t off = (size_t)sb * sizeof(block_nvfp4);
        __m512 S16[NG];
        for (int g = 0; g < NG; ++g) {
            __m128i sc = _mm_cvtsi32_si128(ld32(wr[4 * g] + off));
            if (4 * g + 1 < R) sc = _mm_insert_epi32(sc, ld32(wr[4 * g + 1] + off), 1);
            if (4 * g + 2 < R) sc = _mm_insert_epi32(sc, ld32(wr[4 * g + 2] + off), 2);
            if (4 * g + 3 < R) sc = _mm_insert_epi32(sc, ld32(wr[4 * g + 3] + off), 3);
            S16[g] = _mm512_mul_ps(nvfp4r_decode16v(sc), DX);
        }
        for (int r = 0; r < R; ++r) {
            const char* wp = wr[r] + off;
            pf1<H>(wp + pf);
            const block_nvfp4* wb = reinterpret_cast<const block_nvfp4*>(wp);
            const __m512i Q = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)wb->qs));
            const __m512i N = _mm512_and_si512(_mm512_srlv_epi64(Q, SH), M0F);
            const __m512i Wu = _mm512_shuffle_epi8(TBL, N);
            const __m512i I = _mm512_sub_epi32(_mm512_madd_epi16(_mm512_maddubs_epi16(Wu, XP), ONES16), C);
            const __m512i idx = (r & 3) == 0 ? IDXR0 : (r & 3) == 1 ? IDXR1 : (r & 3) == 2 ? IDXR2 : IDXR3;
            const __m512 S = _mm512_permutexvar_ps(idx, S16[r >> 2]);
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(I), S, acc[r]);
        }
    }
    // fully unrolled so that acc[] stays in registers (a rolled epilogue made gcc store all R accumulators every super-block)
#pragma GCC unroll 8
    for (int r = 0; r < R; ++r) y[r] = hsum16(acc[r]);
}

// Variant with the per-super-block activation vectors precomputed once per call (nvfp4r_prepare_act): saves the XP / C / DX work
// (vinserti64x4, vpermq, vpmaddubsw, vpmaddwd, vcvtph2ps, vpermps ~ 8 ALU uops per group super-block) at the cost of 3 L1 loads.
struct alignas(64) NvAct {
    __m512i XP;   // activation bytes in the shuffled nibble order
    __m512i C;    // sum(12 * x) per int32 lane
    __m512 DX;    // [dx0 dx0 dx1 dx1] x 4
};
constexpr int kNvActMax = 72;   // K <= 4608

inline void nvfp4r_prepare_act(const block_q8_0* x, int nsb, NvAct* act) {
    const __m512i IDXQ = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    const __m512i ONES16 = _mm512_set1_epi16(1);
    const __m512i U12 = _mm512_set1_epi8(12);
    const __m512i IDXDX = _mm512_setr_epi32(0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1);
    for (int sb = 0; sb < nsb; ++sb) {
        const block_q8_0* xb = x + 2 * sb;
        const __m512i XP = _mm512_permutexvar_epi64(IDXQ, ld2(xb[0].qs, xb[1].qs));
        act[sb].XP = XP;
        act[sb].C = _mm512_madd_epi16(_mm512_maddubs_epi16(U12, XP), ONES16);
        const __m128 dx2 = _mm_cvtph_ps(_mm_cvtsi32_si128((int)((uint32_t)xb[0].d | ((uint32_t)xb[1].d << 16))));
        act[sb].DX = _mm512_permutexvar_ps(IDXDX, _mm512_castps128_ps512(dx2));
    }
}

template <int R, int H>
void nvfp4r_rows_pre(const char* w, size_t rs, const void* xv, int nsb, float* y, int pf) {
    static_assert(R == 1 || R == 2 || R == 4 || R == 8, "row group");
    const NvAct* act = static_cast<const NvAct*>(xv);
    const __m512i TBL = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m512i M0F = _mm512_set1_epi8(0x0F);
    const __m512i SH = _mm512_setr_epi64(0, 0, 0, 0, 4, 4, 4, 4);
    const __m512i ONES16 = _mm512_set1_epi16(1);
    const __m512i IDXR0 = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3);
    const __m512i IDXR1 = _mm512_add_epi32(IDXR0, _mm512_set1_epi32(4));
    const __m512i IDXR2 = _mm512_add_epi32(IDXR0, _mm512_set1_epi32(8));
    const __m512i IDXR3 = _mm512_add_epi32(IDXR0, _mm512_set1_epi32(12));
    constexpr int NG = (R + 3) / 4;
    __m512 acc[R];
    const char* wr[R];
    for (int r = 0; r < R; ++r) {
        acc[r] = _mm512_setzero_ps();
        wr[r] = w + (size_t)r * rs;
    }
    auto ld32 = [](const char* p) {
        int32_t v;
        std::memcpy(&v, p, 4);
        return v;
    };
    for (int sb = 0; sb < nsb; ++sb) {
        const __m512i XP = act[sb].XP;
        const __m512i C = act[sb].C;
        const __m512 DX = act[sb].DX;
        const size_t off = (size_t)sb * sizeof(block_nvfp4);
        __m512 S16[NG];
        for (int g = 0; g < NG; ++g) {
            __m128i sc = _mm_cvtsi32_si128(ld32(wr[4 * g] + off));
            if (4 * g + 1 < R) sc = _mm_insert_epi32(sc, ld32(wr[4 * g + 1] + off), 1);
            if (4 * g + 2 < R) sc = _mm_insert_epi32(sc, ld32(wr[4 * g + 2] + off), 2);
            if (4 * g + 3 < R) sc = _mm_insert_epi32(sc, ld32(wr[4 * g + 3] + off), 3);
            S16[g] = _mm512_mul_ps(nvfp4r_decode16v(sc), DX);
        }
        for (int r = 0; r < R; ++r) {
            const char* wp = wr[r] + off;
            pf1<H>(wp + pf);
            const block_nvfp4* wb = reinterpret_cast<const block_nvfp4*>(wp);
            const __m512i Q = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)wb->qs));
            const __m512i N = _mm512_and_si512(_mm512_srlv_epi64(Q, SH), M0F);
            const __m512i Wu = _mm512_shuffle_epi8(TBL, N);
            const __m512i I = _mm512_sub_epi32(_mm512_madd_epi16(_mm512_maddubs_epi16(Wu, XP), ONES16), C);
            const __m512i idx = (r & 3) == 0 ? IDXR0 : (r & 3) == 1 ? IDXR1 : (r & 3) == 2 ? IDXR2 : IDXR3;
            const __m512 S = _mm512_permutexvar_ps(idx, S16[r >> 2]);
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(I), S, acc[r]);
        }
    }
#pragma GCC unroll 8
    for (int r = 0; r < R; ++r) y[r] = hsum16(acc[r]);
}

// ------------------------------------------------------------------ BF16 x F32 and F32 x F32
inline __m512 bf16x16(const char* p) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
}

template <int R, int H>
void bf16_rows(const char* w, size_t rs, const void* xv, int k, float* y, int pf) {
    const float* x = static_cast<const float*>(xv);
    __m512 a0[R], a1[R];
    const char* wr[R];
    for (int r = 0; r < R; ++r) {
        a0[r] = a1[r] = _mm512_setzero_ps();
        wr[r] = w + (size_t)r * rs;
    }
    int i = 0;
    for (; i + 32 <= k; i += 32) {
        const __m512 x0 = _mm512_loadu_ps(x + i), x1 = _mm512_loadu_ps(x + i + 16);
        for (int r = 0; r < R; ++r) {
            const char* wp = wr[r] + (size_t)i * 2;
            pf1<H>(wp + pf);
            a0[r] = _mm512_fmadd_ps(bf16x16(wp), x0, a0[r]);
            a1[r] = _mm512_fmadd_ps(bf16x16(wp + 32), x1, a1[r]);
        }
    }
    if (i < k) {
        const __m512 x0 = _mm512_loadu_ps(x + i);
        for (int r = 0; r < R; ++r) a0[r] = _mm512_fmadd_ps(bf16x16(wr[r] + (size_t)i * 2), x0, a0[r]);
    }
    for (int r = 0; r < R; ++r) y[r] = hsum16(_mm512_add_ps(a0[r], a1[r]));
}

template <int R, int H>
void f32_rows(const char* w, size_t rs, const void* xv, int k, float* y, int pf) {
    const float* x = static_cast<const float*>(xv);
    __m512 a0[R], a1[R];
    const char* wr[R];
    for (int r = 0; r < R; ++r) {
        a0[r] = a1[r] = _mm512_setzero_ps();
        wr[r] = w + (size_t)r * rs;
    }
    int i = 0;
    for (; i + 32 <= k; i += 32) {
        const __m512 x0 = _mm512_loadu_ps(x + i), x1 = _mm512_loadu_ps(x + i + 16);
        for (int r = 0; r < R; ++r) {
            const char* wp = wr[r] + (size_t)i * 4;
            pf1<H>(wp + pf);
            pf1<H>(wp + pf + 64);
            a0[r] = _mm512_fmadd_ps(_mm512_loadu_ps((const float*)wp), x0, a0[r]);
            a1[r] = _mm512_fmadd_ps(_mm512_loadu_ps((const float*)(wp + 64)), x1, a1[r]);
        }
    }
    if (i < k) {
        const __m512 x0 = _mm512_loadu_ps(x + i);
        for (int r = 0; r < R; ++r) a0[r] = _mm512_fmadd_ps(_mm512_loadu_ps((const float*)(wr[r] + (size_t)i * 4)), x0, a0[r]);
    }
    for (int r = 0; r < R; ++r) y[r] = hsum16(_mm512_add_ps(a0[r], a1[r]));
}

// ------------------------------------------------------------------ row-group driver (4 rows, then 2, then 1)
typedef void (*RowsFn)(const char* w, size_t rs, const void* x, int n, float* y, int pf);

// Prefetch policy: the first `pf` bytes of every row of the NEXT 4-row group are requested while the current group is
// computed (whole-group lead time); inside the kernels every row stream prefetches `pf` bytes ahead of its read pointer.
template <int H>
inline void drive(const char* w, size_t rs, size_t rowbytes, int nrows, float* y, const void* x, int n, RowsFn f4,
                  RowsFn f2, RowsFn f1, int pf) {
    const int hb = (int)std::min<size_t>((size_t)std::max(pf, 0), rowbytes);
    const int first = nrows < 4 ? nrows : 4;
    for (int i = 0; i < first; ++i) pf_head<H>(w + (size_t)i * rs, hb);
    int r = 0;
    for (; r + 4 <= nrows; r += 4) {
        const int nxt = std::min(4, nrows - (r + 4));
        for (int i = 0; i < nxt; ++i) pf_head<H>(w + (size_t)(r + 4 + i) * rs, hb);
        f4(w + (size_t)r * rs, rs, x, n, y + r, pf);
    }
    int rem = nrows - r;
    if (rem >= 2) {
        f2(w + (size_t)r * rs, rs, x, n, y + r, pf);
        r += 2;
        rem -= 2;
    }
    if (rem == 1) f1(w + (size_t)r * rs, rs, x, n, y + r, pf);
}

#define INK_DRIVE(ROWS, RB, N)                                                                                   \
    switch (g_gemv_prefetch_hint) {                                                                              \
        case 1: drive<1>((const char*)w, rs, RB, nrows, y, x, N, ROWS<4, 1>, ROWS<2, 1>, ROWS<1, 1>, pf); break; \
        case 2: drive<2>((const char*)w, rs, RB, nrows, y, x, N, ROWS<4, 2>, ROWS<2, 2>, ROWS<1, 2>, pf); break; \
        case 3: drive<3>((const char*)w, rs, RB, nrows, y, x, N, ROWS<4, 3>, ROWS<2, 3>, ROWS<1, 3>, pf); break; \
        default: drive<0>((const char*)w, rs, RB, nrows, y, x, N, ROWS<4, 0>, ROWS<2, 0>, ROWS<1, 0>, pf); break; \
    }

// 8-row groups (nvfp4r): the next group's row heads are prefetched while the current group runs; tails use 4, 2, 1 rows.
template <int H>
inline void drive8(const char* w, size_t rs, size_t rowbytes, int nrows, float* y, const void* x, int n, RowsFn f8, RowsFn f4,
                   RowsFn f2, RowsFn f1, int pf) {
    const int hb = (int)std::min<size_t>((size_t)std::max(pf, 0), rowbytes);
    const int first = nrows < 8 ? nrows : 8;
    for (int i = 0; i < first; ++i) pf_head<H>(w + (size_t)i * rs, hb);
    int r = 0;
    for (; r + 8 <= nrows; r += 8) {
        const int nxt = std::min(8, nrows - (r + 8));
        for (int i = 0; i < nxt; ++i) pf_head<H>(w + (size_t)(r + 8 + i) * rs, hb);
        f8(w + (size_t)r * rs, rs, x, n, y + r, pf);
    }
    int rem = nrows - r;
    if (rem >= 4) {
        f4(w + (size_t)r * rs, rs, x, n, y + r, pf);
        r += 4;
        rem -= 4;
    }
    if (rem >= 2) {
        f2(w + (size_t)r * rs, rs, x, n, y + r, pf);
        r += 2;
        rem -= 2;
    }
    if (rem == 1) f1(w + (size_t)r * rs, rs, x, n, y + r, pf);
}

#define INK_DRIVE8(ROWS, RB, N)                                                                                               \
    switch (g_gemv_prefetch_hint) {                                                                                           \
        case 1: drive8<1>((const char*)w, rs, RB, nrows, y, x, N, ROWS<8, 1>, ROWS<4, 1>, ROWS<2, 1>, ROWS<1, 1>, pf); break; \
        case 2: drive8<2>((const char*)w, rs, RB, nrows, y, x, N, ROWS<8, 2>, ROWS<4, 2>, ROWS<2, 2>, ROWS<1, 2>, pf); break; \
        case 3: drive8<3>((const char*)w, rs, RB, nrows, y, x, N, ROWS<8, 3>, ROWS<4, 3>, ROWS<2, 3>, ROWS<1, 3>, pf); break; \
        default: drive8<0>((const char*)w, rs, RB, nrows, y, x, N, ROWS<8, 0>, ROWS<4, 0>, ROWS<2, 0>, ROWS<1, 0>, pf); break; \
    }

}  // namespace

// ------------------------------------------------------------------ public single-matrix kernels
void gemv_q8_0(const void* w, size_t rs, const block_q8_0* x, int k, float* y, int nrows) {
    if (nrows <= 0) return;
    if (k <= 0 || k % QK8_0) bad_k("gemv_q8_0", k);
    const int pf = g_gemv_prefetch_bytes;
    INK_DRIVE(q8_rows, row_bytes_q8_0(k), k / QK8_0)
}

void gemv_nvfp4(const void* w, size_t rs, const block_q8_0* x, int k, float* y, int nrows) {
    if (nrows <= 0) return;
    if (k <= 0 || k % QK_NVFP4) bad_k("gemv_nvfp4", k);
    const int pf = g_gemv_prefetch_bytes;
    INK_DRIVE(nvfp4_rows, row_bytes_nvfp4(k), k / QK_NVFP4)
}

namespace {
void gemv_nvfp4r_pre(const void* w, size_t rs, const NvAct* x, int k, float* y, int nrows) {
    const int pf = g_gemv_prefetch_bytes;
    if (g_gemv_nvfp4r_group == 8) {
        INK_DRIVE8(nvfp4r_rows_pre, row_bytes_nvfp4(k), k / QK_NVFP4)
    } else {
        INK_DRIVE(nvfp4r_rows_pre, row_bytes_nvfp4(k), k / QK_NVFP4)
    }
}
}  // namespace

void gemv_nvfp4r(const void* w, size_t rs, const block_q8_0* x, int k, float* y, int nrows) {
    if (nrows <= 0) return;
    if (k <= 0 || k % QK_NVFP4) bad_k("gemv_nvfp4r", k);
    const int nsb = k / QK_NVFP4;
    if (g_gemv_nvfp4r_pre && nrows >= 8 && nsb <= kNvActMax) {   // the precompute pays off from one 8-row group on
        alignas(64) NvAct act[kNvActMax];
        nvfp4r_prepare_act(x, nsb, act);
        gemv_nvfp4r_pre(w, rs, act, k, y, nrows);
        return;
    }
    const int pf = g_gemv_prefetch_bytes;
    if (g_gemv_nvfp4r_group == 8) {
        INK_DRIVE8(nvfp4r_rows, row_bytes_nvfp4(k), nsb)
    } else {
        INK_DRIVE(nvfp4r_rows, row_bytes_nvfp4(k), nsb)
    }
}

void gemv_bf16(const void* w, size_t rs, const float* x, int k, float* y, int nrows) {
    if (nrows <= 0) return;
    if (k <= 0 || k % 16) bad_k("gemv_bf16", k);
    const int pf = g_gemv_prefetch_bytes;
    INK_DRIVE(bf16_rows, row_bytes_bf16(k), k)
}

void gemv_f32(const void* w, size_t rs, const float* x, int k, float* y, int nrows) {
    if (nrows <= 0) return;
    if (k <= 0 || k % 16) bad_k("gemv_f32", k);
    const int pf = g_gemv_prefetch_bytes;
    INK_DRIVE(f32_rows, row_bytes_f32(k), k)
}

// ------------------------------------------------------------------ fused multi-matrix entry points
namespace {
template <class Fn>
inline void multi_walk(const GemvMat* m, int nm, int64_t vr0, int64_t vr1, Fn fn) {
    int64_t base = 0;
    for (int i = 0; i < nm && base < vr1; ++i) {
        const int64_t n = m[i].nrows;
        const int64_t lo = std::max(vr0, base), hi = std::min(vr1, base + n);
        if (lo < hi) {
            const int64_t off = lo - base;
            fn((const char*)m[i].w + (size_t)off * m[i].row_stride, m[i].row_stride, m[i].x, m[i].y + off, (int)(hi - lo));
        }
        base += n;
    }
}
}  // namespace

void gemv_multi_q8_0(const GemvMat* m, int nm, int k, int64_t vr0, int64_t vr1) {
    multi_walk(m, nm, vr0, vr1, [&](const void* w, size_t rs, const void* x, float* y, int n) {
        gemv_q8_0(w, rs, (const block_q8_0*)x, k, y, n);
    });
}
void gemv_multi_nvfp4(const GemvMat* m, int nm, int k, int64_t vr0, int64_t vr1) {
    multi_walk(m, nm, vr0, vr1, [&](const void* w, size_t rs, const void* x, float* y, int n) {
        gemv_nvfp4(w, rs, (const block_q8_0*)x, k, y, n);
    });
}
void gemv_multi_nvfp4r(const GemvMat* m, int nm, int k, int64_t vr0, int64_t vr1) {
    multi_walk(m, nm, vr0, vr1, [&](const void* w, size_t rs, const void* x, float* y, int n) {
        gemv_nvfp4r(w, rs, (const block_q8_0*)x, k, y, n);
    });
}
void gemv_multi_bf16(const GemvMat* m, int nm, int k, int64_t vr0, int64_t vr1) {
    multi_walk(m, nm, vr0, vr1,
               [&](const void* w, size_t rs, const void* x, float* y, int n) { gemv_bf16(w, rs, (const float*)x, k, y, n); });
}
void gemv_multi_f32(const GemvMat* m, int nm, int k, int64_t vr0, int64_t vr1) {
    multi_walk(m, nm, vr0, vr1,
               [&](const void* w, size_t rs, const void* x, float* y, int n) { gemv_f32(w, rs, (const float*)x, k, y, n); });
}

// ------------------------------------------------------------------ helpers
void split_rows(int64_t n, int nth, int tid, int64_t align, int64_t* begin, int64_t* end) {
    if (align < 1) align = 1;
    const int64_t units = (n + align - 1) / align;
    const int64_t per = units / nth, extra = units % nth;
    const int64_t u0 = (int64_t)tid * per + std::min<int64_t>(tid, extra);
    const int64_t u1 = u0 + per + (tid < extra ? 1 : 0);
    *begin = std::min(u0 * align, n);
    *end = std::min(u1 * align, n);
}

void* alloc_weight_buffer(size_t bytes, bool hugepage) {
    const size_t al = hugepage ? ((size_t)2 << 20) : (size_t)4096;
    const size_t sz = (bytes + al - 1) / al * al;
    void* p = std::aligned_alloc(al, sz);
    if (p && hugepage) madvise(p, sz, MADV_HUGEPAGE);
    return p;
}
void free_weight_buffer(void* p) { std::free(p); }

// ------------------------------------------------------------------ Q8_0 activation quantiser (bit-exact with the ref)
namespace {
// roundf(): halves away from zero (cvtps2dq would round halves to even)
inline __m512 round_away(__m512 v) {
    const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    const __m512 signbit = _mm512_castsi512_ps(_mm512_set1_epi32((int)0x80000000));
    const __m512 t = _mm512_roundscale_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m512 diff = _mm512_sub_ps(v, t);  // exact
    const __mmask16 k = _mm512_cmp_ps_mask(_mm512_and_ps(diff, absmask), _mm512_set1_ps(0.5f), _CMP_GE_OQ);
    const __m512 one_signed = _mm512_or_ps(_mm512_set1_ps(1.0f), _mm512_and_ps(v, signbit));
    return _mm512_mask_add_ps(t, k, t, one_signed);
}
}  // namespace

void quantize_row_q8_0(const float* x, block_q8_0* y, int k) {
    if (k % QK8_0) bad_k("quantize_row_q8_0", k);
    const int nb = k / QK8_0;
    const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    for (int i = 0; i < nb; ++i) {
        const __m512 v0 = _mm512_loadu_ps(x + 32 * i), v1 = _mm512_loadu_ps(x + 32 * i + 16);
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_and_ps(v0, absmask), _mm512_and_ps(v1, absmask)));
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = f32_to_fp16(d);
        const __m512 idv = _mm512_set1_ps(id);
        const __m512i i0 = _mm512_cvttps_epi32(round_away(_mm512_mul_ps(v0, idv)));
        const __m512i i1 = _mm512_cvttps_epi32(round_away(_mm512_mul_ps(v1, idv)));
        _mm_storeu_si128((__m128i*)(y[i].qs), _mm512_cvtepi32_epi8(i0));
        _mm_storeu_si128((__m128i*)(y[i].qs + 16), _mm512_cvtepi32_epi8(i1));
    }
}

}  // namespace ink

// Small-n multi-activation GEMV kernels (Q8_0 / NVFP4 / NVFP4 re-encoded scales x N <= 8 Q8_0 activations). Owner: agent K2 (wave 1).
// API and numerics contract: include/ink/kernels_gemv_multix.h. Design: docs/status/K2.md (wave 1) / docs/KERNELS_GEMV.md.
//
// Structure (per call): the N activations are lane-permuted once into a thread_local scratch (int8 quants + float block scales, the GEMM's
// physical lane order: Q8_0 dword d = (block d&1, group d>>1); NVFP4 dword d = (sub-block d&3, group d>>2)). Then the rows are streamed in
// groups of R = 8 / 4 / 2 rows (N <= 2 / <= 4 / <= 8, so that R x N <= 16 zmm accumulators) with GEMV-style software prefetch; per 64-weight chunk
// the R rows are decoded in registers (|w| or |kv| bytes, sign mask, scale vector pre-broadcast) and applied to the N activations:
//   XS = x sign-transferred under the weight sign mask, I = madd(maddubs(|w|, XS), 1) = exact int sums of 4 products per lane,
//   acc[r][j] = fma((float)I, fl(dw * dx), acc[r][j])                     (dw * dx = ONE float multiply of a broadcast row scale and a broadcast activation scale)
// and y = hsum16(vpermps(unperm, acc)): identical float operations and reduction tree as gemv_q8_0 / gemv_nvfp4 (lanes only live in another
// physical order until the final permutation), hence bit-identical results.
#include <immintrin.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ink/kernels_gemv.h"
#include "ink/kernels_gemv_multix.h"
#include "ink/nvfp4_repack.h"

namespace ink {

namespace {

constexpr int KMAX = 8192;
constexpr int MX = kMultixMaxX;
constexpr int FMT_Q8 = 0, FMT_NV = 1;
constexpr int ENC_GGUF = 0, ENC_R = 1;

alignas(64) float g_tab[256];  // exact copy of ue4m3_to_f32 for every byte value (same table as gemv.cpp / gemm.cpp)
struct TabInit {
    TabInit() {
        for (int i = 0; i < 256; ++i) g_tab[i] = ue4m3_to_f32((uint8_t)i);
    }
} g_tab_init;

[[noreturn]] void bad_arg(const char* fn, const char* what, long v) {
    std::fprintf(stderr, "ink::%s: unsupported %s=%ld\n", fn, what, v);
    std::abort();
}

// fixed reduction tree (same as gemv.cpp hsum16): lanes j+8, j+4, j+2, j+1
inline float hsum16(__m512 v) {
    const __m256 t = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
    const __m128 u = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
    __m128 w = _mm_add_ps(u, _mm_movehl_ps(u, u));
    w = _mm_add_ss(w, _mm_shuffle_ps(w, w, 1));
    return _mm_cvtss_f32(w);
}

inline __m512i ld2(const void* a, const void* b) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)a)), _mm256_loadu_si256((const __m256i*)b), 1);
}

// ---- lane permutations (identical to gemm.cpp so that the scale vector is a plain broadcast)
inline __m512i idx_q8() { return _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15); }      // both W and X
inline __m512i idx_nv_x() { return _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15); }  // activation dwords
inline __m512i idx_nv_w() { return _mm512_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7); }       // broadcast nibble bytes
template <int FMT>
inline __m512i unperm() {  // logical (gemv.cpp) lane l <- physical lane idx[l]
    if constexpr (FMT == FMT_Q8)
        return _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    else
        return _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
}

inline void pf1(const char* p, int hint) {
    switch (hint) {
        case 0: _mm_prefetch(p, _MM_HINT_T0); break;
        case 1: _mm_prefetch(p, _MM_HINT_T1); break;
        case 2: _mm_prefetch(p, _MM_HINT_NTA); break;
        default: break;
    }
}
inline void pf_head(const char* p, int bytes, int hint) {
    if (hint == 3) return;
    for (int o = 0; o < bytes; o += 64) pf1(p + o, hint);
}

// ---- activation scratch (one call): lane-permuted int8 quants + float scales per activation
struct alignas(64) XScratch {
    alignas(64) int8_t xq[MX][KMAX];
    alignas(64) float dx[MX][KMAX / 16 + 16];  // Q8_0: 2 floats per chunk (K/32); NVFP4: 4 per chunk (K/16, dx of the block duplicated per sub-block)
};
XScratch& xscratch() {
    static thread_local XScratch s;
    return s;
}

template <int FMT>
inline void pack_x(const block_q8_0* xb, int nc, int8_t* q, float* d) {
    const __m512i IDX = FMT == FMT_Q8 ? idx_q8() : idx_nv_x();
    for (int c = 0; c < nc; ++c) {
        const block_q8_0* b = xb + 2 * c;
        _mm512_store_si512((void*)(q + 64 * c), _mm512_permutexvar_epi32(IDX, ld2(b[0].qs, b[1].qs)));
        const float a = _cvtsh_ss(b[0].d), e = _cvtsh_ss(b[1].d);
        if constexpr (FMT == FMT_Q8) {
            d[2 * c] = a;
            d[2 * c + 1] = e;
        } else {
            d[4 * c] = a;
            d[4 * c + 1] = a;
            d[4 * c + 2] = e;
            d[4 * c + 3] = e;
        }
    }
}

// ---- the kernel: R rows x N activations, all chunks of the row
template <int FMT, int ENC, int R, int N>
void kern(const char* w, size_t rs, int nc, const int8_t* const* xq, const float* const* dx, float* y, size_t ldy, int pf, int hint) {
    constexpr size_t CB = FMT == FMT_Q8 ? 2 * sizeof(block_q8_0) : sizeof(block_nvfp4);  // bytes per 64-weight chunk (68 / 36)
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i zero = _mm512_setzero_si512();
    [[maybe_unused]] const __m512i IDXQ = idx_q8();
    [[maybe_unused]] const __m512i IDXW = idx_nv_w();
    [[maybe_unused]] const __m512i SH = _mm512_setr_epi64(0, 0, 0, 0, 4, 4, 4, 4);
    [[maybe_unused]] const __m512i M0F = _mm512_set1_epi8(0x0F);
    [[maybe_unused]] const __m512i M08 = _mm512_set1_epi8(0x08);
    [[maybe_unused]] const __m512i TBL = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    [[maybe_unused]] const __m128 POW = _mm_set1_ps(kNvfp4rPow);
    __m512 acc[R][N];
    const char* wr[R];
    for (int r = 0; r < R; ++r) {
        wr[r] = w + (size_t)r * rs;
        for (int j = 0; j < N; ++j) acc[r][j] = _mm512_setzero_ps();
    }
    // per row and chunk: decoded |w| bytes, sign mask, pre-broadcast scale vector (same arithmetic for both loop orders below)
    auto decode = [&](const char* p, __m512i& AW, __mmask64& km, __m512& SC) {
        pf1(p + pf, hint);
        if constexpr (FMT == FMT_Q8) {
            pf1(p + pf + 64, hint);
            const __m512i W = _mm512_permutexvar_epi32(IDXQ, ld2(p + 2, p + 2 + sizeof(block_q8_0)));
            AW = _mm512_abs_epi8(W);
            km = _mm512_movepi8_mask(W);
            uint16_t d0, d1;
            std::memcpy(&d0, p, 2);
            std::memcpy(&d1, p + sizeof(block_q8_0), 2);
            const __m128 dw = _mm_cvtph_ps(_mm_cvtsi32_si128((int)((uint32_t)d0 | ((uint32_t)d1 << 16))));  // (dw0, dw1, 0, 0)
            SC = _mm512_castpd_ps(_mm512_broadcastsd_pd(_mm_castps_pd(dw)));                            // (dw0, dw1) x 8
        } else {
            const __m512i Q = _mm512_permutexvar_epi32(IDXW, _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)(p + 4))));
            const __m512i Nn = _mm512_and_si512(_mm512_srlv_epi64(Q, SH), M0F);
            AW = _mm512_shuffle_epi8(TBL, Nn);
            km = _mm512_test_epi8_mask(Nn, M08);
            __m128 sc;
            if constexpr (ENC == ENC_GGUF) {
                sc = _mm_setr_ps(g_tab[(uint8_t)p[0]], g_tab[(uint8_t)p[1]], g_tab[(uint8_t)p[2]], g_tab[(uint8_t)p[3]]);
            } else {
                int32_t b4;
                std::memcpy(&b4, p, 4);
                sc = _mm_mul_ps(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(b4)), 20)), POW);  // == nvfp4r_decode per byte
            }
            SC = _mm512_broadcast_f32x4(sc);  // (sc0..sc3) x 4
        }
    };
    // re-encoded scales of 4 rows at once (ENC_R): 4 x 4 bytes -> 16 floats with one cvtepu8 / slli / mul (== nvfp4r_decode16), then one 128-bit lane
    // broadcast per row; the same per-lane operations as the per-row path, so the floats are identical
    [[maybe_unused]] const __m512 POW16 = _mm512_set1_ps(kNvfp4rPow);
    [[maybe_unused]] auto scales4r = [&](const char* p0, const char* p1, const char* p2, const char* p3, __m512* SC) {
        __m128i b = _mm_cvtsi32_si128(*(const int32_t*)(const void*)p0);
        b = _mm_insert_epi32(b, *(const int32_t*)(const void*)p1, 1);
        b = _mm_insert_epi32(b, *(const int32_t*)(const void*)p2, 2);
        b = _mm_insert_epi32(b, *(const int32_t*)(const void*)p3, 3);
        const __m512 v = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(b), 20)), POW16);
        SC[0] = _mm512_shuffle_f32x4(v, v, 0x00);
        SC[1] = _mm512_shuffle_f32x4(v, v, 0x55);
        SC[2] = _mm512_shuffle_f32x4(v, v, 0xAA);
        SC[3] = _mm512_shuffle_f32x4(v, v, 0xFF);
    };
    // GGUF UE4M3 scale bytes of 4 rows at once, exactly ue4m3_to_f32 per byte (bit-identical to the table): b == 0x00 or 0x7F -> 0; v = b & 0x7F (the sign bit is
    // ignored like in ue4m3_to_f32); exponent e = v >> 3 > 0 -> (1 + man/8) * 2^(e-8) = bitcast((v << 20) + (119 << 23)) (exact: e+119 is the fp32 exponent field,
    // man the top 3 mantissa bits); e == 0 (codes 1..7) -> man * 2^-10 = cvt(v) * 2^-10 (exact). 13 vector uops per 16 scales instead of 16 table loads + 12 inserts.
    [[maybe_unused]] auto scales4g = [&](const char* p0, const char* p1, const char* p2, const char* p3, __m512* SC) {
        __m128i b = _mm_cvtsi32_si128(*(const int32_t*)(const void*)p0);
        b = _mm_insert_epi32(b, *(const int32_t*)(const void*)p1, 1);
        b = _mm_insert_epi32(b, *(const int32_t*)(const void*)p2, 2);
        b = _mm_insert_epi32(b, *(const int32_t*)(const void*)p3, 3);
        const __m512i B = _mm512_cvtepu8_epi32(b);
        const __mmask16 nz = _mm512_cmpneq_epi32_mask(B, _mm512_setzero_si512()) & _mm512_cmpneq_epi32_mask(B, _mm512_set1_epi32(0x7F));
        const __m512i V = _mm512_and_si512(B, _mm512_set1_epi32(0x7F));
        const __mmask16 den = _mm512_cmpeq_epi32_mask(_mm512_and_si512(V, _mm512_set1_epi32(0x78)), _mm512_setzero_si512());
        const __m512 normal = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_slli_epi32(V, 20), _mm512_set1_epi32(119 << 23)));
        const __m512 denorm = _mm512_mul_ps(_mm512_cvtepi32_ps(V), _mm512_set1_ps(0x1p-10f));
        const __m512 v = _mm512_maskz_mov_ps(nz, _mm512_mask_blend_ps(den, normal, denorm));
        SC[0] = _mm512_shuffle_f32x4(v, v, 0x00);
        SC[1] = _mm512_shuffle_f32x4(v, v, 0x55);
        SC[2] = _mm512_shuffle_f32x4(v, v, 0xAA);
        SC[3] = _mm512_shuffle_f32x4(v, v, 0xFF);
    };
    // |w| bytes + sign mask only (scale decoded elsewhere)
    [[maybe_unused]] auto decode_nv_w = [&](const char* p, __m512i& AW, __mmask64& km) {
        pf1(p + pf, hint);
        const __m512i Q = _mm512_permutexvar_epi32(IDXW, _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)(p + 4))));
        const __m512i Nn = _mm512_and_si512(_mm512_srlv_epi64(Q, SH), M0F);
        AW = _mm512_shuffle_epi8(TBL, Nn);
        km = _mm512_test_epi8_mask(Nn, M08);
    };
    auto load_x = [&](int j, int c, __m512i& X, __m512& DX) {
        X = _mm512_load_si512((const void*)(xq[j] + 64 * c));
        if constexpr (FMT == FMT_Q8) {
            double dd;
            std::memcpy(&dd, dx[j] + 2 * c, 8);
            DX = _mm512_castpd_ps(_mm512_set1_pd(dd));  // (dx0, dx1) x 8
        } else {
            DX = _mm512_broadcast_f32x4(_mm_loadu_ps(dx[j] + 4 * c));  // (dx0, dx0, dx1, dx1) x 4
        }
    };
    auto step = [&](__m512& a, __m512i AW, __mmask64 km, __m512 SC, __m512i X, __m512 DX) {
        const __m512i XS = _mm512_mask_sub_epi8(X, km, zero, X);
        const __m512i I = _mm512_madd_epi16(_mm512_maddubs_epi16(AW, XS), ones);
        a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(I), _mm512_mul_ps(SC, DX), a);
    };
    for (int c = 0; c < nc; ++c) {
        const size_t off = (size_t)c * CB;
        if constexpr (R >= 8) {
            // 8 row streams, N <= 2: keep the N activations live, decode one row at a time (16 accumulators + 2N + 3 live vectors)
            __m512i X[N];
            __m512 DX[N];
            for (int j = 0; j < N; ++j) load_x(j, c, X[j], DX[j]);
            if constexpr (FMT == FMT_NV) {
                for (int r0 = 0; r0 < R; r0 += 4) {
                    __m512 SC[4];
                    if constexpr (ENC == ENC_R) scales4r(wr[r0] + off, wr[r0 + 1] + off, wr[r0 + 2] + off, wr[r0 + 3] + off, SC);
                    else scales4g(wr[r0] + off, wr[r0 + 1] + off, wr[r0 + 2] + off, wr[r0 + 3] + off, SC);
                    for (int r = r0; r < r0 + 4; ++r) {
                        __m512i AW;
                        __mmask64 km;
                        decode_nv_w(wr[r] + off, AW, km);
                        for (int j = 0; j < N; ++j) step(acc[r][j], AW, km, SC[r - r0], X[j], DX[j]);
                    }
                }
            } else {
                for (int r = 0; r < R; ++r) {
                    __m512i AW;
                    __mmask64 km;
                    __m512 SC;
                    decode(wr[r] + off, AW, km, SC);
                    for (int j = 0; j < N; ++j) step(acc[r][j], AW, km, SC, X[j], DX[j]);
                }
            }
        } else {
            // <= 4 row streams, N <= 8: keep the R decoded rows live, load one activation at a time (16 accumulators + 2R + 2 live vectors)
            __m512i AW[R];
            __mmask64 km[R];
            __m512 SC[R];
            if constexpr (FMT == FMT_NV && R == 4) {
                if constexpr (ENC == ENC_R) scales4r(wr[0] + off, wr[1] + off, wr[2] + off, wr[3] + off, SC);
                else scales4g(wr[0] + off, wr[1] + off, wr[2] + off, wr[3] + off, SC);
                for (int r = 0; r < R; ++r) decode_nv_w(wr[r] + off, AW[r], km[r]);
            } else {
                for (int r = 0; r < R; ++r) decode(wr[r] + off, AW[r], km[r], SC[r]);
            }
            for (int j = 0; j < N; ++j) {
                __m512i X;
                __m512 DX;
                load_x(j, c, X, DX);
                for (int r = 0; r < R; ++r) step(acc[r][j], AW[r], km[r], SC[r], X, DX);
            }
        }
    }
    const __m512i unp = unperm<FMT>();
    for (int j = 0; j < N; ++j)
        for (int r = 0; r < R; ++r) y[(size_t)j * ldy + r] = hsum16(_mm512_permutexvar_ps(unp, acc[r][j]));
}

template <int FMT, int ENC, int R>
inline void call_n(int n, const char* w, size_t rs, int nc, const int8_t* const* xq, const float* const* dx, float* y, size_t ldy, int pf, int hint) {
    constexpr int NMAX = R >= 8 ? 2 : R >= 4 ? 4 : 8;
    switch (n) {
        case 1: kern<FMT, ENC, R, 1>(w, rs, nc, xq, dx, y, ldy, pf, hint); break;
        case 2: kern<FMT, ENC, R, 2>(w, rs, nc, xq, dx, y, ldy, pf, hint); break;
        case 3: if constexpr (NMAX >= 3) kern<FMT, ENC, R, 3>(w, rs, nc, xq, dx, y, ldy, pf, hint); break;
        case 4: if constexpr (NMAX >= 4) kern<FMT, ENC, R, 4>(w, rs, nc, xq, dx, y, ldy, pf, hint); break;
        case 5: if constexpr (NMAX >= 5) kern<FMT, ENC, R, 5>(w, rs, nc, xq, dx, y, ldy, pf, hint); break;
        case 6: if constexpr (NMAX >= 6) kern<FMT, ENC, R, 6>(w, rs, nc, xq, dx, y, ldy, pf, hint); break;
        case 7: if constexpr (NMAX >= 7) kern<FMT, ENC, R, 7>(w, rs, nc, xq, dx, y, ldy, pf, hint); break;
        default: if constexpr (NMAX >= 8) kern<FMT, ENC, R, 8>(w, rs, nc, xq, dx, y, ldy, pf, hint); break;
    }
}

inline int rows_per_group(int nx) { return nx <= 2 ? 8 : nx <= 4 ? 4 : 2; }

// Rows in groups of RG (8 / 4 / 2 by N); a tail is split into power-of-two sub-groups (4, 2, 1). The first `hb` bytes of every row of the NEXT group are
// prefetched while the current group runs (whole-group lead time), inside the kernel every row stream prefetches `pf` bytes ahead (gemv.cpp policy).
template <int FMT, int ENC>
void drive(const char* w, size_t rs, int nrows, const int8_t* const* xq, const float* const* dx, int n, int nc, float* y, size_t ldy, int pf, int hint) {
    constexpr size_t CB = FMT == FMT_Q8 ? 2 * sizeof(block_q8_0) : sizeof(block_nvfp4);
    const size_t rowbytes = (size_t)nc * CB;
    const int hb = (int)std::min<size_t>((size_t)std::max(pf, 0), rowbytes);
    const int RG = rows_per_group(n);
    for (int i = 0; i < std::min(RG, nrows); ++i) pf_head(w + (size_t)i * rs, hb, hint);
    int r = 0;
    while (r < nrows) {
        int R = std::min(RG, nrows - r);
        R = R >= 8 ? 8 : R >= 4 ? 4 : R >= 2 ? 2 : 1;
        const int nxt = std::min(RG, nrows - (r + R));
        for (int i = 0; i < nxt; ++i) pf_head(w + (size_t)(r + R + i) * rs, hb, hint);
        const char* wg = w + (size_t)r * rs;
        switch (R) {
            case 8: call_n<FMT, ENC, 8>(n, wg, rs, nc, xq, dx, y + r, ldy, pf, hint); break;
            case 4: call_n<FMT, ENC, 4>(n, wg, rs, nc, xq, dx, y + r, ldy, pf, hint); break;
            case 2: call_n<FMT, ENC, 2>(n, wg, rs, nc, xq, dx, y + r, ldy, pf, hint); break;
            default: call_n<FMT, ENC, 1>(n, wg, rs, nc, xq, dx, y + r, ldy, pf, hint); break;
        }
        r += R;
    }
}

template <int FMT, int ENC>
void multix(const void* w, size_t rs, int nrows, const block_q8_0* const* x, int nx, int k, float* y, size_t ldy, const char* name) {
    if (nrows <= 0 || nx <= 0) return;
    if (k <= 0 || k % 64 || k > KMAX) bad_arg(name, "K", k);
    const int nc = k / 64;
    XScratch& S = xscratch();
    const int pf = g_gemv_prefetch_bytes, hint = g_gemv_prefetch_hint;
    for (int j0 = 0; j0 < nx; j0 += MX) {
        const int n = std::min(MX, nx - j0);
        const int8_t* xq[MX];
        const float* dx[MX];
        for (int j = 0; j < n; ++j) {
            pack_x<FMT>(x[j0 + j], nc, S.xq[j], S.dx[j]);
            xq[j] = S.xq[j];
            dx[j] = S.dx[j];
        }
        drive<FMT, ENC>((const char*)w, rs, nrows, xq, dx, n, nc, y + (size_t)j0 * ldy, ldy, pf, hint);
    }
}

}  // namespace

void gemv_q8_0_multix(const void* w, size_t rs, int nrows, const block_q8_0* const* x, int nx, int k, float* y, size_t ldy) {
    multix<FMT_Q8, ENC_GGUF>(w, rs, nrows, x, nx, k, y, ldy, "gemv_q8_0_multix");
}
void gemv_nvfp4_multix(const void* w, size_t rs, int nrows, const block_q8_0* const* x, int nx, int k, float* y, size_t ldy) {
    multix<FMT_NV, ENC_GGUF>(w, rs, nrows, x, nx, k, y, ldy, "gemv_nvfp4_multix");
}
void gemv_nvfp4r_multix(const void* w, size_t rs, int nrows, const block_q8_0* const* x, int nx, int k, float* y, size_t ldy) {
    multix<FMT_NV, ENC_R>(w, rs, nrows, x, nx, k, y, ldy, "gemv_nvfp4r_multix");
}
int gemv_multix_rows_per_group(int nx) { return rows_per_group(nx); }

}  // namespace ink

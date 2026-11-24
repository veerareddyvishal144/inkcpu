// Batched Q8_0 / NVFP4 x Q8_0 GEMM (AVX-512 F/BW/DQ, no VNNI) + MoE grouped driver. Owner: agent G. Design: docs/KERNELS_GEMM.md.
//
// Bit-identity with B1's gemv kernels: every (row, token) pair keeps B1's 16 float accumulator lanes; chunk c (64 weights) adds
//   acc_l = fmaf((float)I_l, S_l, acc_l),  I_l = exact int sum of 4 products, S_l = fl(dw * dx)  (one float multiply)
// and the result is hsum16 (same tree) of the accumulators in B1's lane order. Only the PHYSICAL lane order is different (a fixed
// permutation chosen so that the scale vector is a plain memory broadcast); one vpermps restores B1's order before hsum16.
//   Q8_0 : physical dword d = (block d&1, group d>>1)           logical lane l = (l&7)*2 + (l>>3) <-> physical;   S = [s0 s1] x8
//   NVFP4: physical dword d = (sub-block d&3, group d>>2 = q)   logical lane l <-> physical 8*(l>>3) + 4*(l&1) + ((l>>1)&3); S = [sc0..sc3] x4
// X4 PROTOTYPE (numerics-deviating, OFF unless the caller uses the *_x4 entry points; see docs/status/K3.md): weights are unpacked as kv+12
// (unsigned 0..24, B1's gemv table) with NO sign masks, so the inner loop is maddubs/madd/cvt/fma = 4 vector uops per (row, token, 64 MACs)
// instead of 5. The bias 12*sum(x) is removed in the float domain: for each (row, token) corr = sum_s dw[r][s] * Xs[t][s] with
// Xs[t][s] = 12 * dx[t][s] * sum_{i in sub-block s} x[i] (packed once per token, GEMM_FMT_NVFP4X4), a K/16-long dot product (0.25 uop per
// pair-chunk), subtracted after the batched reduction. Mathematically identical, rounding differs (reported by test_gemm, section x4).
// NVFP4 scale encodings (template ENC): ENC_GGUF = UE4M3 bytes as stored in the GGUF (256-entry table, 4 scalar loads + 4 scalar stores
// per row super-block); ENC_R = bytes re-encoded by nvfp4r_repack_rows (include/ink/nvfp4_repack.h): the 4 scale dwords of 4 consecutive
// super-blocks of a row are gathered (3 shuffles), decoded with nvfp4r_decode16 (3 uops) and stored as ONE 64-byte vector into the same
// dw[row][4*sb + s] layout (code 0 -> +0.0f gives the zero padding of the last group for free). Same floats, so the micro-kernel is shared.
#include <immintrin.h>

#include <time.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ink/kernels_gemm.h"
#include "ink/nvfp4_repack.h"

namespace ink {

int g_gemm_group = 64;
int g_gemm_tile = 4;
int g_gemm_prefetch = 1;
int g_gemm_epilogue = 1;
int g_gemm_tail5 = 1;

namespace {

constexpr int KMAX = 8192;
constexpr int DWMAX = KMAX / 16;  // max floats of a scale array (NVFP4: K/16)
constexpr int TMAX = 5;           // tokens per register tile (4 rows x TMAX tokens)
enum { ENC_GGUF = 0, ENC_R = 1 };  // NVFP4 scale byte encoding (see the header comment)

[[noreturn]] void bad_arg(const char* fn, const char* what, long v) {
    std::fprintf(stderr, "ink::%s: unsupported %s=%ld\n", fn, what, v);
    std::abort();
}

alignas(64) float g_ue4m3[256];  // exact copy of ue4m3_to_f32 (same function) for every byte
struct TabInit {
    TabInit() {
        for (int i = 0; i < 256; ++i) g_ue4m3[i] = ue4m3_to_f32((uint8_t)i);
    }
} g_tab_init;

inline float hsum16(__m512 v) {  // identical tree to B1 (lanes j+8, j+4, j+2, j+1)
    const __m256 t = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
    const __m128 u = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
    __m128 w = _mm_add_ps(u, _mm_movehl_ps(u, u));
    w = _mm_add_ss(w, _mm_shuffle_ps(w, w, 1));
    return _mm_cvtss_f32(w);
}

inline int pad16(int n) { return (n + 15) & ~15; }
inline int dxn_of(int fmt, int k) { return pad16(fmt == GEMM_FMT_Q8 ? k / 32 : k / 16); }
inline int xsn_of(int fmt, int k) { return fmt == GEMM_FMT_NVFP4X4 ? pad16(k / 16) : 0; }
inline int xbn_of(int fmt, int k) { return fmt == GEMM_FMT_I16 ? pad16(k / 16) : 0; }  // P2: int32 sub-block sums per token
inline bool fmt_ok(int want, bool x4, int have) { return have == (x4 ? GEMM_FMT_NVFP4X4 : want) || (!x4 && want == GEMM_FMT_NVFP4 && have == GEMM_FMT_I16); }
inline size_t align64(size_t n) { return (n + 63) & ~(size_t)63; }

inline __m512i ld2(const void* a, const void* b) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)a)), _mm256_loadu_si256((const __m256i*)b), 1);
}

// ---- lane permutations
inline __m512i idx_q8() { return _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15); }      // both W and X
inline __m512i idx_nv_x() { return _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15); }  // activation
inline __m512i idx_nv_w() { return _mm512_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7); }       // broadcast nibble bytes
template <int FMT>
inline __m512i unperm() {  // logical lane l <- physical lane idx[l]
    if constexpr (FMT == GEMM_FMT_Q8)
        return _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    else
        return _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
}

// ---------------------------------------------------------------- activation packing
void pack_one(int fmt, const block_q8_0* xr, int k, int8_t* q, float* d, int dxn, float* xs, int32_t* xb) {
    const int nc = k / 64;
    if (fmt == GEMM_FMT_I16) {  // P2: xb[s] = 12 * exact int sum of the 16 quants of sub-block s (the weight-offset bias of the i16 kernels)
        const int ns = k / 16, nsp = pad16(ns);
        for (int sb = 0; sb < ns; ++sb) {
            const int8_t* qs = xr[sb / 2].qs + (sb & 1) * 16;
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += qs[i];
            xb[sb] = 12 * sum;
        }
        for (int i = ns; i < nsp; ++i) xb[i] = 0;
    }
    if (fmt == GEMM_FMT_NVFP4X4) {  // x4 prototype: Xs[s] = 12 * dx(block of s) * sum of the 16 quants of sub-block s (exact int sum, one float multiply)
        const int ns = k / 16, nsp = pad16(ns);
        for (int sb = 0; sb < ns; ++sb) {
            const block_q8_0* b = xr + sb / 2;
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += b->qs[(sb & 1) * 16 + i];
            xs[sb] = _cvtsh_ss(b->d) * (float)(12 * sum);
        }
        for (int i = ns; i < nsp; ++i) xs[i] = 0.f;
    }
    if (fmt == GEMM_FMT_Q8) {
        const __m512i IDX = idx_q8();
        for (int c = 0; c < nc; ++c) {
            const block_q8_0* b = xr + 2 * c;
            _mm512_storeu_si512(q + 64 * c, _mm512_permutexvar_epi32(IDX, ld2(b[0].qs, b[1].qs)));
            d[2 * c] = _cvtsh_ss(b[0].d);
            d[2 * c + 1] = _cvtsh_ss(b[1].d);
        }
        for (int i = 2 * nc; i < dxn; ++i) d[i] = 0.f;
    } else {
        const __m512i IDX = idx_nv_x();
        for (int c = 0; c < nc; ++c) {
            const block_q8_0* b = xr + 2 * c;
            _mm512_storeu_si512(q + 64 * c, _mm512_permutexvar_epi32(IDX, ld2(b[0].qs, b[1].qs)));
            const float a = _cvtsh_ss(b[0].d), e = _cvtsh_ss(b[1].d);
            d[4 * c] = a;
            d[4 * c + 1] = a;
            d[4 * c + 2] = e;
            d[4 * c + 3] = e;
        }
        for (int i = 4 * nc; i < dxn; ++i) d[i] = 0.f;
    }
}

// ---------------------------------------------------------------- weight panel (4 rows) in the lane-permuted unpacked form
struct alignas(64) Scratch {
    // row strides are NOT powers of two (+64 B): rows / P arrays would otherwise map to the same L1 sets and thrash (8-way, 4 KB period)
    alignas(64) uint8_t mag[4][KMAX + 64];         // |w| (Q8_0) or |kv| (NVFP4), physical lane order
    alignas(64) uint64_t sgn[4][KMAX / 64 + 8];    // sign mask per 64 weights
    alignas(64) float dw[4][DWMAX + 16];           // weight scales (Q8_0: per block; NVFP4: 4 per super-block), zero padded
    alignas(64) uint16_t d16[DWMAX];               // fp16 staging
    alignas(64) float P[4 * TMAX][DWMAX + 16];     // per-tile scale products dw*dx
};
Scratch& scratch() {
    static thread_local Scratch s;
    return s;
}

inline void unpack_row_q8(const char* row, int k, uint8_t* mag, uint64_t* sgn, float* dw, uint16_t* d16) {
    const int nc = k / 64, nb = k / 32, nbp = pad16(nb);
    const __m512i IDX = idx_q8();
    for (int c = 0; c < nc; ++c) {
        const char* p = row + (size_t)c * 68;
        const __m512i W = _mm512_permutexvar_epi32(IDX, ld2(p + 2, p + 36));
        _mm512_store_si512((void*)(mag + 64 * c), _mm512_abs_epi8(W));
        sgn[c] = (uint64_t)_mm512_movepi8_mask(W);
        uint16_t a, b;
        std::memcpy(&a, p, 2);
        std::memcpy(&b, p + 34, 2);
        d16[2 * c] = a;
        d16[2 * c + 1] = b;
    }
    for (int i = nb; i < nbp; ++i) d16[i] = 0;
    for (int i = 0; i < nbp; i += 16) _mm512_store_ps(dw + i, _mm512_cvtph_ps(_mm256_load_si256((const __m256i*)(d16 + i))));
}

inline void unpack_row_nvfp4(const char* row, int k, uint8_t* mag, uint64_t* sgn, float* dw) {
    const int nc = k / 64, n4 = 4 * nc, np = pad16(n4);
    const __m512i IDX = idx_nv_w();
    const __m512i SH = _mm512_setr_epi64(0, 0, 0, 0, 4, 4, 4, 4);
    const __m512i M0F = _mm512_set1_epi8(0x0F), M08 = _mm512_set1_epi8(0x08);
    const __m512i TBL = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    for (int c = 0; c < nc; ++c) {
        const char* p = row + (size_t)c * 36;
        const __m512i Q = _mm512_permutexvar_epi32(IDX, _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)(p + 4))));
        const __m512i N = _mm512_and_si512(_mm512_srlv_epi64(Q, SH), M0F);
        _mm512_store_si512((void*)(mag + 64 * c), _mm512_shuffle_epi8(TBL, N));
        sgn[c] = (uint64_t)_mm512_test_epi8_mask(N, M08);
        dw[4 * c] = g_ue4m3[(uint8_t)p[0]];
        dw[4 * c + 1] = g_ue4m3[(uint8_t)p[1]];
        dw[4 * c + 2] = g_ue4m3[(uint8_t)p[2]];
        dw[4 * c + 3] = g_ue4m3[(uint8_t)p[3]];
    }
    for (int i = n4; i < np; ++i) dw[i] = 0.f;
}

// Re-encoded scales (ENC_R). Nibbles/signs exactly as above; the sign mask is stored straight from the k register (kmovq m64, k: no p0 uop).
// Scales: 4 super-blocks per step, dword s of the gathered xmm = the 4 scale bytes of super-block c+s, so byte 4*s + i decodes to
// dw[4*(c+s) + i]: one aligned 64-byte store per 4 super-blocks (dw rows are 64-byte aligned, c % 4 == 0). Missing super-blocks of the
// last step are zero dwords -> +0.0f = the zero padding that compute_P relies on (np = pad16(4*nc) = 16 * ceil(nc/4)).
// The unpack is bound by the 4-wide front end, not by the vector ports (the GGUF version issues ~23 fused uops per row super-block), so the
// 4-super-block body is fully unrolled, branch-free, with constant offsets, and kept out of line so the constants live in registers.
inline void unpack_sb_nvfp4(const char* p, uint8_t* mag, uint64_t* sgn, __m512i IDX, __m512i SH, __m512i M0F, __m512i M08, __m512i TBL) {
    const __m512i Q = _mm512_permutexvar_epi32(IDX, _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)(p + 4))));
    const __m512i N = _mm512_and_si512(_mm512_srlv_epi64(Q, SH), M0F);
    _mm512_store_si512((void*)mag, _mm512_shuffle_epi8(TBL, N));
    _store_mask64((__mmask64*)sgn, _mm512_test_epi8_mask(N, M08));
}
inline __m512 decode_scales4(__m128i a, __m128i b, __m128i c, __m128i d, __m512 POW) {  // 4 dwords of scale codes -> 16 floats
    const __m128i b16 = _mm_unpacklo_epi64(_mm_unpacklo_epi32(a, b), _mm_unpacklo_epi32(c, d));
    return _mm512_mul_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(b16), 20)), POW);
}
__attribute__((noinline)) void unpack_row_nvfp4r(const char* row, int k, uint8_t* mag, uint64_t* sgn, float* dw) {
    const int nc = k / 64;
    const __m512i IDX = idx_nv_w();
    const __m512i SH = _mm512_setr_epi64(0, 0, 0, 0, 4, 4, 4, 4);
    const __m512i M0F = _mm512_set1_epi8(0x0F), M08 = _mm512_set1_epi8(0x08);
    const __m512i TBL = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    const __m512 POW = _mm512_set1_ps(kNvfp4rPow);
    const char* p = row;
    int c = 0;
    for (; c + 4 <= nc; c += 4, p += 4 * 36, mag += 256, sgn += 4, dw += 16) {
        unpack_sb_nvfp4(p, mag, sgn, IDX, SH, M0F, M08, TBL);
        unpack_sb_nvfp4(p + 36, mag + 64, sgn + 1, IDX, SH, M0F, M08, TBL);
        unpack_sb_nvfp4(p + 72, mag + 128, sgn + 2, IDX, SH, M0F, M08, TBL);
        unpack_sb_nvfp4(p + 108, mag + 192, sgn + 3, IDX, SH, M0F, M08, TBL);
        _mm512_store_ps(dw, decode_scales4(_mm_loadu_si32(p), _mm_loadu_si32(p + 36), _mm_loadu_si32(p + 72), _mm_loadu_si32(p + 108), POW));
    }
    if (c < nc) {  // 1..3 remaining super-blocks; absent ones decode to +0.0f (= the padding)
        const int rem = nc - c;
        for (int i = 0; i < rem; ++i) unpack_sb_nvfp4(p + 36 * i, mag + 64 * i, sgn + i, IDX, SH, M0F, M08, TBL);
        const __m128i z = _mm_setzero_si128();
        _mm512_store_ps(dw, decode_scales4(_mm_loadu_si32(p), rem > 1 ? _mm_loadu_si32(p + 36) : z, rem > 2 ? _mm_loadu_si32(p + 72) : z, z, POW));
    }
}

// x4 prototype unpack: mag = kv + 12 (B1's unsigned table), no sign masks (4 vector uops + 1 store per row super-block).
inline void unpack_sb_nvfp4_x4(const char* p, uint8_t* mag, __m512i IDX, __m512i SH, __m512i M0F, __m512i TBL12) {
    const __m512i Q = _mm512_permutexvar_epi32(IDX, _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)(p + 4))));
    const __m512i N = _mm512_and_si512(_mm512_srlv_epi64(Q, SH), M0F);
    _mm512_store_si512((void*)mag, _mm512_shuffle_epi8(TBL12, N));
}
__attribute__((noinline)) void unpack_row_nvfp4r_x4(const char* row, int k, uint8_t* mag, float* dw) {
    const int nc = k / 64;
    const __m512i IDX = idx_nv_w();
    const __m512i SH = _mm512_setr_epi64(0, 0, 0, 0, 4, 4, 4, 4);
    const __m512i M0F = _mm512_set1_epi8(0x0F);
    const __m512i TBL12 = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m512 POW = _mm512_set1_ps(kNvfp4rPow);
    const char* p = row;
    int c = 0;
    for (; c + 4 <= nc; c += 4, p += 4 * 36, mag += 256, dw += 16) {
        unpack_sb_nvfp4_x4(p, mag, IDX, SH, M0F, TBL12);
        unpack_sb_nvfp4_x4(p + 36, mag + 64, IDX, SH, M0F, TBL12);
        unpack_sb_nvfp4_x4(p + 72, mag + 128, IDX, SH, M0F, TBL12);
        unpack_sb_nvfp4_x4(p + 108, mag + 192, IDX, SH, M0F, TBL12);
        _mm512_store_ps(dw, decode_scales4(_mm_loadu_si32(p), _mm_loadu_si32(p + 36), _mm_loadu_si32(p + 72), _mm_loadu_si32(p + 108), POW));
    }
    if (c < nc) {
        const int rem = nc - c;
        for (int i = 0; i < rem; ++i) unpack_sb_nvfp4_x4(p + 36 * i, mag + 64 * i, IDX, SH, M0F, TBL12);
        const __m128i z = _mm_setzero_si128();
        _mm512_store_ps(dw, decode_scales4(_mm_loadu_si32(p), rem > 1 ? _mm_loadu_si32(p + 36) : z, rem > 2 ? _mm_loadu_si32(p + 72) : z, z, POW));
    }
}

template <int FMT, int ENC, bool X4 = false>
inline void unpack_row(const char* row, int k, Scratch& S, int j) {
    if constexpr (X4)
        unpack_row_nvfp4r_x4(row, k, S.mag[j], S.dw[j]);
    else if constexpr (FMT == GEMM_FMT_Q8)
        unpack_row_q8(row, k, S.mag[j], S.sgn[j], S.dw[j], S.d16);
    else if constexpr (ENC == ENC_GGUF)
        unpack_row_nvfp4(row, k, S.mag[j], S.sgn[j], S.dw[j]);
    else
        unpack_row_nvfp4r(row, k, S.mag[j], S.sgn[j], S.dw[j]);
}

// ---------------------------------------------------------------- micro-kernel: 4 rows x T tokens, 5 vector uops per pair and chunk
// P holds the scale products of ONE K segment (SEG chunks): computing it right before its use lets the out-of-order core overlap the
// store-bound P phase with the tail of the previous segment's multiply loop, and keeps the P scratch small (L1).
constexpr int SEG = 16;
template <int T>
inline void compute_P(Scratch& S, const float* const dx[], int f0, int f1) {
    for (int r = 0; r < 4; ++r)
        for (int i = f0; i < f1; i += 16) {
            const __m512 w = _mm512_load_ps(S.dw[r] + i);
            for (int t = 0; t < T; ++t) _mm512_store_ps(S.P[r * T + t] + (i - f0), _mm512_mul_ps(w, _mm512_loadu_ps(dx[t] + i)));
        }
}

// Batched bit-identical epilogue for a full 4x4 tile. hsum16 adds lanes (l, l+8), then (l, l+4), (l, l+2), (l, l+1) in B1's LOGICAL order;
// in the kernels' PHYSICAL lane order those four steps pair lanes at distances 8, 2, 1, 4 (NVFP4) and 1, 8, 4, 1 (Q8_0; verified by
// brute force over all 256 step orders against hsum16 on random data, build-k3/epi_check.cpp). Each step
// combines two vectors with 2 shuffles + 1 add (the same two float operands as hsum16, so the same bits), halving the vector count:
// 16 -> 8 -> 4 -> 2 -> 1 = 30 shuffles + 15 adds for 16 results instead of 16 x (vpermps + 4 shuffles + 4 adds). The final vector holds
// token q's 4 rows in 128-bit quarter q (accumulator a[n] <-> (r, t) by idx<FMT>), stored with one masked 128-bit store per token.
inline __m512 red8(__m512 a, __m512 b) { return _mm512_add_ps(_mm512_shuffle_f32x4(a, b, 0x44), _mm512_shuffle_f32x4(a, b, 0xEE)); }
inline __m512 red4(__m512 a, __m512 b) { return _mm512_add_ps(_mm512_shuffle_f32x4(a, b, 0x88), _mm512_shuffle_f32x4(a, b, 0xDD)); }
inline __m512 red2(__m512 a, __m512 b) {
    return _mm512_add_ps(_mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(a), _mm512_castps_pd(b))),
                         _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(a), _mm512_castps_pd(b))));
}
inline __m512 red1(__m512 a, __m512 b) {
    return _mm512_add_ps(_mm512_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0)), _mm512_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)));
}
template <int FMT>
constexpr int acc_idx(int r, int t) {  // slot of acc[r][t] in the 16-input tree so that quarter q of the result = token q, rows 0..3
    return FMT == GEMM_FMT_Q8 ? 2 * t + (r & 1) + 8 * (r >> 1) : 8 * (t >> 1) + (t & 1) + 2 * r;
}
template <int FMT>
inline __m512 reduce16(const __m512 (&acc)[4][4]) {  // -> quarter q = token q, rows 0..3 (bit-identical to hsum16 of each accumulator)
    __m512 a[16];
    for (int r = 0; r < 4; ++r)
        for (int t = 0; t < 4; ++t) a[acc_idx<FMT>(r, t)] = acc[r][t];
    __m512 s1[8], s2[4], s3[2], s4;
    if constexpr (FMT == GEMM_FMT_NVFP4) {
        for (int i = 0; i < 8; ++i) s1[i] = red8(a[2 * i], a[2 * i + 1]);
        for (int i = 0; i < 4; ++i) s2[i] = red2(s1[2 * i], s1[2 * i + 1]);
        for (int i = 0; i < 2; ++i) s3[i] = red1(s2[2 * i], s2[2 * i + 1]);
        s4 = red4(s3[0], s3[1]);
    } else {
        for (int i = 0; i < 8; ++i) s1[i] = red1(a[2 * i], a[2 * i + 1]);
        for (int i = 0; i < 4; ++i) s2[i] = red8(s1[2 * i], s1[2 * i + 1]);
        for (int i = 0; i < 2; ++i) s3[i] = red4(s2[2 * i], s2[2 * i + 1]);
        s4 = red1(s3[0], s3[1]);
    }
    return s4;
}
inline void store16(__m512 s4, int rn, float* y, size_t ldy) {
    const __mmask8 m = (__mmask8)((1u << rn) - 1);
    _mm_mask_storeu_ps(y, m, _mm512_castps512_ps128(s4));
    _mm_mask_storeu_ps(y + ldy, m, _mm512_extractf32x4_ps(s4, 1));
    _mm_mask_storeu_ps(y + 2 * ldy, m, _mm512_extractf32x4_ps(s4, 2));
    _mm_mask_storeu_ps(y + 3 * ldy, m, _mm512_extractf32x4_ps(s4, 3));
}

template <int FMT, int T, bool X4 = false>
void tile_kernel(Scratch& S, int nc, int pn, const int8_t* const xq[], const float* const dx[], int rn, float* y, size_t ldy,
                 const float* const xs[] = nullptr) {
    constexpr int BC = FMT == GEMM_FMT_Q8 ? 2 : 4;  // floats of P per chunk
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc[4][T];
    for (int r = 0; r < 4; ++r)
        for (int t = 0; t < T; ++t) acc[r][t] = _mm512_setzero_ps();
    const int8_t* xp[T];
    for (int t = 0; t < T; ++t) xp[t] = xq[t];
    for (int c0 = 0; c0 < nc; c0 += SEG) {
      const int c1 = std::min(nc, c0 + SEG);
      compute_P<T>(S, dx, c0 * BC, std::min(pn, c0 * BC + SEG * BC));
      for (int c = c0; c < c1; ++c) {
        for (int r = 0; r < 4; ++r) {
            const __m512i AW = _mm512_load_si512((const void*)(S.mag[r] + 64 * c));
            const __mmask64 km = X4 ? (__mmask64)0 : (__mmask64)S.sgn[r][c];
            for (int t = 0; t < T; ++t) {
                const __m512i X = _mm512_loadu_si512((const void*)(xp[t] + 64 * c));
                const __m512i XS = X4 ? X : _mm512_mask_sub_epi8(X, km, zero, X);
                const __m512i I = _mm512_madd_epi16(_mm512_maddubs_epi16(AW, XS), ones);
                const float* ps = S.P[r * T + t] + BC * (c - c0);
                __m512 sv;
                if constexpr (FMT == GEMM_FMT_Q8) {
                    double dd;
                    std::memcpy(&dd, ps, 8);
                    sv = _mm512_castpd_ps(_mm512_set1_pd(dd));
                } else {
                    sv = _mm512_broadcast_f32x4(_mm_loadu_ps(ps));
                }
                acc[r][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(I), sv, acc[r][t]);
            }
        }
      }
    }
    if constexpr (X4) {  // bias correction: corr[r][t] = dot(dw[r], Xs[t]) over the K/16 sub-blocks (zero padded to pn floats)
        __m512 cv[4][T];
        for (int r = 0; r < 4; ++r)
            for (int t = 0; t < T; ++t) cv[r][t] = _mm512_setzero_ps();
        for (int i = 0; i < pn; i += 16)
            for (int r = 0; r < 4; ++r) {
                const __m512 w = _mm512_load_ps(S.dw[r] + i);
                for (int t = 0; t < T; ++t) cv[r][t] = _mm512_fmadd_ps(w, _mm512_loadu_ps(xs[t] + i), cv[r][t]);
            }
        if constexpr (T == 4) {
            store16(_mm512_sub_ps(reduce16<FMT>(acc), reduce16<FMT>(cv)), rn, y, ldy);
        } else {
            const __m512i unp = unperm<FMT>();
            for (int t = 0; t < T; ++t)
                for (int r = 0; r < rn; ++r) y[(size_t)t * ldy + r] = hsum16(_mm512_permutexvar_ps(unp, _mm512_sub_ps(acc[r][t], cv[r][t])));
        }
        return;
    }
    if constexpr (T == 4) {
        if (g_gemm_epilogue) {
            store16(reduce16<FMT>(acc), rn, y, ldy);
            return;
        }
    } else if constexpr (T == 5) {  // 4 tokens batched + the 5th token through hsum16 (bit-identical either way)
        if (g_gemm_epilogue) {
            __m512 a4[4][4];
            for (int r = 0; r < 4; ++r)
                for (int t = 0; t < 4; ++t) a4[r][t] = acc[r][t];
            store16(reduce16<FMT>(a4), rn, y, ldy);
            const __m512i unp = unperm<FMT>();
            for (int r = 0; r < rn; ++r) y[(size_t)4 * ldy + r] = hsum16(_mm512_permutexvar_ps(unp, acc[r][4]));
            return;
        }
    } else if constexpr (T == 2 || T == 3) {  // tail tiles: same batched tree on zero-padded accumulators (lanes never mix across accumulators)
        if (g_gemm_epilogue) {
            __m512 a4[4][4];
            for (int r = 0; r < 4; ++r)
                for (int t = 0; t < 4; ++t) a4[r][t] = t < T ? acc[r][t] : _mm512_setzero_ps();
            const __m512 s4 = reduce16<FMT>(a4);
            const __mmask8 m = (__mmask8)((1u << rn) - 1);
            _mm_mask_storeu_ps(y, m, _mm512_castps512_ps128(s4));
            _mm_mask_storeu_ps(y + ldy, m, _mm512_extractf32x4_ps(s4, 1));
            if (T == 3) _mm_mask_storeu_ps(y + 2 * ldy, m, _mm512_extractf32x4_ps(s4, 2));
            return;
        }
    }
    const __m512i unp = unperm<FMT>();
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < rn; ++r) y[(size_t)t * ldy + r] = hsum16(_mm512_permutexvar_ps(unp, acc[r][t]));
}

template <int FMT, int T, bool X4>
inline void run_tile_t(Scratch& S, int nc, int pn, const int8_t* const xq[], const float* const dx[], int rn, float* y, size_t ldy,
                       const float* const xs[]) {
    tile_kernel<FMT, T, X4>(S, nc, pn, xq, dx, rn, y, ldy, xs);
}

template <int FMT, bool X4 = false>
inline void run_tile(Scratch& S, int tn, int nc, int pn, const int8_t* const xq[], const float* const dx[], int rn, float* y, size_t ldy,
                     const float* const xs[] = nullptr) {
    switch (tn) {
        case 1: run_tile_t<FMT, 1, X4>(S, nc, pn, xq, dx, rn, y, ldy, xs); break;
        case 2: run_tile_t<FMT, 2, X4>(S, nc, pn, xq, dx, rn, y, ldy, xs); break;
        case 3: run_tile_t<FMT, 3, X4>(S, nc, pn, xq, dx, rn, y, ldy, xs); break;
        case 4: run_tile_t<FMT, 4, X4>(S, nc, pn, xq, dx, rn, y, ldy, xs); break;
        default: run_tile_t<FMT, 5, X4>(S, nc, pn, xq, dx, rn, y, ldy, xs); break;
    }
}

// ---------------------------------------------------------------- driver
// Tokens i = 0..n-1 use activation row (arows ? arows[i] : arow0 + i) and write to y + i*ldy (+ row).
template <int FMT, int ENC, bool X4 = false>
void gemm_core(const char* w, size_t rs, const GemmAct& a, const int32_t* arows, int arow0, int n, int k, float* y, size_t ldy, int nrows,
               int ngroup) {
    if (n <= 0 || nrows <= 0) return;
    Scratch& S = scratch();
    const int nc = k / 64, pn = a.dxn;
    const size_t rowbytes = FMT == GEMM_FMT_Q8 ? row_bytes_q8_0(k) : row_bytes_nvfp4(k);
    const int TM = std::min(std::max(g_gemm_tile, 1), TMAX);
    const bool pf = g_gemm_prefetch != 0;
    if (ngroup < TM) ngroup = TM;
    for (int g0 = 0; g0 < n; g0 += ngroup) {
        const int g1 = std::min(n, g0 + ngroup);
        const int ntiles = (g1 - g0 + TM - 1) / TM;
        // the first rows of the first panel (later panels are prefetched while the previous one computes)
        for (int r0 = 0; r0 < nrows; r0 += 4) {
            const int rn = std::min(4, nrows - r0);
            for (int j = 0; j < rn; ++j) unpack_row<FMT, ENC, X4>(w + (size_t)(r0 + j) * rs, k, S, j);
            const int nxt = std::min(4, nrows - (r0 + 4));
            int ti = 0;
            for (int t0 = g0; t0 < g1; ++ti) {
                // tile width: TM, except that a 1-token tail (9, 13, 17 pairs: the common MoE expert sizes) is absorbed into one 5-token
                // tile (a T=1 tile pays the 4-row P phase and the per-tile epilogue for a single token). Results do not depend on the split.
                int tn = std::min(TM, g1 - t0);
                if (TM == 4 && g_gemm_tail5 && g1 - t0 >= 5 && (g1 - t0) % 4 == 1) tn = 5;
                const int t0_next = t0 + tn;
                const int8_t* xq[TMAX];
                const float* dx[TMAX];
                const float* xs[TMAX];
                for (int j = 0; j < tn; ++j) {
                    const size_t row = (size_t)(arows ? arows[t0 + j] : arow0 + t0 + j);
                    xq[j] = a.xq + row * (size_t)k;
                    dx[j] = a.dx + row * (size_t)pn;
                    if constexpr (X4) xs[j] = a.xs + row * (size_t)pn;  // Xs has pn = pad16(K/16) floats per token, like dx
                }
                run_tile<FMT, X4>(S, tn, nc, pn, xq, dx, rn, y + (size_t)t0 * ldy + r0, ldy, xs);
                // prefetch the next panel's rows, spread over the tiles of this panel
                if (pf && nxt > 0) {
                    const int tix = std::min(ti, ntiles - 1);
                    const size_t b0 = (rowbytes * (size_t)tix / (size_t)ntiles) & ~(size_t)63;
                    const size_t b1 = (tix + 1 == ntiles || t0_next >= g1) ? rowbytes : (rowbytes * (size_t)(tix + 1) / (size_t)ntiles) & ~(size_t)63;
                    for (int j = 0; j < nxt; ++j) {
                        const char* base = w + (size_t)(r0 + 4 + j) * rs;
                        if (g_gemm_prefetch == 2)
                            for (size_t o = b0; o < b1; o += 64) _mm_prefetch(base + o, _MM_HINT_T0);
                        else
                            for (size_t o = b0; o < b1; o += 64) _mm_prefetch(base + o, _MM_HINT_T1);
                    }
                }
                t0 = t0_next;
            }
        }
    }
}

template <int FMT, int ENC = ENC_GGUF, bool X4 = false>
void gemm_dense(const void* w, size_t rs, const GemmAct& a, int t0, int t1, int k, float* y, size_t ldy, int nrows, const char* name) {
    if (nrows <= 0 || t1 <= t0) return;
    if (k <= 0 || k % 64 || k > KMAX) bad_arg(name, "K", k);
    if (!fmt_ok(FMT, X4, a.fmt) || a.k != k) bad_arg(name, "activation pack (fmt/k mismatch)", a.fmt);
    if (t0 < 0 || t1 > a.nt) bad_arg(name, "token range", t1);
    gemm_core<FMT, ENC, X4>((const char*)w, rs, a, nullptr, t0, t1 - t0, k, y + (size_t)t0 * ldy, ldy, nrows, g_gemm_group);
}

template <int FMT, int ENC = ENC_GGUF, bool X4 = false>
void moe_core(const void* w, size_t es, size_t rs, const GemmAct& a, const int32_t* act_rows, const MoeRoute& r, const MoeItem& it, int k,
              float* y, size_t ldy, const char* name) {
    if (k <= 0 || k % 64 || k > KMAX) bad_arg(name, "K", k);
    if (!fmt_ok(FMT, X4, a.fmt) || a.k != k) bad_arg(name, "activation pack (fmt/k mismatch)", a.fmt);
    const int p0 = r.offset[it.expert], p1 = r.offset[it.expert + 1];
    const int n = p1 - p0;
    if (n <= 0 || it.row1 <= it.row0) return;
    const char* wp = (const char*)w + (size_t)it.expert * es + (size_t)it.row0 * rs;
    gemm_core<FMT, ENC, X4>(wp, rs, a, act_rows ? act_rows + p0 : nullptr, act_rows ? 0 : p0, n, k, y + (size_t)p0 * ldy + it.row0, ldy,
                            it.row1 - it.row0, g_gemm_group);
}

}  // namespace

// Benchmark-only helper (bench_gemm --what unpack): unpacks rows [0, nrows) of an NVFP4 matrix `reps` times into the thread-local scratch and
// returns the wall time in ns. enc: 0 = GGUF scale bytes, 1 = re-encoded (nvfp4r), 2 = x4 prototype unpack (re-encoded, no sign masks).
double gemm_bench_unpack_ns(const void* w, size_t rs, int k, int nrows, int enc, int reps) {
    Scratch& S = scratch();
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < reps; ++it)
        for (int r = 0; r < nrows; ++r) {
            const char* row = (const char*)w + (size_t)r * rs;
            if (enc == 0) unpack_row<GEMM_FMT_NVFP4, ENC_GGUF>(row, k, S, r & 3);
            else if (enc == 1) unpack_row<GEMM_FMT_NVFP4, ENC_R>(row, k, S, r & 3);
            else unpack_row<GEMM_FMT_NVFP4, ENC_R, true>(row, k, S, r & 3);
        }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    asm volatile("" ::: "memory");
    return (double)(t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec);
}

// ---------------------------------------------------------------- public API
size_t gemm_act_bytes(int fmt, int nt, int k) {
    return align64((size_t)nt * (size_t)k) + (size_t)nt * (size_t)dxn_of(fmt, k) * 4 + (size_t)nt * (size_t)xsn_of(fmt, k) * 4 +
           (size_t)nt * (size_t)xbn_of(fmt, k) * 4;
}

GemmAct gemm_act_view(int fmt, const void* buf, int nt, int k) {
    GemmAct a;
    a.fmt = fmt;
    a.nt = nt;
    a.k = k;
    a.dxn = dxn_of(fmt, k);
    a.xq = (const int8_t*)buf;
    a.dx = (const float*)((const char*)buf + align64((size_t)nt * (size_t)k));
    a.xs = fmt == GEMM_FMT_NVFP4X4 ? a.dx + (size_t)nt * (size_t)a.dxn : nullptr;
    a.xb = fmt == GEMM_FMT_I16 ? (const int32_t*)(a.dx + (size_t)nt * (size_t)a.dxn) : nullptr;
    return a;
}

void gemm_pack_act(int fmt, const block_q8_0* x, int t0, int t1, int nt, int k, void* buf) {
    if (k <= 0 || k % 64 || k > KMAX) bad_arg("gemm_pack_act", "K", k);
    const GemmAct a = gemm_act_view(fmt, buf, nt, k);
    const int nb = k / 32;
    for (int t = t0; t < t1; ++t)
        pack_one(fmt, x + (size_t)t * nb, k, const_cast<int8_t*>(a.xq) + (size_t)t * k, const_cast<float*>(a.dx) + (size_t)t * a.dxn, a.dxn,
                 a.xs ? const_cast<float*>(a.xs) + (size_t)t * a.dxn : nullptr, a.xb ? const_cast<int32_t*>(a.xb) + (size_t)t * a.dxn : nullptr);
}

void gemm_quantize_pack_act(int fmt, const float* x, size_t ldx, int t0, int t1, int nt, int k, void* buf) {
    if (k <= 0 || k % 64 || k > KMAX) bad_arg("gemm_quantize_pack_act", "K", k);
    alignas(64) block_q8_0 tmp[KMAX / 32];
    const GemmAct a = gemm_act_view(fmt, buf, nt, k);
    for (int t = t0; t < t1; ++t) {
        quantize_row_q8_0(x + (size_t)t * ldx, tmp, k);
        pack_one(fmt, tmp, k, const_cast<int8_t*>(a.xq) + (size_t)t * k, const_cast<float*>(a.dx) + (size_t)t * a.dxn, a.dxn,
                 a.xs ? const_cast<float*>(a.xs) + (size_t)t * a.dxn : nullptr, a.xb ? const_cast<int32_t*>(a.xb) + (size_t)t * a.dxn : nullptr);
    }
}

void gemm_q8_0(const void* w, size_t rs, const GemmAct& a, int t0, int t1, int k, float* y, size_t ldy, int nrows) {
    gemm_dense<GEMM_FMT_Q8>(w, rs, a, t0, t1, k, y, ldy, nrows, "gemm_q8_0");
}
void gemm_nvfp4(const void* w, size_t rs, const GemmAct& a, int t0, int t1, int k, float* y, size_t ldy, int nrows) {
    gemm_dense<GEMM_FMT_NVFP4>(w, rs, a, t0, t1, k, y, ldy, nrows, "gemm_nvfp4");
}
void gemm_nvfp4r(const void* w, size_t rs, const GemmAct& a, int t0, int t1, int k, float* y, size_t ldy, int nrows) {
    gemm_dense<GEMM_FMT_NVFP4, ENC_R>(w, rs, a, t0, t1, k, y, ldy, nrows, "gemm_nvfp4r");
}
void gemm_nvfp4r_x4(const void* w, size_t rs, const GemmAct& a, int t0, int t1, int k, float* y, size_t ldy, int nrows) {
    gemm_dense<GEMM_FMT_NVFP4, ENC_R, true>(w, rs, a, t0, t1, k, y, ldy, nrows, "gemm_nvfp4r_x4");
}

void gemm_moe_q8_0(const void* w, size_t es, size_t rs, const GemmAct& a, const int32_t* act_rows, const MoeRoute& r, const MoeItem& it, int k,
                   float* y, size_t ldy) {
    moe_core<GEMM_FMT_Q8>(w, es, rs, a, act_rows, r, it, k, y, ldy, "gemm_moe_q8_0");
}
void gemm_moe_nvfp4(const void* w, size_t es, size_t rs, const GemmAct& a, const int32_t* act_rows, const MoeRoute& r, const MoeItem& it,
                    int k, float* y, size_t ldy) {
    moe_core<GEMM_FMT_NVFP4>(w, es, rs, a, act_rows, r, it, k, y, ldy, "gemm_moe_nvfp4");
}
void gemm_moe_nvfp4r(const void* w, size_t es, size_t rs, const GemmAct& a, const int32_t* act_rows, const MoeRoute& r, const MoeItem& it,
                     int k, float* y, size_t ldy) {
    moe_core<GEMM_FMT_NVFP4, ENC_R>(w, es, rs, a, act_rows, r, it, k, y, ldy, "gemm_moe_nvfp4r");
}
void gemm_moe_nvfp4r_x4(const void* w, size_t es, size_t rs, const GemmAct& a, const int32_t* act_rows, const MoeRoute& r, const MoeItem& it,
                        int k, float* y, size_t ldy) {
    moe_core<GEMM_FMT_NVFP4, ENC_R, true>(w, es, rs, a, act_rows, r, it, k, y, ldy, "gemm_moe_nvfp4r_x4");
}

void moe_build_route(const int32_t* ids, int n_tokens, int topk, int n_experts, MoeRoute& r) {
    r.n_experts = n_experts;
    r.n_tokens = n_tokens;
    r.topk = topk;
    const long np = (long)n_tokens * topk;
    r.offset.assign((size_t)n_experts + 1, 0);
    for (long i = 0; i < np; ++i) {
        if (ids[i] < 0 || ids[i] >= n_experts) bad_arg("moe_build_route", "expert id", ids[i]);
        r.offset[(size_t)ids[i] + 1]++;
    }
    for (int e = 0; e < n_experts; ++e) r.offset[(size_t)e + 1] += r.offset[(size_t)e];
    std::vector<int32_t> cur(r.offset.begin(), r.offset.end() - 1);
    r.pair_token.assign((size_t)np, 0);
    r.pair_slot.assign((size_t)np, 0);
    for (int t = 0; t < n_tokens; ++t)
        for (int j = 0; j < topk; ++j) {
            const int e = ids[(size_t)t * topk + j];
            const int p = cur[(size_t)e]++;
            r.pair_token[(size_t)p] = t;
            r.pair_slot[(size_t)p] = t * topk + j;
        }
}

void moe_work_items(const MoeRoute& r, int nrows, int chunk_rows, std::vector<MoeItem>& out) {
    out.clear();
    if (chunk_rows < 4) chunk_rows = 4;
    chunk_rows &= ~3;
    std::vector<int> order;
    for (int e = 0; e < r.n_experts; ++e)
        if (r.offset[(size_t)e + 1] > r.offset[(size_t)e]) order.push_back(e);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return (r.offset[(size_t)a + 1] - r.offset[(size_t)a]) > (r.offset[(size_t)b + 1] - r.offset[(size_t)b]);
    });
    for (int e : order)
        for (int r0 = 0; r0 < nrows; r0 += chunk_rows) out.push_back(MoeItem{e, r0, std::min(nrows, r0 + chunk_rows)});
}

}  // namespace ink

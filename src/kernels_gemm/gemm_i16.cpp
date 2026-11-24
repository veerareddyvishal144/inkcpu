// gemm_i16.cpp (agent P2, optimisation wave 2, 2026-11-24): int16-accumulation NVFP4 x Q8_0 GEMM for PREFILL PROMPT CHUNKS.
// Design: docs/KERNELS_GEMM.md section 9, docs/research/wave2_compute.md (approach #1). NUMERICS-DEVIATING from gemv_nvfp4 (fewer roundings,
// different bits): never use it where bit-identity with the decode path is required (verify, decode).
//
// Layout. A PANEL is 16 weight rows. Per K SEGMENT (g_gemm_i16_seg super-blocks of 64 k) the panel is unpacked into
//   W[sb*4 + q] : one zmm = 16 rows (dword lanes) x 4 consecutive k (bytes) as u8 = kv + 12 (0..24), for sub-block sb (16 k) and quad q
//   SW[sb]      : 16 floats = the 16 rows' UE4M3 scale of sub-block sb (nvfp4r-encoded bytes decoded with the vector form of nvfp4_repack.h)
// Unpack per (16 rows, super-block): 16 ymm loads of the 32 nibble bytes -> 16x8 dword transpose (8 vinserti64x4 + 24 shuffles) -> T_d (lane r =
// dword d of row r); sub-block s uses T_{2s} (low nibbles = k 0..3, high = k 8..11) and T_{2s+1} (k 4..7 / 12..15): vpand / vpsrlw+vpand, vpshufb
// with the kv+12 table = 16 output zmm; scales: one vpgatherdd of the 4 scale bytes of the 16 rows, then per sub-block 2 shifts + 1 vmulps.
// ~90 vector uops per 16 rows x 64 k, amortised over ALL tokens of the call (the panel is unpacked once per segment, every token tile re-reads it).
//
// Micro-kernel: 16 rows x N tokens (N <= g_gemm_i16_tile), per Q8_0 block b (sub-blocks 2b, 2b+1) and token t:
//   p  = vpmaddubsw(W[2b][q], bcast4(x_t[..]))  q = 0..3, summed with vpaddw           (|pair| <= 2*24*128 = 6144, |sum of 4| <= 24576: exact)
//   I0 = vpmaddwd(p, ones) - bcast(xb_t[2b])                                            (= EXACT sum_k w_k x_k over the 16 k of sub-block 2b)
//   I1 = same for sub-block 2b+1
//   acc_t = fma( fma(cvt(I1), SW[2b+1], cvt(I0) * SW[2b]), bcast(dx_t[b]), acc_t )
// = 8 maddubs + 6 paddw + 2 maddwd + 2 psubd + 2 cvt + 1 mul + 2 fma = 23 p0/p5 uops per 512 MACs (2.875 per 64; the exact kernel: 5).
// The activation operands are vpbroadcastd loads (load ports only) straight from the lane-permuted NVFP4 pack: the 4 k of (sub-block s, quad q)
// of chunk c sit at byte 64c + 4(4q + s) (gemm.cpp idx_nv_x); dx_t[b] is the NVFP4 pack's dx[4c + 2h] (block b = 2c + h); xb comes from the I16 pack.
// Results are independent of N, the segment length, the row chunking and the thread: the float chain of a (row, token) is fixed (sequential over
// blocks); partial accumulators between segments are kept in y itself (the stored float IS the register value).
#include <immintrin.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ink/kernels_gemm.h"
#include "ink/nvfp4_repack.h"

namespace ink {

int g_gemm_i16_tile = 8;
int g_gemm_i16_seg = 16;
int g_gemm_i16_pf = 1;
int g_gemm_i16_uniform = 1;

namespace {

constexpr int KMAX = 8192;
constexpr int SBMAX = KMAX / 64;  // super-blocks per row
constexpr int NMAX = 12;          // max tokens per register tile (acc + 8 W + 2 SW + temps must fit 32 zmm)

[[noreturn]] void bad_arg(const char* fn, const char* what, long v) {
    std::fprintf(stderr, "ink::%s: unsupported %s=%ld\n", fn, what, v);
    std::abort();
}

struct alignas(64) PanelI16 {
    alignas(64) uint8_t w[SBMAX * 16][64];  // [sb*4 + q] -> 16 rows x 4 k (u8 = kv + 12); 16 zmm per super-block = 128 KB at KMAX
    alignas(64) float sw[SBMAX * 4][16];   // [sb] -> 16 rows' sub-block scales                 32 KB at KMAX
};
PanelI16& panel() {
    static thread_local PanelI16 p;
    return p;
}

// Weight rows of a panel: 16 pointers (rows beyond the panel's row count point at the last valid row; their lanes are masked on store).
struct Rows16 {
    const uint8_t* p[16];
    size_t rs = 0;         // row stride; uniform == true means p[r] == p[0] + r * rs for all 16 rows (full panel: the fast unpack path)
    bool uniform = false;
};

// ---------------------------------------------------------------- unpack of one super-block (64 k) of 16 rows
struct UnpackConst {
    __m512i lut, m0f, mhi, mb0, mb1, mb2;
    __m512 pow;
    __m512i gidx;  // gather index: byte offset of row r's data relative to row 0's
};

template <bool UNIFORM>
__attribute__((always_inline)) inline void unpack_sb16(const Rows16& R, size_t off, uint8_t (*W)[64], float (*SW)[16], const UnpackConst& C) {
    // Z_i = [row i | row i+4] (i = 0..3), Z_{4+i} = [row 8+i | row 12+i]: after the 8x8 transposes of both 256-bit halves and the 128-bit lane
    // shuffles below, T_d holds dword d of rows 0..15 in natural lane order. UNIFORM: the 16 row addresses are base + r*rs (4 base pointers +
    // stride multiples in registers, no pointer reloads: the unpack is front-end bound); else the pointer table (partial panels).
    __m512i Z[8];
    if constexpr (UNIFORM) {
        const uint8_t* q = R.p[0] + off + 4;
        const size_t s1 = R.rs, s4 = 4 * s1, s8 = 8 * s1, s12 = 12 * s1;
        const uint8_t* q1 = q + s1;
        const uint8_t* q2 = q1 + s1;
        const uint8_t* q3 = q2 + s1;
        Z[0] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)q)), _mm256_loadu_si256((const __m256i*)(q + s4)), 1);
        Z[1] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)q1)), _mm256_loadu_si256((const __m256i*)(q1 + s4)), 1);
        Z[2] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)q2)), _mm256_loadu_si256((const __m256i*)(q2 + s4)), 1);
        Z[3] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)q3)), _mm256_loadu_si256((const __m256i*)(q3 + s4)), 1);
        Z[4] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(q + s8))), _mm256_loadu_si256((const __m256i*)(q + s12)), 1);
        Z[5] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(q1 + s8))), _mm256_loadu_si256((const __m256i*)(q1 + s12)), 1);
        Z[6] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(q2 + s8))), _mm256_loadu_si256((const __m256i*)(q2 + s12)), 1);
        Z[7] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(q3 + s8))), _mm256_loadu_si256((const __m256i*)(q3 + s12)), 1);
    } else {
        for (int i = 0; i < 4; ++i) {
            Z[i] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(R.p[i] + off + 4))),
                                      _mm256_loadu_si256((const __m256i*)(R.p[i + 4] + off + 4)), 1);
            Z[4 + i] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(R.p[8 + i] + off + 4))),
                                          _mm256_loadu_si256((const __m256i*)(R.p[12 + i] + off + 4)), 1);
        }
    }
    __m512i A[8], B[8], T[8];
    for (int i = 0; i < 4; ++i) {
        A[2 * i] = _mm512_unpacklo_epi32(Z[2 * i], Z[2 * i + 1]);
        A[2 * i + 1] = _mm512_unpackhi_epi32(Z[2 * i], Z[2 * i + 1]);
    }
    for (int g = 0; g < 2; ++g) {  // g = 0: rows 0-7 (Z0..Z3), g = 1: rows 8-15 (Z4..Z7)
        const __m512i a0 = A[4 * g], a1 = A[4 * g + 1], a2 = A[4 * g + 2], a3 = A[4 * g + 3];
        B[4 * g + 0] = _mm512_unpacklo_epi64(a0, a2);  // d0 | d4
        B[4 * g + 1] = _mm512_unpackhi_epi64(a0, a2);  // d1 | d5
        B[4 * g + 2] = _mm512_unpacklo_epi64(a1, a3);  // d2 | d6
        B[4 * g + 3] = _mm512_unpackhi_epi64(a1, a3);  // d3 | d7
    }
    for (int d = 0; d < 4; ++d) {
        T[d] = _mm512_shuffle_i32x4(B[d], B[4 + d], 0x88);      // lanes (L0, L2 | L0, L2): dword d of rows 0-3, 4-7, 8-11, 12-15
        T[4 + d] = _mm512_shuffle_i32x4(B[d], B[4 + d], 0xDD);  // lanes (L1, L3 | L1, L3): dword 4+d
    }
    // sub-block s: T[2s] = bytes 0..3 (low nibbles k 0..3 -> quad 0, high nibbles k 8..11 -> quad 2), T[2s+1] = bytes 4..7 (quads 1 and 3)
    for (int s = 0; s < 4; ++s) {
        const __m512i t0 = T[2 * s], t1 = T[2 * s + 1];
        _mm512_store_si512((void*)W[4 * s + 0], _mm512_shuffle_epi8(C.lut, _mm512_and_si512(t0, C.m0f)));
        _mm512_store_si512((void*)W[4 * s + 1], _mm512_shuffle_epi8(C.lut, _mm512_and_si512(t1, C.m0f)));
        _mm512_store_si512((void*)W[4 * s + 2], _mm512_shuffle_epi8(C.lut, _mm512_and_si512(_mm512_srli_epi16(t0, 4), C.m0f)));
        _mm512_store_si512((void*)W[4 * s + 3], _mm512_shuffle_epi8(C.lut, _mm512_and_si512(_mm512_srli_epi16(t1, 4), C.m0f)));
    }
    // scales: D lane r = the 4 re-encoded scale bytes of row r; byte s -> bits 20..27 -> * 2^116 (nvfp4r_decode16, same floats)
    // (isolate byte s, then shift it to bits 20..27: the other three bytes must not leak into the mantissa / exponent bits)
    const __m512i D = _mm512_i32gather_epi32(C.gidx, (const void*)(R.p[0] + off), 1);
    _mm512_store_ps(SW[0], _mm512_mul_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_and_si512(D, C.mb0), 20)), C.pow));
    _mm512_store_ps(SW[1], _mm512_mul_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_and_si512(D, C.mb1), 12)), C.pow));
    _mm512_store_ps(SW[2], _mm512_mul_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_and_si512(D, C.mb2), 4)), C.pow));
    _mm512_store_ps(SW[3], _mm512_mul_ps(_mm512_castsi512_ps(_mm512_and_si512(_mm512_srli_epi32(D, 4), C.mhi)), C.pow));
}

// Unpacks super-blocks [c0, c0 + nsb) of the 16 rows into the panel (W index relative to the segment start).
__attribute__((noinline)) void unpack_segment(const Rows16& R, int c0, int nsb, PanelI16& P) {
    UnpackConst C;
    C.lut = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    C.m0f = _mm512_set1_epi8(0x0F);
    C.mhi = _mm512_set1_epi32(0x0FF00000);
    C.mb0 = _mm512_set1_epi32(0x000000FF);
    C.mb1 = _mm512_set1_epi32(0x0000FF00);
    C.mb2 = _mm512_set1_epi32(0x00FF0000);
    C.pow = _mm512_set1_ps(kNvfp4rPow);
    alignas(64) int32_t gi[16];
    for (int r = 0; r < 16; ++r) gi[r] = (int32_t)(R.p[r] - R.p[0]);
    C.gidx = _mm512_load_si512((const void*)gi);
    if (R.uniform) {
        for (int i = 0; i < nsb; ++i) unpack_sb16<true>(R, (size_t)(c0 + i) * sizeof(block_nvfp4), P.w + 16 * i, P.sw + 4 * i, C);
    } else {
        for (int i = 0; i < nsb; ++i) unpack_sb16<false>(R, (size_t)(c0 + i) * sizeof(block_nvfp4), P.w + 16 * i, P.sw + 4 * i, C);
    }
}

// ---------------------------------------------------------------- micro-kernel: 16 rows x N tokens over one segment
struct TileArgs {
    const int8_t* xq[NMAX];   // token quant rows (lane-permuted NVFP4 pack)
    const int32_t* xb[NMAX];  // 12 * sub-block sums
    const float* dx[NMAX];    // NVFP4-pack dx (block b of chunk c at 4c + 2(b&1))
    float* y[NMAX];           // output / partial-accumulator rows (16 floats at the panel's rows)
    __mmask16 km;             // rows of the panel that exist
    bool init;                // first segment: accumulators start at 0, else continue from y
    // software prefetch of the next segment's weight bytes (first tile of a segment only): rows + byte range, or pf_n = 0
    const Rows16* pf_rows;
    size_t pf_off0, pf_off1;
};

inline __m512i bc4(const int8_t* p) {  // vpbroadcastd zmm, m32 (load port only)
    int32_t v;
    std::memcpy(&v, p, 4);
    return _mm512_set1_epi32(v);
}

template <int N>
__attribute__((noinline)) void tile_i16(const PanelI16& P, int c0, int nsb, const TileArgs& A) {
    const __m512i ones = _mm512_set1_epi16(1);
    __m512 acc[N];
    _Pragma("GCC unroll 16") for (int j = 0; j < N; ++j) acc[j] = A.init ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(A.km, A.y[j]);
    const int nblk = 2 * nsb;
    // prefetch plan: iteration b < 16 prefetches row b's byte range of the next segment (T1: into L2)
    const int pf_lines = A.pf_rows ? (int)((A.pf_off1 - A.pf_off0 + 63) / 64) + 1 : 0;
    for (int b = 0; b < nblk; ++b) {
        const __m512i* Wp = (const __m512i*)P.w[8 * b];
        const __m512i W0 = _mm512_load_si512(Wp + 0), W1 = _mm512_load_si512(Wp + 1), W2 = _mm512_load_si512(Wp + 2), W3 = _mm512_load_si512(Wp + 3);
        const __m512i W4 = _mm512_load_si512(Wp + 4), W5 = _mm512_load_si512(Wp + 5), W6 = _mm512_load_si512(Wp + 6), W7 = _mm512_load_si512(Wp + 7);
        const __m512 sw0 = _mm512_load_ps(P.sw[2 * b]), sw1 = _mm512_load_ps(P.sw[2 * b + 1]);
        const int c = c0 + (b >> 1), h = b & 1;        // chunk (super-block) and half (Q8_0 block within it)
        const size_t xo = (size_t)c * 64 + (size_t)h * 8;  // byte offset of (sub-block 2h, quad 0) in the token's quants
        const int sbi = 4 * c + 2 * h;                   // index of sub-block 2h in xb / dx
        if (pf_lines && b < 16) {
            const uint8_t* q = A.pf_rows->p[b] + A.pf_off0;
            for (int l = 0; l < pf_lines; ++l) _mm_prefetch((const char*)q + 64 * l, _MM_HINT_T1);
        }
        _Pragma("GCC unroll 16") for (int j = 0; j < N; ++j) {
            const int8_t* xp = A.xq[j] + xo;
            __m512i p = _mm512_maddubs_epi16(W0, bc4(xp));
            p = _mm512_add_epi16(p, _mm512_maddubs_epi16(W1, bc4(xp + 16)));
            p = _mm512_add_epi16(p, _mm512_maddubs_epi16(W2, bc4(xp + 32)));
            p = _mm512_add_epi16(p, _mm512_maddubs_epi16(W3, bc4(xp + 48)));
            const __m512i i0 = _mm512_sub_epi32(_mm512_madd_epi16(p, ones), _mm512_set1_epi32(A.xb[j][sbi]));
            __m512i r = _mm512_maddubs_epi16(W4, bc4(xp + 4));
            r = _mm512_add_epi16(r, _mm512_maddubs_epi16(W5, bc4(xp + 20)));
            r = _mm512_add_epi16(r, _mm512_maddubs_epi16(W6, bc4(xp + 36)));
            r = _mm512_add_epi16(r, _mm512_maddubs_epi16(W7, bc4(xp + 52)));
            const __m512i i1 = _mm512_sub_epi32(_mm512_madd_epi16(r, ones), _mm512_set1_epi32(A.xb[j][sbi + 1]));
            __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(i0), sw0);
            f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i1), sw1, f);
            acc[j] = _mm512_fmadd_ps(f, _mm512_set1_ps(A.dx[j][sbi]), acc[j]);
        }
    }
    _Pragma("GCC unroll 16") for (int j = 0; j < N; ++j) _mm512_mask_storeu_ps(A.y[j], A.km, acc[j]);
}

void run_tile(int n, const PanelI16& P, int c0, int nsb, const TileArgs& A) {
    switch (n) {
        case 1: tile_i16<1>(P, c0, nsb, A); break;
        case 2: tile_i16<2>(P, c0, nsb, A); break;
        case 3: tile_i16<3>(P, c0, nsb, A); break;
        case 4: tile_i16<4>(P, c0, nsb, A); break;
        case 5: tile_i16<5>(P, c0, nsb, A); break;
        case 6: tile_i16<6>(P, c0, nsb, A); break;
        case 7: tile_i16<7>(P, c0, nsb, A); break;
        case 8: tile_i16<8>(P, c0, nsb, A); break;
        case 9: tile_i16<9>(P, c0, nsb, A); break;
        case 10: tile_i16<10>(P, c0, nsb, A); break;
        case 11: tile_i16<11>(P, c0, nsb, A); break;
        default: tile_i16<12>(P, c0, nsb, A); break;
    }
}

// ---------------------------------------------------------------- driver: panels of 16 rows x K segments x balanced token tiles
// Tokens i = 0..n-1 use activation row (arows ? arows[i] : arow0 + i) and write y + i*ldy (+ row).
void i16_core(const uint8_t* w, size_t rs, const GemmAct& a, const int32_t* arows, int arow0, int n, int k, float* y, size_t ldy, int nrows) {
    if (n <= 0 || nrows <= 0) return;
    PanelI16& P = panel();
    const int nsb_all = k / 64;
    const int seg = std::min(std::max(g_gemm_i16_seg, 1), SBMAX);
    const int NT = std::min(std::max(g_gemm_i16_tile, 1), NMAX);
    const int ntiles = (n + NT - 1) / NT, base = n / ntiles, rem = n % ntiles;  // balanced tiles (sizes base or base + 1)
    const bool pf = g_gemm_i16_pf != 0;
    Rows16 R, Rnext;
    for (int r0 = 0; r0 < nrows; r0 += 16) {
        const int rn = std::min(16, nrows - r0);
        for (int r = 0; r < 16; ++r) R.p[r] = w + (size_t)(r0 + std::min(r, rn - 1)) * rs;
        R.rs = rs;
        R.uniform = rn == 16 && g_gemm_i16_uniform != 0;
        const bool last_panel = r0 + 16 >= nrows;
        if (!last_panel) {
            const int rn2 = std::min(16, nrows - (r0 + 16));
            for (int r = 0; r < 16; ++r) Rnext.p[r] = w + (size_t)(r0 + 16 + std::min(r, rn2 - 1)) * rs;
            Rnext.rs = rs;
            Rnext.uniform = rn2 == 16 && g_gemm_i16_uniform != 0;
        }
        const __mmask16 km = rn == 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << rn) - 1);
        for (int c0 = 0; c0 < nsb_all; c0 += seg) {
            const int nsb = std::min(seg, nsb_all - c0);
            unpack_segment(R, c0, nsb, P);
            TileArgs A;
            A.km = km;
            A.init = c0 == 0;
            A.pf_rows = nullptr;
            A.pf_off0 = A.pf_off1 = 0;
            if (pf) {  // next segment of this panel, else the first segment of the next panel
                if (c0 + nsb < nsb_all) {
                    A.pf_rows = &R;
                    A.pf_off0 = (size_t)(c0 + nsb) * sizeof(block_nvfp4);
                    A.pf_off1 = (size_t)std::min(nsb_all, c0 + nsb + seg) * sizeof(block_nvfp4);
                } else if (!last_panel) {
                    A.pf_rows = &Rnext;
                    A.pf_off0 = 0;
                    A.pf_off1 = (size_t)std::min(nsb_all, seg) * sizeof(block_nvfp4);
                }
            }
            int t0 = 0;
            for (int ti = 0; ti < ntiles; ++ti) {
                const int tn = base + (ti < rem ? 1 : 0);
                for (int j = 0; j < tn; ++j) {
                    const size_t row = (size_t)(arows ? arows[t0 + j] : arow0 + t0 + j);
                    A.xq[j] = a.xq + row * (size_t)k;
                    A.xb[j] = a.xb + row * (size_t)a.dxn;
                    A.dx[j] = a.dx + row * (size_t)a.dxn;
                    A.y[j] = y + (size_t)(t0 + j) * ldy + r0;
                }
                run_tile(tn, P, c0, nsb, A);
                A.pf_rows = nullptr;  // prefetch once per segment (first tile)
                t0 += tn;
            }
        }
    }
}

void check_args(const char* name, const GemmAct& a, int k) {
    if (k <= 0 || k % 64 || k > KMAX) bad_arg(name, "K", k);
    if (a.fmt != GEMM_FMT_I16 || a.k != k || !a.xb) bad_arg(name, "activation pack (needs GEMM_FMT_I16 with the same K)", a.fmt);
}

}  // namespace

void gemm_nvfp4r_i16(const void* w, size_t rs, const GemmAct& a, int t0, int t1, int k, float* y, size_t ldy, int nrows) {
    if (nrows <= 0 || t1 <= t0) return;
    check_args("gemm_nvfp4r_i16", a, k);
    if (t0 < 0 || t1 > a.nt) bad_arg("gemm_nvfp4r_i16", "token range", t1);
    i16_core((const uint8_t*)w, rs, a, nullptr, t0, t1 - t0, k, y + (size_t)t0 * ldy, ldy, nrows);
}

void gemm_moe_nvfp4r_i16(const void* w, size_t es, size_t rs, const GemmAct& a, const int32_t* act_rows, const MoeRoute& r, const MoeItem& it, int k,
                         float* y, size_t ldy) {
    check_args("gemm_moe_nvfp4r_i16", a, k);
    const int p0 = r.offset[it.expert], p1 = r.offset[it.expert + 1];
    const int n = p1 - p0;
    if (n <= 0 || it.row1 <= it.row0) return;
    const uint8_t* wp = (const uint8_t*)w + (size_t)it.expert * es + (size_t)it.row0 * rs;
    i16_core(wp, rs, a, act_rows ? act_rows + p0 : nullptr, act_rows ? 0 : p0, n, k, y + (size_t)p0 * ldy + it.row0, ldy, it.row1 - it.row0);
}

}  // namespace ink

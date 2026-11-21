// Batched Q8_0 / NVFP4 x Q8_0 GEMM (AVX-512 F/BW/DQ, no VNNI) + MoE grouped driver. Owner: agent G. Design: docs/KERNELS_GEMM.md.
//
// Bit-identity with B1's gemv kernels: every (row, token) pair keeps B1's 16 float accumulator lanes; chunk c (64 weights) adds
//   acc_l = fmaf((float)I_l, S_l, acc_l),  I_l = exact int sum of 4 products, S_l = fl(dw * dx)  (one float multiply)
// and the result is hsum16 (same tree) of the accumulators in B1's lane order. Only the PHYSICAL lane order is different (a fixed
// permutation chosen so that the scale vector is a plain memory broadcast); one vpermps restores B1's order before hsum16.
//   Q8_0 : physical dword d = (block d&1, group d>>1)           logical lane l = (l&7)*2 + (l>>3) <-> physical;   S = [s0 s1] x8
//   NVFP4: physical dword d = (sub-block d&3, group d>>2 = q)   logical lane l <-> physical 8*(l>>3) + 4*(l&1) + ((l>>1)&3); S = [sc0..sc3] x4
#include <immintrin.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ink/kernels_gemm.h"

namespace ink {

int g_gemm_group = 64;
int g_gemm_tile = 4;
int g_gemm_prefetch = 1;

namespace {

constexpr int KMAX = 8192;
constexpr int DWMAX = KMAX / 16;  // max floats of a scale array (NVFP4: K/16)
constexpr int TMAX = 5;           // tokens per register tile (4 rows x TMAX tokens)

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
void pack_one(int fmt, const block_q8_0* xr, int k, int8_t* q, float* d, int dxn) {
    const int nc = k / 64;
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

template <int FMT>
inline void unpack_row(const char* row, int k, Scratch& S, int j) {
    if constexpr (FMT == GEMM_FMT_Q8)
        unpack_row_q8(row, k, S.mag[j], S.sgn[j], S.dw[j], S.d16);
    else
        unpack_row_nvfp4(row, k, S.mag[j], S.sgn[j], S.dw[j]);
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

template <int FMT, int T>
void tile_kernel(Scratch& S, int nc, int pn, const int8_t* const xq[], const float* const dx[], int rn, float* y, size_t ldy) {
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
            const __mmask64 km = (__mmask64)S.sgn[r][c];
            for (int t = 0; t < T; ++t) {
                const __m512i X = _mm512_loadu_si512((const void*)(xp[t] + 64 * c));
                const __m512i XS = _mm512_mask_sub_epi8(X, km, zero, X);
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
    const __m512i unp = unperm<FMT>();
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < rn; ++r) y[(size_t)t * ldy + r] = hsum16(_mm512_permutexvar_ps(unp, acc[r][t]));
}

template <int FMT, int T>
inline void run_tile_t(Scratch& S, int nc, int pn, const int8_t* const xq[], const float* const dx[], int rn, float* y, size_t ldy) {
    tile_kernel<FMT, T>(S, nc, pn, xq, dx, rn, y, ldy);
}

template <int FMT>
inline void run_tile(Scratch& S, int tn, int nc, int pn, const int8_t* const xq[], const float* const dx[], int rn, float* y, size_t ldy) {
    switch (tn) {
        case 1: run_tile_t<FMT, 1>(S, nc, pn, xq, dx, rn, y, ldy); break;
        case 2: run_tile_t<FMT, 2>(S, nc, pn, xq, dx, rn, y, ldy); break;
        case 3: run_tile_t<FMT, 3>(S, nc, pn, xq, dx, rn, y, ldy); break;
        case 4: run_tile_t<FMT, 4>(S, nc, pn, xq, dx, rn, y, ldy); break;
        default: run_tile_t<FMT, 5>(S, nc, pn, xq, dx, rn, y, ldy); break;
    }
}

// ---------------------------------------------------------------- driver
// Tokens i = 0..n-1 use activation row (arows ? arows[i] : arow0 + i) and write to y + i*ldy (+ row).
template <int FMT>
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
            for (int j = 0; j < rn; ++j) unpack_row<FMT>(w + (size_t)(r0 + j) * rs, k, S, j);
            const int nxt = std::min(4, nrows - (r0 + 4));
            int ti = 0;
            for (int t0 = g0; t0 < g1; t0 += TM, ++ti) {
                const int tn = std::min(TM, g1 - t0);
                const int8_t* xq[TMAX];
                const float* dx[TMAX];
                for (int j = 0; j < tn; ++j) {
                    const size_t row = (size_t)(arows ? arows[t0 + j] : arow0 + t0 + j);
                    xq[j] = a.xq + row * (size_t)k;
                    dx[j] = a.dx + row * (size_t)pn;
                }
                run_tile<FMT>(S, tn, nc, pn, xq, dx, rn, y + (size_t)t0 * ldy + r0, ldy);
                // prefetch the next panel's rows, spread over the tiles of this panel
                if (pf && nxt > 0) {
                    const size_t b0 = (rowbytes * (size_t)ti / (size_t)ntiles) & ~(size_t)63;
                    const size_t b1 = ti + 1 == ntiles ? rowbytes : (rowbytes * (size_t)(ti + 1) / (size_t)ntiles) & ~(size_t)63;
                    for (int j = 0; j < nxt; ++j) {
                        const char* base = w + (size_t)(r0 + 4 + j) * rs;
                        for (size_t o = b0; o < b1; o += 64) _mm_prefetch(base + o, _MM_HINT_T1);
                    }
                }
            }
        }
    }
}

template <int FMT>
void gemm_dense(const void* w, size_t rs, const GemmAct& a, int t0, int t1, int k, float* y, size_t ldy, int nrows, const char* name) {
    if (nrows <= 0 || t1 <= t0) return;
    if (k <= 0 || k % 64 || k > KMAX) bad_arg(name, "K", k);
    if (a.fmt != FMT || a.k != k) bad_arg(name, "activation pack (fmt/k mismatch)", a.fmt);
    if (t0 < 0 || t1 > a.nt) bad_arg(name, "token range", t1);
    gemm_core<FMT>((const char*)w, rs, a, nullptr, t0, t1 - t0, k, y + (size_t)t0 * ldy, ldy, nrows, g_gemm_group);
}

template <int FMT>
void moe_core(const void* w, size_t es, size_t rs, const GemmAct& a, const int32_t* act_rows, const MoeRoute& r, const MoeItem& it, int k,
              float* y, size_t ldy, const char* name) {
    if (k <= 0 || k % 64 || k > KMAX) bad_arg(name, "K", k);
    if (a.fmt != FMT || a.k != k) bad_arg(name, "activation pack (fmt/k mismatch)", a.fmt);
    const int p0 = r.offset[it.expert], p1 = r.offset[it.expert + 1];
    const int n = p1 - p0;
    if (n <= 0 || it.row1 <= it.row0) return;
    const char* wp = (const char*)w + (size_t)it.expert * es + (size_t)it.row0 * rs;
    gemm_core<FMT>(wp, rs, a, act_rows ? act_rows + p0 : nullptr, act_rows ? 0 : p0, n, k, y + (size_t)p0 * ldy + it.row0, ldy,
                   it.row1 - it.row0, g_gemm_group);
}

}  // namespace

// ---------------------------------------------------------------- public API
size_t gemm_act_bytes(int fmt, int nt, int k) { return align64((size_t)nt * (size_t)k) + (size_t)nt * (size_t)dxn_of(fmt, k) * 4; }

GemmAct gemm_act_view(int fmt, const void* buf, int nt, int k) {
    GemmAct a;
    a.fmt = fmt;
    a.nt = nt;
    a.k = k;
    a.dxn = dxn_of(fmt, k);
    a.xq = (const int8_t*)buf;
    a.dx = (const float*)((const char*)buf + align64((size_t)nt * (size_t)k));
    return a;
}

void gemm_pack_act(int fmt, const block_q8_0* x, int t0, int t1, int nt, int k, void* buf) {
    if (k <= 0 || k % 64 || k > KMAX) bad_arg("gemm_pack_act", "K", k);
    const GemmAct a = gemm_act_view(fmt, buf, nt, k);
    const int nb = k / 32;
    for (int t = t0; t < t1; ++t)
        pack_one(fmt, x + (size_t)t * nb, k, const_cast<int8_t*>(a.xq) + (size_t)t * k, const_cast<float*>(a.dx) + (size_t)t * a.dxn, a.dxn);
}

void gemm_quantize_pack_act(int fmt, const float* x, size_t ldx, int t0, int t1, int nt, int k, void* buf) {
    if (k <= 0 || k % 64 || k > KMAX) bad_arg("gemm_quantize_pack_act", "K", k);
    alignas(64) block_q8_0 tmp[KMAX / 32];
    const GemmAct a = gemm_act_view(fmt, buf, nt, k);
    for (int t = t0; t < t1; ++t) {
        quantize_row_q8_0(x + (size_t)t * ldx, tmp, k);
        pack_one(fmt, tmp, k, const_cast<int8_t*>(a.xq) + (size_t)t * k, const_cast<float*>(a.dx) + (size_t)t * a.dxn, a.dxn);
    }
}

void gemm_q8_0(const void* w, size_t rs, const GemmAct& a, int t0, int t1, int k, float* y, size_t ldy, int nrows) {
    gemm_dense<GEMM_FMT_Q8>(w, rs, a, t0, t1, k, y, ldy, nrows, "gemm_q8_0");
}
void gemm_nvfp4(const void* w, size_t rs, const GemmAct& a, int t0, int t1, int k, float* y, size_t ldy, int nrows) {
    gemm_dense<GEMM_FMT_NVFP4>(w, rs, a, t0, t1, k, y, ldy, nrows, "gemm_nvfp4");
}

void gemm_moe_q8_0(const void* w, size_t es, size_t rs, const GemmAct& a, const int32_t* act_rows, const MoeRoute& r, const MoeItem& it, int k,
                   float* y, size_t ldy) {
    moe_core<GEMM_FMT_Q8>(w, es, rs, a, act_rows, r, it, k, y, ldy, "gemm_moe_q8_0");
}
void gemm_moe_nvfp4(const void* w, size_t es, size_t rs, const GemmAct& a, const int32_t* act_rows, const MoeRoute& r, const MoeItem& it,
                    int k, float* y, size_t ldy) {
    moe_core<GEMM_FMT_NVFP4>(w, es, rs, a, act_rows, r, it, k, y, ldy, "gemm_moe_nvfp4");
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

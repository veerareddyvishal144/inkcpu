// Fold of the Markov head correction into the lm_head (agent S1): lm_head . (hidden_out lat + b) = M lat + m0, M = lm_head . hidden_out ([200058 x 256]),
// m0 = lm_head . b. fp32 accumulation, computed once at create (rows split over the target's pool), cached on disk (content-hash key). docs/DRAFTER.md 3.
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "drafter_impl.h"

namespace ink {
namespace dr {

namespace {

constexpr int KC = 512;      // k block of the fold GEMM
constexpr int RB = 48;       // rows per chunk (8 row groups of 6; 200058 = 6 * 33343)

struct FoldBuf {
    alignas(64) float A[RB * KC];      // dequantised lm_head rows, one k block
    alignas(64) float C[RB * RANK];    // M rows of the chunk
    alignas(64) float m0[RB];
};

// 6 rows x 64 columns over kn steps: C[r][0..63] += sum_k A[r*lda + k] * B[k*64 + c]
INK_AI void micro6x64(const float* A, int lda, const float* B, int kn, float* C, int ldc) {
    __m512 acc[6][4];
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 4; ++c) acc[r][c] = _mm512_loadu_ps(C + (size_t)r * ldc + c * 16);
    for (int k = 0; k < kn; ++k) {
        const __m512 b0 = _mm512_loadu_ps(B + (size_t)k * 64), b1 = _mm512_loadu_ps(B + (size_t)k * 64 + 16),
                     b2 = _mm512_loadu_ps(B + (size_t)k * 64 + 32), b3 = _mm512_loadu_ps(B + (size_t)k * 64 + 48);
#pragma GCC unroll 6
        for (int r = 0; r < 6; ++r) {
            const __m512 a = _mm512_set1_ps(A[(size_t)r * lda + k]);
            acc[r][0] = _mm512_fmadd_ps(a, b0, acc[r][0]);
            acc[r][1] = _mm512_fmadd_ps(a, b1, acc[r][1]);
            acc[r][2] = _mm512_fmadd_ps(a, b2, acc[r][2]);
            acc[r][3] = _mm512_fmadd_ps(a, b3, acc[r][3]);
        }
    }
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 4; ++c) _mm512_storeu_ps(C + (size_t)r * ldc + c * 16, acc[r][c]);
}

uint64_t hash_range(const uint8_t* p, size_t n) {
    constexpr uint64_t K = 0xFF51AFD7ED558CCDull;
    uint64_t h[4] = {0x9E3779B97F4A7C15ull ^ n, 0xC2B2AE3D27D4EB4Full, 0x165667B19E3779F9ull, 0x27D4EB2F165667C5ull};
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        uint64_t w[4];
        std::memcpy(w, p + i, 32);
        for (int j = 0; j < 4; ++j) {
            h[j] = (h[j] ^ w[j]) * K;
            h[j] ^= h[j] >> 32;
        }
    }
    for (; i < n; ++i) {
        h[0] = (h[0] ^ p[i]) * K;
        h[0] ^= h[0] >> 32;
    }
    uint64_t r = 0x9E3779B97F4A7C15ull;
    for (int j = 0; j < 4; ++j) {
        r = (r ^ h[j]) * K;
        r ^= r >> 29;
    }
    return r;
}

}  // namespace

uint64_t hash_bytes_parallel(ThreadPool& pool, const void* p, size_t n) {
    constexpr int NS = 64;
    uint64_t part[NS] = {};
    const size_t per = (n + NS - 1) / NS;
    pool.run_chunks(NS, 1, [&](int, int64_t b, int64_t e) {
        for (int64_t s = b; s < e; ++s) {
            const size_t lo = std::min(n, (size_t)s * per), hi = std::min(n, lo + per);
            part[s] = hash_range((const uint8_t*)p + lo, hi - lo);
        }
    });
    uint64_t r = 0x84222325CBF29CE4ull ^ n;
    for (int s = 0; s < NS; ++s) {
        r = (r ^ part[s]) * 0x100000001B3ull;
        r ^= r >> 31;
    }
    return r;
}

void fold_compute(ThreadPool& pool, const Mat& lm, int n_vocab, const uint16_t* hout_w, const float* hout_b, bool q8, void* out_M, size_t out_rs,
                  float* out_m0) {
    // B strips: bf16 [2304][256] -> f32 [4][2304][64]
    float* Bs = (float*)alloc_weight_buffer((size_t)4 * D * 64 * sizeof(float), false);
    float* bvec = (float*)alloc_weight_buffer((size_t)D * sizeof(float), false);
    for (int k = 0; k < D; ++k) {
        bvec[k] = hout_b[k];
        for (int s = 0; s < 4; ++s) bf16_to_f32(hout_w + (size_t)k * RANK + s * 64, Bs + ((size_t)s * D + k) * 64, 64);
    }
    std::vector<FoldBuf*> bufs(pool.size());
    for (auto& b : bufs) b = (FoldBuf*)alloc_weight_buffer(sizeof(FoldBuf), false);
    const int64_t nchunks = (n_vocab + RB - 1) / RB;
    pool.run_chunks(nchunks, 4, [&](int tid, int64_t cb, int64_t ce) {
        FoldBuf& F = *bufs[tid];
        for (int64_t ch = cb; ch < ce; ++ch) {
            const int64_t r0 = ch * RB;
            const int nr = (int)std::min<int64_t>(RB, n_vocab - r0);   // multiple of 6 for 200058; generic below
            std::memset(F.C, 0, sizeof(float) * (size_t)RB * RANK);
            std::memset(F.m0, 0, sizeof(F.m0));
            for (int kc0 = 0; kc0 < D; kc0 += KC) {
                const int kn = std::min(KC, D - kc0);
                for (int r = 0; r < nr; ++r) {   // dequantise
                    const block_q8_0* blk = (const block_q8_0*)((const char*)lm.w + (size_t)(r0 + r) * lm.rs) + kc0 / 32;
                    float* a = F.A + (size_t)r * KC;
                    __m512 dot = _mm512_setzero_ps();
                    for (int b = 0; b < kn / 32; ++b) {
                        const __m512 d = _mm512_set1_ps(fp16_to_f32(blk[b].d));
                        const __m512 v0 = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)blk[b].qs))));
                        const __m512 v1 = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(blk[b].qs + 16)))));
                        _mm512_storeu_ps(a + b * 32, v0);
                        _mm512_storeu_ps(a + b * 32 + 16, v1);
                        dot = _mm512_fmadd_ps(v0, _mm512_loadu_ps(bvec + kc0 + b * 32), dot);
                        dot = _mm512_fmadd_ps(v1, _mm512_loadu_ps(bvec + kc0 + b * 32 + 16), dot);
                    }
                    F.m0[r] += _mm512_reduce_add_ps(dot);
                }
                for (int s = 0; s < 4; ++s) {
                    const float* B = Bs + ((size_t)s * D + kc0) * 64;
                    int r = 0;
                    for (; r + 6 <= nr; r += 6) micro6x64(F.A + (size_t)r * KC, KC, B, kn, F.C + (size_t)r * RANK + s * 64, RANK);
                    for (; r < nr; ++r) {   // tail rows (not used for 200058)
                        for (int c = 0; c < 64; ++c) {
                            float acc = F.C[(size_t)r * RANK + s * 64 + c];
                            for (int k = 0; k < kn; ++k) acc += F.A[(size_t)r * KC + k] * B[(size_t)k * 64 + c];
                            F.C[(size_t)r * RANK + s * 64 + c] = acc;
                        }
                    }
                }
            }
            for (int r = 0; r < nr; ++r) {
                char* dst = (char*)out_M + (size_t)(r0 + r) * out_rs;
                if (q8)
                    quantize_row_q8_0_ref(F.C + (size_t)r * RANK, (block_q8_0*)dst, RANK);
                else
                    f32_to_bf16(F.C + (size_t)r * RANK, (uint16_t*)dst, RANK);
                out_m0[r0 + r] = F.m0[r];
            }
        }
    });
    for (auto b : bufs) free_weight_buffer(b);
    free_weight_buffer(Bs);
    free_weight_buffer(bvec);
}

namespace {
struct CacheHdr {
    char magic[8];
    uint64_t key;
    uint32_t q8, n_vocab;
    uint64_t M_bytes, m0_bytes;
    uint64_t reserved[3];
};
}  // namespace

bool fold_cache_load(const std::string& path, uint64_t key, bool q8, int n_vocab, void* M, size_t M_bytes, float* m0) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    CacheHdr h;
    bool ok = std::fread(&h, sizeof(h), 1, f) == 1 && std::memcmp(h.magic, "INKFOLD1", 8) == 0 && h.key == key && h.q8 == (uint32_t)q8 &&
              h.n_vocab == (uint32_t)n_vocab && h.M_bytes == M_bytes && h.m0_bytes == (uint64_t)n_vocab * 4;
    if (ok) ok = std::fread(M, 1, M_bytes, f) == M_bytes && std::fread(m0, 4, (size_t)n_vocab, f) == (size_t)n_vocab;
    std::fclose(f);
    return ok;
}

bool fold_cache_store(const std::string& path, uint64_t key, bool q8, int n_vocab, const void* M, size_t M_bytes, const float* m0) {
    const std::string tmp = path + ".tmp" + std::to_string((long)getpid());
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    CacheHdr h;
    std::memset(&h, 0, sizeof(h));
    std::memcpy(h.magic, "INKFOLD1", 8);
    h.key = key;
    h.q8 = q8;
    h.n_vocab = (uint32_t)n_vocab;
    h.M_bytes = M_bytes;
    h.m0_bytes = (uint64_t)n_vocab * 4;
    bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1 && std::fwrite(M, 1, M_bytes, f) == M_bytes && std::fwrite(m0, 4, (size_t)n_vocab, f) == (size_t)n_vocab;
    ok = (std::fclose(f) == 0) && ok;
    if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) std::remove(tmp.c_str());
    return ok;
}

}  // namespace dr
}  // namespace ink

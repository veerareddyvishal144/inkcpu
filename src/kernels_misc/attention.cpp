// Attention for Inkling 2 Mini: fp16 KV cache (K transposed in 64-key blocks), decode and prefill kernels with the learned
// relative-position bias, per-query sink, tau scaling and sigmoid output gate.  Owner: agent B2.  See docs/KERNELS_MISC.md.
//
// score(h, j) = (tau*q_h . k_j) / sqrt(128) + sum_f rel_bias[d][f] * tau * r_h[f]      (d = pos - j; bias only for d < 1024)
// out_h = (sum_j exp(score_j) v_j) / (exp(t_h) + sum_j exp(score_j)) * sigmoid(gate)   (t_h = sink logit; softmax in fp32)
// Streaming: keys are processed in chunks of 16 blocks (1024 keys); per chunk the scores of all rows are stored (<= 24 KiB),
// the chunk max is taken, p = exp(s - max) is formed, and p.V is accumulated; chunk states (m, l, acc) are merged in order.
#include <algorithm>
#include <cmath>
#include <cstring>
#include "ink/kernels_misc.h"
#include "ink/kernels_misc_simd.h"

namespace ink {
using namespace simd;

namespace {
constexpr float kNegBig = -1.0e30f;   // masked score
constexpr float kNegHalf = -1.0e29f;  // "is masked" threshold
constexpr size_t kGroupStride = (size_t)kKvBlock * kHeadDim;  // fp16 elements per (block, kv head) = 8192

inline float exp_s(float x) { return _mm_cvtss_f32(_mm512_castps512_ps128(exp16(_mm512_set1_ps(x)))); }

// bit i set <=> lane i of the 64-key block is inside [lo_rel, hi_rel]
inline uint64_t lane_mask64(int64_t lo_rel, int64_t hi_rel) {
    if (hi_rel < 0 || lo_rel > 63 || hi_rel < lo_rel) return 0;
    if (lo_rel < 0) lo_rel = 0;
    if (hi_rel > 63) hi_rel = 63;
    const uint64_t upper = (hi_rel == 63) ? ~0ull : ((1ull << (hi_rel + 1)) - 1ull);
    return upper & (~0ull << lo_rel);
}

inline size_t phys_block(const KvLayer& L, int64_t b) { return L.ring ? (size_t)(b % (int64_t)L.n_blocks) : (size_t)b; }

// Core: NQ query tokens x 3 heads of kv group g over position blocks [b0, b1).  Produces the merged (m, l, acc) per row.
//   qs [NQ*3][128] pre-scaled by tau/sqrt(128);  rs [NQ*3][16] pre-scaled by tau;  qpos/qlo = visible key range [qlo, qpos].
template <int NQ>
void attn_range(const KvLayer& L, const float* table, const float* qs, const float* rs, const int64_t* qpos, const int64_t* qlo,
                int g, int64_t b0, int64_t b1, float* m_out, float* l_out, float* acc_out) {
    constexpr int R = NQ * 3;
    constexpr int DG = (NQ == 1) ? 8 : 4;  // 16-wide d vectors per PV pass (R*DG accumulators <= 24)
    for (int r = 0; r < R; ++r) {
        m_out[r] = -__builtin_inff();
        l_out[r] = 0.0f;
    }
    std::memset(acc_out, 0, sizeof(float) * R * kHeadDim);
    if (b1 <= b0) return;
    int64_t min_lo = qlo[0], max_pos = qpos[0];
    for (int i = 1; i < NQ; ++i) {
        min_lo = std::min(min_lo, qlo[i]);
        max_pos = std::max(max_pos, qpos[i]);
    }
    alignas(64) float S[R][1024];
    const __m512 negbig = _mm512_set1_ps(kNegBig);

    for (int64_t cb = b0; cb < b1; cb += 16) {
        const int nb = (int)std::min<int64_t>(16, b1 - cb);

        // ---------------- pass 1: scores of every row for the chunk's keys
        for (int bi = 0; bi < nb; ++bi) {
            const int64_t b = cb + bi;
            const int64_t jb = b * kKvBlock;
            const uint16_t* Kp = L.k + phys_block(L, b) * kKvBlockElems + (size_t)g * kGroupStride;
            __m512 acc[R][4];
#pragma GCC unroll 6
            for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = _mm512_setzero_ps();
            for (int d = 0; d < kHeadDim; ++d) {
                const uint16_t* kd = Kp + (size_t)d * kKvBlock;
                const __m512 k0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(kd)));
                const __m512 k1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(kd + 16)));
                const __m512 k2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(kd + 32)));
                const __m512 k3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(kd + 48)));
#pragma GCC unroll 6
                for (int r = 0; r < R; ++r) {
                    const __m512 qb = _mm512_set1_ps(qs[r * kHeadDim + d]);
                    acc[r][0] = _mm512_fmadd_ps(qb, k0, acc[r][0]);
                    acc[r][1] = _mm512_fmadd_ps(qb, k1, acc[r][1]);
                    acc[r][2] = _mm512_fmadd_ps(qb, k2, acc[r][2]);
                    acc[r][3] = _mm512_fmadd_ps(qb, k3, acc[r][3]);
                }
            }
            // learned relative-position bias: table row f holds bias reversed in distance, e = j - pos + 1023 (+64 pad)
            int64_t e_base[NQ];
            bool need[NQ], any = false;
            for (int i = 0; i < NQ; ++i) {
                e_base[i] = jb - qpos[i] + (kRelExtent - 1);
                need[i] = (jb <= qpos[i]) && (e_base[i] + (kKvBlock - 1) >= 0);
                any = any || need[i];
            }
            if (any) {
                for (int f = 0; f < kRelFeat; ++f) {
#pragma GCC unroll 2
                    for (int i = 0; i < NQ; ++i) {
                        if (!need[i]) continue;
                        const float* tf = table + (size_t)f * kRelTableLen + 64 + e_base[i];
                        const __m512 t0 = _mm512_loadu_ps(tf), t1 = _mm512_loadu_ps(tf + 16);
                        const __m512 t2 = _mm512_loadu_ps(tf + 32), t3 = _mm512_loadu_ps(tf + 48);
#pragma GCC unroll 3
                        for (int h = 0; h < 3; ++h) {
                            const __m512 rf = _mm512_set1_ps(rs[(i * 3 + h) * kRelFeat + f]);
                            acc[i * 3 + h][0] = _mm512_fmadd_ps(rf, t0, acc[i * 3 + h][0]);
                            acc[i * 3 + h][1] = _mm512_fmadd_ps(rf, t1, acc[i * 3 + h][1]);
                            acc[i * 3 + h][2] = _mm512_fmadd_ps(rf, t2, acc[i * 3 + h][2]);
                            acc[i * 3 + h][3] = _mm512_fmadd_ps(rf, t3, acc[i * 3 + h][3]);
                        }
                    }
                }
            }
            // causal / window mask and store
            for (int r = 0; r < R; ++r) {
                const int i = r / 3;
                const uint64_t m64 = lane_mask64(qlo[i] - jb, qpos[i] - jb);
                float* srow = S[r] + bi * kKvBlock;
                for (int c = 0; c < 4; ++c) {
                    const __mmask16 mk = (__mmask16)((m64 >> (16 * c)) & 0xFFFFu);
                    _mm512_store_ps(srow + 16 * c, _mm512_mask_blend_ps(mk, negbig, acc[r][c]));
                }
            }
        }

        // ---------------- chunk max, p = exp(s - max), l; merge scale factors
        const int len = nb * kKvBlock;
        float a_old[R], a_new[R];
        for (int r = 0; r < R; ++r) {
            __m512 mx = negbig;
            for (int i = 0; i < len; i += 16) mx = _mm512_max_ps(mx, _mm512_load_ps(S[r] + i));
            const float mrow = _mm512_reduce_max_ps(mx);
            if (mrow < kNegHalf) {  // no visible key of this row in the chunk
                for (int i = 0; i < len; i += 16) _mm512_store_ps(S[r] + i, _mm512_setzero_ps());
                a_old[r] = 1.0f;
                a_new[r] = 0.0f;
                continue;
            }
            const __m512 vm = _mm512_set1_ps(mrow);
            const __m512 thr = _mm512_set1_ps(kNegHalf);
            __m512 lsum = _mm512_setzero_ps();
            for (int i = 0; i < len; i += 16) {
                const __m512 s = _mm512_load_ps(S[r] + i);
                const __mmask16 valid = _mm512_cmp_ps_mask(s, thr, _CMP_GT_OQ);
                const __m512 p = _mm512_maskz_mov_ps(valid, exp16(_mm512_sub_ps(s, vm)));
                _mm512_store_ps(S[r] + i, p);
                lsum = _mm512_add_ps(lsum, p);
            }
            const float lrow = _mm512_reduce_add_ps(lsum);
            const float mn = std::max(m_out[r], mrow);
            a_old[r] = exp_s(m_out[r] - mn);
            a_new[r] = exp_s(mrow - mn);
            l_out[r] = l_out[r] * a_old[r] + lrow * a_new[r];
            m_out[r] = mn;
        }

        // ---------------- pass 2: acc = acc*a_old + a_new * sum_k p_k v_k
        for (int d0 = 0; d0 < 8; d0 += DG) {
            __m512 pa[R][DG];
#pragma GCC unroll 6
            for (int r = 0; r < R; ++r)
#pragma GCC unroll 8
                for (int j = 0; j < DG; ++j) pa[r][j] = _mm512_setzero_ps();
            for (int bi = 0; bi < nb; ++bi) {
                const int64_t b = cb + bi;
                const int64_t jb = b * kKvBlock;
                const int kmin = (int)std::max<int64_t>(0, min_lo - jb);
                const int kmax = (int)std::min<int64_t>(kKvBlock - 1, max_pos - jb);
                if (kmin > kmax) continue;
                const uint16_t* Vp = L.v + phys_block(L, b) * kKvBlockElems + (size_t)g * kGroupStride + (size_t)d0 * 16;
                for (int kk = kmin; kk <= kmax; ++kk) {
                    const uint16_t* vr = Vp + (size_t)kk * kHeadDim;
                    __m512 pb[R];
#pragma GCC unroll 6
                    for (int r = 0; r < R; ++r) pb[r] = _mm512_set1_ps(S[r][bi * kKvBlock + kk]);
#pragma GCC unroll 8
                    for (int j = 0; j < DG; ++j) {
                        const __m512 vv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vr + 16 * j)));
#pragma GCC unroll 6
                        for (int r = 0; r < R; ++r) pa[r][j] = _mm512_fmadd_ps(pb[r], vv, pa[r][j]);
                    }
                }
            }
            for (int r = 0; r < R; ++r) {
                const __m512 ao = _mm512_set1_ps(a_old[r]), an = _mm512_set1_ps(a_new[r]);
                for (int j = 0; j < DG; ++j) {
                    float* ap = acc_out + r * kHeadDim + (d0 + j) * 16;
                    _mm512_storeu_ps(ap, _mm512_fmadd_ps(_mm512_loadu_ps(ap), ao, _mm512_mul_ps(pa[r][j], an)));
                }
            }
        }
    }
}

// out[0..128) = (sum_s acc_s * e^{m_s - M}) / (e^{t - M} + sum_s l_s e^{m_s - M}) * sigmoid(gate)
template <class MG, class LG, class AG>
inline void finish_row(float t, int ns, MG mget, LG lget, AG aget, const float* gate, float* out) {
    float M = t;
    for (int s = 0; s < ns; ++s) M = std::max(M, mget(s));
    float L = exp_s(t - M);
    alignas(64) float A[kHeadDim];
    for (int i = 0; i < kHeadDim; ++i) A[i] = 0.0f;
    for (int s = 0; s < ns; ++s) {
        const float ms = mget(s);
        if (!(ms > kNegHalf)) continue;  // empty split
        const float w = exp_s(ms - M);
        L += lget(s) * w;
        const float* a = aget(s);
        const __m512 vw = _mm512_set1_ps(w);
        for (int i = 0; i < kHeadDim; i += 16)
            _mm512_store_ps(A + i, _mm512_fmadd_ps(_mm512_loadu_ps(a + i), vw, _mm512_load_ps(A + i)));
    }
    const __m512 vinv = _mm512_set1_ps(1.0f / L);
    for (int i = 0; i < kHeadDim; i += 16) {
        __m512 v = _mm512_mul_ps(_mm512_load_ps(A + i), vinv);
        if (gate) v = _mm512_mul_ps(v, sigmoid16(_mm512_loadu_ps(gate + i)));
        _mm512_storeu_ps(out + i, v);
    }
}

template <int NQ>
void prefill_tile(const KvLayer& L, const float* table, const AttnPrefill& P, int g, int t, float* out) {
    constexpr int R = NQ * 3;
    alignas(64) float qs[R * kHeadDim];
    alignas(64) float rs[R * kRelFeat];
    int64_t qpos[NQ], qlo[NQ];
    const float kq = 1.0f / sqrtf((float)kHeadDim);
    int64_t lo_min = INT64_MAX, pos_max = 0;
    for (int i = 0; i < NQ; ++i) {
        const int tt = t + i;
        const float tau = P.tau ? P.tau[tt] : 1.0f;
        vscale(P.q + (size_t)tt * (kNQHeads * kHeadDim) + (size_t)g * 3 * kHeadDim, tau * kq, qs + (size_t)i * 3 * kHeadDim, 3 * kHeadDim);
        vscale(P.r + (size_t)tt * (kNQHeads * kRelFeat) + (size_t)g * 3 * kRelFeat, tau, rs + (size_t)i * 3 * kRelFeat, 3 * kRelFeat);
        qpos[i] = P.pos0 + tt;
        qlo[i] = L.ring ? std::max<int64_t>(0, qpos[i] - (kSwaWindow - 1)) : 0;
        lo_min = std::min(lo_min, qlo[i]);
        pos_max = std::max(pos_max, qpos[i]);
    }
    alignas(64) float m[R], l[R];
    alignas(64) float acc[R * kHeadDim];
    attn_range<NQ>(L, table, qs, rs, qpos, qlo, g, lo_min / kKvBlock, pos_max / kKvBlock + 1, m, l, acc);
    for (int i = 0; i < NQ; ++i) {
        const int tt = t + i;
        for (int h = 0; h < 3; ++h) {
            const int r = i * 3 + h, head = g * 3 + h;
            finish_row(
                P.sink[(size_t)tt * kNQHeads + head], 1, [&](int) { return m[r]; }, [&](int) { return l[r]; },
                [&](int) { return acc + r * kHeadDim; }, P.gate ? P.gate + (size_t)tt * kNQHeads * kHeadDim + (size_t)head * kHeadDim : nullptr,
                out + (size_t)tt * kNQHeads * kHeadDim + (size_t)head * kHeadDim);
        }
    }
}
}  // namespace

// ------------------------------------------------------------------ cache
void kv_append(const KvLayer& L, int64_t pos, const float* k, const float* v, int g0, int g1) {
    const size_t slot = L.ring ? (size_t)(pos % ((int64_t)L.n_blocks * kKvBlock)) : (size_t)pos;
    const size_t b = slot / kKvBlock, o = slot % kKvBlock;
    for (int g = g0; g < g1; ++g) {
        uint16_t* vp = L.v + b * kKvBlockElems + (size_t)g * kGroupStride + o * kHeadDim;
        uint16_t* kp = L.k + b * kKvBlockElems + (size_t)g * kGroupStride + o;
        alignas(32) uint16_t tmp[kHeadDim];
        for (int j = 0; j < 8; ++j) {
            _mm256_storeu_si256((__m256i*)(tmp + 16 * j), f32x16_to_f16(_mm512_loadu_ps(k + g * kHeadDim + 16 * j)));
            _mm256_storeu_si256((__m256i*)(vp + 16 * j), f32x16_to_f16(_mm512_loadu_ps(v + g * kHeadDim + 16 * j)));
        }
        for (int d = 0; d < kHeadDim; ++d) kp[(size_t)d * kKvBlock] = tmp[d];
    }
}

void kv_append_n(const KvLayer& L, int64_t pos0, int n, const float* k, const float* v) {
    for (int t = 0; t < n; ++t) kv_append(L, pos0 + t, k + (size_t)t * kNKvHeads * kHeadDim, v + (size_t)t * kNKvHeads * kHeadDim);
}

void attn_repack_rel_bias(const float* rel_bias, float* table) {
    for (int f = 0; f < kRelFeat; ++f) {
        float* row = table + (size_t)f * kRelTableLen;
        for (int i = 0; i < 64; ++i) row[i] = 0.0f;
        for (int e = 0; e < kRelExtent; ++e) row[64 + e] = rel_bias[(size_t)(kRelExtent - 1 - e) * kRelFeat + f];
        for (int i = 0; i < 64; ++i) row[64 + kRelExtent + i] = 0.0f;
    }
}

// ------------------------------------------------------------------ decode
void attn_decode_task(const KvLayer& L, const float* rel_table, const AttnQuery& q, int g, int split, int nsplit, AttnPartial* out) {
    alignas(64) float qs[3 * kHeadDim];
    alignas(64) float rs[3 * kRelFeat];
    const float kq = 1.0f / sqrtf((float)kHeadDim);
    vscale(q.q + (size_t)g * 3 * kHeadDim, q.tau * kq, qs, 3 * kHeadDim);
    vscale(q.r + (size_t)g * 3 * kRelFeat, q.tau, rs, 3 * kRelFeat);
    const int64_t pos = q.pos;
    const int64_t lo = L.ring ? std::max<int64_t>(0, pos - (kSwaWindow - 1)) : 0;
    const int64_t fb = lo / kKvBlock, nbk = pos / kKvBlock - fb + 1;
    const int64_t b0 = fb + nbk * split / nsplit, b1 = fb + nbk * (split + 1) / nsplit;
    attn_range<1>(L, rel_table, qs, rs, &pos, &lo, g, b0, b1, out->m, out->l, out->acc);
}

void attn_decode_finish(const AttnQuery& q, int g, const AttnPartial* parts, int nsplit, float* out) {
    for (int h = 0; h < 3; ++h) {
        const int head = g * 3 + h;
        finish_row(
            q.sink[head], nsplit, [&](int s) { return parts[s].m[h]; }, [&](int s) { return parts[s].l[h]; },
            [&](int s) { return parts[s].acc + h * kHeadDim; }, q.gate ? q.gate + (size_t)head * kHeadDim : nullptr,
            out + (size_t)head * kHeadDim);
    }
}

void attn_decode(const KvLayer& L, const float* rel_table, const AttnQuery& q, int nsplit, AttnPartial* scratch, float* out) {
    for (int g = 0; g < kNKvHeads; ++g)
        for (int s = 0; s < nsplit; ++s) attn_decode_task(L, rel_table, q, g, s, nsplit, scratch + (size_t)g * nsplit + s);
    for (int g = 0; g < kNKvHeads; ++g) attn_decode_finish(q, g, scratch + (size_t)g * nsplit, nsplit, out);
}

// ------------------------------------------------------------------ prefill
void attn_prefill_group(const KvLayer& L, const float* rel_table, const AttnPrefill& P, int g, int t0, int t1, float* out) {
    int t = t0;
    for (; t + 2 <= t1; t += 2) prefill_tile<2>(L, rel_table, P, g, t, out);
    if (t < t1) prefill_tile<1>(L, rel_table, P, g, t, out);
}

}  // namespace ink

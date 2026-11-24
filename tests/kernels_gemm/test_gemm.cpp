// Tests for the batched GEMM kernels (agent G). Every output must be BIT-IDENTICAL to the per-token gemv kernels of agent B1
// (optimised gemv_* and, on small cases, the scalar gemv_*_ref). Shapes: K in {64,192,1152,2304,3072,4608}, rows incl. odd counts,
// nt in {1,2,3,5,8,16,17,64,256}, 4 value modes (normal, extreme, zero scales, random fp16/UE4M3 bit patterns), padded row strides,
// padded ldy (guard values must survive), row/token splitting, MoE routing (token-indexed and pair-indexed activations).
// NVFP4 cases also run gemm_nvfp4r / gemm_moe_nvfp4r on a copy whose scale bytes were re-encoded with nvfp4r_repack_rows
// (include/ink/nvfp4_repack.h) and require the same bits as gemv_nvfp4 on the ORIGINAL bytes (all 256 scale byte values occur in mode RANDBITS).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ink/kernels_gemm.h"
#include "ink/nvfp4_repack.h"
#include "testutil.h"

using namespace ink;

namespace {

enum Mode { NORMAL = 0, EXTREME = 1, ZEROS = 2, RANDBITS = 3, NMODES = 4 };

uint16_t rand_fp16(tu::Rng& rng, int mode) {
    switch (mode) {
        case NORMAL: return f32_to_fp16(0.0005f + 0.2f * rng.uni());
        case EXTREME: {
            static const uint16_t v[4] = {0x7BFF /*65504*/, 0x0001 /*denormal*/, 0x0400 /*min normal*/, 0x3C00 /*1.0*/};
            return v[rng.u32() & 3];
        }
        case ZEROS: return (rng.u32() % 3 == 0) ? f32_to_fp16(rng.uni()) : 0;
        default: {
            uint16_t h;
            do h = (uint16_t)(rng.u32() & 0xFFFF);
            while (((h >> 10) & 31) == 31);  // no inf/nan
            return h;
        }
    }
}
int8_t rand_q(tu::Rng& rng, int mode, bool act) {
    if (mode == EXTREME) {
        static const int8_t w[5] = {-128, 127, -127, 0, 1};
        static const int8_t a[5] = {-127, 127, 0, 1, -1};
        return act ? a[rng.u32() % 5] : w[rng.u32() % 5];
    }
    return (int8_t)rng.range(act ? -127 : -128, 127);
}

struct Mat {
    std::vector<uint8_t> buf;
    size_t rs = 0;
    const void* ptr() const { return buf.data() + 4; }  // deliberately misaligned base
};

Mat make_q8(int rows, int k, int pad, int mode, tu::Rng& rng) {
    Mat m;
    m.rs = row_bytes_q8_0(k) + (size_t)pad;
    m.buf.assign(m.rs * (size_t)rows + 16, 0xA5);
    for (int r = 0; r < rows; ++r) {
        block_q8_0* b = (block_q8_0*)((char*)m.buf.data() + 4 + (size_t)r * m.rs);  // misaligned on purpose
        for (int i = 0; i < k / 32; ++i) {
            std::memset(&b[i], 0, sizeof(block_q8_0));
            uint16_t d = rand_fp16(rng, mode);
            std::memcpy(&b[i].d, &d, 2);
            for (int j = 0; j < 32; ++j) b[i].qs[j] = rand_q(rng, mode, false);
        }
    }
    return m;
}
Mat make_nvfp4(int rows, int k, int pad, int mode, tu::Rng& rng) {
    Mat m;
    m.rs = row_bytes_nvfp4(k) + (size_t)pad;
    m.buf.assign(m.rs * (size_t)rows + 16, 0xA5);
    for (int r = 0; r < rows; ++r) {
        uint8_t* p = m.buf.data() + 4 + (size_t)r * m.rs;
        for (int i = 0; i < k / 64; ++i) {
            uint8_t* b = p + (size_t)i * 36;
            for (int s = 0; s < 4; ++s) {
                uint8_t d;
                switch (mode) {
                    case NORMAL: d = (uint8_t)rng.range(0x28, 0x48); break;
                    case EXTREME: {
                        static const uint8_t v[6] = {0x7E, 0x01, 0x00, 0x7F, 0x08, 0x07};
                        d = v[rng.u32() % 6];
                        break;
                    }
                    case ZEROS: d = (rng.u32() % 3 == 0) ? (uint8_t)rng.range(0x20, 0x50) : (rng.u32() & 1 ? 0x00 : 0x7F); break;
                    default: d = (uint8_t)(rng.u32() & 0xFF);
                }
                b[s] = d;
            }
            for (int j = 0; j < 32; ++j) b[4 + j] = (uint8_t)(rng.u32() & 0xFF);
        }
    }
    return m;
}
// copy of an NVFP4 matrix with the scale bytes re-encoded in place (same misaligned base, same padded stride)
Mat reencode(const Mat& m, int rows, int k) {
    Mat r = m;
    nvfp4r_repack_rows(r.buf.data() + 4, r.rs, rows, k / 64);
    return r;
}
long g_r_cases = 0;

std::vector<block_q8_0> make_acts(int nt, int k, int mode, tu::Rng& rng) {
    std::vector<block_q8_0> a((size_t)nt * (k / 32));
    for (auto& b : a) {
        std::memset(&b, 0, sizeof b);
        b.d = rand_fp16(rng, mode);
        for (int j = 0; j < 32; ++j) b.qs[j] = rand_q(rng, mode, true);
    }
    return a;
}

struct AlignedBuf {
    void* p = nullptr;
    explicit AlignedBuf(size_t n) { p = std::aligned_alloc(64, (n + 63) & ~(size_t)63); }
    ~AlignedBuf() { std::free(p); }
};

constexpr uint32_t kSentinel = 0x7FC0DEADu;
bool same_bits(const float* a, const float* b, size_t n, size_t* first = nullptr) {
    for (size_t i = 0; i < n; ++i)
        if (std::memcmp(a + i, b + i, 4) != 0) {
            if (first) *first = i;
            return false;
        }
    return true;
}

long g_cases = 0;
long g_ref_cases = 0;

typedef void (*GemvFn)(const void*, size_t, const block_q8_0*, int, float*, int);

// dense Q8_0 / NVFP4
void check_dense(int fmt, int k, int rows, int nt, int mode, bool pad_stride, tu::Rng& rng) {
    const int pad = pad_stride ? 6 : 0;
    Mat W = fmt == GEMM_FMT_Q8 ? make_q8(rows, k, pad, mode, rng) : make_nvfp4(rows, k, pad, mode, rng);
    std::vector<block_q8_0> acts = make_acts(nt, k, mode, rng);
    const int nb = k / 32;
    AlignedBuf buf(gemm_act_bytes(fmt, nt, k));
    // pack in two token ranges (as two threads would)
    const int tsplit = nt / 2;
    gemm_pack_act(fmt, acts.data(), 0, tsplit, nt, k, buf.p);
    gemm_pack_act(fmt, acts.data(), tsplit, nt, nt, k, buf.p);
    const GemmAct act = gemm_act_view(fmt, buf.p, nt, k);

    const size_t ldy = (size_t)rows + 3;
    std::vector<float> y((size_t)nt * ldy), yref((size_t)nt * ldy), yscal((size_t)nt * ldy);
    uint32_t sv = kSentinel;
    float sf;
    std::memcpy(&sf, &sv, 4);
    std::fill(y.begin(), y.end(), sf);
    std::fill(yref.begin(), yref.end(), sf);
    std::fill(yscal.begin(), yscal.end(), sf);
    GemvFn gv = fmt == GEMM_FMT_Q8 ? gemv_q8_0 : gemv_nvfp4;
    GemvFn gr = fmt == GEMM_FMT_Q8 ? gemv_q8_0_ref : gemv_nvfp4_ref;
    for (int t = 0; t < nt; ++t) gv(W.ptr(), W.rs, acts.data() + (size_t)t * nb, k, yref.data() + (size_t)t * ldy, rows);
    // gemm: row split at a multiple of 4 and token split at nt/3, four calls
    const int r1 = std::min(rows, (rows / 2) & ~3);
    const int t1 = nt / 3;
    auto call = [&](int ra, int rb, int ta, int tb) {
        if (rb <= ra || tb <= ta) return;
        const char* w = (const char*)W.ptr() + (size_t)ra * W.rs;
        if (fmt == GEMM_FMT_Q8)
            gemm_q8_0(w, W.rs, act, ta, tb, k, y.data() + ra, ldy, rb - ra);
        else
            gemm_nvfp4(w, W.rs, act, ta, tb, k, y.data() + ra, ldy, rb - ra);
    };
    call(0, r1, 0, t1);
    call(r1, rows, 0, t1);
    call(0, r1, t1, nt);
    call(r1, rows, t1, nt);
    ++g_cases;
    for (int t = 0; t < nt; ++t) {
        size_t first = 0;
        if (!same_bits(y.data() + (size_t)t * ldy, yref.data() + (size_t)t * ldy, rows, &first)) {
            std::fprintf(stderr, "MISMATCH fmt=%d K=%d rows=%d nt=%d mode=%d pad=%d token=%d row=%zu gemm=%.9g gemv=%.9g\n", fmt, k, rows, nt, mode,
                         (int)pad_stride, t, first, (double)y[(size_t)t * ldy + first], (double)yref[(size_t)t * ldy + first]);
            ++tu::g_fail;
            return;
        }
        for (size_t i = rows; i < ldy; ++i)
            if (std::memcmp(&y[(size_t)t * ldy + i], &sf, 4) != 0) {
                std::fprintf(stderr, "GUARD overwritten fmt=%d K=%d rows=%d nt=%d token=%d\n", fmt, k, rows, nt, t);
                ++tu::g_fail;
                return;
            }
    }
    if (fmt == GEMM_FMT_NVFP4) {  // re-encoded scales: gemm_nvfp4r on the repacked copy must reproduce gemv_nvfp4 on the original bytes
        const Mat Wr = reencode(W, rows, k);
        std::vector<float> yr((size_t)nt * ldy, sf);
        auto callr = [&](int ra, int rb, int ta, int tb) {
            if (rb <= ra || tb <= ta) return;
            gemm_nvfp4r((const char*)Wr.ptr() + (size_t)ra * Wr.rs, Wr.rs, act, ta, tb, k, yr.data() + ra, ldy, rb - ra);
        };
        callr(0, r1, 0, t1);
        callr(r1, rows, 0, t1);
        callr(0, r1, t1, nt);
        callr(r1, rows, t1, nt);
        ++g_r_cases;
        for (int t = 0; t < nt; ++t) {
            size_t first = 0;
            if (!same_bits(yr.data() + (size_t)t * ldy, yref.data() + (size_t)t * ldy, ldy, &first)) {
                std::fprintf(stderr, "MISMATCH nvfp4r K=%d rows=%d nt=%d mode=%d pad=%d token=%d idx=%zu gemm_r=%.9g gemv=%.9g\n", k, rows, nt, mode,
                             (int)pad_stride, t, first, (double)yr[(size_t)t * ldy + first], (double)yref[(size_t)t * ldy + first]);
                ++tu::g_fail;
                return;
            }
        }
    }
    if ((double)rows * nt * k <= 4e6) {  // scalar reference on small cases
        ++g_ref_cases;
        for (int t = 0; t < nt; ++t) gr(W.ptr(), W.rs, acts.data() + (size_t)t * nb, k, yscal.data() + (size_t)t * ldy, rows);
        for (int t = 0; t < nt; ++t) {
            size_t first = 0;
            if (!same_bits(y.data() + (size_t)t * ldy, yscal.data() + (size_t)t * ldy, rows, &first)) {
                std::fprintf(stderr, "MISMATCH vs scalar ref fmt=%d K=%d rows=%d nt=%d mode=%d token=%d row=%zu\n", fmt, k, rows, nt, mode, t, first);
                ++tu::g_fail;
                return;
            }
        }
    }
}

// BF16 / F32
void check_small(bool bf16, int k, int rows, int nt, tu::Rng& rng) {
    const size_t esz = bf16 ? 2 : 4;
    const size_t rs = (size_t)k * esz + (rng.u32() & 1 ? 8 : 0);
    std::vector<uint8_t> wb(rs * rows + 16, 0);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < k; ++i) {
            const float v = rng.gauss() * (rng.u32() % 50 == 0 ? 1e3f : 1.f);
            uint8_t* p = wb.data() + 4 + (size_t)r * rs + (size_t)i * esz;
            if (bf16) {
                const uint16_t b = f32_to_bf16(v);
                std::memcpy(p, &b, 2);
            } else {
                std::memcpy(p, &v, 4);
            }
        }
    const size_t ldx = (size_t)k + 5;
    std::vector<float> x((size_t)nt * ldx);
    for (auto& v : x) v = rng.gauss() * (rng.u32() % 40 == 0 ? 100.f : 1.f);
    const size_t ldy = (size_t)rows + 2;
    std::vector<float> y((size_t)nt * ldy, 0.f), yref((size_t)nt * ldy, 0.f), yscal((size_t)nt * ldy, 0.f);
    const void* w = wb.data() + 4;
    for (int t = 0; t < nt; ++t) {
        if (bf16) {
            gemv_bf16(w, rs, x.data() + (size_t)t * ldx, k, yref.data() + (size_t)t * ldy, rows);
            gemv_bf16_ref(w, rs, x.data() + (size_t)t * ldx, k, yscal.data() + (size_t)t * ldy, rows);
        } else {
            gemv_f32(w, rs, x.data() + (size_t)t * ldx, k, yref.data() + (size_t)t * ldy, rows);
            gemv_f32_ref(w, rs, x.data() + (size_t)t * ldx, k, yscal.data() + (size_t)t * ldy, rows);
        }
    }
    if (bf16)
        gemm_bf16(w, rs, x.data(), ldx, nt, k, y.data(), ldy, rows);
    else
        gemm_f32(w, rs, x.data(), ldx, nt, k, y.data(), ldy, rows);
    ++g_cases;
    for (int t = 0; t < nt; ++t) {
        size_t first = 0;
        if (!same_bits(y.data() + (size_t)t * ldy, yref.data() + (size_t)t * ldy, rows, &first) ||
            !same_bits(y.data() + (size_t)t * ldy, yscal.data() + (size_t)t * ldy, rows, &first)) {
            std::fprintf(stderr, "MISMATCH small bf16=%d K=%d rows=%d nt=%d token=%d row=%zu\n", (int)bf16, k, rows, nt, t, first);
            ++tu::g_fail;
            return;
        }
    }
}

// MoE grouped: E experts, token-indexed activations (gate_up style) and pair-indexed activations (down style)
void check_moe(int fmt, int k, int rows, int E, int ntok, int topk, int chunk_rows, tu::Rng& rng) {
    std::vector<Mat> mats;
    // experts stored back to back in one buffer with an expert stride
    Mat all;
    const size_t rs = fmt == GEMM_FMT_Q8 ? row_bytes_q8_0(k) : row_bytes_nvfp4(k);
    const size_t es = rs * (size_t)rows + 24;
    std::vector<uint8_t> wbuf(es * E + 16, 0x5A);
    for (int e = 0; e < E; ++e) {
        Mat m = fmt == GEMM_FMT_Q8 ? make_q8(rows, k, 0, e % NMODES, rng) : make_nvfp4(rows, k, 0, e % NMODES, rng);
        std::memcpy(wbuf.data() + 4 + (size_t)e * es, m.buf.data() + 4, rs * (size_t)rows);
    }
    std::vector<int32_t> ids((size_t)ntok * topk);
    for (auto& v : ids) v = (int32_t)(rng.u32() % (uint32_t)E);
    if (E > 3)
        for (int t = 0; t < ntok; ++t)
            if (rng.u32() % 4 == 0) ids[(size_t)t * topk] = ids[(size_t)t * topk + topk - 1];  // repeated experts in a token are legal
    MoeRoute r;
    moe_build_route(ids.data(), ntok, topk, E, r);
    const int np = r.n_pairs();
    std::vector<MoeItem> items;
    moe_work_items(r, rows, chunk_rows, items);
    // activation per token (gate_up) and per pair (down)
    std::vector<block_q8_0> at = make_acts(ntok, k, NORMAL, rng), ap = make_acts(np, k, NORMAL, rng);
    AlignedBuf bt(gemm_act_bytes(fmt, ntok, k)), bp(gemm_act_bytes(fmt, np, k));
    gemm_pack_act(fmt, at.data(), 0, ntok, ntok, k, bt.p);
    gemm_pack_act(fmt, ap.data(), 0, np, np, k, bp.p);
    const GemmAct acts_t = gemm_act_view(fmt, bt.p, ntok, k), acts_p = gemm_act_view(fmt, bp.p, np, k);
    const size_t ldy = (size_t)rows + 1;
    uint32_t sv = kSentinel;
    float sf;
    std::memcpy(&sf, &sv, 4);
    std::vector<float> y1((size_t)np * ldy, sf), y2((size_t)np * ldy, sf), ref1((size_t)np * ldy, sf), ref2((size_t)np * ldy, sf);
    GemvFn gv = fmt == GEMM_FMT_Q8 ? gemv_q8_0 : gemv_nvfp4;
    const int nb = k / 32;
    for (int e = 0; e < E; ++e)
        for (int p = r.offset[e]; p < r.offset[e + 1]; ++p) {
            const void* w = wbuf.data() + 4 + (size_t)e * es;
            gv(w, rs, at.data() + (size_t)r.pair_token[p] * nb, k, ref1.data() + (size_t)p * ldy, rows);
            gv(w, rs, ap.data() + (size_t)p * nb, k, ref2.data() + (size_t)p * ldy, rows);
        }
    // run the items in a shuffled order
    std::vector<int> order(items.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[rng.u32() % i]);
    for (int oi : order) {
        const MoeItem& it = items[(size_t)oi];
        if (fmt == GEMM_FMT_Q8) {
            gemm_moe_q8_0(wbuf.data() + 4, es, rs, acts_t, r.pair_token.data(), r, it, k, y1.data(), ldy);
            gemm_moe_q8_0(wbuf.data() + 4, es, rs, acts_p, nullptr, r, it, k, y2.data(), ldy);
        } else {
            gemm_moe_nvfp4(wbuf.data() + 4, es, rs, acts_t, r.pair_token.data(), r, it, k, y1.data(), ldy);
            gemm_moe_nvfp4(wbuf.data() + 4, es, rs, acts_p, nullptr, r, it, k, y2.data(), ldy);
        }
    }
    ++g_cases;
    if (fmt == GEMM_FMT_NVFP4) {  // re-encoded expert weights -> gemm_moe_nvfp4r, same reference
        std::vector<uint8_t> wr = wbuf;
        for (int e = 0; e < E; ++e) nvfp4r_repack_rows(wr.data() + 4 + (size_t)e * es, rs, rows, k / 64);
        std::vector<float> z1((size_t)np * ldy, sf), z2((size_t)np * ldy, sf);
        for (int oi : order) {
            const MoeItem& it = items[(size_t)oi];
            gemm_moe_nvfp4r(wr.data() + 4, es, rs, acts_t, r.pair_token.data(), r, it, k, z1.data(), ldy);
            gemm_moe_nvfp4r(wr.data() + 4, es, rs, acts_p, nullptr, r, it, k, z2.data(), ldy);
        }
        ++g_r_cases;
        for (int p = 0; p < np; ++p) {
            size_t f = 0;
            if (!same_bits(z1.data() + (size_t)p * ldy, ref1.data() + (size_t)p * ldy, ldy, &f) ||
                !same_bits(z2.data() + (size_t)p * ldy, ref2.data() + (size_t)p * ldy, ldy, &f)) {
                std::fprintf(stderr, "MISMATCH moe nvfp4r K=%d rows=%d E=%d ntok=%d topk=%d chunk=%d pair=%d idx=%zu\n", k, rows, E, ntok, topk, chunk_rows, p, f);
                ++tu::g_fail;
                return;
            }
        }
    }
    // slots are a permutation of 0..np-1
    std::vector<char> seen((size_t)np, 0);
    for (int p = 0; p < np; ++p) {
        const int s = r.pair_slot[(size_t)p];
        if (s < 0 || s >= np || seen[(size_t)s] || ids[(size_t)s] != (int32_t)(std::upper_bound(r.offset.begin(), r.offset.end(), p) - r.offset.begin() - 1))
            TU_CHECK(false);
        else
            seen[(size_t)s] = 1;
    }
    for (int p = 0; p < np; ++p) {
        size_t f = 0;
        if (!same_bits(y1.data() + (size_t)p * ldy, ref1.data() + (size_t)p * ldy, ldy, &f) ||
            !same_bits(y2.data() + (size_t)p * ldy, ref2.data() + (size_t)p * ldy, ldy, &f)) {
            std::fprintf(stderr, "MISMATCH moe fmt=%d K=%d rows=%d E=%d ntok=%d topk=%d chunk=%d pair=%d idx=%zu\n", fmt, k, rows, E, ntok, topk,
                         chunk_rows, p, f);
            ++tu::g_fail;
            return;
        }
    }
}

// ---- X4 prototype numerics (not a bit-identity check): deviation vs a double-precision reference and Q8_0 re-quantisation flips.
// Realistic setting: one expert gate/up matrix (2304 x 2304, NORMAL scales, random nibbles), Gaussian activations quantised with
// quantize_row_q8_0; the SiTU proxy silu(gate) * up is re-quantised to Q8_0 (as the engine does before the down GEMM) for both kernels.
double ref_dot(const uint8_t* wrow, int k, const block_q8_0* x) {  // exact in double: sum kv * ue4m3(d) * q * dx
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    double acc = 0;
    for (int sb = 0; sb < k / 64; ++sb) {
        const uint8_t* b = wrow + (size_t)sb * 36;
        for (int e = 0; e < 64; ++e) {
            const int s = e >> 4, i = e & 15;
            const int byte = b[4 + 8 * s + (i & 7)];
            const int nib = i < 8 ? (byte & 0xF) : (byte >> 4);
            const block_q8_0* xb = x + 2 * sb + (e >> 5);
            acc += (double)kv[nib] * (double)ue4m3_to_f32(b[s]) * (double)xb->qs[e & 31] * (double)fp16_to_f32(xb->d);
        }
    }
    return acc;
}
double ref_abs(const uint8_t* wrow, int k, const block_q8_0* x) {  // sum |w * x|
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    double acc = 0;
    for (int sb = 0; sb < k / 64; ++sb) {
        const uint8_t* b = wrow + (size_t)sb * 36;
        for (int e = 0; e < 64; ++e) {
            const int s = e >> 4, i = e & 15;
            const int byte = b[4 + 8 * s + (i & 7)];
            const int nib = i < 8 ? (byte & 0xF) : (byte >> 4);
            const block_q8_0* xb = x + 2 * sb + (e >> 5);
            acc += std::fabs((double)kv[nib] * (double)ue4m3_to_f32(b[s]) * (double)xb->qs[e & 31] * (double)fp16_to_f32(xb->d));
        }
    }
    return acc;
}
void check_x4(tu::Rng& rng) {
    const int k = 2304, rows = 2304, nt = 24;
    Mat G = make_nvfp4(rows, k, 0, NORMAL, rng), U = make_nvfp4(rows, k, 0, NORMAL, rng);
    Mat Gr = reencode(G, rows, k), Ur = reencode(U, rows, k);
    std::vector<float> xf((size_t)nt * k);
    for (auto& v : xf) v = rng.gauss();
    std::vector<block_q8_0> acts((size_t)nt * (k / 32));
    for (int t = 0; t < nt; ++t) quantize_row_q8_0(xf.data() + (size_t)t * k, acts.data() + (size_t)t * (k / 32), k);
    AlignedBuf b1(gemm_act_bytes(GEMM_FMT_NVFP4, nt, k)), b2(gemm_act_bytes(GEMM_FMT_NVFP4X4, nt, k));
    gemm_pack_act(GEMM_FMT_NVFP4, acts.data(), 0, nt, nt, k, b1.p);
    gemm_pack_act(GEMM_FMT_NVFP4X4, acts.data(), 0, nt, nt, k, b2.p);
    const GemmAct a1 = gemm_act_view(GEMM_FMT_NVFP4, b1.p, nt, k), a2 = gemm_act_view(GEMM_FMT_NVFP4X4, b2.p, nt, k);
    std::vector<float> ge((size_t)nt * rows), gx((size_t)nt * rows), ue((size_t)nt * rows), ux((size_t)nt * rows);
    gemm_nvfp4r(Gr.ptr(), Gr.rs, a1, 0, nt, k, ge.data(), rows, rows);
    gemm_nvfp4r_x4(Gr.ptr(), Gr.rs, a2, 0, nt, k, gx.data(), rows, rows);
    gemm_nvfp4r(Ur.ptr(), Ur.rs, a1, 0, nt, k, ue.data(), rows, rows);
    gemm_nvfp4r_x4(Ur.ptr(), Ur.rs, a2, 0, nt, k, ux.data(), rows, rows);
    // deviation vs double reference on a subset of (row, token)
    double max_e = 0, max_x = 0, max_xe = 0, maxabs = 0, sum_e = 0, sum_x = 0;
    long nref = 0;
    for (int t = 0; t < nt; t += 3)
        for (int r = 0; r < rows; r += 7) {
            const uint8_t* wr = (const uint8_t*)G.ptr() + (size_t)r * G.rs;
            const block_q8_0* x = acts.data() + (size_t)t * (k / 32);
            const double ref = ref_dot(wr, k, x), ab = ref_abs(wr, k, x);
            const double de = std::fabs((double)ge[(size_t)t * rows + r] - ref) / ab, dx = std::fabs((double)gx[(size_t)t * rows + r] - ref) / ab;
            max_e = std::max(max_e, de);
            max_x = std::max(max_x, dx);
            sum_e += de;
            sum_x += dx;
            max_xe = std::max(max_xe, std::fabs((double)gx[(size_t)t * rows + r] - (double)ge[(size_t)t * rows + r]));
            maxabs = std::max(maxabs, std::fabs(ref));
            ++nref;
        }
    std::printf("x4 numerics (2304x2304, 24 tokens, %ld (row,token) samples vs double): max |y-ref|/sum|wx|: exact %.3g, x4 %.3g; mean: exact %.3g, x4 %.3g; "
                "max |x4-exact| %.3g (max |y| %.3g)\n", nref, max_e, max_x, sum_e / nref, sum_x / nref, max_xe, maxabs);
    TU_CHECK(max_x < 1e-5);  // sanity: the prototype must stay at float-rounding level
    // Q8_0 re-quantisation flips: raw GEMM output rows and the SiTU proxy silu(g) * u
    long flips_raw = 0, flips_act = 0, dflips_raw = 0, dflips_act = 0, n = 0;
    std::vector<float> he((size_t)rows), hx((size_t)rows);
    std::vector<block_q8_0> qa(rows / 32), qb(rows / 32);
    auto flips = [&](const float* a, const float* b, long& fq, long& fd) {
        quantize_row_q8_0(a, qa.data(), rows);
        quantize_row_q8_0(b, qb.data(), rows);
        for (int i = 0; i < rows / 32; ++i) {
            if (qa[i].d != qb[i].d) ++fd;
            for (int j = 0; j < 32; ++j) fq += qa[i].qs[j] != qb[i].qs[j];
        }
    };
    for (int t = 0; t < nt; ++t) {
        flips(ge.data() + (size_t)t * rows, gx.data() + (size_t)t * rows, flips_raw, dflips_raw);
        for (int r = 0; r < rows; ++r) {
            const float g1 = ge[(size_t)t * rows + r], g2 = gx[(size_t)t * rows + r];
            he[(size_t)r] = g1 / (1.f + std::exp(-g1)) * ue[(size_t)t * rows + r];
            hx[(size_t)r] = g2 / (1.f + std::exp(-g2)) * ux[(size_t)t * rows + r];
        }
        flips(he.data(), hx.data(), flips_act, dflips_act);
        n += rows;
    }
    std::printf("x4 Q8_0 re-quantisation flips vs exact kernel: raw GEMM output %ld of %ld int8 values (%.4f%%), %ld of %ld block scales; "
                "silu(gate)*up %ld of %ld (%.4f%%), %ld block scales\n", flips_raw, n, 100.0 * flips_raw / n, dflips_raw, n / 32, flips_act, n,
                100.0 * flips_act / n, dflips_act);
}


// ---- I16 kernels (P2, wave 2): NOT bit-identical to gemv_nvfp4 by design. Reference = a scalar model of the kernel's arithmetic: exact integer
// dot products per 16-k sub-block (reconstructed with plain ints from the ORIGINAL GGUF bytes) and the documented float order
//   acc = fmaf( fmaf((float)I1, sw1, (float)I0 * sw0), dx, acc )  per Q8_0 block, sequential over blocks.
// The kernel must reproduce the model BIT FOR BIT (any int16 saturation, wrong bias or lane mix-up would change the floats), for every token
// tile width / K segment length / row chunking, in dense and MoE form, incl. the adversarial saturation cases (x = +-127 / -128, w = +-12).
float i16_model_dot(const uint8_t* wrow, int k, const block_q8_0* x) {
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    float acc = 0.f;
    for (int b = 0; b < k / 32; ++b) {
        const uint8_t* blk = wrow + (size_t)(b / 2) * 36;
        const int h = b & 1;
        int I[2] = {0, 0};
        for (int s = 0; s < 2; ++s) {
            const int sb = 2 * h + s;  // sub-block within the super-block
            for (int i = 0; i < 16; ++i) {
                const int byte = blk[4 + 8 * sb + (i & 7)];
                const int nib = i < 8 ? (byte & 0xF) : (byte >> 4);
                I[s] += kv[nib] * (int)x[b].qs[16 * s + i];
            }
        }
        const float sw0 = ue4m3_to_f32(blk[2 * h]), sw1 = ue4m3_to_f32(blk[2 * h + 1]);
        const float dx = fp16_to_f32(x[b].d);
        float f = (float)I[0] * sw0;
        f = fmaf((float)I[1], sw1, f);
        acc = fmaf(f, dx, acc);
    }
    return acc;
}
long g_i16_cases = 0;

// dense: y[t][r] vs the model, guards beyond `rows` in a padded ldy, several tile / segment settings (same bits required)
bool check_i16_dense(const Mat& W /*original bytes*/, const Mat& Wr /*re-encoded*/, int k, int rows, const std::vector<block_q8_0>& acts, int nt, const char* label) {
    AlignedBuf buf(gemm_act_bytes(GEMM_FMT_I16, nt, k));
    gemm_pack_act(GEMM_FMT_I16, acts.data(), 0, nt, nt, k, buf.p);
    const GemmAct act = gemm_act_view(GEMM_FMT_I16, buf.p, nt, k);
    const size_t ldy = (size_t)rows + 3;
    uint32_t sv = kSentinel;
    float sf;
    std::memcpy(&sf, &sv, 4);
    std::vector<float> ref((size_t)nt * ldy, sf);
    const int nb = k / 32;
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < rows; ++r) ref[(size_t)t * ldy + r] = i16_model_dot((const uint8_t*)W.ptr() + (size_t)r * W.rs, k, acts.data() + (size_t)t * nb);
    const int tiles[] = {8, 1, 3, 12, 5};
    const int segs[] = {16, 1, 5, 128, 2};
    bool ok = true;
    for (int v = 0; v < 5 && ok; ++v) {
        g_gemm_i16_tile = tiles[v];
        g_gemm_i16_seg = segs[v];
        std::vector<float> y((size_t)nt * ldy, sf);
        // split rows at a multiple of 16 and tokens at nt/3 for v == 1 (results must not depend on the split)
        if (v == 1 && rows > 16 && nt >= 3) {
            const int ra = 16, ta = nt / 3;
            gemm_nvfp4r_i16(Wr.ptr(), Wr.rs, act, 0, ta, k, y.data(), ldy, ra);
            gemm_nvfp4r_i16((const char*)Wr.ptr() + (size_t)ra * Wr.rs, Wr.rs, act, 0, ta, k, y.data() + ra, ldy, rows - ra);
            gemm_nvfp4r_i16(Wr.ptr(), Wr.rs, act, ta, nt, k, y.data(), ldy, rows);
        } else {
            gemm_nvfp4r_i16(Wr.ptr(), Wr.rs, act, 0, nt, k, y.data(), ldy, rows);
        }
        size_t f = 0;
        if (!same_bits(y.data(), ref.data(), y.size(), &f)) {
            std::fprintf(stderr, "MISMATCH i16 %s K=%d rows=%d nt=%d tile=%d seg=%d: token %zu idx %zu got %.9g model %.9g\n", label, k, rows, nt, tiles[v], segs[v],
                         f / ldy, f % ldy, (double)y[f], (double)ref[f]);
            ++tu::g_fail;
            ok = false;
        }
    }
    g_gemm_i16_tile = 8;
    g_gemm_i16_seg = 16;
    ++g_i16_cases;
    return ok;
}

void check_i16_moe(int k, int rows, int E, int ntok, int topk, int chunk_rows, tu::Rng& rng) {
    const size_t rs = row_bytes_nvfp4(k);
    const size_t es = rs * (size_t)rows + 24;
    std::vector<uint8_t> wbuf(es * E + 16, 0x5A);
    for (int e = 0; e < E; ++e) {
        Mat m = make_nvfp4(rows, k, 0, e % NMODES, rng);
        std::memcpy(wbuf.data() + 4 + (size_t)e * es, m.buf.data() + 4, rs * (size_t)rows);
    }
    std::vector<uint8_t> wr = wbuf;
    for (int e = 0; e < E; ++e) nvfp4r_repack_rows(wr.data() + 4 + (size_t)e * es, rs, rows, k / 64);
    std::vector<int32_t> ids((size_t)ntok * topk);
    for (auto& v : ids) v = (int32_t)(rng.u32() % (uint32_t)E);
    MoeRoute r;
    moe_build_route(ids.data(), ntok, topk, E, r);
    const int np = r.n_pairs();
    std::vector<MoeItem> items;
    moe_work_items(r, rows, chunk_rows, items);
    std::vector<block_q8_0> at = make_acts(ntok, k, NORMAL, rng), ap = make_acts(np, k, NORMAL, rng);
    AlignedBuf bt(gemm_act_bytes(GEMM_FMT_I16, ntok, k)), bp(gemm_act_bytes(GEMM_FMT_I16, np, k));
    gemm_pack_act(GEMM_FMT_I16, at.data(), 0, ntok, ntok, k, bt.p);
    gemm_pack_act(GEMM_FMT_I16, ap.data(), 0, np, np, k, bp.p);
    const GemmAct acts_t = gemm_act_view(GEMM_FMT_I16, bt.p, ntok, k), acts_p = gemm_act_view(GEMM_FMT_I16, bp.p, np, k);
    const size_t ldy = (size_t)rows + 1;
    uint32_t sv = kSentinel;
    float sf;
    std::memcpy(&sf, &sv, 4);
    std::vector<float> y1((size_t)np * ldy, sf), y2((size_t)np * ldy, sf), ref1((size_t)np * ldy, sf), ref2((size_t)np * ldy, sf), z((size_t)np * ldy, sf);
    const int nb = k / 32;
    for (int e = 0; e < E; ++e)
        for (int p = r.offset[e]; p < r.offset[e + 1]; ++p)
            for (int row = 0; row < rows; ++row) {
                const uint8_t* w = wbuf.data() + 4 + (size_t)e * es + (size_t)row * rs;
                ref1[(size_t)p * ldy + row] = i16_model_dot(w, k, at.data() + (size_t)r.pair_token[p] * nb);
                ref2[(size_t)p * ldy + row] = i16_model_dot(w, k, ap.data() + (size_t)p * nb);
            }
    std::vector<int> order(items.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[rng.u32() % i]);
    for (int oi : order) {
        const MoeItem& it = items[(size_t)oi];
        gemm_moe_nvfp4r_i16(wr.data() + 4, es, rs, acts_t, r.pair_token.data(), r, it, k, y1.data(), ldy);
        gemm_moe_nvfp4r_i16(wr.data() + 4, es, rs, acts_p, nullptr, r, it, k, y2.data(), ldy);
    }
    // the exact kernel must accept the I16 pack (superset of the NVFP4 pack) and still equal gemv_nvfp4
    for (int oi : order) gemm_moe_nvfp4r(wr.data() + 4, es, rs, acts_t, r.pair_token.data(), r, items[(size_t)oi], k, z.data(), ldy);
    std::vector<float> gv((size_t)rows);
    ++g_i16_cases;
    for (int p = 0; p < np; ++p) {
        size_t f = 0;
        if (!same_bits(y1.data() + (size_t)p * ldy, ref1.data() + (size_t)p * ldy, ldy, &f) || !same_bits(y2.data() + (size_t)p * ldy, ref2.data() + (size_t)p * ldy, ldy, &f)) {
            std::fprintf(stderr, "MISMATCH i16 moe K=%d rows=%d E=%d ntok=%d topk=%d chunk=%d pair=%d idx=%zu: y1 %.9g ref1 %.9g | y2 %.9g ref2 %.9g\n", k, rows, E, ntok, topk,
                         chunk_rows, p, f, (double)y1[(size_t)p * ldy + f], (double)ref1[(size_t)p * ldy + f], (double)y2[(size_t)p * ldy + f], (double)ref2[(size_t)p * ldy + f]);
            ++tu::g_fail;
            return;
        }
        const int e = (int)(std::upper_bound(r.offset.begin(), r.offset.end(), p) - r.offset.begin() - 1);
        gemv_nvfp4(wbuf.data() + 4 + (size_t)e * es, rs, at.data() + (size_t)r.pair_token[p] * nb, k, gv.data(), rows);
        if (!same_bits(z.data() + (size_t)p * ldy, gv.data(), (size_t)rows, &f)) {
            std::fprintf(stderr, "MISMATCH exact kernel on I16 pack K=%d rows=%d pair=%d idx=%zu\n", k, rows, p, f);
            ++tu::g_fail;
            return;
        }
    }
}

// adversarial saturation: every weight +-12 (nibbles 7 / 15), every activation +-127 or -128 (the largest magnitudes the formats allow):
// int16 pair sums reach 2*24*128 = 6144 and 4-pair sums 24576 (< 32767); the model must be reproduced exactly, and the double reference within rounding
void check_i16_adversarial(tu::Rng& rng) {
    const int k = 2304, rows = 48, nt = 6;
    struct Pat { uint8_t wbyte; int8_t xval; const char* name; };
    const Pat pats[] = {{0x77, 127, "w=+12 x=+127"}, {0x77, -127, "w=+12 x=-127"}, {0x77, -128, "w=+12 x=-128"}, {0xFF, 127, "w=-12 x=+127"},
                        {0xFF, -128, "w=-12 x=-128"}, {0x7F, -128, "w=+-12 alternating, x=-128"}, {0xF7, 127, "w=-+12 alternating, x=+127"}};
    for (const Pat& P : pats) {
        Mat W = make_nvfp4(rows, k, 0, NORMAL, rng);
        for (int r = 0; r < rows; ++r) {
            uint8_t* row = W.buf.data() + 4 + (size_t)r * W.rs;
            for (int c = 0; c < k / 64; ++c) std::memset(row + (size_t)c * 36 + 4, P.wbyte, 32);
        }
        std::vector<block_q8_0> acts((size_t)nt * (k / 32));
        for (size_t i = 0; i < acts.size(); ++i) {
            acts[i].d = f32_to_fp16(0.01f + 0.001f * (float)(i % 17));
            for (int j = 0; j < 32; ++j) acts[i].qs[j] = (i % 3 == 2 && (j & 1)) ? (int8_t)-P.xval : P.xval;  // one block in three alternates the sign
        }
        Mat Wr = reencode(W, rows, k);
        if (!check_i16_dense(W, Wr, k, rows, acts, nt, P.name)) continue;
        // double reference (independent of any accumulation order)
        AlignedBuf buf(gemm_act_bytes(GEMM_FMT_I16, nt, k));
        gemm_pack_act(GEMM_FMT_I16, acts.data(), 0, nt, nt, k, buf.p);
        std::vector<float> y((size_t)nt * rows);
        gemm_nvfp4r_i16(Wr.ptr(), Wr.rs, gemm_act_view(GEMM_FMT_I16, buf.p, nt, k), 0, nt, k, y.data(), rows, rows);
        double worst = 0;
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < rows; ++r) {
                const uint8_t* wr = (const uint8_t*)W.ptr() + (size_t)r * W.rs;
                const double ref = ref_dot(wr, k, acts.data() + (size_t)t * (k / 32)), ab = ref_abs(wr, k, acts.data() + (size_t)t * (k / 32));
                worst = std::max(worst, std::fabs((double)y[(size_t)t * rows + r] - ref) / std::max(ab, 1e-30));
            }
        std::printf("  i16 adversarial %-32s model bit-exact, max |y-ref|/sum|wx| vs double %.3g\n", P.name, worst);
        TU_CHECK(worst < 1e-6);
    }
}

void check_i16(tu::Rng& rng) {
    // dense shapes x value modes
    for (int mode = 0; mode < NMODES; ++mode)
        for (int k : {64, 192, 1152, 2304})
            for (int rows : {1, 5, 16, 17, 32, 33, 48})
                for (int nt : {1, 2, 3, 5, 8, 9, 12, 13, 17, 24, 33}) {
                    if (mode != NORMAL && (rows > 17 || nt > 13)) continue;
                    Mat W = make_nvfp4(rows, k, (rows + nt) % 2 ? 24 : 0, mode, rng);
                    Mat Wr = reencode(W, rows, k);
                    std::vector<block_q8_0> acts = make_acts(nt, k, mode == RANDBITS ? NORMAL : mode, rng);
                    check_i16_dense(W, Wr, k, rows, acts, nt, "dense");
                }
    std::printf("i16 dense done, cases %ld, fail %d\n", g_i16_cases, tu::g_fail);
    check_i16_adversarial(rng);
    check_i16_moe(2304, 36, 12, 70, 8, 16, rng);
    check_i16_moe(1152, 40, 9, 64, 8, 32, rng);
    check_i16_moe(256, 22, 7, 37, 3, 8, rng);
    check_i16_moe(64, 5, 3, 9, 1, 4, rng);
    check_i16_moe(2304, 64, 4, 200, 8, 32, rng);  // up to ~400 pairs per expert: many tiles
    std::printf("i16 moe done, cases %ld, fail %d\n", g_i16_cases, tu::g_fail);
    // numerics vs double on the realistic case (2304x2304 gate/up, Gaussian activations): exact kernel vs i16 kernel, plus Q8_0 re-quantisation flips
    {
        const int k = 2304, rows = 2304, nt = 24;
        Mat G = make_nvfp4(rows, k, 0, NORMAL, rng), U = make_nvfp4(rows, k, 0, NORMAL, rng);
        Mat Gr = reencode(G, rows, k), Ur = reencode(U, rows, k);
        std::vector<float> xf((size_t)nt * k);
        for (auto& v : xf) v = rng.gauss();
        std::vector<block_q8_0> acts((size_t)nt * (k / 32));
        for (int t = 0; t < nt; ++t) quantize_row_q8_0(xf.data() + (size_t)t * k, acts.data() + (size_t)t * (k / 32), k);
        AlignedBuf b(gemm_act_bytes(GEMM_FMT_I16, nt, k));
        gemm_pack_act(GEMM_FMT_I16, acts.data(), 0, nt, nt, k, b.p);
        const GemmAct a = gemm_act_view(GEMM_FMT_I16, b.p, nt, k);
        std::vector<float> ge((size_t)nt * rows), gi((size_t)nt * rows), ue((size_t)nt * rows), ui((size_t)nt * rows);
        gemm_nvfp4r(Gr.ptr(), Gr.rs, a, 0, nt, k, ge.data(), rows, rows);  // exact kernel on the I16 pack (superset)
        gemm_nvfp4r_i16(Gr.ptr(), Gr.rs, a, 0, nt, k, gi.data(), rows, rows);
        gemm_nvfp4r(Ur.ptr(), Ur.rs, a, 0, nt, k, ue.data(), rows, rows);
        gemm_nvfp4r_i16(Ur.ptr(), Ur.rs, a, 0, nt, k, ui.data(), rows, rows);
        double max_e = 0, max_i = 0, sum_e = 0, sum_i = 0, max_ie = 0, maxabs = 0;
        long nref = 0;
        for (int t = 0; t < nt; t += 3)
            for (int r = 0; r < rows; r += 7) {
                const uint8_t* wr = (const uint8_t*)G.ptr() + (size_t)r * G.rs;
                const block_q8_0* x = acts.data() + (size_t)t * (k / 32);
                const double ref = ref_dot(wr, k, x), ab = ref_abs(wr, k, x);
                const double de = std::fabs((double)ge[(size_t)t * rows + r] - ref) / ab, di = std::fabs((double)gi[(size_t)t * rows + r] - ref) / ab;
                max_e = std::max(max_e, de);
                max_i = std::max(max_i, di);
                sum_e += de;
                sum_i += di;
                max_ie = std::max(max_ie, std::fabs((double)gi[(size_t)t * rows + r] - (double)ge[(size_t)t * rows + r]));
                maxabs = std::max(maxabs, std::fabs(ref));
                ++nref;
            }
        std::printf("i16 numerics (2304x2304, 24 tokens, %ld samples vs double): max |y-ref|/sum|wx| exact %.3g, i16 %.3g; mean exact %.3g, i16 %.3g; max |i16-exact| %.3g (max |y| %.3g)\n",
                    nref, max_e, max_i, sum_e / nref, sum_i / nref, max_ie, maxabs);
        TU_CHECK(max_i < 1e-6);
        long flips_raw = 0, flips_act = 0, dflips_raw = 0, dflips_act = 0, n = 0;
        std::vector<float> he((size_t)rows), hi((size_t)rows);
        std::vector<block_q8_0> qa(rows / 32), qb(rows / 32);
        auto flips = [&](const float* p, const float* q, long& fq, long& fd) {
            quantize_row_q8_0(p, qa.data(), rows);
            quantize_row_q8_0(q, qb.data(), rows);
            for (int i = 0; i < rows / 32; ++i) {
                if (qa[i].d != qb[i].d) ++fd;
                for (int j = 0; j < 32; ++j) fq += qa[i].qs[j] != qb[i].qs[j];
            }
        };
        for (int t = 0; t < nt; ++t) {
            flips(ge.data() + (size_t)t * rows, gi.data() + (size_t)t * rows, flips_raw, dflips_raw);
            for (int r = 0; r < rows; ++r) {
                const float g1 = ge[(size_t)t * rows + r], g2 = gi[(size_t)t * rows + r];
                he[(size_t)r] = g1 / (1.f + std::exp(-g1)) * ue[(size_t)t * rows + r];
                hi[(size_t)r] = g2 / (1.f + std::exp(-g2)) * ui[(size_t)t * rows + r];
            }
            flips(he.data(), hi.data(), flips_act, dflips_act);
            n += rows;
        }
        std::printf("i16 Q8_0 re-quantisation flips vs exact kernel: raw GEMM output %ld of %ld int8 values (%.4f%%), %ld of %ld block scales; silu(gate)*up %ld of %ld (%.4f%%), %ld block scales\n",
                    flips_raw, n, 100.0 * flips_raw / n, dflips_raw, n / 32, flips_act, n, 100.0 * flips_act / n, dflips_act);
    }
}

}  // namespace

int main() {
    std::printf("test_gemm  loadavg %s\n", tu::loadavg().c_str());
    TU_CHECK(nvfp4r_selfcheck() == 0);  // every UE4M3 byte value re-encodes and decodes to the same float (scalar and vector)
    // optional: exercise the non-default code paths (old per-accumulator epilogue, no 5-token tail tiles); results must not change
    if (const char* e = std::getenv("INK_GEMM_EPILOGUE")) g_gemm_epilogue = std::atoi(e);
    if (const char* e = std::getenv("INK_GEMM_TAIL5")) g_gemm_tail5 = std::atoi(e);
    if (const char* e = std::getenv("INK_GEMM_TILE")) g_gemm_tile = std::atoi(e);
    std::printf("flags: epilogue %d tail5 %d tile %d\n", g_gemm_epilogue, g_gemm_tail5, g_gemm_tile);
    tu::Rng rng(20261121);
    const int Ks[] = {64, 192, 1152, 2304, 3072, 4608};
    const int Rows[] = {1, 2, 3, 4, 5, 7, 13, 33, 64, 65};
    const int NTs[] = {1, 2, 3, 5, 8, 16, 17, 64, 256};
    for (int fmt = 0; fmt < 2; ++fmt) {
        // full shape sweep with normal values
        for (int k : Ks)
            for (int rows : Rows)
                for (int nt : NTs) {
                    if ((long)k * rows * nt > 40000000L && (rows % 2 == 0)) continue;  // keep the run short, odd row counts still run
                    check_dense(fmt, k, rows, nt, NORMAL, (rows + nt) % 2 == 1, rng);
                }
        // value modes on a subset
        for (int mode = 1; mode < NMODES; ++mode)
            for (int k : {64, 1152, 2304})
                for (int rows : {1, 5, 33})
                    for (int nt : {1, 5, 17, 64}) check_dense(fmt, k, rows, nt, mode, mode & 1, rng);
        std::printf("dense fmt=%d done, cases %ld (scalar-ref cases %ld), fail %d\n", fmt, g_cases, g_ref_cases, tu::g_fail);
    }
    for (int bf = 0; bf < 2; ++bf)
        for (int k : {16, 32, 48, 64, 1152, 2304, 3072, 4608})
            for (int rows : {1, 2, 3, 5, 24, 32, 33, 128})
                for (int nt : NTs) check_small(bf == 1, k, rows, nt, rng);
    std::printf("small done, cases %ld, fail %d\n", g_cases, tu::g_fail);
    for (int fmt = 0; fmt < 2; ++fmt) {
        check_moe(fmt, 128, 22, 7, 37, 3, 8, rng);
        check_moe(fmt, 256, 40, 9, 64, 8, 12, rng);
        check_moe(fmt, 2304, 36, 12, 70, 8, 16, rng);
        check_moe(fmt, 1152, 21, 5, 130, 2, 64, rng);
        check_moe(fmt, 64, 5, 3, 9, 1, 4, rng);
    }
    // quantize+pack == quantize then pack
    for (int fmt = 0; fmt < 2; ++fmt) {
        const int k = 2304, nt = 9;
        std::vector<float> x((size_t)nt * (k + 3));
        for (auto& v : x) v = rng.gauss() * 3.f;
        std::vector<block_q8_0> q((size_t)nt * (k / 32));
        for (int t = 0; t < nt; ++t) quantize_row_q8_0(x.data() + (size_t)t * (k + 3), q.data() + (size_t)t * (k / 32), k);
        AlignedBuf a(gemm_act_bytes(fmt, nt, k)), b(gemm_act_bytes(fmt, nt, k));
        gemm_pack_act(fmt, q.data(), 0, nt, nt, k, a.p);
        gemm_quantize_pack_act(fmt, x.data(), (size_t)k + 3, 0, nt, nt, k, b.p);
        TU_CHECK(std::memcmp(a.p, b.p, gemm_act_bytes(fmt, nt, k)) == 0);
    }
    check_x4(rng);
    if (!std::getenv("INK_TEST_SKIP_I16")) check_i16(rng);
    std::printf("moe + pack done, cases %ld (scalar-ref cases %ld, re-encoded nvfp4r cases %ld), failures %d, loadavg %s\n", g_cases, g_ref_cases,
                g_r_cases, tu::g_fail, tu::loadavg().c_str());
    std::printf(tu::g_fail ? "FAILED\n" : "ALL PASS (bit-identical to B1 gemv, incl. gemm_nvfp4r / gemm_moe_nvfp4r on re-encoded scales; i16 kernels == scalar model, %ld cases)\n", g_i16_cases);
    return tu::g_fail ? 1 : 0;
}

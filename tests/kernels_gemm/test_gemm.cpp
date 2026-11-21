// Tests for the batched GEMM kernels (agent G). Every output must be BIT-IDENTICAL to the per-token gemv kernels of agent B1
// (optimised gemv_* and, on small cases, the scalar gemv_*_ref). Shapes: K in {64,192,1152,2304,3072,4608}, rows incl. odd counts,
// nt in {1,2,3,5,8,16,17,64,256}, 4 value modes (normal, extreme, zero scales, random fp16/UE4M3 bit patterns), padded row strides,
// padded ldy (guard values must survive), row/token splitting, MoE routing (token-indexed and pair-indexed activations).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ink/kernels_gemm.h"
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

}  // namespace

int main() {
    std::printf("test_gemm  loadavg %s\n", tu::loadavg().c_str());
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
    std::printf("moe + pack done, cases %ld (scalar-ref cases %ld), failures %d, loadavg %s\n", g_cases, g_ref_cases, tu::g_fail,
                tu::loadavg().c_str());
    std::printf(tu::g_fail ? "FAILED\n" : "ALL PASS (bit-identical to B1 gemv)\n");
    return tu::g_fail ? 1 : 0;
}

// GEMV kernel tests (agent B1). For every format (Q8_0, NVFP4, BF16, F32), K, row count, row stride and value mode:
//   (a) kernel == scalar reference BIT-FOR-BIT (same lane structure / fma / reduction tree; integer parts exact),
//   (b) kernel vs a double-precision dequantised dot: |err| <= 2e-4 * sum|w*x|   (relative to the sum of absolute terms,
//       the meaningful scale for a dot product; a cancelling sum has no useful relative error),
//   (c) UNIT mode (unit scales, small integers): the float result must equal the exact integer dot,
//   (d) results do not depend on how rows are split into calls (bitwise) and the fused multi-matrix entry point matches,
//   (e) Q8_0 / NVFP4 vs ggml's generic dot (dlopen of the llama.cpp build) within 3e-5 * sum|w*x|.
#include <algorithm>
#include <cstring>
#include <vector>

#include "ink/kernels_gemv.h"
#include "ink/nvfp4_repack.h"
#include "testutil.h"

using namespace ink;

enum Kind { Q8, NVFP4, BF16, F32, NKINDS };
enum Mode { NORMAL, EXTREME, ZEROSCALE, TINYHUGE, UNIT, NMODES };
static const char* kKindName[] = {"q8_0", "nvfp4", "bf16", "f32"};
static const char* kModeName[] = {"normal", "extreme", "zeroscale", "tinyhuge", "unit"};

static tu::Oracle g_oracle;
static bool g_have_oracle = false;

static bool k_ok(int kind, int k) {
    if (kind == Q8) return k % 32 == 0;
    if (kind == NVFP4) return k % 64 == 0;
    return k % 16 == 0;
}
static size_t row_bytes(int kind, int k) {
    return kind == Q8 ? row_bytes_q8_0(k) : kind == NVFP4 ? row_bytes_nvfp4(k) : kind == BF16 ? row_bytes_bf16(k) : row_bytes_f32(k);
}
static size_t pad_for(int kind) { return kind == Q8 ? 6 : kind == NVFP4 ? 7 : kind == BF16 ? 6 : 8; }

struct Prob {
    int kind = 0, k = 0, rows = 0, mode = 0;
    size_t rb = 0, rs = 0, off = 0;
    std::vector<uint8_t> wbuf;
    std::vector<block_q8_0> xq;
    std::vector<float> xf;
    const uint8_t* w() const { return wbuf.data() + off; }
};

static float fp16_of(float v) { return fp16_to_f32(f32_to_fp16(v)); }

static void gen_q8_row(tu::Rng& rng, int mode, int rowidx, block_q8_0* b, int nb) {
    for (int i = 0; i < nb; ++i) {
        float d = 0.f;
        switch (mode) {
            case NORMAL: d = 0.001f + 0.05f * rng.uni(); break;
            case EXTREME: d = 1e3f + 6e4f * rng.uni(); break;
            case ZEROSCALE: d = ((rowidx + i) % 3 == 0 || rowidx % 4 == 1) ? 0.f : 0.01f; break;
            case TINYHUGE: d = (rng.u32() & 1) ? 5.96e-8f : 6.0e4f; break;
            default: d = 1.f; break;
        }
        b[i].d = f32_to_fp16(d);
        for (int j = 0; j < 32; ++j) {
            int q;
            if (mode == NORMAL) q = std::max(-127, std::min(127, (int)std::lround(rng.gauss() * 40.f)));
            else if (mode == EXTREME) { static const int e[4] = {-128, 127, -127, 127}; q = e[rng.u32() & 3]; }
            else if (mode == UNIT) q = rng.range(-20, 20);
            else q = rng.range(-128, 127);
            b[i].qs[j] = (int8_t)q;
        }
    }
}

static void gen_nvfp4_row(tu::Rng& rng, int mode, int rowidx, block_nvfp4* b, int nsb) {
    for (int i = 0; i < nsb; ++i) {
        for (int s = 0; s < 4; ++s) {
            uint8_t d;
            switch (mode) {
                case NORMAL: d = (uint8_t)(rng.u32() & 0xFF); break;  // every byte value incl. 0x00, 0x7F, >= 0x80
                case EXTREME: { static const uint8_t e[4] = {0xFE, 0xFF, 0x7E, 0x7D}; d = e[rng.u32() & 3]; break; }
                case ZEROSCALE: d = ((rowidx + i + s) % 3 == 0 || rowidx % 4 == 1) ? ((rng.u32() & 1) ? 0x00 : 0x7F) : 0x38; break;
                case TINYHUGE: d = (rng.u32() & 1) ? 0x01 : 0x7E; break;
                default: d = 0x40; break;  // 1.0
            }
            b[i].d[s] = d;
        }
        for (int j = 0; j < 32; ++j) {
            uint8_t q = (uint8_t)(rng.u32() & 0xFF);
            if (mode == EXTREME) { static const uint8_t e[4] = {0x77, 0xFF, 0x7F, 0xF7}; q = e[rng.u32() & 3]; }
            b[i].qs[j] = q;
        }
    }
}

static float gen_float(tu::Rng& rng, int mode, int idx) {
    switch (mode) {
        case NORMAL: return rng.gauss();
        case EXTREME: { float m = std::pow(10.f, 10.f + 5.f * rng.uni()); return (rng.u32() & 1) ? m : -m; }
        case ZEROSCALE: return (idx % 3 == 0) ? 0.f : rng.gauss() * 0.01f;
        case TINYHUGE: { float m = (rng.u32() & 1) ? 1e-20f : 1e15f; return (rng.u32() & 1) ? m : -m; }
        default: return (float)rng.range(-20, 20);
    }
}

static void make_prob(Prob& p, int kind, int k, int rows, int mode, int stride_variant, tu::Rng& rng) {
    p.kind = kind; p.k = k; p.rows = rows; p.mode = mode;
    p.rb = row_bytes(kind, k);
    const size_t pad = pad_for(kind);
    p.rs = stride_variant == 0 ? p.rb : stride_variant == 1 ? p.rb + pad : 2 * p.rb + 4;
    p.off = (kind == F32) ? 4 : 2;
    p.wbuf.assign(p.off + (size_t)rows * p.rs + 128, 0xA5);  // padding + guard bytes are garbage the kernels must ignore
    for (int r = 0; r < rows; ++r) {
        uint8_t* row = p.wbuf.data() + p.off + (size_t)r * p.rs;
        if (kind == Q8) gen_q8_row(rng, mode, r, (block_q8_0*)row, k / 32);
        else if (kind == NVFP4) gen_nvfp4_row(rng, mode, r, (block_nvfp4*)row, k / 64);
        else if (kind == BF16) { uint16_t* h = (uint16_t*)row; for (int i = 0; i < k; ++i) h[i] = f32_to_bf16(gen_float(rng, mode, i + r)); }
        else { float* f = (float*)row; for (int i = 0; i < k; ++i) f[i] = gen_float(rng, mode, i + r); }
    }
    // activation
    if (kind == Q8 || kind == NVFP4) {
        p.xq.resize(k / 32);
        if (mode == UNIT) {
            for (auto& b : p.xq) { b.d = f32_to_fp16(1.f); for (int j = 0; j < 32; ++j) b.qs[j] = (int8_t)rng.range(-20, 20); }
        } else {
            std::vector<float> x(k);
            for (int i = 0; i < k; ++i) {
                switch (mode) {
                    case NORMAL: x[i] = rng.gauss() * 3.f; break;
                    case EXTREME: x[i] = rng.sym() * 1e5f; break;
                    case ZEROSCALE: x[i] = ((i / 32) % 2 == 0) ? 0.f : rng.gauss(); break;
                    default: x[i] = (rng.u32() & 1) ? 1e-6f * rng.sym() : 1e6f * rng.sym(); break;
                }
            }
            quantize_row_q8_0(x.data(), p.xq.data(), k);
        }
    } else {
        p.xf.resize(k);
        for (int i = 0; i < k; ++i) p.xf[i] = gen_float(rng, mode, i * 7 + 1);
        if (mode == TINYHUGE || mode == EXTREME) for (int i = 0; i < k; ++i) if (std::fabs(p.xf[i]) > 1e5f) p.xf[i] = p.xf[i] * 1e-10f;  // keep sums finite
    }
}

static void run_kernel(const Prob& p, int r0, int n, float* y) {
    const void* w = p.w() + (size_t)r0 * p.rs;
    switch (p.kind) {
        case Q8: gemv_q8_0(w, p.rs, p.xq.data(), p.k, y, n); break;
        case NVFP4: gemv_nvfp4(w, p.rs, p.xq.data(), p.k, y, n); break;
        case BF16: gemv_bf16(w, p.rs, p.xf.data(), p.k, y, n); break;
        default: gemv_f32(w, p.rs, p.xf.data(), p.k, y, n); break;
    }
}
static void run_ref(const Prob& p, float* y) {
    switch (p.kind) {
        case Q8: gemv_q8_0_ref(p.w(), p.rs, p.xq.data(), p.k, y, p.rows); break;
        case NVFP4: gemv_nvfp4_ref(p.w(), p.rs, p.xq.data(), p.k, y, p.rows); break;
        case BF16: gemv_bf16_ref(p.w(), p.rs, p.xf.data(), p.k, y, p.rows); break;
        default: gemv_f32_ref(p.w(), p.rs, p.xf.data(), p.k, y, p.rows); break;
    }
}

// double-precision dequantised dot and sum of absolute terms
static void run_double(const Prob& p, std::vector<double>& dot, std::vector<double>& S) {
    dot.assign(p.rows, 0.0);
    S.assign(p.rows, 0.0);
    for (int r = 0; r < p.rows; ++r) {
        const uint8_t* row = p.w() + (size_t)r * p.rs;
        double d = 0, s = 0;
        auto acc = [&](double t) { d += t; s += std::fabs(t); };
        if (p.kind == Q8) {
            const block_q8_0* wb = (const block_q8_0*)row;
            for (int b = 0; b < p.k / 32; ++b) {
                const double sc = (double)fp16_to_f32(wb[b].d) * (double)fp16_to_f32(p.xq[b].d);
                for (int j = 0; j < 32; ++j) acc(sc * (double)wb[b].qs[j] * (double)p.xq[b].qs[j]);
            }
        } else if (p.kind == NVFP4) {
            const block_nvfp4* wb = (const block_nvfp4*)row;
            for (int sb = 0; sb < p.k / 64; ++sb)
                for (int s4 = 0; s4 < 4; ++s4) {
                    const double sc = (double)ue4m3_to_f32(wb[sb].d[s4]) * (double)fp16_to_f32(p.xq[2 * sb + (s4 >> 1)].d);
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t q = wb[sb].qs[8 * s4 + j];
                        const int elo = 16 * s4 + j, ehi = 16 * s4 + 8 + j;
                        acc(sc * (double)kvalues_fp4[q & 15] * (double)p.xq[2 * sb + (elo >> 5)].qs[elo & 31]);
                        acc(sc * (double)kvalues_fp4[q >> 4] * (double)p.xq[2 * sb + (ehi >> 5)].qs[ehi & 31]);
                    }
                }
        } else if (p.kind == BF16) {
            const uint16_t* h = (const uint16_t*)row;
            for (int i = 0; i < p.k; ++i) acc((double)bf16_to_f32(h[i]) * (double)p.xf[i]);
        } else {
            const float* f = (const float*)row;
            for (int i = 0; i < p.k; ++i) acc((double)f[i] * (double)p.xf[i]);
        }
        dot[r] = d;
        S[r] = s;
    }
}

static int g_cases = 0;

static void test_one(int kind, int k, int rows, int mode, int sv, tu::Rng& rng) {
    Prob p;
    make_prob(p, kind, k, rows, mode, sv, rng);
    std::vector<float> y(rows + 1, -777.f), yr(rows), ys(rows);
    run_kernel(p, 0, rows, y.data());
    TU_CHECK(y[rows] == -777.f);  // no write past the last row
    run_ref(p, yr.data());
    std::vector<double> dd, S;
    run_double(p, dd, S);
    bool bad = false;
    for (int r = 0; r < rows; ++r) {
        if (!(y[r] == yr[r])) {
            std::fprintf(stderr, "MISMATCH kernel!=ref %s k=%d rows=%d mode=%s sv=%d row %d: %.9g vs %.9g\n", kKindName[kind], k, rows, kModeName[mode], sv, r, y[r], yr[r]);
            bad = true;
        }
        const double err = std::fabs((double)y[r] - dd[r]);
        if (!(err <= 2e-4 * S[r])) {
            std::fprintf(stderr, "MISMATCH kernel vs double %s k=%d rows=%d mode=%s sv=%d row %d: %.9g vs %.9g (S=%g)\n", kKindName[kind], k, rows, kModeName[mode], sv, r, y[r], dd[r], S[r]);
            bad = true;
        }
        if (mode == UNIT && !(y[r] == (float)dd[r])) {
            std::fprintf(stderr, "UNIT inexact %s k=%d row %d: %.9g vs %.9g\n", kKindName[kind], k, r, y[r], dd[r]);
            bad = true;
        }
        if (bad) break;
    }
    TU_CHECK(!bad);

    // split invariance: random partition of the rows into separate calls
    if (rows >= 3) {
        int r = 0;
        while (r < rows) {
            int n = std::min(rows - r, rng.range(1, 9));
            run_kernel(p, r, n, ys.data() + r);
            r += n;
        }
        bool same = true;
        for (int i = 0; i < rows; ++i) same = same && (ys[i] == y[i]);
        if (!same) std::fprintf(stderr, "SPLIT variance %s k=%d rows=%d mode=%s\n", kKindName[kind], k, rows, kModeName[mode]);
        TU_CHECK(same);
    }

    // nvfp4r (agent K1): the kernel on RE-ENCODED scale bytes == gemv_nvfp4 on the original bytes, bit for bit; == its scalar ref;
    // split-invariant (8/4/2/1-row tails) and identical for both row-group sizes
    if (kind == NVFP4) {
        Prob q = p;
        nvfp4r_repack_rows(q.wbuf.data() + q.off, q.rs, rows, k / QK_NVFP4);   // only the 4 scale bytes of every block change
        for (int r = 0; r < rows; ++r)   // the nibbles and the padding/guard garbage are untouched
            for (int sb = 0; sb < k / QK_NVFP4; ++sb)
                TU_CHECK(!std::memcmp(((const block_nvfp4*)(q.w() + (size_t)r * q.rs))[sb].qs, ((const block_nvfp4*)(p.w() + (size_t)r * p.rs))[sb].qs, 32));
        TU_CHECK(!std::memcmp(q.w() + (size_t)rows * q.rs, p.w() + (size_t)rows * p.rs, 64));
        std::vector<float> yk(rows + 1, -777.f), yr2(rows), ys2(rows);
        const int oldg = g_gemv_nvfp4r_group, oldpre = g_gemv_nvfp4r_pre;
        for (int variant = 0; variant < 4; ++variant) {
            const int grp = (variant & 1) ? 4 : 8;
            g_gemv_nvfp4r_group = grp;
            g_gemv_nvfp4r_pre = variant >> 1;
            std::fill(yk.begin(), yk.end(), -777.f);
            gemv_nvfp4r(q.w(), q.rs, q.xq.data(), k, yk.data(), rows);
            TU_CHECK(yk[rows] == -777.f);
            gemv_nvfp4r_ref(q.w(), q.rs, q.xq.data(), k, yr2.data(), rows);
            bool same = true;
            for (int r = 0; r < rows; ++r) same = same && (yk[r] == y[r]) && (yr2[r] == y[r]);
            if (!same) std::fprintf(stderr, "NVFP4R mismatch k=%d rows=%d mode=%s sv=%d group=%d pre=%d\n", k, rows, kModeName[mode], sv, grp, g_gemv_nvfp4r_pre);
            TU_CHECK(same);
            if (rows >= 3) {
                int r = 0;
                while (r < rows) {
                    const int n = std::min(rows - r, rng.range(1, 11));
                    gemv_nvfp4r(q.w() + (size_t)r * q.rs, q.rs, q.xq.data(), k, ys2.data() + r, n);
                    r += n;
                }
                bool s2 = true;
                for (int i = 0; i < rows; ++i) s2 = s2 && (ys2[i] == y[i]);
                if (!s2) std::fprintf(stderr, "NVFP4R split variance k=%d rows=%d mode=%s group=%d pre=%d\n", k, rows, kModeName[mode], grp, g_gemv_nvfp4r_pre);
                TU_CHECK(s2);
            }
        }
        g_gemv_nvfp4r_group = oldg;
        g_gemv_nvfp4r_pre = oldpre;
    }

    // ggml oracle (generic dot) for the quantised formats
    if (g_have_oracle && (kind == Q8 || kind == NVFP4) && mode != TINYHUGE) {
        for (int r = 0; r < rows; ++r) {
            float s = 0.f;
            const void* wrow = p.w() + (size_t)r * p.rs;
            (kind == Q8 ? g_oracle.dot_q8 : g_oracle.dot_nvfp4)(k, &s, 0, wrow, 0, p.xq.data(), 0, 1);
            const double err = std::fabs((double)y[r] - (double)s);
            if (!(err <= 3e-5 * S[r])) {
                std::fprintf(stderr, "ORACLE mismatch %s k=%d rows=%d mode=%s row %d: mine %.9g ggml %.9g (S=%g)\n", kKindName[kind], k, rows, kModeName[mode], r, y[r], s, S[r]);
                TU_CHECK(false);
                break;
            }
        }
    }
    ++g_cases;
}

// fused multi-matrix entry points vs per-matrix single calls
static void test_multi(int kind, int k, int mode, tu::Rng& rng) {
    const int nm = 3;
    const int nr[nm] = {5, 3, 9};
    Prob p[nm];
    std::vector<std::vector<float>> yfull(nm), ymulti(nm);
    GemvMat mats[nm];
    const bool share_x = (rng.u32() & 1) != 0;
    for (int i = 0; i < nm; ++i) {
        make_prob(p[i], kind, k, nr[i], mode, i % 3, rng);
        if (share_x && i > 0) { p[i].xq = p[0].xq; p[i].xf = p[0].xf; }
        yfull[i].assign(nr[i], 0.f);
        run_kernel(p[i], 0, nr[i], yfull[i].data());
        ymulti[i].assign(nr[i], -1.f);
        mats[i] = GemvMat{p[i].w(), p[i].rs, (kind == Q8 || kind == NVFP4) ? (const void*)p[i].xq.data() : (const void*)p[i].xf.data(), ymulti[i].data(), nr[i]};
        if (share_x) { /* identical x contents but separate buffers are fine */ }
    }
    // recompute reference with the (possibly shared) activations
    for (int i = 0; i < nm; ++i) run_kernel(p[i], 0, nr[i], yfull[i].data());
    const int total = nr[0] + nr[1] + nr[2];
    int v = 0;
    while (v < total) {
        const int n = std::min(total - v, rng.range(1, 7));
        switch (kind) {
            case Q8: gemv_multi_q8_0(mats, nm, k, v, v + n); break;
            case NVFP4: gemv_multi_nvfp4(mats, nm, k, v, v + n); break;
            case BF16: gemv_multi_bf16(mats, nm, k, v, v + n); break;
            default: gemv_multi_f32(mats, nm, k, v, v + n); break;
        }
        v += n;
    }
    bool same = true;
    for (int i = 0; i < nm; ++i)
        for (int r = 0; r < nr[i]; ++r) same = same && (ymulti[i][r] == yfull[i][r]);
    if (!same) std::fprintf(stderr, "MULTI mismatch %s k=%d mode=%s\n", kKindName[kind], k, kModeName[mode]);
    TU_CHECK(same);
    if (kind == NVFP4) {   // fused nvfp4r entry point on re-encoded copies, random virtual-row splits
        Prob q[nm];
        GemvMat rm[nm];
        std::vector<std::vector<float>> yrm(nm);
        for (int i = 0; i < nm; ++i) {
            q[i] = p[i];
            nvfp4r_repack_rows(q[i].wbuf.data() + q[i].off, q[i].rs, nr[i], k / QK_NVFP4);
            yrm[i].assign(nr[i], -1.f);
            rm[i] = GemvMat{q[i].w(), q[i].rs, q[i].xq.data(), yrm[i].data(), nr[i]};
        }
        for (int variant = 0; variant < 4; ++variant) {
            const int grp = (variant & 1) ? 4 : 8;
            const int oldg = g_gemv_nvfp4r_group, oldpre = g_gemv_nvfp4r_pre;
            g_gemv_nvfp4r_group = grp;
            g_gemv_nvfp4r_pre = variant >> 1;
            int v2 = 0;
            while (v2 < total) {
                const int n = std::min(total - v2, rng.range(1, 17));
                gemv_multi_nvfp4r(rm, nm, k, v2, v2 + n);
                v2 += n;
            }
            g_gemv_nvfp4r_group = oldg;
            g_gemv_nvfp4r_pre = oldpre;
            bool s3 = true;
            for (int i = 0; i < nm; ++i)
                for (int r = 0; r < nr[i]; ++r) s3 = s3 && (yrm[i][r] == yfull[i][r]);
            if (!s3) std::fprintf(stderr, "MULTI nvfp4r mismatch k=%d mode=%s group=%d\n", k, kModeName[mode], grp);
            TU_CHECK(s3);
        }
    }
    ++g_cases;
}

int main() {
    g_have_oracle = g_oracle.load();
    std::printf("ggml oracle: %s\n", g_have_oracle ? "loaded" : "NOT available (skipped)");
    tu::Rng rng(777);
    const int ks[] = {32, 64, 96, 160, 16, 48, 1152, 2304, 3072, 4608, 352};
    const int rowsl[] = {1, 2, 3, 4, 5, 7, 8, 9, 13, 33};
    int cnt = 0;
    for (int kind = 0; kind < NKINDS; ++kind)
        for (int k : ks) {
            if (!k_ok(kind, k)) continue;
            for (int rows : rowsl)
                for (int mode = 0; mode < NMODES; ++mode) {
                    // all three stride variants for small K; random one for the big shapes (keeps the runtime low)
                    if (k <= 352) { for (int sv = 0; sv < 3; ++sv) test_one(kind, k, rows, mode, sv, rng); }
                    else test_one(kind, k, rows, mode, (cnt++) % 3, rng);
                }
            for (int mode = 0; mode < NMODES; ++mode) test_multi(kind, k, mode, rng);
        }
    {   // nvfp4r encoding: self-check of the header, byte table == nvfp4r_encode, block repack == per-byte encode
        TU_CHECK(nvfp4r_selfcheck() == 0);
        const uint8_t* tab = nvfp4r_encode_table();
        for (int b = 0; b < 256; ++b) TU_CHECK(tab[b] == nvfp4r_encode((uint8_t)b));
        std::vector<uint8_t> blk(36 * 300), blk2;
        for (size_t i = 0; i < blk.size(); ++i) blk[i] = (uint8_t)rng.u32();
        blk2 = blk;
        nvfp4r_repack_blocks(blk2.data(), 300);
        for (int i = 0; i < 300; ++i) {
            for (int s = 0; s < 4; ++s) TU_CHECK(blk2[36 * i + s] == nvfp4r_encode(blk[36 * i + s]));
            TU_CHECK(!std::memcmp(&blk2[36 * i + 4], &blk[36 * i + 4], 32));
        }
        ++g_cases;
    }
    {   // nrows == 0 must be a no-op
        float y = 5.f;
        gemv_q8_0(nullptr, 0, nullptr, 32, &y, 0);
        gemv_nvfp4(nullptr, 0, nullptr, 64, &y, 0);
        gemv_nvfp4r(nullptr, 0, nullptr, 64, &y, 0);
        gemv_bf16(nullptr, 0, nullptr, 32, &y, 0);
        gemv_f32(nullptr, 0, nullptr, 32, &y, 0);
        TU_CHECK(y == 5.f);
    }
    {   // prefetch knobs must not change results
        Prob p;
        make_prob(p, NVFP4, 2304, 13, NORMAL, 1, rng);
        std::vector<float> a(13), b(13);
        const int oldpf = g_gemv_prefetch_bytes, oldh = g_gemv_prefetch_hint;
        run_kernel(p, 0, 13, a.data());
        for (int h = 0; h < 4; ++h)
            for (int pf : {0, 64, 512, 4096}) {
                g_gemv_prefetch_hint = h; g_gemv_prefetch_bytes = pf;
                run_kernel(p, 0, 13, b.data());
                for (int i = 0; i < 13; ++i) TU_CHECK(a[i] == b[i]);
            }
        g_gemv_prefetch_bytes = oldpf; g_gemv_prefetch_hint = oldh;
        // same for nvfp4r (both group sizes) on the re-encoded copy
        nvfp4r_repack_rows(p.wbuf.data() + p.off, p.rs, 13, 2304 / QK_NVFP4);
        gemv_nvfp4r(p.w(), p.rs, p.xq.data(), 2304, b.data(), 13);
        for (int i = 0; i < 13; ++i) TU_CHECK(a[i] == b[i]);
        for (int grp : {4, 8})
            for (int h = 0; h < 4; ++h)
                for (int pf : {0, 64, 512, 4096}) {
                    g_gemv_prefetch_hint = h; g_gemv_prefetch_bytes = pf; g_gemv_nvfp4r_group = grp;
                    gemv_nvfp4r(p.w(), p.rs, p.xq.data(), 2304, b.data(), 13);
                    for (int i = 0; i < 13; ++i) TU_CHECK(a[i] == b[i]);
                }
        g_gemv_prefetch_bytes = oldpf; g_gemv_prefetch_hint = oldh; g_gemv_nvfp4r_group = 8;
    }
    std::printf("test_gemv: %d cases (kernel==ref bitwise, vs double, unit-exact, split-invariant, multi, nvfp4r==nvfp4%s)\n", g_cases, g_have_oracle ? ", vs ggml" : "");
    if (tu::g_fail) {
        std::printf("test_gemv: %d FAILURES\n", tu::g_fail);
        return 1;
    }
    std::printf("test_gemv: all ok\n");
    return 0;
}

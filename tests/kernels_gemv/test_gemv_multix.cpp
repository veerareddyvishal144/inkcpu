// Tests for the small-n multi-activation GEMV kernels (agent K2, wave 1). Every y[j][r] must be BIT-IDENTICAL to gemv_q8_0 / gemv_nvfp4 applied to
// activation j and row r (and to the scalar gemv_*_ref on small cases); gemv_nvfp4r_multix on re-encoded scale bytes (nvfp4_repack.h) must equal
// gemv_nvfp4 on the original bytes. Shapes: K in {64,192,1152,2304,3072,4608}, rows incl. odd counts and every row-group tail, nx 1..8 (+ 9, 17 for the
// > kMultixMaxX loop), 4 value modes (normal, extreme +-128 / 65504 / denormal / zero scales, random fp16 / UE4M3 bit patterns), padded row strides,
// padded ldy with guard values, indirect activation pointers (shuffled), prefetch knobs.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ink/kernels_gemv.h"
#include "ink/kernels_gemv_multix.h"
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
    void* mptr() { return buf.data() + 4; }
};

Mat make_q8(int rows, int k, int pad, int mode, tu::Rng& rng) {
    Mat m;
    m.rs = row_bytes_q8_0(k) + (size_t)pad;
    m.buf.assign(m.rs * (size_t)rows + 16, 0xA5);
    for (int r = 0; r < rows; ++r) {
        block_q8_0* b = (block_q8_0*)((char*)m.buf.data() + 4 + (size_t)r * m.rs);
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

constexpr uint32_t kSentinel = 0x7FC0DEADu;
bool same_bits(const float* a, const float* b, size_t n, size_t* first = nullptr) {
    for (size_t i = 0; i < n; ++i)
        if (std::memcmp(a + i, b + i, 4) != 0) {
            if (first) *first = i;
            return false;
        }
    return true;
}

long g_cases = 0, g_ref_cases = 0, g_r_cases = 0;

// one case: fmt (0 Q8, 1 NVFP4), shape, mode; checks multix vs gemv (+ ref), and nvfp4r multix on a re-encoded copy
void check(int fmt, int k, int rows, int nx, int mode, bool pad_stride, tu::Rng& rng) {
    const int pad = pad_stride ? 6 : 0;
    Mat W = fmt == 0 ? make_q8(rows, k, pad, mode, rng) : make_nvfp4(rows, k, pad, mode, rng);
    const int nb = k / 32;
    // activation pool larger than nx, addressed indirectly in a shuffled order (some activations may repeat)
    const int pool = nx + 3;
    std::vector<block_q8_0> acts = make_acts(pool, k, mode, rng);
    std::vector<const block_q8_0*> xs((size_t)nx);
    for (int j = 0; j < nx; ++j) xs[(size_t)j] = acts.data() + (size_t)(rng.u32() % (uint32_t)pool) * nb;
    const size_t ldy = (size_t)rows + 3;
    uint32_t sv = kSentinel;
    float sf;
    std::memcpy(&sf, &sv, 4);
    std::vector<float> y((size_t)nx * ldy, sf), yref((size_t)nx * ldy, sf), yscal((size_t)nx * ldy, sf);
    for (int j = 0; j < nx; ++j) {
        if (fmt == 0)
            gemv_q8_0(W.ptr(), W.rs, xs[(size_t)j], k, yref.data() + (size_t)j * ldy, rows);
        else
            gemv_nvfp4(W.ptr(), W.rs, xs[(size_t)j], k, yref.data() + (size_t)j * ldy, rows);
    }
    if (fmt == 0)
        gemv_q8_0_multix(W.ptr(), W.rs, rows, xs.data(), nx, k, y.data(), ldy);
    else
        gemv_nvfp4_multix(W.ptr(), W.rs, rows, xs.data(), nx, k, y.data(), ldy);
    ++g_cases;
    for (int j = 0; j < nx; ++j) {
        size_t first = 0;
        if (!same_bits(y.data() + (size_t)j * ldy, yref.data() + (size_t)j * ldy, rows, &first)) {
            std::fprintf(stderr, "MISMATCH fmt=%d K=%d rows=%d nx=%d mode=%d pad=%d act=%d row=%zu multix=%.9g gemv=%.9g\n", fmt, k, rows, nx, mode, (int)pad_stride, j,
                         first, (double)y[(size_t)j * ldy + first], (double)yref[(size_t)j * ldy + first]);
            ++tu::g_fail;
            return;
        }
        for (size_t i = rows; i < ldy; ++i)
            if (std::memcmp(&y[(size_t)j * ldy + i], &sf, 4) != 0) {
                std::fprintf(stderr, "GUARD overwritten fmt=%d K=%d rows=%d nx=%d act=%d\n", fmt, k, rows, nx, j);
                ++tu::g_fail;
                return;
            }
    }
    if ((double)rows * nx * k <= 4e6) {  // scalar reference on small cases
        ++g_ref_cases;
        for (int j = 0; j < nx; ++j) {
            if (fmt == 0)
                gemv_q8_0_ref(W.ptr(), W.rs, xs[(size_t)j], k, yscal.data() + (size_t)j * ldy, rows);
            else
                gemv_nvfp4_ref(W.ptr(), W.rs, xs[(size_t)j], k, yscal.data() + (size_t)j * ldy, rows);
            size_t first = 0;
            if (!same_bits(y.data() + (size_t)j * ldy, yscal.data() + (size_t)j * ldy, rows, &first)) {
                std::fprintf(stderr, "MISMATCH vs scalar ref fmt=%d K=%d rows=%d nx=%d mode=%d act=%d row=%zu\n", fmt, k, rows, nx, mode, j, first);
                ++tu::g_fail;
                return;
            }
        }
    }
    if (fmt == 1) {  // re-encoded scales: nvfp4r multix on the repacked copy == gemv_nvfp4 on the original (bits)
        Mat Wr = W;
        nvfp4r_repack_rows((uint8_t*)Wr.mptr(), Wr.rs, rows, k / 64);
        std::vector<float> yr((size_t)nx * ldy, sf);
        gemv_nvfp4r_multix(Wr.ptr(), Wr.rs, rows, xs.data(), nx, k, yr.data(), ldy);
        ++g_r_cases;
        for (int j = 0; j < nx; ++j) {
            size_t first = 0;
            if (!same_bits(yr.data() + (size_t)j * ldy, yref.data() + (size_t)j * ldy, rows, &first)) {
                std::fprintf(stderr, "MISMATCH nvfp4r K=%d rows=%d nx=%d mode=%d act=%d row=%zu r=%.9g gemv=%.9g\n", k, rows, nx, mode, j, first,
                             (double)yr[(size_t)j * ldy + first], (double)yref[(size_t)j * ldy + first]);
                ++tu::g_fail;
                return;
            }
        }
    }
}

}  // namespace

int main() {
    std::printf("test_gemv_multix  loadavg %s\n", tu::loadavg().c_str());
    if (int rc = nvfp4r_selfcheck()) {
        std::fprintf(stderr, "nvfp4r_selfcheck failed at byte %d\n", rc - 1);
        return 1;
    }
    tu::Rng rng(20261123);
    const int Ks[] = {64, 192, 1152, 2304, 3072, 4608};
    const int Rows[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 32, 33};
    const int NXs[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 17};
    for (int fmt = 0; fmt < 2; ++fmt) {
        for (int k : Ks)
            for (int rows : Rows)
                for (int nx : NXs) {
                    if ((long)k * rows * nx > 6000000L && (rows % 2 == 0)) continue;  // keep it short; odd row counts still run everywhere
                    check(fmt, k, rows, nx, NORMAL, (rows + nx) % 2 == 1, rng);
                }
        for (int mode = 1; mode < NMODES; ++mode)
            for (int k : {64, 1152, 2304})
                for (int rows : {1, 3, 5, 8, 9, 33})
                    for (int nx : {1, 2, 3, 4, 5, 8, 9}) check(fmt, k, rows, nx, mode, mode & 1, rng);
        std::printf("fmt=%d done, cases %ld (scalar-ref cases %ld, nvfp4r cases %ld), fail %d\n", fmt, g_cases, g_ref_cases, g_r_cases, tu::g_fail);
    }
    // every UE4M3 scale byte value (0..255, incl. 0x00 / 0x7F = 0, the ignored sign bit, denormal codes 1..7) through the 4-row vector decoders: 8 rows x K=2304 = 1152 scale
    // bytes per matrix cycling through all 256 values at three phase offsets; nx 1 (R=8 path), 3 (R=4 path), 9 (two groups)
    for (int phase = 0; phase < 3; ++phase) {
        const int k = 2304, rows = 8, nc = k / 64;
        Mat W = make_nvfp4(rows, k, 0, NORMAL, rng);
        int b = phase * 85;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < nc; ++c)
                for (int q = 0; q < 4; ++q) W.buf[4 + (size_t)r * W.rs + (size_t)c * 36 + q] = (uint8_t)(b++ & 0xFF);
        for (int nx : {1, 3, 9}) {
            std::vector<block_q8_0> acts = make_acts(nx, k, NORMAL, rng);
            std::vector<const block_q8_0*> xs((size_t)nx);
            for (int j = 0; j < nx; ++j) xs[(size_t)j] = acts.data() + (size_t)j * (k / 32);
            std::vector<float> y((size_t)nx * rows), yref((size_t)nx * rows), yr((size_t)nx * rows);
            for (int j = 0; j < nx; ++j) gemv_nvfp4(W.ptr(), W.rs, xs[(size_t)j], k, yref.data() + (size_t)j * rows, rows);
            gemv_nvfp4_multix(W.ptr(), W.rs, rows, xs.data(), nx, k, y.data(), rows);
            Mat Wr = W;
            nvfp4r_repack_rows((uint8_t*)Wr.mptr(), Wr.rs, rows, nc);
            gemv_nvfp4r_multix(Wr.ptr(), Wr.rs, rows, xs.data(), nx, k, yr.data(), rows);
            ++g_cases;
            TU_CHECK(same_bits(y.data(), yref.data(), y.size()));
            TU_CHECK(same_bits(yr.data(), yref.data(), y.size()));
        }
    }
    std::printf("all-256 scale bytes: done, fail %d\n", tu::g_fail);
    // prefetch knobs must not change results
    for (int hint : {3, 1, 2, 0}) {
        g_gemv_prefetch_hint = hint;
        g_gemv_prefetch_bytes = hint == 1 ? 1024 : 512;
        for (int fmt = 0; fmt < 2; ++fmt) check(fmt, 2304, 13, 5, NORMAL, true, rng);
    }
    g_gemv_prefetch_hint = 0;
    g_gemv_prefetch_bytes = 512;
    std::printf("done, cases %ld (scalar-ref %ld, nvfp4r %ld), failures %d, loadavg %s\n", g_cases, g_ref_cases, g_r_cases, tu::g_fail, tu::loadavg().c_str());
    std::printf(tu::g_fail ? "FAILED\n" : "ALL PASS (bit-identical to B1 gemv, scalar ref and nvfp4 on re-encoded scales)\n");
    return tu::g_fail ? 1 : 0;
}

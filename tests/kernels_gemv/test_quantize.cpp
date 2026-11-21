// quantize_row_q8_0 (AVX-512) must be bit-exact with quantize_row_q8_0_ref (scalar copy of ggml's algorithm) and, when
// the llama.cpp build-sys libraries can be dlopen'ed, with ggml's own quantize_row_q8_0_ref. Agent B1.
#include <vector>

#include "ink/kernels_gemv.h"
#include "testutil.h"

using namespace ink;

static tu::Oracle g_oracle;
static bool g_have_oracle = false;

static void check_vec(const std::vector<float>& x, const char* what) {
    const int k = (int)x.size();
    const int nb = k / QK8_0;
    std::vector<block_q8_0> a(nb), b(nb), c(nb);
    quantize_row_q8_0(x.data(), a.data(), k);
    quantize_row_q8_0_ref(x.data(), b.data(), k);
    const bool ok1 = std::memcmp(a.data(), b.data(), nb * sizeof(block_q8_0)) == 0;
    if (!ok1) {
        for (int i = 0; i < nb; ++i)
            if (std::memcmp(&a[i], &b[i], sizeof(block_q8_0))) {
                std::fprintf(stderr, "simd!=ref (%s) k=%d block %d: d %04x vs %04x\n", what, k, i, a[i].d, b[i].d);
                for (int j = 0; j < 32; ++j)
                    if (a[i].qs[j] != b[i].qs[j])
                        std::fprintf(stderr, "  j=%d x=%a simd=%d ref=%d\n", j, x[i * 32 + j], a[i].qs[j], b[i].qs[j]);
                break;
            }
    }
    TU_CHECK(ok1);
    if (g_have_oracle) {
        g_oracle.q8_ref(x.data(), c.data(), k);
        const bool ok2 = std::memcmp(b.data(), c.data(), nb * sizeof(block_q8_0)) == 0;
        if (!ok2) std::fprintf(stderr, "ref!=ggml (%s) k=%d\n", what, k);
        TU_CHECK(ok2);
    }
}

int main() {
    g_have_oracle = g_oracle.load();
    std::printf("ggml oracle: %s\n", g_have_oracle ? "loaded" : "NOT available (skipped)");
    tu::Rng rng(12345);
    const int ks[] = {32, 64, 96, 1152, 2304, 3072, 4608};
    const float scales[] = {1e-6f, 1e-3f, 1.f, 30.f, 1e3f, 1e6f, 1e20f};
    int nvec = 0;
    for (int k : ks)
        for (float sc : scales)
            for (int trial = 0; trial < 40; ++trial) {
                std::vector<float> x(k);
                for (int i = 0; i < k; ++i) x[i] = rng.gauss() * sc;
                if (trial % 5 == 1)  // sparse blocks with outliers
                    for (int i = 0; i < k; ++i)
                        if (rng.u32() % 8) x[i] *= 1e-3f;
                if (trial % 5 == 2)  // exact grid values: many exact .5 ties after scaling
                    for (int i = 0; i < k; ++i) x[i] = (float)rng.range(-127, 127) * sc * 0.5f;
                if (trial % 5 == 3)  // some blocks all zero, some with only one non-zero
                    for (int b = 0; b < k / 32; ++b) {
                        if (b % 3 == 0) for (int j = 0; j < 32; ++j) x[b * 32 + j] = 0.f;
                        if (b % 3 == 1) for (int j = 1; j < 32; ++j) x[b * 32 + j] = 0.f;
                    }
                check_vec(x, "random");
                ++nvec;
            }
    // constructed exact ties: amax = 127 -> d = 1, id = 1, so m + 0.5 must round away from zero
    for (int shift = -20; shift <= 20; shift += 5) {
        std::vector<float> x(64);
        for (int b = 0; b < 2; ++b) {
            x[b * 32] = 127.f;
            for (int j = 1; j < 32; ++j) x[b * 32 + j] = (float)(j - 16 + shift) + (((j + b) & 1) ? 0.5f : -0.5f);
        }
        check_vec(x, "ties");
        ++nvec;
    }
    {   // special: negative zero, denormals, huge, single element
        std::vector<float> x(96, 0.f);
        x[0] = -0.0f; x[33] = 1e-40f; x[34] = -1e-40f; x[64] = 3.0e38f; x[65] = -3.0e38f; x[66] = 1.0f;
        check_vec(x, "special");
        ++nvec;
        std::vector<float> z(32, 0.f);
        check_vec(z, "zeros");
        ++nvec;
    }
    std::printf("quantize: %d vectors compared (SIMD vs scalar%s)\n", nvec, g_have_oracle ? " vs ggml" : "");
    if (tu::g_fail) {
        std::printf("test_quantize: %d FAILURES\n", tu::g_fail);
        return 1;
    }
    std::printf("test_quantize: all ok\n");
    return 0;
}

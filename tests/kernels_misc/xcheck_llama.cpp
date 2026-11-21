// Cross-check of the private Q8_0 quantiser against the REAL llama.cpp binaries (build-sys shared libs, loaded with dlopen):
//   quantize_row_q8_0_ref  (libggml-base.so)  <->  ink::quantize_q8_0(Q8Mode::Ref)
//   quantize_row_q8_0      (libggml-cpu.so, the AVX path llama.cpp runs on this CPU)  <->  ink::quantize_q8_0(Q8Mode::X86)
// Not part of the default build:  g++ -O2 -std=c++17 -march=skylake-avx512 -Iinclude tests/kernels_misc/xcheck_llama.cpp
//   build-b2s/kernels_misc/libink_kernels_misc.a -ldl -o build-b2s/xcheck_llama && build-b2s/xcheck_llama
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "ink/kernels_misc.h"

using namespace ink;
typedef void (*qfn)(const float*, void*, int64_t);

int main() {
    const char* dir = "/data/inkling-mini/llama.cpp/build-sys/bin/";
    std::string pb = std::string(dir) + "libggml-base.so", pc = std::string(dir) + "libggml-cpu.so";
    void* hb = dlopen(pb.c_str(), RTLD_NOW | RTLD_GLOBAL);
    void* hc = dlopen(pc.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!hb || !hc) { std::printf("dlopen failed: %s\n", dlerror()); return 2; }
    qfn ref_fn = (qfn)dlsym(hb, "quantize_row_q8_0_ref");
    qfn x86_fn = (qfn)dlsym(hc, "quantize_row_q8_0");
    if (!ref_fn || !x86_fn) { std::printf("dlsym failed\n"); return 2; }

    std::mt19937_64 rng(99);
    const size_t nb = 1 << 16, n = nb * 32;
    std::vector<float> x(n);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    for (size_t b = 0; b < nb; ++b) {
        const float s = std::pow(10.0f, ud(rng));
        for (int j = 0; j < 32; ++j) x[b * 32 + j] = nd(rng) * s;
    }
    // exact-tie blocks: amax 127 -> d = 1, id = 1, values k + 0.5
    for (size_t b = 0; b < 2048; ++b) {
        for (int j = 0; j < 32; ++j) x[b * 32 + j] = (float)(((int)(rng() % 126)) + 0.5) * ((rng() & 1) ? 1.0f : -1.0f);
        x[b * 32 + 31] = 127.0f;
    }
    std::vector<block_q8_0> lr(nb), lx(nb), mr(nb), mx(nb);
    ref_fn(x.data(), lr.data(), (int64_t)n);
    x86_fn(x.data(), lx.data(), (int64_t)n);
    quantize_q8_0(x.data(), mr.data(), n, Q8Mode::Ref);
    quantize_q8_0(x.data(), mx.data(), n, Q8Mode::X86);
    size_t bad_ref = 0, bad_x86 = 0, ggml_differ = 0;
    for (size_t b = 0; b < nb; ++b) {
        bad_ref += std::memcmp(&lr[b], &mr[b], 34) != 0;
        bad_x86 += std::memcmp(&lx[b], &mx[b], 34) != 0;
        ggml_differ += std::memcmp(&lr[b], &lx[b], 34) != 0;
    }
    std::printf("blocks %zu: ink Ref vs llama quantize_row_q8_0_ref mismatches %zu; ink X86 vs llama quantize_row_q8_0 (AVX) mismatches %zu\n", nb, bad_ref, bad_x86);
    std::printf("llama's own ref vs AVX quantisers differ in %zu of %zu blocks (%.3f%%)\n", ggml_differ, nb, 100.0 * ggml_differ / nb);
    return (bad_ref || bad_x86) ? 1 : 0;
}

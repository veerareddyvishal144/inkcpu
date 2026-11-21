// Shared helpers for the kernels_gemv tests and benchmarks (agent B1).
#pragma once
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/ggml_types.h"

namespace tu {

inline int g_fail = 0;
#define TU_CHECK(c)                                                                  \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::fprintf(stderr, "CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++tu::g_fail;                                                            \
        }                                                                            \
    } while (0)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ull;
    }
    uint32_t u32() { return (uint32_t)(next() >> 32); }
    int range(int lo, int hi) { return lo + (int)(u32() % (uint32_t)(hi - lo + 1)); }  // inclusive
    float uni() { return (float)(next() >> 40) * (1.0f / 16777216.0f); }               // [0,1)
    float sym() { return uni() * 2.f - 1.f; }
    float gauss() { return (uni() + uni() + uni() + uni() - 2.f) * 1.7320508f; }       // ~N(0,1)
};

inline std::string loadavg() {
    char buf[128] = {0};
    if (FILE* f = std::fopen("/proc/loadavg", "r")) {
        if (!std::fgets(buf, sizeof buf, f)) buf[0] = 0;
        std::fclose(f);
    }
    std::string s(buf);
    size_t p = 0;
    for (int i = 0; i < 3 && p != std::string::npos; ++i) p = s.find(' ', p + 1);
    return p == std::string::npos ? s : s.substr(0, p);
}

// ggml oracle (llama.cpp build-sys shared libraries, loaded with dlopen; tests skip the oracle if unavailable)
struct Oracle {
    typedef void (*dot_t)(int, float*, size_t, const void*, size_t, const void*, size_t, int);
    void* hb = nullptr;
    void* hc = nullptr;
    void (*q8_ref)(const float*, ink::block_q8_0*, int64_t) = nullptr;
    dot_t dot_q8 = nullptr;
    dot_t dot_nvfp4 = nullptr;
    bool load() {
        const char* dir = std::getenv("INK_GGML_LIB_DIR");
        const std::string d = dir ? dir : "/data/inkling-mini/llama.cpp/build-sys/bin";
        hb = dlopen((d + "/libggml-base.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!hb) {
            std::fprintf(stderr, "oracle unavailable: %s\n", dlerror());
            return false;
        }
        hc = dlopen((d + "/libggml-cpu.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!hc) {
            std::fprintf(stderr, "oracle unavailable: %s\n", dlerror());
            return false;
        }
        // ggml fills its fp16 / ue4m3 conversion tables in ggml_init() / ggml_cpu_init(); the generic dots read them
        struct InitParams { size_t mem_size; void* mem_buffer; bool no_alloc; };
        auto ggml_init_fn = (void* (*)(InitParams))dlsym(hb, "ggml_init");
        auto ggml_free_fn = (void (*)(void*))dlsym(hb, "ggml_free");
        auto cpu_init_fn = (void (*)())dlsym(hc, "ggml_cpu_init");
        if (ggml_init_fn && ggml_free_fn) ggml_free_fn(ggml_init_fn(InitParams{1 << 20, nullptr, true}));
        if (cpu_init_fn) cpu_init_fn();
        q8_ref = (decltype(q8_ref))dlsym(hb, "quantize_row_q8_0_ref");
        dot_q8 = (dot_t)dlsym(hc, "ggml_vec_dot_q8_0_q8_0_generic");
        dot_nvfp4 = (dot_t)dlsym(hc, "ggml_vec_dot_nvfp4_q8_0_generic");
        return q8_ref && dot_q8 && dot_nvfp4;
    }
};

}  // namespace tu

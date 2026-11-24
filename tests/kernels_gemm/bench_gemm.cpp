// Benchmarks for the batched GEMM kernels (agent G). Run through tools/with_bench_lock.sh. Prints GMAC/s, GMAC/s per thread and the
// equivalent prompt tokens/s (5.35 GMAC per token), plus /proc/loadavg before and after every section.
//   bench_gemm [--what st|dense|nvfp4|moe|small|rvs|all] [--threads 16] [--cpus 2,4,...] [--reps 7] [--trace PATH] [--visit 12] [--group 64] [--chunk 64]
// --what rvs (K3): single-thread A/B of gemm_nvfp4 (GGUF scale bytes) vs gemm_nvfp4r (re-encoded scales), interleaved reps in the same
// window, plus the per-panel unpack cost derived from the nt=4 / nt=64 timings (T(nt) = U + nt*C). The moe section runs both kernels too.
#include <immintrin.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/kernels_gemm.h"
#include "ink/kernels_gemv_multix.h"
#include "ink/nvfp4_repack.h"
#include "ink/threadpool.h"
#include "testutil.h"

using namespace ink;

static double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static constexpr double kGmacPerToken = 5.35;

// Same-window ceiling: the 5-uop mix (mask_sub, maddubs, madd, cvt, fma; L1 operands) of bench_peak, GMAC/s of ONE core right now.
__attribute__((noinline)) static float peak_loop(long iters, const int8_t* xb, const float* sb) {
    constexpr int NP = 12;
    __m512 acc[NP];
    _Pragma("GCC unroll 12") for (int p = 0; p < NP; ++p) acc[p] = _mm512_set1_ps(0.5f + 0.001f * (float)p);
    const __m512i ones = _mm512_set1_epi16(1), zero = _mm512_setzero_si512(), aw = _mm512_set1_epi8(37);
    const __mmask64 km = (__mmask64)0xF0F0F0F00F0F0F0Full;
    for (long it = 0; it < iters; ++it) {
        asm volatile("" ::: "memory");
        _Pragma("GCC unroll 12") for (int p = 0; p < NP; ++p) {
            __m512i X = _mm512_load_si512(xb + 64 * p);
            X = _mm512_mask_sub_epi8(X, km, zero, X);
            const __m512i I = _mm512_madd_epi16(_mm512_maddubs_epi16(aw, X), ones);
            const __m512 S = _mm512_castpd_ps(_mm512_set1_pd(*(const double*)(sb + 16 * p)));
            acc[p] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(I), S, acc[p]);
        }
    }
    __m512 t = acc[0];
    _Pragma("GCC unroll 12") for (int p = 1; p < NP; ++p) t = _mm512_add_ps(t, acc[p]);
    return _mm512_reduce_add_ps(t);
}
static double peak_gmacs() {
    alignas(64) static thread_local int8_t xb[12 * 64];
    alignas(64) static thread_local float sb[12 * 16];
    for (int i = 0; i < 12 * 64; ++i) xb[i] = (int8_t)((i * 37 + 11) % 255 - 127);
    for (int i = 0; i < 12 * 16; ++i) sb[i] = 0.001f * (float)((i % 7) + 1);
    const long iters = 100000;
    volatile float sink = 0;
    double best = 1e30;
    for (int rep = 0; rep < 6; ++rep) {
        const double t0 = now_s();
        sink = sink + peak_loop(iters, xb, sb);
        best = std::min(best, now_s() - t0);
    }
    return 64.0 * (double)iters * 12.0 / best * 1e-9;
}

struct Opt {
    std::string what = "all", trace = "/data/inkling-mini/engine/tools/route_traces/code.rtrace";
    int threads = 16, reps = 7, visit = 12, group = 64, chunk = 64;
    std::vector<int> cpus;  // --cpus a,b,c (default 2,4,...,30,0)
    std::vector<int> ntok = {64, 128, 256};  // --ntok a,b,c: token chunks for the MoE section
    int i16tile = 0, i16seg = 0, i16pf = -1;    // --i16tile N / --i16seg S / --i16pf 0|1: override the i16 kernel tunables (P2)
    std::vector<int> mx = {8, 4, 0};             // --mx a,b,c: multix thresholds of the engine-mix arms in --what moe (experts with <= mx pairs run gemv_nvfp4r_multix)
} opt;

template <class F>
static std::pair<double, double> time_best(F&& f, int reps) {  // (best, median) seconds
    f();                                                      // warm-up
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) {
        const double t0 = now_s();
        f();
        v.push_back(now_s() - t0);
    }
    std::sort(v.begin(), v.end());
    return {v.front(), v[v.size() / 2]};
}
static void report(const char* name, double gmac, double best, double med, int thr, double peak_total = 0) {
    char pk[48] = "";
    if (peak_total > 0) std::snprintf(pk, sizeof pk, " %5.1f%% of peak %.0f", 100.0 * gmac / best / peak_total, peak_total);
    std::printf("%-44s thr=%2d best %8.3f ms (med %8.3f) %7.1f GMAC/s %6.2f/thr = %6.1f tok/s%s  load %s\n", name, thr, best * 1e3,
                med * 1e3, gmac / best, gmac / best / thr, (gmac / best) / kGmacPerToken, pk, tu::loadavg().c_str());
    std::fflush(stdout);
}

static uint64_t g_s = 88172645463325252ull;
static inline uint64_t xr() {
    g_s ^= g_s << 13;
    g_s ^= g_s >> 7;
    g_s ^= g_s << 17;
    return g_s;
}
static void fill_random(void* p, size_t n) {
    uint64_t* q = (uint64_t*)p;
    for (size_t i = 0; i < n / 8; ++i) q[i] = xr();
}
static void fix_q8(char* base, size_t rows, size_t rs, int k) {  // sane fp16 scales
    for (size_t r = 0; r < rows; ++r)
        for (int b = 0; b < k / 32; ++b) {
            uint16_t d = 0x2800 + (uint16_t)(xr() & 0xFF);
            std::memcpy(base + r * rs + (size_t)b * 34, &d, 2);
        }
}
static void fix_nv(char* base, size_t rows, size_t rs, int k) {  // sane UE4M3 scales
    for (size_t r = 0; r < rows; ++r)
        for (int b = 0; b < k / 64; ++b)
            for (int s = 0; s < 4; ++s) base[r * rs + (size_t)b * 36 + s] = (char)(0x30 + (xr() % 0x18));
}
static std::vector<block_q8_0> make_acts(int nt, int k) {
    std::vector<block_q8_0> a((size_t)nt * (k / 32));
    for (auto& b : a) {
        b.d = 0x2400 + (uint16_t)(xr() & 0x3F);
        for (int j = 0; j < 32; ++j) b.qs[j] = (int8_t)((int)(xr() % 255) - 127);
    }
    return a;
}
struct ActBuf {
    void* p = nullptr;
    GemmAct a;
    ActBuf(int fmt, int nt, int k, const std::vector<block_q8_0>* shared = nullptr) {
        p = std::aligned_alloc(64, (gemm_act_bytes(fmt, nt, k) + 63) & ~(size_t)63);
        std::vector<block_q8_0> q = shared ? *shared : make_acts(nt, k);
        gemm_pack_act(fmt, q.data(), 0, nt, nt, k, p);
        a = gemm_act_view(fmt, p, nt, k);
    }
    ~ActBuf() { std::free(p); }
};

static double peak_mt(ThreadPool& pool) {
    std::vector<double> v((size_t)pool.size(), 0.0);
    pool.run([&](int tid, int) {
        pool.barrier();
        v[(size_t)tid] = peak_gmacs();
        pool.barrier();
    });
    double s = 0;
    for (double x : v) s += x;
    return s;
}
static double g_peak_mt = 0;

static std::vector<int> default_cpus() {
    if (!opt.cpus.empty()) return opt.cpus;
    std::vector<int> c;
    for (int i = 1; i <= 15; ++i) c.push_back(2 * i);
    c.push_back(0);
    return c;
}
// re-encode the scale bytes of an NVFP4 matrix / expert stack in place (bench weights are synthetic UE4M3 bytes from fix_nv)
static void reencode_rows(char* base, size_t rows, size_t rs, int k) { nvfp4r_repack_rows((uint8_t*)base, rs, (int64_t)rows, k / 64); }

// ------------------------------------------------------------------ epilogue A/B: per-accumulator hsum16 vs batched 4x4 reduction, same kernel
static void bench_epi() {
    const int cpu = default_cpus()[0];
    std::printf("== epilogue A/B (g_gemm_epilogue 0 vs 1), single thread CPU %d, interleaved, load %s\n", cpu, tu::loadavg().c_str());
    ThreadPool pool(1, {cpu});
    struct Cfg { int fmt, k, rows, nt; };
    const Cfg cfgs[] = {{0, 2304, 192, 64}, {0, 2304, 192, 256}, {0, 3072, 192, 64}, {1, 2304, 192, 16}, {1, 2304, 192, 64}, {1, 1152, 192, 16}, {1, 1152, 192, 64}, {1, 2304, 2304, 16}};
    for (const Cfg& c : cfgs) {
        const size_t rs = c.fmt == 0 ? row_bytes_q8_0(c.k) : row_bytes_nvfp4(c.k);
        char* w = (char*)alloc_weight_buffer(rs * c.rows, true);
        fill_random(w, rs * c.rows);
        if (c.fmt == 0) fix_q8(w, c.rows, rs, c.k); else { fix_nv(w, c.rows, rs, c.k); reencode_rows(w, c.rows, rs, c.k); }
        ActBuf ab(c.fmt, c.nt, c.k);
        std::vector<float> y0((size_t)c.nt * c.rows), y1((size_t)c.nt * c.rows);
        auto run = [&](int epi, float* y) {
            g_gemm_epilogue = epi;
            pool.run([&](int, int) {
                if (c.fmt == 0) gemm_q8_0(w, rs, ab.a, 0, c.nt, c.k, y, c.rows, c.rows);
                else gemm_nvfp4r(w, rs, ab.a, 0, c.nt, c.k, y, c.rows, c.rows);
            });
        };
        run(0, y0.data()); run(1, y1.data());
        if (std::memcmp(y0.data(), y1.data(), y0.size() * 4) != 0) std::printf("!! MISMATCH epilogue 0 vs 1 (fmt=%d K=%d nt=%d)\n", c.fmt, c.k, c.nt);
        const int reps = std::max(3, opt.reps * (c.rows > 1000 ? 1 : 3));
        double b0 = 1e30, b1 = 1e30;
        std::vector<double> ratios;
        for (int i = 0; i < reps; ++i) {
            double t0 = now_s(); run(0, y0.data()); const double ta = now_s() - t0; b0 = std::min(b0, ta);
            t0 = now_s(); run(1, y1.data()); const double tb = now_s() - t0; b1 = std::min(b1, tb);
            ratios.push_back(ta / tb);
        }
        std::sort(ratios.begin(), ratios.end());
        g_gemm_epilogue = 1;
        const double gm = (double)c.rows * c.nt * c.k * 1e-9;
        std::printf("%s %4dx%d nt=%3d  epi0 %8.3f ms %5.1f GMAC/s | epi1 %8.3f ms %5.1f GMAC/s | speed %.3fx (median %.3fx)  load %s\n", c.fmt == 0 ? "Q8_0 " : "NVFP4r", c.rows, c.k, c.nt,
                    b0 * 1e3, gm / b0, b1 * 1e3, gm / b1, b0 / b1, ratios[ratios.size() / 2], tu::loadavg().c_str());
        std::fflush(stdout);
        free_weight_buffer(w);
    }
}

// ------------------------------------------------------------------ x4 prototype A/B: gemm_nvfp4r (exact, 5 uops) vs gemm_nvfp4r_x4 (4 uops)
static void bench_x4() {
    const int cpu = default_cpus()[0];
    std::printf("== x4 prototype A/B (gemm_nvfp4r exact vs gemm_nvfp4r_x4), single thread CPU %d, interleaved, load %s\n", cpu, tu::loadavg().c_str());
    ThreadPool pool(1, {cpu});
    struct Cfg { int k, rows, nt; };
    const Cfg cfgs[] = {{2304, 192, 4}, {2304, 192, 16}, {2304, 192, 64}, {2304, 192, 256}, {1152, 192, 16}, {1152, 192, 64}, {2304, 2304, 16}, {2304, 2304, 64}, {1152, 2304, 16}};
    for (const Cfg& c : cfgs) {
        const size_t rs = row_bytes_nvfp4(c.k);
        char* w = (char*)alloc_weight_buffer(rs * c.rows, true);
        fill_random(w, rs * c.rows);
        fix_nv(w, c.rows, rs, c.k);
        reencode_rows(w, c.rows, rs, c.k);
        const std::vector<block_q8_0> q = make_acts(c.nt, c.k);
        ActBuf a1(1, c.nt, c.k, &q), a2(2, c.nt, c.k, &q);  // same quants, both pack formats
        std::vector<float> y0((size_t)c.nt * c.rows), y1((size_t)c.nt * c.rows);
        auto run0 = [&]() { pool.run([&](int, int) { gemm_nvfp4r(w, rs, a1.a, 0, c.nt, c.k, y0.data(), c.rows, c.rows); }); };
        auto run1 = [&]() { pool.run([&](int, int) { gemm_nvfp4r_x4(w, rs, a2.a, 0, c.nt, c.k, y1.data(), c.rows, c.rows); }); };
        run0(); run1();
        double md = 0, ma = 0;
        for (size_t i = 0; i < y0.size(); ++i) { md = std::max(md, (double)std::fabs(y0[i] - y1[i])); ma = std::max(ma, (double)std::fabs(y0[i])); }
        const int reps = std::max(3, opt.reps * (c.rows > 1000 ? 1 : 3));
        double b0 = 1e30, b1 = 1e30;
        std::vector<double> ratios;
        for (int i = 0; i < reps; ++i) {
            double t0 = now_s(); run0(); const double ta = now_s() - t0; b0 = std::min(b0, ta);
            t0 = now_s(); run1(); const double tb = now_s() - t0; b1 = std::min(b1, tb);
            ratios.push_back(ta / tb);
        }
        std::sort(ratios.begin(), ratios.end());
        const double gm = (double)c.rows * c.nt * c.k * 1e-9;
        std::printf("NVFP4r %4dx%d nt=%3d  exact %8.3f ms %5.1f GMAC/s | x4 %8.3f ms %5.1f GMAC/s | x4/exact speed %.3fx (median %.3fx) | max|dy| %.2g of max|y| %.3g  load %s\n",
                    c.rows, c.k, c.nt, b0 * 1e3, gm / b0, b1 * 1e3, gm / b1, b0 / b1, ratios[ratios.size() / 2], md, ma, tu::loadavg().c_str());
        std::fflush(stdout);
        free_weight_buffer(w);
    }
}


// ------------------------------------------------------------------ P2: i16 kernel A/B (gemm_nvfp4r exact vs gemm_nvfp4r_i16), single thread, interleaved
static void bench_i16() {
    const int cpu = default_cpus()[0];
    std::printf("== i16 A/B (gemm_nvfp4r exact vs gemm_nvfp4r_i16), single thread CPU %d, interleaved, tile %d seg %d pf %d, load %s\n", cpu, g_gemm_i16_tile,
                g_gemm_i16_seg, g_gemm_i16_pf, tu::loadavg().c_str());
    ThreadPool pool(1, {cpu});
    struct Cfg { int k, rows, nt; };
    const Cfg cfgs[] = {{2304, 192, 1}, {2304, 192, 2}, {2304, 192, 4}, {2304, 192, 9}, {2304, 192, 16}, {2304, 192, 64}, {2304, 192, 256},
                        {1152, 192, 4}, {1152, 192, 9}, {1152, 192, 16}, {1152, 192, 64}, {2304, 2304, 9}, {2304, 2304, 16}, {2304, 2304, 64}, {1152, 2304, 16}};
    for (const Cfg& c : cfgs) {
        const size_t rs = row_bytes_nvfp4(c.k);
        char* w = (char*)alloc_weight_buffer(rs * c.rows, true);
        fill_random(w, rs * c.rows);
        fix_nv(w, c.rows, rs, c.k);
        reencode_rows(w, c.rows, rs, c.k);
        const std::vector<block_q8_0> q = make_acts(c.nt, c.k);
        ActBuf a1(GEMM_FMT_I16, c.nt, c.k, &q);  // the I16 pack serves both kernels
        std::vector<float> y0((size_t)c.nt * c.rows), y1((size_t)c.nt * c.rows);
        auto run0 = [&]() { pool.run([&](int, int) { gemm_nvfp4r(w, rs, a1.a, 0, c.nt, c.k, y0.data(), c.rows, c.rows); }); };
        auto run1 = [&]() { pool.run([&](int, int) { gemm_nvfp4r_i16(w, rs, a1.a, 0, c.nt, c.k, y1.data(), c.rows, c.rows); }); };
        run0(); run1();
        double md = 0, ma = 0;
        for (size_t i = 0; i < y0.size(); ++i) { md = std::max(md, (double)std::fabs(y0[i] - y1[i])); ma = std::max(ma, (double)std::fabs(y0[i])); }
        const int reps = std::max(3, opt.reps * (c.rows > 1000 ? 1 : 3));
        double b0 = 1e30, b1 = 1e30;
        std::vector<double> ratios;
        for (int i = 0; i < reps; ++i) {
            double t0 = now_s(); run0(); const double ta = now_s() - t0; b0 = std::min(b0, ta);
            t0 = now_s(); run1(); const double tb = now_s() - t0; b1 = std::min(b1, tb);
            ratios.push_back(ta / tb);
        }
        std::sort(ratios.begin(), ratios.end());
        const double gm = (double)c.rows * c.nt * c.k * 1e-9;
        std::printf("NVFP4r %4dx%d nt=%3d  exact %8.3f ms %5.1f GMAC/s | i16 %8.3f ms %5.1f GMAC/s | i16/exact speed %.3fx (median %.3fx) | max|dy| %.2g of max|y| %.3g  load %s\n",
                    c.rows, c.k, c.nt, b0 * 1e3, gm / b0, b1 * 1e3, gm / b1, b0 / b1, ratios[ratios.size() / 2], md, ma, tu::loadavg().c_str());
        std::fflush(stdout);
        free_weight_buffer(w);
    }
}


// ------------------------------------------------------------------ P2: unpack fast path A/B (pointer-table unpack vs uniform-stride unpack), same i16 kernel
static void bench_i16u() {
    const int cpu = default_cpus()[0];
    std::printf("== i16 unpack A/B (g_gemm_i16_uniform 0 vs 1), single thread CPU %d, interleaved, tile %d seg %d, load %s\n", cpu, g_gemm_i16_tile, g_gemm_i16_seg,
                tu::loadavg().c_str());
    ThreadPool pool(1, {cpu});
    struct Cfg { int k, rows, nt; };
    const Cfg cfgs[] = {{2304, 192, 4}, {2304, 192, 9}, {2304, 192, 16}, {2304, 192, 64}, {1152, 192, 9}, {1152, 192, 16}, {2304, 2304, 9}, {2304, 2304, 16}};
    for (const Cfg& c : cfgs) {
        const size_t rs = row_bytes_nvfp4(c.k);
        char* w = (char*)alloc_weight_buffer(rs * c.rows, true);
        fill_random(w, rs * c.rows);
        fix_nv(w, c.rows, rs, c.k);
        reencode_rows(w, c.rows, rs, c.k);
        const std::vector<block_q8_0> q = make_acts(c.nt, c.k);
        ActBuf a1(GEMM_FMT_I16, c.nt, c.k, &q);
        std::vector<float> y0((size_t)c.nt * c.rows), y1((size_t)c.nt * c.rows);
        auto run0 = [&]() { g_gemm_i16_uniform = 0; pool.run([&](int, int) { gemm_nvfp4r_i16(w, rs, a1.a, 0, c.nt, c.k, y0.data(), c.rows, c.rows); }); };
        auto run1 = [&]() { g_gemm_i16_uniform = 1; pool.run([&](int, int) { gemm_nvfp4r_i16(w, rs, a1.a, 0, c.nt, c.k, y1.data(), c.rows, c.rows); }); };
        run0(); run1();
        const bool same = std::memcmp(y0.data(), y1.data(), y0.size() * 4) == 0;
        const int reps = std::max(3, opt.reps * (c.rows > 1000 ? 1 : 3));
        double b0 = 1e30, b1 = 1e30;
        std::vector<double> ratios;
        for (int i = 0; i < reps; ++i) {
            double t0 = now_s(); run0(); const double ta = now_s() - t0; b0 = std::min(b0, ta);
            t0 = now_s(); run1(); const double tb = now_s() - t0; b1 = std::min(b1, tb);
            ratios.push_back(ta / tb);
        }
        std::sort(ratios.begin(), ratios.end());
        const double gm = (double)c.rows * c.nt * c.k * 1e-9;
        std::printf("i16 %4dx%d nt=%3d  table-unpack %8.3f ms %5.1f GMAC/s | uniform-unpack %8.3f ms %5.1f GMAC/s | speed %.3fx (median %.3fx) | bits %s  load %s\n",
                    c.rows, c.k, c.nt, b0 * 1e3, gm / b0, b1 * 1e3, gm / b1, b0 / b1, ratios[ratios.size() / 2], same ? "identical" : "DIFFERENT", tu::loadavg().c_str());
        std::fflush(stdout);
        free_weight_buffer(w);
    }
    g_gemm_i16_uniform = 1;
}

// ------------------------------------------------------------------ direct unpack timing (gguf / nvfp4r / x4), L1- and L2-resident rows
static void bench_unpack() {
    const int cpu = default_cpus()[0];
    std::printf("== direct unpack timing (gemm_bench_unpack_ns), single thread CPU %d, load %s\n", cpu, tu::loadavg().c_str());
    ThreadPool pool(1, {cpu});
    struct Cfg { int k, rows; const char* note; };
    const Cfg cfgs[] = {{2304, 16, "L1"}, {1152, 16, "L1"}, {2304, 192, "L2"}, {1152, 192, "L2"}, {2304, 2304, "L2/L3 8 MB"}};
    for (const Cfg& c : cfgs) {
        const size_t rs = row_bytes_nvfp4(c.k);
        char* w = (char*)alloc_weight_buffer(rs * c.rows, true);
        char* wr = (char*)alloc_weight_buffer(rs * c.rows, true);
        fill_random(w, rs * c.rows);
        fix_nv(w, c.rows, rs, c.k);
        std::memcpy(wr, w, rs * c.rows);
        reencode_rows(wr, c.rows, rs, c.k);
        const int reps = std::max(2, 4000 / c.rows);
        double best[3] = {1e30, 1e30, 1e30};
        for (int it = 0; it < opt.reps; ++it)
            for (int enc = 0; enc < 3; ++enc) {
                double ns = 0;
                pool.run([&](int, int) { ns = gemm_bench_unpack_ns(enc == 0 ? w : wr, rs, c.k, c.rows, enc, reps); });
                best[enc] = std::min(best[enc], ns);
            }
        const double per = 1.0 / ((double)reps * c.rows * (c.k / 64));
        std::printf("K=%d rows=%d (%s): ns per (row, super-block): gguf %.2f | nvfp4r %.2f (%.2fx faster) | x4 %.2f (%.2fx)  load %s\n", c.k, c.rows, c.note, best[0] * per,
                    best[1] * per, best[0] / best[1], best[2] * per, best[0] / best[2], tu::loadavg().c_str());
        free_weight_buffer(w);
        free_weight_buffer(wr);
    }
}

// ------------------------------------------------------------------ tail5 A/B on MoE-like token counts (same kernel, flag 0 vs 1)
static void bench_tail() {
    const int cpu = default_cpus()[0];
    std::printf("== tail5 A/B (g_gemm_tail5 0 vs 1), single thread CPU %d, interleaved, load %s\n", cpu, tu::loadavg().c_str());
    ThreadPool pool(1, {cpu});
    struct Cfg { int fmt, k, nt; };
    const Cfg cfgs[] = {{1, 2304, 9}, {1, 2304, 13}, {1, 2304, 17}, {1, 2304, 21}, {1, 1152, 9}, {1, 1152, 13}, {1, 1152, 17}, {0, 2304, 9}, {1, 2304, 8}};
    for (const Cfg& c : cfgs) {
        const int rows = 192;
        const size_t rs = c.fmt == 0 ? row_bytes_q8_0(c.k) : row_bytes_nvfp4(c.k);
        char* w = (char*)alloc_weight_buffer(rs * rows, true);
        fill_random(w, rs * rows);
        if (c.fmt == 0) fix_q8(w, rows, rs, c.k); else { fix_nv(w, rows, rs, c.k); reencode_rows(w, rows, rs, c.k); }
        ActBuf ab(c.fmt, c.nt, c.k);
        std::vector<float> y0((size_t)c.nt * rows), y1((size_t)c.nt * rows);
        auto run = [&](int tail5, float* y) {
            g_gemm_tail5 = tail5;
            pool.run([&](int, int) {
                if (c.fmt == 0) gemm_q8_0(w, rs, ab.a, 0, c.nt, c.k, y, rows, rows);
                else gemm_nvfp4r(w, rs, ab.a, 0, c.nt, c.k, y, rows, rows);
            });
        };
        run(0, y0.data()); run(1, y1.data());
        if (std::memcmp(y0.data(), y1.data(), y0.size() * 4) != 0) std::printf("!! MISMATCH tail5 0 vs 1 (fmt=%d K=%d nt=%d)\n", c.fmt, c.k, c.nt);
        double b0 = 1e30, b1 = 1e30;
        std::vector<double> ratios;
        for (int i = 0; i < opt.reps * 3; ++i) {
            double t0 = now_s(); run(0, y0.data()); const double ta = now_s() - t0; b0 = std::min(b0, ta);
            t0 = now_s(); run(1, y1.data()); const double tb = now_s() - t0; b1 = std::min(b1, tb);
            ratios.push_back(ta / tb);
        }
        std::sort(ratios.begin(), ratios.end());
        g_gemm_tail5 = 1;
        const double gm = (double)rows * c.nt * c.k * 1e-9;
        std::printf("%s 192x%d nt=%2d  tail5=0 %7.3f ms %5.1f GMAC/s | tail5=1 %7.3f ms %5.1f GMAC/s | speed %.3fx (median %.3fx)  load %s\n", c.fmt == 0 ? "Q8_0  " : "NVFP4r", c.k, c.nt,
                    b0 * 1e3, gm / b0, b1 * 1e3, gm / b1, b0 / b1, ratios[ratios.size() / 2], tu::loadavg().c_str());
        std::fflush(stdout);
        free_weight_buffer(w);
    }
}

// ------------------------------------------------------------------ GGUF vs re-encoded scales, single thread, interleaved A/B
static void bench_rvs() {
    const int cpu = default_cpus()[0];
    std::printf("== gemm_nvfp4 (GGUF scales) vs gemm_nvfp4r (re-encoded), single thread CPU %d, interleaved, load %s\n", cpu, tu::loadavg().c_str());
    ThreadPool pool(1, {cpu});
    struct Cfg {
        int k, rows, nt;
        const char* note;
    };
    // 192 rows: slice L2-resident (G's st setting); 2304 rows x 2304: one expert matrix (8.3 MB) streamed from L3/DRAM like the MoE kernel
    const Cfg cfgs[] = {{2304, 192, 4, "L2"},  {2304, 192, 16, "L2"},  {2304, 192, 64, "L2"},  {2304, 192, 256, "L2"},
                        {1152, 192, 4, "L2"},  {1152, 192, 16, "L2"},  {1152, 192, 64, "L2"},
                        {1152, 16, 4, "L1"},   {1152, 16, 64, "L1"},   {2304, 16, 4, "L1"},    {2304, 16, 64, "L1"},
                        {2304, 2304, 4, "mem"}, {2304, 2304, 16, "mem"}, {2304, 2304, 64, "mem"}, {1152, 2304, 16, "mem"}};
    // [k index][rows index: 0 = 192 (L2), 1 = 16 (L1)][enc] best seconds per call for the per-panel derivation T(nt) = U + nt*C
    double t4[2][2][2] = {}, t64[2][2][2] = {};
    for (const Cfg& c : cfgs) {
        const size_t rs = row_bytes_nvfp4(c.k);
        char* w = (char*)alloc_weight_buffer(rs * c.rows, true);
        char* wr = (char*)alloc_weight_buffer(rs * c.rows, true);
        fill_random(w, rs * c.rows);
        fix_nv(w, c.rows, rs, c.k);
        std::memcpy(wr, w, rs * c.rows);
        reencode_rows(wr, c.rows, rs, c.k);
        ActBuf ab(1, c.nt, c.k);
        std::vector<float> y((size_t)c.nt * c.rows), yr((size_t)c.nt * c.rows);
        auto run_g = [&]() { pool.run([&](int, int) { gemm_nvfp4(w, rs, ab.a, 0, c.nt, c.k, y.data(), c.rows, c.rows); }); };
        auto run_r = [&]() { pool.run([&](int, int) { gemm_nvfp4r(wr, rs, ab.a, 0, c.nt, c.k, yr.data(), c.rows, c.rows); }); };
        run_g();
        run_r();
        if (std::memcmp(y.data(), yr.data(), y.size() * 4) != 0) std::printf("!! MISMATCH gemm_nvfp4 vs gemm_nvfp4r (K=%d rows=%d nt=%d)\n", c.k, c.rows, c.nt);
        const int reps = std::max(3, opt.reps * (c.rows > 1000 ? 1 : 3));
        double bg = 1e30, br = 1e30;
        std::vector<double> ratios;
        for (int i = 0; i < reps; ++i) {  // interleaved: same window for both; per-rep ratio is robust against bursts
            double t0 = now_s(); run_g(); const double tg = now_s() - t0; bg = std::min(bg, tg);
            t0 = now_s(); run_r(); const double tr = now_s() - t0; br = std::min(br, tr);
            ratios.push_back(tg / tr);
        }
        std::sort(ratios.begin(), ratios.end());
        const double gm = (double)c.rows * c.nt * c.k * 1e-9;
        std::printf("NVFP4 %4dx%d nt=%3d (%s)  gguf %8.3f ms %5.1f GMAC/s | r %8.3f ms %5.1f GMAC/s | r/gguf speed %.3fx (median of per-rep %.3fx)  load %s\n", c.rows, c.k, c.nt, c.note,
                    bg * 1e3, gm / bg, br * 1e3, gm / br, bg / br, ratios[ratios.size() / 2], tu::loadavg().c_str());
        std::fflush(stdout);
        if (c.rows == 192 || c.rows == 16) {
            const int ki = c.k == 2304 ? 0 : 1, ri = c.rows == 192 ? 0 : 1;
            if (c.nt == 4) { t4[ki][ri][0] = bg; t4[ki][ri][1] = br; }
            if (c.nt == 64) { t64[ki][ri][0] = bg; t64[ki][ri][1] = br; }
        }
        free_weight_buffer(w);
        free_weight_buffer(wr);
    }
    for (int ri = 0; ri < 2; ++ri)
        for (int ki = 0; ki < 2; ++ki) {
            const int k = ki == 0 ? 2304 : 1152, nc = k / 64, rows = ri == 0 ? 192 : 16, panels = rows / 4;
            for (int enc = 0; enc < 2; ++enc) {
                if (t4[ki][ri][enc] <= 0 || t64[ki][ri][enc] <= 0) continue;
                const double C = (t64[ki][ri][enc] - t4[ki][ri][enc]) / 60.0, U = t4[ki][ri][enc] - 4.0 * C;  // per call: U = fixed (unpack) part, C = per token
                std::printf("K=%d rows=%d (%s) %s: fixed part U %.1f us per call = %.1f ns per 4-row panel = %.2f ns per (row, super-block); per-token part C %.1f us (%.2f ns per (row, sb, token))\n",
                            k, rows, ri ? "L1" : "L2", enc ? "nvfp4r" : "nvfp4 ", U * 1e6, U / panels * 1e9, U / panels / (4.0 * nc) * 1e9, C * 1e6, C / ((double)rows * nc) * 1e9);
            }
        }
}

// ------------------------------------------------------------------ single thread, weights resident in L2
static void bench_st() {
    std::printf("== single thread (CPU 2), weight slice resident (192 rows), load %s\n", tu::loadavg().c_str());
    ThreadPool pool(1, {2});
    struct Cfg {
        int fmt, k, nt;
    };
    const Cfg cfgs[] = {{0, 2304, 8}, {0, 2304, 64}, {0, 2304, 256}, {0, 3072, 64}, {1, 2304, 4}, {1, 2304, 16}, {1, 2304, 64}, {1, 1152, 16}, {1, 1152, 64}};
    for (const Cfg& c : cfgs) {
        const int rows = 192;
        const size_t rs = c.fmt == 0 ? row_bytes_q8_0(c.k) : row_bytes_nvfp4(c.k);
        char* w = (char*)alloc_weight_buffer(rs * rows, true);
        fill_random(w, rs * rows);
        if (c.fmt == 0) fix_q8(w, rows, rs, c.k); else fix_nv(w, rows, rs, c.k);
        ActBuf ab(c.fmt, c.nt, c.k);
        std::vector<float> y((size_t)c.nt * rows);
        const double gm = (double)rows * c.nt * c.k * 1e-9;
        char name[128];
        auto run = [&]() {
            pool.run([&](int, int) {
                if (c.fmt == 0) gemm_q8_0(w, rs, ab.a, 0, c.nt, c.k, y.data(), rows, rows);
                else gemm_nvfp4(w, rs, ab.a, 0, c.nt, c.k, y.data(), rows, rows);
            });
        };
        double pk = peak_gmacs();
        auto r = time_best(run, opt.reps * 3);
        pk = std::max(pk, peak_gmacs());
        std::snprintf(name, sizeof name, "%s 192x%d nt=%d (L2-resident W)", c.fmt == 0 ? "Q8_0 " : "NVFP4", c.k, c.nt);
        report(name, gm, r.first, r.second, 1, pk);
        // B1 gemv for the same work (per token), as a reference point
        const int ntg = std::min(c.nt, 8);
        std::vector<block_q8_0> q = make_acts(ntg, c.k);
        auto g = time_best(
            [&]() {
                pool.run([&](int, int) {
                    for (int t = 0; t < ntg; ++t) {
                        if (c.fmt == 0) gemv_q8_0(w, rs, q.data() + (size_t)t * (c.k / 32), c.k, y.data() + (size_t)t * rows, rows);
                        else gemv_nvfp4(w, rs, q.data() + (size_t)t * (c.k / 32), c.k, y.data() + (size_t)t * rows, rows);
                    }
                });
            },
            opt.reps * 3);
        std::snprintf(name, sizeof name, "   B1 gemv x%d tokens (same matrix)", ntg);
        report(name, (double)rows * ntg * c.k * 1e-9, g.first, g.second, 1);
        free_weight_buffer(w);
    }
}

// ------------------------------------------------------------------ multi-thread dense
static void bench_dense_mt(ThreadPool& pool, int fmt, int rows, int k, int nt, int chunk) {
    const size_t rs = fmt == 0 ? row_bytes_q8_0(k) : row_bytes_nvfp4(k);
    char* w = (char*)alloc_weight_buffer(rs * rows, true);
    fill_random(w, rs * rows);
    if (fmt == 0) fix_q8(w, rows, rs, k); else fix_nv(w, rows, rs, k);
    ActBuf ab(fmt, nt, k);
    std::vector<float> y((size_t)nt * rows);
    auto run = [&]() {
        pool.run_chunks(rows, chunk, [&](int, int64_t b, int64_t e) {
            if (fmt == 0) gemm_q8_0(w + (size_t)b * rs, rs, ab.a, 0, nt, k, y.data() + b, rows, (int)(e - b));
            else gemm_nvfp4(w + (size_t)b * rs, rs, ab.a, 0, nt, k, y.data() + b, rows, (int)(e - b));
        });
    };
    g_peak_mt = peak_mt(pool);
    auto r = time_best(run, opt.reps);
    g_peak_mt = std::max(g_peak_mt, peak_mt(pool));
    char name[128];
    std::snprintf(name, sizeof name, "%s %dx%d nt=%d chunk=%d", fmt == 0 ? "Q8_0 " : "NVFP4", rows, k, nt, chunk);
    report(name, (double)rows * nt * k * 1e-9, r.first, r.second, pool.size(), g_peak_mt);
    free_weight_buffer(w);
}
static void bench_dense(ThreadPool& pool) {
    std::printf("== dense, %d threads\n", pool.size());
    for (int nt : {64, 256}) {
        bench_dense_mt(pool, 0, 3072, 2304, nt, opt.chunk);   // wq / attn_gate
        bench_dense_mt(pool, 0, 2304, 3072, nt, opt.chunk);   // wo
    }
    bench_dense_mt(pool, 0, 1024, 2304, 256, 32);              // wk / wv
}
static void bench_nvfp4(ThreadPool& pool) {
    std::printf("== NVFP4 single matrix (one expert), %d threads\n", pool.size());
    for (int nt : {16, 64}) {
        bench_dense_mt(pool, 1, 2304, 2304, nt, 32);
        bench_dense_mt(pool, 1, 2304, 1152, nt, 32);
    }
}

// ------------------------------------------------------------------ MoE with real routing
struct TraceRec {
    int fwd, visit, layer, ntok, topk;
    std::vector<int32_t> ids;
};
static bool load_trace(const char* path, std::vector<TraceRec>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    for (;;) {
        int32_t h[8];
        if (std::fread(h, 4, 8, f) != 8 || h[0] != 0x52545243) break;
        TraceRec r{h[1], h[2], h[3], h[4], h[5], {}};
        r.ids.resize((size_t)r.ntok * r.topk);
        if (std::fread(r.ids.data(), 4, r.ids.size(), f) != r.ids.size()) break;
        out.push_back(std::move(r));
    }
    std::fclose(f);
    return !out.empty();
}

static void bench_moe(ThreadPool& pool) {
    std::vector<TraceRec> tr;
    if (!load_trace(opt.trace.c_str(), tr)) {
        std::printf("== MoE: trace %s not readable, skipped\n", opt.trace.c_str());
        return;
    }
    const TraceRec* rec = nullptr;
    for (const TraceRec& r : tr)
        if (r.visit == opt.visit) { rec = &r; break; }
    if (!rec) rec = &tr[0];
    const int E = 128, N = 2304, Kg = 2304, Kd = 1152;
    std::printf("== MoE grouped NVFP4, %d threads, trace %s fwd %d visit %d layer %d (%d tokens x %d)\n", pool.size(), opt.trace.c_str(), rec->fwd,
                rec->visit, rec->layer, rec->ntok, rec->topk);
    const size_t rsg = row_bytes_nvfp4(Kg), rsd = row_bytes_nvfp4(Kd);
    const size_t esg = rsg * N, esd = rsd * N;
    char* wg = (char*)alloc_weight_buffer(esg * E, true);
    char* wd = (char*)alloc_weight_buffer(esd * E, true);
    fill_random(wg, esg * E);
    fill_random(wd, esd * E);
    fix_nv(wg, (size_t)N * E, rsg, Kg);
    fix_nv(wd, (size_t)N * E, rsd, Kd);
    // re-encoded copies for gemm_moe_nvfp4r (same bytes except the 4 scale bytes per super-block)
    char* wgr = (char*)alloc_weight_buffer(esg * E, true);
    char* wdr = (char*)alloc_weight_buffer(esd * E, true);
    std::memcpy(wgr, wg, esg * E);
    std::memcpy(wdr, wd, esd * E);
    reencode_rows(wgr, (size_t)N * E, rsg, Kg);
    reencode_rows(wdr, (size_t)N * E, rsd, Kd);
    for (int nt : opt.ntok) {
        if (nt > rec->ntok) continue;
        MoeRoute r;
        moe_build_route(rec->ids.data(), nt, rec->topk, E, r);
        const int np = r.n_pairs();
        int maxn = 0, zero = 0, small = 0;
        for (int e = 0; e < E; ++e) {
            const int n = r.offset[e + 1] - r.offset[e];
            maxn = std::max(maxn, n);
            zero += n == 0;
            small += (n > 0 && n <= 2);
        }
        std::vector<MoeItem> items;
        moe_work_items(r, N, opt.chunk, items);
        ActBuf at(1, nt, Kg), ap(1, np, Kd);
        ActBuf at4(2, nt, Kg), ap4(2, np, Kd);  // x4 prototype packs
        ActBuf at16(GEMM_FMT_I16, nt, Kg), ap16(GEMM_FMT_I16, np, Kd);  // P2 i16 packs
        // pre-gathered variant: activation of pair p = row pair_token[p] of the token pack, copied into pair order
        std::vector<block_q8_0> qg = make_acts(nt, Kg);
        std::vector<block_q8_0> qgp((size_t)np * (Kg / 32));
        for (int p = 0; p < np; ++p) std::memcpy(&qgp[(size_t)p * (Kg / 32)], &qg[(size_t)r.pair_token[p] * (Kg / 32)], (Kg / 32) * sizeof(block_q8_0));
        void* gbuf = std::aligned_alloc(64, (gemm_act_bytes(1, np, Kg) + 63) & ~(size_t)63);
        gemm_pack_act(1, qgp.data(), 0, np, np, Kg, gbuf);
        const GemmAct agat = gemm_act_view(1, gbuf, np, Kg);
        std::vector<float> yg((size_t)np * N), yd((size_t)np * N);
        std::printf("-- chunk=%d tokens: %d pairs, %zu items, max pairs/expert %d, experts unused %d, experts with 1-2 pairs %d, mean %.1f per used expert\n", nt, np,
                    items.size(), maxn, zero, small, (double)np / std::max(1, E - zero));
        char name[160];
        auto run_gu = [&](const GemmAct& a, const int32_t* rows_idx, int kind) {  // kind 0 gguf, 1 nvfp4r, 2 x4 prototype
            return [&, rows_idx, kind]() {
                pool.run_chunks((int64_t)items.size(), 1, [&](int, int64_t b, int64_t e) {
                    for (int64_t i = b; i < e; ++i) {
                        if (kind == 3) gemm_moe_nvfp4r_i16(wgr, esg, rsg, at16.a, rows_idx, r, items[(size_t)i], Kg, yg.data(), N);
                        else if (kind == 2) gemm_moe_nvfp4r_x4(wgr, esg, rsg, at4.a, rows_idx, r, items[(size_t)i], Kg, yg.data(), N);
                        else if (kind == 1) gemm_moe_nvfp4r(wgr, esg, rsg, a, rows_idx, r, items[(size_t)i], Kg, yg.data(), N);
                        else gemm_moe_nvfp4(wg, esg, rsg, a, rows_idx, r, items[(size_t)i], Kg, yg.data(), N);
                    }
                });
            };
        };
        auto run_dn = [&](int kind) {
            return [&, kind]() {
                pool.run_chunks((int64_t)items.size(), 1, [&](int, int64_t b, int64_t e) {
                    for (int64_t i = b; i < e; ++i) {
                        if (kind == 3) gemm_moe_nvfp4r_i16(wdr, esd, rsd, ap16.a, nullptr, r, items[(size_t)i], Kd, yd.data(), N);
                        else if (kind == 2) gemm_moe_nvfp4r_x4(wdr, esd, rsd, ap4.a, nullptr, r, items[(size_t)i], Kd, yd.data(), N);
                        else if (kind == 1) gemm_moe_nvfp4r(wdr, esd, rsd, ap.a, nullptr, r, items[(size_t)i], Kd, yd.data(), N);
                        else gemm_moe_nvfp4(wd, esd, rsd, ap.a, nullptr, r, items[(size_t)i], Kd, yd.data(), N);
                    }
                });
            };
        };
        const double gm_gu = (double)np * N * Kg * 1e-9, gm_dn = (double)np * N * Kd * 1e-9;
        g_peak_mt = peak_mt(pool);
        // interleaved A/B: gguf and re-encoded alternate rep by rep (same window)
        auto ab_time = [&](auto&& fg, auto&& fr) {
            fg(); fr();
            double bg = 1e30, br = 1e30;
            for (int i = 0; i < opt.reps; ++i) {
                double t0 = now_s(); fg(); bg = std::min(bg, now_s() - t0);
                t0 = now_s(); fr(); br = std::min(br, now_s() - t0);
            }
            return std::pair<double, double>(bg, br);
        };
        auto a = ab_time(run_gu(at.a, r.pair_token.data(), 0), run_gu(at.a, r.pair_token.data(), 1));
        std::snprintf(name, sizeof name, "moe gate_up %dx%d, indirect, gguf", N, Kg);
        report(name, gm_gu, a.first, a.first, pool.size(), g_peak_mt);
        std::snprintf(name, sizeof name, "moe gate_up %dx%d, indirect, nvfp4r", N, Kg);
        report(name, gm_gu, a.second, a.second, pool.size(), g_peak_mt);
        auto a4 = ab_time(run_gu(at.a, r.pair_token.data(), 1), run_gu(at.a, r.pair_token.data(), 2));
        std::snprintf(name, sizeof name, "moe gate_up %dx%d, indirect, x4 PROTOTYPE", N, Kg);
        report(name, gm_gu, a4.second, a4.second, pool.size(), g_peak_mt);
        auto a16 = ab_time(run_gu(at.a, r.pair_token.data(), 1), run_gu(at.a, r.pair_token.data(), 3));
        std::snprintf(name, sizeof name, "moe gate_up %dx%d, indirect, i16 (P2)", N, Kg);
        report(name, gm_gu, a16.second, a16.second, pool.size(), g_peak_mt);
        auto b = time_best(run_gu(agat, nullptr, 0), opt.reps);
        std::snprintf(name, sizeof name, "moe gate_up %dx%d, pre-gathered, gguf", N, Kg);
        report(name, gm_gu, b.first, b.second, pool.size(), g_peak_mt);
        auto c = ab_time(run_dn(0), run_dn(1));
        auto c4 = ab_time(run_dn(1), run_dn(2));
        std::snprintf(name, sizeof name, "moe down %dx%d (per-pair act), x4 PROTOTYPE", N, Kd);
        report(name, gm_dn, c4.second, c4.second, pool.size(), g_peak_mt);
        auto c16 = ab_time(run_dn(1), run_dn(3));
        std::snprintf(name, sizeof name, "moe down %dx%d (per-pair act), i16 (P2)", N, Kd);
        report(name, gm_dn, c16.second, c16.second, pool.size(), g_peak_mt);
        std::snprintf(name, sizeof name, "moe layer (gate_up + down), i16 (P2)");
        report(name, gm_gu + gm_dn, a16.second + c16.second, a16.second + c16.second, pool.size(), g_peak_mt);
        std::printf("   i16 / nvfp4r (same-pair A/B): gate_up %.3fx, down %.3fx, layer %.3fx  (i16 tile %d seg %d pf %d)\n", a16.first / a16.second, c16.first / c16.second,
                    (a16.first + c16.first) / (a16.second + c16.second), g_gemm_i16_tile, g_gemm_i16_seg, g_gemm_i16_pf);
        // engine mix (phases G/H as wired): experts with <= mx pairs -> gemv_nvfp4r_multix (K2), bigger ones -> nvfp4r (exact) or i16 (P2); per item
        {
            std::vector<block_q8_0> qd = make_acts(np, Kd);
            const int nbg = Kg / 32, nbd = Kd / 32;
            auto run_mix = [&](int mx, bool i16, bool down) {
                return [&, mx, i16, down]() {
                    pool.run_chunks((int64_t)items.size(), 1, [&](int, int64_t b, int64_t e) {
                        for (int64_t i = b; i < e; ++i) {
                            const MoeItem& it = items[(size_t)i];
                            const int p0 = r.offset[it.expert], npe = r.offset[it.expert + 1] - p0, nr = it.row1 - it.row0;
                            if (npe <= mx) {
                                const block_q8_0* xs[kMultixMaxX > 8 ? kMultixMaxX : 8];
                                for (int q = 0; q < npe && q < 8; ++q) xs[q] = down ? &qd[(size_t)(p0 + q) * nbd] : &qg[(size_t)r.pair_token[(size_t)(p0 + q)] * nbg];
                                if (down) gemv_nvfp4r_multix(wdr + (size_t)it.expert * esd + (size_t)it.row0 * rsd, rsd, nr, xs, npe, Kd, yd.data() + (size_t)p0 * N + it.row0, N);
                                else gemv_nvfp4r_multix(wgr + (size_t)it.expert * esg + (size_t)it.row0 * rsg, rsg, nr, xs, npe, Kg, yg.data() + (size_t)p0 * N + it.row0, N);
                            } else if (down) {
                                if (i16) gemm_moe_nvfp4r_i16(wdr, esd, rsd, ap16.a, nullptr, r, it, Kd, yd.data(), N);
                                else gemm_moe_nvfp4r(wdr, esd, rsd, ap.a, nullptr, r, it, Kd, yd.data(), N);
                            } else {
                                if (i16) gemm_moe_nvfp4r_i16(wgr, esg, rsg, at16.a, r.pair_token.data(), r, it, Kg, yg.data(), N);
                                else gemm_moe_nvfp4r(wgr, esg, rsg, at.a, r.pair_token.data(), r, it, Kg, yg.data(), N);
                            }
                        }
                    });
                };
            };
            for (int mx : opt.mx) {
                if (mx > 8) continue;
                auto g = ab_time(run_mix(mx, false, false), run_mix(mx, true, false));
                auto d = ab_time(run_mix(mx, false, true), run_mix(mx, true, true));
                std::snprintf(name, sizeof name, "engine mix mx=%d: multix + nvfp4r (exact)", mx);
                report(name, gm_gu + gm_dn, g.first + d.first, g.first + d.first, pool.size(), g_peak_mt);
                std::snprintf(name, sizeof name, "engine mix mx=%d: multix + i16", mx);
                report(name, gm_gu + gm_dn, g.second + d.second, g.second + d.second, pool.size(), g_peak_mt);
                std::printf("   mx=%d i16-mix / exact-mix: gate_up %.3fx, down %.3fx, layer %.3fx\n", mx, g.first / g.second, d.first / d.second, (g.first + d.first) / (g.second + d.second));
            }
        }
        std::snprintf(name, sizeof name, "moe down %dx%d (per-pair act), gguf", N, Kd);
        report(name, gm_dn, c.first, c.first, pool.size(), g_peak_mt);
        std::snprintf(name, sizeof name, "moe down %dx%d (per-pair act), nvfp4r", N, Kd);
        report(name, gm_dn, c.second, c.second, pool.size(), g_peak_mt);
        std::snprintf(name, sizeof name, "moe layer (gate_up + down), gguf");
        report(name, gm_gu + gm_dn, a.first + c.first, a.first + c.first, pool.size(), g_peak_mt);
        std::snprintf(name, sizeof name, "moe layer (gate_up + down), nvfp4r");
        report(name, gm_gu + gm_dn, a.second + c.second, a.second + c.second, pool.size(), g_peak_mt);
        std::printf("   nvfp4r / gguf speed: gate_up %.3fx, down %.3fx, layer %.3fx;  x4 / nvfp4r (same-pair A/B): gate_up %.3fx, down %.3fx, layer %.3fx\n",
                    a.first / a.second, c.first / c.second, (a.first + c.first) / (a.second + c.second), a4.first / a4.second, c4.first / c4.second,
                    (a4.first + c4.first) / (a4.second + c4.second));
        std::free(gbuf);
    }
    free_weight_buffer(wg);
    free_weight_buffer(wd);
    free_weight_buffer(wgr);
    free_weight_buffer(wdr);
}

// ------------------------------------------------------------------ tunables on the MoE kernel (nvfp4r): group / tile / prefetch arms, interleaved
static void bench_tune(ThreadPool& pool) {
    std::vector<TraceRec> tr;
    if (!load_trace(opt.trace.c_str(), tr)) { std::printf("== tune: trace not readable\n"); return; }
    const TraceRec* rec = &tr[0];
    for (const TraceRec& r : tr) if (r.visit == opt.visit) { rec = &r; break; }
    const int E = 128, N = 2304, Kg = 2304, Kd = 1152;
    const size_t rsg = row_bytes_nvfp4(Kg), rsd = row_bytes_nvfp4(Kd), esg = rsg * N, esd = rsd * N;
    char* wg = (char*)alloc_weight_buffer(esg * E, true);
    char* wd = (char*)alloc_weight_buffer(esd * E, true);
    fill_random(wg, esg * E); fill_random(wd, esd * E);
    fix_nv(wg, (size_t)N * E, rsg, Kg); fix_nv(wd, (size_t)N * E, rsd, Kd);
    reencode_rows(wg, (size_t)N * E, rsg, Kg); reencode_rows(wd, (size_t)N * E, rsd, Kd);
    struct Arm { const char* name; int group, tile, pf, tail5; };
    const Arm arms[] = {{"group64 tile4 pf1 (default)", 64, 4, 1, 1}, {"group32", 32, 4, 1, 1}, {"group128", 128, 4, 1, 1}, {"group256", 256, 4, 1, 1},
                        {"tile5", 64, 5, 1, 1}, {"prefetch off", 64, 4, 0, 1}, {"prefetch T0 (L1)", 64, 4, 2, 1}, {"group128 pf0", 128, 4, 0, 1}, {"tail5 off", 64, 4, 1, 0}};
    const int narm = (int)(sizeof(arms) / sizeof(arms[0]));
    std::printf("== tune (gemm_moe_nvfp4r, gate_up + down, %d threads, row chunk %d), arms interleaved per rep, load %s\n", pool.size(), opt.chunk, tu::loadavg().c_str());
    for (int nt : opt.ntok) {
        if (nt > rec->ntok) continue;
        MoeRoute r;
        moe_build_route(rec->ids.data(), nt, rec->topk, E, r);
        const int np = r.n_pairs();
        std::vector<MoeItem> items;
        moe_work_items(r, N, opt.chunk, items);
        ActBuf at(1, nt, Kg), ap(1, np, Kd);
        std::vector<float> yg((size_t)np * N), yd((size_t)np * N);
        auto run = [&]() {
            pool.run_chunks((int64_t)items.size(), 1, [&](int, int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) gemm_moe_nvfp4r(wg, esg, rsg, at.a, r.pair_token.data(), r, items[(size_t)i], Kg, yg.data(), N);
            });
            pool.run_chunks((int64_t)items.size(), 1, [&](int, int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) gemm_moe_nvfp4r(wd, esd, rsd, ap.a, nullptr, r, items[(size_t)i], Kd, yd.data(), N);
            });
        };
        std::vector<double> best(narm, 1e30);
        std::vector<std::vector<double>> times(narm);
        for (int rep = 0; rep < opt.reps + 1; ++rep)
            for (int a = 0; a < narm; ++a) {
                g_gemm_group = arms[a].group; g_gemm_tile = arms[a].tile; g_gemm_prefetch = arms[a].pf; g_gemm_tail5 = arms[a].tail5;
                const double t0 = now_s();
                run();
                const double t = now_s() - t0;
                if (rep > 0) { best[a] = std::min(best[a], t); times[a].push_back(t); }
            }
        g_gemm_group = opt.group; g_gemm_tile = 4; g_gemm_prefetch = 1; g_gemm_tail5 = 1;
        const double gm = (double)np * N * (Kg + Kd) * 1e-9;
        std::printf("-- %d tokens (%d pairs, mean %.1f per used expert): ", nt, np, (double)np / std::max(1, E - (int)std::count(r.offset.begin(), r.offset.end() - 1, -1)));
        std::printf("\n");
        for (int a = 0; a < narm; ++a) {
            std::vector<double> ratios;
            for (size_t i = 0; i < times[a].size(); ++i) ratios.push_back(times[0][i] / times[a][i]);
            std::sort(ratios.begin(), ratios.end());
            std::printf("   %-28s best %8.1f ms %6.1f GMAC/s  vs default %.3fx (median per-rep %.3fx)  load %s\n", arms[a].name, best[a] * 1e3, gm / best[a],
                        best[0] / best[a], ratios[ratios.size() / 2], tu::loadavg().c_str());
        }
        std::fflush(stdout);
    }
    free_weight_buffer(wg); free_weight_buffer(wd);
}

// ------------------------------------------------------------------ BF16 / F32 small
static void bench_small(ThreadPool& pool) {
    std::printf("== small matrices, nt=256 (1 thread: whole matrix; %d threads: rows split)\n", pool.size());
    struct Cfg {
        bool bf;
        int rows, k;
        const char* nm;
    };
    const Cfg cfgs[] = {{false, 128, 2304, "router F32 128x2304"}, {true, 24, 2304, "sink BF16 24x2304"}, {true, 32, 2304, "taps BF16 32x2304"}, {false, 64, 4608, "ctrl F32 64x4608"}};
    for (const Cfg& c : cfgs) {
        const size_t esz = c.bf ? 2 : 4;
        std::vector<uint8_t> w((size_t)c.rows * c.k * esz);
        fill_random(w.data(), w.size() / 8 * 8);
        if (!c.bf) {  // keep floats sane
            float* f = (float*)w.data();
            for (size_t i = 0; i < (size_t)c.rows * c.k; ++i) f[i] = (float)((int)(xr() % 2001) - 1000) * 1e-3f;
        } else {
            uint16_t* h = (uint16_t*)w.data();
            for (size_t i = 0; i < (size_t)c.rows * c.k; ++i) h[i] = f32_to_bf16((float)((int)(xr() % 2001) - 1000) * 1e-3f);
        }
        const int nt = 256;
        std::vector<float> x((size_t)nt * c.k), y((size_t)nt * c.rows);
        for (auto& v : x) v = (float)((int)(xr() % 2001) - 1000) * 1e-3f;
        const size_t rs = (size_t)c.k * esz;
        const double gm = (double)c.rows * nt * c.k * 1e-9;
        auto one = [&](int tid_unused, int64_t b, int64_t e) {
            (void)tid_unused;
            if (c.bf) gemm_bf16(w.data() + (size_t)b * rs, rs, x.data(), c.k, nt, c.k, y.data() + b, c.rows, (int)(e - b));
            else gemm_f32(w.data() + (size_t)b * rs, rs, x.data(), c.k, nt, c.k, y.data() + b, c.rows, (int)(e - b));
        };
        auto r1 = time_best([&]() { one(0, 0, c.rows); }, opt.reps * 2);
        report((std::string(c.nm) + " 1 thread").c_str(), gm, r1.first, r1.second, 1);
        auto rn = time_best([&]() { pool.run_chunks(c.rows, 2, one); }, opt.reps * 2);
        report((std::string(c.nm) + " pool").c_str(), gm, rn.first, rn.second, pool.size());
    }
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--what") opt.what = next();
        else if (a == "--i16tile") opt.i16tile = std::stoi(next());
        else if (a == "--i16seg") opt.i16seg = std::stoi(next());
        else if (a == "--i16pf") opt.i16pf = std::stoi(next());
        else if (a == "--mx") { opt.mx.clear(); for (const char* q = next().c_str(); *q;) { opt.mx.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q == ',') ++q; } }
        else if (a == "--threads") opt.threads = std::atoi(next().c_str());
        else if (a == "--cpus" || a == "--ntok") {
            std::string l = next();
            size_t p = 0;
            std::vector<int>& dst = a == "--cpus" ? opt.cpus : opt.ntok;
            dst.clear();
            while (p < l.size()) {
                size_t q = l.find(',', p);
                if (q == std::string::npos) q = l.size();
                dst.push_back(std::atoi(l.substr(p, q - p).c_str()));
                p = q + 1;
            }
        }
        else if (a == "--reps") opt.reps = std::atoi(next().c_str());
        else if (a == "--trace") opt.trace = next();
        else if (a == "--visit") opt.visit = std::atoi(next().c_str());
        else if (a == "--group") opt.group = std::atoi(next().c_str());
        else if (a == "--chunk") opt.chunk = std::atoi(next().c_str());
        else if (a == "--tile") g_gemm_tile = std::atoi(next().c_str());
        else if (a == "--prefetch") g_gemm_prefetch = std::atoi(next().c_str());
        else if (a == "--epilogue") g_gemm_epilogue = std::atoi(next().c_str());
        else if (a == "--tail5") g_gemm_tail5 = std::atoi(next().c_str());
    }
    g_gemm_group = opt.group;
    std::printf("bench_gemm loadavg %s, group=%d chunk=%d reps=%d\n", tu::loadavg().c_str(), opt.group, opt.chunk, opt.reps);
    if (opt.i16tile > 0) g_gemm_i16_tile = opt.i16tile;
    if (opt.i16seg > 0) g_gemm_i16_seg = opt.i16seg;
    if (opt.i16pf >= 0) g_gemm_i16_pf = opt.i16pf;
    const bool all = opt.what == "all";
    if (all || opt.what == "st") bench_st();
    if (opt.what == "rvs") bench_rvs();
    if (opt.what == "epi") bench_epi();
    if (opt.what == "x4") bench_x4();
    if (opt.what == "i16") bench_i16();
    if (opt.what == "i16u") bench_i16u();
    if (opt.what == "unpack") bench_unpack();
    if (opt.what == "tail") bench_tail();
    if (all || opt.what == "dense" || opt.what == "nvfp4" || opt.what == "moe" || opt.what == "small" || opt.what == "tune") {
        ThreadPool pool(opt.threads, default_cpus());
        if (all || opt.what == "dense") bench_dense(pool);
        if (all || opt.what == "nvfp4") bench_nvfp4(pool);
        if (all || opt.what == "moe") bench_moe(pool);
        if (opt.what == "tune") bench_tune(pool);
        if (all || opt.what == "small") bench_small(pool);
    }
    std::printf("loadavg at end %s\n", tu::loadavg().c_str());
    return 0;
}

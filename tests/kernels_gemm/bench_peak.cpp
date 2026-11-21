// Peak-throughput microbenchmark of the GEMM inner-loop uop mixes on this CPU (agent G).
// Each "pair" is one (weight row, token) accumulator updated for 64 MACs. Operands come from L1 (loads) so the mixes
// match what the real micro-kernel does.  Mixes:
//   fma  : 1 vfmadd per pair                       (calibrates the AVX-512 clock: 2 FMA/cycle on this 2-FMA-unit Xeon)
//   mad  : maddubs only                            (is vpmaddubsw p05 or p0 only?)
//   mix4 : maddubs, madd, cvt, fma                 (no sign transfer)
//   mix5 : mask_sub, maddubs, madd, cvt, fma       (the real kernel: scale broadcast from memory)
//   mix6 : mix5 + vmulps for the scale product     (what B1's gemv does per row)
// Usage: bench_peak [threads=1] [cpus-list e.g. 2,4,...]   (run through tools/with_bench_lock.sh)
#include <immintrin.h>
#include <time.h>

#include <algorithm>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/threadpool.h"

static double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static std::string loadavg() {
    char b[128] = {0};
    if (FILE* f = fopen("/proc/loadavg", "r")) {
        if (!fgets(b, sizeof b, f)) b[0] = 0;
        fclose(f);
    }
    std::string s(b);
    size_t p = 0;
    for (int i = 0; i < 3 && p != std::string::npos; ++i) p = s.find(' ', p + 1);
    return p == std::string::npos ? s : s.substr(0, p);
}

constexpr int NP = 12;  // independent pairs per iteration

struct Bufs {
    alignas(64) int8_t x[NP * 64];
    alignas(64) float s[NP * 16];
    alignas(64) float dx[NP * 16];
};

template <int MIX>
__attribute__((noinline)) float run_mix(long iters, const Bufs& b, uint64_t kmask) {
    __m512 acc[NP];
    _Pragma("GCC unroll 12") for (int p = 0; p < NP; ++p) acc[p] = _mm512_set1_ps(0.5f + 0.001f * (float)p);
    const __m512i ones = _mm512_set1_epi16(1), zero = _mm512_setzero_si512(), aw = _mm512_set1_epi8(37);
    const __m512 dw = _mm512_set1_ps(0.37f), vb = _mm512_set1_ps(1.0009f);
    const __mmask64 km = (__mmask64)kmask;
    for (long it = 0; it < iters; ++it) {
        asm volatile("" ::: "memory");  // forces reloads: nothing below can be hoisted out of the loop
#pragma GCC unroll 12
        for (int p = 0; p < NP; ++p) {
            if constexpr (MIX == 0) {
                acc[p] = _mm512_fmadd_ps(_mm512_set1_ps(b.s[p * 16]), vb, acc[p]);
            } else if constexpr (MIX == 1) {
                const __m512i X = _mm512_load_si512(b.x + 64 * p);
                const __m512i m = _mm512_maddubs_epi16(aw, X);
                acc[p] = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(acc[p]), m));  // keeps it alive; +1 p05 uop (accounted below)
            } else {
                __m512i X = _mm512_load_si512(b.x + 64 * p);
                if constexpr (MIX >= 3) X = _mm512_mask_sub_epi8(X, km, zero, X);
                const __m512i I = _mm512_madd_epi16(_mm512_maddubs_epi16(aw, X), ones);
                __m512 S;
                if constexpr (MIX == 4)
                    S = _mm512_mul_ps(dw, _mm512_load_ps(b.dx + 16 * p));
                else
                    S = _mm512_castpd_ps(_mm512_set1_pd(*(const double*)(b.s + 16 * p)));
                acc[p] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(I), S, acc[p]);
            }
        }
    }
    __m512 t = acc[0];
    _Pragma("GCC unroll 12") for (int p = 1; p < NP; ++p) t = _mm512_add_ps(t, acc[p]);
    return _mm512_reduce_add_ps(t);
}

struct Res {
    double ns_per_pair;
};

template <int MIX>
double measure(const Bufs& b, double secs) {
    long iters = 200000;
    double t = 0;
    volatile float sink = 0;
    for (int rep = 0; rep < 8; ++rep) {
        const double t0 = now_s();
        sink = sink + run_mix<MIX>(iters, b, 0xF0F0F0F00F0F0F0Full);
        t = now_s() - t0;
        if (t > secs * 0.5) break;
        iters = (long)(iters * (secs * 0.6 / (t > 1e-6 ? t : 1e-6)));
    }
    // final measurement: best of 5
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now_s();
        sink = sink + run_mix<MIX>(iters, b, 0xF0F0F0F00F0F0F0Full);
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return best / ((double)iters * NP) * 1e9;  // ns per pair
}

int main(int argc, char** argv) {
    int nthr = argc > 1 ? atoi(argv[1]) : 1;
    std::vector<int> cpus;
    if (argc > 2) {
        std::string s = argv[2];
        size_t p = 0;
        while (p < s.size()) {
            size_t q = s.find(',', p);
            if (q == std::string::npos) q = s.size();
            cpus.push_back(atoi(s.substr(p, q - p).c_str()));
            p = q + 1;
        }
    } else {
        for (int i = 1; i <= 15; ++i) cpus.push_back(2 * i);
        cpus.insert(cpus.begin(), 2);  // tid0 -> cpu 2 ... placeholder, fixed below
        cpus.clear();
        for (int i = 1; i <= 15; ++i) cpus.push_back(2 * i);
        cpus.push_back(0);
    }
    std::printf("loadavg before: %s   threads=%d\n", loadavg().c_str(), nthr);
    ink::ThreadPool pool(nthr, cpus);
    std::vector<Bufs> bufs(nthr);
    for (auto& b : bufs) {
        for (int i = 0; i < NP * 64; ++i) b.x[i] = (int8_t)((i * 37 + 11) % 255 - 127);
        for (int i = 0; i < NP * 16; ++i) {
            b.s[i] = 0.001f * (float)((i % 7) + 1);
            b.dx[i] = 0.002f * (float)((i % 5) + 1);
        }
    }
    const char* names[5] = {"fma   (1 uop/pair)", "madubs(1+1 uop)", "mix4  (4 uop)", "mix5  (5 uop)", "mix6  (6 uop)"};
    std::vector<std::vector<double>> res(5, std::vector<double>(nthr, 0.0));
    auto go = [&](auto tag, int idx) {
        constexpr int M = decltype(tag)::value;
        pool.run([&](int tid, int) {
            pool.barrier();
            res[idx][tid] = measure<M>(bufs[tid], 0.25);
            pool.barrier();
        });
    };
    go(std::integral_constant<int, 0>{}, 0);
    go(std::integral_constant<int, 1>{}, 1);
    go(std::integral_constant<int, 2>{}, 2);
    go(std::integral_constant<int, 3>{}, 3);
    go(std::integral_constant<int, 4>{}, 4);
    // FMA mix: 1 FMA per pair at 2 FMA/cycle -> cycles = 0.5 per pair -> f = 0.5 / ns
    std::printf("%-20s %12s %12s %12s %12s\n", "mix", "ns/pair(min)", "ns/pair(med)", "cyc/pair", "GMAC/s/core(max-thread)");
    double fghz_all = 0;
    for (int i = 0; i < 5; ++i) {
        std::vector<double> v = res[i];
        std::sort(v.begin(), v.end());
        const double mn = v.front(), md = v[v.size() / 2];
        if (i == 0) fghz_all = 0.5 / mn;  // GHz estimate of the fastest thread
        const double cyc = mn * fghz_all;
        // pair = 64 MAC (mix4..6); fma/mad rows are not MACs
        std::printf("%-20s %12.4f %12.4f %12.3f %12.2f\n", names[i], mn, md, cyc, i >= 2 ? 64.0 / mn : 0.0);
    }
    std::printf("estimated AVX-512 FMA clock of fastest thread: %.3f GHz (2 FMA/cycle assumption)\n", fghz_all);
    if (nthr > 1) {
        double agg = 0;
        for (int t = 0; t < nthr; ++t) agg += 64.0 / res[3][t];
        std::printf("aggregate mix5 over %d threads: %.1f GMAC/s  (per-thread min/med/max ns/pair: ", nthr, agg);
        std::vector<double> v = res[3];
        std::sort(v.begin(), v.end());
        std::printf("%.3f/%.3f/%.3f)\n", v.front(), v[v.size() / 2], v.back());
        agg = 0;
        for (int t = 0; t < nthr; ++t) agg += 64.0 / res[4][t];
        std::printf("aggregate mix6 over %d threads: %.1f GMAC/s\n", nthr, agg);
    }
    std::printf("loadavg after: %s\n", loadavg().c_str());
    return 0;
}

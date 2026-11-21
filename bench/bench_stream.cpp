// Pure streaming-read ceiling of the box at the time of the run (same pool, same hugepage memory, same dynamic chunking as the engine).
// Usage (under the shared lock): tools/with_bench_lock.sh build-c1c/engine/bench_stream [threads list] [GB]
#include <immintrin.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ink/engine.h"
#include "ink/kernels_gemv.h"
#include "ink/threadpool.h"

using namespace ink;
int main(int argc, char** argv) {
    std::vector<int> th = {8, 12, 16};
    if (argc > 1) {
        th.clear();
        for (const char* p = argv[1]; *p;) {
            th.push_back(std::atoi(p));
            while (*p && *p != ',') ++p;
            if (*p == ',') ++p;
        }
    }
    const double gb = argc > 2 ? std::atof(argv[2]) : 3.0;
    const size_t bytes = (size_t)(gb * 1e9) / 4096 * 4096;
    char* buf = (char*)alloc_weight_buffer(bytes, true);
    std::memset(buf, 1, bytes);
    for (int n : th) {
        ThreadPool pool(n, engine_default_cpus(n));
        // 1) one big streaming pass, 64 KB chunks
        double best = 0;
        std::vector<double> sums(n * 8, 0.0);
        for (int rep = 0; rep < 6; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            pool.run_chunks((int64_t)(bytes / 65536), 1, [&](int tid, int64_t b, int64_t e) {
                __m512i acc = _mm512_setzero_si512();
                for (int64_t c = b; c < e; ++c) {
                    const char* p = buf + (size_t)c * 65536;
                    for (int o = 0; o < 65536; o += 256) {
                        acc = _mm512_add_epi32(acc, _mm512_loadu_si512(p + o));
                        acc = _mm512_add_epi32(acc, _mm512_loadu_si512(p + o + 64));
                        acc = _mm512_add_epi32(acc, _mm512_loadu_si512(p + o + 128));
                        acc = _mm512_add_epi32(acc, _mm512_loadu_si512(p + o + 192));
                    }
                }
                sums[tid * 8] += (double)_mm512_reduce_add_epi32(acc);
            });
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            best = std::max(best, bytes / s / 1e9);
        }
        // 2) engine-like: 24 MB "phases" (barrier separated), each streamed by dynamic 32 KB chunks
        double best2 = 0;
        const size_t phase_bytes = 24u << 20;
        const int nphase = (int)(bytes / phase_bytes);
        for (int rep = 0; rep < 4; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            std::atomic<int64_t> ctr[64];
            for (auto& c : ctr) c.store(0);
            pool.run([&](int tid, int nth) {
                __m512i acc = _mm512_setzero_si512();
                for (int ph = 0; ph < nphase; ++ph) {
                    const char* base = buf + (size_t)ph * phase_bytes;
                    const int64_t nch = phase_bytes / 32768;
                    std::atomic<int64_t>& a = ctr[ph & 63];
                    if (tid == 0 && ph >= 32) a.store(0);   // (reset is safe: counter (ph-32) was used 32 phases ago)
                    for (;;) {
                        const int64_t c = a.fetch_add(1, std::memory_order_relaxed);
                        if (c >= nch) break;
                        const char* p = base + (size_t)c * 32768;
                        for (int o = 0; o < 32768; o += 256) {
                            acc = _mm512_add_epi32(acc, _mm512_loadu_si512(p + o));
                            acc = _mm512_add_epi32(acc, _mm512_loadu_si512(p + o + 64));
                            acc = _mm512_add_epi32(acc, _mm512_loadu_si512(p + o + 128));
                            acc = _mm512_add_epi32(acc, _mm512_loadu_si512(p + o + 192));
                        }
                    }
                    pool.barrier();
                }
                sums[tid * 8] += (double)_mm512_reduce_add_epi32(acc);
            });
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            best2 = std::max(best2, (double)nphase * phase_bytes / s / 1e9);
            if (nphase > 64) break;
        }
        std::FILE* f = std::fopen("/proc/loadavg", "r");
        char l[64] = {0};
        if (f) {
            if (!std::fgets(l, sizeof l, f)) l[0] = 0;
            std::fclose(f);
        }
        *std::strchr(l, ' ') = 0;
        std::printf("threads %2d: stream (64 KB chunks) %.1f GB/s, 24 MB barrier-separated phases %.1f GB/s   loadavg %s   (checksum %g)\n", n, best, best2, l, sums[0]);
        std::fflush(stdout);
    }
    return 0;
}

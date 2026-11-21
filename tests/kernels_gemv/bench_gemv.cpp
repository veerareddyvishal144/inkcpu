// GEMV bandwidth benchmark (agent B1). Working set >= 2 GB of weights (rotating through distinct matrices so nothing is
// cache resident), ThreadPool with one thread per physical core, rows split with split_rows(align 4) over the fused
// multi-matrix entry points. Reports GB/s of weight bytes, % of the 92.5 GB/s measured streaming ceiling, and the same
// numbers for a pure streaming-sum kernel that reads exactly the same bytes with the same thread split (upper bound).
//
// usage: bench_gemv [--threads 8,12,16] [--shapes substr] [--reps 3] [--time 0.25] [--pool-gb 2.5]
//                   [--pf bytes] [--hint 0|1|2|3] [--nohuge] [--nostream]
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/kernels_gemv.h"
#include "ink/threadpool.h"
#include "testutil.h"

using namespace ink;

static const double kPeakGBs = 92.5;
static bool g_dynamic = true;            // dynamic chunk scheduling (--static: contiguous equal slices)
static std::vector<int> g_cpus;           // --cpus list (empty = pool default: even CPUs)
enum { Q8 = 0, NVFP4 = 1, BF16 = 2, F32 = 3 };
struct Shape {
    const char* name;
    int kind, k, rows, nm;  // nm = matrices per call (experts), rows = rows per matrix
    bool x_per_mat;
};
static const Shape kShapes[] = {
    {"q8_0 3072x2304 (q proj)", Q8, 2304, 3072, 1, false},
    {"q8_0 2304x3072 (wo)", Q8, 3072, 2304, 1, false},
    {"q8_0 1024x2304 (k/v proj)", Q8, 2304, 1024, 1, false},
    {"q8_0 5120x2304 (q|k|v fused)", Q8, 2304, 5120, 1, false},
    {"q8_0 200058x2304 (lm_head)", Q8, 2304, 200058, 1, false},
    {"nvfp4 8x gate_up 2304x2304", NVFP4, 2304, 2304, 8, false},
    {"nvfp4 8x down 2304x1152", NVFP4, 1152, 2304, 8, true},
    {"bf16 3072x2304", BF16, 2304, 3072, 1, false},
    {"f32 2304x2304", F32, 2304, 2304, 1, false},
};

static size_t row_bytes_of(int kind, int k) {
    return kind == Q8 ? row_bytes_q8_0(k) : kind == NVFP4 ? row_bytes_nvfp4(k) : kind == BF16 ? row_bytes_bf16(k) : row_bytes_f32(k);
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void fill_tile(int kind, std::vector<uint8_t>& tile, tu::Rng& rng) {
    const size_t unit = kind == Q8 ? 34 : kind == NVFP4 ? 36 : 64;
    const size_t n = ((size_t)1 << 20) / unit * unit;
    tile.assign(n, 0);
    if (kind == Q8) {
        for (size_t i = 0; i < n; i += 34) {
            block_q8_0* b = (block_q8_0*)&tile[i];
            b->d = f32_to_fp16(0.002f + 0.02f * rng.uni());
            for (int j = 0; j < 32; ++j) b->qs[j] = (int8_t)rng.range(-127, 127);
        }
    } else if (kind == NVFP4) {
        for (size_t i = 0; i < n; i += 36) {
            block_nvfp4* b = (block_nvfp4*)&tile[i];
            for (int s = 0; s < 4; ++s) b->d[s] = (uint8_t)(0x30 + rng.u32() % 0x28);
            for (int j = 0; j < 32; ++j) b->qs[j] = (uint8_t)rng.u32();
        }
    } else if (kind == BF16) {
        uint16_t* h = (uint16_t*)tile.data();
        for (size_t i = 0; i < n / 2; ++i) h[i] = f32_to_bf16(rng.sym());
    } else {
        float* f = (float*)tile.data();
        for (size_t i = 0; i < n / 4; ++i) f[i] = rng.sym();
    }
}

static void fill_pool(ThreadPool& tp, uint8_t* buf, size_t bytes, const std::vector<uint8_t>& tile) {
    const size_t T = tile.size(), ntiles = (bytes + T - 1) / T;
    tp.run([&](int tid, int n) {
        const size_t t0 = ntiles * tid / n, t1 = ntiles * (tid + 1) / n;
        for (size_t t = t0; t < t1; ++t) {
            const size_t off = t * T, len = std::min(T, bytes - off);
            std::memcpy(buf + off, tile.data(), len);
        }
    });
}

static inline uint64_t stream_bytes(const uint8_t* p, size_t n, int pf) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 256 <= n; i += 256) {
        if (pf) {
            _mm_prefetch((const char*)p + i + pf, _MM_HINT_T0);
            _mm_prefetch((const char*)p + i + pf + 64, _MM_HINT_T0);
            _mm_prefetch((const char*)p + i + pf + 128, _MM_HINT_T0);
            _mm_prefetch((const char*)p + i + pf + 192, _MM_HINT_T0);
        }
        a0 = _mm512_add_epi64(a0, _mm512_loadu_si512((const void*)(p + i)));
        a1 = _mm512_add_epi64(a1, _mm512_loadu_si512((const void*)(p + i + 64)));
        a2 = _mm512_add_epi64(a2, _mm512_loadu_si512((const void*)(p + i + 128)));
        a3 = _mm512_add_epi64(a3, _mm512_loadu_si512((const void*)(p + i + 192)));
    }
    for (; i + 64 <= n; i += 64) a0 = _mm512_add_epi64(a0, _mm512_loadu_si512((const void*)(p + i)));
    a0 = _mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3));
    return (uint64_t)_mm512_reduce_add_epi64(a0);
}

struct Meas {
    double gbps = 0;
    std::string load;
};

// mode 0 = the GEMV kernel, 1 = streaming sum with the current prefetch distance, 2 = streaming sum without prefetch
static Meas measure(ThreadPool& pool, const Shape& s, uint8_t* buf, size_t buf_bytes, int mode, int reps, double min_time) {
    tu::Rng rng(99);
    const size_t rb = row_bytes_of(s.kind, s.k);
    const size_t mat_bytes = (size_t)s.rows * rb;
    const size_t count = buf_bytes / mat_bytes;
    const int nm = s.nm;
    const int nx = s.x_per_mat ? nm : 1;
    std::vector<std::vector<block_q8_0>> xq(nx);
    std::vector<std::vector<float>> xf(nx);
    for (int i = 0; i < nx; ++i) {
        std::vector<float> x(s.k);
        for (int j = 0; j < s.k; ++j) x[j] = rng.gauss();
        if (s.kind == Q8 || s.kind == NVFP4) {
            xq[i].resize(s.k / 32);
            quantize_row_q8_0(x.data(), xq[i].data(), s.k);
        } else {
            xf[i] = x;
        }
    }
    std::vector<float> y_all((size_t)nm * s.rows);
    std::vector<GemvMat> mats(nm);
    std::vector<uint64_t> sink(64 * 8, 0);
    size_t cursor = 0;
    const int pf_stream = mode == 2 ? 0 : g_gemv_prefetch_bytes;
    auto one_call = [&]() {
        for (int j = 0; j < nm; ++j) {
            const size_t idx = (cursor + j) % count;
            const int xi = s.x_per_mat ? j : 0;
            mats[j] = GemvMat{buf + idx * mat_bytes, rb, (s.kind == Q8 || s.kind == NVFP4) ? (const void*)xq[xi].data() : (const void*)xf[xi].data(),
                              y_all.data() + (size_t)j * s.rows, s.rows};
        }
        cursor = (cursor + nm) % count;
        auto work = [&](int tid, int64_t b, int64_t e) {
            if (mode == 0) {
                switch (s.kind) {
                    case Q8: gemv_multi_q8_0(mats.data(), nm, s.k, b, e); break;
                    case NVFP4: gemv_multi_nvfp4(mats.data(), nm, s.k, b, e); break;
                    case BF16: gemv_multi_bf16(mats.data(), nm, s.k, b, e); break;
                    default: gemv_multi_f32(mats.data(), nm, s.k, b, e); break;
                }
            } else {
                int64_t base = 0;
                uint64_t acc = 0;
                for (int i = 0; i < nm && base < e; ++i) {
                    const int64_t lo = std::max(b, base), hi = std::min(e, base + s.rows);
                    if (lo < hi) acc += stream_bytes((const uint8_t*)mats[i].w + (size_t)(lo - base) * rb, (size_t)(hi - lo) * rb, pf_stream);
                    base += s.rows;
                }
                sink[tid * 8] += acc;
            }
        };
        const int64_t total = (int64_t)nm * s.rows;
        if (g_dynamic) {
            int64_t chunk = total / ((int64_t)pool.size() * 6);
            chunk = std::max<int64_t>(16, std::min<int64_t>(512, chunk)) & ~(int64_t)3;
            pool.run_chunks(total, chunk, work);
        } else {
            pool.run([&](int tid, int n) {
                int64_t b, e;
                split_rows(total, n, tid, 4, &b, &e);
                if (b < e) work(tid, b, e);
            });
        }
    };
    Meas best;
    const double call_bytes = (double)nm * (double)mat_bytes;
    for (int rep = 0; rep < reps; ++rep) {
        const std::string la = tu::loadavg();
        for (int i = 0; i < 3; ++i) one_call();
        size_t calls = 0;
        const double t0 = now_s();
        double el;
        do {
            for (int i = 0; i < 4; ++i) one_call();
            calls += 4;
            el = now_s() - t0;
        } while (el < min_time || calls < 8);
        const double gbps = (double)calls * call_bytes / el / 1e9;
        if (gbps > best.gbps) { best.gbps = gbps; best.load = la; }
    }
    return best;
}

static long anon_huge_kb() {
    long v = -1;
    if (FILE* f = std::fopen("/proc/self/smaps_rollup", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (!std::strncmp(line, "AnonHugePages:", 14)) v = std::atol(line + 14);
        std::fclose(f);
    }
    return v;
}

int main(int argc, char** argv) {
    std::vector<int> threads = {8, 12, 16};
    std::string filter;
    int reps = 3;
    double min_time = 0.25, pool_gb = 2.5;
    bool huge = true, do_stream = true;
    int ov_rows = 0, ov_nm = 0;  // --rows/--nm override the shape (cache-resident compute-throughput experiments)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--threads") {
            threads.clear();
            std::string v = next();
            size_t p = 0;
            while (p < v.size()) { threads.push_back(std::atoi(v.c_str() + p)); p = v.find(',', p); if (p == std::string::npos) break; ++p; }
        } else if (a == "--shapes") filter = next();
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--time") min_time = std::atof(next());
        else if (a == "--pool-gb") pool_gb = std::atof(next());
        else if (a == "--pf") g_gemv_prefetch_bytes = std::atoi(next());
        else if (a == "--hint") g_gemv_prefetch_hint = std::atoi(next());
        else if (a == "--nohuge") huge = false;
        else if (a == "--nostream") do_stream = false;
        else if (a == "--static") g_dynamic = false;
        else if (a == "--cpus") {
            g_cpus.clear();
            std::string v = next();
            size_t p = 0;
            while (p < v.size()) { g_cpus.push_back(std::atoi(v.c_str() + p)); p = v.find(',', p); if (p == std::string::npos) break; ++p; }
        }
        else if (a == "--rows") ov_rows = std::atoi(next());
        else if (a == "--nm") ov_nm = std::atoi(next());
    }
    const size_t bytes = (size_t)(pool_gb * 1e9);
    uint8_t* buf = (uint8_t*)alloc_weight_buffer(bytes, huge);
    if (!buf) { std::fprintf(stderr, "alloc failed\n"); return 1; }
    ThreadPool fillpool(16);
    std::printf("%s scheduling; pool %.2f GB, hugepage madvise=%d, prefetch %d B hint %d, reps %d x %.2fs, peak %.1f GB/s\n", g_dynamic ? "dynamic" : "static", (double)bytes / 1e9, (int)huge,
                g_gemv_prefetch_bytes, g_gemv_prefetch_hint, reps, min_time, kPeakGBs);
    std::printf("%-32s %3s | %8s %6s | %8s %8s | %6s | load(1/5/15)\n", "shape", "thr", "kernel", "%peak", "stream", "stream", "k/str");
    std::printf("%-32s %3s | %8s %6s | %8s %8s | %6s |\n", "", "", "GB/s", "", "pf GB/s", "nopf", "");
    int cur_kind = -1;
    for (const Shape& s0 : kShapes) {
        Shape s = s0;
        if (ov_rows) s.rows = ov_rows;
        if (ov_nm) s.nm = ov_nm;
        if (!filter.empty() && std::string(s0.name).find(filter) == std::string::npos) continue;
        if (s.kind != cur_kind) {
            tu::Rng rng(5 + s.kind);
            std::vector<uint8_t> tile;
            fill_tile(s.kind, tile, rng);
            fill_pool(fillpool, buf, bytes, tile);
            cur_kind = s.kind;
            std::printf("# filled pool for kind %d, AnonHugePages %ld kB\n", s.kind, anon_huge_kb());
        }
        for (int nt : threads) {
            std::vector<int> cl;
            for (int i = 0; i < nt && i < (int)g_cpus.size(); ++i) cl.push_back(g_cpus[i]);
            ThreadPool pool(nt, cl);
            // interleave kernel / stream windows so that both see the same background load; keep the best window of each
            Meas k, s1, s2;
            for (int rep = 0; rep < reps; ++rep) {
                const Meas a = measure(pool, s, buf, bytes, 0, 1, min_time);
                if (a.gbps > k.gbps) k = a;
                if (do_stream) {
                    const Meas b = measure(pool, s, buf, bytes, 1, 1, min_time);
                    if (b.gbps > s1.gbps) s1 = b;
                    const Meas c = measure(pool, s, buf, bytes, 2, 1, min_time);
                    if (c.gbps > s2.gbps) s2 = c;
                }
            }
            std::printf("%-32s %3d | %8.1f %5.1f%% | %8.1f %8.1f | %5.2f | %s\n", s.name, nt, k.gbps, 100.0 * k.gbps / kPeakGBs, s1.gbps, s2.gbps,
                        std::max(s1.gbps, s2.gbps) > 0 ? k.gbps / std::max(s1.gbps, s2.gbps) : 0.0, k.load.c_str());
            std::fflush(stdout);
        }
    }
    free_weight_buffer(buf);
    return 0;
}

// Benchmarks for the batched GEMM kernels (agent G). Run through tools/with_bench_lock.sh. Prints GMAC/s, GMAC/s per thread and the
// equivalent prompt tokens/s (5.35 GMAC per token), plus /proc/loadavg before and after every section.
//   bench_gemm [--what st|dense|nvfp4|moe|small|all] [--threads 16] [--reps 7] [--trace PATH] [--visit 12] [--group 64] [--chunk 64]
#include <immintrin.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/kernels_gemm.h"
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
    ActBuf(int fmt, int nt, int k) {
        p = std::aligned_alloc(64, (gemm_act_bytes(fmt, nt, k) + 63) & ~(size_t)63);
        std::vector<block_q8_0> q = make_acts(nt, k);
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
    std::vector<int> c;
    for (int i = 1; i <= 15; ++i) c.push_back(2 * i);
    c.push_back(0);
    return c;
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
    for (int nt : {64, 128, 256}) {
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
        auto run_gu = [&](const GemmAct& a, const int32_t* rows_idx) {
            return [&, rows_idx]() {
                pool.run_chunks((int64_t)items.size(), 1, [&](int, int64_t b, int64_t e) {
                    for (int64_t i = b; i < e; ++i) gemm_moe_nvfp4(wg, esg, rsg, a, rows_idx, r, items[(size_t)i], Kg, yg.data(), N);
                });
            };
        };
        auto run_dn = [&]() {
            pool.run_chunks((int64_t)items.size(), 1, [&](int, int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) gemm_moe_nvfp4(wd, esd, rsd, ap.a, nullptr, r, items[(size_t)i], Kd, yd.data(), N);
            });
        };
        const double gm_gu = (double)np * N * Kg * 1e-9, gm_dn = (double)np * N * Kd * 1e-9;
        g_peak_mt = peak_mt(pool);
        auto a = time_best(run_gu(at.a, r.pair_token.data()), opt.reps);
        std::snprintf(name, sizeof name, "moe gate_up %dx%d, indirect act rows", N, Kg);
        report(name, gm_gu, a.first, a.second, pool.size(), g_peak_mt);
        auto b = time_best(run_gu(agat, nullptr), opt.reps);
        std::snprintf(name, sizeof name, "moe gate_up %dx%d, pre-gathered act", N, Kg);
        report(name, gm_gu, b.first, b.second, pool.size(), g_peak_mt);
        auto c = time_best(run_dn, opt.reps);
        std::snprintf(name, sizeof name, "moe down %dx%d (per-pair act)", N, Kd);
        report(name, gm_dn, c.first, c.second, pool.size(), g_peak_mt);
        std::snprintf(name, sizeof name, "moe layer (gate_up + down, indirect)");
        report(name, gm_gu + gm_dn, a.first + c.first, a.second + c.second, pool.size(), g_peak_mt);
        std::free(gbuf);
    }
    free_weight_buffer(wg);
    free_weight_buffer(wd);
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
        else if (a == "--threads") opt.threads = std::atoi(next().c_str());
        else if (a == "--reps") opt.reps = std::atoi(next().c_str());
        else if (a == "--trace") opt.trace = next();
        else if (a == "--visit") opt.visit = std::atoi(next().c_str());
        else if (a == "--group") opt.group = std::atoi(next().c_str());
        else if (a == "--chunk") opt.chunk = std::atoi(next().c_str());
        else if (a == "--tile") g_gemm_tile = std::atoi(next().c_str());
        else if (a == "--prefetch") g_gemm_prefetch = std::atoi(next().c_str());
    }
    g_gemm_group = opt.group;
    std::printf("bench_gemm loadavg %s, group=%d chunk=%d reps=%d\n", tu::loadavg().c_str(), opt.group, opt.chunk, opt.reps);
    const bool all = opt.what == "all";
    if (all || opt.what == "st") bench_st();
    if (all || opt.what == "dense" || opt.what == "nvfp4" || opt.what == "moe" || opt.what == "small") {
        ThreadPool pool(opt.threads, default_cpus());
        if (all || opt.what == "dense") bench_dense(pool);
        if (all || opt.what == "nvfp4") bench_nvfp4(pool);
        if (all || opt.what == "moe") bench_moe(pool);
        if (all || opt.what == "small") bench_small(pool);
    }
    std::printf("loadavg at end %s\n", tu::loadavg().c_str());
    return 0;
}

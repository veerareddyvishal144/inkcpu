// Microbenchmarks for ink_kernels_misc.  Owner: agent B2.  Prints the host load average first: numbers at load > 2 are indicative only.
// usage: bench_kernels_misc [threads=16]
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <thread>
#include <vector>
#include "ink/kernels_misc.h"

using namespace ink;
using clk = std::chrono::steady_clock;

static std::mt19937_64 rng(7);
static float nrand() { return std::normal_distribution<float>(0.0f, 1.0f)(rng); }
static void print_load(const char* tag) {
    FILE* f = std::fopen("/proc/loadavg", "r");
    char buf[128] = {0};
    if (f) { if (!std::fgets(buf, sizeof buf, f)) buf[0] = 0; std::fclose(f); }
    buf[std::strcspn(buf, "\n")] = 0;
    std::printf("[load %s: %s]\n", tag, buf);
}
// best-of-N seconds per call
static double timeit(const std::function<void()>& fn, double min_total = 0.05, int trials = 7) {
    fn();
    size_t iters = 1;
    for (;;) {
        auto t0 = clk::now();
        for (size_t i = 0; i < iters; ++i) fn();
        const double dt = std::chrono::duration<double>(clk::now() - t0).count();
        if (dt >= min_total / 4) break;
        iters *= 2;
    }
    double best = 1e30;
    for (int t = 0; t < trials; ++t) {
        auto t0 = clk::now();
        for (size_t i = 0; i < iters; ++i) fn();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count() / iters);
    }
    return best;
}
static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

static void bench_vecmath() {
    std::printf("== vector math (single thread, pinned CPU 0)\n");
    for (size_t n : {(size_t)201024, (size_t)9216}) {
        std::vector<float> x(n), y(n);
        for (auto& v : x) v = nrand() * 8.0f;
        struct B { const char* name; std::function<void()> opt, scalar; };
        std::vector<float> u(n);
        for (auto& v : u) v = nrand() * 8.0f;
        const B bs[] = {
            {"tanh", [&] { vtanh(x.data(), y.data(), n); }, [&] { for (size_t i = 0; i < n; ++i) y[i] = tanhf(x[i]); }},
            {"sigmoid", [&] { vsigmoid(x.data(), y.data(), n); }, [&] { for (size_t i = 0; i < n; ++i) y[i] = 1.0f / (1.0f + expf(-x[i])); }},
            {"exp", [&] { vexp(x.data(), y.data(), n); }, [&] { for (size_t i = 0; i < n; ++i) y[i] = expf(x[i]); }},
            {"softcap30", [&] { vsoftcap(x.data(), y.data(), n, 30.0f); }, [&] { for (size_t i = 0; i < n; ++i) y[i] = 30.0f * tanhf(x[i] * (1.0f / 30.0f)); }},
            {"situ", [&] { vsitu(x.data(), u.data(), y.data(), n); },
             [&] { for (size_t i = 0; i < n; ++i) y[i] = 4.0f * tanhf(x[i] * 0.25f) * (1.0f / (1.0f + expf(-x[i]))) * 25.0f * tanhf(u[i] * (1.0f / 25.0f)); }},
        };
        for (const B& b : bs) {
            const double to = timeit(b.opt), ts = timeit(b.scalar);
            std::printf("  n=%-7zu %-9s avx512 %7.3f ns/elem (%.2f Gelem/s, %.1f us/call)   scalar libm %7.3f ns/elem   speedup %.1fx\n", n, b.name,
                        to / n * 1e9, n / to / 1e9, to * 1e6, ts / n * 1e9, ts / to);
        }
    }
}

static void bench_small() {
    std::printf("== norms, clips, conv, router, mix (single thread)\n");
    const size_t n = 2304;
    std::vector<float> x(n), w(n), y(n), y2(n);
    for (size_t i = 0; i < n; ++i) { x[i] = nrand(); w[i] = 1.0f + 0.1f * nrand(); }
    std::vector<block_q8_0> q(n / 32);
    auto rep = [&](const char* name, double t, double bytes) { std::printf("  %-34s %8.3f us/call%s\n", name, t * 1e6, bytes > 0 ? "" : ""); (void)bytes; };
    rep("rmsnorm n=2304 (w)", timeit([&] { rmsnorm(x.data(), w.data(), y.data(), n); }), 0);
    rep("rmsnorm_q8_0 n=2304 (fused)", timeit([&] { rmsnorm_q8_0(x.data(), w.data(), q.data(), n); }), 0);
    {
        std::vector<float> g(9216), u(9216);
        for (auto& v : g) v = nrand() * 3;
        for (auto& v : u) v = nrand() * 3;
        std::vector<block_q8_0> qq(9216 / 32);
        rep("vsitu_q8_0 8 experts x 1152 (fused)", timeit([&] { vsitu_q8_0(g.data(), u.data(), qq.data(), 9216); }), 0);
    }
    rep("rmsnorm + quantize_q8_0 (2 passes)", timeit([&] { rmsnorm(x.data(), w.data(), y.data(), n); quantize_q8_0(y.data(), q.data(), n); }), 0);
    std::vector<float> qh(3072), sc(3072, 1.1f), kh(1024), rh(384), rs(384, 0.9f);
    for (auto& v : qh) v = nrand() * 5;
    for (auto& v : kh) v = nrand() * 5;
    for (auto& v : rh) v = nrand() * 3;
    std::vector<float> qh0 = qh, kh0 = kh, rh0 = rh;
    rep("l2clip Q 24x128 (chan scale)", timeit([&] { std::memcpy(qh.data(), qh0.data(), 3072 * 4); l2clip_heads(qh.data(), sc.data(), 24, 128, 22.627f); }), 0);
    rep("l2clip K 8x128", timeit([&] { std::memcpy(kh.data(), kh0.data(), 4096); l2clip_heads(kh.data(), nullptr, 8, 128, 22.627f); }), 0);
    rep("l2clip R 24x16 (scale, tau)", timeit([&] { std::memcpy(rh.data(), rh0.data(), 384 * 4); l2clip_heads(rh.data(), rs.data(), 24, 16, 8.0f, 1.1f); }), 0);
    rep("memcpy baseline 3072 floats", timeit([&] { std::memcpy(qh.data(), qh0.data(), 3072 * 4); }), 0);
    rep("vnorm V 8x128", timeit([&] { vnorm_heads(kh0.data(), kh.data(), 8, 128); }), 0);
    for (size_t C : {(size_t)1024, (size_t)2304}) {
        std::vector<float> wsoa(4 * C, 0.3f), hist(3 * C, 0.1f), xin(C, 0.5f), yout(C), dyn(C / 128 * 4, 0.2f);
        char nm[64];
        std::snprintf(nm, sizeof nm, "sconv_step C=%zu static", C);
        rep(nm, timeit([&] { sconv_step(xin.data(), hist.data(), wsoa.data(), yout.data(), C, 0, C); }), 0);
        std::snprintf(nm, sizeof nm, "sconv_step C=%zu dynamic taps", C);
        rep(nm, timeit([&] { sconv_step(xin.data(), hist.data(), wsoa.data(), yout.data(), C, 0, C, dyn.data()); }), 0);
    }
    {
        std::vector<std::vector<float>> st(5, std::vector<float>(n));
        std::vector<const float*> sp(5);
        for (int s = 0; s < 5; ++s) { for (auto& v : st[s]) v = nrand(); sp[s] = st[s].data(); }
        std::vector<float> qq(4 * n), o0(n), o1(n), o2(n), o3(n);
        for (auto& v : qq) v = nrand();
        float* outs[4] = {o0.data(), o1.data(), o2.data(), o3.data()};
        for (int nq : {1, 3, 4}) {
            char nm[64];
            std::snprintf(nm, sizeof nm, "depth_mix ns=5 nq=%d n=2304", nq);
            rep(nm, timeit([&] { depth_mix(sp.data(), 5, qq.data(), nq, outs, n, 1.0f / std::sqrt(2304.0f)); }), 0);
        }
        std::vector<float> a(n), b(n), c(n), d(n);
        for (size_t i = 0; i < n; ++i) { a[i] = nrand(); b[i] = nrand(); c[i] = nrand(); d[i] = nrand(); }
        rep("ctrl_apply n=2304 (2 tanh)", timeit([&] { ctrl_apply(a.data(), b.data(), c.data(), d.data(), y.data(), n); }), 0);
        rep("vadd n=2304", timeit([&] { vadd(a.data(), b.data(), y.data(), n); }), 0);
        rep("vmul_sigmoid n=3072", timeit([&] { vmul_sigmoid(qh0.data(), qh.data(), y2.data() , 2304); }), 0);
    }
    {
        std::vector<float> lg(128), bias(128);
        for (auto& v : lg) v = nrand() * 2.5f;
        for (auto& v : bias) v = nrand() * 0.1f;
        int ids[8];
        float wt[8];
        rep("router_topk 128 -> 8", timeit([&] { router_topk(lg.data(), bias.data(), 128, 8, 8.0f, 1e-20f, ids, wt); }), 0);
        rep("ref::router_topk (std::sort)", timeit([&] { ref::router_topk(lg.data(), bias.data(), 128, 8, 8.0f, 1e-20f, ids, wt); }), 0);
    }
    {
        std::vector<float> k(1024), v(1024);
        for (auto& e : k) e = nrand();
        for (auto& e : v) e = nrand();
        KvLayer L;
        L.ring = true; L.n_blocks = 16;
        L.k = (uint16_t*)std::aligned_alloc(64, L.bytes_per_tensor());
        L.v = (uint16_t*)std::aligned_alloc(64, L.bytes_per_tensor());
        std::memset(L.k, 0, L.bytes_per_tensor()); std::memset(L.v, 0, L.bytes_per_tensor());
        int64_t pos = 0;
        rep("kv_append (8 heads, K transposed)", timeit([&] { kv_append(L, pos++, k.data(), v.data()); }), 0);
        std::free(L.k); std::free(L.v);
    }
}

// ---- attention
struct Cache {
    KvLayer L;
    Cache(bool ring, uint32_t nblk) {
        L.ring = ring; L.n_blocks = nblk;
        L.k = (uint16_t*)std::aligned_alloc(64, L.bytes_per_tensor());
        L.v = (uint16_t*)std::aligned_alloc(64, L.bytes_per_tensor());
        uint64_t s = 88172645463325252ull;
        for (uint16_t* p : {L.k, L.v}) {
            const size_t ne = L.bytes_per_tensor() / 2;
            if (ne > ((size_t)1 << 28)) { std::memset(p, 0x30, ne * 2); continue; }  // huge caches: constant finite fp16 (0x3030)
            for (size_t i = 0; i < ne; ++i) {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                p[i] = (uint16_t)(((s & 1) << 15) | (0x2C00 + ((s >> 8) & 0x0BFF)));  // finite fp16, magnitudes ~ 2^-4 .. 2
            }
        }
    }
    ~Cache() { std::free(L.k); std::free(L.v); }
};

struct Pool {  // persistent pinned spinning workers running task(thread_idx)
    int T;
    std::vector<std::thread> th;
    std::atomic<int> gen{0}, done{0};
    std::atomic<bool> quit{false};
    std::function<void(int)> task;
    explicit Pool(int T_) : T(T_) {
        for (int i = 1; i < T; ++i)
            th.emplace_back([this, i] {
                pin(2 * i);
                int seen = 0;
                for (;;) {
                    while (gen.load(std::memory_order_acquire) == seen && !quit.load()) _mm_pause();
                    if (quit.load()) return;
                    seen = gen.load();
                    task(i);
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
            });
    }
    void run(const std::function<void(int)>& f) {
        task = f;
        done.store(0);
        gen.fetch_add(1, std::memory_order_acq_rel);
        task(0);
        while (done.load(std::memory_order_acquire) < T - 1) _mm_pause();
    }
    ~Pool() { quit.store(true); for (auto& t : th) t.join(); }
};

static bool g_big = false;
static void bench_attention(int T) {
    std::printf("== attention decode (1 query token, 24 q heads / 8 kv heads / d128, fp16 KV)\n");
    std::vector<float> rb(1024 * 16), table(16 * kRelTableLen), q(3072), r(384), sink(24), gate(3072), out(3072);
    for (auto& v : rb) v = nrand() * 0.5f;
    attn_repack_rel_bias(rb.data(), table.data());
    for (auto& v : q) v = nrand();
    for (auto& v : r) v = nrand();
    for (auto& v : sink) v = nrand();
    for (auto& v : gate) v = nrand();
    pin(0);
    {
        Cache c(true, 16);
        AttnQuery aq{q.data(), r.data(), sink.data(), gate.data(), 5000, 1.0f};
        std::vector<AttnPartial> sc(8);
        const double t = timeit([&] { attn_decode(c.L, table.data(), aq, 1, sc.data(), out.data()); });
        const double bytes = 8.0 * 1024 * 128 * 2 * 2;
        std::printf("  SWA ring 1024 keys, 1 thread (all 8 groups): %8.1f us/visit  (KV %.1f MB -> %.1f GB/s, L2/L3 resident)\n", t * 1e6, bytes / 1e6, bytes / t / 1e9);
        // 40 SWA visits per token
        std::printf("      x40 SWA visits per token = %.2f ms single-thread\n", t * 40 * 1e3);
        if (T > 1) {
            Pool pool(T);
            for (int nsplit : {1, 2}) {
                std::vector<AttnPartial> sc2(8 * nsplit);
                auto run = [&] {
                    pool.run([&](int ti) {
                        for (int item = ti; item < 8 * nsplit; item += T) attn_decode_task(c.L, table.data(), aq, item / nsplit, item % nsplit, nsplit, &sc2[item]);
                    });
                    for (int g = 0; g < 8; ++g) attn_decode_finish(aq, g, &sc2[g * nsplit], nsplit, out.data());
                };
                const double tt = timeit(run, 0.05, 5);
                std::printf("  SWA ring, %d threads, nsplit %d (%d tasks): %8.1f us/visit (incl. spin-barrier)  x40 = %.2f ms/token  (%.1f GB/s)\n", T, nsplit, 8 * nsplit, tt * 1e6, tt * 40 * 1e3, bytes / tt / 1e9);
            }
        }
    }
    if (g_big) {  // one global visit at 1M positions (4.3 GB of K+V in this single layer visit); validates 64-bit indexing at the model's max context
        const int64_t ctx = 1048576;
        Cache c(false, (uint32_t)(ctx / 64));
        AttnQuery aq{q.data(), r.data(), sink.data(), gate.data(), ctx - 1, 1.3f};
        const double bytes = 8.0 * ctx * 128 * 2 * 2;
        Pool pool(T);
        const int nsplit = std::max(1, T / 8) * 2;
        std::vector<AttnPartial> sc2(8 * nsplit);
        auto run = [&] {
            pool.run([&](int ti) {
                for (int item = ti; item < 8 * nsplit; item += T) attn_decode_task(c.L, table.data(), aq, item / nsplit, item % nsplit, nsplit, &sc2[item]);
            });
            for (int g = 0; g < 8; ++g) attn_decode_finish(aq, g, &sc2[g * nsplit], nsplit, out.data());
        };
        const double tt = timeit(run, 0.05, 3);
        bool finite = true;
        for (float v : out) finite = finite && std::isfinite(v);
        std::printf("  global ctx 1M: %d threads nsplit %d: %.2f ms/visit  KV %.2f GB -> %.1f GB/s  (output finite: %d); x8 visits = %.1f ms/token\n", T, nsplit, tt * 1e3, bytes / 1e9, bytes / tt / 1e9, (int)finite, tt * 8 * 1e3);
    }
    for (int64_t ctx : {8192, 32768, 131072}) {
        Cache c(false, (uint32_t)(ctx / 64));
        AttnQuery aq{q.data(), r.data(), sink.data(), gate.data(), ctx - 1, 1.2f};
        std::vector<AttnPartial> sc(8 * 4);
        const double bytes = 8.0 * ctx * 128 * 2 * 2;
        const double t1 = timeit([&] { attn_decode(c.L, table.data(), aq, 1, sc.data(), out.data()); }, 0.05, 3);
        std::printf("  global ctx %-7ld 1 thread:  %9.1f us/visit  KV %7.1f MB -> %5.1f GB/s", (long)ctx, t1 * 1e6, bytes / 1e6, bytes / t1 / 1e9);
        if (T > 1) {
            Pool pool(T);
            const int nsplit = std::max(1, T / 8);
            std::vector<AttnPartial> sc2(8 * nsplit);
            auto run = [&] {
                pool.run([&](int ti) {
                    for (int item = ti; item < 8 * nsplit; item += T) attn_decode_task(c.L, table.data(), aq, item / nsplit, item % nsplit, nsplit, &sc2[item]);
                });
                for (int g = 0; g < 8; ++g) attn_decode_finish(aq, g, &sc2[g * nsplit], nsplit, out.data());
            };
            const double tt = timeit(run, 0.05, 5);
            std::printf("   | %d threads (nsplit %d): %9.1f us  -> %5.1f GB/s (%.0f%% of 92.5 GB/s)", T, nsplit, tt * 1e6, bytes / tt / 1e9, 100.0 * bytes / tt / 1e9 / 92.5);
        }
        std::printf("\n");
    }
    std::printf("== attention prefill (SWA ring 2048 slots, pos0 3000, 512 tokens; global ctx 8192, 128 tokens), 1 thread, 8 groups\n");
    {
        const int n = 512;
        Cache c(true, 32);
        std::vector<float> pq(n * 3072), pr(n * 384), ps(n * 24), pg(n * 3072), po(n * 3072);
        for (auto& v : pq) v = nrand();
        for (auto& v : pr) v = nrand();
        for (auto& v : ps) v = nrand();
        for (auto& v : pg) v = nrand();
        AttnPrefill pf{pq.data(), pr.data(), ps.data(), pg.data(), nullptr, 3000, n};
        const double t = timeit([&] { for (int g = 0; g < 8; ++g) attn_prefill_group(c.L, table.data(), pf, g, 0, n, po.data()); }, 0.05, 3);
        std::vector<AttnPartial> sc(8);
        AttnQuery aq{q.data(), r.data(), sink.data(), gate.data(), 5000, 1.0f};
        const double td = timeit([&] { attn_decode(c.L, table.data(), aq, 1, sc.data(), out.data()); });
        std::printf("  SWA prefill: %.1f us/token (%.2fx the decode-kernel cost per token %.1f us)\n", t / n * 1e6, td / (t / n), td * 1e6);
    }
    {
        const int n = 128;
        Cache c(false, 8192 / 64 + 4);
        std::vector<float> pq(n * 3072), pr(n * 384), ps(n * 24), pg(n * 3072), po(n * 3072), tau(n, 1.1f);
        for (auto& v : pq) v = nrand();
        for (auto& v : pr) v = nrand();
        for (auto& v : ps) v = nrand();
        for (auto& v : pg) v = nrand();
        AttnPrefill pf{pq.data(), pr.data(), ps.data(), pg.data(), tau.data(), 8192 - n, n};
        const double t = timeit([&] { for (int g = 0; g < 8; ++g) attn_prefill_group(c.L, table.data(), pf, g, 0, n, po.data()); }, 0.05, 3);
        const double flop = 2.0 * 2.0 * 24 * 128 * (8192.0 - n / 2.0) * n;
        std::printf("  global prefill (~8K keys/query): %.1f us/token  (%.1f GFLOP/s MAC-equivalent single thread)\n", t / n * 1e6, flop / t / 1e9);
    }
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 16;
    g_big = argc > 2 && std::atoi(argv[2]) != 0;  // second argument 1: also run the 1M-context global visit (allocates 4.3 GB)
    pin(0);
    print_load("start");
    bench_vecmath();
    bench_small();
    bench_attention(T);
    print_load("end");
    return 0;
}

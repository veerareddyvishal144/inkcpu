// bench_drafter: time of Drafter::draft() per draft length, per phase (rdtsc on thread 0), for several context lengths; light target (real lm_head, no 17 GB
// engine) so it can run next to other jobs; use INK_CPUS / --threads. Shared host: numbers are indicative (the load average is printed). Owner: agent S1.
//   bench_drafter [--threads 12] [--ctx 128,1024,4096] [--n 1,3,7] [--reps 20] [--sample] [--bf16] [--commit K]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <numeric>
#include <vector>

#include "../tests/drafter/light_target.h"

using namespace ink;
using namespace ink::dr;

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

int main(int argc, char** argv) {
    int threads = 12, reps = 20, commit_k = 3;
    std::vector<int> ctxs = {128, 1024, 4096}, ns = {1, 3, 7};
    bool sample = false, bf16 = false;
    std::string target = "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf", drafter = "/data/inkling-mini/models/dspark-Inkling-2-Mini-Q8_0.gguf";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctxs = parse_list(argv[++i]);
        else if (a == "--n" && i + 1 < argc) ns = parse_list(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--commit" && i + 1 < argc) commit_k = std::atoi(argv[++i]);
        else if (a == "--sample") sample = true;
        else if (a == "--bf16") bf16 = true;
        else if (a == "--target" && i + 1 < argc) target = argv[++i];
        else if (a == "--drafter" && i + 1 < argc) drafter = argv[++i];
    }
    int maxctx = *std::max_element(ctxs.begin(), ctxs.end());
    LightTarget lt;
    std::string err;
    if (!lt.init(target, threads, maxctx + 64, &err)) { std::fprintf(stderr, "light target: %s\n", err.c_str()); return 2; }
    DrafterOptions opt;
    opt.fold_q8 = !bf16;
    opt.max_states = 2;
    double t0 = now_ms();
    auto D = DrafterImpl::make(lt.t, drafter, opt, &err);
    if (!D) { std::fprintf(stderr, "create: %s\n", err.c_str()); return 2; }
    std::printf("drafter create %.2f s (fold %s %.2f s), weights+fold %.0f MB, %d threads, M as %s\n", (now_ms() - t0) / 1e3, D->fold_from_cache ? "from cache" : "computed",
                D->fold_seconds, D->weight_bytes() / 1e6, threads, bf16 ? "bf16" : "Q8_0");
    std::vector<float> taps((size_t)256 * kTapFloats);
    uint64_t rs = 88172645463325252ull;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * (1.0 / 9007199254740992.0)) - 0.5f; };
    for (auto& x : taps) x = rnd() * 6.0f;
    const int st = D->state_new();
    for (int ctx : ctxs) {
        D->state_reset(st);
        double tc = now_ms();
        for (int done = 0; done < ctx; done += 256) D->commit(st, taps.data(), std::min(256, ctx - done));
        D->flush(st);
        tc = now_ms() - tc;
        std::printf("\n=== ctx %d (committed %.1f ms = %.2f ms per 64 rows) loadavg ", ctx, tc, tc / (ctx / 64.0));
        if (FILE* f = std::fopen("/proc/loadavg", "r")) { char b[64] = {}; if (std::fgets(b, sizeof b, f)) std::printf("%s", b); std::fclose(f); }
        for (int n : ns) {
            DraftRequest rq;
            rq.anchor = 1000; rq.n = n; rq.sample = sample; rq.temperature = 1.0f;
            int32_t tok[8];
            std::vector<float> q(sample ? (size_t)7 * lt.t.n_vocab : 0);
            for (int w = 0; w < 3; ++w) D->draft(st, rq, tok, sample ? q.data() : nullptr);
            D->profile_reset();
            std::vector<double> ts;
            for (int r = 0; r < reps; ++r) {
                rq.seed = r;
                const double t1 = now_ms();
                D->draft(st, rq, tok, sample ? q.data() : nullptr);
                ts.push_back(now_ms() - t1);
            }
            std::sort(ts.begin(), ts.end());
            std::printf("\ndraft n=%d%s: min %.2f ms  median %.2f ms  mean %.2f ms  (reps %d)\n", n, sample ? " sampled" : "", ts[0], ts[ts.size() / 2],
                        std::accumulate(ts.begin(), ts.end(), 0.0) / ts.size(), reps);
            D->profile_report(stdout);
        }
        // spec-loop pattern: commit k rows then draft (lazy flush inside the draft)
        {
            DraftRequest rq;
            rq.anchor = 1000; rq.n = 3;
            int32_t tok[8];
            D->profile_reset();
            std::vector<double> ts;
            for (int r = 0; r < reps; ++r) {
                if (D->state_pos(st) + commit_k >= maxctx) break;
                D->commit(st, taps.data(), commit_k);
                const double t1 = now_ms();
                D->draft(st, rq, tok, nullptr);
                ts.push_back(now_ms() - t1);
            }
            std::sort(ts.begin(), ts.end());
            if (!ts.empty()) std::printf("\ncommit(%d rows) + draft n=3: min %.2f ms  median %.2f ms\n", commit_k, ts[0], ts[ts.size() / 2]);
            D->profile_report(stdout);
        }
    }
    return 0;
}

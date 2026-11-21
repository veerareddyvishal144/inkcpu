// Decode throughput benchmark (agent C1). Run under the shared lock:  tools/with_bench_lock.sh build-c1b/engine/bench_decode [options]
//   --threads 8,12,16   thread counts            --ctx 16,1000   context lengths decoded after (prefilled once per ctx, slot_copy per run)
//   --n 64              tokens timed per run     --warm 4        untimed warm-up tokens per run     --reps 2   timed runs per cell (best + all printed)
//   --model PATH        GGUF                     INK_PROFILE=1   per-phase table of the last rep of every cell
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/engine.h"

using namespace ink;
using clk = std::chrono::steady_clock;

static std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}
static std::string load1() {
    FILE* f = std::fopen("/proc/loadavg", "r");
    char b[64] = {0};
    if (f) {
        if (!std::fgets(b, sizeof b, f)) b[0] = 0;
        std::fclose(f);
    }
    char* sp = std::strchr(b, ' ');
    if (sp) *sp = 0;
    return b;
}
static std::vector<int32_t> read_i32(const std::string& p) {
    std::vector<int32_t> v;
    if (FILE* f = std::fopen(p.c_str(), "rb")) {
        int32_t x;
        while (std::fread(&x, 4, 1, f) == 1) v.push_back(x);
        std::fclose(f);
    }
    return v;
}
static int argmax(const float* x, size_t n) {
    size_t b = 0;
    for (size_t i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return (int)b;
}

int main(int argc, char** argv) {
    std::string model = "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf";
    std::vector<int> threads = {8, 12, 16}, ctxs = {16, 1000}, splits;
    int n = 64, warm = 4, reps = 2;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() { return i + 1 < argc ? argv[++i] : (const char*)""; };
        if (a == "--threads") threads = parse_list(val());
        else if (a == "--ctx") ctxs = parse_list(val());
        else if (a == "--n") n = std::atoi(val());
        else if (a == "--warm") warm = std::atoi(val());
        else if (a == "--reps") reps = std::atoi(val());
        else if (a == "--model") model = val();
        else if (a == "--splits") splits = parse_list(val());   // attention key-split granule (tune.split_keys) variants, used for ctx >= 256
    }
    // context text: the oracle prompts concatenated and repeated
    const std::string root = "/data/inkling-mini/engine/tools/oracle/";
    std::vector<int32_t> pool_toks;
    for (const char* nm : {"s1_en_capital", "s2_code_add", "s3_en_story", "m1_en_llm", "m2_code_c", "c1_chat_user"}) {
        auto t = read_i32(root + nm + "/tokens.i32");
        pool_toks.insert(pool_toks.end(), t.begin(), t.end());
    }
    if (pool_toks.empty()) pool_toks = {371, 151451, 220};
    int max_ctx = 0;
    for (int c : ctxs) max_ctx = std::max(max_ctx, c);
    EngineOptions o;
    o.n_threads = threads.empty() ? 12 : threads[0];
    o.n_slots = 2;
    o.max_ctx = max_ctx + warm + n + 8;
    std::string err;
    auto e = Engine::create(model, o, &err);
    if (!e) {
        std::fprintf(stderr, "create failed: %s\n", err.c_str());
        return 2;
    }
    std::printf("loaded in %.1f s, %.2f GB packed; loadavg at start %s\n", e->load_seconds, e->weight_bytes / 1e9, load1().c_str());
    std::vector<float> lg(e->n_vocab());
    const int s0 = e->slot_new(), s1 = e->slot_new();
    std::printf("\n%6s %8s %9s %9s %12s %s\n", "ctx", "threads", "tok/s", "ms/tok", "all reps", "loadavg(before/after)");
    for (int ctx : ctxs) {
        e->slot_reset(s0);
        const auto tp = clk::now();
        for (int i = 0; i < ctx; ++i) e->decode(s0, pool_toks[i % pool_toks.size()], nullptr);
        std::printf("# prefill %d tokens via decode path: %.1f s\n", ctx, std::chrono::duration<double>(clk::now() - tp).count());
        std::vector<int> sp = (ctx >= 256 && !splits.empty()) ? splits : std::vector<int>{e->tune.split_keys};
        for (int th : threads)
          for (int spk : sp) {
            e->tune.split_keys = spk;
            if (e->opt.n_threads != th) e->set_threads(th);
            double best = 0;
            std::string allr;
            const std::string l0 = load1();
            for (int r = 0; r < reps; ++r) {
                e->slot_copy(s0, s1);
                int tok = pool_toks[ctx % pool_toks.size()];
                for (int i = 0; i < warm; ++i) {
                    e->decode(s1, tok, lg.data());
                    tok = argmax(lg.data(), lg.size());
                }
                e->profile_reset();
                const auto t0 = clk::now();
                for (int i = 0; i < n; ++i) {
                    e->decode(s1, tok, lg.data());
                    tok = argmax(lg.data(), lg.size());
                }
                const double s = std::chrono::duration<double>(clk::now() - t0).count();
                const double tps = n / s;
                best = std::max(best, tps);
                char b[32];
                std::snprintf(b, sizeof b, "%.2f ", tps);
                allr += b;
                if (e->profiling() && r + 1 == reps) {
                    std::printf("--- profile ctx %d, %d threads ---\n", ctx, th);
                    e->profile_report(stdout);
                }
            }
            std::printf("%6d %8d %9.2f %9.1f   %-12s %s/%s  split_keys=%d\n", ctx, th, best, 1000.0 / best, allr.c_str(), l0.c_str(), load1().c_str(), spk);
            std::fflush(stdout);
        }
    }
    return 0;
}

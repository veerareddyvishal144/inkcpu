// Decode throughput benchmark (agent C1, extended by E1). Run under the shared lock:  tools/with_bench_lock.sh build-e1/src/engine/bench_decode [options]
//   --threads 8,12,16   thread counts            --ctx 16,1000   context lengths decoded after (built once per ctx, slot_copy per run)
//   --n 64              tokens timed per run     --warm 4        untimed warm-up tokens per run     --reps 2   timed runs per cell (best + all printed)
//   --model PATH        GGUF                     INK_PROFILE=1   per-phase table of the last rep of every cell
//   --prefill 1|0       build the context with Engine::forward (batched prefill, default) or token by token through decode()
//   --tokens FILE       int32 token file for the context (default: the oracle prompts concatenated and repeated)
//   --dump PREFIX       write the logits of every timed token of the FIRST rep of every cell to PREFIX_ctx<ctx>_t<threads>.f32 (bit-identity checks)
//   --splits 384,...    attention key-split granule variants (tune.split_keys), used for ctx >= 256
#include <dirent.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/engine.h"
#include "../src/engine/engine_priv.h"   // engine_fine_report (INK_FINE=1)

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
static std::string rss_mb() {   // VmRSS of this process in MB (hugepage impact of the KV mappings)
    FILE* f = std::fopen("/proc/self/status", "r");
    char line[256];
    std::string r = "?";
    while (f && std::fgets(line, sizeof line, f))
        if (!std::strncmp(line, "VmRSS:", 6)) {
            r = std::to_string(std::atol(line + 6) / 1024) + " MB";
            break;
        }
    if (f) std::fclose(f);
    return r;
}
static long ctxt_switches(bool nonvoluntary) {   // sum over all threads of this process (preemptions of the spin-barrier threads by other tenants)
    long sum = 0;
    if (DIR* d = opendir("/proc/self/task")) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            char p[256];
            std::snprintf(p, sizeof p, "/proc/self/task/%s/status", e->d_name);
            FILE* f = std::fopen(p, "r");
            char line[256];
            while (f && std::fgets(line, sizeof line, f))
                if (!std::strncmp(line, nonvoluntary ? "nonvoluntary_ctxt_switches:" : "voluntary_ctxt_switches:", nonvoluntary ? 27 : 24))
                    sum += std::atol(std::strchr(line, ':') + 1);
            if (f) std::fclose(f);
        }
        closedir(d);
    }
    return sum;
}
static int argmax(const float* x, size_t n) {
    size_t b = 0;
    for (size_t i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return (int)b;
}

int main(int argc, char** argv) {
    std::string model = "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf";
    std::string tokfile, dump;
    std::vector<int> threads = {8, 12, 16}, ctxs = {16, 1000}, splits, granules;
    int n = 64, warm = 4, reps = 2, use_prefill = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() { return i + 1 < argc ? argv[++i] : (const char*)""; };
        if (a == "--threads") threads = parse_list(val());
        else if (a == "--ctx") ctxs = parse_list(val());
        else if (a == "--n") n = std::atoi(val());
        else if (a == "--warm") warm = std::atoi(val());
        else if (a == "--reps") reps = std::atoi(val());
        else if (a == "--model") model = val();
        else if (a == "--prefill") use_prefill = std::atoi(val());
        else if (a == "--tokens") tokfile = val();
        else if (a == "--dump") dump = val();
        else if (a == "--splits") splits = parse_list(val());   // attention key-split granule (tune.split_keys) variants, used for ctx >= 256
        else if (a == "--granules") granules = parse_list(val());   // attn_set_split_granule variants (0 = legacy rule), used for ctx >= 256
    }
    // context text: a token file, else the oracle prompts concatenated and repeated
    const std::string root = "/data/inkling-mini/engine/tools/oracle/";
    std::vector<int32_t> pool_toks;
    if (!tokfile.empty()) pool_toks = read_i32(tokfile);
    if (pool_toks.empty())
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
    std::printf("loaded in %.1f s, %.2f GB packed; loadavg at start %s; context source %s (%zu tokens, repeated)\n", e->load_seconds,
                e->weight_bytes / 1e9, load1().c_str(), tokfile.empty() ? "oracle prompts" : tokfile.c_str(), pool_toks.size());
    std::vector<float> lg(e->n_vocab());
    const int s0 = e->slot_new(), s1 = e->slot_new();
    std::printf("\n%6s %8s %9s %9s %12s %s\n", "ctx", "threads", "tok/s", "ms/tok", "all reps", "loadavg(before/after)");
    for (int ctx : ctxs) {
        e->slot_reset(s0);
        const auto tp = clk::now();
        std::vector<int32_t> ctoks((size_t)ctx);
        for (int i = 0; i < ctx; ++i) ctoks[i] = pool_toks[i % pool_toks.size()];
        if (use_prefill && ctx > 0) {
            const int rc = e->forward(s0, ctoks.data(), ctx, nullptr);
            if (rc) {
                std::fprintf(stderr, "forward failed: %d\n", rc);
                return 3;
            }
        } else {
            for (int i = 0; i < ctx; ++i) e->decode(s0, ctoks[i], nullptr);
        }
        const double ps = std::chrono::duration<double>(clk::now() - tp).count();
        std::printf("# context of %d tokens built via %s: %.1f s (%.1f tok/s), loadavg %s, RSS %s\n", ctx, use_prefill ? "Engine::forward (prefill)" : "decode path", ps,
                    ctx / std::max(ps, 1e-9), load1().c_str(), rss_mb().c_str());
        std::vector<int> sp = (ctx >= 256 && !splits.empty()) ? splits : std::vector<int>{e->tune.split_keys};
        std::vector<int> gr = (ctx >= 256 && !granules.empty()) ? granules : std::vector<int>{attn_set_split_granule(-1)};
        for (int th : threads)
          for (int spk : sp)
          for (int gran : gr) {
            e->tune.split_keys = spk;
            attn_set_split_granule(gran);
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
                FILE* df = nullptr;
                if (!dump.empty() && r == 0) {
                    char nm[512];
                    if (gr.size() > 1 || sp.size() > 1)
                        std::snprintf(nm, sizeof nm, "%s_ctx%d_t%d_s%d_g%d.f32", dump.c_str(), ctx, th, spk, gran);
                    else
                        std::snprintf(nm, sizeof nm, "%s_ctx%d_t%d.f32", dump.c_str(), ctx, th);
                    df = std::fopen(nm, "wb");
                }
                if (r == 0) e->profile_reset();   // the per-phase table averages over ALL reps of the cell (less rep-to-rep noise)
                const long nv0 = ctxt_switches(true), vol0 = ctxt_switches(false);
                const auto t0 = clk::now();
                for (int i = 0; i < n; ++i) {
                    e->decode(s1, tok, lg.data());
                    if (df) std::fwrite(lg.data(), sizeof(float), lg.size(), df);
                    tok = argmax(lg.data(), lg.size());
                }
                const double s = std::chrono::duration<double>(clk::now() - t0).count();
                std::printf("# rep %d: %ld nonvoluntary + %ld voluntary context switches over all threads in %.2f s (%.0f preemptions/s)\n", r,
                            ctxt_switches(true) - nv0, ctxt_switches(false) - vol0, s, (ctxt_switches(true) - nv0) / s);
                if (df) std::fclose(df);
                const double tps = n / s;
                best = std::max(best, tps);
                char b[32];
                std::snprintf(b, sizeof b, "%.2f ", tps);
                allr += b;
                if (e->profiling() && r + 1 == reps) {
                    std::printf("--- profile ctx %d, %d threads, loadavg %s (all %d reps) ---\n", ctx, th, load1().c_str(), reps);
                    e->profile_report(stdout);
                    engine_fine_report(stdout, e->prof.tsc_per_us);
                }
            }
            std::printf("%6d %8d %9.2f %9.1f   %-12s %s/%s  split_keys=%d granule=%d\n", ctx, th, best, 1000.0 / best, allr.c_str(), l0.c_str(), load1().c_str(), spk, gran);
            std::fflush(stdout);
        }
    }
    return 0;
}

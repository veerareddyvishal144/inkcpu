// Prefill throughput benchmark (agent P). Run under the shared lock:  tools/with_bench_lock.sh build-p/src/engine/bench_prefill [options]
//   --threads 12,16     thread counts            --prompts 16,64,256,1024   prompt lengths (tokens)      --reps 2    timed runs per cell (best + all printed)
//   --chunk 256         chunk size               --model PATH                --decode N  also time N decode tokens after the prefill (baseline)
//   --attn decode       attention through the decode kernels                  INK_PF_PROFILE=1  per-phase table of the last rep of every cell
// Prints tok/s per cell with the load average before and after (numbers at load > 2 are indicative only).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/engine.h"
#include "ink/engine_prefill.h"

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

int main(int argc, char** argv) {
    std::string model = "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf";
    std::vector<int> threads = {12, 16}, prompts = {16, 64, 256, 1024};
    int reps = 2, chunk = 256, ndecode = 0;
    bool dk = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() { return i + 1 < argc ? argv[++i] : (const char*)""; };
        if (a == "--threads") threads = parse_list(val());
        else if (a == "--prompts") prompts = parse_list(val());
        else if (a == "--reps") reps = std::atoi(val());
        else if (a == "--chunk") chunk = std::atoi(val());
        else if (a == "--model") model = val();
        else if (a == "--decode") ndecode = std::atoi(val());
        else if (a == "--attn") dk = std::string(val()) == "decode";
    }
    // real text for the prompts: the 1500-token wikitext data of the engine tests, repeated if needed
    std::vector<int32_t> text = read_i32("/data/inkling-mini/engine/build-c1b/data/ring/tokens.i32");
    if (text.size() < 64) text = read_i32("/data/inkling-mini/engine/tools/oracle/m1_en_llm/tokens.i32");
    int maxp = 0;
    for (int p : prompts) maxp = std::max(maxp, p);
    EngineOptions o;
    o.n_threads = threads.empty() ? 12 : threads[0];
    o.n_slots = 1;
    o.max_ctx = std::max(4096, maxp + 64);
    std::string err;
    auto e = Engine::create(model, o, &err);
    if (!e) {
        std::fprintf(stderr, "create failed: %s\n", err.c_str());
        return 2;
    }
    const int slot = e->slot_new();
    prefill_tune().chunk = chunk;
    prefill_tune().attn_decode_kernels = dk;
    std::vector<float> lg(e->n_vocab());
    std::printf("bench_prefill: chunk %d, attention %s, reps %d; load at start %s\n", chunk, dk ? "decode kernels" : "prefill kernel", reps, load1().c_str());
    std::printf("%-8s %-7s %-9s %-9s %-24s %s\n", "threads", "prompt", "best s", "tok/s", "all reps (tok/s)", "load before/after");
    for (int nth : threads) {
        if (e->opt.n_threads != nth) e->set_threads(nth);
        // warm-up (first touch of the scratch, TLB, page cache)
        {
            std::vector<int32_t> w(text.begin(), text.begin() + 32);
            e->slot_reset(slot);
            engine_prefill(*e, slot, w.data(), (int)w.size(), lg.data());
        }
        for (int p : prompts) {
            std::vector<int32_t> toks;
            for (int i = 0; i < p; ++i) toks.push_back(text[(size_t)i % text.size()]);
            double best = 1e30;
            std::string all;
            const std::string l0 = load1();
            for (int r = 0; r < reps; ++r) {
                e->slot_reset(slot);
                engine_prefill_profile_reset(*e);
                const auto t0 = clk::now();
                const int rc = engine_prefill(*e, slot, toks.data(), p, lg.data());
                const double s = std::chrono::duration<double>(clk::now() - t0).count();
                if (rc) {
                    std::fprintf(stderr, "prefill rc %d\n", rc);
                    return 3;
                }
                best = std::min(best, s);
                char b[32];
                std::snprintf(b, sizeof b, "%.1f ", p / s);
                all += b;
                if (r + 1 == reps && (e->prof_on || std::getenv("INK_PF_PROFILE"))) engine_prefill_profile_report(*e, stdout);
            }
            std::printf("%-8d %-7d %-9.3f %-9.1f %-24s %s/%s\n", nth, p, best, p / best, all.c_str(), l0.c_str(), load1().c_str());
            std::fflush(stdout);
        }
        if (ndecode > 0) {   // decode baseline on the context left by the last prefill
            const int32_t tk = text[0];
            const auto t0 = clk::now();
            for (int i = 0; i < ndecode; ++i) e->decode(slot, tk, lg.data());
            const double s = std::chrono::duration<double>(clk::now() - t0).count();
            std::printf("  decode baseline at %d threads: %d tokens in %.2f s = %.2f tok/s\n", nth, ndecode, s, ndecode / s);
        }
    }
    return 0;
}

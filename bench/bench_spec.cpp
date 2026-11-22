// bench_spec (agent S2): plain vs fixed T=4 vs fixed T=8 vs auto on the DSpark prompts with the real drafter (S1) or a test double.
//   INK_CPUS=... build/src/spec/bench_spec [--n 80] [--modes plain,4,8,auto] [--prompts name,name|all] [--drafter PATH|oracle|none] [--exact -1|0|1] [--temp 0.0] [--seed 1]
// Every mode starts from a fresh prefill of the prompt (the tap hook feeds the drafter during the prefill, as in the server) and generates n tokens greedily (or sampled with --temp).
// Reports per prompt and mode: tok/s of the generation (wall), L = tokens per speculative step, acceptance, mean draft / verify / plain step times, the T histogram (auto), and the
// number of tokens that differ from the plain run (0 expected in exact mode and for greedy agreement; the fast attention kernel may flip low-margin tokens above 768 keys).
// Timings on the shared host are indicative (load printed).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "spec_test_util.h"

using namespace sut;

namespace {
struct Row {
    std::string mode;
    int tokens = 0, steps = 0, spec_steps = 0, drafted = 0, accepted = 0, mismatches = 0, hist[9] = {};
    double wall = 0, draft_us = 0, verify_us = 0, accept_us = 0, plain_us = 0, prefill_s = 0;
    double L() const { return spec_steps ? (double)(tokens - (steps - spec_steps)) / spec_steps : 0; }
    double toks() const { return tokens / std::max(wall, 1e-9); }
};
int mode_id(const std::string& m) {
    if (m == "plain" || m == "off" || m == "0") return 0;
    if (m == "auto") return 1;
    return std::atoi(m.c_str());
}
std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) v.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    int n = 80, exact = -1, seed = 1;
    float temp = 0.0f;
    std::string modes = "plain,4,8,auto", prompts = "all", drafter_path = "/data/inkling-mini/models/dspark-Inkling-2-Mini-Q8_0.gguf";
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* d = "") { return i + 1 < argc ? std::string(argv[++i]) : std::string(d); };
        if (!std::strcmp(argv[i], "--n")) n = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--modes")) modes = next();
        else if (!std::strcmp(argv[i], "--prompts")) prompts = next();
        else if (!std::strcmp(argv[i], "--drafter")) drafter_path = next();
        else if (!std::strcmp(argv[i], "--exact")) exact = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--temp")) temp = (float)std::atof(next().c_str());
        else if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(next().c_str());
    }
    auto Ep = make_engine(1, 4096);
    Engine& E = *Ep;
    const int A = E.slot_new();
    const int V = E.n_vocab();
    std::vector<std::string> names = prompts == "all" ? prompt_names() : split(prompts);
    std::vector<std::string> mode_list = split(modes);
    std::printf("bench_spec: n=%d modes=%s exact=%d temp=%.2f threads=%d load %s drafter=%s\n", n, modes.c_str(), exact, temp, E.opt.n_threads, load1().c_str(), drafter_path.c_str());

    // the drafter: real (S1), oracle double (upper bound: needs the plain tokens, filled per prompt), or none
    std::unique_ptr<Drafter> dr;
    DoubleDrafter* oracle = nullptr;
    if (drafter_path == "oracle") {
        auto d = std::make_unique<DoubleDrafter>(DoubleDrafter::ORACLE, V);
        oracle = d.get();
        dr = std::move(d);
    } else if (drafter_path != "none") {
#ifdef INK_HAVE_DRAFTER
        std::string err;
        const double t0 = now_s();
        dr = Drafter::create(E, drafter_path, &err);
        if (!dr) {
            std::fprintf(stderr, "drafter load failed: %s\n", err.c_str());
            return 1;
        }
        std::printf("drafter loaded in %.1f s\n", now_s() - t0);
#else
        std::fprintf(stderr, "built without the drafter module\n");
        return 1;
#endif
    }
    SpecOptions so;
    so.mode = 1;
    so.exact = exact;
    SpecDecoder dec(E, std::move(dr), so);
    std::vector<float> lg((size_t)V);
    std::map<std::string, std::vector<Row>> all;
    std::vector<int32_t> plain_tokens;
    for (const std::string& name : names) {
        std::vector<int32_t> pr = prompt(name);
        if (pr.empty()) {
            std::printf("%s: no prompt file\n", name.c_str());
            continue;
        }
        std::printf("\n%s (p = %zu), load %s\n", name.c_str(), pr.size(), load1().c_str());
        std::printf("  %-6s %8s %6s %7s %9s %9s %9s %9s  %s\n", "mode", "tok/s", "L", "accept", "plain ms", "draft ms", "verify ms", "accept ms", "T hist / mismatches vs plain");
        for (const std::string& m : mode_list) {
            const int mode = mode_id(m);
            Row r;
            r.mode = m;
            E.slot_reset(A);
            double t0 = now_s();
            if (E.forward(A, pr.data(), (int)pr.size(), lg.data()) != 0) {
                std::printf("  prefill failed\n");
                break;
            }
            r.prefill_s = now_s() - t0;
            if (oracle) {   // the oracle needs the true continuation: plain run first
                oracle->truth = pr;
                oracle->truth.insert(oracle->truth.end(), plain_tokens.begin(), plain_tokens.end());
                oracle->set_pos(A, (int64_t)pr.size());
            }
            SamplingParams sp;
            sp.temperature = temp;
            SpecRng rng((uint64_t)seed);
            int32_t anchor = temp > 0 ? Sampler(V).sample(lg.data(), sp, rng) : argmax(lg.data(), V);
            std::vector<int32_t> out;
            out.push_back(anchor);
            uint64_t s = rng.next_seed();
            t0 = now_s();
            while ((int)out.size() - 1 < n) {
                SpecRequest rq;
                rq.anchor = anchor;
                rq.sp = sp;
                rq.seed = s;
                rq.mode = mode;
                rq.max_emit = 8;
                SpecResult res;
                const int rc = dec.step(A, rq, &res);
                if (rc) {
                    std::printf("  step failed rc %d\n", rc);
                    break;
                }
                ++r.steps;
                ++r.hist[res.T];
                if (res.T > 1) {
                    ++r.spec_steps;
                    r.drafted += res.drafted;
                    r.accepted += res.accepted;
                    r.draft_us += res.draft_us;
                    r.verify_us += res.verify_us;
                    r.accept_us += res.accept_us;
                } else {
                    r.plain_us += res.verify_us + res.accept_us;
                }
                for (int i = 0; i < res.n_tokens; ++i) out.push_back(res.tokens[i]);
                anchor = res.tokens[res.n_tokens - 1];
                s = res.next_seed;
            }
            r.wall = now_s() - t0;
            r.tokens = (int)out.size() - 1;
            if (mode == 0) plain_tokens.assign(out.begin() + 1, out.end());
            else
                for (size_t i = 1; i < out.size() && i - 1 < plain_tokens.size(); ++i) r.mismatches += out[i] != plain_tokens[i - 1];
            const int n_plain = r.steps - r.spec_steps;
            std::printf("  %-6s %8.2f %6.2f %7s %9.0f %9.0f %9.0f %9.1f  ", m.c_str(), r.toks(), r.L(), r.drafted ? (std::to_string(r.accepted) + "/" + std::to_string(r.drafted)).c_str() : "-",
                        n_plain ? r.plain_us / n_plain / 1e3 : 0.0, r.spec_steps ? r.draft_us / r.spec_steps / 1e3 : 0.0, r.spec_steps ? r.verify_us / r.spec_steps / 1e3 : 0.0,
                        r.spec_steps ? r.accept_us / r.spec_steps / 1e3 : 0.0);
            for (int T = 1; T <= 8; ++T)
                if (r.hist[T]) std::printf("%d:%d ", T, r.hist[T]);
            if (mode != 0) std::printf(" mismatches %d/%d", r.mismatches, r.tokens);
            std::printf("  (prefill %.1f s, gen %.1f s)\n", r.prefill_s, r.wall);
            std::fflush(stdout);
            all[m].push_back(r);
        }
    }
    std::printf("\npooled over %zu prompts (n = %d each), load %s:\n  %-6s %8s %6s %7s %s\n", names.size(), n, load1().c_str(), "mode", "tok/s", "L", "accept", "mismatches");
    for (const std::string& m : mode_list) {
        double tok = 0, wall = 0, Lsum = 0;
        int nL = 0, acc = 0, dr_ = 0, mm = 0;
        for (const Row& r : all[m]) {
            tok += r.tokens;
            wall += r.wall;
            if (r.spec_steps) {
                Lsum += r.L();
                ++nL;
            }
            acc += r.accepted;
            dr_ += r.drafted;
            mm += r.mismatches;
        }
        std::printf("  %-6s %8.2f %6.2f %7s %d\n", m.c_str(), tok / std::max(wall, 1e-9), nL ? Lsum / nL : 0.0, dr_ ? (std::to_string(acc) + "/" + std::to_string(dr_)).c_str() : "-", mm);
    }
    return 0;
}

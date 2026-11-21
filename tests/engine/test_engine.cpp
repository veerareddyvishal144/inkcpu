// Validation gates for the decode engine (agent C1). Usage: test_engine <mode> [args]
//   oracle                 gate (a): 6 prompts token by token vs llama.cpp logits_last (argmax, top-10 overlap, cosine) + cosine vs RefModel
//   greedy                 gate (b): 5 prompts x 64 greedy tokens vs tools/oracle/greedy (first divergence + llama.cpp margin)
//   nll <dirA> [dirB]      gate (c): mean NLL over 256-token chunks (dir has tokens.i32 + logits_all.f32 from oracle_logits --all-logits)
//   determinism            gate (d): bit-identical logits across repeats and thread counts 8/12/16
//   slots [ring_dir]       gate (e): reset/continue, interleaved slots, ring wrap (ring_dir: tokens.i32 + logits_last.f32 of >1024 tokens)
//   trace [ntok]           per-visit comparison of Q/K/V/R/attn_out/ffn_inp/l_out against the scalar reference
// env: INK_GGUF (model path), INK_THREADS (default 12), INK_REFCACHE (dir for cached RefModel logits)
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ink/engine.h"
#include "ref_forward.h"

using namespace ink;

namespace {

const std::string kRoot = INK_SOURCE_DIR;
std::string gguf_path() {
    const char* e = std::getenv("INK_GGUF");
    return e ? e : "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf";
}
int env_threads() {
    const char* e = std::getenv("INK_THREADS");
    return e ? std::atoi(e) : 12;
}
std::string refcache_dir() {
    const char* e = std::getenv("INK_REFCACHE");
    return e ? e : kRoot + "/build-c1/refcache";
}

template <class T>
std::vector<T> read_file(const std::string& p) {
    std::vector<T> v;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)n / sizeof(T));
    if (std::fread(v.data(), sizeof(T), v.size(), f) != v.size()) v.clear();
    std::fclose(f);
    return v;
}
void write_f32(const std::string& p, const std::vector<float>& v) {
    FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) return;
    std::fwrite(v.data(), 4, v.size(), f);
    std::fclose(f);
}
double cosine(const float* a, const float* b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; ++i) {
        ab += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    return ab / std::sqrt(aa * bb + 1e-300);
}
int argmax(const float* x, size_t n) {
    size_t b = 0;
    for (size_t i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return (int)b;
}
std::vector<int> topk(const float* x, size_t n, int k) {
    std::vector<int> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    idx.resize(k);
    return idx;
}
int overlap(const float* a, const float* b, size_t n, int k) {
    auto ta = topk(a, n, k), tb = topk(b, n, k);
    int c = 0;
    for (int x : ta)
        for (int y : tb)
            if (x == y) ++c;
    return c;
}
double logsumexp(const float* x, size_t n) {
    double m = x[0];
    for (size_t i = 1; i < n; ++i) m = std::max<double>(m, x[i]);
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += std::exp((double)x[i] - m);
    return m + std::log(s);
}

std::unique_ptr<Engine> make_engine(int threads, int slots, int max_ctx) {
    EngineOptions o;
    o.n_threads = threads;
    o.n_slots = slots;
    o.max_ctx = max_ctx;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    auto e = Engine::create(gguf_path(), o, &err);
    if (!e) {
        std::fprintf(stderr, "engine create failed: %s\n", err.c_str());
        std::exit(2);
    }
    std::printf("engine loaded in %.1f s (%.2f GB packed, %d threads, %d slots)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), e->weight_bytes / 1e9, threads, slots);
    return e;
}

struct RefHolder {
    std::unique_ptr<GgufFile> g;
    std::unique_ptr<RefModel> m;
    RefModel& get() {
        if (!m) {
            std::string err;
            g = GgufFile::open(gguf_path(), &err);
            RefOptions ro;
            ro.n_threads = 8;
            m.reset(new RefModel(*g, ro));
        }
        return *m;
    }
};
RefHolder g_ref;

std::vector<float> ref_logits_cached(const std::string& name, const std::vector<int32_t>& toks, bool compute = false) {
    const std::string dir = refcache_dir();
    ::mkdir(dir.c_str(), 0755);
    const std::string p = dir + "/" + name + ".f32";
    auto v = read_file<float>(p);
    if (!v.empty()) return v;
    if (!compute && !std::getenv("INK_REFWAIT")) return v;   // run `test_engine refcache` (slow, background) to fill the cache
    std::printf("  [ref] computing RefModel logits for %s (%zu tokens, ~1 s/token)...\n", name.c_str(), toks.size());
    std::fflush(stdout);
    RefModel& m = g_ref.get();
    RefState st = m.new_state();
    v = m.run(st, toks);
    write_f32(p, v);
    return v;
}

std::vector<float> run_prompt(Engine& e, int slot, const std::vector<int32_t>& toks) {
    std::vector<float> lg(e.n_vocab());
    e.slot_reset(slot);
    for (size_t i = 0; i < toks.size(); ++i)
        if (e.decode(slot, toks[i], i + 1 == toks.size() ? lg.data() : nullptr)) {
            std::fprintf(stderr, "decode failed at %zu\n", i);
            std::exit(3);
        }
    return lg;
}

// ------------------------------------------------------------------------------------------------ gate (a)
int mode_oracle() {
    auto e = make_engine(env_threads(), 1, 4096);
    const int slot = e->slot_new();
    const char* names[] = {"s1_en_capital", "s2_code_add", "s3_en_story", "c1_chat_user", "m1_en_llm", "m2_code_c"};
    int fails = 0;
    std::printf("%-15s %4s %9s %9s %6s %10s %10s %12s\n", "prompt", "ntok", "argmax(o/e)", "", "top10", "cos(oracle)", "cos(ref)", "cos(ref,oracle)");
    for (const char* nm : names) {
        const std::string dir = kRoot + "/tools/oracle/" + nm;
        auto toks = read_file<int32_t>(dir + "/tokens.i32");
        auto orc = read_file<float>(dir + "/logits_last.f32");
        if (toks.empty() || orc.empty()) {
            std::printf("%s: oracle data missing\n", nm);
            ++fails;
            continue;
        }
        const size_t nv = orc.size();
        auto lg = run_prompt(*e, slot, toks);
        auto rf = ref_logits_cached(nm, toks);
        if (rf.size() != nv) rf.assign(nv, 0.0f);   // not cached yet: cos(ref) prints as 0
        const int ao = argmax(orc.data(), nv), ae = argmax(lg.data(), nv);
        const int ov = overlap(lg.data(), orc.data(), nv, 10);
        const double c1 = cosine(lg.data(), orc.data(), nv), c2 = cosine(lg.data(), rf.data(), nv), c3 = cosine(rf.data(), orc.data(), nv);
        const bool ok = ao == ae && ov >= 9 && c1 >= 0.999;
        std::printf("%-15s %4zu %9d %9d %6d %10.6f %10.6f %12.6f  %s\n", nm, toks.size(), ao, ae, ov, c1, c2, c3, ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
    }
    std::printf("gate (a) %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}

// ------------------------------------------------------------------------------------------------ gate (b)
int mode_greedy() {
    auto e = make_engine(env_threads(), 1, 4096);
    const int slot = e->slot_new();
    const char* names[] = {"s1_en_capital", "s2_code_add", "m1_en_llm", "m2_code_c", "c1_chat_user"};
    int fails = 0;
    for (const char* nm : names) {
        const std::string dir = kRoot + "/tools/oracle/greedy/" + nm;
        auto prompt = read_file<int32_t>(dir + "/prompt_tokens.i32");
        auto gen = read_file<int32_t>(dir + "/gen_tokens.i32");
        std::vector<double> margin(gen.size(), 1e9);
        if (FILE* f = std::fopen((dir + "/steps.txt").c_str(), "r")) {
            char line[4096];
            while (std::fgets(line, sizeof line, f)) {
                int step, tok, t2;
                double l1, l2, mg;
                if (std::sscanf(line, "%d %d %lf %d %lf %lf", &step, &tok, &l1, &t2, &l2, &mg) == 6 && step >= 0 && step < (int)gen.size()) margin[step] = mg;
            }
            std::fclose(f);
        }
        if (prompt.empty() || gen.empty()) {
            std::printf("%s: data missing\n", nm);
            ++fails;
            continue;
        }
        auto lg = run_prompt(*e, slot, prompt);
        int div = -1;
        for (size_t s = 0; s < gen.size(); ++s) {
            const int t = argmax(lg.data(), lg.size());
            if (t != gen[s]) {
                div = (int)s;
                break;
            }
            if (s + 1 < gen.size()) e->decode(slot, t, lg.data());
        }
        if (div < 0) {
            std::printf("%-15s %zu/%zu tokens identical (no divergence)  PASS\n", nm, gen.size(), gen.size());
        } else {
            const bool bad = margin[div] >= 2.0;
            std::printf("%-15s first divergence at step %d (llama.cpp top1-top2 margin %.3f)  %s\n", nm, div, margin[div], bad ? "FAIL" : "ok (near-tie)");
            if (bad) ++fails;
        }
    }
    std::printf("gate (b) %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}

// ------------------------------------------------------------------------------------------------ gate (c)
int mode_nll(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "nll <dirA> [dirB]\n");
        return 2;
    }
    auto e = make_engine(env_threads(), 1, 4096);
    const int slot = e->slot_new();
    const size_t nv = e->n_vocab();
    int fails = 0;
    for (int a = 2; a < argc; ++a) {
        const std::string dir = argv[a];
        auto toks = read_file<int32_t>(dir + "/tokens.i32");
        auto all = read_file<float>(dir + "/logits_all.f32");
        if (toks.size() < 2 || all.size() != toks.size() * nv) {
            std::printf("%s: bad data (tokens %zu, logits %zu)\n", dir.c_str(), toks.size(), all.size());
            ++fails;
            continue;
        }
        e->slot_reset(slot);
        std::vector<float> lg(nv);
        double nll_e = 0, nll_l = 0;
        const size_t n = toks.size() - 1;
        for (size_t i = 0; i < n; ++i) {
            e->decode(slot, toks[i], lg.data());
            const int nx = toks[i + 1];
            nll_e += logsumexp(lg.data(), nv) - lg[nx];
            nll_l += logsumexp(all.data() + i * nv, nv) - all[i * nv + nx];
        }
        nll_e /= n;
        nll_l /= n;
        const bool ok = std::fabs(nll_e - nll_l) <= 0.01;
        std::printf("%s: %zu positions  NLL engine %.5f  llama.cpp %.5f  |delta| %.5f  %s\n", dir.c_str(), n, nll_e, nll_l, std::fabs(nll_e - nll_l), ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
    }
    std::printf("gate (c) %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}

// ------------------------------------------------------------------------------------------------ gate (d)
int mode_determinism() {
    auto e = make_engine(16, 1, 4096);
    const int slot = e->slot_new();
    auto toks = read_file<int32_t>(kRoot + "/tools/oracle/m1_en_llm/tokens.i32");
    toks.resize(std::min<size_t>(toks.size(), 40));
    const size_t nv = e->n_vocab();
    std::vector<float> first;
    int fails = 0;
    const int cfgs[][2] = {{16, 0}, {16, 1}, {12, 0}, {8, 0}, {8, 1}, {5, 0}};
    for (auto& cf : cfgs) {
        if (e->opt.n_threads != cf[0]) e->set_threads(cf[0]);
        e->slot_reset(slot);
        std::vector<float> all(toks.size() * nv);
        for (size_t i = 0; i < toks.size(); ++i) e->decode(slot, toks[i], all.data() + i * nv);
        if (first.empty()) first = all;
        const bool same = std::memcmp(first.data(), all.data(), all.size() * 4) == 0;
        double md = 0;
        for (size_t i = 0; i < all.size(); ++i) md = std::max<double>(md, std::fabs((double)all[i] - first[i]));
        std::printf("threads %2d run %d: %s (max |diff| vs first %.3g over %zu positions)\n", cf[0], cf[1], same ? "bit-identical" : "DIFFERENT", md, toks.size());
        if (!same) ++fails;
    }
    std::printf("gate (d) %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}

// ------------------------------------------------------------------------------------------------ gate (e)
int mode_slots(int argc, char** argv) {
    auto e = make_engine(env_threads(), 3, 4096);
    const size_t nv = e->n_vocab();
    int fails = 0;
    auto t1 = read_file<int32_t>(kRoot + "/tools/oracle/s1_en_capital/tokens.i32");
    auto t2 = read_file<int32_t>(kRoot + "/tools/oracle/s2_code_add/tokens.i32");
    const int a = e->slot_new(), b = e->slot_new();
    auto L1 = run_prompt(*e, a, t1);
    auto L2 = run_prompt(*e, b, t2);
    auto chk = [&](const char* what, const std::vector<float>& x, const std::vector<float>& y) {
        const bool same = std::memcmp(x.data(), y.data(), nv * 4) == 0;
        std::printf("  %-52s %s\n", what, same ? "bit-identical PASS" : "DIFFERENT FAIL");
        if (!same) ++fails;
    };
    // reset + rerun
    auto L1b = run_prompt(*e, a, t1);
    chk("reset + rerun s1 in the same slot", L1, L1b);
    // continue: split the prompt across two calls
    e->slot_reset(a);
    std::vector<float> L1c(nv);
    e->forward(a, t1.data(), 7, nullptr);
    e->forward(a, t1.data() + 7, (int)t1.size() - 7, L1c.data());
    chk("continue (7 + 6 tokens) == one pass", L1, L1c);
    // interleaved slots
    e->slot_reset(a);
    e->slot_reset(b);
    std::vector<float> I1(nv), I2(nv);
    for (size_t i = 0; i < std::max(t1.size(), t2.size()); ++i) {
        if (i < t1.size()) e->decode(a, t1[i], i + 1 == t1.size() ? I1.data() : nullptr);
        if (i < t2.size()) e->decode(b, t2[i], i + 1 == t2.size() ? I2.data() : nullptr);
    }
    chk("interleaved slot A == isolated s1", L1, I1);
    chk("interleaved slot B == isolated s2", L2, I2);
    // slot_copy
    const int c = e->slot_new();
    e->slot_reset(a);
    e->forward(a, t1.data(), 7, nullptr);
    e->slot_copy(a, c);
    std::vector<float> C1(nv);
    e->forward(c, t1.data() + 7, (int)t1.size() - 7, C1.data());
    chk("slot_copy at 7 tokens then continue == isolated s1", L1, C1);
    std::printf("  ink_slot_pos check: %lld (expect %zu)\n", (long long)e->slot_pos(c), t1.size());

    // ring wrap: > 1024 tokens
    std::string rdir = argc > 2 ? argv[2] : kRoot + "/build-c1/ring";
    auto rt = read_file<int32_t>(rdir + "/tokens.i32");
    auto rl = read_file<float>(rdir + "/logits_last.f32");
    if (rt.size() > 1030) {
        auto t0 = std::chrono::steady_clock::now();
        e->slot_reset(a);
        std::vector<float> lg(nv);
        for (size_t i = 0; i < rt.size(); ++i) e->decode(a, rt[i], i + 1 == rt.size() ? lg.data() : nullptr);
        bool finite = true;
        float mx = 0;
        for (size_t i = 0; i < nv; ++i) {
            if (!std::isfinite(lg[i])) finite = false;
            mx = std::max(mx, std::fabs(lg[i]));
        }
        std::printf("  ring wrap: %zu tokens decoded in %.1f s, logits finite=%d max|logit|=%.2f\n", rt.size(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), finite, mx);
        if (!finite || mx > 31) ++fails;
        if (rl.size() == nv) {
            const double cs = cosine(lg.data(), rl.data(), nv);
            const int ov = overlap(lg.data(), rl.data(), nv, 10);
            std::printf("  ring wrap vs llama.cpp logits_last: argmax %d/%d top10 overlap %d cosine %.6f %s\n", argmax(rl.data(), nv), argmax(lg.data(), nv), ov, cs,
                        (cs >= 0.99 && argmax(rl.data(), nv) == argmax(lg.data(), nv)) ? "PASS" : "CHECK");
            if (cs < 0.99) ++fails;
        }
        // determinism of the long run across thread counts
        std::vector<float> lg2(nv);
        e->set_threads(8);
        e->slot_reset(a);
        for (size_t i = 0; i < rt.size(); ++i) e->decode(a, rt[i], i + 1 == rt.size() ? lg2.data() : nullptr);
        chk("ring-wrap run, 8 threads == 12 threads", lg, lg2);
    } else {
        std::printf("  ring wrap: no token file in %s (need > 1030 tokens), skipped\n", rdir.c_str());
    }
    std::printf("gate (e) %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}

// ------------------------------------------------------------------------------------------------ trace vs RefModel
int mode_trace(int argc, char** argv) {
    const int ntok = argc > 2 ? std::atoi(argv[2]) : 2;
    auto e = make_engine(env_threads(), 1, 1024);
    const int slot = e->slot_new();
    auto toks = read_file<int32_t>(kRoot + "/tools/oracle/s1_en_capital/tokens.i32");
    RefModel& rm = g_ref.get();
    std::map<int, int> il2v;
    for (const Visit& v : e->plan) il2v[v.llama_slot] = v.index;
    const char* names[] = {"Qcur", "Kcur", "Vcur", "Rcur", "attn_out", "ffn_inp", "l_out"};
    typedef std::map<std::pair<std::string, int>, std::vector<float>> Cap;
    Cap rc, ec;
    bool cap_ref = true;
    rm.set_trace([&](const char* n, int il, const float* d, size_t cnt) {
        for (const char* w : names)
            if (!std::strcmp(w, n)) (cap_ref ? rc : ec)[{n, il2v[il]}].assign(d, d + cnt);
    });
    e->set_trace([&](const char* n, int v, const float* d, size_t cnt) { ec[{n, v}].assign(d, d + cnt); });
    RefState st = rm.new_state();
    std::vector<float> lr(e->n_vocab()), le(e->n_vocab());
    for (int t = 0; t < ntok && t < (int)toks.size(); ++t) {
        rc.clear();
        ec.clear();
        cap_ref = true;
        rm.forward(st, toks[t], lr.data());
        e->decode(slot, toks[t], le.data());
        std::printf("token %d (id %d): cosine(engine, ref) = %.7f  argmax %d/%d\n", t, toks[t], cosine(le.data(), lr.data(), le.size()), argmax(lr.data(), lr.size()), argmax(le.data(), le.size()));
        int first_bad = -1;
        std::printf("  l_out rel-L2 per visit:");
        for (int v = 0; v < kVisits; ++v) {
            auto& a = ec[{"l_out", v}];
            auto& b = rc[{"l_out", v}];
            double num = 0, den = 0;
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                num += (double)(a[i] - b[i]) * (a[i] - b[i]);
                den += (double)b[i] * b[i];
            }
            const double rel = std::sqrt(num / (den + 1e-30));
            if (v % 12 == 0) std::printf("\n   ");
            std::printf(" %.1e", rel);
            if (rel > 1e-3 && first_bad < 0) first_bad = v;
        }
        std::printf("\n");
        if (first_bad >= 0 || t == 0) {
            const int v = first_bad >= 0 ? first_bad : 0;
            std::printf("  per-tensor rel-L2 at visit %d%s:", v, first_bad >= 0 ? " (first with l_out > 1e-3)" : "");
            for (const char* w : names) {
                auto& a = ec[{w, v}];
                auto& b = rc[{w, v}];
                double num = 0, den = 0;
                for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                    num += (double)(a[i] - b[i]) * (a[i] - b[i]);
                    den += (double)b[i] * b[i];
                }
                std::printf(" %s=%.2e(n=%zu/%zu)", w, std::sqrt(num / (den + 1e-30)), a.size(), b.size());
            }
            std::printf("\n");
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string m = argc > 1 ? argv[1] : "oracle";
    if (m == "refcache") {
        for (const char* nm : {"s1_en_capital", "s2_code_add", "s3_en_story", "c1_chat_user", "m1_en_llm", "m2_code_c"})
            ref_logits_cached(nm, read_file<int32_t>(kRoot + "/tools/oracle/" + nm + "/tokens.i32"), true);
        return 0;
    }
    if (m == "oracle") return mode_oracle();
    if (m == "greedy") return mode_greedy();
    if (m == "nll") return mode_nll(argc, argv);
    if (m == "determinism") return mode_determinism();
    if (m == "slots") return mode_slots(argc, argv);
    if (m == "trace") return mode_trace(argc, argv);
    std::fprintf(stderr, "unknown mode %s\n", m.c_str());
    return 2;
}

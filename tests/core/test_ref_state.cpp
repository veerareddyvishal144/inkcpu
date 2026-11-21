// State handling of RefModel: determinism, reset, fork (copy) + continue, position accounting, SWA ring behaviour.
// Uses the first tokens of the oracle prompt s1_en_capital (any ids would do). Float-exact (memcmp) comparisons: the reference is
// deterministic by construction (fixed summation orders, no thread-count dependence of the per-row dots).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ink/gguf.h"
#include "ref_forward.h"
#include "test_util.h"

using namespace ink;

static int g_fail = 0;
#define CHECK(cond)                                                                            \
    do {                                                                                       \
        if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
    } while (0)

static bool same(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * 4); }

int main() {
    std::string err;
    auto g = GgufFile::open(tutil::gguf_path(), &err);
    if (!g) { std::printf("FAIL open: %s\n", err.c_str()); return 1; }
    RefOptions opt;
    opt.n_threads = 8;
    RefModel m(*g, opt);
    const int NV = m.config().n_vocab;
    std::vector<int32_t> all;
    const std::string dir = std::string(INK_SOURCE_DIR) + "/tools/oracle/s1_en_capital/tokens.i32";
    if (!tutil::read_raw(dir, &all) || all.size() < 10) { std::printf("FAIL cannot read %s\n", dir.c_str()); return 1; }
    const std::vector<int32_t> P(all.begin(), all.begin() + 6);

    // 1. determinism + reset: fresh state twice, and the same state after reset()
    RefState a = m.new_state();
    std::vector<float> l1 = m.run(a, P);
    CHECK(a.pos == 6);
    RefState b = m.new_state();
    std::vector<float> l2 = m.run(b, P);
    CHECK(same(l1, l2));
    m.reset(a);
    CHECK(a.pos == 0);
    for (const auto& v : a.visits) CHECK(v.n_cells == 0);
    std::vector<float> l3 = m.run(a, P);
    CHECK(same(l1, l3));
    std::printf("determinism / reset: %s (logits[0..2] = %.6f %.6f %.6f)\n", same(l1, l3) ? "bit-identical" : "DIFFERENT", l1[0], l1[1], l1[2]);

    // 2. position accounting and cache sizes: every visit holds one cell per token (window 1024 not reached)
    const auto& plan = m.plan();
    for (const Visit& v : plan) CHECK(a.visits[v.index].n_cells == 6 && a.visits[v.index].cell_pos.size() == 6);
    CHECK(plan.size() == 48);

    // 3. fork: run 4 tokens, copy the state, continue both with the same 2 tokens -> identical to the straight 6-token run
    RefState c = m.new_state();
    m.run(c, std::vector<int32_t>(P.begin(), P.begin() + 4));
    RefState d = c;   // deep copy of every cache and conv history
    std::vector<float> lc = m.run(c, std::vector<int32_t>(P.begin() + 4, P.end()));
    std::vector<float> ld = m.run(d, std::vector<int32_t>(P.begin() + 4, P.end()));
    CHECK(same(lc, ld));
    CHECK(same(lc, l1));
    std::printf("fork + continue:     %s\n", same(lc, l1) && same(ld, l1) ? "identical to the straight run" : "DIFFERENT");

    // 4. different continuation must give different logits (the state really influences the output)
    RefState e = m.new_state();
    m.run(e, std::vector<int32_t>(P.begin(), P.begin() + 4));
    std::vector<float> le = m.run(e, std::vector<int32_t>{P[5], P[4]});   // swapped order
    tutil::Cmp cmp = tutil::compare(le.data(), l1.data(), NV);
    CHECK(cmp.cosine < 0.999999);
    std::printf("order sensitivity:   cosine(swapped, original) = %.6f\n", cmp.cosine);

    // 5. state independence: advancing d must not touch c (deep copy)
    const int pos_c = c.pos;
    m.run(d, std::vector<int32_t>{P[0]});
    CHECK(c.pos == pos_c && d.pos == pos_c + 1);

    // 6. SWA ring: a window override of 4 over a 10-token prompt keeps 4 cells on SWA visits, all 10 on global visits;
    //    a window larger than the prompt must not change anything (bit-identical to the default).
    const std::vector<int32_t> Q(all.begin(), all.begin() + 10);
    RefOptions o4 = opt;
    o4.sliding_window = 4;
    RefModel m4(*g, o4);
    RefState s4 = m4.new_state();
    std::vector<float> l4 = m4.run(s4, Q);
    int n_swa = 0, n_glob = 0;
    for (const Visit& v : plan) {
        const auto& vs = s4.visits[v.index];
        if (v.is_swa) { ++n_swa; CHECK(vs.n_cells == 4); } else { ++n_glob; CHECK(vs.n_cells == 10); }
        // ring cells hold the newest 4 positions {6,7,8,9}
        if (v.is_swa) { int mn = 99; for (int p : vs.cell_pos) mn = std::min(mn, p); CHECK(mn == 6); }
    }
    CHECK(n_swa == 40 && n_glob == 8);
    RefOptions o64 = opt;
    o64.sliding_window = 64;
    RefModel m64(*g, o64);
    RefState s64 = m64.new_state();
    std::vector<float> l64 = m64.run(s64, Q);
    RefState sd = m.new_state();
    std::vector<float> ld10 = m.run(sd, Q);
    CHECK(same(l64, ld10));
    tutil::Cmp c4 = tutil::compare(l4.data(), ld10.data(), NV);
    CHECK(c4.cosine < 0.9999999);   // the window must be observable
    std::printf("SWA ring:            window 4 keeps 4 cells on 40 SWA visits, 10 on 8 global visits; window 64 == default (bit-identical); cosine(w4, default) = %.6f\n", c4.cosine);

    std::printf(g_fail ? "test_ref_state: %d FAILURES\n" : "test_ref_state: all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}

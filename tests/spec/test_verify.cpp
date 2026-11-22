// test_verify (agent S2, milestone 2): the verify path and its exact rollback, engine level, no drafter.
//   build-s2-b/spec/test_verify [quick|block|wrap|all]      (INK_CPUS=10,12,...; INK_SPEC_EXACT is NOT read here: both attention modes are tested explicitly)
// For a slot A prefilled with p tokens of real text and rows toks[0..T-1] = the next T text tokens:
//   (1) engine_verify logits of EVERY row and the taps of every row == the same rows fed one by one through Engine::decode (bit for bit; decode-kernel attention at any length,
//       the fast kernel below 768 keys),
//   (2) engine_verify_commit(keep = j) leaves the slot EXACTLY as if only the first j rows had been processed: pos, all 48 x 4 conv histories (memcmp), every global KV cell of
//       positions < p + j, every SWA ring cell inside the window of position p + j, and the logits of the following 3 decode steps (bit for bit), for every j in 1..T and T in {2, 4, 8}.
//   "block": p = 61 (rows cross the 64-position KV block boundary), "wrap": p = 2044 (rows cross the global block 2048 and wrap the SWA ring slot 2047 -> 0), quick = p 40 + block.
#include "spec_test_util.h"

using namespace sut;

static void run_p(Engine& E, const std::vector<int32_t>& text, int p, const std::vector<int>& Ts, const std::vector<int>& exacts) {
    const int V = E.n_vocab();
    const int A = 0, X = 1, R = 2, W1 = 3, W2 = 4;
    E.slot_reset(A);
    {
        const double t0 = now_s();
        E.forward(A, text.data(), p, nullptr);
        std::printf("p = %d: prefill %.1f s (load %s)\n", p, now_s() - t0, load1().c_str());
        std::fflush(stdout);
    }
    std::vector<float> vl((size_t)kVerifyMaxRows * V), vt((size_t)kVerifyMaxRows * kEngineTapFloats), rl((size_t)kVerifyMaxRows * V), l1((size_t)V), l2((size_t)V);
    std::vector<float> rt((size_t)kVerifyMaxRows * kEngineTapFloats);
    for (int exact : exacts) {
        for (int T : Ts) {
            const int32_t* toks = text.data() + p;
            std::printf("  p=%d T=%d exact=%d:", p, T, exact);
            std::fflush(stdout);
            E.slot_copy(A, R);
            int nfail0 = g_fail;
            for (int j = 1; j <= T; ++j) {
                // reference: row j-1 through the decode path (state after j rows), with its logits and taps
                E.tap_sink = [&](int, int64_t, int rows, const float* t) { std::memcpy(rt.data() + (size_t)(j - 1) * kEngineTapFloats, t, (size_t)rows * kEngineTapFloats * 4); };
                SUT_CHECK(E.decode(R, toks[j - 1], rl.data() + (size_t)(j - 1) * V) == 0, "decode");
                E.tap_sink = nullptr;
                // verify T rows on a copy of A, keep j
                E.slot_copy(A, X);
                int rc = engine_verify(E, X, toks, T, vl.data(), vt.data(), exact);
                SUT_CHECK(rc == 0, "engine_verify rc %d", rc);
                if (rc) return;
                SUT_CHECK(E.slot_pos(X) == p + T, "pos after verify %lld", (long long)E.slot_pos(X));
                if (j == 1) {   // double verify without commit must be refused
                    SUT_CHECK(engine_verify(E, X, toks, T, vl.data() + 0, nullptr, exact) == -8, "second verify must be refused");
                }
                for (int i = 0; i < j; ++i) {   // rows < j are known: logits and taps bit-identical to the decode path
                    // (reference rows are produced progressively; row i of the verify must equal the decode logits of row i)
                    SUT_CHECK(bits_equal(vl.data() + (size_t)i * V, rl.data() + (size_t)i * V, V), "T=%d keep=%d row %d logits differ (max abs %.3g)", T, j, i,
                              max_abs_diff(vl.data() + (size_t)i * V, rl.data() + (size_t)i * V, V));
                    SUT_CHECK(bits_equal(vt.data() + (size_t)i * kEngineTapFloats, rt.data() + (size_t)i * kEngineTapFloats, kEngineTapFloats), "T=%d keep=%d row %d taps differ", T, j, i);
                }
                rc = engine_verify_commit(E, X, j);
                SUT_CHECK(rc == 0, "commit rc %d", rc);
                SUT_CHECK(engine_verify_commit(E, X, j) == -7, "second commit must be refused");
                std::string why;
                const bool eq = slots_equal(E, X, R, &why);
                SUT_CHECK(eq, "T=%d keep=%d: state after rollback != reference: %s", T, j, why.c_str());
                // the next 3 decode steps (copies, so X and R stay for the next iteration) must produce bit-identical logits
                E.slot_copy(R, W1);
                E.slot_copy(X, W2);
                for (int s = 0; s < 3; ++s) {
                    const int32_t tk = text[(size_t)(p + T + s)];
                    SUT_CHECK(E.decode(W1, tk, l1.data()) == 0 && E.decode(W2, tk, l2.data()) == 0, "follow-up decode");
                    SUT_CHECK(bits_equal(l1.data(), l2.data(), V), "T=%d keep=%d: follow-up step %d logits differ (max abs %.3g)", T, j, s, max_abs_diff(l1.data(), l2.data(), V));
                }
                std::printf(" %d%s", j, g_fail == nfail0 ? "" : "!");
                std::fflush(stdout);
            }
            std::printf("  %s\n", g_fail == nfail0 ? "ok" : "FAILED");
            std::fflush(stdout);
        }
    }
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "quick";
    auto Ep = make_engine(6, 4096);
    Engine& E = *Ep;
    for (int i = 0; i < 6; ++i) E.slot_new();
    std::vector<int32_t> text = prompt("long_context_qa");   // 715 tokens of real text
    while (text.size() < 2200) text.insert(text.end(), text.begin(), text.begin() + std::min<size_t>(715, 2200 - text.size()));   // repeated text for the wrap case
    std::printf("test_verify %s, load %s\n", mode.c_str(), load1().c_str());
    if (mode == "quick" || mode == "all") run_p(E, text, 40, {2, 4, 8}, {1, 0});
    if (mode == "quick" || mode == "block" || mode == "all") run_p(E, text, 61, {8}, {1});
    if (mode == "wrap" || mode == "all") run_p(E, text, 2044, {8}, {1});
    std::printf("test_verify %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

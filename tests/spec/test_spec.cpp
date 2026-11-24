// test_spec (agent S2, milestone 2): the FULL speculative loop (SpecDecoder) against plain greedy decode, with test-double drafters (no S1 code needed).
//   build/spec/test_spec [quick|full|stop|sample|all]            INK_CPUS=10,12,... ; env SPEC_N (new tokens per prompt, default 200), SPEC_PROMPTS (count, default 8),
//                                                                 SPEC_EXACT (1 = decode attention kernels in verify: bit-exact gate; 0 = fast kernel: token agreement gate),
//                                                                 SPEC_SAMPLE_STEPS (spec steps of the sampled test, default 300), SPEC_WRONG_N (tokens of the all-reject run, default 64)
// Greedy exactness: for every prompt the plain greedy continuation gen[0..N+7] (gen[0] = first anchor) is produced by Engine::decode; then the loop runs from the same restored state
// (slot_copy) with (a) the ORACLE drafter (= true continuation: full-accept path), (b) the WRONG drafter (all-reject path), (c) the NOISY oracle (errors injected at random positions:
// every accepted length 0..n), each with the schedules fixed T = 4, fixed T = 8 and MIXED (modes 0/2..8 and max_emit budgets 1..8 varied per step). Every emitted token must equal the plain
// token (bit-exact rows in exact mode); at the end the slot state must equal the plain slot (pos, conv histories, KV cells in window) and the next decode step must be bit-identical.
// Stop handling: a stop id in the middle of a verify window ends the step there, stopped = 1, and the slot equals "only the tokens BEFORE the stop were processed"; budget truncation (max_emit).
// Sampled mode: temperature 0.8 / top_k 8: the first emitted token of SPEC_SAMPLE_STEPS speculative steps (NOISY drafter sampling from a dense q) from the same restored state must follow the
// target distribution p (chi-square against the analytic p from the plain logits); 1500 plain draws likewise; same seed => identical result.
#include <functional>

#include "spec_test_util.h"

using namespace sut;

namespace {

struct Ref {
    std::vector<int32_t> gen;      // gen[0] = argmax after the prompt (first anchor), gen[i] = argmax after feeding gen[i-1]
    std::vector<float> margin;     // top-1 minus top-2 logit of the step that produced gen[i]
    double plain_us = 0;           // mean decode + argmax microseconds
};

float top2_margin(const float* x, int n, int best) {
    float m2 = -INFINITY;
    for (int i = 0; i < n; ++i)
        if (i != best) m2 = std::max(m2, x[i]);
    return x[best] - m2;
}

// plain greedy on slot R (prefilled with the prompt, logits of the prompt in lg): n tokens
Ref make_ref(Engine& E, int R, const float* prompt_logits, int n) {
    Ref r;
    const int V = E.n_vocab();
    std::vector<float> lg((size_t)V);
    int best = argmax(prompt_logits, V);
    r.gen.push_back(best);
    r.margin.push_back(top2_margin(prompt_logits, V, best));
    const double t0 = now_s();
    for (int i = 1; i < n; ++i) {
        E.decode(R, r.gen.back(), lg.data());
        best = argmax(lg.data(), V);
        r.gen.push_back(best);
        r.margin.push_back(top2_margin(lg.data(), V, best));
    }
    r.plain_us = (now_s() - t0) * 1e6 / std::max(1, n - 1);
    return r;
}

struct Sched {
    const char* name;
    std::function<void(int step, SpecRequest&)> set;
};
const std::vector<Sched>& scheds() {
    static const std::vector<Sched> s = {
        {"T4", [](int, SpecRequest& q) { q.mode = 4; q.max_emit = 8; }},
        {"T8", [](int, SpecRequest& q) { q.mode = 8; q.max_emit = 8; }},
        {"mixed", [](int i, SpecRequest& q) {
             static const int pat[10][2] = {{4, 8}, {8, 8}, {0, 8}, {8, 3}, {2, 8}, {5, 8}, {8, 1}, {3, 2}, {8, 8}, {6, 5}};
             q.mode = pat[i % 10][0];
             q.max_emit = pat[i % 10][1];
         }},
    };
    return s;
}

struct RunStats {
    int steps = 0, spec_steps = 0, emitted = 0, drafted = 0, accepted = 0, mismatches = 0, first_mismatch = -1;
    float first_mismatch_margin = 0;
    double secs = 0;
};

// Runs the loop on slot X (a copy of the prompt state at position p) until >= N tokens were emitted; compares with ref. Returns the stats; failures via SUT_CHECK (exact mode) or counted (fast mode).
RunStats run_loop(Engine& E, SpecDecoder& dec, DoubleDrafter& dd, int X, int p, const Ref& ref, const Sched& sc, int N, bool exact_gate) {
    RunStats st;
    dd.set_pos(X, p);
    int32_t anchor = ref.gen[0];
    std::vector<int32_t> out;
    const double t0 = now_s();
    for (int step = 0; (int)out.size() < N; ++step) {
        SpecRequest rq;
        rq.anchor = anchor;
        rq.seed = 0;
        sc.set(step, rq);
        SpecResult res;
        const int rc = dec.step(X, rq, &res);
        SUT_CHECK(rc == 0, "%s step %d rc %d", sc.name, step, rc);
        if (rc) break;
        SUT_CHECK(res.n_tokens >= 1 && res.n_tokens <= rq.max_emit, "%s step %d n_tokens %d (max_emit %d)", sc.name, step, res.n_tokens, rq.max_emit);
        SUT_CHECK(E.slot_pos(X) == p + (int64_t)out.size() + res.n_tokens, "%s step %d slot_pos %lld, expected %lld", sc.name, step, (long long)E.slot_pos(X),
                  (long long)(p + out.size() + res.n_tokens));
        SUT_CHECK(res.stopped == 0, "%s step %d stopped without stop ids", sc.name, step);
        SUT_CHECK(dec.synced(X), "%s step %d drafter out of sync (drafter pos %lld, slot %lld)", sc.name, step, (long long)dd.state_pos(X), (long long)E.slot_pos(X));
        ++st.steps;
        if (res.T > 1) {
            ++st.spec_steps;
            st.drafted += res.drafted;
            st.accepted += res.accepted;
            if (dd.kind == DoubleDrafter::ORACLE) SUT_CHECK(res.accepted == res.drafted, "%s step %d oracle: accepted %d of %d", sc.name, step, res.accepted, res.drafted);
            if (dd.kind == DoubleDrafter::WRONG) SUT_CHECK(res.accepted == 0, "%s step %d wrong drafter: accepted %d", sc.name, step, res.accepted);
        }
        for (int i = 0; i < res.n_tokens; ++i) {
            const size_t idx = out.size() + 1;   // ref index of this emitted token
            if (idx < ref.gen.size() && res.tokens[i] != ref.gen[idx]) {
                if (st.mismatches == 0) {
                    st.first_mismatch = (int)idx;
                    st.first_mismatch_margin = ref.margin[idx];
                }
                ++st.mismatches;
                if (exact_gate) SUT_CHECK(false, "%s: token %zu = %d, plain %d (margin %.4f)", sc.name, idx, res.tokens[i], ref.gen[idx], ref.margin[idx]);
            }
            out.push_back(res.tokens[i]);
        }
        anchor = res.tokens[res.n_tokens - 1];
        if (st.mismatches) break;   // streams diverge from here: nothing more to compare
    }
    st.secs = now_s() - t0;
    st.emitted = (int)out.size();
    return st;
}

// After a mismatch free run: feed the remaining plain tokens so the slot has processed prompt + gen[0..N+6] like the reference slot R; compare states and the next step.
void check_final_state(Engine& E, int X, int R, int W1, int W2, int p, const Ref& ref, int N, int emitted, const char* tag) {
    const int V = E.n_vocab();
    std::vector<float> l1((size_t)V), l2((size_t)V);
    for (int i = emitted; i < N + 7; ++i) SUT_CHECK(E.decode(X, ref.gen[(size_t)i], nullptr) == 0, "%s: fill decode", tag);
    SUT_CHECK(E.slot_pos(X) == p + N + 7 && E.slot_pos(R) == p + N + 7, "%s: positions %lld / %lld", tag, (long long)E.slot_pos(X), (long long)E.slot_pos(R));
    std::string why;
    SUT_CHECK(slots_equal(E, X, R, &why), "%s: final slot state != plain: %s", tag, why.c_str());
    E.slot_copy(R, W1);
    E.slot_copy(X, W2);
    const int32_t tk = ref.gen[(size_t)N + 7];
    SUT_CHECK(E.decode(W1, tk, l1.data()) == 0 && E.decode(W2, tk, l2.data()) == 0, "%s: follow-up decode", tag);
    SUT_CHECK(bits_equal(l1.data(), l2.data(), V), "%s: follow-up logits differ (max abs %.3g)", tag, max_abs_diff(l1.data(), l2.data(), V));
}

SpecOptions opts(int exact) {
    SpecOptions o;
    o.mode = 1;
    o.exact = exact;
    return o;
}

// ------------------------------------------------------------------------------------------------------------ session-lookup drafts (agent L): exactness
// No drafter. Mode 9 (lookup only) with a 2-token key, firing on every match whatever its predicted speed-up, so that most steps verify (mostly wrong) lookup drafts of every
// length 1..7 while some copy runs get accepted: every emitted token must equal plain greedy, the slot state at the end must equal the plain slot, and the history the decoder
// keeps must equal the tokens the slot processed. Also: auto with the ORACLE double (lookup first, DSpark fallback) and a mid-run truncation (note_truncate = rewind).
void lookup_suite(Engine& E, int n_prompts, int N, int exact) {
    const int A = 0, R = 1, X = 2, W1 = 3, W2 = 4;
    const int V = E.n_vocab();
    std::vector<float> lg((size_t)V);
    const bool exact_gate = exact != 0;
    std::printf("lookup exactness: %d prompts x %d tokens, exact=%d, load %s\n", n_prompts, N, exact, load1().c_str());
    int fired_total[3] = {};   // per config over all prompts
    int width_seen[9] = {};
    for (int pi = 0; pi < n_prompts && pi < (int)prompt_names().size(); ++pi) {
        const std::string& name = prompt_names()[(size_t)pi];
        std::vector<int32_t> pr = prompt(name);
        if (pr.empty()) continue;
        const int p = (int)pr.size();
        E.slot_reset(A);
        SUT_CHECK(E.forward(A, pr.data(), p, lg.data()) == 0, "prefill %s", name.c_str());
        E.slot_copy(A, R);
        Ref ref = make_ref(E, R, lg.data(), N + 8);
        std::vector<int32_t> truth = pr;
        truth.insert(truth.end(), ref.gen.begin(), ref.gen.end());
        struct Cfg {
            const char* name;
            int n_min, mode;
            bool oracle;
        };
        const Cfg cfgs[3] = {{"lookup n_min=2 forced", 2, kSpecModeLookup, false}, {"lookup n_min=4 forced", 4, kSpecModeLookup, false}, {"auto lookup+oracle", 3, 1, true}};
        for (int ci = 0; ci < 3; ++ci) {
            const Cfg& cf = cfgs[ci];
            SpecOptions o = opts(exact);
            o.mode = cf.mode;
            o.lookup.n_min = cf.n_min;
            o.lookup.enter = 0.0;       // fire on every match
            o.lookup.probe_every = 1;
            for (int b = 0; b < 4; ++b) o.lookup.prior_c[b] = 0.99;   // start at T = 8; rejections shrink the width, so every width 2..8 gets exercised
            std::unique_ptr<DoubleDrafter> ddp;
            DoubleDrafter* dd = nullptr;
            if (cf.oracle) {
                ddp = std::make_unique<DoubleDrafter>(DoubleDrafter::ORACLE, V, 0.0, 77);
                ddp->truth = truth;
                dd = ddp.get();
            }
            SpecDecoder dec(E, std::move(ddp), o);
            E.slot_copy(A, X);
            dec.note_tokens(X, 0, pr.data(), p);   // the prompt enters the history (what ink_forward does)
            if (dd) dd->set_pos(X, p);
            const int fails0 = g_fail;
            int32_t anchor = ref.gen[0];
            std::vector<int32_t> out;
            int steps = 0, lk_steps = 0, lk_drafted = 0, lk_acc = 0, ds_steps = 0, truncs = 0, mismatches = 0;
            int hist_len[8] = {};
            const double t0 = now_s();
            for (int step = 0; (int)out.size() < N; ++step) {
                // a "rewind" in the middle: drop the last few emitted tokens from the engine (slot_copy of a reference state) and tell the decoder
                if (!cf.oracle && step == 12 && out.size() >= 6) {
                    const int keep = (int)out.size() - 4;
                    E.slot_copy(A, X);
                    for (int i = 0; i < keep; ++i) SUT_CHECK(E.decode(X, ref.gen[(size_t)i], nullptr) == 0, "rewind refill");
                    dec.note_truncate(X, p + keep);
                    out.resize((size_t)keep);
                    anchor = ref.gen[(size_t)keep];
                    ++truncs;
                }
                SpecRequest rq;
                rq.anchor = anchor;
                rq.seed = 0;
                rq.mode = cf.mode;
                rq.max_emit = 2 + step % 7;   // budgets 2..8 cap the verify width, so widths 2..8 are all exercised
                SpecResult res;
                const int rc = dec.step(X, rq, &res);
                SUT_CHECK(rc == 0, "%s step %d rc %d", cf.name, step, rc);
                if (rc) break;
                SUT_CHECK(res.n_tokens >= 1 && res.n_tokens <= rq.max_emit, "%s step %d n_tokens %d", cf.name, step, res.n_tokens);
                SUT_CHECK(E.slot_pos(X) == p + (int64_t)out.size() + res.n_tokens, "%s step %d slot_pos", cf.name, step);
                ++steps;
                if (res.src == 2) {
                    ++lk_steps;
                    lk_drafted += res.drafted;
                    lk_acc += res.accepted;
                    ++hist_len[res.T - 1];
                    ++width_seen[res.T];
                    ++fired_total[ci];
                } else if (res.src == 1) {
                    ++ds_steps;
                }
                for (int i = 0; i < res.n_tokens; ++i) {
                    const size_t idx = out.size() + 1;
                    if (idx < ref.gen.size() && res.tokens[i] != ref.gen[idx]) {
                        ++mismatches;
                        if (exact_gate) SUT_CHECK(false, "%s: token %zu = %d, plain %d (margin %.4f)", cf.name, idx, res.tokens[i], ref.gen[idx], ref.margin[idx]);
                    }
                    out.push_back(res.tokens[i]);
                }
                anchor = res.tokens[res.n_tokens - 1];
                // the decoder's history must be exactly prompt ++ gen[0..out.size()-1] (the last emitted token is the pending anchor)
                const LookupIndex* lk = dec.lookup(X);
                SUT_CHECK(lk && lk->size() == p + (int64_t)out.size(), "%s step %d: history %lld vs %lld", cf.name, step, lk ? (long long)lk->size() : -1LL, (long long)(p + out.size()));
                if (lk && lk->size() == p + (int64_t)out.size() && !mismatches)
                    for (int64_t i = p; i < lk->size(); i += 3) SUT_CHECK(lk->at(i) == truth[(size_t)i], "%s step %d: history[%lld] = %d vs %d", cf.name, step, (long long)i, lk->at(i), truth[(size_t)i]);
                if (mismatches) break;
            }
            const double secs = now_s() - t0;
            std::printf("    %-22s steps %3d (lookup %3d: acc %d/%d, dspark %d) emitted %3d  %.1f s  T hist", cf.name, steps, lk_steps, lk_acc, lk_drafted, ds_steps, (int)out.size(), secs);
            for (int t = 1; t <= 8; ++t)
                if (hist_len[t - 1]) std::printf(" %d:%d", t, hist_len[t - 1]);
            if (truncs) std::printf("  (rewind x%d)", truncs);
            if (!mismatches) {
                check_final_state(E, X, R, W1, W2, p, ref, N, (int)out.size(), cf.name);
                std::printf(g_fail == fails0 ? "  tokens == plain, state == plain" : "  STATE DIFFERS");
            } else {
                std::printf("  MISMATCH");
            }
            std::printf("%s\n", g_fail == fails0 ? "" : "  FAILED");
            std::fflush(stdout);
        }
    }
    for (int ci = 0; ci < 3; ++ci) SUT_CHECK(fired_total[ci] > 0, "config %d: the lookup drafter never fired on any prompt", ci);
    std::printf("  lookup widths exercised:");
    for (int t = 2; t <= 8; ++t) std::printf(" T%d:%d", t, width_seen[t]);
    std::printf("\n");
    int n_widths = 0;
    for (int t = 2; t <= 8; ++t) n_widths += width_seen[t] > 0;
    SUT_CHECK(n_widths >= 5 && width_seen[2] > 0 && width_seen[8] > 0, "at least 5 of the widths 2..8 incl. 2 and 8 must be exercised (%d)", n_widths);
}

// ------------------------------------------------------------------------------------------------------------ greedy exactness over the prompts
void greedy_suite(Engine& E, int n_prompts, int N, int exact) {
    const int A = 0, R = 1, X = 2, W1 = 3, W2 = 4;
    const int V = E.n_vocab();
    std::vector<float> lg((size_t)V);
    const bool exact_gate = exact != 0;
    std::printf("greedy exactness: %d prompts x %d tokens, exact=%d (%s), load %s\n", n_prompts, N, exact, exact_gate ? "bit-exact gate" : "token agreement gate", load1().c_str());
    double sum_plain_us = 0;
    for (int pi = 0; pi < n_prompts && pi < (int)prompt_names().size(); ++pi) {
        const std::string& name = prompt_names()[(size_t)pi];
        std::vector<int32_t> pr = prompt(name);
        if (pr.empty()) {
            std::printf("  %s: no prompt file, skipped\n", name.c_str());
            continue;
        }
        const int p = (int)pr.size();
        E.slot_reset(A);
        SUT_CHECK(E.forward(A, pr.data(), p, lg.data()) == 0, "prefill %s", name.c_str());
        E.slot_copy(A, R);
        Ref ref = make_ref(E, R, lg.data(), N + 8);
        sum_plain_us += ref.plain_us;
        std::vector<int32_t> truth = pr;
        truth.insert(truth.end(), ref.gen.begin(), ref.gen.end());
        std::printf("  %-18s p=%4d plain %.0f ms/tok (%.2f tok/s)\n", name.c_str(), p, ref.plain_us / 1e3, 1e6 / ref.plain_us);
        std::fflush(stdout);
        struct Kind {
            DoubleDrafter::Kind k;
            const char* name;
            double err;
        };
        const Kind kinds[3] = {{DoubleDrafter::ORACLE, "oracle", 0}, {DoubleDrafter::NOISY, "noisy", 0.3}, {DoubleDrafter::WRONG, "wrong", 0}};
        for (const Kind& kd : kinds) {
            for (const Sched& sc : scheds()) {
                if (kd.k == DoubleDrafter::WRONG && std::strcmp(sc.name, "mixed") != 0) continue;   // all-reject: one schedule (each step costs a verify and yields one token)
                const int n_run = kd.k == DoubleDrafter::WRONG ? std::min(N, env_int("SPEC_WRONG_N", 64)) : N;
                auto ddp = std::make_unique<DoubleDrafter>(kd.k, V, kd.err, 1234 + pi);
                DoubleDrafter& dd = *ddp;
                dd.truth = truth;
                SpecDecoder dec(E, std::move(ddp), opts(exact));
                E.slot_copy(A, X);
                const int fails0 = g_fail;
                RunStats st = run_loop(E, dec, dd, X, p, ref, sc, n_run, exact_gate);
                const double L = st.spec_steps ? (double)(st.emitted - (st.steps - st.spec_steps)) / st.spec_steps : 0;
                std::printf("    %-7s %-6s steps %3d (spec %3d) emitted %3d  L %.2f  acc %d/%d  %.1f s (%.2f tok/s)%s", kd.name, sc.name, st.steps, st.spec_steps, st.emitted, L, st.accepted,
                            st.drafted, st.secs, st.emitted / std::max(st.secs, 1e-9), st.mismatches ? "" : "  tokens == plain");
                if (st.mismatches) std::printf("  MISMATCH at token %d (plain margin %.4f)", st.first_mismatch, st.first_mismatch_margin);
                if (!st.mismatches && n_run == N) {
                    check_final_state(E, X, R, W1, W2, p, ref, N, st.emitted, sc.name);
                    std::printf(g_fail == fails0 ? "  state == plain" : "  STATE DIFFERS");
                }
                std::printf("%s\n", g_fail == fails0 ? "" : "  FAILED");
                std::fflush(stdout);
            }
        }
    }
    std::printf("mean plain decode %.0f ms/tok over the prompts (indicative, shared host)\n", sum_plain_us / std::max(1, n_prompts) / 1e3);
}

// ------------------------------------------------------------------------------------------------------------ stop ids and budgets
void stop_suite(Engine& E, int exact) {
    const int A = 0, R = 1, X = 2, W1 = 3, W2 = 4;
    const int V = E.n_vocab();
    std::vector<float> lg((size_t)V), l1((size_t)V), l2((size_t)V);
    std::vector<int32_t> pr = prompt("chat_code");
    const int p = (int)pr.size();
    E.slot_reset(A);
    SUT_CHECK(E.forward(A, pr.data(), p, lg.data()) == 0, "prefill");
    E.slot_copy(A, R);
    Ref ref = make_ref(E, R, lg.data(), 24);
    std::vector<int32_t> truth = pr;
    truth.insert(truth.end(), ref.gen.begin(), ref.gen.end());
    std::printf("stop / budget suite on chat_code (p = %d), exact=%d\n", p, exact);
    // reference slot fed exactly gen[0..m-1]
    auto ref_state = [&](int m) {
        E.slot_copy(A, W1);
        for (int i = 0; i < m; ++i) E.decode(W1, ref.gen[(size_t)i], nullptr);
    };
    auto same_as_ref = [&](int m, const char* tag) {
        ref_state(m);
        std::string why;
        SUT_CHECK(E.slot_pos(X) == p + m, "%s: pos %lld, expected %d", tag, (long long)E.slot_pos(X), p + m);
        SUT_CHECK(slots_equal(E, X, W1, &why), "%s: state != plain fed %d tokens: %s", tag, m, why.c_str());
        E.slot_copy(X, W2);
        const int32_t tk = ref.gen[(size_t)m];
        SUT_CHECK(E.decode(W1, tk, l1.data()) == 0 && E.decode(W2, tk, l2.data()) == 0, "%s: follow-up decode", tag);
        SUT_CHECK(bits_equal(l1.data(), l2.data(), V), "%s: follow-up logits differ", tag);
    };
    struct Case {
        const char* name;
        int mode, max_emit;
        std::vector<int> stop_idx;   // ref indices used as stop ids
        int expect_n, expect_stopped;
    };
    const std::vector<Case> cases = {
        {"T8 stop at draft 5 (mid window)", 8, 8, {5}, 5, 1},
        {"T8 stop = bonus token (row 8)", 8, 8, {8}, 8, 1},
        {"T8 stop = first emitted", 8, 8, {1}, 1, 1},
        {"T8 two stop ids, first wins", 8, 8, {7, 3}, 3, 1},
        {"T4 stop beyond the window (no stop)", 4, 8, {9}, 4, 0},
        {"plain stop", 0, 8, {1}, 1, 1},
        {"plain no stop", 0, 8, {2}, 1, 0},
        {"T8 budget 3", 8, 3, {}, 3, 0},
        {"T8 budget 1", 8, 1, {}, 1, 0},
        {"T8 budget 3 then stop inside budget", 8, 3, {2}, 2, 1},
        {"T8 budget 3, stop outside budget", 8, 3, {5}, 3, 0},
    };
    for (const Case& c : cases) {
        auto ddp = std::make_unique<DoubleDrafter>(DoubleDrafter::ORACLE, V);
        DoubleDrafter& dd = *ddp;
        dd.truth = truth;
        SpecDecoder dec(E, std::move(ddp), opts(exact));
        E.slot_copy(A, X);
        dd.set_pos(X, p);
        std::vector<int32_t> stops;
        for (int i : c.stop_idx) stops.push_back(ref.gen[(size_t)i]);
        SpecRequest rq;
        rq.anchor = ref.gen[0];
        rq.mode = c.mode;
        rq.max_emit = c.max_emit;
        rq.stop_ids = stops.data();
        rq.n_stop = (int)stops.size();
        SpecResult res;
        const int f0 = g_fail;
        SUT_CHECK(dec.step(X, rq, &res) == 0, "%s: rc %d", c.name, res.rc);
        SUT_CHECK(res.n_tokens == c.expect_n, "%s: n_tokens %d, expected %d", c.name, res.n_tokens, c.expect_n);
        SUT_CHECK(res.stopped == c.expect_stopped, "%s: stopped %d, expected %d", c.name, res.stopped, c.expect_stopped);
        for (int i = 0; i < res.n_tokens; ++i) SUT_CHECK(res.tokens[i] == ref.gen[(size_t)i + 1], "%s: token %d = %d, plain %d", c.name, i, res.tokens[i], ref.gen[(size_t)i + 1]);
        same_as_ref(res.n_tokens, c.name);   // rows processed = anchor + emitted tokens before the last one = n_tokens rows; the stop token itself is never fed
        SUT_CHECK(dd.state_pos(X) == E.slot_pos(X), "%s: drafter pos %lld != slot %lld", c.name, (long long)dd.state_pos(X), (long long)E.slot_pos(X));
        if (!res.stopped) {   // continue: the next step must carry on the plain stream
            SpecRequest rq2 = rq;
            rq2.anchor = res.tokens[res.n_tokens - 1];
            rq2.max_emit = 8;
            rq2.mode = 4;
            SpecResult r2;
            SUT_CHECK(dec.step(X, rq2, &r2) == 0, "%s: second step", c.name);
            for (int i = 0; i < r2.n_tokens; ++i)
                SUT_CHECK(r2.tokens[i] == ref.gen[(size_t)res.n_tokens + 1 + i], "%s: second step token %d differs", c.name, i);
        }
        std::printf("  %-42s n=%d stopped=%d pos=%lld %s\n", c.name, res.n_tokens, res.stopped, (long long)E.slot_pos(X), g_fail == f0 ? "ok" : "FAILED");
    }
    // the stop token inside a window with a NOISY drafter: whatever the accepted length, the slot must equal "tokens before the stop" (run several seeds)
    for (int s = 0; s < 4; ++s) {
        auto ddp = std::make_unique<DoubleDrafter>(DoubleDrafter::NOISY, V, 0.4, 77 + s);
        DoubleDrafter& dd = *ddp;
        dd.truth = truth;
        SpecDecoder dec(E, std::move(ddp), opts(exact));
        E.slot_copy(A, X);
        dd.set_pos(X, p);
        const int32_t stop = ref.gen[3];
        SpecRequest rq;
        rq.anchor = ref.gen[0];
        rq.mode = 8;
        rq.stop_ids = &stop;
        rq.n_stop = 1;
        SpecResult res;
        const int f0 = g_fail;
        int total = 0, acc = 0, dr = 0, steps = 0;   // a rejected first draft emits one token per step: loop to the stop
        for (;;) {
            SUT_CHECK(dec.step(X, rq, &res) == 0, "noisy stop seed %d rc", s);
            total += res.n_tokens;
            acc += res.accepted;
            dr += res.drafted;
            ++steps;
            if (res.stopped || total >= 3 || steps > 3) break;
            rq.anchor = res.tokens[res.n_tokens - 1];
        }
        SUT_CHECK(total == 3 && res.stopped == 1 && res.tokens[res.n_tokens - 1] == stop, "noisy stop seed %d: total %d stopped %d", s, total, res.stopped);
        same_as_ref(3, "noisy stop");
        std::printf("  noisy drafter, stop at token 3 reached in %d step(s), accepted %d/%d: %s\n", steps, acc, dr, g_fail == f0 ? "ok" : "FAILED");
    }
}

// ------------------------------------------------------------------------------------------------------------ sampled mode
double chi_square(const std::vector<double>& expected, const std::vector<int>& counts, int* df) {
    // merges bins with expected < 5 into the last one
    double chi = 0, e_acc = 0;
    int c_acc = 0, bins = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        e_acc += expected[i];
        c_acc += counts[i];
        if (e_acc >= 5.0 || i + 1 == expected.size()) {
            if (e_acc > 0) {
                chi += (c_acc - e_acc) * (c_acc - e_acc) / e_acc;
                ++bins;
            }
            e_acc = 0;
            c_acc = 0;
        }
    }
    *df = std::max(1, bins - 1);
    return chi;
}

void sample_suite(Engine& E, int exact) {
    const int A = 0, X = 2;
    const int V = E.n_vocab();
    std::vector<float> lg((size_t)V);
    std::vector<int32_t> pr = prompt("chat_answer");
    const int p = (int)pr.size();
    E.slot_reset(A);
    SUT_CHECK(E.forward(A, pr.data(), p, lg.data()) == 0, "prefill");
    SamplingParams sp;
    sp.temperature = 0.8f;
    sp.top_k = 8;
    Sampler smp(V);
    TargetDist P;
    // pick, along the first 16 greedy tokens, the position whose target distribution is the flattest (lowest top-1 probability): a near one-hot p makes the test vacuous.
    // The state A is advanced to that position (plain decode) and the anchor is the greedy token there; the plain logits after the anchor give the analytic p.
    int32_t anchor = argmax(lg.data(), V);
    {
        const int B = 3;   // spare slot for the scan
        E.slot_copy(A, B);
        int32_t a = anchor;
        float best_top1 = 2.0f;
        int best_i = 0;
        std::vector<float> l2((size_t)V);
        for (int i = 0; i < 16; ++i) {
            SUT_CHECK(E.decode(B, a, l2.data()) == 0, "scan decode");
            TargetDist d;
            smp.build(l2.data(), sp, &d);
            const float top1 = d.probs.empty() ? 1.0f : d.probs[0];
            if (top1 < best_top1) {
                best_top1 = top1;
                best_i = i;
            }
            a = argmax(l2.data(), V);
        }
        a = anchor;   // advance A to position p + best_i (anchor = the greedy token there)
        for (int i = 0; i < best_i; ++i) {
            SUT_CHECK(E.decode(A, a, l2.data()) == 0, "advance decode");
            a = argmax(l2.data(), V);
        }
        anchor = a;
        std::printf("  flattest of the first 16 positions: offset %d (top-1 prob %.3f at temperature 0.8)\n", best_i, best_top1);
    }
    const int p_eff = (int)E.slot_pos(A);
    // the target distribution after the anchor (plain path: one decode)
    E.slot_copy(A, X);
    SUT_CHECK(E.decode(X, anchor, lg.data()) == 0, "decode");
    smp.build(lg.data(), sp, &P);
    SUT_CHECK(!P.dense && P.ids.size() == 8, "top_k 8 distribution has %zu ids", P.ids.size());
    std::printf("sampled mode on chat_answer (p = %d), temperature 0.8 top_k 8, exact=%d. p = [", p, exact);
    for (size_t i = 0; i < P.ids.size(); ++i) std::printf("%s%d:%.3f", i ? " " : "", P.ids[i], P.probs[i]);
    std::printf("]\n");
    auto bin_of = [&](int32_t t) {
        for (size_t i = 0; i < P.ids.size(); ++i)
            if (P.ids[i] == t) return (int)i;
        return (int)P.ids.size();   // outside the support
    };
    // (1) plain draws: 1500 samples (Sampler::sample on the same logits with different seeds)
    {
        const int n = 1500;
        std::vector<int> cnt(P.ids.size() + 1, 0);
        for (int i = 0; i < n; ++i) {
            SpecRng rng(1000 + (uint64_t)i * 7919);
            ++cnt[(size_t)bin_of(smp.sample(lg.data(), sp, rng))];
        }
        std::vector<double> ex;
        for (float q : P.probs) ex.push_back(q * n);
        int df;
        const double chi = chi_square(ex, cnt, &df);
        SUT_CHECK(cnt.back() == 0, "plain draw outside the top-k support (%d)", cnt.back());
        SUT_CHECK(chi < 30.0, "plain draws: chi-square %.1f (df %d)", chi, df);
        std::printf("  plain sampler: %d draws, chi-square %.2f (df %d, bound 30) %s\n", n, chi, df, chi < 30 ? "ok" : "FAILED");
    }
    // (2) speculative steps from the same restored state: first emitted token ~ p. NOISY drafter in sample mode (dense q).
    {
        const int n = env_int("SPEC_SAMPLE_STEPS", 300);
        std::vector<int> cnt(P.ids.size() + 1, 0);
        int acc = 0, dr = 0, outside = 0;
        std::vector<int32_t> pr_gen;   // truth for the double: prompt + anchor + some plausible continuation (irrelevant for the first-token test: the drafts are sampled from q)
        std::vector<int32_t> truth((size_t)p_eff, 0);
        truth.push_back(anchor);
        for (int i = 0; i < 16; ++i) truth.push_back(P.ids[(size_t)(i % P.ids.size())]);
        auto ddp = std::make_unique<DoubleDrafter>(DoubleDrafter::NOISY, V, 0.3, 99);
        DoubleDrafter& dd = *ddp;
        dd.truth = truth;
        SpecOptions o = opts(exact);
        o.draft_sample = true;
        SpecDecoder dec(E, std::move(ddp), o);
        const double t0 = now_s();
        SpecResult first;
        for (int i = 0; i < n; ++i) {
            E.slot_copy(A, X);
            dd.set_pos(X, p_eff);
            SpecRequest rq;
            rq.anchor = anchor;
            rq.sp = sp;
            rq.seed = 5000 + (uint64_t)i * 104729;
            rq.mode = 4;
            SpecResult res;
            SUT_CHECK(dec.step(X, rq, &res) == 0, "spec step %d rc %d", i, res.rc);
            if (res.rc) break;
            SUT_CHECK(res.T == 4 && res.drafted == 3, "spec step %d: T %d drafted %d", i, res.T, res.drafted);
            const int b = bin_of(res.tokens[0]);
            if (b == (int)P.ids.size()) ++outside;
            ++cnt[(size_t)b];
            acc += res.accepted;
            dr += res.drafted;
            if (i == 0) first = res;
            if (i == 0) {   // determinism: the same seed from the same state gives the same result
                E.slot_copy(A, X);
                dd.set_pos(X, p_eff);
                SpecResult again;
                SUT_CHECK(dec.step(X, rq, &again) == 0, "repeat step");
                SUT_CHECK(again.n_tokens == res.n_tokens && std::memcmp(again.tokens, res.tokens, sizeof(int32_t) * (size_t)res.n_tokens) == 0 && again.accepted == res.accepted,
                          "same seed, different result");
            }
        }
        std::vector<double> ex;
        for (float q : P.probs) ex.push_back(q * n);
        int df;
        const double chi = chi_square(ex, cnt, &df);
        SUT_CHECK(outside == 0, "%d first tokens outside the target support", outside);
        SUT_CHECK(chi < 30.0, "spec first tokens: chi-square %.1f (df %d)", chi, df);
        std::printf("  speculative steps: %d, first-token chi-square %.2f (df %d, bound 30), accepted %d/%d (%.2f), %.1f s; counts [", n, chi, df, acc, dr, dr ? (double)acc / dr : 0,
                    now_s() - t0);
        for (size_t i = 0; i < cnt.size(); ++i) std::printf("%s%d", i ? " " : "", cnt[i]);
        std::printf("] %s\n", chi < 30 && outside == 0 ? "ok" : "FAILED");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "quick";
    const int exact = env_int("SPEC_EXACT", 1);
    auto Ep = make_engine(5, 4096);
    Engine& E = *Ep;
    for (int i = 0; i < 5; ++i) E.slot_new();
    std::printf("test_spec %s, load %s, threads %d\n", mode.c_str(), load1().c_str(), E.opt.n_threads);
    if (mode == "quick") greedy_suite(E, env_int("SPEC_PROMPTS", 2), env_int("SPEC_N", 40), exact);
    if (mode == "full" || mode == "all") greedy_suite(E, env_int("SPEC_PROMPTS", 8), env_int("SPEC_N", 200), exact);
    if (mode == "quick" || mode == "lookup" || mode == "all") lookup_suite(E, env_int("SPEC_PROMPTS", 2), env_int("SPEC_N", 40), exact);
    if (mode == "quick" || mode == "stop" || mode == "all") stop_suite(E, exact);
    if (mode == "sample" || mode == "all") sample_suite(E, exact);
    std::printf("test_spec %s: %s (%d failures)\n", mode.c_str(), g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

// test_policy (agent S2): deterministic simulation of SpecPolicy (no engine, no clock). A simulated target with per-position conditional acceptance c and fixed step costs
// (plain 200 ms; verify(T) = plain x bytes-model ratio; draft(n) = plain x (0.21 + 0.013 n)) is driven by the policy; checks: code-like regime (c = 0.9) -> speculates with T = 8 and
// gains >= 1.4x; chat-like regime (c = 0.45) -> stays plain (>= 85 % of the steps), probes back off to the cap, loses < 10 %; regime switch -> re-enters within the probe cap;
// modes off / fixed; max_T clamp; two identical runs give identical decisions.
#include <cmath>
#include <cstdio>
#include <vector>

#include "ink/spec.h"

using namespace ink;
static int g_fail = 0;
#define CHECK(c, ...)                                                   \
    do {                                                                \
        if (!(c)) {                                                     \
            std::printf("  FAIL %d: %s -- ", __LINE__, #c);             \
            std::printf(__VA_ARGS__);                                   \
            std::printf("\n");                                          \
            ++g_fail;                                                   \
        }                                                               \
    } while (0)

struct Sim {
    double plain = 200000;                                                   // us
    double vratio[9] = {0, 1.0, 1.59, 2.0, 2.47, 2.8, 3.1, 3.4, 3.72};
    double da = 0.21, db = 0.013;
    uint64_t rng = 1;
    double u() {
        rng = spec_mix64(rng);
        return (double)(rng >> 11) * (1.0 / 9007199254740992.0);
    }
    double jitter() { return 0.9 + 0.2 * u(); }                             // +-10 % timing noise
    struct Out {
        int tokens = 0, steps = 0, spec_steps = 0, probes = 0;
        double us = 0;
        std::vector<int> Ts;
        int hist_T[9] = {};
    };
    // c(step) = conditional acceptance at this step (same for every position)
    template <class CF>
    Out run(SpecPolicy& pol, int n_tokens, CF cf, int max_T = 8) {
        Out o;
        while (o.tokens < n_tokens) {
            const int T = pol.choose(max_T);
            o.Ts.push_back(T);
            ++o.hist_T[T];
            ++o.steps;
            if (T == 1) {
                const double t = plain * jitter();
                pol.observe_plain(t);
                o.us += t;
                o.tokens += 1;
                continue;
            }
            ++o.spec_steps;
            const double c = cf(o.tokens);
            int acc = 0;
            while (acc < T - 1 && u() < c) ++acc;
            const double dus = plain * (da + db * (T - 1)) * jitter(), vus = plain * vratio[T] * jitter();
            pol.observe_spec(T, acc, dus, vus);
            o.us += dus + vus;
            o.tokens += acc + 1;
        }
        return o;
    }
};

static void report(const char* name, const Sim::Out& o, double plain) {
    const double speed = (double)o.tokens / o.us * plain;   // tokens per plain-step time, i.e. speed-up vs plain
    std::printf("  %-28s steps %4d (spec %4d) tokens %4d  speed-up %.2fx  T hist: 1:%d 4:%d 8:%d\n", name, o.steps, o.spec_steps, o.tokens, speed, o.hist_T[1], o.hist_T[4], o.hist_T[8]);
}

int main() {
    Sim sim;
    SpecPolicyOptions po;
    // --- code-like regime
    {
        SpecPolicy pol(po);
        Sim s = sim;
        auto o = s.run(pol, 600, [](int) { return 0.9; });
        const double speed = (double)o.tokens / o.us * sim.plain;
        report("code-like c=0.9", o, sim.plain);
        CHECK(speed >= 1.4, "speed-up %.2f", speed);
        CHECK(o.hist_T[8] > o.hist_T[4] && o.hist_T[8] > o.hist_T[1], "T=8 should dominate (1:%d 4:%d 8:%d)", o.hist_T[1], o.hist_T[4], o.hist_T[8]);
        CHECK(pol.speculating(), "should be speculating at the end");
        CHECK(pol.L(8) > 4.0 && pol.L(8) < 6.5, "L(8) estimate %.2f", pol.L(8));
        // determinism
        SpecPolicy pol2(po);
        Sim s2 = sim;
        auto o2 = s2.run(pol2, 600, [](int) { return 0.9; });
        CHECK(o.Ts == o2.Ts, "two identical runs differ");
    }
    // --- chat-like regime: speculation does not pay (L(4) ~ 1.8 < 2), policy must stay plain and probe rarely
    {
        SpecPolicy pol(po);
        Sim s = sim;
        auto o = s.run(pol, 1500, [](int) { return 0.45; });
        const double speed = (double)o.tokens / o.us * sim.plain;
        report("chat-like c=0.45", o, sim.plain);
        CHECK(speed >= 0.90, "chat regime loses %.1f %%", (1 - speed) * 100);
        CHECK(o.hist_T[1] >= (int)(0.85 * o.steps), "plain steps %d of %d", o.hist_T[1], o.steps);
        CHECK(!pol.speculating(), "should be plain at the end");
        CHECK(pol.probe_interval() == po.reprobe_max, "probe interval %d (cap %d)", pol.probe_interval(), po.reprobe_max);
        CHECK(pol.L(4) < 2.1, "L(4) estimate %.2f", pol.L(4));
    }
    // --- regime switch: chat for 600 tokens then code; must re-enter within the probe cap + a few steps
    {
        SpecPolicy pol(po);
        Sim s = sim;
        auto o = s.run(pol, 1400, [](int tok) { return tok < 600 ? 0.45 : 0.92; });
        report("switch chat->code at 600", o, sim.plain);
        // find the first step after the switch where T = 8 was chosen twice in a row
        int tok = 0, entered_at = -1, run8 = 0;
        for (size_t i = 0; i < o.Ts.size(); ++i) {
            if (tok >= 600) {
                run8 = o.Ts[i] == 8 ? run8 + 1 : 0;
                if (run8 >= 2) {
                    entered_at = tok;
                    break;
                }
            }
            tok += o.Ts[i] == 1 ? 1 : 1;   // lower bound on tokens; precise enough for the bound below
        }
        CHECK(entered_at >= 0 && entered_at - 600 <= po.reprobe_max + 60, "re-entered at token %d (switch at 600, cap %d)", entered_at, po.reprobe_max);
        CHECK(pol.speculating(), "should be speculating at the end");
    }
    // --- priors only: the first step is plain (needs t_plain), the probe comes right after; after a fully accepted probe pair it enters
    {
        SpecPolicy pol(po);
        CHECK(pol.choose(8) == 1, "first step must be plain");
        pol.observe_plain(sim.plain);
        const int T2 = pol.choose(8);
        CHECK(T2 == 4, "second step should probe with T=4, got %d", T2);
        pol.observe_spec(4, 3, sim.plain * 0.25, sim.plain * 2.47);
        int steps_to_enter = 0;
        while (!pol.speculating() && steps_to_enter < 20) {
            const int T = pol.choose(8);
            if (T == 1) pol.observe_plain(sim.plain);
            else pol.observe_spec(T, T - 1, sim.plain * (0.21 + 0.013 * (T - 1)), sim.plain * sim.vratio[T]);
            ++steps_to_enter;
        }
        CHECK(pol.speculating(), "did not enter after %d perfect steps", steps_to_enter);
        std::printf("  from priors: entered speculation after %d steps of full acceptance\n", steps_to_enter + 2);
    }
    // --- modes and clamps
    {
        SpecPolicyOptions o0 = po;
        o0.mode = 0;
        SpecPolicy off(o0);
        for (int i = 0; i < 5; ++i) {
            CHECK(off.choose(8) == 1, "mode off chose > 1");
            off.observe_plain(1000);
        }
        SpecPolicyOptions o4 = po;
        o4.mode = 4;
        SpecPolicy fixed4(o4);
        CHECK(fixed4.choose(8) == 4 && fixed4.choose(3) == 3 && fixed4.choose(1) == 1, "fixed 4 clamp");
        fixed4.observe_spec(4, 0, 100, 100);
        CHECK(fixed4.choose(8) == 4, "fixed 4 must ignore acceptance");
        SpecPolicyOptions o8 = po;
        o8.mode = 8;
        SpecPolicy fixed8(o8);
        CHECK(fixed8.choose(8) == 8 && fixed8.choose(5) == 5, "fixed 8 clamp");
        SpecPolicy a(po);
        a.observe_plain(sim.plain);
        for (int i = 0; i < 6; ++i) a.observe_spec(8, 7, sim.plain * 0.3, sim.plain * 3.72);
        CHECK(a.choose(8) == 8, "auto with perfect acceptance should pick 8");
        CHECK(a.choose(4) == 4, "max_T 4 clamp");
        CHECK(a.choose(2) == 1 || a.choose(2) == 2, "max_T 2");
        a.set_mode(0);
        CHECK(a.choose(8) == 1, "set_mode off");
        std::printf("  modes / clamps ok\n");
    }
    std::printf("test_policy %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

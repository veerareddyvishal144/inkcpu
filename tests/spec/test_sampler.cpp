// test_sampler (agent S2): the C++ target sampler and exact speculative sampling on synthetic logits, no engine, < 2 s.
//   (1) Sampler::build semantics: temperature / top_k / top_p / min_p against a straightforward double-precision implementation of server/sampling.py (same kept set, probs within 1e-6).
//   (2) Draws follow the built distribution (chi-square).
//   (3) Exact speculative sampling: with d ~ q (dense q, or one-hot greedy drafts) the FIRST emitted token of spec_accept is distributed as p (chi-square over 200k rounds),
//       for q close to p, q far from p, q with mass outside supp(p), and a one-hot q on a token with p(d) = 0 (must always reject and draw from p).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

static std::vector<float> rand_logits(int V, uint64_t seed, float scale) {
    std::vector<float> x((size_t)V);
    uint64_t s = seed;
    for (int i = 0; i < V; ++i) {
        s = spec_mix64(s + 1);
        const double u = (double)(s >> 11) * (1.0 / 9007199254740992.0);
        x[(size_t)i] = (float)(scale * std::log(u / (1 - u)));   // logistic, heavy-ish tails
    }
    return x;
}

// reference of server/sampling.py in double precision (full sort; kept set and renormalised probs)
static void ref_build(const std::vector<float>& x, const SamplingParams& sp, std::vector<int32_t>* ids, std::vector<double>* probs) {
    const int V = (int)x.size();
    float m = *std::max_element(x.begin(), x.end());
    std::vector<double> e((size_t)V);
    for (int i = 0; i < V; ++i) {
        float v = x[(size_t)i] - m;
        if (sp.temperature != 1.0f) v *= (float)(1.0 / (double)sp.temperature);
        e[(size_t)i] = std::exp((double)std::max(v, -80.0f));
    }
    std::vector<int32_t> order((size_t)V);
    for (int i = 0; i < V; ++i) order[(size_t)i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return e[(size_t)a] > e[(size_t)b]; });
    size_t n = (size_t)V;
    if (sp.top_k > 0 && sp.top_k < V) n = (size_t)sp.top_k;
    double z = 0;
    for (size_t i = 0; i < n; ++i) z += e[(size_t)order[i]];
    size_t keep = n;
    if (sp.top_p < 1.0f) {
        double cum = 0;
        for (size_t i = 0; i < n; ++i) {
            cum += e[(size_t)order[i]];
            if (cum / z >= (double)sp.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    if (sp.min_p > 0.0f) {
        size_t km = 0;
        while (km < n && e[(size_t)order[km]] >= (double)sp.min_p * e[(size_t)order[0]]) ++km;
        keep = std::min(keep, km);
    }
    keep = std::max<size_t>(keep, 1);
    double s = 0;
    for (size_t i = 0; i < keep; ++i) s += e[(size_t)order[i]];
    ids->assign(order.begin(), order.begin() + (long)keep);
    probs->resize(keep);
    for (size_t i = 0; i < keep; ++i) (*probs)[i] = e[(size_t)order[i]] / s;
}

static double chi2(const std::vector<double>& ex, const std::vector<int>& cnt, int* df) {
    double chi = 0, ea = 0;
    int ca = 0, bins = 0;
    for (size_t i = 0; i < ex.size(); ++i) {
        ea += ex[i];
        ca += cnt[i];
        if (ea >= 5 || i + 1 == ex.size()) {
            if (ea > 0) {
                chi += (ca - ea) * (ca - ea) / ea;
                ++bins;
            }
            ea = 0;
            ca = 0;
        }
    }
    *df = std::max(1, bins - 1);
    return chi;
}

static void test_build(int V) {
    Sampler smp(V);
    struct P {
        float t, tp, mp;
        int k;
    };
    const P ps[] = {{1.0f, 1.0f, 0.0f, 0}, {0.7f, 1.0f, 0.0f, 40}, {0.9f, 0.9f, 0.0f, 0}, {1.0f, 0.95f, 0.0f, 50}, {0.8f, 1.0f, 0.05f, 0}, {1.2f, 0.8f, 0.02f, 20}, {0.5f, 1.0f, 0.0f, 1}, {0.0f, 1.0f, 0.0f, 0}};
    int n_ok = 0, n = 0;
    for (int seed = 1; seed <= 6; ++seed) {
        std::vector<float> x = rand_logits(V, (uint64_t)seed, seed % 2 ? 2.0f : 5.0f);
        for (const P& q : ps) {
            SamplingParams sp;
            sp.temperature = q.t;
            sp.top_k = q.k;
            sp.top_p = q.tp;
            sp.min_p = q.mp;
            TargetDist d;
            smp.build(x.data(), sp, &d);
            ++n;
            if (sp.greedy()) {
                const int32_t a = (int32_t)(std::max_element(x.begin(), x.end()) - x.begin());
                CHECK(d.greedy && d.greedy_id == a, "greedy id %d vs %d", d.greedy_id, a);
                n_ok += d.greedy && d.greedy_id == a;
                continue;
            }
            std::vector<int32_t> rid;
            std::vector<double> rp;
            ref_build(x, sp, &rid, &rp);
            bool ok = true;
            if (d.dense) {
                ok = rid.size() == (size_t)V;
                for (size_t i = 0; ok && i < rid.size(); ++i) ok = std::fabs(d.probs[(size_t)rid[i]] - rp[i]) < 1e-6;
            } else {
                ok = d.ids.size() == rid.size();
                for (size_t i = 0; ok && i < rid.size(); ++i) ok = d.ids[i] == rid[i] && std::fabs((double)d.probs[i] - rp[i]) < 1e-6;
            }
            CHECK(ok, "build differs from the reference: T %.2f k %d p %.2f minp %.2f (kept %zu vs %zu)", q.t, q.k, q.tp, q.mp, d.dense ? (size_t)V : d.ids.size(), rid.size());
            n_ok += ok;
        }
    }
    std::printf("build vs reference: %d/%d parameter sets agree\n", n_ok, n);
}

static void test_draws(int V) {
    Sampler smp(V);
    std::vector<float> x = rand_logits(V, 11, 3.0f);
    SamplingParams sp;
    sp.temperature = 0.9f;
    sp.top_k = 12;
    TargetDist d;
    smp.build(x.data(), sp, &d);
    const int n = 60000;
    std::vector<int> cnt(d.ids.size() + 1, 0);
    SpecRng rng(42);
    for (int i = 0; i < n; ++i) {
        const int32_t t = Sampler::draw(d, rng);
        size_t b = d.ids.size();
        for (size_t j = 0; j < d.ids.size(); ++j)
            if (d.ids[j] == t) b = j;
        ++cnt[b];
    }
    std::vector<double> ex;
    for (float q : d.probs) ex.push_back(q * n);
    int df;
    const double c = chi2(ex, cnt, &df);
    CHECK(cnt.back() == 0 && c < 30, "draws: chi-square %.1f df %d outside %d", c, df, cnt.back());
    std::printf("draws from a top_k 12 distribution: chi-square %.2f (df %d)\n", c, df);
}

// first emitted token of spec_accept with d ~ q must follow p
static void test_spec_accept(int V) {
    Sampler smp(V);
    std::vector<float> x = rand_logits(V, 5, 3.0f);
    SamplingParams sp;
    sp.temperature = 0.8f;
    sp.top_k = 8;
    TargetDist P;
    smp.build(x.data(), sp, &P);
    std::vector<float> logits((size_t)2 * V);   // row 0 = x, row 1 = x shifted (bonus row, irrelevant for the first token)
    std::memcpy(logits.data(), x.data(), (size_t)V * 4);
    std::memcpy(logits.data() + V, x.data(), (size_t)V * 4);
    auto bin_of = [&](int32_t t) {
        for (size_t i = 0; i < P.ids.size(); ++i)
            if (P.ids[i] == t) return (int)i;
        return (int)P.ids.size();
    };
    struct Q {
        const char* name;
        std::vector<float> q;   // dense, or empty = one-hot on `d`
        int32_t d;
    };
    std::vector<Q> qs;
    {   // q close to p (p with a bit of noise, renormalised over supp(p))
        Q q{"q ~ p", std::vector<float>((size_t)V, 0.0f), -1};
        double s = 0;
        for (size_t i = 0; i < P.ids.size(); ++i) s += q.q[(size_t)P.ids[i]] = P.probs[i] * (0.7f + 0.6f * (float)(i % 3) / 2);
        for (float& v : q.q) v = (float)(v / s);
        qs.push_back(q);
    }
    {   // q far from p: 0.5 on the least likely kept token, 0.3 outside the support, 0.2 on the best
        Q q{"q far from p (30 % outside supp p)", std::vector<float>((size_t)V, 0.0f), -1};
        q.q[(size_t)P.ids.back()] = 0.5f;
        q.q[(size_t)((P.ids[0] + 1234) % V)] = 0.3f;
        q.q[(size_t)P.ids[0]] = 0.2f;
        qs.push_back(q);
    }
    {   // uniform over the whole vocabulary
        Q q{"q uniform", std::vector<float>((size_t)V, 1.0f / V), -1};
        qs.push_back(q);
    }
    qs.push_back({"greedy draft = argmax", {}, P.ids[0]});
    qs.push_back({"greedy draft = 3rd best", {}, P.ids[2]});
    qs.push_back({"greedy draft outside supp p", {}, (int32_t)((P.ids[0] + 999) % V)});
    for (const Q& q : qs) {
        const int n = 200000;
        std::vector<int> cnt(P.ids.size() + 1, 0);
        SpecRng drng(777);
        int acc = 0;
        for (int i = 0; i < n; ++i) {
            int32_t d = q.d;
            if (!q.q.empty()) {   // d ~ q
                const double u = drng.uniform();
                double c = 0;
                d = V - 1;
                for (int t = 0; t < V; ++t) {
                    c += q.q[(size_t)t];
                    if (c > u) {
                        d = t;
                        break;
                    }
                }
            }
            SpecRng rng(10000 + (uint64_t)i);
            int32_t out[2];
            const int k = spec_accept(smp, logits.data(), 1, &d, q.q.empty() ? nullptr : q.q.data(), sp, rng, out);
            acc += k;
            ++cnt[(size_t)bin_of(out[0])];
        }
        std::vector<double> ex;
        for (float pr : P.probs) ex.push_back(pr * n);
        int df;
        const double c = chi2(ex, cnt, &df);
        CHECK(cnt.back() == 0, "%s: %d emitted tokens outside supp p", q.name, cnt.back());
        CHECK(c < 30, "%s: chi-square %.1f df %d", q.name, c, df);
        std::printf("spec_accept %-36s acceptance %.3f  chi-square %.2f (df %d) %s\n", q.name, (double)acc / n, c, df, c < 30 && cnt.back() == 0 ? "ok" : "FAILED");
    }
    // greedy params: accept iff draft == argmax, correction = argmax
    {
        SamplingParams g;
        g.temperature = 0.0f;
        int32_t d = P.ids[0], out[2];
        SpecRng rng(1);
        CHECK(spec_accept(smp, logits.data(), 1, &d, nullptr, g, rng, out) == 1 && out[0] == P.ids[0] && out[1] == P.ids[0], "greedy accept");
        d = P.ids[1];
        CHECK(spec_accept(smp, logits.data(), 1, &d, nullptr, g, rng, out) == 0 && out[0] == P.ids[0], "greedy reject");
        CHECK(rng.k == 0, "greedy mode consumed %llu uniforms", (unsigned long long)rng.k);
    }
    // seed threading: next_seed differs and depends on the number of draws
    {
        SpecRng a(5), b(5);
        a.uniform();
        CHECK(a.next_seed() != b.next_seed() && a.next_seed() != 5, "next_seed");
    }
}

int main() {
    const int V = 4096;
    test_build(V);
    test_draws(V);
    test_spec_accept(V);
    std::printf("test_sampler %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

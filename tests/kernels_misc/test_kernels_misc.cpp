// Randomised tests for ink_kernels_misc: every AVX-512 kernel against its scalar/double reference.  Owner: agent B2.
// Plain asserts, returns non-zero on failure.  Tolerances are the task's: elementwise <= 2e-6 relative, attention <= 1e-5.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "ink/kernels_misc.h"

using namespace ink;

static int g_fail = 0;
#define CHECK(cond, ...)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            ++g_fail;                                             \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);      \
            std::printf(__VA_ARGS__);                             \
            std::printf("\n");                                    \
        }                                                         \
    } while (0)

static std::mt19937_64 rng(20261121);
static float urand(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
static float nrand() { return std::normal_distribution<float>(0.0f, 1.0f)(rng); }
static float h16(float x) { return fp16_to_f32(f32_to_fp16(x)); }

static double rel(double a, double r) { return r == 0.0 ? std::fabs(a) : std::fabs(a - r) / std::fabs(r); }
// max |a-b| / max|ref|
static double normrel(const float* a, const float* ref, size_t n) {
    double mx = 0, mr = 0;
    for (size_t i = 0; i < n; ++i) { mx = std::max(mx, (double)std::fabs((double)a[i] - ref[i])); mr = std::max(mr, (double)std::fabs(ref[i])); }
    return mx / std::max(mr, 1e-30);
}
static bool close_abs(double a, double r, double rtol, double atol) { return std::fabs(a - r) <= rtol * std::fabs(r) + atol; }

// ------------------------------------------------------------------ 1. vector math
static double d_exp(double x) { return std::exp(x); }
static double d_tanh(double x) { return std::tanh(x); }
static double d_sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double d_cap(double x) { return 30.0 * std::tanh(x / 30.0); }
static void vsoftcap30(const float* x, float* y, size_t n) { vsoftcap(x, y, n, 30.0f); }

static void test_vecmath() {
    struct Fn { const char* name; void (*opt)(const float*, float*, size_t); double (*ref)(double); double lo, hi; };
    const Fn fns[] = {{"exp", vexp, d_exp, -87.3, 88.7}, {"tanh", vtanh, d_tanh, -60, 60}, {"sigmoid", vsigmoid, d_sigm, -87.3, 60},
                      {"softcap30", vsoftcap30, d_cap, -1500, 1500}};
    const size_t N = 1u << 21;
    for (const Fn& f : fns) {
        std::vector<float> x(N), y(N);
        for (size_t i = 0; i < N; ++i) {
            if (i & 1) x[i] = urand((float)f.lo, (float)f.hi);
            else {
                const float mag = std::exp(urand(std::log(1e-8f), std::log((float)std::max(std::fabs(f.lo), std::fabs(f.hi)))));
                x[i] = std::min(std::max((i & 2) ? mag : -mag, (float)f.lo), (float)f.hi);
            }
        }
        f.opt(x.data(), y.data(), N);
        double mx = 0;
        for (size_t i = 0; i < N; ++i) mx = std::max(mx, rel(y[i], (double)(float)f.ref((double)x[i])) );
        // compare against the true double value (not its float rounding) for the reported number
        double mx2 = 0;
        for (size_t i = 0; i < N; ++i) mx2 = std::max(mx2, rel(y[i], f.ref((double)x[i])));
        std::printf("  vecmath %-9s range [%g, %g]: max rel err vs double = %.3e\n", f.name, f.lo, f.hi, mx2);
        CHECK(mx2 <= 2e-6, "%s max rel err %.3e > 2e-6", f.name, mx2);
        (void)mx;
        // tails, in-place, no overrun
        for (size_t n : {0, 1, 2, 15, 16, 17, 31, 33, 47, 100, 1000}) {
            std::vector<float> o(n + 32, 777.0f), ip(x.begin(), x.begin() + n);
            f.opt(x.data(), o.data() + 8, n);
            f.opt(ip.data(), ip.data(), n);
            bool ok = true;
            for (size_t i = 0; i < n; ++i) ok = ok && o[8 + i] == y[i] && ip[i] == y[i];
            for (size_t i = 0; i < 8; ++i) ok = ok && o[i] == 777.0f;
            for (size_t i = 8 + n; i < o.size(); ++i) ok = ok && o[i] == 777.0f;
            CHECK(ok, "%s tail/in-place n=%zu", f.name, n);
        }
    }
    // SiTU
    {
        const size_t n = 1u << 20;
        std::vector<float> g(n), u(n), y(n), yr(n);
        for (size_t i = 0; i < n; ++i) {
            g[i] = (i & 1) ? urand(-60, 60) : nrand() * 3.0f;
            u[i] = (i & 2) ? urand(-400, 400) : nrand() * 10.0f;
            if (u[i] == 0.0f) u[i] = 1e-3f;
        }
        vsitu(g.data(), u.data(), y.data(), n);
        double mx = 0;
        for (size_t i = 0; i < n; ++i) {
            const double gd = g[i], ud = u[i];
            const double r = 4.0 * std::tanh(gd / 4.0) * (1.0 / (1.0 + std::exp(-gd))) * 25.0 * std::tanh(ud / 25.0);
            mx = std::max(mx, rel(y[i], r));
        }
        std::printf("  vecmath situ(4,25) g in [-60,60], u in [-400,400]: max rel err vs double = %.3e\n", mx);
        CHECK(mx <= 2e-6, "situ max rel err %.3e", mx);
        ref::vsitu(g.data(), u.data(), yr.data(), n);
        CHECK(normrel(y.data(), yr.data(), n) < 1e-6, "situ vs ref");
        for (int m = 0; m < 2; ++m) {  // fused SiTU + Q8_0 is bit-identical to the two-step form
            const size_t nn = 1152;
            std::vector<block_q8_0> a(nn / 32), b(nn / 32);
            std::vector<float> s(nn);
            vsitu(g.data(), u.data(), s.data(), nn);
            quantize_q8_0(s.data(), a.data(), nn, m ? Q8Mode::X86 : Q8Mode::Ref);
            vsitu_q8_0(g.data(), u.data(), b.data(), nn, 4.0f, 25.0f, m ? Q8Mode::X86 : Q8Mode::Ref);
            CHECK(std::memcmp(a.data(), b.data(), a.size() * 34) == 0, "vsitu_q8_0 != vsitu+quantize mode=%d", m);
        }
        // fused layout and in place
        std::vector<float> gu(2 * 1152), o1(1152), o2(1152);
        for (int i = 0; i < 1152; ++i) { gu[i] = g[i]; gu[1152 + i] = u[i]; }
        vsitu_fused(gu.data(), o1.data(), 1152);
        vsitu(g.data(), u.data(), o2.data(), 1152);
        CHECK(std::memcmp(o1.data(), o2.data(), 1152 * 4) == 0, "situ_fused != situ");
    }
}

// ------------------------------------------------------------------ 2/3. norms, q8_0, clips
static void test_norms() {
    for (size_t n : {1, 7, 16, 17, 31, 32, 64, 100, 128, 2304, 4099}) {
        std::vector<float> x(n), w(n), y(n), yr(n);
        for (size_t i = 0; i < n; ++i) { x[i] = nrand() * (1.0f + 5.0f * (i % 7 == 0)); w[i] = 0.5f + urand(0, 1.5f); }
        for (int use_w = 0; use_w < 2; ++use_w) {
            rmsnorm(x.data(), use_w ? w.data() : nullptr, y.data(), n);
            ref::rmsnorm(x.data(), use_w ? w.data() : nullptr, yr.data(), n);
            double mx = 0;
            for (size_t i = 0; i < n; ++i) mx = std::max(mx, rel(y[i], yr[i]));
            CHECK(mx <= 2e-6, "rmsnorm n=%zu w=%d rel %.3e", n, use_w, mx);
            std::vector<float> ip = x;
            rmsnorm(ip.data(), use_w ? w.data() : nullptr, ip.data(), n);
            CHECK(std::memcmp(ip.data(), y.data(), n * 4) == 0, "rmsnorm in place n=%zu", n);
        }
    }
    // Q8_0: vector quantiser == scalar ggml reference, bit exact, both modes; fused == two-step
    for (int mode_i = 0; mode_i < 2; ++mode_i) {
        const Q8Mode mode = mode_i ? Q8Mode::X86 : Q8Mode::Ref;
        for (size_t n : {32, 64, 96, 2304, 4096}) {
            std::vector<float> x(n), w(n), y(n);
            for (size_t i = 0; i < n; ++i) { x[i] = nrand() * ((i / 32) % 5 == 0 ? 0.001f : 3.0f); w[i] = urand(0.3f, 2.0f); }
            for (size_t i = 0; i < 32 && i < n; ++i) x[i] = 0.0f;  // an all-zero block
            std::vector<block_q8_0> a(n / 32), b(n / 32), c(n / 32);
            quantize_q8_0(x.data(), a.data(), n, mode);
            ref::quantize_q8_0(x.data(), b.data(), n, mode);
            CHECK(std::memcmp(a.data(), b.data(), a.size() * 34) == 0, "quantize_q8_0 != ref n=%zu mode=%d", n, mode_i);
            rmsnorm(x.data(), w.data(), y.data(), n);
            quantize_q8_0(y.data(), a.data(), n, mode);
            rmsnorm_q8_0(x.data(), w.data(), c.data(), n, 1e-6f, mode);
            CHECK(std::memcmp(a.data(), c.data(), a.size() * 34) == 0, "rmsnorm_q8_0 != rmsnorm+quantize n=%zu mode=%d", n, mode_i);
        }
        // exact ties: amax = 127 -> d = 1, id = 1; values k+0.5 round away (Ref) / to even (X86)
        float t[32];
        for (int i = 0; i < 32; ++i) t[i] = (i % 2 ? -1.0f : 1.0f) * ((i / 2) + 0.5f);
        t[31] = 127.0f;
        block_q8_0 a, b;
        quantize_q8_0(t, &a, 32, mode);
        ref::quantize_q8_0(t, &b, 32, mode);
        CHECK(std::memcmp(&a, &b, 34) == 0, "q8 tie handling mode=%d", mode_i);
    }
    {  // thread-splittable forms (rms_scale + ranged apply) are bit-identical to the one-shot kernels
        const size_t n = 2304;
        std::vector<float> x(n), w(n), y1(n), y2(n);
        for (size_t i = 0; i < n; ++i) { x[i] = nrand() * 2; w[i] = urand(0.5f, 1.5f); }
        rmsnorm(x.data(), w.data(), y1.data(), n);
        const float s = rms_scale(x.data(), n);
        rmsnorm_apply(x.data(), w.data(), s, y2.data(), 0, 803);
        rmsnorm_apply(x.data(), w.data(), s, y2.data(), 803, n);
        CHECK(std::memcmp(y1.data(), y2.data(), n * 4) == 0, "rmsnorm_apply split");
        for (int m = 0; m < 2; ++m) {
            std::vector<block_q8_0> a(n / 32), b(n / 32);
            rmsnorm_q8_0(x.data(), w.data(), a.data(), n, 1e-6f, m ? Q8Mode::X86 : Q8Mode::Ref);
            rmsnorm_apply_q8_0(x.data(), w.data(), s, b.data(), 0, 20, m ? Q8Mode::X86 : Q8Mode::Ref);
            rmsnorm_apply_q8_0(x.data(), w.data(), s, b.data(), 20, n / 32, m ? Q8Mode::X86 : Q8Mode::Ref);
            CHECK(std::memcmp(a.data(), b.data(), a.size() * 34) == 0, "rmsnorm_apply_q8_0 split mode=%d", m);
        }
    }
    // L2 clip (Q-like with chan scale, K-like, R-like with tau)
    struct C { int nh, dim; float thr, post; bool scale; };
    for (const C& c : {C{24, 128, 22.627f, 1.0f, true}, C{8, 128, 22.627f, 1.0f, false}, C{24, 16, 8.0f, 1.17f, true}, C{3, 21, 5.0f, 1.0f, true}}) {
        std::vector<float> x(c.nh * c.dim), sc(c.nh * c.dim);
        for (auto& v : x) v = nrand() * urand(0.3f, 8.0f);
        for (auto& v : sc) v = urand(0.5f, 3.0f);
        std::vector<float> a = x, b = x;
        l2clip_heads(a.data(), c.scale ? sc.data() : nullptr, c.nh, c.dim, c.thr, c.post);
        ref::l2clip_heads(b.data(), c.scale ? sc.data() : nullptr, c.nh, c.dim, c.thr, c.post);
        double mx = 0;
        for (size_t i = 0; i < a.size(); ++i) mx = std::max(mx, rel(a[i], b[i]));
        CHECK(mx <= 2e-6, "l2clip nh=%d dim=%d rel %.3e", c.nh, c.dim, mx);
        for (int h = 0; h < c.nh && c.post == 1.0f; ++h) {
            double ss = 0;
            for (int i = 0; i < c.dim; ++i) ss += (double)a[h * c.dim + i] * a[h * c.dim + i];
            CHECK(std::sqrt(ss) <= c.thr * (1 + 1e-5), "l2clip norm %f > thr", std::sqrt(ss));
        }
    }
    {  // V norm
        std::vector<float> x(8 * 128), a(8 * 128), b(8 * 128);
        for (auto& v : x) v = nrand() * 3;
        vnorm_heads(x.data(), a.data(), 8, 128);
        ref::vnorm_heads(x.data(), b.data(), 8, 128);
        double mx = 0;
        for (size_t i = 0; i < a.size(); ++i) mx = std::max(mx, rel(a[i], b[i]));
        CHECK(mx <= 2e-6, "vnorm rel %.3e", mx);
    }
}

// ------------------------------------------------------------------ 4. short conv
static void test_sconv() {
    struct Cfg { size_t C; int head_dim; bool dyn; size_t n; };
    for (const Cfg& cf : {Cfg{2304, 128, false, 9}, Cfg{1024, 128, false, 1}, Cfg{1024, 128, true, 11}, Cfg{100, 128, false, 5}, Cfg{1024, 128, true, 1}}) {
        const size_t C = cf.C, n = cf.n, nh = C / cf.head_dim;
        std::vector<float> w(C * 4), hist(C * 3), x(n * C), dyn(n * nh * 4);
        for (auto& v : w) v = nrand() * 0.7f;
        for (auto& v : hist) v = nrand();
        for (auto& v : x) v = nrand();
        for (auto& v : dyn) v = nrand() * 0.5f;
        const float* dp = cf.dyn ? dyn.data() : nullptr;
        // reference (ggml layout)
        std::vector<float> yr(n * C), hr = hist;
        ref::sconv_seq(x.data(), hr.data(), w.data(), yr.data(), n, C, dp, cf.head_dim);
        // optimised: SoA
        std::vector<float> wsoa(4 * C), hs(3 * C);
        sconv_repack_static(w.data(), wsoa.data(), C);
        for (size_t c = 0; c < C; ++c) for (int i = 0; i < 3; ++i) hs[i * C + c] = hist[c * 3 + i];
        auto check_hist = [&](const std::vector<float>& h, const char* tag) {
            for (size_t c = 0; c < C; ++c) for (int i = 0; i < 3; ++i)
                CHECK(close_abs(h[i * C + c], hr[c * 3 + i], 1e-6, 1e-6), "sconv %s hist C=%zu", tag, C);
        };
        std::vector<float> y1(n * C), y2(n * C), y3(n * C);
        // (a) one-shot sequence
        std::vector<float> h1 = hs;
        sconv_seq(x.data(), h1.data(), wsoa.data(), y1.data(), n, C, 0, C, dp, cf.head_dim);
        // (b) token by token
        std::vector<float> h2 = hs;
        for (size_t t = 0; t < n; ++t) sconv_step(x.data() + t * C, h2.data(), wsoa.data(), y2.data() + t * C, C, 0, C, dp ? dp + t * nh * 4 : nullptr, cf.head_dim);
        // (c) channel ranges split in two (threading): must be bit identical to (a)
        std::vector<float> h3 = hs;
        const size_t mid = (C / 2) / 16 * 16;
        sconv_seq(x.data(), h3.data(), wsoa.data(), y3.data(), n, C, 0, mid, dp, cf.head_dim);
        sconv_seq(x.data(), h3.data(), wsoa.data(), y3.data(), n, C, mid, C, dp, cf.head_dim);
        double mx = 0;
        for (size_t i = 0; i < n * C; ++i) {
            CHECK(close_abs(y1[i], yr[i], 2e-6, 4e-6), "sconv seq C=%zu dyn=%d i=%zu %g vs %g", C, cf.dyn, i, y1[i], yr[i]);
            CHECK(close_abs(y2[i], yr[i], 2e-6, 4e-6), "sconv step C=%zu dyn=%d i=%zu", C, cf.dyn, i);
            mx = std::max(mx, std::fabs((double)y1[i] - yr[i]));
        }
        CHECK(std::memcmp(y1.data(), y3.data(), n * C * 4) == 0 && std::memcmp(h1.data(), h3.data(), 3 * C * 4) == 0, "sconv channel split not bit-identical");
        CHECK(std::memcmp(y1.data(), y2.data(), n * C * 4) == 0, "sconv seq != step (C=%zu)", C);
        check_hist(h1, "seq");
        check_hist(h2, "step");
        std::printf("  sconv C=%zu n=%zu dyn=%d: max abs err vs double %.2e\n", C, n, cf.dyn, mx);
    }
}

// ------------------------------------------------------------------ 6/7. router, depth mix, controller, converters
static void test_router_mix() {
    // selection: exact vs std::sort reference on distinct scores
    for (int trial = 0; trial < 2000; ++trial) {
        std::vector<float> s(128);
        for (auto& v : s) v = urand(0, 1);
        int a[8], b[8];
        router_select(s.data(), 128, 8, a);
        ref::router_select(s.data(), 128, 8, b);
        for (int i = 0; i < 8; ++i) CHECK(a[i] == b[i], "router_select trial %d pos %d: %d vs %d", trial, i, a[i], b[i]);
    }
    {  // ties: lowest index first
        std::vector<float> s(128, 0.5f);
        int a[8];
        router_select(s.data(), 128, 8, a);
        for (int i = 0; i < 8; ++i) CHECK(a[i] == i, "router tie order %d", a[i]);
        s[100] = s[20] = s[77] = 0.9f;
        router_select(s.data(), 128, 8, a);
        CHECK(a[0] == 20 && a[1] == 77 && a[2] == 100 && a[3] == 0, "router tie order 2");
    }
    int nskip = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        std::vector<float> lg(128), bias(128);
        for (auto& v : lg) v = nrand() * 2.5f;
        for (auto& v : bias) v = nrand() * 0.1f;
        const bool use_bias = trial % 4 != 3;
        int a[8], b[8];
        float wa[8], wb[8];
        router_topk(lg.data(), use_bias ? bias.data() : nullptr, 128, 8, 8.0f, 1e-20f, a, wa);
        ref::router_topk(lg.data(), use_bias ? bias.data() : nullptr, 128, 8, 8.0f, 1e-20f, b, wb);
        // skip near-ties at the selection boundary (sigmoid implementations differ by ~3e-7)
        std::vector<float> sel(128);
        for (int i = 0; i < 128; ++i) sel[i] = 1.0f / (1.0f + std::exp(-lg[i])) + (use_bias ? bias[i] : 0.0f);
        std::vector<float> ss = sel;
        std::sort(ss.begin(), ss.end(), std::greater<float>());
        bool near = false;
        for (int i = 0; i < 9; ++i) near = near || (ss[i] - ss[i + 1] < 3e-6f);
        if (near) { ++nskip; continue; }
        double wsum = 0;
        for (int i = 0; i < 8; ++i) {
            CHECK(a[i] == b[i], "router_topk id %d: %d vs %d", i, a[i], b[i]);
            CHECK(rel(wa[i], wb[i]) <= 2e-6, "router weight rel %.3e", rel(wa[i], wb[i]));
            wsum += wa[i];
        }
        CHECK(std::fabs(wsum - 8.0) < 1e-4, "router weights sum %f", wsum);
    }
    std::printf("  router: 3000 random trials (%d near-tie trials skipped), ids exact, weights <= 2e-6\n", nskip);

    // depth mix
    for (size_t n : {2304, 100, 16}) {
        for (int ns : {2, 3, 5, 8}) {
            for (int nq : {1, 3, 4}) {
                std::vector<std::vector<float>> st(ns, std::vector<float>(n));
                std::vector<const float*> sp(ns);
                for (int s = 0; s < ns; ++s) {
                    const float amp = std::pow(10.0f, urand(-2, 2));
                    for (auto& v : st[s]) v = nrand() * amp + 0.3f * amp;
                    sp[s] = st[s].data();
                }
                std::vector<float> q(nq * n);
                for (auto& v : q) v = nrand() * 3.0f;
                std::vector<std::vector<float>> oa(nq, std::vector<float>(n)), ob(nq, std::vector<float>(n));
                std::vector<float*> pa(nq), pb(nq);
                for (int i = 0; i < nq; ++i) { pa[i] = oa[i].data(); pb[i] = ob[i].data(); }
                const float scale = 1.0f / std::sqrt(2304.0f);
                depth_mix(sp.data(), ns, q.data(), nq, pa.data(), n, scale);
                ref::depth_mix(sp.data(), ns, q.data(), nq, pb.data(), n, scale);
                for (int i = 0; i < nq; ++i) CHECK(normrel(oa[i].data(), ob[i].data(), n) <= 2e-6, "depth_mix n=%zu ns=%d nq=%d err %.3e", n, ns, nq, normrel(oa[i].data(), ob[i].data(), n));
            }
        }
    }
    // controller
    {
        const size_t n = 2304;
        std::vector<float> cur(n), msg(n), gp(n), fp(n), a(n), b(n), gi(2 * n), gir(2 * n);
        for (size_t i = 0; i < n; ++i) { cur[i] = nrand(); msg[i] = nrand() * 2; gp[i] = nrand() * 2; fp[i] = nrand() * 2; }
        ctrl_gate_input(cur.data(), msg.data(), gi.data(), n);
        ref::ctrl_gate_input(cur.data(), msg.data(), gir.data(), n);
        CHECK(normrel(gi.data(), gir.data(), 2 * n) <= 2e-6, "ctrl_gate_input");
        ctrl_apply(cur.data(), msg.data(), gp.data(), fp.data(), a.data(), n);
        ref::ctrl_apply(cur.data(), msg.data(), gp.data(), fp.data(), b.data(), n);
        double mx = 0;
        for (size_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs((double)a[i] - b[i]));
        CHECK(mx <= 2e-6 * 4, "ctrl_apply abs err %.3e", mx);
        std::vector<float> ip = cur;
        ctrl_apply(ip.data(), msg.data(), gp.data(), fp.data(), ip.data(), n);
        CHECK(std::memcmp(ip.data(), a.data(), n * 4) == 0, "ctrl_apply in place");
    }
}

static void test_converters() {
    for (size_t n : {0, 1, 15, 16, 17, 100, 2304, 5000}) {
        std::vector<uint16_t> bf(n), hf(n), bf2(n), hf2(n), bfr(n), hfr(n);
        std::vector<float> x(n), a(n), b(n), c(n);
        for (size_t i = 0; i < n; ++i) { x[i] = nrand() * 100.0f; bf[i] = f32_to_bf16(x[i]); hf[i] = f32_to_fp16(x[i]); }
        bf16_to_f32(bf.data(), a.data(), n);
        f16_to_f32(hf.data(), b.data(), n);
        for (size_t i = 0; i < n; ++i) { CHECK(a[i] == bf16_to_f32(bf[i]), "bf16_to_f32 i=%zu", i); CHECK(b[i] == fp16_to_f32(hf[i]), "f16_to_f32 i=%zu", i); }
        f32_to_bf16(x.data(), bf2.data(), n);
        f32_to_f16(x.data(), hf2.data(), n);
        CHECK(n == 0 || (std::memcmp(bf.data(), bf2.data(), n * 2) == 0 && std::memcmp(hf.data(), hf2.data(), n * 2) == 0), "f32->bf16/f16 n=%zu", n);
        // fused helpers
        std::vector<float> y(n), z(n);
        for (size_t i = 0; i < n; ++i) { y[i] = nrand(); z[i] = nrand(); }
        vadd(y.data(), z.data(), a.data(), n);
        vmul(y.data(), z.data(), b.data(), n);
        vscale(y.data(), 0.37f, c.data(), n);
        for (size_t i = 0; i < n; ++i) { CHECK(a[i] == y[i] + z[i], "vadd"); CHECK(b[i] == y[i] * z[i], "vmul"); CHECK(c[i] == y[i] * 0.37f, "vscale"); }
        std::vector<float> ax = y;
        vaxpy(ax.data(), 1.5f, z.data(), n);
        vadd_scale(y.data(), z.data(), 0.2f, a.data(), n);
        vmul_sigmoid(y.data(), z.data(), b.data(), n);
        for (size_t i = 0; i < n; ++i) {
            CHECK(close_abs(ax[i], y[i] + 1.5f * z[i], 1e-6, 1e-6), "vaxpy");
            CHECK(a[i] == (y[i] + z[i]) * 0.2f, "vadd_scale");
            CHECK(close_abs(b[i], y[i] / (1.0 + std::exp(-(double)z[i])), 2e-6, 1e-30), "vmul_sigmoid");
        }
    }
}

// ------------------------------------------------------------------ 5. attention
struct TestCache {
    KvLayer L;
    std::vector<float> Kf, Vf;  // plain fp16-rounded K, V [pos][1024]
    TestCache(bool ring, uint32_t n_blocks, size_t max_pos) {
        L.ring = ring;
        L.n_blocks = n_blocks;
        L.k = (uint16_t*)std::aligned_alloc(64, L.bytes_per_tensor());
        L.v = (uint16_t*)std::aligned_alloc(64, L.bytes_per_tensor());
        std::memset(L.k, 0, L.bytes_per_tensor());
        std::memset(L.v, 0, L.bytes_per_tensor());
        Kf.resize(max_pos * 1024);
        Vf.resize(max_pos * 1024);
    }
    ~TestCache() { std::free(L.k); std::free(L.v); }
    void gen(size_t pos) {
        for (int i = 0; i < 1024; ++i) { Kf[pos * 1024 + i] = h16(nrand() * 0.8f); Vf[pos * 1024 + i] = h16(nrand()); }
    }
    void append(size_t pos) { kv_append(L, (int64_t)pos, &Kf[pos * 1024], &Vf[pos * 1024]); }
};

static void test_attention() {
    std::vector<float> rb(1024 * 16), table(16 * kRelTableLen);
    for (auto& v : rb) v = nrand() * 0.5f;
    attn_repack_rel_bias(rb.data(), table.data());
    // repack spot checks
    CHECK(table[3 * kRelTableLen + 64 + 0] == rb[1023 * 16 + 3] && table[5 * kRelTableLen + 64 + 1023] == rb[0 * 16 + 5] &&
          table[7 * kRelTableLen + 10] == 0.0f && table[7 * kRelTableLen + 64 + 1024 + 10] == 0.0f, "rel table repack");

    {  // kv_append by disjoint head ranges == one call
        TestCache a(true, 16, 300), b(true, 16, 300);
        for (size_t p = 0; p < 300; ++p) {
            a.gen(p);
            a.append(p);
            kv_append(b.L, (int64_t)p, &a.Kf[p * 1024], &a.Vf[p * 1024], 0, 3);
            kv_append(b.L, (int64_t)p, &a.Kf[p * 1024], &a.Vf[p * 1024], 3, 8);
        }
        CHECK(std::memcmp(a.L.k, b.L.k, a.L.bytes_per_tensor()) == 0 && std::memcmp(a.L.v, b.L.v, a.L.bytes_per_tensor()) == 0, "kv_append head split");
    }
    const std::vector<int> checks = {0, 1, 5, 62, 63, 64, 65, 127, 500, 1022, 1023, 1024, 1025, 1087, 1088, 1500, 2047, 2048, 2049, 3000, 4000, 4999};
    for (int mode = 0; mode < 2; ++mode) {  // 0 = SWA ring (cap 1024), 1 = global
        const bool swa = mode == 0;
        const size_t P = 5000;
        TestCache tc(swa, swa ? 16 : 80, P);
        std::vector<float> q(3072), r(384), sink(24), gate(3072), out(3072), outr(3072);
        double worst = 0;
        for (size_t p = 0; p < P; ++p) {
            tc.gen(p);
            tc.append(p);
            if (std::find(checks.begin(), checks.end(), (int)p) == checks.end()) continue;
            for (auto& v : q) v = nrand();
            for (auto& v : r) v = nrand();
            for (auto& v : sink) v = nrand();
            for (auto& v : gate) v = nrand() * 2;
            const float tau = swa ? 1.0f : 1.0f + 0.1f * std::log(std::max(1.0f, (float)(p + 1) / 8192.0f)) + (p % 3) * 0.05f;
            AttnQuery aq{q.data(), r.data(), sink.data(), gate.data(), (int64_t)p, tau};
            ref::attn_decode(tc.Kf.data(), tc.Vf.data(), (int64_t)p, swa, rb.data(), q.data(), r.data(), sink.data(), gate.data(), tau, outr.data());
            for (int nsplit : {1, 2, 5}) {
                std::vector<AttnPartial> scratch(8 * nsplit);
                attn_decode(tc.L, table.data(), aq, nsplit, scratch.data(), out.data());
                const double e = normrel(out.data(), outr.data(), 3072);
                worst = std::max(worst, e);
                CHECK(e <= 1e-5, "attn %s pos=%zu nsplit=%d err %.3e", swa ? "swa" : "global", p, nsplit, e);
                if (nsplit == 5) {  // thread-order independence: run the (g,s) items in reverse order, finish in any group order
                    std::vector<AttnPartial> s2(8 * nsplit);
                    std::vector<float> out2(3072);
                    for (int g = 7; g >= 0; --g) for (int s = nsplit - 1; s >= 0; --s) attn_decode_task(tc.L, table.data(), aq, g, s, nsplit, &s2[g * nsplit + s]);
                    for (int g = 0; g < 8; ++g) attn_decode_finish(aq, g, &s2[g * nsplit], nsplit, out2.data());
                    CHECK(std::memcmp(out.data(), out2.data(), 3072 * 4) == 0, "attn not bit-reproducible across item order pos=%zu", p);
                }
            }
            // no gate variant
            AttnQuery ang = aq;
            ang.gate = nullptr;
            std::vector<AttnPartial> scratch(8);
            attn_decode(tc.L, table.data(), ang, 1, scratch.data(), out.data());
            ref::attn_decode(tc.Kf.data(), tc.Vf.data(), (int64_t)p, swa, rb.data(), q.data(), r.data(), sink.data(), nullptr, tau, outr.data());
            CHECK(normrel(out.data(), outr.data(), 3072) <= 1e-5, "attn no-gate pos=%zu", p);
        }
        std::printf("  attention decode %-6s positions 0..%zu, nsplit 1/2/5: worst err vs double %.3e\n", swa ? "SWA" : "global", P - 1, worst);
    }

    // prefill: ring of 2048 slots (chunk <= 1025) and global; query tokens vs per-token double reference
    for (int mode = 0; mode < 2; ++mode) {
        const bool swa = mode == 0;
        const size_t pre = swa ? 1500 : 2900, n = swa ? 701 : 301, P = pre + n;
        TestCache tc(swa, swa ? 32 : 60, P);
        for (size_t p = 0; p < pre; ++p) { tc.gen(p); tc.append(p); }
        for (size_t p = pre; p < P; ++p) tc.gen(p);
        kv_append_n(tc.L, (int64_t)pre, (int)n, &tc.Kf[pre * 1024], &tc.Vf[pre * 1024]);
        std::vector<float> q(n * 3072), r(n * 384), sink(n * 24), gate(n * 3072), tau(n), out(n * 3072, -1.0f), outr(3072);
        for (auto& v : q) v = nrand();
        for (auto& v : r) v = nrand();
        for (auto& v : sink) v = nrand();
        for (auto& v : gate) v = nrand() * 2;
        for (size_t t = 0; t < n; ++t) tau[t] = swa ? 1.0f : 1.0f + 0.1f * std::log(std::max(1.0f, (float)(pre + t + 1) / 8192.0f)) + 0.1f;
        AttnPrefill pf{q.data(), r.data(), sink.data(), gate.data(), tau.data(), (int64_t)pre, (int)n};
        for (int g = 0; g < 8; ++g) {  // two token ranges per group, the second starting at an odd token
            attn_prefill_group(tc.L, table.data(), pf, g, 0, 101, out.data());
            attn_prefill_group(tc.L, table.data(), pf, g, 101, (int)n, out.data());
        }
        double worst = 0;
        for (size_t t = 0; t < n; ++t) {
            if (!(t % 13 == 0 || t == n - 1 || t == 100 || t == 101)) continue;
            ref::attn_decode(tc.Kf.data(), tc.Vf.data(), (int64_t)(pre + t), swa, rb.data(), &q[t * 3072], &r[t * 384], &sink[t * 24], &gate[t * 3072], tau[t], outr.data());
            const double e = normrel(&out[t * 3072], outr.data(), 3072);
            worst = std::max(worst, e);
            CHECK(e <= 1e-5, "prefill %s token %zu err %.3e", swa ? "swa" : "global", t, e);
        }
        std::printf("  attention prefill %-6s pos0=%zu n=%zu: worst err vs double %.3e\n", swa ? "SWA" : "global", pre, n, worst);
    }
}

int main() {
    std::printf("kernels_misc tests\n");
    test_vecmath();
    test_norms();
    test_sconv();
    test_router_mix();
    test_converters();
    test_attention();
    std::printf(g_fail ? "FAILED: %d checks\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}

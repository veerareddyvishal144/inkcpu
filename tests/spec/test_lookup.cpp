// test_lookup (agent L): the session-lookup index and its width policy, no engine. LookupIndex::find against the brute-force reference on random (small alphabet: long
// chains) and repetitive (copied substrings: long matches) token streams, continuation() against the virtual-sequence definition (incl. cyclic self-extension), truncate /
// reset consistency (the incrementally maintained index must equal one rebuilt from the shorter history, chain for chain), unknown padding never matches, and the policy:
// all-accept history widens to T = 8, all-reject history stops firing in that bucket only, timing model from measured steps.
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

static uint64_t g_rng = 12345;
static uint32_t rnd() {
    g_rng = spec_mix64(g_rng);
    return (uint32_t)(g_rng >> 32);
}

// reference continuation on the virtual sequence s = hist ++ ctx ++ out
static std::vector<int32_t> ref_cont(const std::vector<int32_t>& hist, int64_t end, const std::vector<int32_t>& ctx, int n, int32_t max_id) {
    std::vector<int32_t> s = hist;
    s.insert(s.end(), ctx.begin(), ctx.end());
    std::vector<int32_t> out;
    for (int k = 0; k < n; ++k) {
        const size_t i = (size_t)(end + 1 + k);
        const int32_t t = s[i];
        if (t < 0 || t >= max_id) break;
        out.push_back(t);
        s.push_back(t);
    }
    return out;
}

static void compare_find(const LookupIndex& ix, const std::vector<int32_t>& hist, const std::vector<int32_t>& ctx, const char* tag, int step) {
    const LookupIndex::Match a = ix.find(ctx.data(), (int)ctx.size(), 1 << 30, 64);
    const LookupIndex::Match r = ix.find_reference(ctx.data(), (int)ctx.size(), 64);
    CHECK(a.len == r.len && a.end == r.end, "%s step %d: find (%d @ %lld) != reference (%d @ %lld)", tag, step, a.len, (long long)a.end, r.len, (long long)r.end);
    const LookupIndex::Match b = ix.find(ctx.data(), (int)ctx.size(), 16, 64);
    if (r.len) CHECK(b.len >= ix.n_key() && b.len <= r.len, "%s step %d: 16-candidate find %d (ref %d)", tag, step, b.len, r.len);
    else CHECK(b.len == 0, "%s step %d: 16-candidate find %d but no reference match", tag, step, b.len);
    if (a.len) {
        int32_t out[8];
        const int n = ix.continuation(a.end, ctx.data(), (int)ctx.size(), out, 7, 1 << 30);
        std::vector<int32_t> ref = ref_cont(hist, a.end, ctx, 7, 1 << 30);
        CHECK(n == (int)ref.size() && std::memcmp(out, ref.data(), (size_t)n * 4) == 0, "%s step %d: continuation differs (n %d vs %zu)", tag, step, n, ref.size());
    }
}

static void stream_test(const char* tag, int V, int N, bool copies) {
    LookupIndex ix(8);
    std::vector<int32_t> hist;
    int step = 0;
    while ((int)hist.size() < N) {
        if (copies && hist.size() > 64 && rnd() % 4 == 0) {   // copy a substring of 8..40 tokens from earlier
            const int len = 8 + (int)(rnd() % 33);
            const size_t from = rnd() % (hist.size() - (size_t)len);
            for (int i = 0; i < len; ++i) {
                const int32_t t = hist[from + (size_t)i];
                ix.append(t);
                hist.push_back(t);
            }
        } else {
            const int32_t t = (int32_t)(rnd() % (uint32_t)V);
            ix.append(t);
            hist.push_back(t);
        }
        if (++step % 3 == 0) {
            std::vector<int32_t> ctx = {(int32_t)(rnd() % (uint32_t)V)};
            if (copies && rnd() % 2) ctx[0] = hist[hist.size() - 1 - rnd() % 5];
            compare_find(ix, hist, ctx, tag, step);
            if (rnd() % 4 == 0) {   // multi-token ctx (anchor + drafts so far)
                ctx.push_back(hist[rnd() % hist.size()]);
                ctx.push_back((int32_t)(rnd() % (uint32_t)V));
                compare_find(ix, hist, ctx, tag, step);
            }
        }
    }
    CHECK(ix.size() == (int64_t)hist.size(), "%s: size", tag);
    for (size_t i = 0; i < hist.size(); i += 97) CHECK(ix.at((int64_t)i) == hist[i], "%s: at(%zu)", tag, i);
    std::printf("  %s: %d tokens, %d queries ok, index %.1f KB\n", tag, N, step / 3, ix.bytes() / 1024.0);
}

// periodic text: the continuation must cycle through the query itself
static void periodic_test() {
    LookupIndex ix(8);
    std::vector<int32_t> hist;
    for (int i = 0; i < 40; ++i) {
        ix.append(7);
        hist.push_back(7);
    }
    std::vector<int32_t> ctx = {7};
    compare_find(ix, hist, ctx, "periodic", 0);
    int32_t out[7];
    const LookupIndex::Match m = ix.find(ctx.data(), 1, 16, 64);
    CHECK(m.len == 41 || m.len == 40 || m.len >= 8, "periodic match %d", m.len);
    CHECK(m.end == 39, "periodic end %lld", (long long)m.end);
    const int n = ix.continuation(m.end, ctx.data(), 1, out, 7, 1 << 30);
    CHECK(n == 7, "periodic continuation n %d", n);
    for (int i = 0; i < n; ++i) CHECK(out[i] == 7, "periodic out[%d] = %d", i, out[i]);
    // pattern a b c a b c ... with anchor completing the period
    LookupIndex iy(8);
    std::vector<int32_t> h2;
    for (int i = 0; i < 30; ++i) {
        iy.append(i % 3);
        h2.push_back(i % 3);
    }
    ctx = {0};   // position 30 holds 0 (30 % 3)
    compare_find(iy, h2, ctx, "period3", 0);
    const LookupIndex::Match m2 = iy.find(ctx.data(), 1, 16, 64);
    CHECK(m2.len >= 8 && m2.end == 27, "period3 match %d @ %lld", m2.len, (long long)m2.end);
    const int n2 = iy.continuation(m2.end, ctx.data(), 1, out, 7, 1 << 30);
    CHECK(n2 == 7, "period3 n %d", n2);
    for (int i = 0; i < n2; ++i) CHECK(out[i] == (31 + i) % 3, "period3 out[%d] = %d", i, out[i]);
    std::printf("  periodic ok\n");
}

static bool same_chains(const LookupIndex& a, const LookupIndex& b, const std::vector<int32_t>& hist, int V, int queries) {
    for (int q = 0; q < queries; ++q) {
        std::vector<int32_t> ctx = {(int32_t)(rnd() % (uint32_t)V)};
        if (!hist.empty() && rnd() % 2) ctx[0] = hist[hist.size() - 1 - rnd() % std::min<size_t>(5, hist.size())];
        for (int mc : {1, 3, 16, 1 << 30}) {
            const LookupIndex::Match x = a.find(ctx.data(), 1, mc, 64), y = b.find(ctx.data(), 1, mc, 64);
            if (x.len != y.len || x.end != y.end || x.candidates != y.candidates) {
                std::printf("    chains differ: mc %d: (%d @ %lld, %d cand) vs rebuilt (%d @ %lld, %d cand), hist %zu\n", mc, x.len, (long long)x.end, x.candidates, y.len,
                            (long long)y.end, y.candidates, hist.size());
                return false;
            }
        }
    }
    return true;
}

static void rewind_test(const char* tag, int V) {
    LookupIndex ix(8);
    std::vector<int32_t> hist;
    int ops = 0, truncs = 0, resets = 0;
    for (int it = 0; it < 400; ++it) {
        const int kind = (int)(rnd() % 10);
        if (kind < 6) {   // append 1..60 tokens, sometimes copies
            const int k = 1 + (int)(rnd() % 60);
            for (int i = 0; i < k; ++i) {
                int32_t t = (int32_t)(rnd() % (uint32_t)V);
                if (hist.size() > 20 && rnd() % 3 == 0) t = hist[hist.size() - 1 - rnd() % 20 + 1 - 1];
                ix.append(t);
                hist.push_back(t);
            }
        } else if (kind < 9) {   // truncate to a random earlier position (a rewind)
            const int64_t pos = hist.empty() ? 0 : (int64_t)(rnd() % (hist.size() + 1));
            ix.truncate(pos);
            hist.resize((size_t)pos);
            ++truncs;
        } else {
            if (rnd() % 2) {
                ix.reset();
                hist.clear();
                ++resets;
            } else {   // sync to a position: pad or truncate
                const int64_t pos = (int64_t)(rnd() % 300);
                ix.sync(pos);
                if ((int64_t)hist.size() > pos) hist.resize((size_t)pos);
                while ((int64_t)hist.size() < pos) hist.push_back(-1);
            }
        }
        ++ops;
        CHECK(ix.size() == (int64_t)hist.size(), "%s op %d: size %lld vs %zu", tag, it, (long long)ix.size(), hist.size());
        // rebuild from scratch and compare chain for chain
        LookupIndex fresh(8);
        for (int32_t t : hist) fresh.append(t);
        if (!same_chains(ix, fresh, hist, V, 6)) {
            CHECK(false, "%s op %d (kind %d): incremental index != rebuilt index", tag, it, kind);
            break;
        }
        if (it % 20 == 0) {
            std::vector<int32_t> ctx = {(int32_t)(rnd() % (uint32_t)V)};
            compare_find(ix, hist, ctx, tag, it);
        }
    }
    std::printf("  %s: %d ops (%d truncates, %d resets) consistent with a rebuilt index\n", tag, ops, truncs, resets);
}

static void unknown_test() {
    LookupIndex ix(8);
    std::vector<int32_t> hist;
    for (int i = 0; i < 20; ++i) {
        ix.append(i % 4);
        hist.push_back(i % 4);
    }
    ix.pad_unknown(30);
    for (int i = 20; i < 30; ++i) hist.push_back(-1);
    for (int i = 0; i < 12; ++i) {
        ix.append(i % 4 + 10);
        hist.push_back(i % 4 + 10);
    }
    std::vector<int32_t> ctx = {0};
    LookupIndex::Match m = ix.find(ctx.data(), 1, 16, 64);
    CHECK(m.len == 0, "query over unknown tokens must not match (%d)", m.len);
    // a query whose 8-gram lies entirely after the gap matches only after the gap
    for (int i = 0; i < 12; ++i) {
        ix.append(i % 4 + 10);
        hist.push_back(i % 4 + 10);
    }
    ctx = {10};
    compare_find(ix, hist, ctx, "unknown", 0);
    m = ix.find(ctx.data(), 1, 16, 64);
    CHECK(m.len >= 8 && m.end >= 30, "match after the gap: %d @ %lld", m.len, (long long)m.end);
    int32_t out[7];
    const int n = ix.continuation(m.end, ctx.data(), 1, out, 7, 1 << 30);
    CHECK(n >= 1, "continuation after the gap: %d", n);
    // a continuation that would run into the gap stops there: history 0..9, 4 unknown, 0..8; anchor 9 -> the only match ends at position 9, followed by the gap
    LookupIndex iz(8);
    std::vector<int32_t> h3;
    for (int i = 0; i < 10; ++i) {
        iz.append(i);
        h3.push_back(i);
    }
    iz.pad_unknown(14);
    for (int i = 10; i < 14; ++i) h3.push_back(-1);
    for (int i = 0; i < 9; ++i) {
        iz.append(i);
        h3.push_back(i);
    }
    std::vector<int32_t> c2 = {9};
    compare_find(iz, h3, c2, "gap", 0);
    const LookupIndex::Match m3 = iz.find(c2.data(), 1, 16, 64);
    CHECK(m3.len == 10 && m3.end == 9, "gap continuation match %d @ %lld", m3.len, (long long)m3.end);
    const int n3 = iz.continuation(m3.end, c2.data(), 1, out, 7, 1 << 30);
    CHECK(n3 == 0, "continuation into the gap must stop: %d", n3);
    std::printf("  unknown padding ok\n");
}

static void policy_test() {
    LookupOptions o;
    LookupPolicy lp(o);
    SpecPolicy tm;   // no measurements: ratio priors
    // priors: bucket 3 (32+) fires wide, bucket 0 narrower
    const int T3 = lp.choose(40, 8, tm), T0 = lp.choose(8, 8, tm);
    CHECK(T3 >= 4 && T3 <= 8, "prior choice for long matches %d", T3);
    CHECK(T0 >= 2 && T0 <= T3, "prior choice for short matches %d (long %d)", T0, T3);
    std::printf("  prior widths: m=8 -> T %d, m=12 -> T %d, m=16 -> T %d, m=40 -> T %d (speed-ups %.2f %.2f %.2f %.2f)\n", T0, lp.choose(12, 8, tm), lp.choose(16, 8, tm), T3,
                lp.speedup(0, T0, tm), lp.speedup(1, lp.choose(12, 8, tm), tm), lp.speedup(2, lp.choose(16, 8, tm), tm), lp.speedup(3, T3, tm));
    // all accepted in bucket 0 -> T grows to 8
    for (int i = 0; i < 40; ++i) {
        const int T = lp.choose(9, 8, tm);
        lp.observe(9, T, T - 1);
    }
    CHECK(lp.choose(9, 8, tm) == 8, "all-accept bucket should widen to 8, got %d", lp.choose(9, 8, tm));
    CHECK(lp.choose(9, 5, tm) == 5, "max_T clamp %d", lp.choose(9, 5, tm));
    // all rejected in bucket 1 -> stops firing there apart from a T = 2 probe every 8 declined matches; bucket 0 and 3 unaffected
    int fired = 0, probes = 0;
    for (int i = 0; i < 80; ++i) {
        const int T = lp.choose(13, 8, tm);
        if (T >= 2) {
            ++fired;
            if (T == 2 && i >= 10) ++probes;
            lp.observe(13, T, 0);
        } else {
            lp.note_declined(13);
        }
    }
    CHECK(fired >= 2 && fired <= 14, "all-reject bucket fired %d times in 80 matches (probes %d)", fired, probes);
    CHECK(probes >= 6 && probes <= 9, "recovery probes %d (expected ~every 8th declined match)", probes);
    lp.observe(13, 2, 0);
    CHECK(lp.choose(13, 8, tm) == 1, "all-reject bucket still fires right after a rejected probe");
    CHECK(lp.choose(9, 8, tm) == 8 && lp.choose(40, 8, tm) >= 4, "other buckets affected");
    CHECK(lp.stats(1).steps == fired + 1 && lp.stats(1).accepted == 0 && lp.stats(1).drafted > 0 && lp.stats(1).declined == 80 - fired, "stats bucket 1: steps %lld declined %lld",
          (long long)lp.stats(1).steps, (long long)lp.stats(1).declined);
    // recovery: once the probes get accepted the bucket fires again (within 3 accepted probes the T=2 speed-up exceeds 1)
    int rec = -1;
    for (int i = 0; i < 100 && rec < 0; ++i) {
        const int T = lp.choose(13, 8, tm);
        if (T >= 2) {
            lp.observe(13, T, T - 1);   // all accepted now
            if (lp.choose(13, 8, tm) >= 2 && lp.choose(13, 8, tm) >= 2) rec = i;
        } else {
            lp.note_declined(13);
        }
    }
    CHECK(rec >= 0 && rec <= 30, "bucket 1 did not recover (rec %d)", rec);
    // timing model from measurements: plain 70 ms, verify(4) 150 ms (2.14x), verify(8) 240 (3.43x)
    SpecPolicy tm2;
    tm2.observe_plain(70000);
    tm2.observe_verify_time(4, 150000);
    tm2.observe_verify_time(8, 240000);
    CHECK(std::fabs(tm2.verify_ratio(4) - 150.0 / 70.0) < 1e-9 && std::fabs(tm2.verify_ratio(8) - 240.0 / 70.0) < 1e-9, "verify_ratio from measurements %.3f %.3f",
          tm2.verify_ratio(4), tm2.verify_ratio(8));
    CHECK(tm2.verify_ratio(6) > tm2.verify_ratio(4) && tm2.verify_ratio(6) < tm2.verify_ratio(8), "interpolated ratio %.3f", tm2.verify_ratio(6));
    LookupPolicy lq(o);
    // acceptance 0.5 per position in bucket 0: expected tokens at T=2: 1.5 vs cost 1.45 -> marginal; at T=4: 1.94 vs 2.14 -> no. Policy must pick T <= 2 or 1.
    for (int i = 0; i < 200; ++i) lq.observe(9, 8, (i % 2) ? 1 : 0);
    const int Tm = lq.choose(9, 8, tm2);
    CHECK(Tm <= 2, "p~0.5 regime chose T %d (speed-ups T2 %.3f T4 %.3f)", Tm, lq.speedup(0, 2, tm2), lq.speedup(0, 4, tm2));
    std::printf("  policy ok (all-reject stopped after %d steps; p~0.5 -> T %d)\n", fired, Tm);
}

int main() {
    std::printf("test_lookup\n");
    stream_test("random V=3", 3, 6000, false);
    stream_test("random V=50", 50, 6000, false);
    stream_test("copies V=1000", 1000, 8000, true);
    periodic_test();
    rewind_test("rewind V=4", 4);
    rewind_test("rewind V=300", 300);
    unknown_test();
    policy_test();
    std::printf("test_lookup: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

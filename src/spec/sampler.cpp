// C++ target sampler (mirrors server/sampling.py: temperature -> top_k -> top_p -> min_p -> draw, no penalties) and exact speculative sampling.
// Semantics copied from the Python reference (Sampler._reference / sample): e = exp(max((x - max) / T, -80)); top_k keeps the k largest scores (ties: lower id first),
// top_p renormalises over the top_k set (HF) and keeps the smallest prefix whose cumulative mass reaches top_p, min_p keeps scores >= min_p * best; greedy (T <= 0 or
// top_k == 1) is np.argmax (lowest id on ties). The random stream is the counter based SpecRng (not numpy's PCG64): distributions are equal, the draws are not bit-equal.
#include <x86intrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "ink/kernels_misc.h"
#include "ink/spec.h"

namespace ink {

uint64_t spec_mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double SpecRng::uniform() {
    ++k;
    const uint64_t h = spec_mix64(seed + k * 0x9E3779B97F4A7C15ull);
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);
}

uint64_t SpecRng::next_seed() const { return spec_mix64(seed ^ (0xA5A5A5A5DEADBEEFull + k * 0x2545F4914F6CDD1Dull)); }

float TargetDist::prob_of(int32_t d) const {
    if (greedy) return d == greedy_id ? 1.0f : 0.0f;
    if (dense) return (d >= 0 && (size_t)d < probs.size()) ? probs[(size_t)d] : 0.0f;
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] == d) return probs[i];
    return 0.0f;
}

Sampler::Sampler(int n_vocab) : V_(n_vocab), e_((size_t)n_vocab) { cand_.reserve(4096); }

int32_t Sampler::argmax(const float* x, int n) {
    if (n <= 0) return -1;
    int i = 0;
    __m512 vm = _mm512_set1_ps(-INFINITY);
    for (; i + 16 <= n; i += 16) vm = _mm512_max_ps(vm, _mm512_loadu_ps(x + i));
    float m = _mm512_reduce_max_ps(vm);
    for (int j = i; j < n; ++j) m = std::max(m, x[j]);
    for (int j = 0; j < n; ++j)
        if (x[j] == m) return j;   // first maximum
    return 0;                      // all NaN
}

void Sampler::build(const float* x, const SamplingParams& sp, TargetDist* out) {
    out->ids.clear();
    out->probs.clear();
    out->dense = false;
    out->greedy = sp.greedy();
    const int V = V_;
    if (out->greedy) {
        out->greedy_id = argmax(x, V);
        return;
    }
    float m = x[0];
    for (int i = 1; i < V; ++i) m = std::max(m, x[i]);
    const float inv = sp.temperature != 1.0f ? (float)(1.0 / (double)sp.temperature) : 1.0f;
    float* e = e_.data();
    for (int i = 0; i < V; ++i) {
        float v = x[i] - m;
        if (sp.temperature != 1.0f) v *= inv;
        e[i] = std::max(v, -80.0f);
    }
    vexp(e, e, (size_t)V);
    const bool use_k = sp.top_k > 0 && sp.top_k < V;
    if (!use_k && sp.top_p >= 1.0f && sp.min_p <= 0.0f) {   // pure temperature: the whole vocabulary, token id order
        double z = 0;
        for (int i = 0; i < V; ++i) z += e[i];
        out->dense = true;
        out->probs.resize((size_t)V);
        const double iz = 1.0 / z;
        for (int i = 0; i < V; ++i) out->probs[(size_t)i] = (float)(e[i] * iz);
        return;
    }
    // ---- candidate set (a threshold on e is always a prefix of the sorted vocabulary)
    auto collect = [&](float thr) {
        cand_.clear();
        if (thr <= 0.0f) {
            for (int i = 0; i < V; ++i) cand_.push_back(i);
        } else {
            for (int i = 0; i < V; ++i)
                if (e[i] >= thr) cand_.push_back(i);
        }
    };
    auto by_score = [&](int32_t a, int32_t b) { return e[a] > e[b] || (e[a] == e[b] && a < b); };
    double z = 0;   // normaliser of the top_p test
    if (use_k) {
        float thr = 1e-3f;
        for (;;) {
            collect(thr);
            if ((int)cand_.size() >= sp.top_k || thr <= 0.0f) break;
            thr = thr > 1e-12f ? thr * 1e-3f : 0.0f;
        }
        std::sort(cand_.begin(), cand_.end(), by_score);
        if ((int)cand_.size() > sp.top_k) cand_.resize((size_t)sp.top_k);
        if (sp.top_p < 1.0f)
            for (int32_t i : cand_) z += e[i];
    } else if (sp.top_p < 1.0f) {   // nucleus: widen the threshold until the set covers top_p of the whole mass
        for (int i = 0; i < V; ++i) z += e[i];
        float thr = 1e-3f;
        for (;;) {
            collect(thr);
            double s = 0;
            for (int32_t i : cand_) s += e[i];
            if (thr <= 0.0f || s >= (double)sp.top_p * z || (sp.min_p > 0.0f && thr <= sp.min_p)) break;
            thr = thr > 1e-12f ? thr * 1e-3f : 0.0f;
        }
        std::sort(cand_.begin(), cand_.end(), by_score);
    } else {   // min_p only: exact threshold, always contains the best token (e = 1)
        collect(sp.min_p);
        std::sort(cand_.begin(), cand_.end(), by_score);
    }
    // ---- keep: top_p prefix and min_p prefix of the descending list
    size_t keep = cand_.size();
    if (sp.top_p < 1.0f) {
        double cum = 0;
        size_t kp = keep;
        for (size_t i = 0; i < cand_.size(); ++i) {
            cum += e[cand_[i]];
            if (cum / z >= (double)sp.top_p) {
                kp = i + 1;
                break;
            }
        }
        keep = std::min(keep, kp);
    }
    if (sp.min_p > 0.0f) {
        const float thr = sp.min_p * e[cand_[0]];
        size_t km = 0;
        while (km < cand_.size() && e[cand_[km]] >= thr) ++km;
        keep = std::min(keep, km);
    }
    keep = std::max<size_t>(keep, 1);
    double s = 0;
    for (size_t i = 0; i < keep; ++i) s += e[cand_[i]];
    out->ids.assign(cand_.begin(), cand_.begin() + (long)keep);
    out->probs.resize(keep);
    for (size_t i = 0; i < keep; ++i) out->probs[i] = (float)(e[cand_[i]] / s);
}

int32_t Sampler::draw(const TargetDist& d, SpecRng& rng) {
    if (d.greedy) return d.greedy_id;
    const double u = rng.uniform();   // probs sum to 1
    double c = 0;
    if (d.dense) {
        const size_t n = d.probs.size();
        for (size_t i = 0; i < n; ++i) {
            c += d.probs[i];
            if (c > u) return (int32_t)i;
        }
        return (int32_t)(n - 1);
    }
    for (size_t i = 0; i < d.ids.size(); ++i) {
        c += d.probs[i];
        if (c > u) return d.ids[i];
    }
    return d.ids.back();
}

int32_t Sampler::sample(const float* logits, const SamplingParams& sp, SpecRng& rng) {
    if (sp.greedy()) return argmax(logits, V_);
    TargetDist d;
    build(logits, sp, &d);
    return draw(d, rng);
}

int32_t Sampler::draw_residual(const TargetDist& p, const float* q, int32_t d, SpecRng& rng) {
    // r = max(p - q, 0) over the support of p
    const size_t n = p.dense ? p.probs.size() : p.ids.size();
    auto tok = [&](size_t i) { return p.dense ? (int32_t)i : p.ids[i]; };
    auto qv = [&](int32_t t) { return q ? (double)q[t] : (t == d ? 1.0 : 0.0); };
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += std::max((double)p.probs[i] - qv(tok(i)), 0.0);
    if (!(sum > 0.0)) return draw(p, rng);   // p == q on the support (numerically): any draw from p is exact
    const double u = rng.uniform() * sum;
    double c = 0;
    int32_t last = tok(0);
    for (size_t i = 0; i < n; ++i) {
        const double r = std::max((double)p.probs[i] - qv(tok(i)), 0.0);
        if (r <= 0.0) continue;
        c += r;
        last = tok(i);
        if (c > u) return last;
    }
    return last;
}

int spec_accept(Sampler& smp, const float* logits, int n, const int32_t* drafts, const float* q, const SamplingParams& sp, SpecRng& rng, int32_t* out) {
    const int V = smp.n_vocab();
    if (sp.greedy()) {
        for (int i = 0; i < n; ++i) {
            const int32_t a = Sampler::argmax(logits + (size_t)i * V, V);
            out[i] = a;                        // accepted: a == draft; rejected: the target's own choice
            if (a != drafts[i]) return i;
        }
        out[n] = Sampler::argmax(logits + (size_t)n * V, V);
        return n;
    }
    TargetDist dist;
    for (int i = 0; i < n; ++i) {
        smp.build(logits + (size_t)i * V, sp, &dist);
        const int32_t d = drafts[i];
        const double pd = dist.prob_of(d);
        const double qd = q ? (double)q[(size_t)i * V + d] : 1.0;
        const double u = rng.uniform();
        if (u * qd < pd) {
            out[i] = d;
            continue;
        }
        out[i] = Sampler::draw_residual(dist, q ? q + (size_t)i * V : nullptr, d, rng);
        return i;
    }
    smp.build(logits + (size_t)n * V, sp, &dist);
    out[n] = Sampler::draw(dist, rng);
    return n;
}

}  // namespace ink

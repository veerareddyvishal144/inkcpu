// SpecPolicy: adaptive verify width (docs/DSPARK.md 10, docs/SPEC_DESIGN.md). Pure arithmetic: no clock, no randomness; the caller feeds measured times in microseconds.
//
// Model. c_j = P(draft j accepted | drafts 1..j-1 accepted) is tracked per position by a running mean that turns into an EWMA (weight `alpha`) after a few observations
// (conditioning on the earlier positions removes the censoring bias of the cumulative acceptance). L(T) = 1 + sum_{m=1..T-1} prod_{j<=m} c_j. Positions never observed
// inherit the previous conditional with a 3 % decay. Throughput(T) = L(T) / (draft(T-1) + verify(T)) against plain 1 / t_plain; times are measured (EWMA), the bytes model of
// docs/ROUTING.md only supplies priors scaled by what was already measured. Hysteresis: enter above `enter`, stay above `stay`; while plain, a probe step every
// `probe_iv` plain tokens (doubling after an unprofitable probe, reset after a promising one, which also schedules the next probe soon).
#include <algorithm>
#include <cmath>

#include "ink/spec.h"

namespace ink {

SpecPolicy::SpecPolicy(const SpecPolicyOptions& o) : o_(o), probe_iv_(o.reprobe) { reset_acceptance(); }

void SpecPolicy::reset_acceptance() {
    for (int j = 0; j < 7; ++j) {
        c_[j] = o_.prior_c[j];
        cnt_[j] = 0;
    }
    in_spec_ = false;
    probe_iv_ = o_.reprobe;
    plain_run_ = probe_iv_ - 1;   // the first step must be plain (t_plain), the first probe comes right after it (a new sequence should not wait 32 tokens)
    probing_ = false;
}

double SpecPolicy::c_eff(int j) const {
    if (j == 1) return c_[0];
    if (cnt_[j - 1] >= 2) return c_[j - 1];
    return c_eff(j - 1) * 0.97;
}

double SpecPolicy::L(int T) const {
    double a = 1.0, l = 1.0;
    for (int j = 1; j < T && j <= 7; ++j) {
        a *= c_eff(j);
        l += a;
    }
    return l;
}

double SpecPolicy::verify_us(int T) const {
    if (T < 1) T = 1;
    if (T > 8) T = 8;
    if (T == 1) return tp_;
    if (tv_[T] > 0) return tv_[T];
    double s = 0;
    int k = 0;
    for (int t = 2; t <= 8; ++t)
        if (tv_[t] > 0 && tp_ > 0) {
            s += tv_[t] / (tp_ * o_.verify_ratio[t]);
            ++k;
        }
    return tp_ * o_.verify_ratio[T] * (k ? s / k : 1.0);
}

double SpecPolicy::draft_us(int n) const {
    if (n < 1) return 0;
    if (n > 7) n = 7;
    if (td_[n] > 0) return td_[n];
    double s = 0;
    int k = 0;
    for (int m = 1; m <= 7; ++m)
        if (td_[m] > 0 && tp_ > 0) {
            s += td_[m] / (tp_ * (o_.draft_a + o_.draft_b * m));
            ++k;
        }
    return tp_ * (o_.draft_a + o_.draft_b * n) * (k ? s / k : 1.0);
}

double SpecPolicy::speedup(int T) const {
    if (T <= 1 || tp_ <= 0) return 1.0;
    return L(T) * tp_ / (draft_us(T - 1) + verify_us(T));
}

int SpecPolicy::choose(int max_T) {
    max_T = std::min(max_T, 8);
    if (o_.mode <= 0 || max_T < 2) return 1;
    if (o_.mode >= 2) return std::min(o_.mode, max_T);
    if (tp_ <= 0) return 1;   // auto: one measured plain step first
    int best_T = 1;
    double best = 0;
    for (int i = 0; i < o_.n_widths; ++i) {
        const int W = o_.widths[i];
        if (W < 2 || W > max_T) continue;
        const double s = speedup(W);
        if (s > best) {
            best = s;
            best_T = W;
        }
    }
    if (best_T > 1 && best > (in_spec_ ? o_.stay : o_.enter)) {
        in_spec_ = true;
        if (o_.explore > 0 && ++spec_run_ >= o_.explore) {   // exploration: refresh the estimates of the other width (positions it alone observes, its verify time)
            spec_run_ = 0;
            int other = 0;
            for (int i = 0; i < o_.n_widths; ++i) {
                const int W = o_.widths[i];
                if (W >= 2 && W <= max_T && W != best_T && W > other) other = W;
            }
            if (other) return other;
        }
        return best_T;
    }
    spec_run_ = 0;
    in_spec_ = false;
    if (plain_run_ >= probe_iv_) {   // probe with the narrowest width
        int W = 0;
        for (int i = 0; i < o_.n_widths; ++i)
            if (o_.widths[i] >= 2 && o_.widths[i] <= max_T && (W == 0 || o_.widths[i] < W)) W = o_.widths[i];
        if (W) {
            probing_ = true;
            return W;
        }
    }
    return 1;
}

void SpecPolicy::observe_plain(double us) {
    tp_ = tp_ > 0 ? (1 - o_.alpha_time) * tp_ + o_.alpha_time * us : us;
    ++plain_run_;
}

void SpecPolicy::observe_spec(int T, int accepted, double d_us, double v_us) {
    if (T < 2 || T > 8) return;
    const int n = T - 1;
    for (int j = 1; j <= n && j <= accepted + 1; ++j) {   // position j was tested iff 1..j-1 were accepted
        const double obs = accepted >= j ? 1.0 : 0.0;
        // running mean (2 pseudo-observations at the prior) turning into the EWMA; a probe after a long plain run may see a new regime: weight 1/3
        const double w = probing_ ? std::max(o_.alpha, 1.0 / 3.0) : std::max(o_.alpha, 1.0 / (cnt_[j - 1] + 3.0));
        c_[j - 1] += w * (obs - c_[j - 1]);
        if (cnt_[j - 1] < 1000) ++cnt_[j - 1];
    }
    tv_[T] = tv_[T] > 0 ? (1 - o_.alpha_time) * tv_[T] + o_.alpha_time * v_us : v_us;
    td_[n] = td_[n] > 0 ? (1 - o_.alpha_time) * td_[n] + o_.alpha_time * d_us : d_us;
    if (probing_) {
        probing_ = false;
        // promising = the updated estimate is near the entry threshold or the probe itself went well: look again in 3 plain tokens (a burst of probes re-enters in a few steps);
        // otherwise back off (doubling up to reprobe_max)
        const bool promising = speedup(T) >= 0.85 * o_.enter || accepted >= 2;
        if (promising) {
            probe_iv_ = o_.reprobe;
            plain_run_ = std::max(0, probe_iv_ - 3);
        } else {
            probe_iv_ = std::min(probe_iv_ * 2, o_.reprobe_max);
            plain_run_ = 0;
        }
    } else {
        plain_run_ = 0;
    }
}

}  // namespace ink

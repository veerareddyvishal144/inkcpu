// Session-lookup drafter (agent L, wave 2; docs/SPEC_DESIGN.md 11, docs/research/wave2_speculation.md). Exact n-gram index over the token history of a slot and the
// acceptance-driven width policy for its (free) drafts. No engine code: unit-tested against a brute-force reference in tests/spec/test_lookup.cpp.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "ink/spec.h"

namespace ink {

// ---------------------------------------------------------------------------------------------------------------------------------------- LookupIndex
LookupIndex::LookupIndex(int n_key) : n_(std::max(2, n_key)) {
    tab_.resize(4096);
}

void LookupIndex::reset() {
    hist_.clear();
    prev_.clear();
    n_used_ = 0;
    if (++gen_ == 0) {   // wrapped (never in practice): clear physically
        std::fill(tab_.begin(), tab_.end(), Ent());
        gen_ = 1;
    }
}

uint64_t LookupIndex::key_of(const int32_t* t) const {
    uint64_t h = 0x243F6A8885A308D3ull ^ (uint64_t)n_;
    for (int i = 0; i < n_; ++i) {
        h ^= (uint64_t)(uint32_t)t[i] + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        h *= 0xBF58476D1CE4E5B9ull;
    }
    return spec_mix64(h);
}

bool LookupIndex::indexable(int64_t e) const {
    if (e < n_ - 1) return false;
    const int32_t* t = hist_.data() + (e - n_ + 1);
    for (int i = 0; i < n_; ++i)
        if (t[i] < 0) return false;
    return true;
}

uint64_t LookupIndex::key_at(int64_t e) const { return key_of(hist_.data() + (e - n_ + 1)); }

LookupIndex::Ent* LookupIndex::find_ent(uint64_t key, bool insert) {
    const size_t mask = tab_.size() - 1;
    size_t i = (size_t)key & mask;
    for (;;) {
        Ent& en = tab_[i];
        if (en.gen != gen_) {   // empty (or from an older generation)
            if (!insert) return nullptr;
            en.key = key;
            en.pos = -1;
            en.gen = gen_;
            ++n_used_;
            return &en;
        }
        if (en.key == key) return &en;
        i = (i + 1) & mask;
    }
}

void LookupIndex::grow() {
    std::vector<Ent> old;
    old.swap(tab_);
    tab_.assign(old.size() * 2, Ent());
    n_used_ = 0;
    const size_t mask = tab_.size() - 1;
    for (const Ent& en : old) {
        if (en.gen != gen_ || en.pos < 0) continue;   // dead entries (undone by truncate) are dropped
        size_t i = (size_t)en.key & mask;
        while (tab_[i].gen == gen_) i = (i + 1) & mask;
        tab_[i] = en;
        ++n_used_;
    }
}

void LookupIndex::append(int32_t tok) {
    const int64_t e = (int64_t)hist_.size();
    hist_.push_back(tok);
    prev_.push_back(-1);
    if (!indexable(e)) return;
    if ((n_used_ + 1) * 2 > tab_.size()) grow();
    Ent* en = find_ent(key_at(e), true);
    prev_[(size_t)e] = en->pos;
    en->pos = (int32_t)e;
}

void LookupIndex::append(const int32_t* toks, int n) {
    for (int i = 0; i < n; ++i) append(toks[i]);
}

void LookupIndex::truncate(int64_t pos) {
    if (pos <= 0) {
        reset();
        return;
    }
    while ((int64_t)hist_.size() > pos) {
        const int64_t e = (int64_t)hist_.size() - 1;
        if (indexable(e)) {
            Ent* en = find_ent(key_at(e), false);
            if (en && en->pos == (int32_t)e) en->pos = prev_[(size_t)e];   // exact undo of append(e): the head of the chain was e
        }
        hist_.pop_back();
        prev_.pop_back();
    }
}

void LookupIndex::pad_unknown(int64_t upto) {
    while ((int64_t)hist_.size() < upto) append(-1);
}

void LookupIndex::sync(int64_t pos) {
    if ((int64_t)hist_.size() > pos) truncate(pos);
    else if ((int64_t)hist_.size() < pos) pad_unknown(pos);
}

LookupIndex::Match LookupIndex::find(const int32_t* ctx, int n_ctx, int max_candidates, int max_len) const {
    Match best;
    const int64_t H = (int64_t)hist_.size();
    const int64_t L = H + n_ctx;   // virtual sequence s = hist ++ ctx
    if (n_ctx < 0 || L < n_ || H < 1) return best;
    auto s_at = [&](int64_t i) -> int32_t { return i < H ? hist_[(size_t)i] : ctx[i - H]; };
    int32_t key_buf[64];
    int32_t* kb = key_buf;
    std::vector<int32_t> big;
    if (n_ > 64) {
        big.resize((size_t)n_);
        kb = big.data();
    }
    for (int i = 0; i < n_; ++i) {
        const int32_t t = s_at(L - n_ + i);
        if (t < 0) return best;   // unknown token inside the query
        kb[i] = t;
    }
    const uint64_t key = key_of(kb);
    const size_t mask = tab_.size() - 1;
    size_t i = (size_t)key & mask;
    const Ent* en = nullptr;
    for (;;) {
        const Ent& c = tab_[i];
        if (c.gen != gen_) break;
        if (c.key == key) {
            en = &c;
            break;
        }
        i = (i + 1) & mask;
    }
    if (!en) return best;
    int cand = 0;
    for (int64_t e = en->pos; e >= 0 && cand < max_candidates; e = prev_[(size_t)e]) {
        if (e >= H) break;   // cannot happen with exact undo; defensive
        ++cand;
        // backward extension: s[L-1-k] == hist[e-k]
        int m = 0;
        while (m < max_len && e - m >= 0 && L - 1 - m >= 0) {
            const int32_t a = hist_[(size_t)(e - m)];
            if (a < 0 || a != s_at(L - 1 - m)) break;
            ++m;
        }
        if (m >= n_ && m > best.len) {   // longest wins; ties keep the earlier (more recent) candidate
            best.len = m;
            best.end = e;
            if (m >= max_len) break;
        }
    }
    best.candidates = cand;
    return best;
}

LookupIndex::Match LookupIndex::find_reference(const int32_t* ctx, int n_ctx, int max_len) const {
    Match best;
    const int64_t H = (int64_t)hist_.size();
    const int64_t L = H + n_ctx;
    if (L < n_ || H < 1) return best;
    auto s_at = [&](int64_t i) -> int32_t { return i < H ? hist_[(size_t)i] : ctx[i - H]; };
    for (int64_t e = H - 1; e >= 0; --e) {
        int m = 0;
        while (m < max_len && e - m >= 0 && L - 1 - m >= 0) {
            const int32_t a = hist_[(size_t)(e - m)];
            if (a < 0 || a != s_at(L - 1 - m)) break;
            ++m;
        }
        if (m >= n_ && m > best.len) {
            best.len = m;
            best.end = e;
        }
        ++best.candidates;
    }
    return best;
}

int LookupIndex::continuation(int64_t end, const int32_t* ctx, int n_ctx, int32_t* out, int n, int32_t max_id) const {
    const int64_t H = (int64_t)hist_.size();
    if (end < 0 || end >= H) return 0;
    int k = 0;
    for (; k < n; ++k) {
        const int64_t i = end + 1 + k;
        int32_t t;
        if (i < H) t = hist_[(size_t)i];
        else if (i < H + n_ctx) t = ctx[i - H];
        else t = out[i - H - n_ctx];   // already written (i - H - n_ctx < k)
        if (t < 0 || t >= max_id) break;
        out[k] = t;
    }
    return k;
}

size_t LookupIndex::bytes() const { return hist_.capacity() * 4 + prev_.capacity() * 4 + tab_.capacity() * sizeof(Ent); }

// ---------------------------------------------------------------------------------------------------------------------------------------- LookupPolicy
LookupPolicy::LookupPolicy(const LookupOptions& o) : o_(o) { reset(); }

void LookupPolicy::reset() {
    for (int b = 0; b < 4; ++b) {
        for (int j = 0; j < 7; ++j) {
            c_[b][j] = o_.prior_c[b];
            cnt_[b][j] = 0;
        }
        declined_run_[b] = 0;
        probing_[b] = false;
        st_[b] = Stats();
    }
}

void LookupPolicy::note_declined(int match_len) {
    const int b = bucket(match_len);
    ++st_[b].declined;
    ++declined_run_[b];
}

int LookupPolicy::bucket(int m) {
    if (m < 12) return 0;
    if (m < 16) return 1;
    if (m < 32) return 2;
    return 3;
}

const char* LookupPolicy::bucket_name(int b) {
    static const char* n[4] = {"8-11", "12-15", "16-31", "32+"};
    return n[b < 0 ? 0 : (b > 3 ? 3 : b)];
}

double LookupPolicy::L(int b, int T) const {
    double a = 1.0, l = 1.0;
    for (int j = 1; j < T && j <= 7; ++j) {
        // positions never observed in this bucket inherit the previous conditional with a 3 % decay (same rule as the DSpark policy)
        const double cj = (j == 1 || cnt_[b][j - 1] >= 2) ? c_[b][j - 1] : c_[b][j - 2] * 0.97;
        a *= cj;
        l += a;
    }
    return l;
}

double LookupPolicy::speedup(int b, int T, const SpecPolicy& tm) const {
    if (T <= 1) return 1.0;
    const double r = tm.verify_ratio(T);
    return r > 0 ? L(b, T) / r : 0.0;
}

int LookupPolicy::choose(int match_len, int max_T, const SpecPolicy& tm, double* predicted) {
    const int b = bucket(match_len);
    int best_T = 1;
    double best = o_.enter;
    for (int T = 2; T <= std::min(max_T, 8); ++T) {
        const double s = speedup(b, T, tm);
        if (s > best) {
            best = s;
            best_T = T;
        }
    }
    if (predicted) *predicted = best_T > 1 ? best : speedup(b, 2, tm);
    if (best_T == 1 && o_.probe_every > 0 && declined_run_[b] >= o_.probe_every && max_T >= 2) {   // recovery probe (costs <= 0.45 plain steps when rejected)
        probing_[b] = true;
        return 2;
    }
    return best_T;
}

void LookupPolicy::observe(int match_len, int T, int accepted) {
    if (T < 2 || T > 8) return;
    const int b = bucket(match_len);
    const int n = T - 1;
    for (int j = 1; j <= n && j <= accepted + 1; ++j) {   // position j was tested iff 1..j-1 were accepted
        const double obs = accepted >= j ? 1.0 : 0.0;
        // running mean (2 pseudo-observations at the prior) turning into the EWMA; a recovery probe after a run of declines may see a new regime: weight 1/3
        const double w = probing_[b] ? std::max(o_.alpha, 1.0 / 3.0) : std::max(o_.alpha, 1.0 / (cnt_[b][j - 1] + 3.0));
        c_[b][j - 1] += w * (obs - c_[b][j - 1]);
        if (cnt_[b][j - 1] < 1000000) ++cnt_[b][j - 1];
    }
    declined_run_[b] = 0;
    probing_[b] = false;
    ++st_[b].steps;
    st_[b].drafted += n;
    st_[b].accepted += accepted;
}

LookupOptions lookup_options_from_env(LookupOptions o) {
    if (const char* s = std::getenv("INK_SPEC_LOOKUP"); s && *s) o.enable = std::atoi(s) != 0;
    if (const char* s = std::getenv("INK_SPEC_LOOKUP_NMIN"); s && *s) o.n_min = std::max(2, std::min(64, std::atoi(s)));
    if (const char* s = std::getenv("INK_SPEC_LOOKUP_ENTER"); s && *s) o.enter = std::atof(s);
    if (const char* s = std::getenv("INK_SPEC_LOOKUP_MAXDRAFT"); s && *s) o.max_draft = std::max(1, std::min(kMaxDraft, std::atoi(s)));
    if (const char* s = std::getenv("INK_SPEC_LOOKUP_PROBE"); s && *s) o.probe_every = std::max(0, std::atoi(s));
    return o;
}

}  // namespace ink

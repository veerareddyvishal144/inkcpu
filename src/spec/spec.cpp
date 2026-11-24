// SpecDecoder: the speculative decode loop. One step = [policy] -> draft -> verify (batched executor, T rows) -> accept/reject -> stop / budget truncation ->
// rollback (engine_verify_commit) -> drafter commit. Plain steps (T = 1) run Engine::decode; every processed row reaches the drafter through the engine's tap sink
// (plain decode, prefill chunks) or through the verify taps (only the kept rows), so the drafter context always equals the engine slot. docs/SPEC_DESIGN.md.
// Agent L (wave 2): the drafts may also come from the session-lookup index (LookupIndex, docs/SPEC_DESIGN.md 11): in auto mode a long exact n-gram match of the
// history ++ anchor is drafted first (free), the DSpark policy only runs when no match fires; mode 9 = lookup only. The emitted tokens stay exact: lookup drafts
// go through the same verify + rejection sampler (one-hot q) as DSpark drafts.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "ink/spec.h"

namespace ink {

namespace {
using clk = std::chrono::steady_clock;
double us_between(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
}  // namespace

int spec_mode_from_env(int def) {
    const char* s = std::getenv("INK_SPEC");
    if (!s || !*s) return def;
    if (!std::strcmp(s, "auto")) return 1;
    if (!std::strcmp(s, "off")) return 0;
    if (!std::strcmp(s, "lookup")) return kSpecModeLookup;
    const int v = std::atoi(s);
    if (v == 1) return 0;
    if (v >= 2 && v <= 8) return v;
    if (v == kSpecModeLookup) return v;
    return def;
}

SpecDecoder::SpecDecoder(Engine& e, std::unique_ptr<Drafter> drafter, const SpecOptions& opt)
    : E_(e), drafter_(std::move(drafter)), opt_(opt), smp_(e.n_vocab()) {
    opt_.mode = spec_mode_from_env(opt_.mode);
    opt_.policy.mode = opt_.mode == kSpecModeLookup ? 0 : opt_.mode;
    opt_.lookup = lookup_options_from_env(opt_.lookup);
    opt_.lookup.max_draft = std::max(1, std::min(opt_.lookup.max_draft, kMaxDraft));
    slots_.resize(e.slots.size());
    for (Slot& s : slots_) {
        s.policy = SpecPolicy(opt_.policy);
        s.st = drafter_ ? drafter_->state_new() : -1;
        s.lk = LookupIndex(opt_.lookup.n_min);
        s.lkp = LookupPolicy(opt_.lookup);
    }
    const size_t V = (size_t)e.n_vocab();
    logits_.resize((size_t)kVerifyMaxRows * V);
    q_.resize((size_t)kMaxDraft * V);
    taps_.resize((size_t)kVerifyMaxRows * kEngineTapFloats);
    if (drafter_) e.tap_sink = [this](int slot, int64_t first_pos, int rows, const float* taps) { on_taps(slot, first_pos, rows, taps); };
}

SpecDecoder::~SpecDecoder() {
    if (drafter_) E_.tap_sink = nullptr;
    drafter_.reset();
}

void SpecDecoder::on_taps(int slot, int64_t first_pos, int rows, const float* taps) {
    if (slot < 0 || slot >= (int)slots_.size()) return;
    Slot& s = slots_[slot];
    if (first_pos == 0) {   // a new sequence starts in this slot (slot_reset / slot_new): the lookup history restarts (ink_forward reports the tokens right after)
        s.lk.reset();
        s.lkp.reset();
    }
    if (s.st < 0) return;
    if (first_pos == 0) {   // ... and the drafter context restarts too
        drafter_->state_reset(s.st);
        s.synced = true;
        s.policy.reset_acceptance();
    }
    if (!s.synced || drafter_->state_pos(s.st) != first_pos) {   // rows were missed (slot prefilled before the decoder existed, slot_copy ...): plain decode only until the next reset
        s.synced = false;
        sink_dropped_ += rows;
        return;
    }
    drafter_->commit(s.st, taps, rows);
}

bool SpecDecoder::synced(int slot) const {
    if (!drafter_ || slot < 0 || slot >= (int)slots_.size()) return false;
    const Slot& s = slots_[slot];
    return s.st >= 0 && s.synced && drafter_->state_pos(s.st) == E_.slot_pos(slot);
}

void SpecDecoder::note_tokens(int slot, int64_t first_pos, const int32_t* toks, int n) {
    if (slot < 0 || slot >= (int)slots_.size() || !toks || n <= 0) return;
    Slot& s = slots_[slot];
    if (first_pos == 0) {
        s.lk.reset();
        s.lkp.reset();
    } else {
        s.lk.sync(first_pos);
    }
    s.lk.append(toks, n);
}

void SpecDecoder::note_truncate(int slot, int64_t pos) {
    if (slot < 0 || slot >= (int)slots_.size()) return;
    slots_[slot].lk.sync(std::max<int64_t>(0, pos));
}

void SpecDecoder::note_reset(int slot) {
    if (slot < 0 || slot >= (int)slots_.size()) return;
    slots_[slot].lk.reset();
    slots_[slot].lkp.reset();
}

int SpecDecoder::step(int slot, const SpecRequest& rq, SpecResult* res) {
    const auto t_begin = clk::now();
    *res = SpecResult();
    last_src_ = 0;
    if (slot < 0 || slot >= (int)E_.slots.size() || !E_.slots[slot].used) return res->rc = -1;
    if (rq.max_emit < 1 || rq.n_stop < 0) return res->rc = -4;
    if (rq.anchor < 0 || rq.anchor >= E_.cfg.n_vocab_padded) return res->rc = -3;
    SlotState& S = E_.slots[slot];
    const int64_t p = S.pos;
    if (p + 1 > E_.opt.max_ctx) return res->rc = -2;
    Slot& sl = slots_[slot];
    if (p == 0) {   // fresh sequence, nothing processed yet
        if (drafter_ && sl.st >= 0) {
            drafter_->state_reset(sl.st);
            sl.synced = true;
            sl.policy.reset_acceptance();
        }
        sl.lk.reset();
        sl.lkp.reset();
    } else {
        sl.lk.sync(p);   // rows the decoder did not see (slot_copy, a forward without the C ABI hook) become unknown positions; a missed rewind truncates
    }
    const bool can_spec = drafter_ && sl.st >= 0 && sl.synced && drafter_->state_pos(sl.st) == p;
    const int mode = rq.mode >= 0 ? rq.mode : opt_.mode;
    const bool lookup_on = opt_.lookup.enable && (mode == 1 || mode == kSpecModeLookup);
    sl.policy.set_mode(mode == kSpecModeLookup ? 0 : mode);
    const int max_T = (int)std::min<int64_t>(std::min<int64_t>(kVerifyMaxRows, rq.max_emit), (int64_t)E_.opt.max_ctx - p);
    SpecRng rng(rq.seed);
    auto is_stop = [&](int32_t t) {
        for (int i = 0; i < rq.n_stop; ++i)
            if (rq.stop_ids[i] == t) return true;
        return false;
    };
    // ---- choose the draft source: session lookup (free) first, then the DSpark policy
    int32_t toks[kVerifyMaxRows];
    toks[0] = rq.anchor;
    int T = 1, src = 0;
    LookupIndex::Match mt;
    double lookup_us = 0;
    if (lookup_on && max_T >= 2) {
        const auto tl0 = clk::now();
        mt = sl.lk.find(&rq.anchor, 1, opt_.lookup.max_candidates, opt_.lookup.max_match);
        if (mt.len >= opt_.lookup.n_min) {
            double s_lookup = 0;
            const int Tl = sl.lkp.choose(mt.len, std::min(max_T, 1 + opt_.lookup.max_draft), sl.policy, &s_lookup);
            bool use = Tl >= 2;
            if (use && can_spec && mode == 1) {   // auto with a synced DSpark drafter: the source with the higher predicted speed-up drafts this step
                double s_dspark = 0;
                for (int i = 0; i < opt_.policy.n_widths; ++i) {
                    const int W = opt_.policy.widths[i];
                    if (W >= 2 && W <= max_T) s_dspark = std::max(s_dspark, sl.policy.speedup(W));
                }
                if (s_dspark > s_lookup) {
                    use = false;
                    sl.lkp.note_yielded(mt.len);
                }
            }
            if (use) {
                const int got = sl.lk.continuation(mt.end, &rq.anchor, 1, toks + 1, Tl - 1, E_.cfg.n_vocab);
                if (got >= 1) {
                    T = 1 + got;
                    src = 2;
                }
            } else if (Tl < 2) {
                sl.lkp.note_declined(mt.len);
            }
        }
        lookup_us = us_between(tl0, clk::now());
    }
    if (T == 1 && can_spec) {
        T = sl.policy.choose(max_T);
        if (T > 1) src = 1;
    }
    res->T = T;
    res->src = src;
    res->match_len = src == 2 ? mt.len : 0;
    last_src_ = src;
    if (T == 1) {
        const auto t0 = clk::now();
        int rc = E_.decode(slot, rq.anchor, logits_.data());
        if (rc) return res->rc = rc;
        const auto t1 = clk::now();
        const int32_t tok = smp_.sample(logits_.data(), rq.sp, rng);
        const auto t2 = clk::now();
        res->tokens[0] = tok;
        res->n_tokens = 1;
        res->stopped = is_stop(tok) ? 1 : 0;
        res->draft_us = lookup_us;
        res->verify_us = us_between(t0, t1);
        res->accept_us = us_between(t1, t2);
        sl.policy.observe_plain(us_between(t0, t2));
        sl.lk.append(rq.anchor);
    } else {
        const int n = T - 1;
        const bool dsample = src == 1 && !rq.sp.greedy() && opt_.draft_sample;   // lookup drafts are argmax-like proposals: one-hot q in the rejection sampler
        const auto t0 = clk::now();
        if (src == 1) {
            DraftRequest dq;
            dq.anchor = rq.anchor;
            dq.n = n;
            dq.sample = dsample;
            dq.temperature = rq.sp.temperature;
            dq.seed = spec_mix64(rq.seed ^ 0xD5A9D5A9D5A9D5A9ull);
            int rc = drafter_->draft(sl.st, dq, toks + 1, dsample ? q_.data() : nullptr);
            if (rc) return res->rc = rc;
        }
        const auto t1 = clk::now();
        for (int i = 1; i < T; ++i)
            if (toks[i] < 0 || toks[i] >= E_.cfg.n_vocab) return res->rc = -3;   // a drafter must never propose a padded / invalid id
        int rc = engine_verify(E_, slot, toks, T, logits_.data(), taps_.data(), opt_.exact);
        if (rc) return res->rc = rc;
        const auto t2 = clk::now();
        int32_t emit[kVerifyMaxRows];
        const int k = spec_accept(smp_, logits_.data(), n, toks + 1, dsample ? q_.data() : nullptr, rq.sp, rng, emit);
        const auto t3 = clk::now();
        int n_emit = std::min(k + 1, rq.max_emit);
        int stopped = 0;
        for (int i = 0; i < n_emit; ++i)
            if (is_stop(emit[i])) {   // the stop token is emitted but never fed: only the rows BEFORE it stay processed (rows kept = tokens emitted)
                n_emit = i + 1;
                stopped = 1;
                break;
            }
        rc = engine_verify_commit(E_, slot, n_emit);
        if (rc) return res->rc = rc;
        if (can_spec) {   // keep the DSpark context equal to the slot whatever drafted this step
            rc = drafter_->commit(sl.st, taps_.data(), n_emit);
            if (rc < 0) sl.synced = false;
        } else if (drafter_ && sl.st >= 0) {
            sink_dropped_ += n_emit;
        }
        const auto t4 = clk::now();
        for (int i = 0; i < n_emit; ++i) res->tokens[i] = emit[i];
        res->n_tokens = n_emit;
        res->stopped = stopped;
        res->drafted = n;
        res->accepted = k;
        res->draft_us = src == 1 ? us_between(t0, t1) : lookup_us;
        res->verify_us = us_between(t1, t2);
        res->accept_us = us_between(t2, t3) + us_between(t3, t4);   // accept/reject + rollback + drafter commit
        if (src == 1) {
            sl.policy.observe_spec(T, k, res->draft_us, us_between(t1, t4));
        } else {
            sl.lkp.observe(mt.len, T, k);
            sl.policy.observe_verify_time(T, us_between(t1, t4));
        }
        // history: the anchor and the emitted tokens that were fed (all but the last one, which is the next anchor or the stop id)
        sl.lk.append(rq.anchor);
        for (int i = 0; i + 1 < n_emit; ++i) sl.lk.append(emit[i]);
    }
    res->next_seed = rng.next_seed();
    res->total_us = us_between(t_begin, clk::now());
    return 0;
}

}  // namespace ink

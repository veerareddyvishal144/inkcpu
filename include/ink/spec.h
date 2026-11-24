// Speculative decoding on top of the inkcpu engine (agent S2). Design: docs/SPEC_DESIGN.md. Contract: docs/SPEC_CONTRACT.md.
//
//   Sampler / TargetDist  : the target sampler of server/sampling.py in C++ (temperature -> top_k -> top_p -> min_p, no penalties) + exact speculative sampling
//   SpecPolicy            : adaptive verify width T in {1, 4, 8} from measured acceptance and measured step times (deterministic, clock free, unit testable)
//   SpecDecoder           : the loop: tap sink -> drafter commits, draft, Engine verify (engine_verify), accept/reject, stop handling, rollback, commit
//   LookupIndex / LookupPolicy (agent L, wave 2): draft-model-free "session lookup" drafter: exact n-gram matching over every token the slot has seen
//                           (prompt, generated text, tool results); auto = lookup first, DSpark fallback; mode 9 = lookup only. docs/SPEC_DESIGN.md 11.
#pragma once
#include <climits>
#include <cstdint>
#include <memory>
#include <vector>

#include "ink/drafter.h"
#include "ink/engine.h"
#include "ink/engine_prefill.h"

namespace ink {

// ====================================================================================================================== sampler
struct SamplingParams {
    float temperature = 0.0f;   // <= 0 (or top_k == 1): greedy argmax
    int top_k = 0;              // 0 = off
    float top_p = 1.0f;
    float min_p = 0.0f;
    bool greedy() const { return temperature <= 0.0f || top_k == 1; }
};

// Counter based uniform stream: draw k of a seed is a hash of (seed, k); the caller threads `next_seed()` from one step to the next.
struct SpecRng {
    uint64_t seed;
    uint64_t k = 0;
    explicit SpecRng(uint64_t s) : seed(s) {}
    double uniform();                 // [0, 1), 53 bits
    uint64_t next_seed() const;       // seed of the following step / call (never equal to the current one)
};
uint64_t spec_mix64(uint64_t x);      // splitmix64 finaliser

// The distribution the target sampler draws from (after temperature, top_k, top_p, min_p). Greedy = one-hot on `greedy_id`.
struct TargetDist {
    bool greedy = false;
    int32_t greedy_id = -1;
    bool dense = false;               // true: probs[token id] over the whole vocabulary (pure temperature); false: sparse ids/probs sorted by descending probability
    std::vector<int32_t> ids;         // sparse only
    std::vector<float> probs;         // normalised over the kept set
    float prob_of(int32_t d) const;   // 0 for tokens outside the kept set
};

class Sampler {
public:
    explicit Sampler(int n_vocab);
    // Fills `out` from one logits row (n_vocab floats, softcap applied).
    void build(const float* logits, const SamplingParams& sp, TargetDist* out);
    // One draw (one uniform) from `d` (same inverse-CDF orders as server/sampling.py: vocabulary order for dense, descending order for sparse).
    static int32_t draw(const TargetDist& d, SpecRng& rng);
    // build + draw; greedy draws nothing. The plain-decode sampler.
    int32_t sample(const float* logits, const SamplingParams& sp, SpecRng& rng);
    // Residual after a rejected draft: token ~ norm(max(p - q, 0)); q = dense row over the vocabulary, or null = one-hot on d (greedy drafts).
    static int32_t draw_residual(const TargetDist& p, const float* q, int32_t d, SpecRng& rng);
    int n_vocab() const { return V_; }
    static int32_t argmax(const float* x, int n);   // first maximum (lowest id on ties), like np.argmax

private:
    int V_;
    std::vector<float> e_;
    std::vector<int32_t> cand_;
};

// Exact speculative sampling over one verify step. logits = [n + 1][V] rows of the verify, drafts d_1..d_n (rows 0..n-1 verify them, row n gives the bonus).
// q = dense draft probabilities [n][V] (drafts were sampled from it) or null (greedy drafts, q = one-hot). Greedy params: accept iff draft == argmax.
// out[0..k] receive the emitted tokens (k accepted drafts + the corrected / bonus token); returns k. Draw order: per position one uniform for the accept test, one for the residual.
int spec_accept(Sampler& smp, const float* logits, int n, const int32_t* drafts, const float* q, const SamplingParams& sp, SpecRng& rng, int32_t* out);

// ====================================================================================================================== policy
struct SpecPolicyOptions {
    int mode = 1;                    // 0 = plain only, 1 = auto, 2..8 = fixed verify width T
    int widths[2] = {4, 8};          // widths considered by auto (plus T = 1)
    int n_widths = 2;
    double alpha = 0.15;             // EWMA weight of one observation (steady state; a running mean is used for the first observations)
    double alpha_time = 0.2;
    double enter = 1.04;             // predicted speed-up needed to start speculating
    double stay = 1.00;              // ... and to keep going once speculating
    int reprobe = 32;                // plain tokens between speculative probes while plain
    int reprobe_max = 128;           // back-off cap of the probe interval after unprofitable probes (bounds the re-entry latency after a regime change)
    int explore = 16;                // while speculating: every `explore` spec steps one step uses the OTHER candidate width (keeps its acceptance / time estimates fresh); 0 = off
    double prior_c[7] = {0.60, 0.64, 0.76, 0.78, 0.62, 0.76, 0.73};   // conditional acceptance priors (docs/DSPARK.md 9.2, pooled greedy n = 7)
    // bytes-model priors for times that were not measured yet (relative to one plain step): verify(T) and draft(n)
    double verify_ratio[9] = {0, 1.0, 1.45, 1.9, 2.33, 2.65, 2.97, 3.3, 3.6};   // index = T; the real ratio is measured. K2 2026-11-23: verify/plain measured with the
                                                                               // multix MoE path (12 threads, same window): T=4 2.45 (code) / 2.20 (chat), T=8 3.75 / 3.47; the old
                                                                               // values were the bytes model of docs/ROUTING.md (2.47 / 3.72), which the GEMM path never reached (2.7 / 4.2)
    double draft_a = 0.20, draft_b = 0.025;                                    // draft(n) ~ plain * (a + b n); K2 2026-11-23: measured idle-host drafts 21-23 ms (n=3) / 29-30 (n=7)
                                                                               // vs a 72 ms plain step = 0.30 / 0.41, minus the 11 % multix base pass -> 0.27 / 0.37
};

class SpecPolicy {
public:
    explicit SpecPolicy(const SpecPolicyOptions& o = SpecPolicyOptions());
    void reset_acceptance();                                  // new sequence: acceptance back to the priors (times are kept)
    void set_mode(int mode) { o_.mode = mode; }
    int mode() const { return o_.mode; }
    int choose(int max_T);                                    // T in {1} U widths (<= max_T); updates the probe counter
    void observe_plain(double us);                            // one plain step (decode + sampling) took `us`
    void observe_spec(int T, int accepted, double draft_us, double verify_us);   // spec step: T rows, `accepted` drafts accepted
    double L(int T) const;                                    // expected tokens per spec step with T rows
    double verify_us(int T) const;
    double verify_ratio(int T) const;                         // verify(T) / plain from the measurements, else the prior ratio (usable before any step was timed)
    void observe_verify_time(int T, double us);               // time-only observation of a verify step whose drafts came from another source (session lookup)
    double draft_us(int n) const;
    double plain_us() const { return tp_; }
    double speedup(int T) const;                              // L(T) * plain / (draft(T-1) + verify(T)); 1 for T = 1
    bool speculating() const { return in_spec_; }
    int plain_run() const { return plain_run_; }
    int probe_interval() const { return probe_iv_; }
    double c(int j) const { return c_[j - 1]; }               // conditional acceptance estimate at draft position j (1..7)
    int observations(int j) const { return cnt_[j - 1]; }

private:
    double c_eff(int j) const;
    SpecPolicyOptions o_;
    double c_[7];
    int cnt_[7];
    double tp_ = 0;
    double tv_[9] = {};   // measured verify time by T (0 = unknown)
    double td_[8] = {};   // measured draft time by n
    bool in_spec_ = false;
    int plain_run_ = 0, probe_iv_;
    int spec_run_ = 0;    // spec steps since the last exploration step
    bool probing_ = false;
};

// ====================================================================================================================== session lookup (agent L)
// LookupIndex: the token history of one slot (every position the engine processed: prompt, generated tokens, tool results) with an exact n-gram index: a hash table
// from the hash of the n_key tokens ending at a position to the most recent such position, plus a per-position `prev` chain to the older occurrences. append is O(1)
// without allocation (amortised: the table doubles when half full), truncate(pos) undoes the appends in reverse order (exact: the index equals one built from the
// shorter history, so a slot rewind keeps it consistent), reset is a generation bump. Unknown positions (-1: rows the decoder did not see) never match.
// find(): the longest match of the suffix of history ++ ctx (ctx = the pending anchor [+ drafts]) among <= max_candidates chain entries, backward-extended to its true
// length (<= max_len); ties -> the most recent. continuation(): the tokens that followed the match, cyclically extended through ctx / the draft itself when the
// continuation runs into the query (periodic text).
class LookupIndex {
public:
    explicit LookupIndex(int n_key = 8);
    void reset();
    void append(int32_t tok);
    void append(const int32_t* toks, int n);
    void truncate(int64_t pos);                      // keep positions < pos
    void pad_unknown(int64_t upto);                  // positions size()..upto-1 become unknown (never match)
    void sync(int64_t pos);                          // size() == pos afterwards (truncate or pad)
    int64_t size() const { return (int64_t)hist_.size(); }
    int32_t at(int64_t i) const { return hist_[(size_t)i]; }
    int n_key() const { return n_; }
    struct Match {
        int len = 0;                                 // matched tokens (>= n_key) or 0
        int64_t end = -1;                            // history position of the last matched token
        int candidates = 0;                          // chain entries examined
    };
    Match find(const int32_t* ctx, int n_ctx, int max_candidates = 16, int max_len = 64) const;
    // out[i] = s[end + 1 + i] with s = history ++ ctx ++ out; stops at an unknown token or an id >= max_id; returns the tokens written (<= n)
    int continuation(int64_t end, const int32_t* ctx, int n_ctx, int32_t* out, int n, int32_t max_id = INT32_MAX) const;
    size_t bytes() const;
    // brute-force reference of find() (tests): scans every history position
    Match find_reference(const int32_t* ctx, int n_ctx, int max_len = 64) const;

private:
    struct Ent {
        uint64_t key = 0;
        int32_t pos = -1;
        uint32_t gen = 0;
    };
    uint64_t key_of(const int32_t* t) const;         // hash of n_ tokens
    bool indexable(int64_t e) const;                 // the n_ tokens ending at e are all known
    uint64_t key_at(int64_t e) const;
    Ent* find_ent(uint64_t key, bool insert);
    void grow();
    int n_;
    std::vector<int32_t> hist_;
    std::vector<int32_t> prev_;                      // prev_[e] = previous position with the same key, -1 = none
    std::vector<Ent> tab_;                           // open addressing, power-of-two capacity
    size_t n_used_ = 0;
    uint32_t gen_ = 1;
};

struct LookupOptions {
    bool enable = true;             // env INK_SPEC_LOOKUP=0 disables the lookup drafter (auto behaves as before: DSpark policy only)
    int n_min = 8;                  // key length = minimum match length (env INK_SPEC_LOOKUP_NMIN)
    int max_candidates = 16;        // chain entries examined per query
    int max_match = 64;             // backward extension cap
    int max_draft = 7;              // <= kMaxDraft: T = 1 + draft <= kVerifyMaxRows
    double enter = 1.0;             // predicted speed-up (free draft) needed to fire; the marginal-row rule (lambda ~0.3) in closed form (env INK_SPEC_LOOKUP_ENTER)
    double alpha = 0.10;            // EWMA weight of the acceptance estimates (after a running mean over the first observations)
    int probe_every = 8;            // a bucket whose estimate fell below break-even fires a T = 2 probe every `probe_every` declined matches (recovery; env INK_SPEC_LOOKUP_PROBE)
    double prior_c[4] = {0.70, 0.80, 0.90, 0.95};   // conditional acceptance priors per match-length bucket [8,12) [12,16) [16,32) [32,inf)
};

// LookupPolicy: conditional acceptance c_{b,j} = P(draft j accepted | 1..j-1 accepted) per match-length bucket b and draft position j; verify width T = argmax over 2..max_T
// of L_b(T) * plain / verify(T) (free draft) when that exceeds `enter`, else 1 (do not fire). The time model is the DSpark policy's (measured plain / verify(T) times).
class LookupPolicy {
public:
    explicit LookupPolicy(const LookupOptions& o = LookupOptions());
    void reset();
    static int bucket(int match_len);                  // 0..3
    static const char* bucket_name(int b);
    int choose(int match_len, int max_T, const SpecPolicy& timing, double* predicted = nullptr);   // may schedule a recovery probe (state); predicted = speed-up of the chosen T
    double L(int b, int T) const;                      // expected tokens per step of width T in bucket b
    double speedup(int b, int T, const SpecPolicy& timing) const;
    void observe(int match_len, int T, int accepted);  // T rows (T-1 drafts), `accepted` drafts accepted
    double c(int b, int j) const { return c_[b][j - 1]; }
    struct Stats {
        int64_t steps = 0, drafted = 0, accepted = 0, declined = 0, yielded = 0;   // declined: predicted speed-up below `enter`; yielded: the DSpark policy predicted more
    };
    void note_yielded(int match_len) { ++st_[bucket(match_len)].yielded; }
    const Stats& stats(int b) const { return st_[b]; }
    void note_declined(int match_len);                 // the decoder saw a match but choose() returned 1
    const LookupOptions& options() const { return o_; }

private:
    LookupOptions o_;
    double c_[4][7];
    int cnt_[4][7];
    int declined_run_[4];                              // declined matches since the bucket last fired
    bool probing_[4];                                  // the next observation of the bucket is a recovery probe (weight 1/3)
    Stats st_[4];
};

constexpr int kSpecModeLookup = 9;   // SpecRequest::mode / SpecOptions::mode: lookup drafter only (plain fallback); 1 = auto = lookup first, DSpark policy fallback

// ====================================================================================================================== the loop
struct SpecOptions {
    int mode = 1;                // default policy mode (see SpecPolicyOptions::mode); env INK_SPEC=auto|off|4|8 overrides
    bool draft_sample = true;    // temperature > 0: the drafter SAMPLES its drafts (dense q, as in the reference) instead of argmax (one-hot q)
    int exact = -1;              // verify attention: 1 = decode kernels (bit exact at any length), 0 = fast kernel, -1 = env INK_SPEC_EXACT
    SpecPolicyOptions policy;
    LookupOptions lookup;        // session-lookup drafter (agent L): used by modes 1 (auto, before the DSpark policy) and 9 (lookup only); env INK_SPEC_LOOKUP* override
};

struct SpecRequest {
    int32_t anchor = 0;          // last sampled token, NOT yet processed by the engine (it sits at position slot_pos)
    SamplingParams sp;
    uint64_t seed = 0;
    const int32_t* stop_ids = nullptr;
    int n_stop = 0;
    int max_emit = 8;            // >= 1: tokens the step may emit at most
    int mode = -1;               // -1 = decoder default, else SpecPolicyOptions::mode (0 plain, 1 auto, 2..8 fixed) or kSpecModeLookup
};

struct SpecResult {
    int32_t tokens[kVerifyMaxRows];   // emitted tokens (>= 1)
    int n_tokens = 0;
    int stopped = 0;             // a stop id is the last emitted token (never fed to the engine)
    int drafted = 0, accepted = 0, T = 1;
    int src = 0;                 // draft source: 0 none (plain step), 1 DSpark drafter, 2 session lookup
    int match_len = 0;           // src 2: length of the n-gram match that produced the draft
    double draft_us = 0, verify_us = 0, accept_us = 0, total_us = 0;
    uint64_t next_seed = 0;
    int rc = 0;
};

class SpecDecoder {
public:
    // drafter may be null (plain steps only, no tap commits). The decoder installs Engine::tap_sink (only when a drafter exists) and removes it in the destructor.
    SpecDecoder(Engine& e, std::unique_ptr<Drafter> drafter, const SpecOptions& opt = SpecOptions());
    ~SpecDecoder();
    int step(int slot, const SpecRequest& rq, SpecResult* out);   // 0 or a negative error (-1 bad slot, -2 context full, -3 bad token, -4 bad request)
    SpecPolicy& policy(int slot) { return slots_[slot].policy; }
    Drafter* drafter() { return drafter_.get(); }
    bool synced(int slot) const;                                   // drafter context == engine slot
    Engine& engine() { return E_; }
    SpecOptions& options() { return opt_; }
    int64_t sink_dropped() const { return sink_dropped_; }         // tap rows the drafter could not take (slot prefilled before the decoder existed, slot_copy ...)
    // ---- session-lookup history (agent L). step() records the rows it feeds itself; rows processed elsewhere are reported here (ink_forward / prefill, rewind, reset).
    // A step whose history length differs from the slot position repairs it first (truncate, or pad with unknown tokens that never match), so exactness never depends on it.
    void note_tokens(int slot, int64_t first_pos, const int32_t* toks, int n);
    void note_truncate(int slot, int64_t pos);
    void note_reset(int slot);
    const LookupIndex* lookup(int slot) const { return slot >= 0 && slot < (int)slots_.size() ? &slots_[slot].lk : nullptr; }
    LookupPolicy& lookup_policy(int slot) { return slots_[slot].lkp; }
    int last_src() const { return last_src_; }

private:
    struct Slot {
        int st = -1;
        bool synced = true;
        SpecPolicy policy;
        LookupIndex lk;
        LookupPolicy lkp;
    };
    void on_taps(int slot, int64_t first_pos, int rows, const float* taps);
    Engine& E_;
    std::unique_ptr<Drafter> drafter_;
    SpecOptions opt_;
    std::vector<Slot> slots_;
    Sampler smp_;
    std::vector<float> logits_, q_, taps_;
    int64_t sink_dropped_ = 0;
    int last_src_ = 0;
};

// env INK_SPEC: "auto" | "off" | "lookup" | "4" | "8" (any 2..8) -> SpecPolicyOptions::mode (lookup = kSpecModeLookup); returns `def` when unset / unparsable
int spec_mode_from_env(int def);
// env INK_SPEC_LOOKUP (0/1), INK_SPEC_LOOKUP_NMIN, INK_SPEC_LOOKUP_ENTER applied on top of `o`
LookupOptions lookup_options_from_env(LookupOptions o);

}  // namespace ink

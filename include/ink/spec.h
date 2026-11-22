// Speculative decoding on top of the inkcpu engine (agent S2). Design: docs/SPEC_DESIGN.md. Contract: docs/SPEC_CONTRACT.md.
//
//   Sampler / TargetDist  : the target sampler of server/sampling.py in C++ (temperature -> top_k -> top_p -> min_p, no penalties) + exact speculative sampling
//   SpecPolicy            : adaptive verify width T in {1, 4, 8} from measured acceptance and measured step times (deterministic, clock free, unit testable)
//   SpecDecoder           : the loop: tap sink -> drafter commits, draft, Engine verify (engine_verify), accept/reject, stop handling, rollback, commit
#pragma once
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
    double verify_ratio[9] = {0, 1.0, 1.59, 2.0, 2.47, 2.8, 3.1, 3.4, 3.72};   // x (N + E(T)) / B1 of docs/ROUTING.md, index = T (the real ratio is measured)
    double draft_a = 0.21, draft_b = 0.013;                                    // draft(n) ~ plain * (a + b n)
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

// ====================================================================================================================== the loop
struct SpecOptions {
    int mode = 1;                // default policy mode (see SpecPolicyOptions::mode); env INK_SPEC=auto|off|4|8 overrides
    bool draft_sample = true;    // temperature > 0: the drafter SAMPLES its drafts (dense q, as in the reference) instead of argmax (one-hot q)
    int exact = -1;              // verify attention: 1 = decode kernels (bit exact at any length), 0 = fast kernel, -1 = env INK_SPEC_EXACT
    SpecPolicyOptions policy;
};

struct SpecRequest {
    int32_t anchor = 0;          // last sampled token, NOT yet processed by the engine (it sits at position slot_pos)
    SamplingParams sp;
    uint64_t seed = 0;
    const int32_t* stop_ids = nullptr;
    int n_stop = 0;
    int max_emit = 8;            // >= 1: tokens the step may emit at most
    int mode = -1;               // -1 = decoder default, else SpecPolicyOptions::mode
};

struct SpecResult {
    int32_t tokens[kVerifyMaxRows];   // emitted tokens (>= 1)
    int n_tokens = 0;
    int stopped = 0;             // a stop id is the last emitted token (never fed to the engine)
    int drafted = 0, accepted = 0, T = 1;
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

private:
    struct Slot {
        int st = -1;
        bool synced = true;
        SpecPolicy policy;
    };
    void on_taps(int slot, int64_t first_pos, int rows, const float* taps);
    Engine& E_;
    std::unique_ptr<Drafter> drafter_;
    SpecOptions opt_;
    std::vector<Slot> slots_;
    Sampler smp_;
    std::vector<float> logits_, q_, taps_;
    int64_t sink_dropped_ = 0;
};

// env INK_SPEC: "auto" | "off" | "4" | "8" (any 2..8) -> SpecPolicyOptions::mode; returns `def` when unset / unparsable
int spec_mode_from_env(int def);

}  // namespace ink

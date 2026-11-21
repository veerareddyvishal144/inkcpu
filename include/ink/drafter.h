// Interface between the DSpark drafter (agent S1: src/drafter/) and the speculative loop (agent S2: src/spec/). Initial version written by the
// orchestrator; S1 may EXTEND it (additions only, announce them in docs/status/S1.md), S2 codes against it (and against a test double until S1 lands).
// Model details: docs/DSPARK.md (sections 2-6). Contract and ownership: docs/SPEC_CONTRACT.md.
#pragma once
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "ink/engine.h"

namespace ink {

constexpr int kNTaps = 4;                  // target visits 5, 23, 41 (stream entering the visit, DSPARK.md 4.1) + tap 48 (final normalised hidden state)
constexpr int kTapFloats = kNTaps * kD;    // floats per committed row, layout [tap][2304], F32
constexpr int kMaxDraft = 7;               // the drafter always runs an 8-slot block; n only truncates the Markov chain

struct DraftRequest {
    int32_t anchor = 0;        // the last token the target has sampled and NOT yet processed; it sits at position state_pos(st)
    int n = 3;                 // 1..kMaxDraft tokens to draft
    bool sample = false;       // false: argmax chain (ties -> lowest id). true: d_j ~ softmax(logits_j / temperature), q_j returned in q_out
    float temperature = 1.0f;  // only when sample
    uint64_t seed = 0;         // only when sample (deterministic given seed)
};

// ---- S1 additions (2026-11-21, additions only; see docs/DRAFTER.md) ----
struct DrafterOptions {
    bool fold_q8 = true;        // store the folded matrix M (200058 x 256) as Q8_0 (54 MB, default) or bf16 (102 MB)
    std::string cache_dir;      // fold cache directory: "" = next to the drafter GGUF, "-" = no cache file
    int max_states = 0;         // 0 = Engine::opt.n_slots (at least 1)
    bool profile = true;        // rdtsc per phase on thread 0 (negligible cost)
};

struct DrafterProfile {         // accumulated by draft(); times in milliseconds
    enum { P_FLUSH = 0, P_EMBED, P_QKV, P_ATTN, P_WO, P_FFN, P_FINAL, P_BASE, P_CHAIN_U, P_CHAIN_M, P_POST, P_COUNT };
    double ms[P_COUNT] = {};
    uint64_t calls[P_COUNT] = {};
    uint64_t drafts = 0, drafted_tokens = 0;
    double total_ms = 0;        // wall time of the pool run inside draft()
    static const char* name(int p);
};

class Drafter {
public:
    // Loads dspark-Inkling-2-Mini-Q8_0.gguf and prepares what it borrows from the target: lm_head (Engine::lm_head), the embedding table + embed_norm
    // gain (Engine::tok_embd / tok_norm), the folded matrix M = lm_head . hidden_out (computed once at load; may be cached on disk next to the GGUF).
    // All parallel work runs on target.pool (the calling thread is tid 0): never create a second spinning pool.
    static std::unique_ptr<Drafter> create(Engine& target, const std::string& drafter_gguf, std::string* err);
    // same with options (S1 addition)
    static std::unique_ptr<Drafter> create(Engine& target, const std::string& drafter_gguf, const DrafterOptions& opt, std::string* err);
    virtual ~Drafter() = default;

    // One drafter state per sequence (ctx K/V of committed positions, f_rnn). The speculative loop creates one per target slot.
    virtual int state_new() = 0;                       // -1 if none free
    virtual void state_free(int st) = 0;
    virtual void state_reset(int st) = 0;              // forget everything (positions restart at 0)
    virtual int64_t state_pos(int st) const = 0;       // number of committed positions

    // Appends `rows` committed positions (state_pos(st) .. state_pos(st)+rows-1). taps = rows x kTapFloats F32.
    // May compute lazily (e.g. only when draft() needs the context), but the observable behaviour must equal "computed now".
    virtual int commit(int st, const float* taps, int rows) = 0;

    // Drafts after rq.anchor at position state_pos(st). Does not change committed state (the 8 block K/V are scratch).
    // out_tokens: rq.n ids. q_out (only when rq.sample, else may be null): dense draft probabilities over the real vocabulary,
    // row-major [rq.n][Engine::n_vocab()], the distribution d_j was drawn from. Returns 0 or a negative error.
    virtual int draft(int st, const DraftRequest& rq, int32_t* out_tokens, float* q_out) = 0;

    // ---- S1 additions: all have a default so a test double of the interface keeps compiling ----
    // Rows committed but not yet turned into ctx K/V (commit() is lazy; draft() flushes them inside its single pool run).
    virtual int64_t state_pending(int /*st*/) const { return 0; }
    // Computes the pending ctx K/V now (own pool run). Not needed for correctness.
    virtual int flush(int /*st*/) { return 0; }
    // Bytes of resident drafter memory of one state / of the shared weights + fold (approximate).
    virtual size_t state_bytes(int /*st*/) const { return 0; }
    virtual size_t weight_bytes() const { return 0; }
    // Per-phase profile of draft() (rdtsc on thread 0, indicative on a shared host).
    virtual const DrafterProfile* profile() const { return nullptr; }
    virtual void profile_reset() {}
    virtual void profile_report(FILE* /*f*/) const {}
};

}  // namespace ink

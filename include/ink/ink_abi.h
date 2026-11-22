// C ABI of the engine (libink). The Python server binds this with ctypes. Owner: orchestrator.
// Thread-safety: all calls on one engine must come from ONE thread (the server runs the engine in a dedicated thread).
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ink_engine ink_engine;

// Load a GGUF (arch inkling2_loop). n_threads = worker threads (physical cores recommended, 12-16).
// n_slots = number of independent sequence slots, max_ctx = max tokens per slot. Returns NULL on failure (see ink_last_error).
ink_engine* ink_load(const char* gguf_path, int n_threads, int n_slots, int max_ctx);
void        ink_free(ink_engine*);
const char* ink_last_error(void);
const char* ink_version(void);

int ink_n_vocab(const ink_engine*);  // number of logits returned per call (200058 real tokens, padded rows dropped)

// Slots hold one sequence's state (KV for 8 global visits, SWA rings for 40 visits, conv histories).
int  ink_slot_new(ink_engine*);              // returns slot id >= 0, or -1 if none free
void ink_slot_free(ink_engine*, int slot);
void ink_slot_reset(ink_engine*, int slot);  // drop all history
int  ink_slot_pos(const ink_engine*, int slot);  // number of tokens currently in the slot

// Append n tokens to the slot (prefill if n > 1, decode if n == 1). If logits_out != NULL it receives the logits of
// the LAST token, float[ink_n_vocab] (final softcap already applied: 30*tanh(x/30)). Returns 0 on success.
int ink_forward(ink_engine*, int slot, const int32_t* tokens, int n, float* logits_out);

// Fetch counters for the last ink_forward: wall time in microseconds spent in kernels vs total (best effort, may be 0).
void ink_last_timing(const ink_engine*, double* total_us, double* kernel_us);


// ======================================================================= speculative decoding (agent S2; include/ink/spec.h, docs/SPEC_DESIGN.md)
// Call sequence for a server: ink_load -> ink_spec_load (installs the tap hook: do it BEFORE prefilling slots that should speculate) -> per request: ink_slot_reset,
// ink_forward(prompt, logits) -> first anchor = ink_sample(logits) -> loop { ink_spec_step(anchor) -> emit result.tokens; anchor = tokens[n_tokens-1]; seed = next_seed }
// until result.stopped or the budget. After every step ink_slot_pos == previous pos + result.n_fed, and the slot holds exactly fed[0..n_fed-1] on top of what it had.
// A slot prefilled before ink_spec_load (or filled by ink_forward while the drafter is absent) runs plain steps until it is reset (ink_spec_synced tells).
typedef struct ink_sampling_params {
    float   temperature;   // <= 0: greedy argmax (top_k / top_p / min_p ignored); same semantics as server/sampling.py (temperature -> top_k -> top_p -> min_p)
    int32_t top_k;         // 0 = off; 1 = greedy
    float   top_p;         // 1.0 = off
    float   min_p;         // 0.0 = off
} ink_sampling_params;

typedef struct ink_spec_request {
    int32_t anchor;            // the last sampled token, NOT yet processed by the engine (it sits at position ink_slot_pos)
    int32_t mode;              // -1 = engine default (ink_spec_enable / env INK_SPEC), 0 = plain, 1 = auto, 2..8 = fixed verify width T (rows incl. the anchor)
    ink_sampling_params sp;
    uint64_t seed;             // sampled mode: the step is a deterministic function of (slot state, request); chain result.next_seed into the next step
    const int32_t* stop_ids;   // may be NULL when n_stop == 0
    int32_t n_stop;
    int32_t max_emit;          // 1..8: tokens the step may emit at most (the engine applies the budget: the extra verified rows are rolled back)
} ink_spec_request;

typedef struct ink_spec_result {
    int32_t tokens[8];         // emitted tokens (n_tokens >= 1); tokens[n_tokens-1] is the next anchor unless stopped
    int32_t n_tokens;
    int32_t stopped;           // 1: tokens[n_tokens-1] is a stop id; it was NOT fed; the slot holds exactly the tokens before it (like plain decode)
    int32_t fed[8];            // tokens the engine processed in this step: anchor, tokens[0..n_fed-2]
    int32_t n_fed;             // == n_tokens (ink_slot_pos advanced by n_fed)
    int32_t T;                 // verify width used (1 = plain step)
    int32_t drafted, accepted; // drafts proposed / accepted (0 on a plain step)
    int32_t rc;                // 0, or the error (-1 bad slot, -2 context full, -3 bad token, -4 bad request, -10 bad arguments)
    double  draft_us, verify_us, accept_us, total_us;   // accept_us = acceptance + rollback + drafter commit
    uint64_t next_seed;
} ink_spec_result;

int  ink_spec_load(ink_engine*, const char* drafter_gguf);   // loads the DSpark drafter and installs the tap hook; 0 ok (see ink_last_error)
void ink_spec_free(ink_engine*);                             // removes the drafter (ink_spec_step keeps working as a sampled plain decode step)
int  ink_spec_enable(ink_engine*, int mode);                 // default mode for requests with mode = -1: 0 plain, 1 auto, 2..8 fixed; 0 ok
int  ink_spec_mode(const ink_engine*);                       // the default mode in effect (env INK_SPEC applied)
int  ink_spec_synced(const ink_engine*, int slot);           // 1 when the drafter context equals the slot (speculation possible), else 0
// One step: [draft ->] verify / decode -> sample -> stop / budget -> rollback. Returns result->rc (0 ok). ink_last_timing reports total_us of the step afterwards.
int  ink_spec_step(ink_engine*, int slot, const ink_spec_request*, ink_spec_result*);
// Samples one token from a logits row (ink_n_vocab floats, as written by ink_forward) with the sampler the spec step uses (first anchor); next_seed may be NULL.
int32_t ink_sample(const ink_engine*, const float* logits, const ink_sampling_params*, uint64_t seed, uint64_t* next_seed);

#ifdef __cplusplus
}
#endif

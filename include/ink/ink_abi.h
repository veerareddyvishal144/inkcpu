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

#ifdef __cplusplus
}
#endif

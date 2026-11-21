// Batched prompt processing (prefill) for the inkcpu Engine. Owner: agent P. See docs/PREFILL_DESIGN.md.
//
// engine_prefill appends n tokens to a slot in chunks of up to PrefillTune::chunk tokens (<= 1024). The resulting slot state (global KV,
// SWA rings, conv histories, pos) is the state the decode path would have produced, so Engine::decode can continue afterwards.
// Dense matmuls are bit-identical to the decode path (agent G's GEMM kernels == agent B1's GEMV kernels); only attention (different
// kernel, `attn_prefill_group`) differs in summation order unless INK_PF_ATTN=decode (then the whole prefill is bit-identical to decode).
#pragma once
#include <cstdint>
#include <cstdio>

#include "ink/engine.h"

namespace ink {

struct PrefillTune {                    // read from the environment (INK_PF_*) the first time a scratch is created
    int chunk = 256;                    // tokens per chunk (1..1024); INK_PF_CHUNK
    int chunk_proj = 32;                // row chunk of the q|k|v|gate|rel GEMM items; INK_PF_PROJ
    int chunk_wo = 16;                  // row chunk of the wo GEMM items; INK_PF_WO
    int chunk_gu = 32;                  // gate/up row chunk of the MoE items (multiple of 32); INK_PF_GU
    int chunk_dn = 32;                  // down row chunk of the MoE items (multiple of 4); INK_PF_DN
    int tok_claim = 4;                  // tokens per claim in token-parallel phases; INK_PF_TOK
    int attn_tq = 8;                    // query tokens per attention item; INK_PF_ATTN_TQ
    int attn_decode_kernels = 0;        // 1 = run the decode attention kernels per token (bit-identical to decode); INK_PF_ATTN=decode
    int gemm_group = 0;                 // >0: override ink::g_gemm_group (tokens per weight pass); INK_PF_GROUP
};

// The tunables in effect (edit before the first call; chunk and the item sizes may be changed between calls).
PrefillTune& prefill_tune();

// Appends tokens[0..n) to `slot`; logits_last (n_vocab floats, softcap applied) receives the logits of the LAST token, or nullptr to skip the lm head.
// Returns 0 on success, negative on error (-1 bad slot, -2 context overflow, -3 bad token, -4 n < 1).
int engine_prefill(Engine& e, int slot, const int32_t* tokens, int n, float* logits_last);

// Same, but logits_all receives n_vocab floats for EVERY token (row t at logits_all + t * n_vocab; perplexity tooling). 200 MB per 256 tokens.
int engine_prefill_all(Engine& e, int slot, const int32_t* tokens, int n, float* logits_all);

// Per-phase profile (thread 0 rdtsc between barriers; enabled when e.prof_on or env INK_PF_PROFILE=1).
void engine_prefill_profile_reset(Engine& e);
void engine_prefill_profile_report(Engine& e, FILE* f);
// Releases the scratch of an engine (optional: normally lives as long as the process).
void engine_prefill_release(Engine& e);

}  // namespace ink

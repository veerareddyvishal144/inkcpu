# Atomic Engine (AtomicBot-ai/atomic-engine) first pass, orchestrator, 2026-11-24

What it is: a llama.cpp fork (the "TurboQuant" fork used by Atomic Chat) whose `main`/`dev` carry TurboQuant KV types, Gemma-4 MTP, Qwen NextN,
a "decision model" server mode, MoE expert cache (GPU), and Inkling 2 Mini support merged from upstream PR 25731. The README describes an
"Atomic Engine" (expert streaming from SSD under a memory budget, Metal first) whose code lives on branch `codex/expert-weight-source`
(e110381, 2026-11-23), not on `main`. Checkouts: /data/inkling-mini/research/atomic-engine (main) and ../atomic-engine-ews (engine branch, worktree).

## Checked on `main` (nothing to adopt)
* `ggml_vec_dot_nvfp4_q8_0` (ggml/src/ggml-cpu/arch/x86/quants.c:1004): AVX2 only (no AVX-512 path), per-super-block scalar UE4M3 table
  lookups + `_mm256_set_m128(set1, set1)` scale broadcasts, two FMAs per 64 weights. Our `gemv_nvfp4r` (re-encoded scales, 8-row groups,
  14.8 instructions per 36-byte super-block) is well ahead; upstream llama.cpp has no x86 SIMD NVFP4 dot at all.
* Banded flash attention for Inkling (`GGML_OP_FLASH_ATTN_EXT_BANDED`, ops.cpp:8514): scalar per-key loop (kq_vec_dot per key, rel-bias lookup
  per key); our fused rel-bias table + 16-key vector path is the faster design.
* TurboQuant KV (docs/KV-cache-quantization.md): WHT-rotated 2/3.25/4.25-bit K/V (6.4x/4.9x/3.8x vs f16), runtime-only types; explicitly
  gated OFF for Inkling in this fork (banded attention path not implemented for the rotated cache). Conceptually applicable to our attention
  (the relative bias is added after QK, a shared rotation of Q and K preserves the dot product), but it is a precision trade-off and their own
  doc warns that sink-attention models are K-sensitive. Relevance for us: only at long context (attention 8/11 ms per token at 8k/16k).
* MoE expert cache (docs/backend/MOE-CACHE.md): GPU-side cache of CPU-resident experts; not applicable to a CPU-only engine.
* MTP / NextN / DFlash2 / EAGLE-3 / decision mode: other model families, not Inkling.
* Speculative n-gram family (docs/speculative.md): ngram-simple / ngram-map-k / ngram-map-k4v / ngram-mod with drafts up to 64 tokens
  (GPU verify is cheap). Design notes worth keeping: "MoEs require long drafts", shared hash pool across slots (ngram-mod), per-key
  accepted-token statistics (ngram-map-k). Our lookup drafter already does acceptance tracking; the shared pool across slots is an idea for
  multi-agent use.
* DSpark in the fork (docs/speculative.md 112-143, src/models/dflash.cpp:308): Markov head identical in form to ours; a CONFIDENCE head
  (`--spec-draft-conf-min P`, truncate the block at the first position with predicted acceptance < P) exists in the code path, but neither
  the Unsloth drafter nor the AtomicChat drafter for Inkling has one (`has_confidence_head=False` in both GGUF headers).

## AtomicChat drafter GGUF (HF AtomicChat/Inkling-2-Mini-GGUF @3095098, header only)
58 tensors: same 3 DFlash blocks, block_size 8, target_layers [5,23,41,48]; Markov head over the UNPADDED vocab (markov_w1/w2 [256, 200058] BF16,
102 MB each) plus `markov_w2.bias [200058]`, `markov_ctx_mix [4]`, `markov_ctx_norm [2304]`, `has_markov_context = True`. I.e. a retrained
"Markov context" head: details and measured acceptance delegated to the deep-dive agent (docs/research/atomic_engine_deep_dive.md).

## Their Inkling numbers (README, M4 Max 48 GB, 2026-11-23, 4k-token prompts, Q4_0 KV)
Atomic Q4_K at 20 GiB budget: TTFT 9.95 s / 77 tok/s decode (DSpark 3/32); stock llama.cpp 9.5 s / 32 tok/s; Strata (SQ-10 pack, DSpark, fp16 K)
4.2 s / 113 tok/s. Apple silicon bandwidth (546 GB/s) explains the absolute level; the 2.4x over stock on the same box comes from speculation
plus their expert cache, i.e. the same levers we already use on x86 (our DSpark+lookup auto is 1.0-1.75x plain depending on content).

# What inkcpu can learn from AtomicBot-ai/atomic-engine (orchestrator synthesis, 2026-11-24)

Sources: `atomic_engine_firstpass.md` (branch `main`), `atomic_engine_deep_dive.md` (branch `codex/expert-weight-source`, the actual engine work), plus
two checks run by the orchestrator against the AtomicChat drafter GGUF header/tensors. Checkouts under /data/inkling-mini/research (read-only; nothing built or run).

## Verdict
The fork is a Metal + SSD-streaming llama.cpp for Macs with limited memory. Its Inkling-specific work is the upstream DSpark drafter (re-exported in a
second GGUF layout), a conservative prompt-lookup drafter, seven fused Metal kernels, and a budgeted expert cache for streaming the MoE from SSD. Our engine
already has the equivalent or better of each piece that applies to a RAM-resident x86 CPU. Nothing in it changes the RESEARCH_WAVE2.md roadmap
(D32s_v6 VM, more lookup/draft policy work, dense Q8_0 prefill phases). Two of its negative results confirm our own policy choices.

## Checked and settled
1. **AtomicChat drafter = our drafter.** Its `mask_embd [2304 x 7]` is seven bit-identical copies of the Unsloth `mask_embd [2304]` we broadcast
   (max |diff| 0, verified on the downloaded tensor); `markov_w2` / `markov_w2.bias` are the fold `M = lm_head . hidden_out`, `m0` that our drafter already
   computes (`src/drafter/fold.cpp`); the raw `ctx_mix` weights replace `softmax(fuse)`; their own A/B shows the two files within 2 % tok/s. No acceptance upside.
2. **Their x86 NVFP4 kernel** is AVX2-only with scalar scale lookups; ours (`gemv_nvfp4r`, int16 prefill chain) is far ahead. **Their Inkling attention** on CPU is a
   scalar per-key loop. Nothing to port.
3. **Prompt lookup**: 32-token exact suffix match, fixed depth, greedy, no acceptance feedback, DSpark only when no match. A strict subset of our session-lookup
   drafter (8-token minimum, per-match-length acceptance tracking, caps, predicted-speed-up arbitration with DSpark). Keep ours.
4. **Draft-length policy negatives (their Flash Next A/Bs, not Inkling)**: reactive depth switching 3<->2 lost 8 %; gating the last draft row on a heuristic
   confidence gained 0 % at three thresholds; depth 4 lost 11 % vs depth 3 at 53 % acceptance. Consistent with our EWMA width policy; do not add reactive gating.
5. **Q4_0 KV cache for Inkling** in their README: a Metal kernel convenience with no quality measurement anywhere in the repo; Strata (fastest in their table)
   used fp16 K. Our stance (fp16 KV; Q8_0 KV listed as a measured-trade-off option) stands.
6. **The M4 Max table** (Atomic Q4_K 77 tok/s vs stock 32 vs Strata 113) is measured on a fully repetitive copy fixture at 100 % draft acceptance; the gap to
   Strata is draft depth 3 vs 7 plus their grouped kernel being disabled for Q4_K. Apple bandwidth (546 GB/s) sets the absolute level; nothing transfers to x86.
7. **Multi-row expert GEMV for small verify batches** (their `inkling_group_dot<count>`: decode the weight row once, 1..5 accumulators): we already have it
   (`gemv_*_multix`, K2, wave 1) and use it for every expert with <= 8 pairs in verify. Our remaining verify gap to the bytes floor (2.15x vs 1.70x at T=4) is the
   per-row fixed work (loop controllers run 2.7x their decode cost per verify step, ~16 ms of per-row small phases at T=4), not the expert GEMV.
8. **Expert streaming / budgets / early reads / prime wave / resident banks**: SSD-bound design for machines without enough RAM; our model is resident. Their
   "shadow router" (predicting routes ahead) failed their own numerical gate. Not applicable.

## Worth doing (small)
* Add their copy fixture (a ~2000-token document repeated, cropped to 4000 tokens) and the "handover" summarisation prompt to `bench_spec` as
  `lookup_copy` / `lookup_handover`: a shared yardstick with published numbers (100 % acceptance regime; their L(7) = 4.0 on handover). Expected for us:
  ~3.4x plain (~55 tok/s) on the copy fixture with the lookup drafter. 30 minutes.
* Drafter/target vocabulary identity check at `ink_spec_load` (token text per id), as their loader does: safety for third-party drafter GGUFs. 1 hour.
* If we ever accept prefilled-reasoning continuation requests, diff `common/parsers/inkling.cpp:81-96` against our parser. Correctness only.

## Not worth doing
Markov-context GGUF support (same weights), TurboQuant/Q4_0 KV for Inkling (unmeasured quality, their own fork gates it off), MoE expert cache (GPU), MTP/NextN/
DFlash2/EAGLE-3 (no heads exist for Inkling), fused-kernel ports (we have the CPU fusions; our 7 ms fixed cost is barriers and serial depth mixes, for which the
fork has nothing).

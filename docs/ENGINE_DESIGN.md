# ENGINE_DESIGN: decode executor and C ABI (agent C1)

Files: `include/ink/engine.h` (public API + weight/state descriptors), `src/engine/engine_load.cpp` (loader, slots, profile report),
`src/engine/engine_decode.cpp` (the executor), `src/engine/engine_priv.h` (scratch), `src/capi/ink_capi.cpp` (`libink.so`, implements
`include/ink/ink_abi.h`), `tests/engine/` (gates), `bench/bench_decode.cpp`. Build (the root configure is blocked while `src/kernels_gemm`
is mid-edit, so there is a private top level): `cmake -S src/engine/standalone -B build-c1b -G Ninja && ninja -C build-c1b`.
Targets: `ink_engine` (static), `ink` (libink.so), `test_engine`, `bench_decode`, `bench_stream`. The root CMake (`cmake -S . -B build-c1r -G Ninja`) works again since agent G finished
`src/kernels_gemm` and builds the same targets (`ninja -C build-c1r ink test_engine bench_decode bench_stream`, verified 2026-11-21).

## 1. Structure in one paragraph

`Engine::decode(slot, token, logits)` does ONE `ThreadPool::run()` per token. Inside, every thread executes the same straight-line code for
the 48 visits; serial work runs on tid 0 while the others wait in `pool.barrier()`; matvec phases use dynamic chunk scheduling from a shared
atomic counter per phase (counters are numbered identically on every thread because the control flow is data independent; they are zeroed
before each run). Wave 1 (E1) added an OPT-IN variant (`INK_PF_ROUNDS=1|2`, default 0 = off): the first chunk indices of every phase are assigned statically (chunk tid, tid+nth,
...) so that a thread idle before a barrier (serial phases, the 8 threads without a KV group at short context, early finishers) prefetches its first chunks of the next
phase into L2 (`_mm_prefetch` T1). Rows are grouped exactly as before: bit-identical. Measured 2026-11-23 it LOST ~5 % at ctx 16 against plain dynamic scheduling
(software prefetches drain through the L1 fill buffers at ~3 GB/s per core and delayed the barriers even with the stop epoch), hence off by default.
Software prefetches drain through the L1 fill buffers (~3 GB/s per core here), so the prefetch loops poll a stop epoch (`Scratch::pf_stop`) every 1 KB and quit as soon
as the thread everyone waits for bumps it right before entering the barrier (the serial thread, the first finished attention group, the last thread leaving a dynamic
phase): the prefetched volume adapts to the real idle window and never delays the barrier. In phase C each group thread also prefetches its own cold inputs first
(conv histories and taps of its channels, clip scales, the rel-bias rows of the query's block, the other cores' projection outputs, the K column to be written and at
short context the K block it reads; `INK_PF_ATTN_IN=0` disables). Kernels are B1's (`gemv_*`, thread pool) and B2's (norms, clips, short conv, attention, router, depth mix, SiTU);
the only executor-owned arithmetic is the expert combine and a few glue loops. All results are independent of the thread count (checked
bit-for-bit at 5/8/12/16 threads): every reduction either lives inside one kernel call whose result does not depend on how rows are grouped
(B1) or is computed by one thread in a fixed order; the attention key split `nsplit` depends only on the number of visible keys.

## 2. Thread layout

* `n` threads, one per physical core; tid 0 is the calling thread (re-pinned on first use from a new thread). CPU lists
  (`engine_default_cpus`): n <= 15: CPUs {2,4,..,2n}; n = 16: {2,4,..,30,0}; n > 16: the odd siblings are appended (hyperthreads).
  Physical core 0 (CPUs 0/1) runs at half speed on the dev host, so it is only used as the 16th thread and is always last in the order;
  dynamic chunking keeps it from stalling the barriers. Static work (8 KV groups) goes to tid 0..7.
* `Engine::set_threads(n)` rebuilds the pool without reloading weights (used by the determinism test and the benchmark sweeps).
* `INK_CPUS=3,5,7,...` overrides the CPU list (wraps around); useful to stay off the CPUs another tenant pins to (even vs odd CPUs measured the same at load 19: 13.0 vs 13.4 tok/s).
* Idle workers spin 2 ms after a job and then park on a futex (agent T, wave 1; `INK_POOL_PARK=0` = the old 20 us polling), so the first token after a pause costs a futex wake (median +5 us, p90 +30 us).

## 3. Weights (load)

* Every streamed tensor is copied (parallel memcpy in 4 MB pieces on the pool) into ONE 2 MB-aligned `alloc_weight_buffer()` arena
  (MADV_HUGEPAGE), in layer order: wq, wk, wv, attn_gate, attn_rel, wo, experts gate_up, experts down per layer, lm_head last (200058 rows only,
  the padded rows are dropped: 489.7 MB). Total 16.97 GB. `token_embd` stays in the GGUF mapping (one 2.4 KB row per token); after the copy the
  process drops its page-table references to the mapping (`MADV_DONTNEED`, the page cache keeps the file), so resident memory is about the arena + KV.
* F32 tensors that are streamed (router `ffn_gate_inp`, controller down/gain/forget) are converted to BF16 only if EVERY element is exactly
  representable (low 16 bits zero; checked per tensor class at load, all 24 routers and all 144 controller matrices pass here), otherwise an F32 copy is kept
  (`Mat::type` says which). The activations of these matvecs stay float (no bf16 rounding): they are F32 matmuls in llama.cpp.
  Not representable and kept F32: `*_res_query`, `exp_probs_b`, `loop_msg_query`, expert `.scale`, `output_res_query`.
* Rel-bias table repacked with `attn_repack_rel_bias` (16 x 1152 floats per physical layer, 1.77 MB total), the four conv tap sets with
  `sconv_repack_static` (SoA [4][C]).
* Load time (page cache hot): 1.3-2.3 s for 16.97 GB (12 threads, 2026-11-21, load 4-17). A cold cache is bound by the disk (not measured).
* `LayerPacked` (engine.h) is the per-physical-layer descriptor shared by the 3 passes; `SlotState` is the per-sequence state. A batched prefill
  executor can be added next to `run_token`/`visit` and use both unchanged.

## 4. State per slot (`SlotState`)

* KV: 48 `KvLayer` (B2 layout, fp16, blocks of 64 positions; since wave 1 each tensor is a 2 MiB-aligned anonymous mapping with MADV_HUGEPAGE, `INK_KV_HUGE=0` = 4 KiB pages). SWA visits (40): ring of `kSwaRingBlocks` = 32 blocks = 2048 positions (so a batched prefill can append a whole chunk of up to 1025 tokens
  before the attention runs; the attention window is still 1024 by position mask), 2 x 4 MiB per visit = 320 MiB per slot of virtual memory, committed lazily as positions are written.
  Global visits (8): one anonymous `mmap(MAP_NORESERVE)` of `ceil(max_ctx/64)+1` blocks per tensor (lazy, zero filled), `n_blocks = pos/64 + 1` is set before every token;
  32 KiB of KV per cached token for the 8 global visits.
* Conv histories: SoA `[3][C]` for K (1024), V (1024), wo-out (2304), moe-out (2304) per visit: 79.9 KB per visit, 3.83 MB per slot (`hist`).
* `slot_reset` zeroes the KV blocks that were written (`used_blocks`: ring = min(ring, ceil(pos/64)), global = grown extent) and the histories; `slot_copy(src, dst)` copies everything (used by the benchmark to reuse a prefilled context).
* Per-token transient state (residual ping-pong buffers, up to 4 attention-residual block states, the prelude copy, the message bank, all
  activations) lives in `Scratch` (one per engine, ~0.45 MB, stays in L2).

## 5. Phases of one visit

(b = barrier; "serial" = tid 0 only; "dyn" = dynamic chunks from a shared counter.) Visits of passes 1 and 2 (24 of 48) additionally run the
message mix and the loop controllers (+8 barriers).

| phase | what | work split | bytes (SWA / NVFP4 visit) | barrier after |
|---|---|---|---|---|
| A (PH_MIX) | [msgs: depth mix of the 2 bank states, 4 queries] attention-input depth mix (nq 1 or 3), rmsnorm x norm weight, Q8_0 quantisation, bf16 rounding (sink / dyn-tap input) | serial | 0 | yes (SWA, no msgs: Q8 once, all three inputs share it) |
| ctrl (PH_CTRL) | 3 attention controllers batched: gate input (serial), down matvec 4608->64 (chunks of 4 rows), then per (controller, 64-channel block): gain+forget rows 64->2304 fused with `ctrl_apply` of the block; then norms (serial) | dyn | 4.7 MB per visit (also FFN controller) | 2 + 1 |
| B (PH_PROJ) | wq\|wk\|wv\|attn_gate\|attn_rel (one `gemv_multi_q8_0`, different x per matrix) + BF16 sink / dyn-tap matvecs as extra virtual rows | dyn, chunk 32 -> 8 | 19.9 MB (global 20.0) | yes |
| C (PH_ATTN) | per KV group g (tid g): short conv K,V (dyn taps on global layers), K clip, V norm, Q clip, R clip, `kv_append`; then attention. nsplit = 1: the same thread runs the task, `attn_decode_finish` (with sigmoid gate) and the Q8_0 quantisation of its 384 outputs. nsplit > 1 (>= 2 x `split_keys` = 768 visible keys; `attn_decode_nsplit`: keys / split_keys clamped to [1,16] by default, env INK_ATTN_GRANULE=g for ceil(keys/g)): b, items (g, s) dyn, the thread completing the LAST item of a group runs that group's finish (per-group arrival counters) | static / dyn | K+V fp16 of the visible keys | yes (1 or 2) |
| E (PH_WO) | wo matvec (K = 3072) by 32-row chunks; the thread that produced rows [b,e) also does the post-conv and the residual add for those channels (block-start layers: no residual) | dyn | 7.5 MB | yes |
| F1 (PH_FFNMIX) | push inpSA to the block list (block starts), FFN depth mix, [FFN controller], rmsnorm x ffn_norm, Q8_0 | serial | 0 | yes |
| F2 (PH_ROUTER) | router matvec 128 x 2304 (BF16) by 8-row chunks; then EVERY thread computes `router_topk` redundantly (identical result, no broadcast barrier) | dyn | 0.59 MB | yes |
| G (PH_GATEUP) | unit = (expert slot, 32 SiTU indices): 32 gate rows + the matching 32 up rows, per-expert F32 scale, `vsitu_q8_0` into one Q8_0 block of that expert's down input; 288 units | dyn, 1 unit | 23.9 MB (Q8_0 layers 45.2 MB) | yes |
| H (PH_DOWN) | unit = 16-row tile x 8 experts (chunks of 32 rows, 16 at the tail; tiles inside a chunk run back to back): `d_i = (dot_i * down_scale) * w_i`, summed in top-k order (sequential adds, AVX-512, no FMA), then post-conv and `cur = ffn_inp + ffn_out` in place for those channels | dyn, 32 -> 16 rows | 11.9 MB (Q8_0 layers 22.6 MB) | yes |
| end | after visit 47: final depth mix, rmsnorm, Q8_0 (serial), b, lm_head 200058 rows in 512 -> 64-row chunks, each chunk followed by `vsoftcap` on its own rows | dyn | 489.7 MB | (join) |

8 barriers per SWA visit without messages with nsplit 1 (A, B, C, E, F1, F2, G, H), 9 with nsplit > 1, 14-15 on message visits: 528 barriers per token at ctx < 768, 576 beyond
(before wave 1: 10 / 16-18, i.e. 576 / 672). The barrier is tid 0 polling 15 arrive lines in sequence then one release store (~1.5-2 us when everyone has arrived).

## 6. Numerics decisions

* Activation quantisation: B2's `quantize_q8_0(..., Q8Mode::X86)` everywhere (id = 127/amax, round-half-even: what llama.cpp RUNS on this CPU); B1's standalone
  quantiser (`Ref` variant) is not used. `rmsnorm` + `quantize_q8_0` are called back to back (bit-identical to the fused function; the float result is also needed).
* BF16 weights (sink, dyn-tap projections): the normalised activation is rounded to bf16 (RNE) first; B1's `gemv_bf16` accumulation order differs from ggml's (two FMA chains), so these
  are close to, not bit-equal to, llama.cpp.
* Known deviations from llama.cpp that the scalar reference reproduces and this engine does not: (1) the relative bias (incl. sink term) is not rounded to F16 inside B2's attention
  kernel (it is folded into the score accumulation); MODEL.md measured this rounding at ~5e-4 of cosine on s1; (2) F32/BF16 dot orders of router and controllers (tinyBLAS order) and
  the B1/B2 summation orders; (3) vector exp/tanh/sigmoid approximations (<= 3e-7 relative); (4) FMA in the short conv taps. Flips of Q8_0 activation elements and of the top-8 expert set
  amplify these to the logit level (docs/MODEL.md section 9); the measured parity is in section 8.
* Router: probabilities = sigmoid, selection = p + bias, ties -> lowest index (B2), `w = p / (sum_double + 1e-20) * 8`.
* tau (global layers, pos >= 8191) follows the float operation order of the reference; the attention kernel applies it to Q and R itself (so it is NOT applied in the executor).

## 7. Profiling

`INK_PROFILE=1` (or `EngineOptions::profile`): thread 0 reads rdtsc after every barrier, so a phase is charged the time between barrier completions (it includes waiting for stragglers
and the barrier itself); `Engine::profile_report` prints ms/token, share, calls, MB/token and GB/s per phase type (bytes = weights streamed + KV read, analytical). `bench_decode`
prints it per cell. `ink_free` prints it on exit when profiling is on.

## 8. Validation (real GGUF `Inkling-2-Mini-UD-Q4_K_XL`, 2026-11-21; `build-c1b/engine/test_engine <mode>`, ctest-style plain executables)

(a) `oracle`: 6 prompts token by token, final-position logits vs `tools/oracle/<p>/logits_last.f32` (llama.cpp) and vs the scalar RefModel (cached in `build-c1/refcache`):

| prompt | tokens | argmax oracle / engine | top-10 overlap | cos(engine, oracle) | cos(engine, RefModel) | cos(RefModel, oracle) |
|---|---|---|---|---|---|---|
| s1_en_capital | 13 | 154328 / 154328 | 10 | 0.999831 | 0.999719 | 0.999866 |
| s2_code_add | 14 | 68014 / 68014 | 10 | 0.999824 | 0.999868 | 0.999935 |
| s3_en_story | 15 | 149878 / 149878 | 10 | 0.999124 | 0.999149 | 0.999144 |
| c1_chat_user | 14 | 200008 / 200008 | 9 | 0.999985 | 0.999902 | 0.999893 |
| m1_en_llm | 43 | 151787 / 151787 | 10 | 0.999066 | 0.999569 | 0.998854 |
| m2_code_c | 52 | 407 / 407 | 9 | **0.991664** | 0.997890 | 0.994058 |

Same argmax 6/6, overlap >= 9 on 6/6, cosine >= 0.999 on 5/6. m2_code_c is below 0.999 but this is the documented noise floor: the scalar reference (which reproduces llama.cpp's arithmetic much more closely)
only reaches 0.99406 against llama.cpp on the same prompt, and the engine is within 0.0021 of the reference (0.99789). The known fidelity gaps of the engine are listed in section 6; no structural error
(per-visit comparison against the reference for the first tokens: relative L2 of l_out 1e-3..1e-2, final logits cosine 0.999996 / 0.99996 on tokens 0 / 1 of s1).

(b) `greedy`: 5 prompts x 64 tokens vs `tools/oracle/greedy`: s1 and s2 64/64 identical; m1_en_llm first divergence at step 1 (llama.cpp top1-top2 margin 0.161), m2_code_c step 39 (0.475),
c1_chat_user step 39 (0.270). All divergences are near-ties (margin < 2.0): PASS.

(c) `nll`: mean NLL over two 256-token wikitext-2 test chunks (255 predicted positions each; tokenised by `tools/build/oracle_logits`, llama.cpp side = its `logits_all.f32`, data made by
`tests/engine/make_nll_data.sh`): chunk 1 2.11948 vs 2.11793 (|d| 0.00155), chunk 2 1.70466 vs 1.71145 (|d| 0.00680); limit 0.01: PASS.

(d) `determinism`: logits of 40 positions bit-identical across two runs at 16 threads, 12, 8 (two runs) and 5 threads: PASS. (Also the 1305-token run: 8 vs 12 threads bit-identical.)

(e) `slots`: reset + rerun of the same slot, continue (7 + 6 tokens vs one pass of 13), two interleaved slots (s1 and s2 token by token) and `slot_copy` + continue are all BIT-IDENTICAL to the isolated runs.
Ring wrap: 1305 wikitext tokens decoded (SWA ring wrapped on the 1024-position ring of the first build): logits finite, max |logit| 17.9, argmax identical to llama.cpp, top-10 overlap 10, cosine 0.999375
(llama.cpp ran the same 1305 tokens as one batch). After the coordinator's request the SWA ring is 32 blocks (2048 positions); the wrap of the new ring and the unchanged behaviour are checked in section 8.1.

C ABI: `tests/engine/test_capi.py` (ctypes on libink.so): load, 2 slots, `ink_forward` with n = 13 (argmax 154328, cosine 0.999831 vs oracle), decode, `ink_slot_pos/reset/free`, error return on a bad slot: PASS.

### 8.1 SWA ring of 32 blocks (coordinator request, 2026-11-21)

`kSwaRingBlocks = 32` (2048 positions) replaces the 16-block ring; the decode path is untouched, only `alloc_slot`, `slot_reset` and `slot_copy` changed (`used_blocks`). Checks on the new build (`build-c1d`):
* the 1305-token run (wraps the OLD 1024-position ring, not the new one): logits after the last token are BIT-IDENTICAL between `INK_SWA_BLOCKS=16` and 32 (`cmp` of the saved logits), cosine vs llama.cpp 0.9993749 in both cases.
* `ring2`: 2683 wikitext tokens (wraps the new ring after position 2048; llama.cpp ran them as one batch): all finite, max |logit| 25.9, argmax 53399 = llama.cpp, top-10 overlap 9, cosine 0.9995351: PASS.
* slot semantics (reset/continue/interleave/slot_copy) bit-identical on the new build; determinism / oracle / greedy gates re-run on the new build: see the lines below.
* gate (d) on the new build: 40 positions bit-identical across two runs at 16, 12, 8 (x2) and 5 threads: PASS. Gate (a): the printed table of section 8 is reproduced digit for digit (same 6 argmax, same overlaps, same cosines).
  Gate (b): same divergence steps and margins as before (s1, s2 identical; m1 step 1, m2 step 39, c1 step 39): PASS.

## 9. Performance (`bench/bench_decode.cpp`, run under `tools/with_bench_lock.sh`; best of 2 timed runs of 64 greedy tokens after 4 warm-up tokens)

**Host conditions: the box was shared with other agents all the time (1-min load average 11-24 during every number below, `llama-speculative-simple`, `test_prefill`, llama.cpp oracle runs
at 6-8 busy cores each). All numbers are INDICATIVE, not idle-host numbers; the clean idle-host stock baseline is 5.45 +- 0.05 tok/s (BASELINE.md / ENGINE.md).** THP is effective (16.6 GB of the 17.1 GB anonymous memory in AnonHugePages).

| context | 8 threads | 12 threads | 16 threads | load avg during the cell |
|---|---|---|---|---|
| 16 tokens (decode of 64 tokens after a 16-token prompt) | 10.17 (98.3 ms) | 13.04 (76.7 ms; runs 12.37, 13.04) | 13.91 (71.9 ms; runs 13.35, 13.91) | 16.8-17.8 |
| 1000 tokens | 9.44 (105.9 ms) | 12.55 (79.7 ms; runs 12.00, 12.55) | 12.42 (80.5 ms; runs 11.66, 12.42) | 20.2-22.3 |

Other measurements (same period, ctx 16, 12 threads; the last bullet is from a later cell at load 27-31): 12.2-13.8 tok/s over 6 repeated runs of the unguided / guided scheduling A/B sweep; 24 threads (2 per core) 13.0, 32 threads 9.7-10.8 (worse), 14 threads 13.9.
Oversubscription warning: with 24 threads (2 per core) at load 28-31 the spin barriers collapsed to 4.1 tok/s (12 threads in the same minutes 10.6, 16 threads 12.5): never run more threads than free cores; the hot path never yields the CPU.
Tunable sweeps (INK_PF=256 / 1024, INK_PFH=1, chunk sizes 16 / 32) all landed within noise or worse than the defaults (prefetch 512 B, T0), which are kept. nsplit for attention: at ctx 1000 the 2-way split
(`split_keys` 384) gave 12.55 vs 11.12 tok/s for nsplit 1 at 12 threads (and 12.42 vs 12.10 at 16): kept.
Final root-built binary (SWA ring 32 blocks, `build-c1r`), load 17-19: ctx 16 = 10.41 / 12.97 / 12.98 tok/s at 8 / 12 / 16 threads. Ring A/B at ctx 1000, 12 threads, same window (load 12-17): 16-block ring 12.45 and 12.86, 32-block ring 12.88 and 9.05
(the 9.05 cell overlapped a 11-core `test_prefill` run of another agent, load 12 -> 14): no performance change from the ring size. The ctx-1000 row of the table above comes from the earlier ring-16 build; the same-window A/B shows the same speed.
Contention sensitivity (same binary family, ctx 16, 48 tokens, 3 reps): a later run of the final root-built binary at load 20-26 (agent P's `test_prefill` on ~9 cores and a llama.cpp oracle run on ~7 cores, both pinned to / floating over the same CPUs)
gave only 9.45 tok/s (12 threads) and 10.27 (16 threads); expert phases fell to 30-40 GB/s. The spin barriers make every descheduled or hyperthread-shared thread a stall for all; an idle-host number has NOT been obtained in this session
(ESTIMATE from the streaming ceiling: 92.5 GB/s x 70% = ~65 GB/s on 3.865 GB, i.e. ~16 tok/s).
Final root-built binary, ctx 1000 (load 28-29, rest of the benchmark after my other processes ended): 12 threads 12.62 tok/s (reps 11.41, 12.62), 16 threads 12.48 (reps 11.98, 12.03, 12.48).
Oversubscription pathology (measured by accident, 23:20-23:27): while my own C ABI test and an oracle gate (12 threads each, same default CPU list) overlapped a benchmark at load 24-35, the benchmark collapsed to 4.4-5.4 tok/s (ctx 16) and 5.0 (ctx 1000, 8 threads),
and the 13-token C ABI test needed 237 s; after those processes ended the same benchmark process measured 12.62 and 11.41 tok/s (ctx 1000, 12 threads, load 29). Two spin-barrier pools time-slicing on the same CPUs convoy.
Rule: run engine processes serially or give them disjoint CPUs (`INK_CPUS`, or `EngineOptions::cpus`). Benchmarks must run under `tools/with_bench_lock.sh` and with nothing else of ours running.
Reference points: the target is >= 11.5 tok/s (2x stock); 13.0-13.9 tok/s at load ~17 is 2.4-2.5x the stock 5.45 tok/s, which itself was measured on an idle host.

### Per-phase table (12 threads, ctx 16, 64 tokens, load ~17, rdtsc on thread 0, 76.7 ms/token, 13.1 tok/s if serialised at this speed)

| phase | ms/token | % | per visit (us) | MB/token | GB/s |
|---|---|---|---|---|---|
| embed | 0.07 | 0.1 | | 0 | |
| attention-input mix + norm (serial, +msgs mix on 24 visits) | 1.36 | 1.8 | 28 | 0 | |
| loop controllers (24 visits, 6 barriers each) | 3.11 | 4.1 | 129 per message visit | 113 | 36.5 |
| projections q,k,v,gate,rel,sink (+dyn taps) | 16.20 | 21.2 | 337 | 1015 | 62.7 |
| conv/clip/kv append + attention (ctx 16) | 1.32 | 1.7 | 27 | 10 | |
| wo + post-conv + residual | 6.28 | 8.2 | 131 | 361 | 57.5 |
| ffn mix + norm (serial) | 0.54 | 0.7 | 11 | 0 | |
| router matvec | 0.74 | 1.0 | 15 | 28 | 38.0 |
| experts gate_up + SiTU | 26.04 | 34.1 | 542 | 1232 | 47.3 |
| experts down + post-conv + residual | 14.12 | 18.5 | 294 | 616 | 43.6 |
| final mix + norm | 0.01 | 0 | | 0 | |
| lm_head (200058 rows) + softcap | 6.50 | 8.5 | | 490 | 75.3 |
| total | 76.29 | | | 3865 | 50.7 |

At 16 threads (same conditions): proj 15.2 ms (66.8 GB/s), wo 7.0, gate_up 23.95 (51.4 GB/s), down 12.8 (48.1 GB/s), lm_head 5.8 (84.8 GB/s), ctrl 3.8, serial mixes 3.2, total 74.5 ms.
At ctx 1000 the attention phase grows from 1.3 to 5.8 ms (12 threads; 201 MB of fp16 K/V read per token at ~35 GB/s, 144 calls = 3 barriers per visit), i.e. +4.5 ms for 1000 cached keys.

### What limits the speed (evidence)
* Pure streaming ceiling measured with the same pool/hugepages in the same minutes (`bench/bench_stream.cpp`, load 11-18): 75-78 GB/s at 12 threads, 82-84 at 16, 85 at 24; the same read split into barrier-separated phases
  of 3-12 MB (the engine's phase sizes) loses only 3-10 us per phase (69.5-71.6 GB/s at 12 threads). The engine moves 3.865 GB per token = 50.7 GB/s average at 12 threads (53.7 at 16): about 70% of the ceiling.
* The Q8_0 phases run close to the ceiling (proj 63-67 GB/s, lm_head 75-85 GB/s, wo 52-57). The NVFP4 expert phases, 52% of the time, run at 44-51 GB/s: B1 measured the per-thread NVFP4 compute rate at 7.4 GB/s L2-resident,
  so 12-16 threads need 3.2-3.9 GB/s each = 43-53% of that rate on a host whose hyperthread siblings are busy with other agents' jobs. The kernel (about 23 uops per 36 B row superblock, 13 of them on the two vector ALU ports,
  11 of those for the scale decode and shuffles) is the limit, not the phase structure. If the expert phases ran at the proj phases' efficiency (63 GB/s) the token would take about 65 ms (15.4 tok/s): an ESTIMATE, not measured.
* Fixed per-token overhead outside streaming matvecs (serial mixes and norms, controllers, router, attention at short context, embed, 8-10 barriers per visit): about 7 ms = 9% (controllers 4%).
* Contention with other tenants: the same phases ran 10-25% slower in the 8-thread and loaded cells; this is why the numbers above are marked indicative.

### Reproduce
```
cmake -S src/engine/standalone -B build-c1d -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build-c1d      # or the root CMake: ninja -C <root build> ink test_engine bench_decode
INK_PROFILE=1 tools/with_bench_lock.sh build-c1d/engine/bench_decode --threads 8,12,16 --ctx 16,1000 --n 64 --reps 2 --splits 384,100000
# wave 1 (E1): the context is built with Engine::forward (batched prefill) so long contexts are cheap; --granules 0,256 sweeps the split rule, --dump P writes P_ctx<c>_t<n>.f32
# (64 greedy logits rows) for bit-identity checks (cmp), --tokens FILE uses an int32 token file, INK_FINE=1 adds a thread-0 rdtsc breakdown of phases A / C / F1
INK_PROFILE=1 tools/with_bench_lock.sh build-e1/src/engine/bench_decode --threads 16 --ctx 16,1000,4000,8000,16000 --n 64 --reps 2 --dump build-e1/ref/ref
tools/with_bench_lock.sh build-c1d/engine/bench_stream 12,16 3.0
INK_THREADS=12 nice -n 10 build-c1d/engine/test_engine oracle | greedy | determinism | slots <ring dirs> | nll <dirA> <dirB> | trace 2
tests/engine/make_nll_data.sh build-c1b/data      # llama.cpp side data for (c) and the ring-wrap runs
/data/engine/.venv/bin/python tests/engine/test_capi.py build-c1d/capi/libink.so
```


### 9.2 Wave 1, agent E1 (2026-11-23): long-context attention and fixed per-token overheads (full log: docs/status/E1.md)

Method: `bench_decode` builds the context with the batched prefill, 16 threads, 2 x 64 greedy tokens per cell, same-window A/B under the lock on an idle box
(the loadavg printed during a run is the run's own 16 spinning threads). Baseline = the pre-wave build in the same harness.

| ctx | attention phase ms/token: baseline -> final | KV GB/s | whole token tok/s: baseline -> final (best of 2) |
|---|---|---|---|
| 16 | 1.97 -> 1.5-2.0 | - | 13.65 -> 13.8-14.8 |
| 1000 | 6.80 -> 5.2-6.0 | 30 -> 33-39 | 12.85 -> 13.0-13.3 |
| 4000 | 8.62 -> 7.3-7.5 | 35 -> 40 | 12.75 -> 12.4-13.0 |
| 8000 | 9.95 -> 8.1 | 43 -> 53 | 12.58 -> 12.5-12.9 |
(Identical configurations vary by 5-7 % in whole-token tok/s between runs on this shared box; the attention-phase reduction is the consistent part.)

What did it (all bit-identical: logits dumps at ctx 16 / 1000 / 4000 / 8000 equal the pre-wave build byte for byte; oracle / greedy / determinism / slots+ring / test_prefill
quick unchanged): (1) software prefetch 4 KiB ahead inside the attention kernel across block and chunk boundaries (+17-25 % per-thread DRAM streaming in the
kernel microbench, 7.5 -> 9.4 GB/s at 32K context); (2) the thread that completes the last split of a KV group runs that group's merge (2 barriers per split visit
instead of 3); (3) the controllers' gain+forget matvec fused with ctrl_apply (2 barriers per call instead of 3); (4) each attention group thread prefetches its own
cold inputs first (histories, taps, clip scales, rel rows, other cores' projections, the K column: task 6.9 -> 4.3 us per visit); (5) KV mapped with MADV_HUGEPAGE
(no measurable effect at <= 4000 context; kept). Tried and left OFF: a finer key split above 767 keys (`INK_ATTN_GRANULE`: no gain, changes rounding), static-first
chunk assignment + idle-thread weight prefetch (`INK_PF_ROUNDS`: -5 % even with a stop epoch, because software prefetches drain through the L1 fill buffers at
~3 GB/s per core and delay the barriers).

What limits the rest (measured with `INK_FINE=1`, thread 0 rdtsc + thread 1 arrival stamps): the marginal KV streaming runs at 57-72 GB/s (like the projection
phase); the remaining attention cost is a fixed ~30-40 us per visit of which the barrier after the group work is 17-30 us although the idle threads arrived
15 us earlier -- the box's idle PRODUCTION server preempts the 16 spinning threads ~3000 times per second (220 per token; measured as nonvoluntary context switches,
matched by the server's 3311 voluntary switches/s), i.e. 2-3 ms/token on every decode measured today. Restarting production on the futex-parking pool (agent T)
should recover that for everyone. At ctx 1000 the split visits still lose ~40 % of their time to barrier imbalance (thread 0: items 44 us, final barrier 34-46 us).

## 10. Where the next speed-up is (analysis, not implemented)

1. NVFP4 row kernel (B1's `nvfp4_rows`, 52% of the token). Per 36-byte row superblock it issues ~23 uops, 13 of them on the two vector ALU ports (p0 fused with p1, and p5): srlv, and, pshufb, maddubs, madd, sub for the weights,
   then 3 `vinsertps` + mul + `vpermps` to build the 16-lane scale vector (8 scalar table loads), cvtdq2ps and fma. That is ~6.5 cycles per row superblock on the vector ports, i.e. ~11-13 GB/s per core at AVX-512 clocks,
   and B1 measured 7.4 GB/s resident. Two independent savings: (a) build the scale vectors for the 4 rows of a group together (gather the four 4-byte scale words with one `vmovd` + 3 `vpinsrd`, one `vpmovzxbd zmm`, one shift/multiply for
   all 16 scales, then one `vpermps` per row): about 10.5 instead of 13 ALU uops per row superblock (-19%) and 3 instead of 10 loads; (b) the bit trick `float(byte << 20) * 2^k` replaces the 256-entry table but is only exact for bytes with
   exponent >= 1 (UE4M3 denormals would hit the denormal-multiply assist), so it needs a one-time re-encoding of the scale bytes at load (UE4M3 -> normalised UE5M3, same size, all 256 values exactly representable). Expected gain: 5-10% on the
   expert phases (3-5% of the token), more where the kernel is compute bound (idle host with free hyperthread siblings).
2. Load balance at barriers: phases are 15-50 us granular; a thread descheduled by another tenant stalls every barrier for its time slice (24 threads at load 28: 4.1 tok/s). On a dedicated host this disappears; on a shared one fewer threads than free cores is the only remedy.
3. Smaller items measured in the profile: loop controllers 3.1 ms (4%: 6 barriers per message visit; the 64 -> 2304 up-projections would be cheaper as column-major axpy), serial mixes 1.9 ms (cache the normalised block states),
   overlap of the serial phases with weight prefetch by the idle threads (~1.5% at best, DRAM can deliver only ~1.6 MB in a 20 us window).
4. lm_head already runs at 75-85 GB/s; the Q8_0 matvec phases reach 55-67 GB/s against a 70-77 GB/s phase-streaming ceiling at 12 threads.

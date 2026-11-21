# ENGINE_DESIGN: decode executor and C ABI (agent C1)

Files: `include/ink/engine.h` (public API + weight/state descriptors), `src/engine/engine_load.cpp` (loader, slots, profile report),
`src/engine/engine_decode.cpp` (the executor), `src/engine/engine_priv.h` (scratch), `src/capi/ink_capi.cpp` (`libink.so`, implements
`include/ink/ink_abi.h`), `tests/engine/` (gates), `bench/bench_decode.cpp`. Build (the root configure is blocked while `src/kernels_gemm`
is mid-edit, so there is a private top level): `cmake -S src/engine/standalone -B build-c1b -G Ninja && ninja -C build-c1b`.
Targets: `ink_engine` (static), `ink` (libink.so), `test_engine`, `bench_decode`.

## 1. Structure in one paragraph

`Engine::decode(slot, token, logits)` does ONE `ThreadPool::run()` per token. Inside, every thread executes the same straight-line code for
the 48 visits; serial work runs on tid 0 while the others wait in `pool.barrier()`; matvec phases use dynamic chunk scheduling from a shared
atomic counter per phase (counters are numbered identically on every thread because the control flow is data independent; they are zeroed
before each run). Kernels are B1's (`gemv_*`, thread pool) and B2's (norms, clips, short conv, attention, router, depth mix, SiTU);
the only executor-owned arithmetic is the expert combine and a few glue loops. All results are independent of the thread count (checked
bit-for-bit at 5/8/12/16 threads): every reduction either lives inside one kernel call whose result does not depend on how rows are grouped
(B1) or is computed by one thread in a fixed order; the attention key split `nsplit` depends only on the number of visible keys.

## 2. Thread layout

* `n` threads, one per physical core; tid 0 is the calling thread (re-pinned on first use from a new thread). CPU lists
  (`engine_default_cpus`): n <= 15: CPUs {2,4,..,2n}; n = 16: {2,4,..,30,0}; n > 16: the odd siblings are appended (hyperthreads).
  Physical core 0 (CPUs 0/1) runs at half speed on the dev host, so it is only used as the 16th thread and is always last in the order;
  dynamic chunking keeps it from stalling the barriers. Static work (8 KV groups) goes to tid 0..7.
* `Engine::set_threads(n)` rebuilds the pool without reloading weights (used by the determinism test and the benchmark sweeps).
* Idle workers spin 2 ms after a job and then poll with 20 us sleeps (ThreadPool default), so the first token after a pause costs <= ~100 us extra.

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

* KV: 48 `KvLayer` (B2 layout, fp16, blocks of 64 positions). SWA visits (40): ring of 16 blocks = 1024 positions, 2 x 2 MiB per visit (160 MiB per slot).
  Global visits (8): one anonymous `mmap(MAP_NORESERVE)` of `ceil(max_ctx/64)+1` blocks per tensor (lazy, zero filled), `n_blocks = pos/64 + 1` is set before every token;
  32 KiB of KV per cached token for the 8 global visits.
* Conv histories: SoA `[3][C]` for K (1024), V (1024), wo-out (2304), moe-out (2304) per visit: 79.9 KB per visit, 3.83 MB per slot (`hist`).
* `slot_reset` zeroes the used KV blocks and the histories; `slot_copy(src, dst)` copies everything (used by the benchmark to reuse a prefilled context).
* Per-token transient state (residual ping-pong buffers, up to 4 attention-residual block states, the prelude copy, the message bank, all
  activations) lives in `Scratch` (one per engine, ~0.45 MB, stays in L2).

## 5. Phases of one visit

(b = barrier; "serial" = tid 0 only; "dyn" = dynamic chunks from a shared counter.) Visits of passes 1 and 2 (24 of 48) additionally run the
message mix and the loop controllers (+8 barriers).

| phase | what | work split | bytes (SWA / NVFP4 visit) | barrier after |
|---|---|---|---|---|
| A (PH_MIX) | [msgs: depth mix of the 2 bank states, 4 queries] attention-input depth mix (nq 1 or 3), rmsnorm x norm weight, Q8_0 quantisation, bf16 rounding (sink / dyn-tap input) | serial | 0 | yes (SWA, no msgs: Q8 once, all three inputs share it) |
| ctrl (PH_CTRL) | 3 attention controllers batched: gate input (serial), down matvec 4608->64, gain+forget 64->2304, `ctrl_apply` per channel range; then norms (serial) | dyn | 4.7 MB per visit (also FFN controller) | 3 + 1 |
| B (PH_PROJ) | wq\|wk\|wv\|attn_gate\|attn_rel (one `gemv_multi_q8_0`, different x per matrix) + BF16 sink / dyn-tap matvecs as extra virtual rows | dyn, chunk 32 -> 8 | 19.9 MB (global 20.0) | yes |
| C (PH_ATTN) | per KV group g (tid g): short conv K,V (dyn taps on global layers), K clip, V norm, Q clip, R clip, `kv_append`; then attention. nsplit = 1: the same thread runs the task, `attn_decode_finish` (with sigmoid gate) and the Q8_0 quantisation of its 384 outputs. nsplit > 1 (>= 2 x `split_keys` = 768 visible keys): b, items (g, s) dyn, b, finish per group | static / dyn | K+V fp16 of the visible keys | yes (1 or 3) |
| E (PH_WO) | wo matvec (K = 3072) by 32-row chunks; the thread that produced rows [b,e) also does the post-conv and the residual add for those channels (block-start layers: no residual) | dyn | 7.5 MB | yes |
| F1 (PH_FFNMIX) | push inpSA to the block list (block starts), FFN depth mix, [FFN controller], rmsnorm x ffn_norm, Q8_0 | serial | 0 | yes |
| F2 (PH_ROUTER) | router matvec 128 x 2304 (BF16) by 8-row chunks; then EVERY thread computes `router_topk` redundantly (identical result, no broadcast barrier) | dyn | 0.59 MB | yes |
| G (PH_GATEUP) | unit = (expert slot, 32 SiTU indices): 32 gate rows + the matching 32 up rows, per-expert F32 scale, `vsitu_q8_0` into one Q8_0 block of that expert's down input; 288 units | dyn, 1 unit | 23.9 MB (Q8_0 layers 45.2 MB) | yes |
| H (PH_DOWN) | unit = 16 output rows x 8 experts: `d_i = (dot_i * down_scale) * w_i`, summed in top-k order (sequential adds, AVX-512, no FMA), then post-conv and `cur = ffn_inp + ffn_out` in place for those channels | dyn, 16 rows | 11.9 MB (Q8_0 layers 22.6 MB) | yes |
| end | after visit 47: final depth mix, rmsnorm, Q8_0 (serial), b, lm_head 200058 rows in 512 -> 64-row chunks, each chunk followed by `vsoftcap` on its own rows | dyn | 489.7 MB | (join) |

8 barriers per SWA visit without messages with nsplit 1 (A, B, C, E, F1, F2, G, H), 10 with nsplit > 1, 16-18 on message visits. ~520 barriers per token at ~1-1.5 us.

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

RESULTS_PLACEHOLDER

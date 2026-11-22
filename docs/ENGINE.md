# inkcpu: decision log and measurements

## 2026-11-21 Phase 0 (CPU-only profile of llama.cpp) and the go decision

Setup: stock llama.cpp mainline @2a28df0f9 (loop fix), CPU build with the system toolchain, GGUF `Inkling-2-Mini-UD-Q4_K_XL`
(17.63 GB: experts L0-19 NVFP4, L20-23 Q8_0, attention/embed/lm_head Q8_0, small tensors bf16/F32), quiet host (load < 1).

Baseline (tuned): decode 5.2-5.7 tok/s (best 5.71 at 12 threads with `OMP_WAIT_POLICY=active OMP_PLACES=cores
OMP_PROC_BIND=close`; `OMP_WAIT_POLICY=passive` collapses to 1-2 tok/s); prefill 23 tok/s at 16 threads.
Measured streaming read bandwidth of the box: 92.5 GB/s (16 threads). Bytes read per decoded token for this GGUF
~4.0 GB (estimate) => 17-20 tok/s ceiling at 70-80 GB/s; llama.cpp reaches ~23 GB/s effective (25%).

Where decode time goes (wall time per op as seen by thread 0, 33 tokens, 173 ms/token):
- MoE NVFP4 gate_up+down: 29% (effective ~32 GB/s). llama.cpp has NO x86 SIMD NVFP4 dot; only `ggml_vec_dot_nvfp4_q8_0_generic`.
- UNARY (tanh/sigmoid/softcap etc.): 14.8% = 25 ms/token (scalar libm), should be ~1-2 ms.
- Q8_0 attention matmuls: 19% (~41 GB/s effective); lm_head 3.9% (73 GB/s: fine).
- ~6000 tiny graph nodes per token (MUL/ADD/CONCAT/SCALE/CONT/RMS_NORM/...), each followed by a thread barrier:
  barrier waiting = 25.8% of wall time; tiny ops = ~18% on top of that.
- F32 matmuls (controllers, router; stored F32 in the GGUF although the original is bf16): ~4%.
Where prefill time goes (42.8 ms/token): NVFP4 MoE 55% (scalar kernel), UNARY 7%, Q8_0 attention ~11%, F32 matmuls 7%.

Gate: headroom is large (decode ~3.5x of bandwidth, prefill ~4-5x of int8 compute) and mostly structural (graph
fragmentation + missing SIMD kernels), so a purpose-built engine is justified. Decision: build `inkcpu`, CPU-only.
Plan: scalar reference (oracle-validated against llama.cpp on the same GGUF) -> AVX-512 kernels -> fused phase executor with
pinned spin-barrier thread pool -> prefill GEMM path -> speculative decoding if routing data says it pays -> server.
Targets: decode >= 2x tuned llama.cpp (>= 11.5 tok/s; stretch 3x = 17), prefill >= 3x (>= 70 tok/s), logits cosine >= 0.999
and greedy agreement vs llama.cpp on the same GGUF, resident memory < 25 GB.

Caveats: single measurements on a shared VM; baseline numbers are per `plan/raw/baseline_matrix.txt`; the 4.0 GB/token figure is
an estimate until E's exact table lands (docs/BASELINE.md).

## 2026-11-22 Final results (orchestrator, idle host, `docs/raw_bench_final2.log`, build-orch @ commit 4378df9+)

Conditions: no other engine process on the box, host load 2.3 before the run (the 12-14 load during the run is the benchmark's own threads), `tools/with_bench_lock.sh`,
one pool thread per physical core (CPUs 2,4,...; core 0 avoided), THP hugepage arena 16.97 GB, same GGUF as the baseline. Stock baseline = tuned llama.cpp (`OMP_WAIT_POLICY=active
OMP_PLACES=cores OMP_PROC_BIND=close`, `tools/bench_baseline_clean.csv`, idle host, 3 reps).

| metric | stock llama.cpp | inkcpu | ratio |
|---|---|---|---|
| decode, ctx 16, 12 threads | 5.45 +- 0.05 tok/s | 13.86 tok/s (reps 13.72 / 13.86 / 13.57) | 2.5x |
| decode, ctx 16, 16 threads | 4.71 +- 0.07 | 14.32 (14.12 / 14.32 / 14.32) | 3.0x |
| decode, ctx 1000, 12 / 16 threads | n/a | 13.24 / 14.04 | |
| prefill 256 tokens, 12 threads | 19.02 +- 0.14 | 67.9 (67.9 / 67.5 / 65.8) | 3.6x |
| prefill 256 tokens, 16 threads | 22.4 +- 1.3 (23.5 best) | 82.7 (82.7 / 81.7 / 82.2) | 3.5-3.7x |
| prefill 1024 tokens, 12 / 16 threads | n/a | 65.6 / 81.5 (chunk 256) | |
| resident memory | 17.6 GB mmap | 17.0 GB arena (+0.15 prefill scratch, +0.4 drafter); peak 32-34 GB during load | |

Targets: decode >= 11.5 (met: 13.9-14.3; stretch 17 NOT met), prefill >= 70 (met at 16 threads: 82.7; 12 threads 67.9), parity (met: argmax 6/6 oracle prompts, greedy 64/64 on 2 prompts
and identical to llama.cpp through the server, NLL within 0.002-0.007 nats on 256-token chunks and 0.004-0.005 on 2400-token real text, `docs/VALIDATION_LONGCTX.md`), resident < 25 GB (met).
Per token the engine moves 3.86 GB at ~53 GB/s average (16 threads) against a 75-85 GB/s streaming ceiling; the remaining gap is the NVFP4 expert kernel (44-51 GB/s, ALU-bound without VNNI:
`ENGINE_DESIGN.md` section 10) and ~9% fixed overhead (serial mixes, controllers, barriers).

### Speculative decoding (DSpark drafter), 12 threads, greedy, 120 new tokens, 8 prompts (`bench_spec`)
| mode | pooled tok/s | pooled L (tokens / verify step) | worst prompt | best prompt |
|---|---|---|---|---|
| plain | 13.90 | - | | |
| fixed T=4 | 10.80 | 2.39 | prose 7.22 (L 1.50) | code_completion 14.40 (L 3.08, +5%) |
| fixed T=8 | 8.16 | 2.86 | prose 4.70 | code_completion 13.16 |
| auto (T in {1,4,8}) | 13.72 | 2.14 on the probes | chat_email 13.48 (-4%) | tool_call 13.77 (0%) |
All 32 speculative cells produced exactly the plain tokens (0 mismatches). Step costs: plain 71-74 ms, draft 21-23 ms (n=3) / 29-30 ms (n=7), verify 180-196 ms (T=4) / 277-302 ms (T=8).
The verify step costs 2.6x (T=4) / 4.0x (T=8) a plain step against the bytes model's 2.05x / 2.75x (`docs/DSPARK.md` 8): the batched expert GEMMs over 4-8 rows are ALU-bound on this
AVX-512-without-VNNI core, not bandwidth-bound. Break-even is therefore L ~2.9 (T=4) / ~4.4 (T=8), above the measured acceptance of everything except code completion.
Conclusion: on this CPU, speculation is a wash; the adaptive policy's value is "never more than ~4% slower than plain, +5% on code". It would pay on a CPU with VNNI/AMX (verify closer to
bandwidth-bound) or with a cheaper verify path; both are out of scope here. Default server mode stays `auto`.

### What was built and verified (per-agent logs in `docs/status/`)
Reference forward + oracles (A, E), AVX-512 GEMV/GEMM/misc kernels bit-identical to their scalar references (B1, B2, G), decode executor + `libink.so` (C1), batched prefill == decode bit for bit
up to 767 keys (P), DSpark drafter with acceptance equal to llama.cpp's within 1 SE (R, S1), verify + exact rollback + rejection sampler + adaptive policy + `ink_spec_*` C ABI, greedy speculation
identical to plain decode token for token and state for state (S2), OpenAI-compatible server on the real engine: 64 mock tests, 12 real-engine tests, a real coding-agent task in 207-291 s
(llama-server: 643 s, 9 steps; the engine run needed 5 steps) (D, V; `docs/SERVING.md`).

### Open items
1. NVFP4 expert kernel: 4-row scale decode (recipe in `ENGINE_DESIGN.md` 10) for an estimated 3-5% of decode.
2. Thread pool idle polling burns ~30% of a core (20 us nanosleep with 50 us slack); mitigated by the server with PR_SET_TIMERSLACK (2%), proper fix = futex park after ~2 ms idle.
3. Peak RSS at load is 2x steady (fixed per piece in the loader by S2; re-measure).
4. Sampled `auto` mode is not reproducible per seed across requests (policy state persists); fixed widths are. Fast-attention verify above 767 keys is rounding-level different from decode
   (`INK_SPEC_EXACT=1` for bit-exactness, cost unmeasured).
5. Contexts beyond ~2.7k tokens and tau > 1 (positions >= 8191) are untested against llama.cpp.

## 2026-11-22 GGUF layouts: `inkling2_loop` and `inkling2` (S4)
Unsloth re-uploaded `Inkling-2-Mini-UD-Q4_K_XL.gguf` in llama.cpp's canonical `inkling2` layout. The weights are byte-identical to the
interim `inkling2_loop` file the engine was validated on (588/588 common tensors + 144/144 renamed controller tensors + the message queries,
checked with gguf-py); only names and metadata differ. The engine now loads both (`ModelConfig::from_gguf`, `load_weights` in
`src/core/model.cpp`); nothing downstream (engine, prefill, RefModel, drafter) sees a difference. Mapping rules:

| | `inkling2_loop` (interim, 744 tensors) | `inkling2` (canonical, 734 tensors) |
|---|---|---|
| `block_count` | 24 = physical layers | 48 = visits; physical = 48 - (num_loops-1)*(core_end-core_start) = 24 |
| loop KVs | `loop.start` 6, `loop.end` 18, `loop.count` 3, `loop.varying` false, `loop.gate_rank` 64 | `loop.core_start` 6, `loop.core_end` 18, `num_loops` 3, `loop.gate_rank` 64 (no `varying`) |
| per-layer arrays (`head_count`, `head_count_kv`, `sliding_window_pattern`) | length 24, per physical layer | length 48, per visit in llama.cpp cache-slot order: physical layers 0..23 first, then the extra core passes (slot 24+(pass-1)*12+(phys-6)); the engine keeps entries 0..23 and checks that the repeats agree |
| controllers (core layers 6..17) | `blk.L.loop_{q,k,v,ffn}_ctrl_{down,gain,forget}` | `blk.L.loop_{attn,k,v,mlp}_{down,gain,forget}` (the loader tries the interim name first) |
| message queries | `blk.L.loop_msg_query.weight` [2304, 4], rows Q, K, V, FFN | global `loop_res_query.weight` [2304, 24] and `loop_kv_res_query.weight` [2304, 24]; for layer L, i = L-6: Q = res[2i], FFN = res[2i+1], K = kv_res[2i], V = kv_res[2i+1]. `load_weights` assembles the per-layer [2304, 4] block (Q, K, V, FFN) into memory owned by `Weights::owned_f32` |
| everything else | identical names, shapes and bytes | |

Gates (`docs/status/S4.md`): `test_engine crosscheck <other>` loads the two files one after the other and requires bit-identical logits on the
6 oracle prompts (decode) and a 300-token prefill; `test_engine oracle` and `test_prefill quick` on the canonical file reproduce C1/P.
The local llama.cpp checkouts (2a28df0f9 and llama-unsloth 0a9105d) do NOT load the canonical file: their `inkling2` arch is the
non-looped dense model (`check_tensor_dims: tensor 'blk.24.attn_q.weight' not found`); `tools/oracle_*` therefore still need the interim file.

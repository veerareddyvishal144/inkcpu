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

### Verification of the canonical `inkling2` file (orchestrator, 2026-11-22 21:10-21:45 UTC, 3 SMT threads, production untouched)
Freshly downloaded `/data/inkling-mini/models_fresh/Inkling-2-Mini-UD-Q4_K_XL.gguf` (17,629,759,456 B, `tools/get_model.sh`) vs the validated interim file:
`test_engine crosscheck`: 6 oracle prompts + 300-token prefill, max |diff| 0, 0 differing logits (bit-identical). `test_engine oracle` on the new file: C1's table digit for digit.
`test_prefill quick`: prefill == decode bit-identical, vs llama.cpp cosine 0.99983, argmax equal. `tests/engine/test_capi.py` PASS. `server/tests/test_ink_real.py` on the new file: 15/15
(a-h, speculation i-l, rewind m-o; b re-run after restoring effort-0 semantics). Production now serves the canonical file (`INK_MODEL_DIR=/data/inkling-mini/models_fresh`), identical
greedy output to the interim file. llama.cpp (2a28df0f9, PR 32218) cannot load the canonical file: oracle tooling keeps the interim copy in `models/`.

## 2026-11-23 Optimisation wave 1 (orchestrator + agents K1, K2, K3, E1, T, R; contract `docs/OPT_WAVE1.md`, per-agent logs `docs/status/{K1,K2,K3,E1,T}.md`, research `docs/RESEARCH_WAVE1.md`)

Question: how much more speed is in the engine on this host without changing what it computes? Everything below is bit-identical to the pre-wave build
(same logits at ctx 16 / 1000 / 4000 / 8000, prefill == decode, verify == decode) unless marked otherwise; parity with llama.cpp is therefore unchanged.

### What changed (all default ON unless noted)
1. **NVFP4 sub-block scales re-encoded at load** (`include/ink/nvfp4_repack.h`, K1): code = (e+11)<<3 | man3, decode = bitcast(code<<20) * 2^116, exact for all 256 UE4M3
   bytes (self-check), applied in place to the arena copy of the expert tensors of layers 0-19 while copying (+0.3 s load). 16 scales now decode in 3 vector uops instead of
   16 table loads. `INK_NVFP4R=0` restores the GGUF-encoded kernels.
2. **Decode expert GEMV `gemv_nvfp4r`** (K1): 8-row groups, grouped scale gather, per-call activation precompute; 14.8 instructions per 36-byte row super-block instead of 26.8,
   2.4 loads instead of 9. Phase G (gate_up) 22.3 -> 18.0 ms/token (55 -> 68 GB/s) in K1's same-window run; phase H (down) unchanged at ~52 GB/s: bound by its 16-row call
   structure (144 tiles per visit), not by the kernel (open item).
3. **Prefill / verify GEMM** (K3): `gemm_moe_nvfp4r` unpack with vector scale decode (unpack 1.2-1.6x faster), batched bit-identical epilogue for all Q8_0/NVFP4 GEMMs (49 uops
   instead of ~160 per tile), batched tail tiles. MoE layer at 16 threads on the real `code.rtrace` routing: +9.2 % (64 tokens), +4.8 % (256), +3.5 % (1024).
4. **Small-n expert matmuls `gemv_*_multix`** (K2): one weight stream against up to 8 activation vectors (shared nibble/scale decode, per-token arithmetic identical to the GEMV),
   used by prefill/verify phases G/H for experts with <= 8 routed pairs (`INK_PF_MULTIX`, 0 = GEMM always), by the verify lm_head and by the drafter's lm_head. Verify step
   T=4: 2.58x -> 2.15x a plain step, T=8: 4.11x -> 3.43x (near-idle, 12 threads); speculation policy priors re-measured (`include/ink/spec.h` verify_ratio). Short prefill
   16 tokens +18 %, 64 tokens +9 %.
5. **Decode executor** (E1): KV cache mapped 2 MiB-aligned with MADV_HUGEPAGE (`INK_KV_HUGE=0` off), K/V software prefetch across block boundaries, finish-by-last-split
   (2 barriers instead of 3 per split attention visit), controllers' gain+forget fused with ctrl_apply (2 barriers instead of 3), attention group-input prefetch. Attention phase
   -11..-19 % at ctx >= 1000 (ctx 8000: 9.95 -> 8.1 ms/token, KV read 43 -> 53 GB/s); barriers per token 576 -> 528. Measured and left OFF: finer key split above 767 keys
   (`INK_ATTN_GRANULE`, no gain), idle-thread weight prefetch during serial segments (`INK_PF_ROUNDS`, -5 %: software prefetches drain through the L1 fill buffers).
6. **Thread pool parks idle workers on a futex** (T; `INK_POOL_PARK=0` = old 20 us polling): idle 16-thread pool 0.12-0.16 % of one core instead of 86-120 % (50 us slack) or
   7 % (production's 5 ms slack); hot path unchanged within noise; first dispatch after an idle gap median 183 us vs 81 us. The old polling pool of the idle PRODUCTION server
   preempted every 16-thread benchmark ~3000 times/s (2-3 ms/token of barrier stalls in every measurement taken while it ran).
7. **Server** (T): per-token overhead measured at 0.21-0.36 ms (greedy) / 1.2-1.5 ms (sampled, numpy) = 0.3-2 % of a token; start-up hygiene (affinity before numpy import,
   OPENBLAS/OMP threads = 1); optional C sampler (`--sampler c`, slower than numpy here, off).

### Findings that closed doors
* SMT (T, idle host, lock): decode 14.05 / 13.88 / 14.28 / 10.95 tok/s at 16 / 20 / 24 / 32 threads (ctx 16); prefill 256 tokens 82.3 / 72.3 / 71.4 / 74.4. The GEMMs saturate the FMA
  ports with one thread per core, the spin barriers convoy with more threads than free vCPUs. One thread per physical core stays.
* R1 of the research report (integer-folded sub-block scales): no uop advantage once the scales are gathered per 4-row group; not implemented. ik_llama.cpp / llama.cpp repacked
  4-bit paths all require VNNI; no published no-VNNI 4-bit kernel beats the 5-uop MAC chain (`docs/RESEARCH_WAVE1.md`).
* Numerics-deviating prototype left OFF: K3's 4-uop GEMM inner loop (`gemm_moe_nvfp4r_x4`, +7-11 % on the MoE GEMMs, ESTIMATE ~+6-7 % prefill) flips 1 of 55296 int8 outputs at
  re-quantisation and would end prefill == decode bit-identity. Decision deferred to the owner.

### Gates on the integrated build (`build-final`, default flags)
Unit: test_gemv 3981 cases, test_gemv_multix 2813, test_gemm 2454, test_threadpool, test_quantize, test_kernels_misc, test_policy, test_sampler, test_gguf: all pass; server mock suite 73/73.
Full model (under the lock, `build-final/gates/00_summary.txt`): `test_engine oracle` reproduces the section-8 table of ENGINE_DESIGN.md digit for digit (incl. the m2_code_c
noise-floor label), `greedy` PASS, `determinism` bit-identical over 16/16/12/8/8/5 threads, `slots` + both ring wraps PASS (cosine 0.9993749 / 0.9995351 = C1's values),
`test_prefill quick` prefill == decode bit for bit (logits, conv histories, all KV) in both attention modes, `test_verify quick` PASS (T=2/4/8, exact and fast attention),
`test_rewind quick` ALL PASS, C ABI smoke test PASS, decode logits of the 6 oracle prompts + 32 greedy steps bit-identical between INK_NVFP4R=0 and 1, `test_drafter model`
265/265. Server: 73/73 mock tests, 15/15 real-engine tests (greedy via HTTP == raw libink == llama.cpp, speculation, stop tokens, tool calls, rewind, Caret pattern).
Production restarted on the integrated build at 20:46 UTC (idle CPU 0 % instead of 12 %).

### Final idle-host benchmark (`tools/final_bench_wave1.sh build-orch build-final build-final/final_bench`, 20:47-21:3x UTC, host idle 98 % at start, production idle on the new pool)
Old = the pre-wave build (build-orch, commit 95f2cab), new = build-final with all wave-1 defaults; old and new alternated in the same lock window (two runs each, best rep and all reps).

| metric (16 threads unless noted) | old | new | change |
|---|---|---|---|
| decode ctx 16 (64 tokens x 3 reps, runs A / B) | 13.99 (13.39 13.99 12.85) / 14.34 (14.34 14.12 13.83) tok/s | 14.74 (14.48 14.74 14.32) / 15.76 (15.76 15.52 14.89) | +5..+10 % |
| decode ctx 1000 | 13.40 (13.08 12.70 13.40) / 13.18 (13.18 12.91 12.86) | 14.30 (14.04 14.11 14.30) / 14.86 (14.36 14.86 14.18) | +7..+13 % |
| decode ctx 16, 12 threads | 13.86 (2026-11-22 baseline) | 15.33 (15.18 15.33 15.18) | +11 % |
| decode ctx 8000 (context built by prefill) | n/a | 13.75 (13.75 13.62); attention 8.0 ms/token = 54 GB/s of KV | |
| decode ctx 16000 | n/a | 12.99 (12.72 12.99); attention 11.3 ms/token = 61 GB/s | |
| prefill 256 / 512 / 1024 tokens (chunk 256) | 80.6 / 82.8 / 80.1 tok/s | 86.8 / 83.5 / 82.3 | +8 / +1 / +3 % |
| prefill 1024 tokens, chunk 512 (server default) | n/a | 86.6 | |
SPEC_ROWS_PLACEHOLDER
Run-to-run spread of identical configurations is ~5 % on this VM (the two old runs differ by 2.5 %, the two new runs by 7 %), so the decode gain is "+6..+10 %" rather than a single number;
the per-phase profiles (INK_PROFILE=1 in the result files) attribute it to gate_up (22 -> 18 ms), attention at long context, controllers and fewer barriers. Prefill's gain is the
K3 unpack/epilogue plus K2's multix for small experts (+8 % at 256 tokens, ~+3 % at 1024). What did NOT change: bytes per token (3.87 GB) and therefore the ~20 tok/s exact floor.

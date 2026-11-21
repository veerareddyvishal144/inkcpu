# PREFILL_DESIGN: batched prompt processing (agent P)

Files: `include/ink/engine_prefill.h` (API + tunables), `src/engine/engine_prefill.cpp` (executor, ~800 lines), `tests/engine/test_prefill.cpp` (gates), `bench/bench_prefill.cpp`
(throughput + per-phase profile). Build: `export PATH=/usr/bin:/bin; cmake -S . -B build-p -G Ninja && ninja -C build-p test_prefill bench_prefill`
(the targets exist only when `ink_kernels_gemm` is part of the build: block appended to `src/engine/CMakeLists.txt`).

```cpp
int engine_prefill(Engine& e, int slot, const int32_t* tokens, int n, float* logits_last);   // appends n tokens, logits of the last one (or nullptr)
int engine_prefill_all(Engine& e, int slot, const int32_t* tokens, int n, float* logits_all); // logits of every token (perplexity tooling), [n][n_vocab]
PrefillTune& prefill_tune();   // chunk (default 256, max 1024), item sizes, attention kernel switch; env INK_PF_*
```

## 1. Structure

One `ThreadPool::run()` per chunk of up to `chunk` tokens (default 256, up to 1024; longer prompts loop over chunks, state carries over in the slot). Inside, every thread executes the
same straight-line code for the 48 visits of `visit_plan`; phases are separated by `pool.barrier()`; dynamic scheduling uses one atomic counter per phase (numbered identically on
every thread because control flow is data independent, zeroed after every run), exactly like the decode executor. The scratch (all activations of a chunk, packed GEMM activations, MoE
buffers: ~150 MB at 256 tokens, ~600 MB at 1024) is allocated once per engine, lazily, in one hugepage arena (`get_scratch`; process-global registry keyed by `Engine*`, see docs/requests.md).

Per visit (n = tokens in the chunk; "tok" = token-parallel, claims of 4 tokens; "item" = dynamic work item):

| phase | work | split | barrier |
|---|---|---|---|
| A | [loop message depth mix (2 bank states, 4 queries)] attention-input depth mix (nq 1 or 3) -> [controller gate input] or rmsnorm x norm weight + Q8_0 (`quantize_q8_0`, `Q8Mode::X86`) + bf16 rounding (sink / dyn taps) + `gemm_pack_act` | tok | 1 |
| ctrl (passes 1,2) | batched controllers: down (4 rows x token group) -> gain/forget (32 rows) -> `ctrl_apply` per token; then norm + quantise + pack | items / tok | 3 + 1 |
| B | q, gate, k, v, rel (`gemm_q8_0`, 32-row items over ALL tokens of the chunk) and sink / k-conv-proj / v-conv-proj (`gemm_bf16`, 8-row items); BF16 items first (slowest per row) | item | 1 |
| C1 | per KV group g (8 items): `sconv_seq` K and V over all tokens (dyn taps on global layers), then per token K clip, V norm, Q clip, R clip, `kv_append` of head g | item | 1 |
| C2 | per token range (8 tokens, late tokens first): `attn_prefill_group` for the 8 groups (or the decode kernels per token, `INK_PF_ATTN=decode`), Q8_0 quantise of the 3072 outputs, pack | item | 1 |
| WO | 16-row item: `gemm_q8_0` of wo over all tokens, `sconv_seq` (post-conv) on those channels, residual add (no residual at block starts) | item | 1 |
| F1 | per token: push inpSA at block starts, FFN depth mix, [controller input], rmsnorm x ffn_norm, Q8_0, pack in the expert format (NVFP4 pack on layers 0-19, Q8 pack on 20-23) | tok | 1 (+4 with controller) |
| F2 | router `gemm_bf16/f32` (4-row items, token groups of 24), `router_topk` per token | item / tok | 2 |
| route | `moe_build_route` (counting sort of the n x 8 (token, expert) pairs by expert, stable), `moe_work_items`, inverse pair map; serial on tid 0 | - | 1 |
| G | (expert, 32 gate rows + the matching 32 up rows) items, big experts first: two `gemm_moe_*` calls into a per-thread tile [pairs][64], then per pair: expert scale, SiTU, Q8_0 quantisation into one block of the down input | item | 1 |
| pack | `gemm_pack_act` of the per-pair down activations | tok (pairs) | 1 |
| H | (expert, 32 rows) down items: `gemm_moe_*` -> y_dn[pair][2304] | item | 1 |
| I | per 32-channel range: top-k-ordered combine `((y*down_scale)*w)` + sequential float adds (AVX-512, no FMA, the decode order), `sconv_seq` (ffn post-conv), residual add | item | 1 |

13 barriers per visit (21 with messages), ~830 per chunk. After the visits: final depth mix + norm + quantise (last token, or all), `lm_head` by 512 -> 64-row `gemv_q8_0` chunks
for the last token (`gemm_q8_0` for all tokens in `engine_prefill_all`), softcap. Prelude / pass bank copies are token-parallel memcpys. Only the final chunk computes logits.

Trick used in phase G: `gemm_moe_*` writes `y[pair*ldy + row]` with ABSOLUTE pair and row indices; the executor passes `tmp - pair0*ldy - row0` (byte arithmetic through `uintptr_t`) so that the 32 gate
rows and the 32 up rows of one expert land side by side in a small per-thread tile that stays in L1/L2, and the SiTU + quantisation epilogue consumes it right away (no [pairs][2304] round trip).
The down activations must be packed per pair, which needs complete 36-block rows, hence the separate pack phase.

## 2. Memory and state

* Scratch per engine, allocated on first use for `chunk` tokens (re-allocated if a larger chunk is requested): activations [cap][2304] x ~35, GEMM packs (Q8 and NVFP4 sized), `y_dn`
  [cap*8][2304] floats (19 MB at 256), MoE per-thread tiles. 150 MB at 256 tokens, 300 MB at 512, 600 MB at 1024.
* Slot state is the decode state: `SlotState::kv` (global visits: `n_blocks` is grown to cover the chunk before the run; SWA rings: 32 blocks = 2048 positions, `kSwaRingBlocks`, made by C1
  on request; a chunk appends all its K/V before attending, so it needs >= 1023 + chunk slots). If a slot only had a smaller ring the executor shrinks the chunk to what is safe
  (`chunk_limit`), correct but slow. Conv histories (SoA [3][C]) are advanced by `sconv_seq`; `S.pos` is advanced per chunk. Decode continues from the result (gate (b)).

## 3. Numerics: what is bit-identical to the decode path and what is not

* Dense/MoE matmuls: G's kernels are bit-identical to B1's GEMV per (row, token); activations are quantised with the same function (B2 `quantize_q8_0`, `Q8Mode::X86`) after the same
  rmsnorm; SiTU, scale, expert combine, convs, clips, depth mixes, router, controllers use the same kernels in the same order. Results do not depend on the chunk size, the thread count or the
  item schedule.
* Attention: `attn_prefill_group` (default) vs the decode path's `attn_decode_task/finish`. They agree to the last bit while the decode path uses ONE key split, i.e. up to 767 visible keys on
  global layers (SWA: always); from 768 visible keys the decode path splits the keys (nsplit >= 2, merged in a fixed order) and the prefill kernel does one pass, so the two differ by
  rounding (~1e-6 relative per call, B2's tests), which Q8_0 re-quantisation and the top-8 choice amplify (docs/MODEL.md section 9). `INK_PF_ATTN=decode` runs the decode kernels per token
  (same nsplit rule) and is then bit-identical to decode at any length (validation mode, slower: ~5% at 1000 keys).

## 4. Tunables (env INK_PF_*, or `prefill_tune()` before the call)

`CHUNK` 256 (tokens), `PROJ` 32 / `WO` 16 / `GU` 32 / `DN` 32 (row chunks), `TOK` 4 (tokens per claim), `ATTN_TQ` 8, `ATTN=decode`, `GROUP` (G's tokens per weight pass), `PROFILE=1` (per-phase table incl. per-thread
CPU time). `INK_CPUS=3,5,...` in the test harness pins the threads elsewhere than another job's default CPUs.

## 5. Validation (test_prefill; `nice -n 10`, host load 17-30, 12 threads; reference = Engine::decode token by token)

Reference = `Engine::decode` token by token on slot A, prefill on slot B of the same engine. "fast" = default attention kernel (`attn_prefill_group`), "decode-kernel" = `INK_PF_ATTN=decode`.
Raw logs: `build-p/logs/*.log`, `build-p2` is the final build.

| gate | result |
|---|---|
| (a) 6 oracle prompts (13-52 tokens) + a 300-token wikitext chunk | prefill logits == decode logits **bit for bit** (max abs diff 0, cosine 1.0, same argmax and top-10) in BOTH attention modes; vs llama.cpp the numbers are those of the decode path (cosine 0.99166-0.99998, argmax equal 6/6, top-10 overlap 9-10). |
| (b) prefill m tokens then decode 24 | m = 43 (reference all-decode): 24/24 steps bit-identical in both modes; conv histories and all 48 x (K,V) fp16 caches bit-identical after the prefill and after the 24 decode steps. m = 700 (reference: decode-kernel prefill, itself bit-identical to decode): same, bit-identical. m = 1000: the fast kernel differs from the decode kernel by rounding (see section 3): 24 steps argmax equal, worst cosine 0.998 (informational; the independent check is against llama.cpp, below). |
| (c) chunk sizes | 300 tokens, chunk 1, 2, 7, 64, 256: final logits **bit-identical** to the all-decode run (cosine 1.0). Ring wrap: 2200 tokens (the 2048-slot SWA ring wraps inside the third chunk of 1000): chunk 1000 == chunk 256 bit for bit (logits, conv histories, all KV), and prefill 2192 + 8 decode steps == prefill 2200 bit for bit (decode-kernel mode). 1305 / 1500 tokens: decode-kernel mode bit-identical for chunk 64/256/1000; fast mode cosine 0.9997 / 0.995 against it, and against llama.cpp the fast kernel is as close as the decode kernels: 0.99931 vs 0.99937 (1305 tokens), 0.99550 vs 0.99243 (1500 tokens, repeated text), argmax equal everywhere. 2200 tokens (same text repeated 1.7x, past the 2048-slot ring wrap): decode-kernel mode (== the decode path) cosine 0.977 and fast mode 0.965 against llama.cpp (relL2 0.24 / 0.37, top-10 overlap 7 / 8, argmax equal): both implementations are equally far from llama.cpp, so this is a property of the engine's long-context numerics (flip amplification on a highly repetitive prompt, or a long-context semantic difference), not of the prefill executor; no real-text reference beyond 1305 tokens exists to tell which (open item for C1/B2). |
| (d) NLL, 2 x 256 wikitext tokens, `engine_prefill_all` vs llama.cpp `logits_all` | 2.11948 vs 2.11793 (|delta| 0.00155) and 1.70466 vs 1.71145 (|delta| 0.00680); limit 0.01. Argmax mismatches 5/256 and 3/256 (same as the decode path: prefill == decode bitwise). |
| (e) determinism | 300-token prefill bit-identical across repeated runs and thread counts 16 / 12 / 8 / 5. The schedule never changes arithmetic: dynamic items only decide which thread computes a (row chunk, token range) whose result is independent of the split. |
| fast vs decode attention | bit-identical up to 767 visible keys (`test_prefill split`: logits and the whole slot state, 767 tokens); first deviation exactly at 768 tokens (decode nsplit becomes 2): cosine 0.99996, 0.99992 at 900 tokens. |

Test commands: `build-p2/src/engine/test_prefill {quick|oracle|state|chunks|wrap|long|split|nll DIR..|determinism}` (INK_CPUS=3,5,... pins the threads elsewhere; the all-decode reference of
the 300-token prompt is cached in `build-p/refcache`, decode costs 2.4 s/token at load 24).

## 6. Performance

`tools/with_bench_lock.sh build-p2/src/engine/bench_prefill --threads 12,16 --prompts 16,64,256,1024 --reps 2` (best of 2, chunk 256, final code, two independent runs at 23:15 and 23:31 UTC; **host load
14-16 during run 1 and 8-14 during run 2, so indicative only**: other tenants share the SMT siblings; the host was at load 17-39 for most of the session). Stock llama.cpp on this host: 19.0 tok/s
(12 threads), 23.5 (16 threads) on a 256-token prompt.

| threads | 16 tokens | 64 | 256 | 1024 (chunk 256) | 1024, chunk 512 | 1024, chunk 1024 |
|---|---|---|---|---|---|---|
| 12 | 32.2 / 36.7 | 50.5 / 55.4 | **66.7 / 66.7** | 66.6 / 65.3 | 69.1 | 69.6 |
| 16 | 44.3 / 43.7 | 68.4 / 67.5 | **83.6 / 78.9** | 78.5 / 75.4 | 84.1 | 87.2 |

(run 1 / run 2; chunk 512 / 1024 columns: one run each at load 13-14.) At 256 tokens that is 3.5x (12 threads) and 3.4-3.6x (16 threads) stock llama.cpp. Target 70 tok/s on a 256-token prompt: met at 16 threads
(79-84), 5% short at 12 threads (66.7, both runs). Larger chunks help long prompts (more pairs per expert: the MoE kernel runs 6-8% faster per thread at 512 than at 256 tokens): the default stays 256
(memory 150 MB, spec), `INK_PF_CHUNK=512` or 1024 is recommended for prompts >= 512 tokens (+5-15%).

### Per-phase profile, 16 threads, 256 tokens, load 13 (`INK_PF_PROFILE=1`; wall = thread 0 between barriers, busy = thread CPU time in the phase body, averaged over threads)

| phase | ms/token | % | GMAC/token | GMAC/s (wall) | MB/token | GB/s | busy GMAC/s per thread |
|---|---|---|---|---|---|---|---|
| MoE gate_up + SiTU + Q8_0 (NVFP4 L0-19, Q8_0 L20-23) | 5.09 | 42.7 | 2.038 | 400 | 56.0 | 11.0 | 25.2 |
| MoE down GEMM | 2.80 | 23.4 | 1.019 | 365 | 28.0 | 10.0 | 23.0 |
| proj q,k,v,gate,rel,sink (+ dyn taps) | 2.02 | 16.9 | 0.952 | 472 | 4.0 | 2.0 | 30.9 |
| wo GEMM + post-conv + residual | 0.73 | 6.1 | 0.340 | 465 | 1.4 | 1.9 | 31.0 |
| loop controllers (24 message visits, BF16 K=4608 and K=64) | 0.56 | 4.7 | 0.057 | 100 | 0.4 | 0.8 | 7.2 |
| attention + Q8 quantise | 0.16 | 1.3 | - | - | - | - | - |
| attn-in mix + norm + quant / ffn mix + norm | 0.13 / 0.06 | 1.1 / 0.5 | - | - | - | - | - |
| router GEMM + top-8 / conv+clip+append / combine / route / pack | 0.10 / 0.13 / 0.10 / 0.02 / 0.02 | 0.8 / 1.1 / 0.8 / 0.1 / 0.2 | 0.014 | 147 | 0.1 | 1.2 | 11.0 |
| lm_head (last token) | 0.02 | 0.2 | 0.002 | 87 | 1.9 | 92.6 | 5.5 |
| **total** | **11.93 = 83.9 tok/s** | | 4.43 | 372 | | | busy bound 87.1 tok/s |

### Limiting factors (evidence)
1. **The GEMM kernels, 88% of the time (MoE alone 66%)**. They are bound by the vector ALU ports (5 uops per 64 MACs, KERNELS_GEMM.md): at load 13 a busy thread sustains 31 GMAC/s (dense Q8_0) and
   23-25 GMAC/s (NVFP4 / Q8_0 MoE with 16 pairs per expert) against 35-39 / 29-31 in G's isolated single-thread runs and a ~54-61 GMAC/s idle ceiling. Executor overhead on top of that is small: 4% (busy 11.48 vs wall 11.93 ms/token) is
   barrier waits and stragglers, non-GEMM phases together < 7%, controllers 4.7%.
2. **Host contention halves the kernels**: the same code measured 31 tok/s at load 26 (12 threads: 17 / 14 / 12.5 GMAC/s per busy thread) and 84 tok/s at load 13 (16 threads). `bench_peak` (G's pure-ALU 5-uop loop,
   no memory) gave 25 GMAC/s per thread at load 30 (idle ~54-61): the hyperthread siblings are busy with other jobs, so every thread runs at roughly half speed. ESTIMATE (not measured): on an idle host about
   75-80 tok/s at 12 threads and 95-100 at 16.
3. MoE efficiency grows with pairs per expert (16 at 256 tokens, 64 at 1024 with chunk 1024): 16-token prompts run at 7-8 GMAC/s per thread (unpack not amortised; ~650 MB of expert weights are streamed per prompt token over the 48 visits),
   hence 44 tok/s at 16 tokens vs 79-84 at 256.
4. Controllers: gain/forget are K = 64 BF16 GEMMs whose cost is the horizontal sum per output (7 GMAC/s per thread). A transposed bit-exact reduction would save ~2-3% overall (not done).
5. Barriers: ~830 per chunk (~1 us each when nobody is preempted). Under load the cost is stragglers: busy vs wall gap 4% at load 13, 14% at load 26.

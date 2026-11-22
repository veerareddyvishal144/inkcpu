# DSpark block drafter (`ink::Drafter`, module `src/drafter`, agent S1)

Owner: agent S1. Spec: docs/DSPARK.md sections 4-6 (forward), 8 (cost model), 9.2 (R's acceptance table). Contract: docs/SPEC_CONTRACT.md. Interface: include/ink/drafter.h.
Status log: docs/status/S1.md. All timings in this file were taken on the shared host next to other agents' jobs (load average printed with every run): they are indicative.

## 1. What it is
The drafter is the 3-layer, 8-slot DFlash/DSpark block model shipped as `dspark-Inkling-2-Mini-Q8_0.gguf` (402 MB). Per cycle it takes the target's committed context
(4 taps x 2304 per position, DSPARK.md 4.1), the anchor token and produces n <= 7 draft tokens with one block pass (8 slots: anchor + 7 mask slots) followed by a sequential
Markov/RNN chain over the n drafted positions (DSPARK.md 4.3). It borrows the target's `lm_head` (Q8_0), `token_embd` + `token_embd_norm`, and the target's `ThreadPool`.

## 2. Design
One `ThreadPool::run` per `draft()` (the calling thread is tid 0 and is pinned to the first CPU of the engine's list on first use); phases are separated by `pool.barrier()`;
inside a phase the work items are claimed with a relaxed `fetch_add` on a per-phase counter (`Sched`, identical numbering on every thread; nothing depends on which thread does what).
Nothing is allocated on the hot path: all scratch (`Scr`) and the weight arena (hugepages, `alloc_weight_buffer`) are allocated in `create`; per-state buffers in `state_new`.

| # | phase (profile name) | work | parallel split |
|---|---|---|---|
| 1 | ctx K/V flush | for the pending committed rows (lazy `commit`, see 5): mix the 4 taps with `softmax(fuse_weight)` rows, RMS (dflare_norm), Q8_0 pack, GEMM with `Wk_ctx`/`Wv_ctx` (per layer; the SWA layers 0/2 skip rows that no future query can see), `k_ctx_norm` + RoPE, `kv_append` into the B2 f16 block caches (layer 1 global, layers 0/2 a 16-block ring = the 1024 positions before the block) | rows x layers; 32-row weight strips; (layer, kv group) |
| 2 | embed + rope tables | X[0] = RMS(E[anchor]) * embed_norm gain (target's Q8_0 table); X[1..7] = mask_embd; RoPE tables for positions p..p+7 | tid 0 / slots |
| 3 | norm + QKV gemm + q/k prep | per layer: RMS + Q8_0 quantise + `gemm_pack_act` of the 8 slots; `gemm_q8_0` over wq/wk/wv in 32-row strips (160 items); q_norm/k_norm + RoPE + 1/sqrt(128) | slots; strips; (slot, group) |
| 4 | attention (items + merge) | own AVX-512 kernel (`attn.cpp`): 24 queries of one kv group against the transposed f16 ctx keys (online softmax per 64-key block, key splits for long contexts) + the 8 block keys in fp32; deterministic merge; Q8_0 quantise of the attention output | (group, split); slots |
| 5 | o-proj + residual | `gemm_q8_0` wo in 32-row strips, add into X | 72 strips |
| 6 | FFN | RMS + pack; gate/up GEMM in 64-row strips with fused silu*up and Q8_0 quantisation of the intermediate (two K halves of 4608 because the GEMM kernel takes K <= 8192); pack; down GEMM (two halves summed) + residual | 144 strips; 72 strips |
| 7 | final norm | hid[j] = RMS(X[j]; output_norm) for j = 1..n, quantise + pack (only the n drafted slots; the last layer also computes q/o/FFN only for slots 1..n) | slots |
| 8 | base lm_head gemm + joint parts | ONE batched `gemm_q8_0` over the n activation rows for all 200058 lm_head rows (64-row items, output stride padded to 16) + the chain-independent `joint_proj` columns ([hid ; f_rnn], 4608 of 5120) for all n rows (`gemm_bf16`) | 3126 + 48 items |
| 9 | chain: joint matvec | per step j: `gemv_bf16` of the 512 sequential columns ([s ; markov_w1[prev]]) of `joint_proj` (768 rows); then every thread redundantly applies the GRU update and lat = tanh(.) (768 floats: cheaper than a broadcast) | 48 items |
| 10 | chain: M matvec + softcap + argmax | per step j: `gemv_q8_0` (or `gemv_bf16`) over the folded `M` in 512-row chunks, + base[j] + m0, softcap 30, chunk argmax (greedy) or chunk max / sum of exp (sampling); the final reduction over the 391 chunk results is done redundantly by every thread (no extra barrier) | 391 chunks |
| 11 | sampling post pass | normalise the dense q rows (only when `rq.sample`) | (step, chunk) |

Barriers per draft: 3 per flush round + 1 + 8 per layer (24) + 1 + 1 + 2 per chain step (+1 when sampling): 32 + 2n (+3 per 64 pending rows).

The fold (`fold.cpp`): `M = lm_head . hidden_out` (200058 x 256) and `m0 = lm_head . hidden_out.bias` are computed once at `create` in fp32 (rows split over the pool, ~4.5 s on 4 threads) and cached on disk
next to the drafter GGUF as `dspark-fold-{q8,bf16}-<key>.bin` (header `INKFOLD1`, key = content hash of lm_head, hidden_out weight and bias, n_vocab and the format; loads in < 0.1 s).
`DrafterOptions::cache_dir` = "" (next to the GGUF, default) | "-" (never write/read a cache). Storage: Q8_0 (54.4 MB, default) or bf16 (102.4 MB) (`DrafterOptions::fold_q8`); decision in section 4.
Chain numerics: per step the logits are `softcap(base[j][v] + M[v].lat + m0[v])` with base from the Q8_0 lm_head GEMM (Q8_0-quantised hid, like llama.cpp) and `M.lat` from a Q8_0-quantised lat (fold_q8) or fp32 lat (bf16).

## 3. Memory
| what | bytes | note |
|---|---|---|
| drafter layer weights (Q8_0) | 278.2 MB | 3 x 92.75 MB (q 7.52, k/v/k_ctx/v_ctx 10.03, o 7.52, gate+up 45.12, down 22.56); copied from the GGUF into a hugepage arena |
| joint_proj (bf16) + norms + m0 | 8.7 MB | in the arena |
| folded M | 54.4 MB Q8_0 / 102.4 MB bf16 | in the arena; `weight_bytes()` reports the arena size: 342 MB (Q8_0) / 390 MB (bf16) |
| borrowed from the target | 0 | lm_head 489.7 MB, token_embd, markov_w1 (bf16, 102 MB, read 512 B per step) are the target's / the GGUF mapping |
| scratch (shared by all states) | ~14 MB | base logits 8 x 200064 floats (6.4 MB), internal q rows 7 x 200064 (5.6 MB, only used when sampling without q_out), FFN intermediates, packs, partials |
| per state: pending taps | 2.4 MB | 64 rows x 4 x 2304 floats |
| per state: SWA caches (layers 0, 2) | 2 x 4 MB | 16 blocks x (K 128 KiB + V 128 KiB), `MAP_NORESERVE`, touched as used |
| per state: global cache (layer 1) | (max_ctx/64 + 1) x 256 KiB | 16.6 MB at 4096, 131 MB at 32K; virtual, touched as used; `state_reset` gives the pages back (`MADV_DONTNEED`) |
`state_bytes(st)` reports the touched estimate for the current position.

## 4. Fold decision: Q8_0 vs bf16 for M
Data: `eval_acc --fold-compare --ref-every 6 --max-anchors 72` on code_completion, chat_answer, tool_call (252 chained positions, 4 threads, 2026-11-22 00:17) and `bench_drafter --bf16` (ctx 1024, 4 threads, load1 8).
A = M as Q8_0 (lat quantised to Q8_0 too), B = M as bf16 (fp32 lat), U = unfolded chain (two lm_head products per step with Q8_0-quantised activations, llama.cpp style), E = exact-activation reference
(fp32 `lm_head . (hid + corr)`, no activation quantisation).

| | top-1 = U | top-1 = E | max abs dlogit vs U | max abs dlogit vs E (mean of per-step max) | chain-M ms per chained position, 4 threads | M bytes |
|---|---|---|---|---|---|---|
| A: Q8_0 M (default) | 252/252 = 100.0 % | 98.41 % | 0.065 (mean 0.043) | 0.260 (0.159) | 1.94 ms (29 GB/s) | 54.4 MB |
| B: bf16 M | 252/252 = 100.0 % | 98.41 % | (test_drafter: 0.024-0.032 vs 0.034-0.048 for A) | 0.261 (0.157) | 3.21 ms (32 GB/s) | 102.4 MB |
| U: unfolded (llama.cpp) | - | 98.41 % | - | 0.272 (0.159) | 2 x lm_head = 980 MB per position | - |

Decision: **Q8_0 (default)**. It meets the gate (top-1 agreement with the unfolded reference >= 99.5 %: measured 100 %), its agreement with the exact reference is identical to bf16's and to the
unfolded chain's (the 1.6 % of positions where all three differ from E come from the Q8_0 quantisation of `hid` before the lm_head, which llama.cpp shares and which is the same for every M format),
and it saves 48 MB per chained position (1.3 ms per position on 4 threads, ~0.7 ms at 12 threads). `--bf16` / `DrafterOptions::fold_q8 = false` stays available and tested (`test_drafter model --bf16`: 265 checks pass).
Full-run check (8 prompts, every anchor, `--ref-every 6` not needed for the acceptance numbers): the Q8_0 drafter's greedy tokens are the ones the unfolded reference would draft on 100 % of the 252 checked positions.


## 5. Semantics for the caller (S2)
* `create(Engine&, gguf, [DrafterOptions], &err)`: runs on `Engine::pool` (one fold computation on the first run without a cache, ~4.5 s on 4 threads). Refuses GGUFs with a different softcap / rms eps / target layers / block size.
* States: `state_new()` returns -1 when `max_states` (default `Engine::opt.n_slots`) are in use. The context limit of a state is `Engine::opt.max_ctx` (`commit` returns -2 beyond it).
* `commit(st, taps, rows)` is LAZY: rows are copied into a 64-row pending buffer (`state_pending(st)`); `f_rnn` is taken from the LAST row of every commit immediately. When the pending buffer would overflow,
  the pending rows are flushed first (own pool run); commits of more than the free pending capacity (prompts) are processed straight from the caller's buffer in 256-row pool runs (the pointer must only stay valid during the call).
  `draft()` flushes the pending rows inside its single pool run, so the spec loop should commit the accepted rows and call `draft()` without `flush()`; `flush()` exists for benchmarks / tests. Observable results are identical to eager commits (tested bit-identical).
* `draft(st, rq, out, q_out)`: needs `state_pos(st) > 0` (returns -3 otherwise), `1 <= n <= 7` (-1), `0 <= anchor < n_vocab_padded` (-4). Greedy: ties -> lowest id. Truncation: the n=3 draft is the prefix of the n=7 draft (tested).
  Sampling: `rq.sample && temperature > 0` draws d_j from softmax(logits/T) (inverse CDF, deterministic in (seed, j)) and writes the dense q rows `[n][n_vocab]` (unpadded stride) to `q_out`; temperature 0 with `sample` = greedy with one-hot q rows.
  The committed state is never changed by `draft()` (the 8 block K/V are scratch outside the caches).
* Threads: every call runs on the target's pool from the caller's thread (tid 0); never call the drafter concurrently with an engine forward (same pool), never from another thread than the one that drives the pool.
  Respect `INK_CPUS` / `EngineOptions::cpus`; the drafter pins the calling thread to `cpus[0]` on first use (as the engine does).
* Profile: `profile()` / `profile_report(FILE*)` / `profile_reset()` (rdtsc on thread 0 between barriers, analytical MB and MACs per phase; `DrafterOptions::profile` on by default, negligible cost).

## 6. Tests and benchmarks (all honour `INK_CPUS`, default 4 threads)
```
PATH=/usr/bin:/bin:$PATH cmake -S . -B build-s1 -G Ninja -DCMAKE_BUILD_TYPE=Release && nice -n 19 ninja -j3 -C build-s1 test_drafter eval_acc bench_drafter
INK_CPUS=2,4,6,8 build-s1/src/drafter/test_drafter all [--threads 4] [--bf16] [--pbig 1100]     # unit (rope, attention vs double ref, fold kernel, hash, sampling) + model (scalar reference, semantics)
INK_CPUS=2,4,6,8 build-s1/src/drafter/eval_acc [--threads 4] [--prompts a,b] [--ref-every 8] [--fold-compare] [--bf16]   # acc@k and L on S2's tap dumps (tests/spec/taps/*.taps), fold checks
INK_CPUS=2,4,6,8 build-s1/src/drafter/bench_drafter --threads 4 --ctx 128,1024,4096 --n 1,3,7 --reps 20 [--bf16] [--sample] [--commit 3]
```
`test_drafter model` and `eval_acc` use the "light target" (tests/drafter/light_target.h: the target's lm_head copied into a hugepage arena + token_embd from the mapping, no 17 GB engine), so they can run next to other jobs (~1 GB).
The scalar reference (tests/drafter/ref_drafter.cpp) is written from DSPARK.md 4 with double accumulation and the UNFOLDED chain (two lm_head products per step, llama.cpp style).

## 7. Measurements
### 7.1 Phase table
`bench_drafter --threads 4 --ctx 128,1024,4096 --n 1,3,7 --reps 15`, INK_CPUS=2,4,6,8, host load1 7.6-8.0 (V's server with 5 threads on CPUs 22-30 and S2's tests running: INDICATIVE), 2026-11-22 00:16.
Light target (real Q8_0 lm_head), greedy, Q8_0 M, ctx 1024, median wall of 15 drafts; ms per phase = thread-0 rdtsc between barriers; MB = analytical bytes streamed (weights + KV + fold + logit rows); GB/s and GMAC/s over the phase time.

| phase | n=1 ms | n=3 ms | n=7 ms | MB/draft (n=3) | GB/s (n=3) | GMAC/s (n=3) | bound |
|---|---|---|---|---|---|---|---|
| embed + rope tables | 0.02 | 0.01 | 0.02 | 0 | | | |
| norm + QKV gemm + q/k prep (3 layers) | 3.03 | 3.11 | 3.27 | 37.6 | 12.1 | 79.6 | GEMM compute (nt = 8) |
| attention (items + merge), ctx 1024 | 1.24 | 1.33 | 1.49 | 12.6 | 9.5 | 90.7 | KV read + softmax |
| o-proj + residual | 1.67 | 1.77 | 1.95 | 22.6 | 12.7 | 75.8 | GEMM compute |
| FFN (norm, gate/up, silu, pack, down) | 14.83 | 15.63 | 17.25 | 203.0 | 13.0 | 77.4 | GEMM compute (8 rows; last layer n rows) |
| final norm | 0.01 | 0.01 | 0.01 | 0 | | | |
| base lm_head GEMM (one batched pass, n rows) + joint parts | 21.65 | 27.00 | 38.92 | 496.8 | 23.0 / 18.4 / 12.8 | 21.6 / 51.7 / 83.6 | bandwidth at n=1, GEMM compute at n >= 3 |
| chain: joint matvec (per step 0.8 MB, L2) | 0.04 | 0.13 | 0.29 | 2.4 | 18 | 9 | latency (48 items) |
| chain: M matvec + softcap + argmax | 1.99 | 5.81 | 13.41 | 168.0 (56 per step) | 28.9 | 26.4 | bandwidth (4-thread streaming ceiling ~29 GB/s) |
| **total (median wall)** | **44.5** | **54.9** | **76.7** | 943 (n=3), 1170 (n=7) | 17.2 | 59.6 | |

* The analytical bytes match the DSPARK.md 8 cost model with Q8_0 layers: Dr(3) = 278 + 9 + 490 + 3 x 54 = 940 MB, Dr(7) = 1158 MB.
* Context: total at n=3 is 54.7 / 54.9 / 56.4 ms for ctx 128 / 1024 / 4096 (attention 0.28 / 1.33 / 2.72 ms: layer 1 reads all ctx K/V, layers 0/2 the 1024-position window). Prompt commit 6.9-7.6 ms per 64 rows (~110 ms per 1024 positions) on 4 threads.
* Spec-loop pattern `commit(3 rows)` + `draft(n=3)`: 54.2-56.3 ms; the lazy flush inside the draft costs 0.9 ms (15 MB of ctx projections, 3 barriers).
* Sampling (`--sample`, T=1): 56.2 ms (n=3) / 80.1 ms (n=7); the dense q rows cost +1.3 / +3.4 ms (exp over 200k rows per step) and the post pass 0.1 / 0.26 ms.
* bf16 M (`--bf16`): 60.0 / 87.7 ms (n=3/7); chain M 9.64 / 22.38 ms = 3.2 ms per position at 32 GB/s (bandwidth; 102 MB per step).
* Predecessor's session (load 14, hyperthread siblings busy): 62 / 87 ms (n=3/7) with the same code: the 4-thread numbers move 15-20 % with the host load.

**What limits it.** The chain-M matvec runs at the 4-thread streaming ceiling (29 GB/s = 7.3 GB/s per thread, the same per-thread rate as the engine's lm_head phase at 12-16 threads). The GEMM phases
(FFN, QKV, o-proj, base lm_head at n >= 3) are compute-bound: 75-86 GMAC/s on 4 threads = 19-21 GMAC/s per thread, the rate G measured for the Q8_0 GEMM at nt <= 8 on this CPU (5 ALU uops per 64 MACs, no VNNI;
docs/KERNELS_GEMM.md 5-6). The base lm_head pass therefore costs 0.46 GMAC per drafted row: at n=7 it is 2x its bandwidth floor even at 12 threads. No allocation and no serial section on the hot path
(embed and the chain reductions are done redundantly by every thread); barriers 32 + 2n per draft (~5-10 us each at 12 threads).

**Extrapolation to 12 threads on an idle host (NOT measured: V's server occupied CPUs 22-30 during the whole session, load1 never below 7.6).** Assumptions: GEMM 20-22 GMAC/s per thread (240-260 aggregate,
G's nt <= 8 rate; an idle host may give 26-28 per thread), streaming 70-75 GB/s (ENGINE_DESIGN.md: engine lm_head 75 GB/s at 12 threads), attention and barriers 1 ms.

| n | QKV + o-proj (0.44 GMAC) | FFN (1.21 / 1.46 GMAC) | base lm_head (0.46 GMAC per row vs 6.6 ms bandwidth floor) | chain M (56 MB per step) | misc | **estimate** | budget |
|---|---|---|---|---|---|---|---|
| 3 | 1.8 ms | 4.8 ms | 6.6-7 ms (balanced) | 2.4 ms | 1 ms | **17-19 ms** | <= 25 ms |
| 7 | 1.8 ms | 5.8 ms | 12.5-13.5 ms (compute) | 5.6 ms | 1 ms | **27-30 ms** | <= 45 ms |

With the hyperthread siblings busy (other agents, as in every run of this session) the per-thread GEMM rate drops to ~15 GMAC/s (G: load 17) and the streaming to ~55 GB/s: 24-26 ms (n=3), 38-42 ms (n=7), still inside the budget.
The orchestrator's idle-host run: `tools/with_bench_lock.sh env INK_CPUS=2,4,6,8,10,12,14,16,18,20,22,24 build-s1/src/drafter/bench_drafter --threads 12 --ctx 1024 --n 3,7 --reps 20`.


### 7.2 Accuracy on S2's tap dumps (greedy, every anchor of the 160 generated tokens, n = 7, Q8_0 M)
`eval_acc --threads 4` on S2's dumps `tests/spec/taps/*.taps` (prompt + 160 tokens of OUR engine's greedy continuation, S2's `dump_taps --gen`), 2026-11-22 00:16, 4 threads, load1 7.3-8.
acc@k = fraction of anchors whose first k drafted tokens all equal the target's next k greedy tokens (every anchor of the path, 153 per prompt). L(n) = 1 + sum acc@1..n (per anchor).
**Ls(n)** = mean tokens per verify step of the loop simulated on the same path (after a step that accepts k, the next anchor is k+1 positions later: exactly what the real loop sees, since the drafter's output at
an anchor depends only on the committed prefix); this is R's statistic. R = llama.cpp PR build, DSPARK.md 9.2 (its own 80-token continuations, 13-46 steps per prompt, SE 0.08-0.15 pooled).

| prompt | acc@1 | acc@2 | acc@3 | acc@4 | acc@5 | acc@6 | acc@7 | L(3) | L(7) | **Ls(3)** | **Ls(7)** | R: L n=3 | R: L n=7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat_answer | 0.739 | 0.536 | 0.412 | 0.301 | 0.235 | 0.176 | 0.105 | 2.69 | 3.50 | 2.32 | 2.56 | 1.77 | 1.79 |
| code_completion | 0.895 | 0.745 | 0.667 | 0.549 | 0.464 | 0.392 | 0.294 | 3.31 | 5.01 | 3.16 | 4.43 | 3.67 | 5.92 |
| prose_continuation | 0.379 | 0.124 | 0.039 | 0.000 | 0.000 | 0.000 | 0.000 | 1.54 | 1.54 | 1.50 | 1.50 | 1.67 | 1.66 |
| reasoning | 0.784 | 0.601 | 0.458 | 0.346 | 0.275 | 0.170 | 0.118 | 2.84 | 3.75 | 2.67 | 3.18 | 2.85 | 3.48 |
| tool_call | 0.595 | 0.444 | 0.366 | 0.320 | 0.281 | 0.255 | 0.248 | 2.41 | 3.51 | 1.91 | 2.22 | 3.13 | 5.14 |
| long_context_qa | 0.739 | 0.536 | 0.399 | 0.281 | 0.176 | 0.105 | 0.059 | 2.67 | 3.29 | 2.52 | 2.94 | 2.72 | 3.17 |
| chat_email | 0.699 | 0.471 | 0.320 | 0.216 | 0.144 | 0.065 | 0.052 | 2.49 | 2.97 | 2.37 | 2.72 | 1.73 | 1.85 |
| chat_code | 0.902 | 0.739 | 0.647 | 0.562 | 0.471 | 0.399 | 0.314 | 3.29 | 5.03 | 3.14 | 4.25 | 3.12 | 4.41 |
| **POOLED (1224 anchors)** | 0.717 | 0.525 | 0.413 | 0.322 | 0.256 | 0.195 | 0.149 | 2.65 | 3.58 | **2.32** | **2.68** | **2.37 +- 0.08** | **2.77 +- 0.15** |

Pooled per-step L agrees with R within one standard error (2.32 vs 2.37, 2.68 vs 2.77). Per prompt the continuations differ (ours: 160 tokens of our engine; R: 80 tokens of llama.cpp; tool_call and
chat_answer diverge most), so per-prompt rows are not a like-for-like test; the pooled agreement and the 100 % top-1 agreement with the unfolded reference (section 4) are. The eval-loop draft time was 75-78 ms (n=7, 4 threads).


## 8. Open items
* 12-thread idle-host measurement (above) not taken in either S1 session: only the 4-thread numbers next to other agents exist.
* The base lm_head pass is bound by the Q8_0 GEMM's ALU rate for n >= 3 (G's kernel, not S1's code): at n=7 and 12 threads it is ~13 ms against a 6.6 ms bandwidth floor. A kernel-level change (G) is the only lever; a top-K restriction of the
  chain would not help (the base pass needs every row).
* n=1 could run the base pass with `gemv_q8_0` instead of the GEMM (29 vs 22 GB/s on 4 threads: -5 ms at 4 threads, ~-1 ms at 12); not done because the policy never drafts n=1 (T=1 is plain decode).
* Per-step L (`Ls`) in eval_acc simulates the loop on the dumped greedy path; R's numbers come from llama.cpp's own 80-token continuations: per-prompt differences are expected, the pooled range is what to compare.
* `commit()` of more than the free pending capacity while rows are pending runs two pool runs (flush, then the big commit); the spec loop never does this (<= 8 rows per cycle).
* Layer-1 caches are `MAP_NORESERVE` mappings sized to `Engine::opt.max_ctx` per state (virtual 131 MB at 32K, touched as used); `state_reset` returns the pages.


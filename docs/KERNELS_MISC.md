# kernels_misc (agent B2): vector math, norms, clips, short convs, attention, router, depth mix

Library `ink_kernels_misc` (`src/kernels_misc/`), API `include/ink/kernels_misc.h`, header-only lane functions
`include/ink/kernels_misc_simd.h` (always_inline `exp16/tanh16/sigmoid16/situ16/softcap16/rcp16`, bf16/f16 conversions) so the executor can
fuse them into its own phase loops. Tests `tests/kernels_misc/test_kernels_misc.cpp`, microbenchmarks `bench_kernels_misc.cpp`,
raw benchmark output `docs/status/B2_bench.txt`.
ISA: AVX-512 F/BW/DQ/VL + FMA + F16C only (objdump scan of the archive: no VNNI/BF16/VBMI/FP16 instructions).
Every kernel is single-threaded, allocation-free, deterministic (reduction order fixed by lengths, never by thread count) and exists as
`ink::name` (AVX-512) and, where it matters, `ink::ref::name` (scalar / double). Pointwise kernels accept any length incl. 0 and tails and `y == x`.

Build/run (the root cmake currently fails to configure because B1's `src/kernels_gemv/CMakeLists.txt` is mid-edit; this private top level builds only B2):
`cmake -S tests/kernels_misc/standalone -B build-b2s -G Ninja && ninja -C build-b2s && build-b2s/kernels_misc/test_kernels_misc && build-b2s/kernels_misc/bench_kernels_misc 16 [1]`
(the second bench argument 1 also runs a 1M-context global visit, allocating 4.3 GB). `export PATH=/usr/bin:/bin` first.

## Semantics taken from llama.cpp @2a28df0f9 (and a few that are odd)
1. **clip()** (Q, K, R): `x *= chan_scale` (stored back), per HEAD `norm = sqrtf(sum x^2) + 1e-6` (the 1e-6 is added AFTER the sqrt),
   `x *= clamp(threshold/norm, 0, 1)`. Q: 24x128, scale `attn_q_clip[24][128]`, 22.627. K: 8x128, no scale, 22.627. R: 24x16, scale
   `attn_rel_clip[24][16]`, 8.0. On global layers `tau` multiplies Q and R AFTER the clip (separate multiply), so their norms can exceed the threshold.
2. **V norm**: per KV head RMS norm over 128, eps 1e-6, no gamma.
3. **Short conv** (`convolve`): history = the previous 3 conv INPUTS (pre-conv values). `out[c] = sum_{i=0..3} xin[t-3+i][c]*w[c][i]` (ggml `ssm_conv`:
   tap 3 = current token), dynamic taps (global layers K/V) add `sum_lag xin[t-lag] * dyn[h][lag]` (lag 0 = current) to all 128 channels of head h,
   `y = (x + out) / sqrt((dyn ? 2 : 1)*4 + 1)`: **1/sqrt(5) static, 1/3 dynamic**. Applies to K, V, after `wo` and after the MoE (C = 1024, 1024, 2304, 2304).
   Dynamic taps are the raw matvec output of `attn_{k,v}_conv_proj` ([head][lag], lag fastest).
4. **Attention score** = `(q.k) * 1/sqrt(128) + bias` where the relative bias is added AFTER the scale (llama.cpp folds it into the KQ mask):
   `bias[h][d] = sum_f rel_bias[d][f] * R[h][f]` for distance `d = pos_q - pos_k < 1024` (d = 0 is the token itself), else 0. The sink is
   realised by subtracting `t_h` (= `attn_sink_proj(x)`) from the whole bias table, including the zero row for d >= 1024, plus a zero sink: i.e. the
   softmax denominator gets `+ exp(t_h)` with `t_h` NOT scaled by 1/sqrt(128). With flash attention enabled llama.cpp casts that mask+bias+sink to
   **F16** (our kernels keep fp32). GQA: q head h uses kv head h/3. Global layers: `tau = 1 + 0.1*ln(max(1,(pos+1)/8192))` multiplies Q and R.
   SWA window: distances 0..1023 visible. The gate is `sigmoid(gate_attn(x))` on the 3072-wide output before `wo`.
5. **MoE router**: `p = sigmoid(logits)`, selection score `p + exp_probs_b`, top-8 by `std::sort` (UNSTABLE for exact ties), weights are the UNBIASED `p[id]`,
   `w = p / ((float)sum_double(p) + 1e-20f) * 8.0f`. SiTU: gate = FIRST half of the fused gate_up output, up = second half,
   `4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)`.
6. **Depth mix** (`mix`): the weighted sum is over the RMS-normed states (values are normed too), `scale = 1/sqrtf(2304)` = 1/48 exactly, states are
   concatenated newest-block-first (order is irrelevant mathematically). **Controller**: `gate_in = [rms(cur) | rms(msg)]`, `out = (cur - tanh(f)*cur) + tanh(g)*msg`.
7. Q8_0 activations: the CONTRACT text says `q = roundf(x/d)`; the literal `quantize_row_q8_0_ref` uses `id = 1/d`, `q = roundf(x*id)`, while the code llama.cpp
   RUNS on this CPU (arch/x86 AVX path) uses `id = 127/amax` and round-half-even. Both are implemented (`Q8Mode::Ref` default, `Q8Mode::X86`), bit-exact to scalar copies.

## API summary (details in the header)
| item | functions | notes |
|---|---|---|
| 1 vector math | `vexp vtanh vsigmoid vsitu vsitu_fused vsoftcap`, `vsitu_q8_0` (SiTU + Q8_0 fused) | softplus/log1p not needed anywhere in this graph (tau uses one scalar `log` per token) |
| 2 norms | `rmsnorm`, `rmsnorm_q8_0`, `quantize_q8_0` (private copy of B1's), split forms `rms_scale` + `rmsnorm_apply[_q8_0]` (ranges, for threads) | output `(x*s)*w` (two roundings, as ggml); fused == two-step bit-exactly |
| 3 clips | `l2clip_heads(x, chan_scale, n_heads, dim, thr, post_scale)`, `vnorm_heads` | in place, any dim |
| 4 conv | `sconv_repack_static`, `sconv_step`, `sconv_seq` | SoA layouts: taps `[4][C]` (tap 3 = current), history `[3][C]` (row 2 = t-1); channel range `[c0,c1)` per call for threading; dyn per token |
| 5 attention | `attn_repack_rel_bias`, `KvLayer`, `kv_append[_n]`, `attn_decode_task/finish/attn_decode`, `attn_prefill_group`, `attn_suggest_nsplit` | see below |
| 6 router | `router_topk`, `router_select` | ids descending by selection score, **ties: lowest expert index first** (llama.cpp: unspecified); `ref::router_topk` replicates std::sort |
| 7 mix | `depth_mix(states, ns<=8, queries, nq<=4, outs, n, scale)`, `ctrl_gate_input`, `ctrl_apply` | mix reads each state twice, never materialises normed copies |
| 8 helpers | `bf16_to_f32 f16_to_f32 f32_to_f16 f32_to_bf16 vadd vmul vscale vaxpy vadd_scale vmul_sigmoid` | |

### Attention design
* **KV layout** (per layer visit = cache slot, fp16, allocate ZERO-initialised or any finite values; one allocation per tensor): blocks of 64 positions, block b at
  `k + b*65536` (128 KiB): **K `[kv head 8][d 128][key 64]`** (transposed: one 32-byte load = 16 keys of one dim, so QK needs no horizontal reductions),
  **V `[kv head 8][key 64][d 128]`**. Global layers: slot = pos, grow by appending zeroed blocks (1M tokens = 16384 blocks = 2.1 GB per tensor per visit; use mmap/NORESERVE
  to grow lazily). SWA layers: ring, slot = `pos % (n_blocks*64)`, `n_blocks = 16` for decode; prefill chunks of n tokens need `n_blocks*64 >= 1023 + n` (e.g. 32 blocks, n <= 1025)
  so no key needed by an earlier query of the same chunk is overwritten. Append first (`kv_append`, optional head range for threads), then attend (the query token's own K/V is a key).
* **Rel-bias table**: `attn_repack_rel_bias(rel_bias[1024][16] as in the GGUF, table[16][1152])` stores the bias reversed in distance with 64 zeros on both sides, so 16 consecutive keys
  read 16 consecutive floats (no gathers, zero for distance >= 1024 for free). 73.7 KB per physical layer, shared by the 3 passes.
* **Decode**: work items (group g = kv head 0..7, split s). `attn_decode_task(L, table, q, g, s, nsplit, &partial[g*nsplit+s])` writes only its own `AttnPartial`
  (running max, sum, 3x128 accumulators); `attn_decode_finish(q, g, partials+g*nsplit, nsplit, out)` merges the splits in index order (sink included), applies `sigmoid(gate)` and writes `out[g*384..+384)`.
  Splits cut the visible key range on 64-position block boundaries; results are bit-identical for a fixed `nsplit` whatever thread runs whatever item (tested: reversed item order).
  Use nsplit 1 for SWA (8 items), `ceil(threads/8)` or more for global (`attn_suggest_nsplit`). Scratch: 1.6 KB per partial; stack ~15 KB per call (prefill 30 KB).
* **Split rule shared by the executors (E1, wave 1)**: `attn_decode_nsplit(n_keys, split_keys, max_split)` is THE function every executor that runs the decode kernels per
  token must use (decode does; prefill / verify in their decode-kernel modes should, see docs/requests.md): 1 split below `2*split_keys` (767 keys with the engine default 384,
  the regime where the prefill tile kernel is bit-identical to decode), else by default the old `n_keys / split_keys` clamped to [1, max_split] (numerics identical to every
  earlier build); env `INK_ATTN_GRANULE=g` switches to `ceil(n_keys / g)` clamped to [2, max_split] (g = 256 measured: no gain over 2 splits per SWA visit, so opt-in);
  `attn_set_split_granule()` overrides it per process (benchmarks). It depends on the visible keys only, never on the thread count.
* **Software prefetch (E1, wave 1)**: `attn_range` prefetches 4 KiB ahead in both passes (2 lines per d row of K, 4 lines per V key, `_MM_HINT_T1`) and runs across block and chunk
  boundaries (K of the next block -> V of the chunk's first block -> K of the next chunk), because the L2 streamer restarts at every 4 KiB page and at every 16 KiB block.
  Pure hint, numerics untouched (verified bit-identical on 64-token logits dumps at ctx 1000/4000/8000 against the previous build). The engine maps the KV with MADV_HUGEPAGE since the same wave.
* **Prefill**: `attn_prefill_group(L, table, P, g, t0, t1, out[n][3072])` for token range [t0,t1) of a batch: 2 query tokens x 3 heads per tile, keys streamed in 1024-key chunks
  (scores of a chunk live on the stack), never an n x n matrix. Items (g, token range) are independent. Per-token `tau` array. Odd ranges fall back to a 1-query tile.
* The kernel takes `q`, `R` after clip and applies `tau/sqrt(128)` and `tau` itself; `gate` is the pre-sigmoid matvec output (nullptr = no gate).

## Accuracy (tests/kernels_misc/test_kernels_misc.cpp, vs double; all asserted)
| kernel | range | measured max error | spec |
|---|---|---|---|
| exp | x in [-87.3, 88.7] (clamped outside, exp(-inf) -> 1.2e-38) | 8.4e-8 rel | <= 2e-6 |
| tanh | all finite x (tested +-60 and magnitudes down to 1e-8; saturates to +-1 for |x| > 9) | 1.8e-7 rel | |
| sigmoid | x >= -87.3 (below: abs err < 1.2e-38) | 1.5e-7 rel | |
| softcap 30*tanh(x/30) | [-1500, 1500] | 2.7e-7 rel | |
| SiTU(4, 25) | g in [-60, 60], u in [-400, 400] | 4.2e-7 rel | |
| rmsnorm / clips / vnorm / depth mix / controller | random, many lengths | <= 2e-6 (asserted) | |
| short conv (static + dynamic, step == seq, channel-split bit-identical) | C = 100 .. 2304 | 4.8e-7 abs (O(1) values) | |
| router | 3000 random trials | ids exact vs std::sort reference (near-ties < 3e-6 skipped), weights <= 2e-6 | |
| attention decode / prefill, SWA + global, pos 0..4999, nsplit 1/2/5 | random q,K,V,R,sink,gate; K/V fp16-rounded inputs | 2.1e-6 (max abs diff / max abs ref) | <= 1e-5 |
NaN inputs are not supported by the vector math (they map to a clamp value; no trap).

### Cross-checks against the REAL llama.cpp (build-sys libraries; not in the default build, commands in the file headers)
* `tests/kernels_misc/xcheck_ggml_graph.cpp`: copies of the graph code of `clip`, `mix`, `control`, `convolve` (static + dynamic), the rel-bias / sink / mask / GQA attention section of
  `build_layer` + `build_attn_mha`, the router (`sigmoid`, bias, `argsort_top_k`, `get_rows`, normalise, x8), SiTU and the final softcap are executed by ggml's CPU backend on random data and compared with the
  ink kernels: clip 8e-8/1.3e-7, depth mix 2.7e-7, controller 9e-7, conv 7e-8/1e-7, attention softmax (GQA, bias after scale, sink) 1.3e-7, **whole attention block T=3 at pos 3000 (SWA ring wrap and
  global with tau, gate) 1.2e-6 / 1.0e-6**, router weights 1.1e-7 with **0 of 320 ids different**, SiTU 2.3e-7, softcap 1.9e-7 (all as max abs diff / max abs value).
* `tests/kernels_misc/xcheck_llama.cpp`: `ink::quantize_q8_0(Q8Mode::Ref)` is bit-identical to `quantize_row_q8_0_ref` and `Q8Mode::X86` to the AVX `quantize_row_q8_0` that libggml-cpu runs (0 mismatches in 65536 blocks).
  llama.cpp's own two quantisers differ only on exact rounding ties / last-ulp scale cases: 4 of 63488 random blocks (0.006%) plus every crafted half-integer tie block.

## Measured performance (shared host: load average 4.6-8 during these runs, 12 at worst, so treat as indicative; single thread pinned to CPU 0 unless noted; 2026-11-21)
| kernel | AVX-512 | scalar libm (what ggml does) |
|---|---|---|
| tanh, 201024 floats | 0.89-1.24 ns/elem (0.8-1.1 Gelem/s) | 31-52 ns/elem (35-42x slower) |
| sigmoid / exp | 0.81 / 0.55 ns/elem | 10.2 / 8.1 ns |
| final softcap 30*tanh(x/30), 201024 logits | 1.39 ns/elem = 0.28 ms (16 threads: ~0.02 ms) | 39.6 ns/elem = 8 ms |
| SiTU, 8 experts x 1152 | 2.4-3.8 ns/elem = 22-35 us per visit | 103 ns/elem |
| rmsnorm n=2304 / fused +Q8_0 | 0.26-0.41 us / 1.4-1.8 us | |
| l2clip Q 24x128 / K / R | 1.65 / 0.46 / 0.64 us | |
| sconv_step C=2304 static / dynamic | 1.7 / 3.2 us (C=1024: 1.3 / 1.7) | |
| depth_mix ns=5, nq=1 / 3 / 4 (n=2304) | 2.6 / 7.6 / 9.5 us (measured at load 10-13) | |
| ctrl_apply n=2304 | 5-6 us | |
| router_topk 128 -> 8 | 0.36-0.45 us | 2.1-3.7 us (std::sort) |
| kv_append (8 heads) | 2.3-2.5 us | |
| attention decode SWA (1024 keys), 1 thread, 8 groups | 248-288 us/visit (4.2 MB KV, 15-17 GB/s = single-core L3 bandwidth) | |
| same, 16 threads pinned to even CPUs, nsplit 1 | 35 us/visit (x40 visits = 1.4 ms/token; cache L3-resident in the bench, a real run streams 168 MB/token from DRAM: ~2 ms) | |
| attention decode global, 16 threads, nsplit 2 | ctx 32K: 1.49 ms/visit = 90 GB/s (97% of the 92.5 GB/s ceiling); 128K: 6.5 ms = 83 GB/s (90%) | 1 thread: 7-10 GB/s |
| global 1M context, 16 threads, nsplit 4 | 56.7 ms/visit = 75.8 GB/s (82%), x8 visits = **453 ms/token** (34 GB of fp16 KV per token; fp8 KV would halve it) | |
| attention prefill, 1 thread | SWA 249 us/token (40 visits: 10 ms/token single, ~0.6 ms with 16 threads); global ~8K keys: 1.9 ms/token/8 groups (53 GFLOP/s) | |
Decode attention is memory-bound (bytes of K+V per visit), not FMA-bound; the 2-query prefill tile gives only ~1.06x over the decode kernel per token because each tile still streams
the window from L3. Long-context prefill (>= 32K) would want a larger query tile (4-8 tokens) to cut that traffic: not done.

## Executor call order per layer visit (suggested fused phases)
1. depth mix (`depth_mix`, nq 1 or 3, or 4 for the loop messages) -> `ctrl_gate_input` -> matvec 4608->64 -> matvecs 64->2304 x2 -> `ctrl_apply` -> `rmsnorm_q8_0` (or `rms_scale` once + `rmsnorm_apply_q8_0` per thread range).
2. matvecs wq/wk/wv/attn_rel/sink/gate; `sconv_step` (K, V; thread-split over channels) -> `l2clip_heads` (Q with chan scale, K, R with tau post-scale), `vnorm_heads` (V) -> `kv_append` (split by head).
3. `attn_decode_task` items -> `attn_decode_finish` (gate fused) -> matvec wo -> `sconv_step` (2304) -> `vadd` residual.
4. depth mix -> (controller) -> `rmsnorm` -> router matvec -> `router_topk` -> expert GEMVs; `vsitu_q8_0` between gate_up and down; scale by weights; `sconv_step`; `vadd`.
5. Output: depth mix -> `rmsnorm` -> unembed matvec -> `vsoftcap` (monotone: argmax/top-k may skip it, temperature sampling needs it on all 201024 logits).

## Known limitations / follow-ups
* Prefill tile is 2 queries; a 4-8 query tile would cut L3 traffic for long prompts. K/V are fp16 only (an fp8 cache would halve the 453 ms/token at 1M).
* `Q8Mode::X86` follows the AVX path of llama.cpp; verify against the oracle which activation quantiser reproduces its logits best (B1/orchestrator decision).
* Vector exp saturates at x < -87.3 (no denormal outputs); SiTU/sigmoid inputs below that are irrelevant for this model.
* Tie order in the router differs from llama.cpp only for exactly equal selection scores (documented above).

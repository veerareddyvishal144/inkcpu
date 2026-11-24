# KERNELS_GEMM: batched GEMM kernels for prompt processing (agent G)

Files: `include/ink/kernels_gemm.h`, `src/kernels_gemm/{gemm,gemm_small}.cpp` (static lib `ink_kernels_gemm`, links `ink_kernels_gemv`, built with
`-ffp-contract=off`), tests/benches in `tests/kernels_gemm/` (`test_gemm`, `bench_peak`, `bench_gemm`). No repack of the GGUF weights is needed.

## 1. Verdict in four lines
- **Bit-identical to B1**: every y[t][r] equals `gemv_{q8_0,nvfp4,bf16,f32}` (optimised AND scalar ref) on the same row/token: `test_gemm`, 2454 cases, 0 failures
  (K 64..4608, odd rows, nt 1..256, 4 value modes incl. +-128 weights / 65504 and denormal scales / zero scales / random UE4M3 bytes, padded strides, row+token splits, MoE).
- **Speed vs B1 gemv (same matrix, single thread)**: 2.2-2.5x at nt >= 64 (Q8_0 34-39, NVFP4 29-33 GMAC/s vs 14-16), 1.3-1.8x at 4-16 tokens.
- **Ceiling**: the bit-identical scheme costs 5 vector uops (p0/p5) per (row, token, 64 MACs) = 2.5-2.6 cycles: 61 GMAC/s/core at 2.45 GHz, ~54 at the 16-thread
  clock (2.14 GHz) = ~0.86 TMAC/s aggregate on an idle host. Kernels reach **70-81 % (Q8_0) / 63-72 % (NVFP4, nt >= 64) of the same-window peak** single threaded.
- **Target 0.40 TMAC/s: met only in the best dense Q8_0 window (403 GMAC/s = 75 tok/s equivalent); MoE grouped measured 0.22-0.31 TMAC/s (41-58 tok/s)** at host load 8-18
  where even the pure-ALU peak loop gave only 430-600 GMAC/s at 16 threads. Details and limits in section 6.

## 2. API (see the header for full comments)
```cpp
// activations: quantise each token (quantize_row_q8_0), then pack ONCE per matmul input and FORMAT (Q8_0 and NVFP4 weights need different lane permutations)
size_t  gemm_act_bytes(int fmt, int nt, int k);                        // fmt: GEMM_FMT_Q8 | GEMM_FMT_NVFP4; buffer must be 64-byte aligned
void    gemm_pack_act(int fmt, const block_q8_0* x, int t0, int t1, int nt, int k, void* buf);   // token range [t0,t1): split over threads
void    gemm_quantize_pack_act(int fmt, const float* x, size_t ldx, int t0, int t1, int nt, int k, void* buf);  // quantise + pack from floats
GemmAct gemm_act_view(int fmt, const void* buf, int nt, int k);

// dense: y[t*ldy + r] = dot(W[r], x[t]), r in [0,nrows), t in [t0,t1)  (y pre-offset by the caller's row range; K % 64 == 0, K <= 8192)
void gemm_q8_0 (const void* w, size_t row_stride, const GemmAct& act, int t0, int t1, int k, float* y, size_t ldy, int nrows);
void gemm_nvfp4(...same...);                                           // per-expert F32 scale NOT applied
void gemm_bf16 / gemm_f32(const void* w, size_t rs, const float* x, size_t ldx, int nt, int k, float* y, size_t ldy, int nrows);  // K % 16

// MoE grouped
void moe_build_route(const int32_t* ids /*[ntok][topk]*/, int ntok, int topk, int n_experts, MoeRoute& r);   // pairs sorted by expert (stable)
void moe_work_items(const MoeRoute& r, int nrows, int chunk_rows, std::vector<MoeItem>& out);                // (expert,row-chunk), big experts first
void gemm_moe_nvfp4 / gemm_moe_q8_0(const void* w, size_t expert_stride, size_t row_stride, const GemmAct& act, const int32_t* act_rows,
                                    const MoeRoute& r, const MoeItem& it, int k, float* y, size_t ldy);   // y[pair*ldy + row]
```
`act_rows[p]` = activation row of pair p: `r.pair_token.data()` for gate_up (one activation per token, indirect addressing, no gather copy), `nullptr` for down (one activation per pair).
`r.pair_slot[p]` = original `token*topk + j` for scattering/combining. Tunables (defaults are the tuned values): `g_gemm_group` (64 tokens: activation group kept in L2 while
all row panels stream over it), `g_gemm_tile` (4 tokens per register tile, 1..5), `g_gemm_prefetch` (1).

### Recommended use
- **Dense** (wq|wk|wv, wo, attn_gate, ...): pack the nt (<= 256 per call) activations in parallel by token range, then
  `pool.run_chunks(nrows, 32, [&](int, int64_t b, int64_t e){ gemm_q8_0(w + b*rs, rs, act, 0, nt, k, y + b, ldy, int(e-b)); })`. Row chunk 16-32 (multiple of 4); do not split K;
  each call walks all nt tokens in groups of 64, the thread's weight slice (<= 0.5 MB) stays in L2 between groups. q|k|v share one pack.
- **MoE**: `moe_build_route` once per layer visit, `moe_work_items(r, 2304, 32, items)`, `pool.run_chunks(items.size(), 1, ...)` calling `gemm_moe_nvfp4` (gate_up with `act_rows = pair_token`,
  down with `nullptr` and a per-pair activation pack). Multiply the per-expert F32 scale / routing weight afterwards.
- **Small (BF16/F32)**: `pool.run_chunks(nrows, 2..8, ...)` over rows; they are 0.5 % of the MACs.
- Row counts/chunks that are multiples of 4 and token counts >= 16 per call are the efficient shapes; any count is correct. Experts with 1-3 pairs run a narrow tile (correct; 4 tokens measured at 41-46 % of peak, 1-3 tokens not measured and lower: the unpack is not amortised).

## 3. Numerics: why bit-identical, and the one deliberate difference
B1 keeps 16 float lanes per row; chunk c (64 weights, 2 Q8_0 blocks or 1 NVFP4 super-block) does `acc_l = fmaf((float)I_l, S_l, acc_l)` with `I_l` the exact int sum of 4 products and
`S_l = fl(dw*dx)` one float multiply; result `hsum16(acc)` (lanes j+8, j+4, j+2, j+1). The GEMM keeps exactly that arithmetic per (row, token): same ints (maddubs on |w| and the
sign-transferred activation, madd with ones), same `S_l` (one `vmulps` of the same two floats, done in a compact per-tile array), same `fma`, same hsum tree. The only change is the PHYSICAL lane
order of the accumulator (a fixed permutation chosen so that S is a plain memory broadcast: `vbroadcastsd` of [s_blk0,s_blk1] for Q8_0, `vbroadcastf32x4` of [sc0..sc3] for NVFP4); one
`vpermps` restores B1's order before `hsum16`, so no float operation differs. NVFP4 scales use a 256-entry table built with `ue4m3_to_f32` (identical for all byte values, incl. 0x7F -> 0).
Same dependence on MXCSR (FTZ/DAZ) as B1. Requirement: K % 64 == 0 (B1 allows K % 32 for Q8_0; the model has no such K; fall back to per-token `gemv_q8_0` if ever needed).

## 4. Kernel structure (why these tile shapes)
- Activation pack: int8 quants in the lane-permuted order (one `vpermd` per 64 values) + float scale arrays (Q8_0 dx per block, NVFP4 dx expanded to (dx0,dx0,dx1,dx1) per super-block), zero padded to 16 floats.
- Weight panel = 4 rows, unpacked ONCE per (panel, token group) into thread-local scratch: |w| u8 (Q8_0: `vpabsb`; NVFP4: nibble -> |kv| via `vpermd`+`vpsrlvq`+`vpshufb` with magnitude table
  0,1,2,3,4,6,8,12), a 64-bit sign mask per 64 weights (`vpmovb2m` / `vptestmb` on nibble bit 3), and float weight scales. Scratch strides are not powers of two (L1 set aliasing).
- Micro-kernel 4 rows x 4 tokens (16 zmm accumulators, tail tiles 1..3 tokens): per (row, token, chunk) `vpsubb{k}` (sign transfer), `vpmaddubsw`, `vpmaddwd`, `vcvtdq2ps`, `vfmadd231ps`
  = 5 p0/p5 uops, scale from `vbroadcast*` (load port only). Per tile and 16-chunk K segment: P = dw*dx for the 16 (row,token) pairs (store-bound, overlapped with the previous segment).
  Per panel: prefetch (T1) of the next panel's rows spread over the tiles. 6-uop variant (scale `vmulps` per pair) measured 12 % slower in `bench_peak`.
- Things tried that did not help: `vpblendmb` with precomputed -X instead of the merge-masked `vpsubb` (the reg copy is free via move elimination; blend costs 0.25 uop/pair more),
  tile 5 / group 32 / group 128 / prefetch off (all within noise at load 15; defaults kept).
- BF16/F32: 2 rows x 6 tokens register block, same two-accumulator structure as `gemv_bf16/f32` (a0 for the first 16 of each 32 and the 16-tail, a1 for the second 16).

## 5. Peak microbenchmark (`bench_peak`, uop mixes with L1 operands; load noted)
| mix (per row,token,64 MAC) | idle host, 1 thr | load 16, 1 thr (CPU 2) | load 16-17, 16 thr |
|---|---|---|---|
| FMA-calibrated AVX-512 clock | 2.45 GHz (load 0.7) | ~1.9 GHz (sibling busy) | 2.14 GHz (load 0.7) / 1.7 GHz (load 17) |
| mix4: maddubs, madd, cvt, fma | | 1.83 cyc, 66.6 GMAC/s | 2.19 cyc |
| **mix5: + sign vpsubb (the real kernel)** | 2.5-2.6 cyc ~ 61 GMAC/s | **2.65 cyc, 45.9 GMAC/s** | 2.50 cyc; aggregate 502 GMAC/s (per-thread 1.49/2.10/2.35 ns min/med/max) |
| mix6: + vmulps for the scale | | 2.77 cyc, 43.8 | aggregate 426 GMAC/s |
The 5-uop mix is port-bound as predicted (2 uops/cycle on p0+p5). Idle-host ceiling for the bit-identical scheme: 61 GMAC/s/core (1 thr), ~54 at 16 threads = ~0.86 TMAC/s.
(An earlier run in my status log printed 0.19 ns/pair: the compiler had hoisted the invariant ops; the shipped bench uses a memory clobber and the numbers above.)

## 6. Kernel benchmarks (`bench_gemm`, `tools/with_bench_lock.sh`; "% of peak" = vs the 5-uop loop measured in the same window on the same cores)
**The host was never idle: loadavg 8-18 during all of these (other agents; hyperthread siblings busy), so absolute GMAC/s are 30-50 % below an idle host; the ratio to the same-window peak is the robust figure.**
Single thread, CPU 2, 192-row weight slice resident in L2 (load 10-13 / second column: load 17, K-segmented final code):

| kernel | nt | GMAC/s (load 10-13) | % of peak | GMAC/s (load 17) | % of peak | B1 gemv, same window |
|---|---|---|---|---|---|---|
| Q8_0 192x2304 | 8 | 26-28 | 54-66 | 15.1 | 60 | 14.8 (load 10) / 7.7 (load 17) |
| Q8_0 192x2304 | 64 | 34.3-34.7 | 70-76 | 18.8 | 75 | |
| Q8_0 192x2304 | 256 | 33.6-35.2 | 70-78 | 17.3 | 73 | |
| Q8_0 192x3072 | 64 | 33.5-39.2 | 78-80 | 19.4 | 81 | 16.2 / 7.8 |
| NVFP4 192x2304 | 4 | 17.6-20.6 | 41-46 | 10.5 | 44 | 14.2 / 7.4 |
| NVFP4 192x2304 | 16 | 25.8-30.0 | 57-65 | 16.2 | 65 | |
| NVFP4 192x2304 | 64 | 28.8-30.9 | 63-69 | 17.2 | 72 | |
| NVFP4 192x1152 | 16 / 64 | 23.4-26.4 / 28-32.8 | 51-64 / 61-68 | 13.6 / 16.5 | 59 / 69 | 13.4 / 7.1 |
| BF16/F32 small (router 128x2304, sink, taps, ctrl 64x4608), nt=256 | | 9.5-10.3 GMAC/s per thread | | | | |

16 threads, CPU list {2,4,...,30,0}, dynamic `run_chunks`; "tok/s" = GMAC/s / 5.35 (equivalent prompt tokens/s if the whole token ran at this rate):

| kernel (16 thr) | GMAC/s (best window, load 8.3) | tok/s eq | % of 16-thr peak | GMAC/s (final run, load 17-18, chunk 32) | tok/s eq |
|---|---|---|---|---|---|
| Q8_0 3072x2304, nt=64 | **403** | 75 | 67 | 311 | 58 |
| Q8_0 2304x3072, nt=64 | 364 | 68 | 65 | 255 | 48 |
| Q8_0 3072x2304, nt=256 | 322 | 60 | 60 | 260 | 49 |
| Q8_0 2304x3072, nt=256 | 352 | 66 | 60 | 323 | 60 |
| Q8_0 1024x2304, nt=256 | 334 | 62 | 56 | 254 | 48 |
| NVFP4 2304x2304, nt=16 / 64 | 296 / 321 | 55 / 60 | 49 / 56 | 213 / 295 | 40 / 55 |
| NVFP4 2304x1152 (down), nt=16 / 64 | 279 / 275 | 52 / 51 | 46 / 48 | 213 / 264 | 40 / 49 |
| **MoE layer (gate_up+down), real routing, 64 tokens** | 283 | 53 | 46 | 219 | 41 |
| MoE layer, 128 tokens | 289 | 54 | 48 | 231 | 43 |
| MoE layer, 256 tokens | **310** | 58 | 53 | 254 | 47 |
| router F32 128x2304 nt=256 (pool) | | | | 154 | |
MoE input: `tools/route_traces/code.rtrace` forward 0, visit 12 (layer 12), first 64/128/256 tokens x top-8: 512/1024/2048 pairs, 128 experts, 70/46/24 unused, max 40/79/145 pairs per
expert, mean 8.8/12.5/19.7 per used expert, 21-26 experts with only 1-2 pairs. Weights 128 random NVFP4 experts (gate_up 2304x2304, down 2304x1152, 570 MB), indirect vs pre-gathered
activation rows: 239-310 vs 204-320 GMAC/s (no consistent difference, within noise): **use indirect rows, no gather copy**.

### What limits it (evidence)
1. **ALU ports**: 5 p0/p5 uops per 64 MACs is the floor for the bit-identical scheme (peak loop 2.5-2.65 cycles). The kernels reach 70-81 % of it for Q8_0 at nt >= 64.
2. **Per-tile overhead** (~10-15 %): scale-product phase (stores), 16 x (`vpermps` + `hsum16`) epilogue, accumulator init.
3. **Unpack per 4-row panel is not amortised for small n** (NVFP4 ~7 cycles per (row, super-block), store-port bound: 4 scalar scale stores + mag + mask): nt=4 44 %, 16 -> 58-65 %, 64 -> 63-72 %.
   MoE experts average 9-20 pairs, so the MoE rows sit at 46-53 % of peak. Next step (not done): vector scale decode (gather 16 super-blocks of scale bytes, integer decode of UE4M3) to
   cut the stores from 6 to ~2 per super-block; estimated +5 % on MoE.
4. **Host contention**: the same-window 16-thread peak was 430-600 GMAC/s (vs ~860 idle); `bench_peak` per-thread spread 1.49-2.35 ns/pair. Estimated (not measured) idle-host aggregate at the
   same efficiencies: Q8_0 ~0.55 TMAC/s, MoE ~0.4-0.45 TMAC/s.
5. Core 0 (CPUs 0/1) runs at about half speed (B1): it is last in the CPU list and dynamic chunks absorb it; static row splits would not.

## 7. Tests
`ctest -R test_gemm` / `build-gc/src/kernels_gemm/test_gemm` (5 s). Compares bitwise against B1's optimised gemv per token on every case and against the scalar `gemv_*_ref` when rows*nt*K <= 4e6
(1220 cases); checks guard values beyond `rows` in padded `ldy`, misaligned weight bases, padded row strides, independent token-range packing, 4x split calls (rows split at multiples of 4, tokens at nt/3),
MoE items executed in shuffled order with token-indexed and pair-indexed activations and repeated experts per token, `pair_slot` permutation, quantize+pack == quantize then pack.
Benchmarks: `bench_peak [threads] [cpus]`, `bench_gemm [--what st|dense|nvfp4|moe|small|all] [--threads 16] [--reps N] [--chunk R] [--group G] [--tile T] [--prefetch 0|1] [--trace PATH] [--visit V]`.

## 8. Wave 1 (2026-11-23, agent K3): re-encoded-scale unpack, batched epilogue, x4 prototype

Full log and raw numbers: `docs/status/K3.md`. All ratios are same-window A/B; the host was shared (loadavg noted there).

* **`gemm_nvfp4r` / `gemm_moe_nvfp4r`**: same arguments and the same `GEMM_FMT_NVFP4` activation pack as `gemm_nvfp4` / `gemm_moe_nvfp4`, but the weights' 4 scale
  bytes per super-block must be re-encoded per `include/ink/nvfp4_repack.h` (the engine does this at load when `Tune::nvfp4r` is set and marks `LayerPacked::nvfp4r`;
  never call the `_r` kernels on GGUF-encoded bytes). Unpack: 16 scales = 4 dword loads + 3 shuffles + `nvfp4r_decode16` + ONE 64-byte store into the unchanged
  `dw[row][4 sb + s]` layout, sign mask stored with `kmovq m64, k`, 4-super-block body unrolled. The old unpack was front-end bound (~23 fused uops per row
  super-block), the new one is ~13 with a port-5 floor of 4. Measured: unpack 1.2-1.6x faster; kernel +12.6 % at nt=4, +4.3 % at nt=16, +1-5 % at nt=64/256.
* **Batched bit-identical epilogue** (default on, `g_gemm_epilogue`; all Q8_0/NVFP4 entry points): the 16 accumulators of a tile are reduced pairwise with hsum16's tree in
  physical lane order (49 uops instead of ~160); 2/3/5-token tail tiles are batched too and 1-token tails are absorbed into a 5-token tile (`g_gemm_tail5`, default on).
  +1.7-3.0 % Q8_0, +2.1-4.0 % NVFP4 when L2-resident, 0 % when memory-streamed. Same floats in the same order: `test_gemm` (2454 cases, 651 on re-encoded
  weights incl. all 256 scale byte values) passes bit-identical against `gemv_nvfp4` / `gemv_q8_0` and the scalar references, also with the old paths forced
  (`INK_GEMM_EPILOGUE=0`, `INK_GEMM_TAIL5=0`, `INK_GEMM_TILE=5`).
* **Locked 16-thread MoE layer A/B** (real `code.rtrace` routing, visit 12, default CPU list, loadavg 1.5 at start; same-window ALU peak 723-738 GMAC/s):

  | tokens (pairs per used expert) | GGUF encoding + new epilogue | nvfp4r | ratio (layer) |
  |---|---|---|---|
  | 64 (8.8) | 322 GMAC/s (44.5 % of peak) | 352 (48.6 %) | 1.092x |
  | 256 (19.7) | 384 (51.9 %) | 402 (54.4 %) | 1.048x |
  | 1024 (69.4) | 407 (55.3 %) | 421 (57.2 %) | 1.035x |

  Wave-0 reference (G's table, load 8-18): 283 / 310 GMAC/s at 64 / 256 tokens. ESTIMATE on the PREFILL_DESIGN profile: ~+5.5 % prefill tok/s at 256-token chunks, more for 16-64-token prompts.
* **Tunables re-checked**: group 64 / tile 4 / prefetch T1 stay (group 32 -2..-4 %, tile 5 -3..-7 %, prefetch off / T0 and group 128-256 within noise); indirect activation rows
  beat a pre-gathered copy again. The inner loop runs at ~70 % of the 5-uop port bound; no bit-identical lever is left in it.
* **`gemm_nvfp4r_x4` / `gemm_moe_nvfp4r_x4` + `GEMM_FMT_NVFP4X4` pack (PROTOTYPE, NOT wired, numerics-deviating)**: weights stored as kv+12 (no sign masks), a 4-uop inner loop,
  and the 12*sum(x) bias correction applied as a K/16 float dot product. Kernel 1.13-1.19x vs the exact nvfp4r kernel at K=2304, 1.07-1.10x at K=1152; MoE layer +7-11 % on top
  of nvfp4r at 16 threads. Numerics: max |y - ref| / sum|w x| 5.2e-8 (exact kernel 1.7e-8); Q8_0 re-quantisation of raw outputs flips 1 of 55296 int8 values, 0 block scales.
  Using it would break prefill == decode bit-identity (decode stays exact) in exchange for an ESTIMATED further ~+6-7 % prefill; the parity gate against llama.cpp
  (argmax / top-10 / cosine) must be measured before it is ever enabled. Orchestrator decision pending.
* Bench: `bench_gemm --what rvs|epi|x4|unpack|tail|tune|moe --cpus ... --ntok ...` (median-of-ratios A/B). Under the bench lock use `--ntok 256` only: the full MoE
  setup (two 570 MB random expert stacks + re-encoding + peak calibration) held the lock 17 minutes for 1 minute of timing.

## 9. Wave 2 (2026-11-24, agent P2): int16-accumulation kernels `gemm_nvfp4r_i16` / `gemm_moe_nvfp4r_i16` (prompt chunks only)

Full log: `docs/status/P2.md`. Research basis: `docs/research/wave2_compute.md` approach #1. File: `src/kernels_gemm/gemm_i16.cpp`.
**Numerics-deviating by design** (owner decision in `docs/OPT_WAVE2.md`): the float accumulation runs once per 16 weights (+ once per 32) instead of once per 4,
so the outputs are NOT bit-identical to `gemv_nvfp4` (same accuracy class vs the true dot product: max relative error 3.9e-8 vs 2.0e-8, mean 4.3e-9 vs 3.8e-9,
see below). The engine uses them for PROMPT CHUNKS only (`PrefillTune::i16`, env `INK_PF_I16`, default 1); `engine_verify` and `Engine::decode` never do
(the executor aborts if the verify path ever selects them). Final wiring, parity gates and end-to-end numbers: the "Wave 2b" subsections at the end of this section.

### Scheme
* Weights: u8 = kv + 12 (0..24, the x4 table), no sign masks. Activations: s8 as packed (Q8_0 quants). `vpmaddubsw` pair sums are bounded by 2*24*128 = 6144,
  the sum of the 4 pair-sum vectors of a 16-k sub-block by 24576 < 32767: plain `vpaddw` is exact, no saturation can occur (unit test with w = +-12 everywhere,
  x = +-127 / -128). `vpmaddwd` with ones gives sum (w+12) x over the sub-block; subtracting the per-token int32 `xb[t][sb] = 12 * sum x` (computed in the activation
  pack) leaves the EXACT integer dot product I_sb. Float part per Q8_0 block b: `acc = fmaf(fmaf((float)I_{2b+1}, sw_{2b+1}, (float)I_{2b} * sw_{2b}), dx_t[b], acc)`.
* Per (16 rows, token, Q8_0 block = 512 MACs): 8 maddubs + 6 paddw + 2 maddwd + 2 psubd + 2 cvt + 1 mul + 2 fma = **23 p0/p5 uops = 2.875 per 64 MACs** (exact
  kernel: 5). Activation operands are `vpbroadcastd` loads (load ports only) straight from the existing lane-permuted NVFP4 pack (the 4 k of (sub-block s, quad q)
  of chunk c sit at byte 64c + 4(4q+s)); the bias and dx use embedded broadcasts. Register tile: 16 rows (dword lanes) x N tokens (N <= `g_gemm_i16_tile` = 8;
  an expert's n tokens are split into balanced tiles, e.g. 9 -> 5+4, 17 -> 6+6+5: no lane waste, and the per-token cost is nearly independent of N).
  The accumulator IS the output vector (16 rows of one token): no reduction epilogue; partial sums between K segments live in y itself.
* Unpack per 16-row panel and K segment (`g_gemm_i16_seg` = 16 super-blocks): 16 ymm loads of the nibble bytes -> 16x8 dword transpose (8 `vinserti64x4` +
  16 `vpunpck` + 8 `vshufi32x4`) -> per transposed dword `vpand` / `vpsrlw`+`vpand` and `vpshufb` with the kv+12 table = 16 output zmm; scales: one `vpgatherdd` of the
  16 rows' 4 re-encoded bytes, then per sub-block isolate the byte, shift to bits 20..27, `vmulps` 2^116 (the nvfp4r decode). ~90 vector uops per 16 rows x 64 k,
  i.e. ~2 tokens' worth of compute: amortised over ALL tokens of the call (vs ~13 uops per ROW super-block = 208 per 16 rows in the exact unpack). The next
  segment's rows are software-prefetched (T1) during the first tile of a segment.
* Activation format `GEMM_FMT_I16` = the NVFP4 pack + `xb` (int32 per token per sub-block); every NVFP4 entry point accepts an I16 pack (superset), so the engine
  packs once per layer visit whichever kernel runs (multix reads the raw Q8_0 rows anyway).

### Correctness (test_gemm, section i16)
704 cases bit-exact against a scalar model of the arithmetic (exact ints from the ORIGINAL GGUF bytes + the fmaf order above) over tile widths 1/3/5/8/12 x segment
lengths 1/2/5/16/128, row and token splits, all value modes incl. random UE4M3 bytes, MoE items in shuffled order with token- and pair-indexed activations and up to
~400 pairs per expert; adversarial saturation patterns bit-exact. Realistic 2304x2304 gate/up, 24 Gaussian tokens, 2640 samples vs double: max |y-ref|/sum|wx|
exact kernel 2.0e-8, i16 3.9e-8; mean 3.8e-9 vs 4.3e-9; Q8_0 re-quantisation of the outputs vs the exact kernel: 2 of 55296 int8 flips, 0 block scales,
0 flips on silu(gate)*up. (x4 prototype for comparison: 5.2e-8 / 1.1e-8.) The 2454 cases of the exact kernels are unchanged (ALL PASS).

### Kernel speed (same-window A/B, shared host: loadavg 6.7-13; see P2.md for the exact windows)
| single thread, i16 / exact nvfp4r | nt=1 | 2 | 4 | 9 | 16 | 64 | 256 |
|---|---|---|---|---|---|---|---|
| 192x2304 (L2-resident) | 1.19x (12.6 -> 15.0 GMAC/s) | 1.32x | 1.61x (23.7 -> 38.2) | 1.77x (29.4 -> 52.0) | 1.91x (32.4 -> 62.0) | 2.06x (35.5 -> 73.1) | 1.99x |
| 192x1152 | | | 1.60x | 1.89x | 2.02x (30.5 -> 61.5) | 2.21x | |
| 2304x2304 streamed / 2304x1152 | | | | 1.83x | 1.86x / 2.00x | 1.92x | |
Unpack share of the kernel time (from the nt sweep, L2-resident 192 rows): ~20 % at nt=9, ~12 % at nt=16, ~5 % at nt=64. It is store/port bound
(20 x 64-byte stores + 16 loads + ~90 vector uops per 16 rows x 64 k), not front-end bound: the uniform-stride address fast path (`g_gemm_i16_uniform`,
`bench_gemm --what i16u`) is 0.97-1.01x on L2-resident panels and 1.02-1.10x on streamed 2304-row panels (kept on; bit-identical either way).
MoE layer (gate_up + down), real routing code.rtrace visit 12, 4 threads, all experts through the GEMM: **1.53x at 64 tokens** (8.8 pairs per used expert, 111.5 GMAC/s)
and **1.75x at 256 tokens** (19.7 pairs, 138.8 GMAC/s = 91 % of the same-window 5-uop peak). Tunables: tile 4 is 3-5 % slower at nt >= 16, tile 12 = tile 8,
segment 36 (whole K at once) 5-20 % slower at nt <= 9 (cold unpack burst). Bench arms: `bench_gemm --what i16` (single thread), `--what moe` (i16 arm + engine-mix
arms `--mx 8,4,0`: multix below the threshold, nvfp4r or i16 above), `--i16tile / --i16seg / --i16pf`.

### Wave 2b (P2b, 2026-11-24 03:00-06:00 UTC): final wiring in the prefill executor
* `PrefillTune::i16` (env `INK_PF_I16`, **default 1**): in PROMPT CHUNKS every NVFP4 expert item of phases G (gate_up) and H (down) runs `gemm_moe_nvfp4r_i16`,
  whatever its pair count (the multix threshold `INK_PF_MULTIX` only applies when i16 is off, and in verify). `engine_verify` (speculative decoding, T <= 8 rows)
  and `Engine::decode` never use it (the executor aborts if the verify path ever sees the I16 pack): speculation stays token-identical to plain decode.
  `INK_PF_I16=0` = the wave-1 kernels everywhere: prefill == decode bit for bit. `INK_PF_I16=2` = A/B only: P2's first wiring (multix below 9 pairs, i16 above);
  its logits depend on the chunk boundaries, so it is not a production setting.
* Why ALL experts and not only the ones with > 8 pairs: the i16 result per (row, token) does not depend on the token tile, so with all-i16 the chunk size cannot
  change the logits (gate (c) below: chunk 1/2/7/64/256 of a 300-token prompt bit-identical to each other; decode-attention mode chunk 1000 == chunk 256 at 1305 and
  1500 tokens, exactly as in wave 1). With the mixed wiring the kernel choice depends on the pair count per expert, which depends on the chunking (P2's run: chunk
  1/2/7 == decode, chunk 64/256 not, 1500-token chunk 1000 vs chunk 256 cosine 0.985). All-i16 was also 1-3 % faster in P2's engine-mix sweep (4 threads).
* The activation pack of NVFP4 layers in prompt chunks is `GEMM_FMT_I16` (`efmt`), which the exact kernels and the multix diagnostic accept as a superset;
  `max_act_bytes` covers it. The only edits outside `src/kernels_gemm` / `tests/kernels_gemm` are the kernel selection in phases G/H (`i16`, `efmt`, `mx_max`, the
  verify guard), `max_act_bytes`, and the `i16` tune flag + env parsing.

### Wave 2b: parity gates on the final binary (all-i16 prompt chunks, 16 threads, under the lock, 2026-11-24 03:10-04:10 UTC, loadavg 6-16)
Decode is the engine's exact path (bit-identical to the wave-1 prefill); llama.cpp logits come from `tools/oracle` and `tools/build/oracle_logits` on the same GGUF.
| gate | result |
|---|---|
| unit tests (`test_gemm`, dev CPUs) | ALL PASS: 704 i16 cases bit-exact vs the scalar model (tile 1/3/5/8/12 x segment 1/2/5/16/128, row/token splits, MoE items in shuffled order, up to ~400 pairs per expert); adversarial saturation (w = +-12 everywhere, x = +127 / -127 / -128, alternating) bit-exact, <= 2.4e-7 vs double; realistic numerics exact 2.0e-8 / i16 3.9e-8; the 2454 exact-kernel cases unchanged. |
| (a) 6 oracle prompts + wiki300 (`test_prefill oracle`, `build-p2/oracle_dump.py`) | prefill vs llama.cpp: argmax equal 5/6, top-10 overlap 10/10/10/8/10/9, cosine 0.99939 / 0.99993 / 0.99917 / 0.99988 / 0.99927 / 0.99689 where decode has 0.99983 / 0.99982 / 0.99912 / 0.99998 / 0.99907 / 0.99166; KL(llama.cpp \|\| engine) mean 0.00147 nats vs decode's 0.00163 (prefill closer on s2 / s3 / m2, decode on s1 / m1). The two literal misses are ties: on m1 the engine puts p = 0.331 on its own top token and 0.329 on llama.cpp's (decode: 0.329 on the latter; llama.cpp's own top-2 gap is 0.095 logits), on c1 ranks 2-10 carry ~1e-10 probability (p(top) = 1.0000, KL 0.00000; llama.cpp's 10th-11th gap 0.032). Prefill vs decode: argmax 6/7, top-10 >= 8, cosine 0.9934-0.9999, max abs 0.4-3.3 logits: the flip-amplification regime of any non-bit-identical path (decode vs llama.cpp: 0.3-3.4). |
| (d) NLL vs llama.cpp `logits_all` (`test_prefill nll`) | 9 wikitext chunks, 3063 positions (2 x 256 from wave 1 + a 1024-token chunk + 6 x 256 built from the ring token files): every chunk within the 0.01 limit for both paths (largest: i16 0.00977 on nll2, exact 0.00813 on nll8); mean \|NLL - llama.cpp\| i16 0.00282 vs exact 0.00320; paired (i16 - exact) weighted by positions -0.00005 nats, i16 above exact on 5 of 9 chunks; argmax mismatches vs llama.cpp 76 vs 70 of 3072. 1024-token context: i16 1.91447 vs llama.cpp 1.91375 (exact 1.91488). No measurable direction. Table in `docs/status/P2.md` 03:56. |
| (b) prefill m tokens then 24 greedy decode steps (`test_prefill state`) | m = 43 vs all-decode: argmax agreement 24/24, worst cosine 0.99759, worst max abs 1.52 (no longer bit-identical, as accepted); m = 700 vs the decode-attention-kernel prefill: 24/24 bit-identical; m = 1000: 24/24 argmax, cosine 0.99652 (informational: decode nsplit > 1). |
| (c) chunk sizes (`test_prefill chunks`, `build-p2/chunkcmp.py`) | 300 tokens, chunk 1 / 2 / 7 / 64 / 256: final logits **bit-identical to each other** (pairwise, max abs diff 0), all at cosine 0.99986 / max abs 0.42 vs decode. 1305 and 1500 tokens: decode-attention mode chunk 1000 == chunk 256 **bit-identical**, fast-attention chunk 64 == chunk 256; 1305 tokens vs llama.cpp cosine 0.9975 / 0.9976 (decode 0.9994, exact fast prefill 0.9993; P2's mixed wiring 0.9998 / 0.9984: the last position of this prompt is a flat distribution, p(top) 0.24-0.32, and swings 10x in (1 - cos) between equally valid paths). |
| (e) determinism (`test_prefill determinism`) | 300-token prefill bit-identical across 16 / 16 / 12 / 8 / 8 / 5 threads and repeated runs. |
| speculative verify (`test_verify quick`) | PASS: every `engine_verify` row and tap bit-identical to `Engine::decode`, commit/rollback exact, both attention modes, T = 2 / 4 / 8 at p = 40 and T = 8 at p = 61 (the prefix is prefilled through the i16 path in both arms: verify never touches it). |
| rewind (`test_rewind quick`) | ALL PASS (29 checks, 0 failures): rewind to the prompt end / a mid-prompt chunk checkpoint / the exact bound / a deep checkpoint past the ring wrap, each followed by a 40-token tail prefill and 3-10 decodes: logits and slot state bit-identical to a fresh slot (both arms prefill through the i16 path with the same chunking; the i16 schedule is deterministic). |
| exact path `INK_PF_I16=0` | unchanged: `test_prefill oracle` and `chunks` with INK_PF_I16=0 on the final binary: prefill == decode BIT-IDENTICAL on all 7 prompts and for chunk 1 / 2 / 7 / 64 / 256, decode-attention mode bit-identical at 1305 / 1500 tokens, gates (a) and (c) PASS exactly as in wave 1 (`quick` with INK_PF_I16=0: BIT-IDENTICAL, P2 02:53). |
| final-binary smoke (after adding `INK_PF_I16=2`) | rebuild 38 steps, no warnings in kernel/engine files, `test_gemm` ALL PASS; `quick` with INK_PF_I16=2 reproduces P2's 02:47 mixed-wiring numbers to the last digit (s1 cosine 0.99971341 vs decode), `quick` with the default reproduces the pre-rebuild run (0.99951821), `determinism` PASS (16/16/12/8/8/5 threads). Mixed wiring on the 1305-token prompt: KL 0.0107 vs llama.cpp (exact fast prefill 0.0058, all-i16 0.0272, decode 0.0037); its 6-prompt KL mean 0.00107 (all-i16 0.00147, decode 0.00163): the three paths sit in one band. |

**Default decision (P2b): `INK_PF_I16=1`.** Every gate holds in substance: chunk independence and determinism bit for bit, verify and the exact path untouched, NLL /
KL / argmax rate at the exact path's level over 3000+ positions. The two binary misses of gate (a) (argmax on m1, top-10 on c1) are decided by 0.03-0.1-logit ties in the
reference that any non-bit-identical path (decode included) resolves by chance; they are reported, not hidden. `INK_PF_I16=0` restores wave-1 bit-identity with one env
var and is the fallback if the owner prefers prefill == decode over the 1.3x. The orchestrator's final idle-host run should measure bench_prefill at chunk 512 (server default)
with both settings.

Open items: the dense Q8_0 projections (2.8 of 8.4 ms/token) are the next kernel lever (3-uop int32 chain, ~1.3x on those phases, or the offset trick); ~6 % barrier /
straggler share; `tests/engine/test_prefill.cpp` (not P2's file) still prints FAIL for gates (a) / (b) / (c) on its bit-identity criteria when i16 is on: an INK_PF_I16-aware
criterion (argmax agreement + chunk-pairwise bit identity + NLL) or running those gates with INK_PF_I16=0 for the bit-identity check is requested in docs/requests.md;
docs/PREFILL_DESIGN.md sections 3 and 5 describe the INK_PF_I16=0 behaviour only.

### Wave 2b: end-to-end speed (bench_prefill, 16 threads, default CPU list, under the lock; exact = `INK_PF_I16=0`, i16 = default; one lock hold per A/B/A)
| chunk | prompt tokens | exact tok/s (A, then A again) | i16 tok/s | ratio (vs mean of the two A) | loadavg window |
|---|---|---|---|---|---|
| 512 (server default) | 256 | 87.1 / 84.3 | **112.4** | 1.31x | 12.7-15.7, instantaneous non-idle 0 % at the start |
| 512 | 512 | 87.1 / 87.2 | **119.3** | 1.37x | (target 120 at 512 tokens: reached within noise on a loaded host) |
| 512 | 1024 | 87.0 / 86.9 | **115.6** | 1.33x | |
| 256 | 256 | 87.8 / 85.8 | **113.6** | 1.31x | 13.5-15.8, non-idle 0 % at the start |
| 256 | 512 | 85.9 / 85.5 | **113.6** | 1.33x | |
| 256 | 1024 | 83.9 / 84.3 | **110.8** | 1.32x | |
| 256 (P2's mixed wiring, 02:57, separate holds) | 256 / 512 / 1024 | 87.2 / 85.6 / 83.2 | 116.8 / 113.6 / 110.2 | 1.34x / 1.33x / 1.32x | 10-13.6 |
Wave-2 baseline for reference (build-final, idle host): 86.8 / 83.5 / 82.3 at chunk 256: the exact arms above reproduce it, so the loaded windows are representative.

Per-phase profile (`INK_PF_PROFILE=1`, 512-token prompts, chunk 512, 16 threads, loadavg 16): exact 11.31 ms/token = 88.4 tok/s, i16 **8.40 ms/token = 119.1 tok/s**.
| phase | exact ms/tok (GMAC/s) | i16 ms/tok (GMAC/s) |
|---|---|---|
| MoE gate_up + SiTU + q8 | 4.619 (441) | **2.757 (739)** |
| MoE down GEMM | 2.520 (405) | **1.408 (724)** |
| proj q,k,v,gate,rel,sink GEMM (dense Q8_0) | 2.040 (467) | 2.053 (464) |
| wo GEMM + post-conv + resid (dense Q8_0) | 0.746 | 0.776 |
| loop controllers (batched) | 0.560 | 0.553 |
| attention + quant / conv / mixes / router / combine / lm_head | 0.82 | 0.85 |
| busy-time bound (no barrier wait) | 92.1 tok/s | 126.0 tok/s |
What limits it now: the two MoE phases are 1.71x faster (uop count 5 -> 2.875 = 1.74x theoretical) and now take 4.17 of 8.40 ms; the other half of the token is
unchanged: 2.8 ms of dense Q8_0 GEMMs (int8 x int8 cannot use the int16 chain: |w| <= 127 saturates 4 pair sums; the 3-uop int32 chain would give ~1.3x there),
0.55 ms of controllers, 0.27 ms attention, ~0.6 ms of mixes/quantisation/router/combine, plus ~6 % barrier/straggler wait (126 vs 119 tok/s busy-time bound).
DRAM is not the limit at the server's chunk size: the MoE phases stream 11.5 GB/s of expert weights at 64 MAC/B (= ~36 pairs per used expert at 512-token chunks;
the DRAM ceiling 80 GB/s x 64 MAC/B = 5.1 TMAC/s is 3.4x above the idle-host ALU peak of the i16 chain, 44.5 MAC/cycle/core x 16 x 2.14 GHz = 1.5 TMAC/s); at
256-token chunks (19.7 pairs, 35 MAC/B) it is still ALU-bound, only at 64-token chunks (8.8 pairs, 1.25 TMAC/s DRAM ceiling) do the two meet. The achieved 739 GMAC/s is
~49 % of the idle-host i16 peak in a loaded window (the exact kernel ran at 51 % of its own peak in the same window), i.e. the kernels keep the efficiency of wave 1 and
the whole gain is the shorter uop chain. Next levers (not done): the dense Q8_0 phases (3-uop int32 chain or offset trick), the barrier/straggler share, token-count-aware
work items for the small experts of short prompts.


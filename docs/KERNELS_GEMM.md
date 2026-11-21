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

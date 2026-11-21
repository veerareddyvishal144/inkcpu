# G status (batched GEMM kernels for prefill)

## Plan (written at start, 2026-11-21)
Design for bit-identity with B1's gemv_* (same 16-lane accumulators, same per-block fmaf(float(I), fl(dw*dx), acc), same hsum tree):
1. Activations are packed once per matmul input (`gemm_pack_act`): int8 quants in a lane-permuted order + float scale arrays.
2. Weights: per 4-row panel, unpack ONCE (amortised over all tokens of the group): |w| as u8, sign mask (64-bit per 64 weights), per-block float scales.
   NVFP4: nibble -> |kv| via pshufb, sign from nibble bit 3, scale bytes -> floats through a table equal to ue4m3_to_f32.
3. Shared micro-kernel 4 rows x T tokens (T<=5), 5 vector uops per (row, token, 64 MACs): mask_sub (sign transfer), maddubs, madd, cvtdq2ps, fma;
   the scale product dw*dx is precomputed per tile as a compact float array (one vmulps per 16 floats) and broadcast from memory (vbroadcastsd / vbroadcastf32x4, load-port only).
4. Accumulator lanes are permuted vs B1 (to make the scale broadcast free); one vpermps restores B1's lane order before the identical hsum16.
5. MoE grouped: indirect activation rows (no gather copy needed), (expert, row-chunk) work items sorted by token count.
6. BF16/F32: register-blocked (2 rows x 6 tokens), same lane/accumulation structure as gemv_bf16/f32.

## Milestones
- [x] M0 peak microbenchmark (ceiling per core): see Log
- [x] M1 Q8_0 + NVFP4 GEMM + tests bit-identical (test_gemm: 2454 cases, 0 failures, loadavg 13)
- [x] M2 MoE grouped + BF16/F32 + tests (same test_gemm)
- [x] M3 benchmarks (dense, NVFP4 per expert, MoE with real routing) - numbers in docs/KERNELS_GEMM.md section 6 (host load 8-18 throughout)
- [x] M4 docs/KERNELS_GEMM.md

## Log
- Peak (bench_peak, 5-uop mix = mask_sub, maddubs, madd, cvt, fma with L1 operands): 2.5-2.6 cycles per (row,token,64 MAC) pair, i.e. p0/p5 bound as predicted
  (6-uop mix 2.9). FMA-calibrated AVX-512 clock: 2.45 GHz with 1 thread on an idle host, 2.0-2.14 GHz with 16 threads. Ceiling per core ~53-60 GMAC/s (5-uop mix).
  (An earlier run printed 0.19 ns/pair: LICM had hoisted the loop-invariant work; fixed with a memory clobber, final numbers in KERNELS_GEMM.md.)
- All kernels BIT-IDENTICAL to B1 gemv_* (optimised and scalar ref) on first run: Q8_0, NVFP4, BF16, F32, MoE (token- and pair-indexed), row/token splits.
- Final state (2026-11-21): test_gemm ALL PASS (2454 cases, bit-identical to B1 gemv and gemv_ref). Single thread 70-81% (Q8_0) / 63-72% (NVFP4, nt>=64) of the same-window peak;
  16 threads: Q8_0 dense up to 403 GMAC/s (75 tok/s eq), NVFP4 single matrix 213-321, MoE with real routing 219-310 GMAC/s (41-58 tok/s eq) at host load 8-18.
  Target 0.40 TMAC/s met only for the best dense window; MoE limited by 5-uop port floor + per-panel unpack at 9-20 pairs per expert + host contention (see doc section 6).
- Open: vector UE4M3 scale decode for the NVFP4 unpack (est. +5% on MoE); batched hsum epilogue (~+2%); re-measure on a quiet host.
- Build: `cmake -S . -B build-gc -G Ninja && cmake --build build-gc --target test_gemm bench_gemm bench_peak` (PATH=/usr/bin:/bin).

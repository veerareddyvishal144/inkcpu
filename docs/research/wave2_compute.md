# Wave 2 compute research (R2d): faster bit-exact int8 x NVFP4 kernels on Skylake-SP

Target ISA: Xeon Platinum 8171M (Skylake-SP): AVX-512 F/BW/DQ/VL, FMA, F16C. NO VNNI / BF16 / AMX / VBMI / GFNI.
Status: FINAL, 2026-11-23 ~21:35 UTC. Budget used: 13 web searches, 15 fetches (2 failed/retried), ~25 min wall.
Nothing was built, run or edited outside this file.

## 0. Baseline (from the task brief)
- Prefill GEMM micro-kernel: 5 p05 uops per (row, token, 64 MACs): vpsubb{k} sign transfer, vpmaddubsw, vpmaddwd, vcvtdq2ps, vfmadd231ps.
  Peak = 2 ports x 64 MACs / 5 uops = 25.6 MACs/cycle/core = ~61 GMAC/s/core @2.45 GHz, ~0.86 TMAC/s for 16 threads; engine reaches 50-57% of it
  on MoE layers (85 tok/s, 11.3 ms/token, GEMMs ~85%).
- Decode NVFP4 GEMV: ~9 p05 uops per 64-weight super-block (36 B); gate_up phase at 68 of ~80 GB/s.
- Exactness rule: integer dot product per 4 weights must be exact; float accumulation-order changes allowed only if flagged.

## 1. Key fact that reframes everything: Intel's own peak for this ISA is 3 uops per 64 MACs, not 5
- Intel deck 784471 "AI Acceleration using Intel AMX/TMUL", slide 8 "Accelerated AI with Intel DL Boost" (verified with pdftotext on the fetched PDF):
  "Intel AVX-512: **85 int8 ops/cycle/core** with 2 FMA" (vpmaddubsw -> 16-bit -> vpmaddwd with 16-bit constant -> 32-bit acc)
  vs "Intel AVX-512 (VNNI): 256 int8 ops/cycle/core with 2 FMAs" (vpdpbusd); slide 10: AMX "2048 INT8 ops/cycle/core".
  https://cdrdv2-public.intel.com/784471/784471_AI%20Acceleration%20using%20Intel%20AMX-TMUL.pdf
  85 ops/cycle = 42.5 MACs/cycle = 2 x 64 / 3: vpmaddubsw + vpmaddwd + vpaddd, **int32 accumulation across K, float conversion only in the epilogue**.
- oneDNN's Skylake int8 GEMM does exactly this (src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.cpp):
  `vpmaddubsw(dp_scratch_, src1, src2); vpmaddwd(dp_scratch_, ones_, dp_scratch_); vpaddd(dst, dst, dp_scratch_);`
  tile IGEMM_UNROLL_M_ = 48, IGEMM_UNROLL_N_ = 8 (24 zmm accumulators, 3 A vectors, B via vpbroadcastd).
  https://raw.githubusercontent.com/uxlfoundation/oneDNN/main/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.cpp (+ .hpp)
- FBGEMM paper (arXiv 2101.05615, pdftotext of fetched PDF): "INT8 FMA with accumulation into INT16 is performed with a combination of vpmaddubsw and
  vpaddsw vector instructions ... we use two vector instructions for each vector FMA. Therefore, theoretical peak for accumulating into 16 bits is 2x that
  of FP32." and "INT8 FMA with INT32 accumulation is performed with a combination of vpbroadcastd, vpmaddubsw, vpmaddwd, and vpaddd ... theoretical compute
  peak for INT8 is not better than FP32" (Broadwell/AVX2 counting the broadcast). FBGEMM needs outlier-aware quantisation to keep int16 from saturating;
  **we do not**: |w| <= 12 bounds everything (see 2.1).
- uops.info (SKX): VPMADDUBSW zmm = 1 uop 1*p05, lat 5, tp 0.5; VPSHUFB zmm = 1 uop 1*p5, lat 1, tp 1.0; VPERMW zmm = 2 uops 2*p5, lat 6, tp 2.0.
  https://uops.info/html-instr/VPMADDUBSW_ZMM_ZMM_ZMM.html https://uops.info/html-instr/VPSHUFB_ZMM_ZMM_ZMM.html https://uops.info/html-instr/VPERMW_ZMM_ZMM_ZMM.html
  (vpmaddwd/vpaddw/vpaddd/vcvtdq2ps/vfmadd zmm are all 1 uop p05 on SKX; p1's vector ALU is fused off for 512-bit ops, so p0+p5 is the whole budget.)

## 2. Ranked kernel approaches for THIS ISA (prefill GEMM, 16 rows x 4 k per zmm, activation broadcast per token, as oneDNN/llama.cpp do)

### 2.1 Exactness bounds used below (E2M1 x2 integer codes w in {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}, Q8_0 x in [-127,127])
- Weight-offset trick (recommended): u8 operand = w + 12 in [0,24] produced directly by the existing nibble->vpshufb LUT (no extra uop), s8 operand = x.
  vpmaddubsw pair sum <= 2*24*128 = 6144 (no saturation); four pair sums added with plain vpaddw <= 24576 < 32767 -> exact; five would be 30720 (still exact).
  Correction: sum_k (w_k+12) x_k = sum_k w_k x_k + 12*sum_k x_k. The term 12*sum(x) over a 16-k sub-block is a per-TOKEN integer (compute during Q8_0
  quantisation, <= 12*16*128 = 24576), identical for every row -> 1 broadcast vpsubd per (16 rows x 16 k). No sign transfer, no -128 edge case
  (llama.cpp issue #29351 / PRs #29359, #31211 document that the vpsignb trick breaks for x = -128; the offset trick has no such case).
  Precedent for the offset/compensation idea: llama.cpp PR #933 ("bytes_0 * (bytes_1 - 8) + bytes_1 * (-8) + 64"), hdahroug.github.io (A xor 0x80,
  "Cbiased = (Asigned x Bsigned) + (128 x Bsigned)" corrected with column sums), oneDNN s8s8 compensation.
  https://git.k0r.in/n/llama.cpp/commit/f266259ad9a2bce5a34d919592310147af23f3dc https://hdahroug.github.io/optimizing-int8-gemm-avx2/
- Sign-trick variant (keep current vpsubb{k}): pair sum <= 2*127*12 = 3048, up to 10 pair sums exact in int16, but costs 1 uop per zmm of weights.

### 2.2 Ranking (uops are p05 uops per 64 MACs incl. amortised epilogue; MACs/cycle = 128 / uops)

| # | Approach | uops/64 MACs | MACs/cycle/core | vs now | Exactness | Effort | Risk |
|---|----------|--------------|-----------------|--------|-----------|--------|------|
| 1 | **int16 partial accumulation per 16-k sub-block + weight-offset trick**: per (16 rows x 16 k = 256 MACs): 4 vpmaddubsw + 3 vpaddw + 1 vpmaddwd(ones) + 1 vpsubd(bcast 12*sum x) + 1 vcvtdq2ps + 1 vfmadd (scale = d_x * s_w fp32) = 11 uops | 2.75 | 46.5 | 1.82x | integer dot exact (bounds 2.1); float accumulation now once per 16 weights instead of once per 4 (if the current kernel does cvt+fma per zmm) -> FLAGGED order change, strictly fewer roundings | medium (new micro-kernel, per-token sub-block sums, LUT outputs w+12) | low; verify no-saturation bound in a unit test with adversarial x = -127/+127 and w = +-12 |
| 2 | Same as 1 but int32 accumulation across both 16-k sub-blocks of a 32-k Q8_0 block using **integer sub-block scales** M*2^(e-emin) in the vpmaddwd (precedent: llama.cpp q4_K `p16l = _mm256_madd_epi16(scale_l, p16l); sumi = _mm256_add_epi32(sumi, sumj);` then one `_mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc)` per 256-element super-block) | ~2.6 | ~49 | 1.9x | exact iff 16-bit scale M*2^k <= 32767 (M <= 15 -> k <= 11) and int32 headroom; UE4M3 spread between two adjacent sub-blocks is tiny in practice but must be checked at pack time with a fallback path | high | medium (needs pack-time exponent analysis; only +5% over #1) |
| 3 | Intel/oneDNN 3-uop int32 chain (vpmaddubsw+vpmaddwd+vpaddd) with epilogue per 16-k sub-block: 12 + cvt + fma + sub = 15 / 256 MACs | 3.75 | 34 | 1.33x | exact | low | low. Strictly dominated by #1 (vpaddw is as cheap as vpaddd and halves the maddwd count) |
| 4 | Current kernel minus sign transfer only (offset trick): maddubs + maddwd + cvt + fma + amortised vpsubd | 4.25 | 30 | 1.18x | exact | low | low; smallest step, good first commit |
| 5 | FMA route (int8 -> fp32, vfmadd231ps = 16 MACs/uop): peak 32 MACs/cycle before conversions; weights nibble->fp32 need >= 3 uops per 16 values (and/shift, vpmovzxbd, vpermps-LUT) amortised over tokens, activations vpmovsxbd+vcvtdq2ps per 16 amortised over rows; at 9 tokens/expert effective ~26-28 MACs/cycle | ~4.6 | ~27 | ~1.05x | exact and fewer roundings | medium | simd-ml-kernels measured the fp32-conversion path at "0.95-1.03x our FP32 GEMM: memory savings at FP32 speed, no gain in time" https://github.com/Mattral/simd-ml-kernels/blob/main/docs/DESIGN.md . Not worth it; keep as a reference/validation kernel |
| 6 | LUT (T-MAC / bitnet.cpp TL2 style): exact products |w*x| <= 1524 need int16 tables -> vpermw (2 uops 2*p5 per 32 lookups = 16 MACs/uop) or lossy int8 vpshufb tables | >= 6 (+ table build per token) | <= 16-21 | 0.6-0.8x | vpshufb version NOT exact; bit-plane decomposition inapplicable to non-linear E2M1 codes | high | T-MAC's own x86 4-bit data: Surface Book 3 Llama-2-7B W4 llama.cpp 2.50 / 6.52 tok/s vs T-MAC 3.74 / 9.34 (1 / 4 threads) = 1.4-1.5x in bandwidth-bound tg, vs 2.2-2.5x at W2 https://github.com/microsoft/T-MAC/blob/main/docs/profiling_data.md ; M2-Ultra "1.1x (4-bit), 2.3x (2-bit)" arXiv 2407.00088; bitnet.cpp on i7-13700H at 1.58 bit: TL2_0 20.72 vs I2_S (MAD) 20.62 tok/s (7B), 4.99 vs 5.70 (30B), and "TBL+ADD+CVT took 6.20 ns, approximately 68% longer than the raw latency of a single MAD instruction" https://arxiv.org/html/2502.11880v1 . Verdict: LUT only pays when it shrinks bytes (<= 2 bit) or when the MAD path lacks a good int8 instruction; vpmaddubsw at 64 exact MACs/uop beats any BW-era table lookup. Dead end here. |
| 7 | Packed arithmetic / SWAR (two int4 x int8 products per 16-bit lane, Ozaki-style splits) | n/a | <= 32 | < 1x | (16 w1 + w0) x = 16 w1 x + w0 x with |w0 x| <= 1524 (11 bits) overlaps; a non-overlapping pack needs 2 products per 32-bit lane = 32 MACs/uop < 64 | - | No published exact scheme found for x86; closest art is wrap-tolerant int16 lo/hi splitting (Luminex PR #126) and int8 -> 2 x int4 splitting on a CGLA (arXiv 2609.27453), both weaker than vpmaddubsw. Dead end. |

Theoretical floor on this ISA: 1 vpmaddubsw + ~0.75 vpaddw per 64 MACs + amortised epilogue ~= 2.0-2.2 uops -> 58-64 MACs/cycle (2.3-2.5x current).
Approach #1 gets 85% of that floor with modest complexity. Nibble decode (vpand, vpsrlw+vpand, 2 x vpshufb = 2.5 uops per 128 weights... = 2.5/64)
is amortised over the tokens of an N-tile; at 9 tokens/expert it adds ~0.28 uops per 64 MACs -> decode each 32-48-row panel once into L1/L2 int8 (48 x 2304 = 110 KB)
and reuse it for all tokens of the expert instead of re-decoding per N-tile.

### 2.3 Register-blocking recommendations (Q4) for 9-20 tokens x 2304x2304 panels
- Published shapes, all built around "weights as 16-lane vector, activation broadcast": oneDNN SKX s8u8s32 48x8 (24 acc + 3 A + 1 B + scratch + ones);
  hdahroug AVX2 6x16 (12 acc, "98% of oneDNN", ~360 GOPS on EPYC 7763; oneDNN on AVX-512 8370C 1200 GOPS); llama.cpp x86 repack Q4_0x8: 4 tokens x 8 rows,
  AVX-512 no-VNNI path `_mm512_maddubs_epi16` via mul_sum_us8_pairs_acc_int32x16, int32 accumulation over the 32-element block then one
  `_mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), ...)`; nibble decode `_mm256_shuffle_epi8(signextendlut, raw & m4b)`.
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cpu/arch/x86/repack.cpp
- The engine's 50-57% efficiency is suspiciously close to the N-tail utilisation of a fixed 8-token tile at 9-20 tokens (9 = 8+1 -> 56%; 17 = 8+8+1 -> 71%).
  Choose N per expert so that tiles are full: with Mv = 2 zmm rows (32 rows), 4 k-steps of decoded weights resident (8 regs), N <= 9 tokens per tile
  (18 fp32 acc + 8 w + 4 temps + 2 const = 32 zmm); split 10-20 tokens as 5+5, 6+6, 7+7, 8+8, 9+9 (or 3 tiles) so no tile is below ~55% of 9.
  For #1 the int16 chain (4 maddubs -> 3 paddw -> maddwd) lives in 2 temps per (Mv,n) in flight; ~11 uops per chain, 21-cycle latency,
  SKX's 97-entry scheduler covers >= 4 chains, enough for 2 uops/cycle.
- Measured Skylake-SP no-VNNI int8 efficiency in the literature is only indirect: Intel NMT paper arXiv 1906.00532 "INT8 MatMuls using VNNI provides a speed-up
  of 3.7X over FP32 ... VNNI over AVX512 for INT8 MatMul is 2.3X" -> legacy AVX-512 int8 = 1.6x fp32 vs a 1.33x theoretical ratio (85/64), i.e. MKL's
  3-uop kernel ran at a HIGHER fraction of its peak than the fp32 GEMM did (plausibly 80-90%); IntelCaffe on Skylake 8124M (arXiv 1805.08691): int8 end-to-end
  "1.38X-2.9X ... over the FP32 baseline". No direct "X% of 85 ops/cycle" figure exists in public sources.

## 3. Decode GEMV (Q5): getting below ~9 uops per 64 weights
Current (per brief): nibble unpack 3 + maddubs/madd 2 + bias sub 1 + cvt 1 + fma 1 + scale 1 = 9. Options, all p05-bound:
1. Offset trick instead of bias/sign (-1 uop per 64 weights): LUT emits w+12; correction 12*sum(x_sub) * d_x(blk) * s_w(row,sub) summed over the row is a
   dot product of the row's 144 decoded sub-block scales with a per-token vector -> 9 fma + 9 loads per 2304-row (0.25 uops per 64 weights) instead of 36 vpsubd.
   Net ~8.25 uops/64. Exact.
2. Integer sub-block scales inside the vpmaddwd (llama.cpp q4_K precedent above): per 64-weight super-block the 4 UE4M3 scales become int16 M*2^(e-emin_sb)
   (M = 8+m, or m for subnormals); vpmaddwd(pairsums, scales16) replaces vpmaddwd(ones) at zero extra cost, leaving ONE cvt + ONE fma per super-block with the
   float 2^emin_sb * d_x. Exact iff 15*2^k <= 32767 (k <= 11) and 2*6144*15*2^k < 2^31 (k <= 13) -> essentially always; pack-time check + slow-path flag.
   If the current 9 already includes per-sub-block scale handling ("scale 1" + "cvt 1" + "fma 1" per 64), this folds it to per-super-block: ~7.5 uops/64.
3. int32 accumulation ACROSS super-blocks (defer cvt+fma to once per 2-4 super-blocks) additionally requires the activation scale d_x to be constant over
   those 64-128 activations, i.e. Q8_0 per-32 -> per-128 (or per-token) activation scales for the decode GEMV. That is a numerics change (not just order) -> only
   with an explicit parity decision. Headroom: per lane <= 4 pair sums * 6144 * 15*2^k; with k <= 6 you can add ~360 super-blocks before 2^31, i.e. a whole row.
4. Nibble unpack floor on this ISA is 2.5 uops/64 (vpand; vpsrlw+vpand; 2 x vpshufb per 128 weights). The shift-free variants need vpmultishiftqb (VBMI) or
   vgf2p8affineqb (GFNI), both absent on Skylake-SP; bf16-exponent-gemv and Intel's 2-bit VNNI kernel (arXiv 2508.06753, "vpsrld ymm5, ymm3, 0x4 ... with 2 more
   shuffles") both use the shift. microsoft/BitNet issue #420 confirms the p5-only vpshufb pressure on Intel cores.
   llama.cpp's own AVX2 NVFP4 vec_dot (PR #23961, tested on an i9-7900X = Skylake-X, same ISA class): per-16 fp32 scales via the UE4M3 LUT, int accumulation
   across sub-blocks with _mm256_madd_epi16; Qwen3-4B pp512 went 2.85 -> 30.48 tok/s just by removing ldexpf() from the hot loop. https://github.com/ggml-org/llama.cpp/pull/23961
Realistic decode outcome: 9 -> ~7.5 uops/64 weights (-17% ALU), but gate_up is already at 68/80 GB/s so the end-to-end decode gain is capped near +10-15%;
the ALU reduction mostly buys margin so other phases also stream at the ceiling.

## 4. Verdict: can prefill pass ~100 tok/s on this CPU?
YES, with good confidence, via approach #1 (int16 partial accumulation + weight-offset trick) plus full N-tiles:
- Today: 11.3 ms/token, GEMM ~9.6 ms at 25.6 MACs/cycle peak and 50-57% efficiency (~0.45 TMAC/s achieved of 0.86 peak).
- #1 raises the ALU peak to 46.5 MACs/cycle (1.82x -> ~1.8 TMAC/s for 16 cores @2.45 GHz). At unchanged 55% efficiency GEMM -> ~5.3 ms, token -> ~7.0 ms = ~140 tok/s.
  With N-tiles sized to the token count (efficiency 55 -> ~70%) GEMM -> ~4.1 ms, token -> ~5.8 ms = ~170 tok/s. Step #4 alone (drop sign transfer) gives ~95-100 tok/s.
- Bandwidth check: expert weights give 2 MACs per byte per token; at 9 tokens/expert the DRAM ceiling is 80 GB/s x 2 x 9 = 1.44 TMAC/s, at 20 tokens 3.2 TMAC/s.
  After #1 the ALU peak (1.8 TMAC/s) exceeds the 9-token bandwidth ceiling, so the low end of the 9-20 range becomes co-limited by DRAM; expect 130-160 tok/s
  for typical mixes, ~180 only when tokens/expert >= 15 and the non-GEMM 15% (attention, routing, quantisation) is also trimmed.
- Beyond that there is no ISA headroom: the floor is ~2.0-2.2 uops/64 MACs (58-64 MACs/cycle); fp32 FMA, LUT and SWAR routes are all slower than vpmaddubsw here.
Recommended order: (a) offset trick + per-token sub-block sums (#4, small, exact, immediate ~15%); (b) int16 4-step chain (#1); (c) token-count-aware N tiling
and L2-resident decoded panels; (d) only then consider integer sub-block scales (#2) and the decode changes in section 3.
Validation to add: adversarial unit test (x = +-127 everywhere, w = +-12) proving no vpmaddubsw/vpaddw saturation; parity run flagged for the per-16 float
accumulation granularity.

## 5. What I could not find
- No public measurement of oneDNN/MKL int8 GEMM as a percentage of the 85 ops/cycle Skylake-SP peak (only the indirect 1.6x-fp32 and 1.38x-2.9x figures above).
- No published exact 4-bit x int8 kernel that beats 3 uops per 64 MACs on AVX-512BW; the int16-chain estimate (#1) is derived from FBGEMM's acc16 principle
  and the E2M1 magnitude bound, not from a published NVFP4 kernel.
- No T-MAC x86 data beyond the Surface Book 3 table; no per-core GMAC/s for T-MAC's 4-bit kernel (its paper reports tokens/s and relative speedups only).
- FBGEMM acc16 measured GOPS on Skylake (paper shows Broadwell: 153.6 GOPS single-core peak = 2x fp32; actual values are only in a figure).
- ik_llama.cpp has no AVX-512-without-VNNI path at all (Skylake-SP falls back to AVX2 kernels; PR #2430 comment), so no numbers there.
- T-MAC repo page fetch returned HTTP 503 once (raw.githubusercontent copy used instead); Intel whitepaper "Lower Numerical Precision..." (doc 754765) not
  fetched (budget), its 1.33x figure is reproduced via the 85/64 ratio.

## 6. Source list
- Intel AMX/TMUL deck 784471 (slide 8/10): https://cdrdv2-public.intel.com/784471/784471_AI%20Acceleration%20using%20Intel%20AMX-TMUL.pdf
- oneDNN SKX s8u8s32 kernel: https://github.com/uxlfoundation/oneDNN/blob/main/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.cpp
- oneDNN int8 nuances: https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html
- FBGEMM paper: https://arxiv.org/pdf/2101.05615 ; blog https://engineering.fb.com/2018/11/07/ml-applications/fbgemm/
- uops.info SKX pages for VPMADDUBSW / VPSHUFB / VPERMW zmm (links in section 1)
- T-MAC: https://arxiv.org/html/2407.00088v2 ; https://github.com/microsoft/T-MAC/blob/main/docs/profiling_data.md
- bitnet.cpp: https://arxiv.org/html/2502.11880v1
- llama.cpp: PR #933 commit f266259 (AVX-512 Q4_0 offset trick), PRs #29359/#31211 (-128 sign-trick bug), x86 repack.cpp and quants.c (q4_K madd with scales), PR #23961 (AVX2 NVFP4), PR #6414 (tinyBLAS)
- justine.lol/matmul (tinyBLAS numbers: Mistral 7B q4_0 Skylake prompt 12 -> 28 tok/s)
- hdahroug AVX2 int8 GEMM: https://hdahroug.github.io/optimizing-int8-gemm-avx2/
- simd-ml-kernels DESIGN.md (fp32-conversion path = 0.95-1.03x fp32): https://github.com/Mattral/simd-ml-kernels/blob/main/docs/DESIGN.md
- Intel NMT int8 paper: https://arxiv.org/pdf/1906.00532 ; IntelCaffe int8: https://arxiv.org/pdf/1805.08691
- Intel 2-bit VNNI kernel paper: https://arxiv.org/pdf/2508.06753 ; microsoft/BitNet issue #420 ; ktransformers issue #1897 (E2M1 x2 as exact int8)
- ik_llama.cpp PR #2430, #2662, issue #2409 (VNNI gating, port-5 contention on Cascade Lake)

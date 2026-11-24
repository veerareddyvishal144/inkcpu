# RESEARCH_WAVE1: what the outside world knows that applies to inkcpu (agent R, 2026-11-23)

Scope: Xeon Platinum 8171M, AVX-512 F/BW/DQ/VL only (no VNNI/BF16/AMX/VBMI), 92.5 GB/s DRAM, 35.8 MB L3. Budget used: 13 web searches,
12 page fetches, 2 PDFs mined offline. Read-only; nothing built or measured. Engine numbers quoted below come from docs/ENGINE_DESIGN.md 9-10,
docs/KERNELS_GEMV.md 5, docs/KERNELS_GEMM.md 4-6. "Parity" = the engine's current contract (cosine >= 0.999, greedy agreement vs llama.cpp);
nothing below changes model semantics unless marked "precision trade-off".

## Headline
1. Every published fast AVX-512 4-bit path (ik_llama.cpp IQK, llama.cpp Q4_K repack AVX-512, Strata NVFP4, Unsloth sd.cpp 16-row GEMM) is
   built on `vpdpbusd` (VNNI). On this ISA they all fall back to AVX2-class code. Nobody has published a faster no-VNNI MAC chain than
   `vpmaddubsw + vpmaddwd + vcvtdq2ps + vfmadd` that the engine already uses; the engine's per-core 4-bit GEMV rate (~6 GMAC/s per GHz) matches
   the best no-VNNI number found (llama.cpp PR 933, Tiger Lake). The remaining headroom is entirely in NVFP4 *scale handling* (11 of 13 ALU
   uops per 64 weights in `nvfp4_rows`), not in the dot product.
2. No public CPU-only x86 number for Inkling 2 Mini beats what inkcpu already does; llama.cpp master still has no x86 SIMD NVFP4 dot.
3. Hot-expert L3 caching, routing-aware prefetch and MoE-aware speculation have no CPU evidence and the arithmetic on this box rules them out.
4. At 8k-32k context the attention phase, not the weights, becomes the bottleneck; a good CPU flash-attention kernel streams KV at 81-91 % of
   DRAM roof (measured elsewhere), vs the engine's 35 GB/s today. Lossless KV compression exists but costs ALU we do not have.

## Ranked recommendations (most applicable first)

### R1. Finish the NVFP4 scale-vector rewrite; fold the 4 sub-block scales into the `vpmaddwd` multiplier (integer-exact)
Expected effect: expert phases (52 % of the token) from 44-51 GB/s toward the 63 GB/s the Q8_0 phases reach: 5-10 % of the token (the engine's own
estimate for recipe (a)+(b) in ENGINE_DESIGN 10 is 3-5 %; the fold below removes more uops than (a)). Decode 14.3 -> ~15-15.5 tok/s is the realistic ceiling.
What: (i) the 4-row grouped scale decode and the UE4M3 -> normalised re-encoding already planned (ENGINE_DESIGN.md 10, OPT_WAVE1 header);
(ii) new: write each sub-block scale as I_b * 2^(E_b) with I_b in 0..15 (UE4M3 mantissa with hidden bit, or the bare mantissa for e=0) and
E_b = e-10 (e>=1) / -9 (e=0); per 64-weight superblock take E_min and store M_b = I_b << (E_b - E_min) as an int16 lane pattern. Then replace the
`vpmaddwd(pairs, ones)` widening step by `vpmaddwd(pairs, Mvec)`: the sub-block scale multiply becomes free, and the 16-lane float scale vector
(3 `vinsertps` + `vmulps` + `vpermps` + 8 scalar table loads per row superblock today) collapses to ONE broadcast of 2^(E_min) * row_scale * dx.
Exactness: |pair sum| <= 2*12*127 = 3048; with 15 * 2^spread <= 32767 (spread <= 11) products stay in int32, the 4 sub-block sums fit int32 for spread <= 10;
the integer part is exact, the only difference from the current reference is float summation order (one rounding per superblock instead of four), i.e.
MORE accurate, not bit-identical: the scalar reference must adopt the same formula; keep a per-row fallback flag for superblocks with spread > 10
(the microscaling paper found 97.9-99.97 % of real rows satisfy a stricter spread <= 3 condition at row granularity).
Evidence: the doubled-E2M1-is-integer trick and its exact-embedding proposition, arXiv 2611.00486 "Integer-Exact Microscaling" (Prop. 1: "If s_n <= 3,
|q_n,k| <= 96, so the row is an INT8 vector times one scale"; 1.10-1.57x on MoE models on Apple INT8 units). The ARM NEON NVFP4 kernel in llama.cpp
replaced branchy UE4M3 decode with `ue4m3_scale_lut[128]` (PR 19769 follow-ups, https://github.com/ggml-org/llama.cpp/pull/19769). Engine uop budget:
ENGINE_DESIGN.md 10 (13 ALU uops / 36 B superblock, 11 for scale decode + shuffles).
ISA caveat: `vpermw` is 2 uops on Skylake-X; expand the 4 int16 M values with `vpbroadcastq` + `vpshufb` (1+1 uop) instead. Denormal UE4M3 (e=0) must
be handled in the re-encoding, not at run time (same reason as recipe (b)).

### R2. Treat the decode GEMV as "done" on this ISA; stop looking for a better MAC chain
Expected effect: saves engineering time; sets expectations (stretch 17 tok/s is not reachable by kernel work alone: 3.87 GB at the 75-85 GB/s
phase-streaming ceiling = 45-52 ms, plus ~7 ms fixed = 15.5-17 tok/s only with a perfectly bandwidth-bound expert kernel).
Evidence: llama.cpp PR 933 (first AVX-512 Q4_0 dot, no VNNI used; 60,275 "FLOPS/us" = ~30 GMAC/s on one Tiger Lake core at ~4.7 GHz = ~6.4 GMAC/s/GHz;
engine `nvfp4_rows` 7.4 GB/s per thread = 14.8 GMAC/s at ~2.45 GHz = ~6 GMAC/s/GHz) https://github.com/ggml-org/llama.cpp/pull/933 . The author notes
"VPSIGNB doesn't have an AVX-512 version", the same sign-transfer problem the engine solves with masked `vpsubb`. mhhayashi's AVX-512BW (no VNNI) Q2_0
kernel on a Xeon W-2140B (Skylake-W, same ISA class): 34.68 GB/s single-thread vs 14.93 AVX2 vs 7.74 scalar, tok/s 4.4 -> 9.0 on Q2_0 models
(https://github.com/mhhayashi/llama.cpp/pull/1) - a 2-bit format with ONE fp16 scale per 32 weights, which is exactly why it is cheap: scale handling, again.
ik_llama.cpp: "#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)" gates all IQK
AVX-512 kernels; "If these five macros are not defined ... the build falls back to AVX2" (https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md).
Its R4/R8 speedups (Discussion 164, Zen4 7950X) therefore do not transfer. Strata's AVX-512 NVFP4 kernel also uses `vpdpbusd` (15.4 GB/s single-thread on Zen 4,
https://github.com/Niko1221/Strata/blob/main/docs/NVFP4.md). llama.cpp Q4_K repack AVX-512 (PR 29397) needs VNNI+VBMI.
Caveat: the ONE idea worth a microbenchmark besides R1 is processing two row superblocks per zmm with `k`-masked ops (PR 933's "two blocks at once" structure)
to halve the per-superblock epilogue; the engine's 4-row panel already amortises most of it in GEMM, less so in GEMV.

### R3. For 8k-32k contexts, make attention stream: fewer barriers, bigger key spans, and measure before any KV format change
Expected effect: at ctx 1000 attention is 5.8 ms/token reading 201 MB at ~35 GB/s (ENGINE_DESIGN 9); linear extrapolation to 8k on the
full-attention layers puts the attention phase at the same order as the whole expert phase, so the kernel's 35 GB/s -> 70+ GB/s would be worth
20-40 % of the token at 8k and more at 32k (ESTIMATE; the engine has no measurement above 2.7k).
Evidence that CPU attention can stream: a llama.cpp fork's decode profile at 24.8k context shows FLASH_ATTN_EXT reading 1.945 GB of q8_0 KV
at 259-292 GB/s on a 320 GB/s box (81-91 % of roof), 6.66-7.55 ms (https://github.com/cymru2000/llama.cpp/issues/6). ik_llama.cpp's CPU FA
beat no-FA by 20 % at 16k and 23 % at 32k (Ryzen 7950X) and processes 32k-token prompts at 122 t/s BF16-KV / 113 t/s Q8_0-KV
(https://github.com/ikawrakow/ik_llama.cpp/discussions/25). Mainline guidance: f16 KV is faster than quantised KV whenever it fits, "dequantization
runs on every token" (CUDA data: 64k ctx f16 79.6 vs q8_0 44.3 tok/s, https://sotaaz.com/post/llamacpp-kv-cache-quantization-guide-en).
Concrete steps: (i) one barrier per visit for attention instead of 3 (144 calls per token today), split keys across threads (the 2-way split already
gave +13 %); (ii) keep K and V in the per-head 64-position blocks but lay out K so a 64-key block of one head is one contiguous 64 x 128 x 2 B = 16 KB run
read with software prefetch 2 blocks ahead (THP already covers the hugepage question: 16.6 of 17.1 GB in AnonHugePages; nothing published on explicit
hugetlb for KV was found); (iii) only then decide on formats.
Precision trade-off options (NOT parity): Q8_0 KV halves bytes (llama.cpp -ctk/-ctv q8_0, PPL cost small). Bit-exact option: GAAC lossless KV coding,
1.405-1.407x on BF16 caches, 1.19-1.27x attention speedup at 64k on DGX Spark, 1.02-1.25x on Apple M4 (https://arxiv.org/abs/2610.38210) - but it is
exponent entropy coding decoded on the fly, i.e. ALU work on a core that is already ALU-bound; defer until attention is proven bandwidth-bound here.

### R4. Expert-phase structure: keep row-split dynamic chunks, co-schedule the same expert, prefetch only the first lines of the 8 routed experts
Expected effect: <= 1.5 % (the engine's own bound: DRAM delivers ~1.6 MB in a 20 us serial window). Low priority; included because it is the only
MoE-decode technique in the literature that applies here.
Evidence: KTransformers (SOSP'25) uses a tile-aligned per-expert layout with scales stored separately, "Expert weight matrices are vertically partitioned
into tasks dynamically scheduled across threads", "Dynamic task scheduling prioritizes co-scheduling tasks targeting the same expert", and a lightweight
AVX-512 kernel "consistently outperforming AMX when ARI is four or fewer tokens per expert" giving "up to 1.20x speedup in decode phases" (paper PDF via
https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf ;
kt-kernel backends https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md). The engine already does all of this except the
expert-ordered co-scheduling (its MoE GEMM uses indirect rows, which measured equal to gathered copies).
NOT recommended: an L3 "hot expert set". 8 experts x ~4 MB = ~32 MB per visit against 35.8 MB L3 and 3.87 GB of streaming traffic per token: nothing survives
to the next token, and the 11 intervening layers flush the loop's core layers between passes. The literature is split even for GPU-HBM caches (temporal
locality claimed by arXiv 2512.16473 and ST-MoE 2606.15453; denied by SpecMD 2602.03921, AdapMoE 2408.10284, ProMoE 2410.22134) and none of it is CPU-L3.
Looped/recurrent-depth models: nothing found on expert reuse across loop passes; "Layers 6-17 are looped 3 times", 5.4 B of 26.4 B active
(https://unsloth.ai/docs/models/inkling-2-mini). The sequential dependency between passes prevents fusing them.

### R5. Speculative decoding: the only parity-preserving lever is a cheaper verify GEMM (= R1); otherwise keep `auto` and stop
Expected effect: none beyond R1 unless outputs may change. The engine's measured verify cost (2.6x a plain step for T=4, break-even L ~2.9) is exactly
the literature's MoE effect plus this ISA's ALU bound.
Evidence: "draft tokens collectively activate more weights and can raise verification time by 2-3x, causing slowdowns of up to 1.5x even when token
throughput improves" (Utility-Driven SD, https://arxiv.org/pdf/2506.20675); OLMoE k=16 verify touches ~40 of 64 experts per layer (https://arxiv.org/pdf/2610.34721);
fine-grained MoEs show more routing overlap than independence predicts, modelled with IPF to < 4 % error (https://arxiv.org/abs/2610.38981, GPU-measured);
DraftExpert: under expert offloading "speculative efficiency ... is governed by how quickly the draft and verification stages expand the set of experts"
(https://arxiv.org/html/2607.24434v1). Unsloth's own DSpark numbers: 86.9 -> 96.6 tok/s (1.11x) on a B200 at depth 2 (https://unsloth.ai/docs/models/inkling-2-mini).
Output-preserving tricks: truncate the draft when the drafter's confidence drops (EVICT, https://arxiv.org/html/2605.00342v1; EcoSpec, https://arxiv.org/html/2607.12696)
- the engine's adaptive T policy is already this. Output-CHANGING tricks (not parity): MoE-Spec expert budgets (top 32 of 64 experts carry 93 % of routing
mass on OLMoE, https://bytez.com/docs/arxiv/2602.16052/paper), S2-MoE steering draft tokens to already-active experts (https://arxiv.org/pdf/2608.15018).
No pure-CPU measurement of batched verify paying off on a bandwidth-bound CPU was found.

### R6. Prefill GEMM: nothing published beats the 5-uop loop without VNNI; the known wins are structural
Expected effect: engine already at 70-81 % of its port-bound peak for Q8_0 and 46-53 % on real MoE routing (small per-expert n); the published
repack GEMMs reach their gains from the same source (amortised unpack over 8 rows x 8 columns), which the engine's 4x4 panel already captures.
Evidence: llama.cpp Q4_0_8_8 AVX-512 GEMM (PR 9532): pp512 67.9 -> 103.7 t/s (+52.8 %), tg unchanged (https://github.com/ggml-org/llama.cpp/pull/9532);
llamafile tinyBLAS on Skylake i9-9900: Mistral 7B q4_0 prompt eval 17 -> 28 tok/s, "on Skylake, llamafile users can expect to see a 2x speedup"
(https://justine.lol/matmul/, upstream https://github.com/ggml-org/llama.cpp/pull/6414). Both are AVX-512F + maddubs paths; neither reports uops per 64 MACs.
Only remaining lever found: the MoE GEMM's small-n inefficiency (nt=4: 44 % of peak, nt>=64: 63-72 %), which R1's cheaper per-panel scale decode addresses.
IPEX/oneDNN int4: IPEX WOQ uses custom kernels that assume VNNI/AMX for the int8 compute path ("hardware accelerators like Intel AVX-512, VNNI, and Intel
AMX"); no Skylake-class numbers exist (https://www.intel.com/content/www/us/en/developer/articles/technical/weight-only-quantization-in-llm-inference.html,
https://pytorch.org/blog/genai-acceleration-intel-xeon/ - the latter's AVX-512 int4 kernel packs [N/64, K, 32] tiles, same idea as the engine's panels).

## Q5. What the rest of the world gets on Inkling 2 Mini (CPU) and llama.cpp's NVFP4 CPU state
- CPU-only x86: 6.2 tok/s decode, 17.3 tok/s prefill, 8 vCPUs, ggml-org IQ4_NL GGUF, llama.cpp b12608, 18.3 GiB working set at 8k ctx
  (https://computingforgeeks.com/run-inkling-2-mini-locally/). Apple: M6 Mac mini 24 GB IQ3_XXS 15.52 tg / 28.57 pp
  (https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/discussions/7); M1 Pro 32 GB "about 10 t/s using q4 xl" (https://news.ycombinator.com/item?id=50581260).
  No x86 server-CPU numbers published; inkcpu's 14.3 tok/s on 16 Skylake cores at 92.5 GB/s has no public peer.
- llama.cpp master: `arch-fallback.h` still aliases `ggml_vec_dot_nvfp4_q8_0` to the generic scalar kernel on x86 (https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/arch-fallback.h);
  only ARM NEON got an optimised path (LUT + vpaddq + vfmaq). The closed PR 30690 measured nvfp4 at 11.27 GB/s "quantized (4096)" on master and made it 3 % slower.
  The open PR 29359 (exact -128 handling via operand swap) lists nvfp4 among the x86 int8-helper call sites, so an AVX2 nvfp4 path may exist in that branch; unverified.
  Recent AVX-512 vec_dot PRs for other types: iq2_s 1.15x, iq3_s 1.79x over AVX2 (https://github.com/ggml-org/llama.cpp/pull/32775, /pull/32774).
- Unsloth: "up to 1.9x faster than mainline llama.cpp for Inkling-2-Mini inference decoding" (unspecified "llama.cpp optimizations", GPU table only);
  mainline needs PR 32704 for Unsloth's root GGUFs; the HN "~30 % patch" is Vulkan-only (`ggml_cont` around strided MoE views).
- Thinking Machines: no CPU numbers found.

## Not found / could not verify (within budget)
- Any no-VNNI AVX-512 4-bit kernel with a published uop count per 64 weights (only speedup ratios exist).
- Any pure-CPU measurement of MoE speculative decoding paying off; any CPU-L3 hot-expert result; anything on expert reuse across loop passes.
- Explicit hugetlb vs THP for KV on CPU; CPU decode tok/s tables at 16k-32k with fp16 KV.
- The full text of arXiv 2610.38981 (only the abstract was reachable); PR 29359's absolute GB/s table.

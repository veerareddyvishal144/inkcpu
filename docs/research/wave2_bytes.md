# Wave 2 research: bytes per emitted token (agent R2b, 2026-11-23)

Scope: Q1 lossless weight compression, Q2 exact lm_head shortcuts, Q3 batched MoE decode on CPU, Q4 accepted precision trade-offs,
Q5 other exact byte reductions. Budget used: 18 web searches, 13 fetches (10 pages + 3 GGUF header byte-range downloads), 10 min wall.
Nothing was built, run against the server, or edited besides this file; `/tmp/r2b_*` holds the downloaded headers and PDF text.

Baseline used for all arithmetic (docs/ENGINE_DESIGN.md per-phase table, 12 threads, ctx 16): 3,865 MB/token, 76.3 ms at load
(brief: ~65 ms/token idle, ~60 GB/s average). Phase rates: lm_head 490 MB at 75-85 GB/s (5.8-6.5 ms), Q8_0 projections 1,015 MB at
63-67 GB/s, wo 361 MB at 52-57, NVFP4 experts 1,848 MB at 44-51 GB/s (40 ms), Q8_0 experts of L20-23 271 MB, controllers 113 MB (BF16
already), fp16 K/V 201 MB at ~35 GB/s at ctx 1000 (+4.5 ms). Expert kernel: 23 uops per 36 B superblock, 13 on the two vector ALU ports,
11 of them scale decode + shuffles; 7.4 GB/s per thread L2-resident. The brief's "9 uops" is superseded by this measured figure.
Rule that shapes every recommendation below: on this CPU bytes saved only become ms saved if the replacement kernel keeps the GB/s;
the 4-bit phases run at 44-51 GB/s against 63-85 for Q8_0, so a Q8_0 -> k-quant swap can be byte-neutral-to-negative in time.

---------------------------------------------------------------------------------------------------------------------------------
## (i) Ranked recommendations for THIS engine

### R1. Serve 2 (at most 4) sequences per step for aggregate throughput; keep n = 1 for latency. Exact. Effort low-medium.
Arithmetic (engine's own routing table, docs/ROUTING.md 2(d), tokens of one step share experts like neighbouring tokens): expert bytes per
step x1.59 at n = 2, x2.47 at n = 4; shared streams (proj 1,015 + wo 361 + lm_head 490 + ctrl 113 + router 28 = 2,007 MB) read once.
n = 2: 2,928 + 2,007 + 2 x KV = ~4,955 MB per step = x1.28 bytes for 2 tokens -> 2,478 MB per emitted token (0.64x). If the step stays at
60 GB/s: 83 ms per step = 24 tok/s aggregate, 12 tok/s per stream (vs 15.4 single). n = 4: 4,554 + 2,007 + 4 x KV = ~6,600 MB -> x1.71 ->
110 ms, 36 tok/s aggregate, 9 tok/s per stream. Per-token latency therefore never improves with batch; it is a throughput lever only.
Published curves agree: KTransformers 22.7 tok/s at B=1 -> 35.4 aggregate at B=2 (per request 13.6, -40 %), 36.0 at B=4, 36.6 at B=6
(CPU-centric hybrid, arXiv 2606.10493 Fig. 16: https://arxiv.org/html/2606.10493); "Batched decoding presents a fundamental bottleneck on
CPU-centric systems due to independent MoE activation patterns ... the set of activated experts expands multiplicatively" (same paper).
Intel/KTransformers balance_serve: 17 tok/s single user -> 40 tok/s aggregate at 4-way concurrency on Xeon6 + MRDIMM-8800
(https://github.com/kvcache-ai/ktransformers, doc/en/balance-serve.md via search). GPU reference, same model class (Qwen3-Coder-30B-A3B Q8_0,
llama-batched-bench on DGX Spark): S_TG 52.58 (B=1), 57.27 (B=2), 82.78 (B=4), 116.63 (B=8) tok/s
(https://jetsonhacks.com/wp-content/uploads/2025/10/spark-llamacpp-bench.html). Fiddler (ICLR'25): "On the CPU, the execution latency tends
to increase linearly with the size of the input batch" (https://arxiv.org/pdf/2402.07033) -- i.e. an ALU-bound kernel gains nothing from
batching unless the per-superblock decode (11 of 13 ALU uops here) is shared across the n inputs; the engine's multi-x GEMV (test_gemv_multix)
is the right structure, so the expert phase at n = 2 should cost ~1.3-1.6x of n = 1, not 2x. A badly structured batch path regresses: one
llama.cpp-derived engine measured 2 streams at 0.75x the single-stream aggregate (https://github.com/1bit-MONSTER/engine/pull/643).
Exactness: bit-identical per sequence as long as each row's accumulation order does not depend on n (the engine already proves two interleaved
slots bit-identical, ENGINE_DESIGN.md gate e). Risk: barrier count per visit grows; spin pools convoy under contention (documented).
Combine with the existing DSpark speculation (T-token verify already reads experts at 0.79x/0.62x/0.47x bytes per token at T = 2/4/8).

### R2. Exact lm_head screening (read ~25-40 % of the head, rescore candidates exactly). Exact. 2-4 ms/token (3-6 %). Effort high.
Design that keeps the 80 GB/s Q8_0 kernel: store a Q8_0 copy of the head restricted to the first w dims of a rotated basis
(W' = W R, R from the SVD of calibration hidden states, FEXIPRO's trick, https://www.comp.hkbu.edu.hk/~edisonchan/publication/FEXIPRO.pdf) plus
per-row L2 norm of the tail; per token compute x' = R^T x, partial logits over w dims for all 200,058 rows, upper bound = partial +
||w'_tail|| ||x'_tail|| + strict Q8 error bound (per-block |scale|/2 x ||x'_block||_1), lower bound symmetric. Rows whose upper bound exceeds the
best lower bound are rescored on the ORIGINAL Q8_0 rows with the current kernel, so emitted logits are bit-identical to today's. Greedy =
argmax of lower bounds vs upper bounds; sampling = Gumbel-max on the bounds (per-row Gumbel keyed by position and row), exact in distribution:
Mussmann & Ermon, ICML 2016, Theorem 3.2: "Suppose we have a MIPS technique with O(f(N)) query time. Then the MIPS reduction technique can
produce t exact samples from the model" (https://cs.stanford.edu/~ermon/papers/ICML_MIPS_Gumbel.pdf); top-k / top-p prefixes decided from
bounds as in tqllm's `sample_bounded`.
Bytes: screen at w = 576 of 2304 (25 %): 200,058 x 576 x 1.0625 B = 122 MB + 1.6 MB of norms/bounds; rescoring 8,192 rows = 20 MB.
Time: 124 MB / 80 GB/s = 1.6 ms + rescoring 0.3 ms + bound bookkeeping ~0.3 ms = ~2.2 ms vs 5.8-6.5 ms now -> 3.5-4 ms saved at 25 %, ~2 ms at
40 % dims. Only measured precedent for how often bounds decide on an LLM head: tqllm PR #64 (Qwen3.x-27B fine-tune, 248,320-row int8 head, 4-bit
full-dimension screen, +-6 sigma error model): "88.6 % of sampled tokens need nothing beyond the top-16, 11.4 % a 64-row round, 3.8 % a
1024-row round, 0.46 % a full int8 pass"; "2,056 sampled columns, 0 draws different from the reference"; largest error "4.80 estimated sigmas
over 510 M row-logits" (https://github.com/turbo-tan/tqllm/pull/64). That screen is probabilistic (6 sigma) and uses a 4-bit head copy; with
strict Cauchy-Schwarz bounds on a 25 % prefix the candidate set will be larger and is unknown for Inkling 2 Mini -- a one-day offline
experiment on dumped hidden states and the Q8_0 head settles w and the candidate count before any kernel work. Expected fraction on real logits
in the exact-MIPS literature: LEMP / FEXIPRO / MAXIMUS report 1.8-10x over brute force on recommender matrices (MAXIMUS "outperforms LEMP and
FEXIPRO by 1.78x on average, and up to 10.6x", https://arxiv.org/pdf/1706.01449), but those are 50-200-dim problems; LM-head margins are
smaller (2.4-2.9 % of positions are near-ties per the engine's own parity notes). Risk: medium (new kernel + sampler path, strict proofs needed).

### R3. Re-quantise the Q8_0 experts of layers 20-23 to NVFP4 (kernel exists). NOT identical. ~1.5 ms/token (2.3 %). Effort low.
Bytes: 271 MB -> 271 x 4.5/8.5 = 143 MB. Time: 271 MB at ~60 GB/s = 4.5 ms -> 143 MB at 47 GB/s (measured NVFP4 rate) = 3.0 ms -> 1.5 ms saved.
Evidence on acceptability, Unsloth's own files (GGUF headers downloaded 2026-11-23 from https://huggingface.co/unsloth/Inkling-2-Mini-GGUF):
UD-Q4_K_M differs from UD-Q4_K_XL ONLY in `ffn_gate_up_exps` of layers 20-23 (Q5_K instead of Q8_0; down stays Q8_0); Unsloth measures
mean KLD 0.0108 vs 0.0107 and top-token 97.9 % vs 98.0 % (16.61 vs 17.63 GB) (https://unsloth.ai/docs/models/inkling-2-mini). UD-IQ4_XS
(KLD 0.0109, 97.7 %) goes further: L20-23 gate_up Q5_K/Q4_K/IQ4_XS, down Q5_0 x3 + IQ4_NL x1, output.weight Q6_K, 10 late attn_q/attn_k
tensors Q6_K/Q5_K -- all of that for +0.0002 KLD. Why Unsloth keeps Q8_0 in the recommended file: those 4 layers are MXFP8 in the original
checkpoint ("UD-Q4_K_XL is also nearly lossless with MXFP8 weights dequantized to Q8_0"; "UD-Q8_K_XL upcasts MXFP8 to BF16 so KLD = 0",
same page), so Q8_0 is the cheapest near-exact container, not a sensitivity finding. NVFP4 with calibrated scales (imatrix_unsloth.gguf_file
is in the repo) should land near the IQ4_NL/Q4_K data point, i.e. a few 1e-4 KLD; must be measured with the parity gate.

### R4. Q8_0 KV cache (V first, then K). NOT identical. 1.5-3 ms at ctx 1000, ~6 ms at 8k, ~17 ms at 32k. Effort medium.
KV geometry (HF blog + engine): 48 visits x 8 KV heads x 128 x 2 x fp16 = 192 KiB/token; 40 SWA visits capped at 1,024 positions = 164 MB
(read every token once ctx >= 1024), 8 global visits grow 32 MB per 1,000 tokens. ctx 1000: 201 MB (engine measured, 35 GB/s, +4.5 ms);
ctx 8k: 420 MB; ctx 32k: 1.19 GB -- beyond ~20k context the KV stream is the largest per-token stream. Q8_0 halves it; realised gain at ctx
1000 depends on whether the attention phase is bandwidth- or barrier-bound (35 GB/s says partly barrier-bound; "144 calls = 3 barriers per visit").
Measured deltas elsewhere: llama.cpp PR #7412 (via https://refusalrate.com/kv-cache-quantization and https://modelvram.com/llm-vram-calculator/kv-cache-quantization/):
q8_0 K+V perplexity 6.2322 -> 6.2344, KLD 0.0010 (q4_0: 6.438). Nov 2026 KLD on Qwen3-8B: q8_0 0.0007, fp8 e4m3 0.0068, q4_0 0.0290; Llama-3.1-8B
q8_0 0.0005; "about 1 position in 90 picks a different first token" (https://sotaaz.com/post/fp8-kv-cache-en). Unsloth for this model: "Q8_0 KV
cache ... costs about 3 % to 7 % decode speed" for 2x context (GPU). Caveat: attention-sink models are K-sensitive ("for GPT-OSS and similar,
even q8_0 K changes output distributions", https://github.com/TheTom/llama-cpp-turboquant); Inkling has `attn_sink_proj` and gated attention,
so quantise V first and gate K separately.

### R5. lm_head Q6_K / Q5_K / Q4_K: bytes drop, time probably does not. NOT identical. 0-1 ms. Not recommended alone.
Bytes: 490 -> 378 MB (Q6_K, -112), 317 (Q5_K, -173), 259 (Q4_K, -231). But the Q8_0 head streams at 75-85 GB/s and no 4/6-bit kernel on this
CPU has exceeded ~50 GB/s: Q6_K at 60 GB/s = 6.3 ms (no gain), Q4_K/NVFP4 at 47 GB/s = 5.5 ms (-0.5 ms). Quality evidence: Unsloth UD-IQ4_XS
output.weight Q6_K inside a +0.0002 KLD bundle (above); Ripple-65B: Q8_0 attention+embed+output over Q6_K removes "about two thirds of the KL
gap" 0.0053 -> 0.0036 -> 0.0027 (https://huggingface.co/prometheusAIR/Ripple-65B-Preview-GGUF); bartowski Qwen3.8-27B: Q8_0 0.0009, Q6_K_L
0.0024, Q6_K 0.0030, and his current policy "a small table is kept at Q8_0, a large one follows the file's bitrate"
(https://huggingface.co/bartowski/Qwen3.8-27B-GGUF). Only worth it as the screen copy of R2 (a 4-bit screen as in tqllm) if R2's Q8_0-prefix
variant proves too loose.

### R6. Lossless entropy coding of the NVFP4 / Q8_0 streams: NO on this CPU (negative result with numbers).
Headroom: block-scaled low-bit formats "retain measurable redundancy, typically around 1.1-1.3x above their entropy bound, due to skewed symbol
frequencies and the structured metadata introduced by per-group scaling" (https://arxiv.org/pdf/2606.15789, Sec. II) -> E2M1 codes carry ~3.1-3.6
bits of the 4 stored; best case 1,848 MB -> ~1,500 MB (-19 %) = 6 ms if decode were free. Q8_0 codes are ~1.1x compressible (same paper, Fig. 2
"min: sq8 1.1x"; block-32 absmax int8 of Gaussian weights has ~7.9 bits of entropy) -> nothing there. Decode cost: the fastest CPU entropy decoders
are 16/32-lane interleaved rANS, "sub-3 cycles/byte", ~1 GB/s per core on AVX2 (https://encode.su/threads/2607-AVX-512-and-interleaved-rANS),
Recoil AVX-512 "max. 11+ GB/s decoding throughput" for a whole multi-core CPU (https://arxiv.org/pdf/2306.12141); DFloat11 needs 8-bit LUT
Huffman walks on GPU SRAM and reports no CPU decode (https://arxiv.org/html/2504.11651v3); ZipNN single-thread decompress 1.65 GB/s, 80 GB/s only
with a 224-core 8480+ (https://github.com/zipnn/zipnn). We would need 1.5 GB decoded in <= 30 ms = 50 GB/s on 16 cores = 3.1 GB/s per core of
OUTPUT on a kernel that is already ALU-bound (23 uops per 36 B), i.e. ~2 uops per 64 weights is 10-30x below any published decoder; and without
VBMI the 4-bit LUT is a 16-entry vpshufb, which cannot express variable-length codes. The only system that makes entropy-coded 4-bit weights pay
at the edge does it with a dedicated "0.21 mm2 ASIC decoder unit (1 GHz, 7 nm)" between DRAM and L2 and finds "A Huffman decoder beats an rANS
decoder in rate and area at equal throughput" (LazyWeights, https://arxiv.org/abs/2611.17108). Conclusion: skip; revisit only on AMX/VBMI hosts.

### R7. Expert caching / prefetch across tokens or passes: confirmed out. Exact but zero gain.
Engine data: neighbours share 3.3 of 8 experts; the 16 hottest experts of all layers need 1.97 GB vs 35.8 MiB L3; the same token's three passes
use 14.2 distinct experts of 24, but the passes are separated by >= 1.07 GB of other weights (docs/ROUTING.md). Layer-major reordering of the
passes (which would put a layer's three visits back to back and let its 20 MB of attention weights sit in L3, ~0.5 GB/token) is forbidden by the
model: "each core layer attends over two residual blocks from the previous pass" and reads "a message from the previous pass's outputs"
(https://huggingface.co/blog/inkling2mini) -- pass p of layer L depends on all of pass p-1. External confirmation of weak temporal locality:
adjacent-step expert overlap "mean 2.48 of 10 (median 2)", reuse distance p50 = 4 steps (https://github.com/ddvnguyen/llama.cpp/issues/130).

### R8. Attention-phase efficiency (not bytes): exact, ~2 ms at ctx 1000 and growing with context. Effort medium.
The fp16 K/V stream runs at ~35 GB/s against 63-85 GB/s for weight streams (ENGINE_DESIGN.md). Bringing it to 65 GB/s saves 2.7 ms at ctx 1000,
~6.5 ms at 8k, independent of R4; layout (per-head contiguous 64-position blocks already), fewer barriers (3 per visit now), software prefetch.
This is the exact-output alternative to R4 at short/medium context; R4 and R8 stack at long context.

Summary of ms/token: R1 none per token (throughput x1.5-2.3 aggregate) | R2 2-4 exact | R3 1.5 lossy | R4 1.5-3 at 1k, 6 at 8k, 17 at 32k lossy |
R5 0-1 lossy | R6 0 | R7 0 | R8 2-6 exact. Everything exact together (R2 + R8) is ~4-6 ms at ctx 1000 (65 -> ~60 ms); the exact floor remains
~48 ms because 2.1 GB of expert and 1.4 GB of attention bytes have no exact shortcut on this CPU.

---------------------------------------------------------------------------------------------------------------------------------
## Evidence notes per question

Q1. DFloat11 (https://arxiv.org/pdf/2504.11651): BF16 exponents Huffman-coded, "approximately 70 % of original size", "effective bit width of
around 11 bits", GPU hierarchical 256-entry LUTs decoding 8 bits per lookup; no CPU decode numbers; nothing about 4-bit codes. ZipNN
(https://arxiv.org/pdf/2411.05239): BF16 66.4 % remaining, 1.65 GB/s single-thread decompress; "up to 80 GB/s" with 16 workers on 2 NUMA nodes
of a 224-core Xeon 8480+. Shannon-bound paper (https://arxiv.org/pdf/2606.15789): tile-level rANS fused into GEMM tiles on GPU, "within 0.01-0.1
bits of the Shannon limit", int4 entropy ratios "6-10x" for unscaled int4 (not relevant: ours is block-scaled), block-scaled "1.1-1.3x".
Duda's ANS report (https://th.if.uj.edu.pl/~dudaj/ANSreport.pdf): SIMD Huffman 4-24 GB/s (M4) to 38 GB/s (c8i) on skewed data, machine-wide.

Q2. Exact MIPS: LEMP (https://dl.acm.org/doi/pdf/10.1145/2996452), FEXIPRO (SVD prefix + integer bounds), MAXIMUS (centroid bound + blocked brute
force wins). FlashSampling (https://arxiv.org/html/2603.15854): Gumbel-max fused into the head GEMM but computes every logit (no byte saving).
VocabTrim (https://arxiv.org/html/2506.22694v1): frequency-pruned head for drafters, approximate. tqllm PR #64 as above: practical, token-identical
in tests, probabilistic bound. Mussmann & Ermon 2016 as above (exact samples from exact MIPS; Ram & Gray exact space partitioning "speedups on
empirical datasets between 2 and 20,000"). Sampling without full logits that is exactly equivalent in distribution: Gumbel-max over bounded rows
is the only published family; everything else found (LSH softmax, Mussmann-Levy-Ermon UAI'17 O(sqrt n)) is approximate.

Q3. See R1. Additional: KTransformers DeepSeek tutorial per-request decode 16.8 -> 15.4 -> 14.2 -> 13.0 tok/s as max_batch_size grows
(https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md); PDScope/LayerScope/MoE-APEX only give relative gains.
No CPU-only llama-batched-bench MoE curve exists in the open; the closest CPU-ish point is the regression in 1bit-MONSTER PR #643.

Q4. See R3-R5. Inkling-specific layouts (headers read 2026-11-23): UD-Q4_K_XL: all 24 attn q/k/v/o/gate/rel Q8_0, L0-19 experts NVFP4
(ggml type id 40), L20-23 experts Q8_0, output and token_embd Q8_0 (file_type 7). UD-Q4_K_M: identical except L20-23 ffn_gate_up_exps Q5_K
(file_type 15). UD-IQ4_XS: output Q6_K, L20-23 gate_up Q5_K/Q4_K/IQ4_XS, down Q5_0/IQ4_NL, attn_q Q6_K x6 + Q5_K x1, attn_k Q6_K x4, attn_rel
Q6_K x5, attn_gate Q6_K x3. UD-Q3_K_XL (KLD 0.0174): output and token_embd Q5_K, L20-23 down IQ4_NL, core experts IQ2/IQ3, more attention at Q4-Q6.
Unsloth table (https://unsloth.ai/docs/models/inkling-2-mini): UD-Q4_K_XL 17.63 GB 0.0107 98.0 %; UD-Q4_K_M 16.61 0.0108 97.9 %; UD-IQ4_XS 15.73
0.0109 97.7 %; UD-Q3_K_XL 13.63 0.0174 96.3 %; UD-IQ3_XXS 11.66 0.0269 94.7 %; UD-Q2_K_XL 9.14 0.0595 91.6 % (output Q4_K).

Q5. Architecture facts used: 24 layers, layers 6-17 looped 3x (48 visits), 5 SWA (1,024) + 1 global per 6-layer block, 24 q heads / 8 KV heads,
head_dim 128, hidden 2304, 128 experts top-8 of width 1152, vocab 201,024 (200,058 real) (https://huggingface.co/blog/inkling2mini,
https://sebastianraschka.com/llm-architecture-gallery/models/inkling-2-mini/, https://miles.radixark.com/docs/models/thinkingmachines/inkling-2-mini).
Already exploited exactly in the engine: padded lm_head rows dropped (489.7 MB), BF16-representable F32 routers/controllers stored as BF16,
DSpark speculation. MXFP8 storage for L20-23 instead of Q8_0 would save only 3 % of 271 MB (8.25 vs 8.5 bpw) and would not reproduce the GGUF's
Q8_0 values: skip.

## (ii) What I could not find
- An isolated A/B of output.weight alone at Q6_K vs Q8_0 (all else fixed) with KLD; only bundled recipes (Unsloth IQ4_XS, Ripple-65B, bartowski).
- A CPU-only tokens/s-vs-batch curve for llama.cpp or ktransformers on a 30B-A3B-class MoE; only hybrid (2606.10493, KT balance_serve) and GPU.
- A per-code histogram or entropy figure specifically for NVFP4/E2M1 codes (only the 1.1-1.3x statement for block-scaled formats in general).
- Any CPU decode throughput for DFloat11; any entropy decoder on AVX-512 without VBMI that approaches 3 GB/s per core of decoded output.
- A published fraction of LM-head rows skipped by strict (non-probabilistic) norm bounds on a real LLM; tqllm's numbers use a 6-sigma model.
- KL/perplexity for Q8_0 K vs V separately on a gated / sink-projection attention like Inkling's (only the GPT-OSS warning).

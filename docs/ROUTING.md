# Routing study: expert reuse in verify batches and the bandwidth break-even of speculative decoding

Agent E, 2026-11-21. Question: decode on this CPU is bandwidth-bound, so can a T-token verify step (plus a drafter) read fewer bytes per
emitted token than plain decoding? What matters is how many DISTINCT experts T consecutive tokens touch. Everything here is measured on the
community GGUF `Inkling-2-Mini-UD-Q4_K_XL.gguf` with llama.cpp @2a28df0f9; acceptance of the DSpark drafter is NOT measured and no number
for it is assumed anywhere below.

## 1. Method (reproducible)
- Trace patch: `GGML_ROUTE_TRACE=<file>` in `/data/inkling-mini/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (next to the node profiler). For every
  `MUL_MAT_ID` node whose `src[0]` name contains `gate_up_exps` thread 0 appends a record: `int32 header[8] = {0x52545243, fwd, visit, layer,
  n_tokens, n_expert_used, 0, 0}` + `int32 ids[n_tokens][8]` (the `src[2]` expert ids, in ids-tensor order). `visit` counts the nodes inside one
  graph compute (0..47 = layers 0-5, 6-17 x3, 18-23), `layer` is the physical layer parsed from `blk.<l>.ffn_gate_up_exps.weight`.
  Built into `build-nprof` (`cmake --build build-nprof --target llama-perplexity`, PATH=/usr/bin:/bin); `build-sys` untouched.
- A prompt processed as one batch gives the routing of every position for every visit. Data sets (traces in `tools/route_traces/`):
  | name | source | forwards x tokens |
  |---|---|---|
  | wiki | wikitext-2 test through `llama-perplexity -c 512 --chunks 4 -t 8` (PPL 6.57 +- 0.52, sanity only) | 4 x 512 = 2048 |
  | code | first 3300 bytes of `llama.cpp/src/models/inkling2.cpp` (`tools/route_data/code.txt`) through `oracle_logits` (one batch) | 1 x 1255 |
  | chat | 4-turn transcript with the model's own markers (`tools/route_data/chat.txt`) through `oracle_logits` | 1 x 895 |
- `tools/route_stats.py wiki=... code=... chat=...` (interpreter `/data/engine/.venv/bin/python`) produces all tables; the complete output
  (including the 24-row skew table and the per-layer overlaps) is `tools/route_traces/route_stats_output.md`. Expert and non-expert byte
  counts come from `tools/bytes_per_token.json`, written by `tools/gguf_bytes.py` from the GGUF tensor table (see docs/BASELINE.md).
- Load while tracing: 1-min loadavg 2.6 to 15 (other agents were running); this does not affect routing, only the run time.
- "Mean" = unweighted mean of the three data sets. The spread between the data sets is small (+-0.01 in the ratios of (a)), so the numbers below do not
  depend much on the text type, but all three are English prose / C++ / chat: other languages or very long contexts are not covered.

## 2. Results
### (a) Distinct experts per visit for T consecutive tokens (mean over all 48 visits and all windows)
| T | independent 128(1-(1-8/128)^T) | wiki | code | chat | mean | mean / independent |
|---|---|---|---|---|---|---|
| 1 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 1.000 |
| 2 | 15.50 | 12.59 | 12.89 | 12.55 | 12.68 | 0.818 |
| 3 | 22.53 | 16.39 | 16.71 | 16.31 | 16.47 | 0.731 |
| 4 | 29.12 | 19.65 | 20.00 | 19.52 | 19.72 | 0.677 |
| 6 | 41.10 | 25.14 | 25.63 | 24.89 | 25.22 | 0.614 |
| 8 | 51.62 | 29.68 | 30.44 | 29.31 | 29.81 | 0.577 |

Two neighbouring tokens share about 3.3 of their 8 experts (41%; independent routing would share 0.5). By visit group the ratio to independent is
at T=4: prelude L0-5 0.80, core pass 1 0.64, pass 2 0.67, pass 3 0.68, L18-19 0.59, L20-23 (Q8_0) 0.68; the first layers are the least correlated.

### (b) Same token, expert sets of the three passes (layers 6-17, mean over layers and tokens; |S| = 8)
| data set | S1 and S2 | S2 and S3 | S1 and S3 | all three | S1 or S2 or S3 (of 24) |
|---|---|---|---|---|---|
| wiki | 4.96 (62%) | 4.38 (55%) | 3.86 (48%) | 3.18 | 13.98 |
| code | 4.55 (57%) | 4.62 (58%) | 3.57 (45%) | 3.02 | 14.28 |
| chat | 4.69 (59%) | 4.28 (54%) | 3.41 (43%) | 2.84 | 14.47 |
| mean | 4.73 (59%) | 4.43 (55%) | 3.61 (45%) | 3.01 | 14.24 |
| independent | 0.50 (6%) | 0.50 | 0.50 | 0.03 | 22.53 |

A token is routed to largely the same experts in every pass (14.2 distinct experts over three passes instead of 22.5), but the passes of one layer
are separated by the weights of 11 other core layers (>= 1.07 GB), so no cache holds them: every visit still streams its experts (ARCH.md S5).
Layer 17 (last core layer) is the exception (S1&S2 3.5, S2&S3 2.3, S1&S3 1.7); layers 8-16 overlap most.

### (c) Popularity skew: share of the selections taken by the 16 most used experts of a layer
Mean over the 24 layers (per visit, mean of the data sets): 49.8%; min 35.1% (layer 3), max 75.1% (layer 17); uniform routing would give 12.5%, and
the sampling noise of a random router with the same token count gives 14.3-15.2% (per visit) / 13.5-14.1% (three passes pooled). Layers 0-5: 31-51%;
core layers 6-16: 38-57%; layer 17: 72-77%; layers 18-23: 44-84%. Routing is strongly non-uniform, which is part of why batches overlap (a); the
hot 16 experts of all layers would need 16 x 4.48 MB x 20 + 16 x 8.46 MB x 4 = 1.97 GB, far beyond the 35.8 MiB L3. Full table: `tools/route_traces/route_stats_output.md`.

### (d) Expert bytes of a T-token verify step (44 NVFP4 visits at 4,478,984 B per expert + 4 Q8_0 visits at 8,460,288 B; MB = 10^6 B)
| T | NVFP4 visits MB | Q8_0 visits MB | experts MB (mean) | x one-token step | x per token | independent x | wiki x | code x | chat x |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1576.6 | 270.7 | 1847.3 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 2498.0 | 430.1 | 2928.1 | 1.59 | 0.793 | 1.94 | 1.57 | 1.61 | 1.57 |
| 3 | 3245.4 | 557.7 | 3803.1 | 2.06 | 0.686 | 2.82 | 2.05 | 2.09 | 2.04 |
| 4 | 3888.0 | 665.9 | 4553.8 | 2.47 | 0.616 | 3.64 | 2.46 | 2.50 | 2.44 |
| 5 | 4458.2 | 760.4 | 5218.7 | 2.82 | 0.565 | 4.41 | 2.82 | 2.86 | 2.79 |
| 6 | 4974.6 | 845.1 | 5819.7 | 3.15 | 0.525 | 5.14 | 3.14 | 3.20 | 3.11 |
| 7 | 5447.2 | 922.3 | 6369.6 | 3.45 | 0.493 | 5.82 | 3.44 | 3.51 | 3.40 |
| 8 | 5883.2 | 993.1 | 6876.3 | 3.72 | 0.465 | 6.45 | 3.71 | 3.80 | 3.66 |

The one-token row reproduces ARCH.md (NVFP4 visits 1576.6 MB); the independent column reproduces its x1.94 / x3.64 / x6.45. So a verify batch costs 1.59 /
2.47 / 3.72 times the expert bytes of one token for T = 2 / 4 / 8, instead of 1.94 / 3.64 / 6.45 under independent routing.

## 3. Break-even of speculative decoding on bandwidth grounds
Plain decoding reads B1 = N + E(1) per emitted token. One speculative step reads S(T) = N + E(T) + Dr and emits L tokens (L in [1, T]: accepted drafts + 1):
- N = bytes that are read once per step whatever T is: attention projections, lm_head, router, controllers, small tensors (KV/attention-score reads
  are context dependent, read once per step in both modes, and not included; they raise N and lower L* slightly).
- E(T) = measured expert bytes of T consecutive tokens (table (d), mean of the data sets).
- Dr = drafter bytes per step: DSpark drafter 635.8 MB (bf16; one pass drafts the whole 8-token block, so Dr does not depend on T) plus one lm_head pass of the
  target if the drafter uses the target head (ARCH.md S8). Both cases are shown.
- Speculation beats plain decoding on bandwidth iff S(T)/L < B1, i.e. **L > L\*(T) = S(T)/B1**. `max x` = T x B1 / S(T) is the speed-up if every draft is accepted
  (L = T): an upper bound, not an estimate. L\*(T) - 1 is the average number of accepted drafts per step needed.

Engine plan (F32 router/controllers stored as bf16, lm_head trimmed to 200058 rows, see docs/BASELINE.md): N = 2014.6 MB (lm_head 489.7 MB), E(1) = 1847.3 MB,
B1 = 3861.9 MB per token.

**Drafter + one lm_head pass (Dr = 1125.5 MB):**

| T | verify step N+E(T) MB | step S(T) MB | L\* independent routing | L\* wiki | L\* code | L\* chat | **L\* mean** | L\*/T | max x at L=T |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 4943 | 6068 | 1.74 | 1.57 | 1.58 | 1.56 | **1.57** | 0.79 | 1.27 |
| 3 | 5818 | 6943 | 2.16 | 1.79 | 1.81 | 1.79 | **1.80** | 0.60 | 1.67 |
| 4 | 6568 | 7694 | 2.55 | 1.99 | 2.01 | 1.98 | **1.99** | 0.50 | 2.01 |
| 5 | 7233 | 8359 | 2.92 | 2.16 | 2.18 | 2.15 | **2.16** | 0.43 | 2.31 |
| 6 | 7834 | 8960 | 3.27 | 2.32 | 2.34 | 2.30 | **2.32** | 0.39 | 2.59 |
| 7 | 8384 | 9510 | 3.60 | 2.46 | 2.49 | 2.44 | **2.46** | 0.35 | 2.84 |
| 8 | 8891 | 10016 | 3.90 | 2.59 | 2.63 | 2.56 | **2.59** | 0.32 | 3.08 |

**Drafter only, no lm_head pass (Dr = 635.8 MB):**

| T | step S(T) MB | L\* independent routing | L\* wiki | L\* code | L\* chat | **L\* mean** | L\*/T | max x at L=T |
|---|---|---|---|---|---|---|---|---|
| 2 | 5578 | 1.61 | 1.44 | 1.46 | 1.44 | **1.44** | 0.72 | 1.38 |
| 3 | 6453 | 2.03 | 1.67 | 1.68 | 1.66 | **1.67** | 0.56 | 1.80 |
| 4 | 7204 | 2.43 | 1.86 | 1.88 | 1.85 | **1.87** | 0.47 | 2.14 |
| 5 | 7869 | 2.80 | 2.04 | 2.06 | 2.02 | **2.04** | 0.41 | 2.45 |
| 6 | 8470 | 3.14 | 2.19 | 2.22 | 2.17 | **2.19** | 0.37 | 2.74 |
| 7 | 9020 | 3.47 | 2.33 | 2.36 | 2.31 | **2.34** | 0.33 | 3.00 |
| 8 | 9527 | 3.77 | 2.46 | 2.50 | 2.44 | **2.47** | 0.31 | 3.24 |

GGUF as stored (F32 router/controllers, lm_head 201024 rows; N = 2163.5 MB, B1 = 4010.8 MB): L\* mean with the lm_head pass is 1.55, 1.77, 1.96, 2.12, 2.27, 2.41,
2.54 for T = 2..8 (without it 1.43, 1.65, 1.83, 2.00, 2.15, 2.29, 2.41): 0.01-0.06 lower than the engine plan because a larger N dilutes the extra bytes.

## 4. Reading the numbers
- Routing correlation makes verification cheaper than the independent model assumed in ARCH.md: L\*(8) is 2.59 instead of 3.90, L\*(4) is 1.99 instead of 2.55.
  Experts are no longer the main cost of a step: at T=4 the verify step is 6568 MB (1.70 x B1) and the drafter side adds 1126 MB (0.29 x B1).
- The break-even is a modest ask in relative terms for larger T (L\*/T = 0.50 at T=4, 0.32 at T=8) but needs an average of at least 1.0 accepted drafts per step at T=4
  and 1.6 at T=8 (L\* - 1), and the 29% (T-independent) drafter+head overhead is what keeps small T unattractive (T=2 needs L > 1.57 although at most 2 tokens can be emitted).
- Whether DSpark delivers L above these thresholds on this Q4_K_XL GGUF is UNKNOWN: no acceptance data exists and none was invented. Measuring it needs the drafter
  GGUF (arch `dflash`, `--spec-type draft-dspark`) in llama.cpp; the thresholds above are what the measurement has to beat.
- Caveats: (1) routing was measured on TRUE continuations; rejected draft tokens are off-distribution and may touch more experts (the independent column bounds this
  from above: L\*(4) <= 2.55); (2) only bytes are modelled: with T up to 8 the NVFP4/Q8_0 kernels must stay bandwidth-bound on AVX-512 without VNNI (a rough estimate,
  not measured, puts the int8 peak at several TOPS against about 0.7 TFLOP/s needed at T=8 and 75 GB/s; agent B1's kernel benchmarks must confirm);
  (3) the traces cover 4.2k tokens of English/C++/chat, no long-context or non-English text; (4) no drafter hidden-state or Markov-chain extra reads beyond ARCH.md S8
  (its "up to 7 x 103 MB" per-position Markov re-reads would add up to 0.72 GB to Dr and are not in the tables).

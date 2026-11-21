# Baseline: exact bytes per token of this GGUF, and the stock llama.cpp numbers

Agent E, 2026-11-21. Source: the tensor table of `/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf` (744 tensors, 17,617,780,736 tensor bytes, 63 KV)
read by `tools/gguf_bytes.py` (also writes `tools/bytes_per_token.json`, used by `tools/route_stats.py`). MB = 10^6 bytes. Replaces the "~4.0 GB (estimate)" of ENGINE.md.

## 1. Rule used (decode, batch 1, every visit re-reads its weights from DRAM, no cache reuse)
- 48 visits: layers 0-5 once, core layers 6-17 three times (passes 1-3), layers 18-23 once. A class on all 24 layers therefore has 48 visits, the four global-attention
  classes (layers 5, 11, 17, 23) 8, the experts 44 NVFP4 visits (L0-19) + 4 Q8_0 visits (L20-23).
- Routed experts: 8 of 128 experts per visit, i.e. bytes x 8/128 of `ffn_gate_up_exps` and `ffn_down_exps` (plus the 8 selected entries of the per-layer F32 `.scale` vectors).
- Loop controllers (`loop_{q,k,v,ffn}_ctrl_{down,forget,gain}`) and `loop_msg_query`: passes 2 and 3 only, so 2 visits per core layer = 24 visits.
- Embedding: one row (2304/32 x 34 = 2448 B) per token. lm_head `output.weight` [2304 x 201024 padded rows] Q8_0: every row, once per token.
- Scenario A = tensors as stored. B = every F32 tensor whose values are exactly representable in bf16 is stored as bf16 at load (checked element by element: low 16 bits
  of every float are zero: router `ffn_gate_inp`, all controllers, convs, norms, clips, `attn_rel_bias`; NOT representable and kept F32: `*_res_query`, `exp_probs_b`,
  `loop_msg_query`, expert `.scale`, `output_res_query`: 1.9 MB per token). C = B with lm_head trimmed to the 200058 real vocabulary rows.
- Not included: KV-cache / attention-score reads (grow with context), activations and scratch, the vision tower and the drafter.

## 2. Bytes read per token by tensor class (sorted by A)
| tensor class (blk.N.<class>) | type | tensors | file MB | visits | read MB/token A (as stored) | B (F32->bf16) | C (B + lm_head 200058) | share of A |
|---|---|---|---|---|---|---|---|---|
| ffn_gate_up_exps.weight | NVFP4 | 20 | 7,644.12 | 44 | 1,051.07 | 1,051.07 | 1,051.07 | 26.21% |
| ffn_down_exps.weight | NVFP4 | 20 | 3,822.06 | 44 | 525.53 | 525.53 | 525.53 | 13.10% |
| output.weight | Q8_0 | 1 | 492.11 | 1 | 492.11 | 492.11 | 489.74 | 12.27% |
| attn_gate.weight | Q8_0 | 24 | 180.49 | 48 | 360.97 | 360.97 | 360.97 | 9.00% |
| attn_output.weight | Q8_0 | 24 | 180.49 | 48 | 360.97 | 360.97 | 360.97 | 9.00% |
| attn_q.weight | Q8_0 | 24 | 180.49 | 48 | 360.97 | 360.97 | 360.97 | 9.00% |
| ffn_gate_up_exps.weight | Q8_0 | 4 | 2,887.78 | 4 | 180.49 | 180.49 | 180.49 | 4.50% |
| attn_k.weight | Q8_0 | 24 | 60.16 | 48 | 120.32 | 120.32 | 120.32 | 3.00% |
| attn_v.weight | Q8_0 | 24 | 60.16 | 48 | 120.32 | 120.32 | 120.32 | 3.00% |
| ffn_down_exps.weight | Q8_0 | 4 | 1,443.89 | 4 | 90.24 | 90.24 | 90.24 | 2.25% |
| ffn_gate_inp.weight | F32 (24/24 bf16-exact) | 24 | 28.31 | 48 | 56.62 | 28.31 | 28.31 | 1.41% |
| attn_rel.weight | Q8_0 | 24 | 22.56 | 48 | 45.12 | 45.12 | 45.12 | 1.12% |
| loop_q_ctrl_down.weight | F32 (12/12 bf16-exact) | 12 | 14.16 | 24 | 28.31 | 14.16 | 14.16 | 0.71% |
| loop_k_ctrl_down.weight | F32 (12/12 bf16-exact) | 12 | 14.16 | 24 | 28.31 | 14.16 | 14.16 | 0.71% |
| loop_ffn_ctrl_down.weight | F32 (12/12 bf16-exact) | 12 | 14.16 | 24 | 28.31 | 14.16 | 14.16 | 0.71% |
| loop_v_ctrl_down.weight | F32 (12/12 bf16-exact) | 12 | 14.16 | 24 | 28.31 | 14.16 | 14.16 | 0.71% |
| loop_q_ctrl_forget.weight | F32 (12/12 bf16-exact) | 12 | 7.08 | 24 | 14.16 | 7.08 | 7.08 | 0.35% |
| loop_q_ctrl_gain.weight | F32 (12/12 bf16-exact) | 12 | 7.08 | 24 | 14.16 | 7.08 | 7.08 | 0.35% |
| loop_k_ctrl_forget.weight | F32 (12/12 bf16-exact) | 12 | 7.08 | 24 | 14.16 | 7.08 | 7.08 | 0.35% |
| loop_k_ctrl_gain.weight | F32 (12/12 bf16-exact) | 12 | 7.08 | 24 | 14.16 | 7.08 | 7.08 | 0.35% |
| loop_ffn_ctrl_forget.weight | F32 (12/12 bf16-exact) | 12 | 7.08 | 24 | 14.16 | 7.08 | 7.08 | 0.35% |
| loop_ffn_ctrl_gain.weight | F32 (12/12 bf16-exact) | 12 | 7.08 | 24 | 14.16 | 7.08 | 7.08 | 0.35% |
| loop_v_ctrl_forget.weight | F32 (12/12 bf16-exact) | 12 | 7.08 | 24 | 14.16 | 7.08 | 7.08 | 0.35% |
| loop_v_ctrl_gain.weight | F32 (12/12 bf16-exact) | 12 | 7.08 | 24 | 14.16 | 7.08 | 7.08 | 0.35% |
| attn_sink_proj.weight | BF16 | 24 | 2.65 | 48 | 5.31 | 5.31 | 5.31 | 0.13% |
| attn_rel_bias.weight | F32 (24/24 bf16-exact) | 24 | 1.57 | 48 | 3.15 | 1.57 | 1.57 | 0.08% |
| attn_post_conv.weight | F32 (24/24 bf16-exact) | 24 | 0.88 | 48 | 1.77 | 0.88 | 0.88 | 0.04% |
| ffn_post_conv.weight | F32 (24/24 bf16-exact) | 24 | 0.88 | 48 | 1.77 | 0.88 | 0.88 | 0.04% |
| attn_k_conv_proj.weight | BF16 | 4 | 0.59 | 8 | 1.18 | 1.18 | 1.18 | 0.03% |
| attn_v_conv_proj.weight | BF16 | 4 | 0.59 | 8 | 1.18 | 1.18 | 1.18 | 0.03% |
| loop_msg_query.weight | F32 (0/12 bf16-exact) | 12 | 0.44 | 24 | 0.88 | 0.88 | 0.88 | 0.02% |
| attn_k_conv.weight | F32 (24/24 bf16-exact) | 24 | 0.39 | 48 | 0.79 | 0.39 | 0.39 | 0.02% |
| attn_v_conv.weight | F32 (24/24 bf16-exact) | 24 | 0.39 | 48 | 0.79 | 0.39 | 0.39 | 0.02% |
| attn_q_clip.weight | F32 (24/24 bf16-exact) | 24 | 0.29 | 48 | 0.59 | 0.29 | 0.29 | 0.01% |
| attn_res_query.weight | F32 (0/23 bf16-exact) | 23 | 0.29 | 47 | 0.58 | 0.58 | 0.58 | 0.01% |
| ffn_norm.weight | F32 (24/24 bf16-exact) | 24 | 0.22 | 48 | 0.44 | 0.22 | 0.22 | 0.01% |
| ffn_res_query.weight | F32 (0/24 bf16-exact) | 24 | 0.22 | 48 | 0.44 | 0.44 | 0.44 | 0.01% |
| attn_norm.weight | F32 (20/20 bf16-exact) | 20 | 0.18 | 40 | 0.37 | 0.18 | 0.18 | 0.01% |
| attn_rel_clip.weight | F32 (24/24 bf16-exact) | 24 | 0.04 | 48 | 0.07 | 0.04 | 0.04 | 0.00% |
| attn_k_inp_norm.weight | F32 (4/4 bf16-exact) | 4 | 0.04 | 8 | 0.07 | 0.04 | 0.04 | 0.00% |
| attn_q_inp_norm.weight | F32 (4/4 bf16-exact) | 4 | 0.04 | 8 | 0.07 | 0.04 | 0.04 | 0.00% |
| attn_v_inp_norm.weight | F32 (4/4 bf16-exact) | 4 | 0.04 | 8 | 0.07 | 0.04 | 0.04 | 0.00% |
| exp_probs_b.bias | F32 (0/24 bf16-exact) | 24 | 0.01 | 48 | 0.02 | 0.02 | 0.02 | 0.00% |
| output_norm.weight | F32 (1/1 bf16-exact) | 1 | 0.01 | 1 | 0.01 | 0.00 | 0.00 | 0.00% |
| output_res_query.weight | F32 (0/1 bf16-exact) | 1 | 0.01 | 1 | 0.01 | 0.01 | 0.01 | 0.00% |
| token_embd_norm.weight | F32 (1/1 bf16-exact) | 1 | 0.01 | 1 | 0.01 | 0.00 | 0.00 | 0.00% |
| token_embd.weight | Q8_0 | 1 | 492.11 | 1 | 0.00 | 0.00 | 0.00 | 0.00% |
| ffn_down_exps.scale | F32 (0/20 bf16-exact) | 20 | 0.01 | 44 | 0.00 | 0.00 | 0.00 | 0.00% |
| ffn_gate_up_exps.scale | F32 (0/20 bf16-exact) | 20 | 0.01 | 44 | 0.00 | 0.00 | 0.00 | 0.00% |
| **total** | | 744 | 17,617.78 | | **4,010.82** | **3,864.28** | **3,861.92** | 100% |

| group | MB/token A | B | C | share of A |
|---|---|---|---|---|
| routed experts NVFP4 (L0-19) | 1,576.60 | 1,576.60 | 1,576.60 | 39.31% |
| attention projections (Q8_0) | 1,368.69 | 1,368.69 | 1,368.69 | 34.12% |
| lm_head (output.weight, Q8_0) | 492.11 | 492.11 | 489.74 | 12.27% |
| routed experts Q8_0 (L20-23) | 270.73 | 270.73 | 270.73 | 6.75% |
| loop controllers + msg queries (F32, passes 2-3) | 227.38 | 114.13 | 114.13 | 5.67% |
| router (ffn_gate_inp F32 + exp_probs_b F32) | 56.65 | 28.34 | 28.34 | 1.41% |
| small F32: convs, norms, clips, rel_bias, queries | 11.00 | 6.02 | 6.02 | 0.27% |
| attention BF16 (sink_proj, dyn-conv proj) | 7.67 | 7.67 | 7.67 | 0.19% |
| expert .scale F32 (selected entries) | 0.00 | 0.00 | 0.00 | 0.00% |
| embedding row (token_embd.weight, 1 row) | 0.00 | 0.00 | 0.00 | 0.00% |
| **total** | **4,010.82** | **3,864.28** | **3,861.92** | 100% |

per expert (gate_up + down + 2 scale entries): NVFP4 [4478984] B, Q8_0 [8460288] B; expert layers: 0:NVFP4, 1:NVFP4, 2:NVFP4, 3:NVFP4, 4:NVFP4, 5:NVFP4, 6:NVFP4, 7:NVFP4, 8:NVFP4, 9:NVFP4, 10:NVFP4, 11:NVFP4, 12:NVFP4, 13:NVFP4, 14:NVFP4, 15:NVFP4, 16:NVFP4, 17:NVFP4, 18:NVFP4, 19:NVFP4, 20:Q8_0, 21:Q8_0, 22:Q8_0, 23:Q8_0
F32 classes with at least one tensor that is NOT exactly bf16-representable (kept F32 in B and C): ['attn_res_query.weight', 'exp_probs_b.bias', 'ffn_down_exps.scale', 'ffn_gate_up_exps.scale', 'ffn_res_query.weight', 'loop_msg_query.weight', 'output_res_query.weight']; F32 bytes in file 147.54 MB
sum of tensor bytes 17,617,780,736 (file size minus header/alignment gives the rest)

## 3. What the conversions buy
- As stored (A): **4,010.8 MB per token** (ENGINE.md's estimate was 4.0 GB). Experts 1,847.3 MB (46.1%: NVFP4 1,576.6 + Q8_0 270.7), attention projections (Q8_0, 6 matrices) 1,368.7 MB (34.1%),
  lm_head 492.1 MB (12.3%), loop controllers 227.4 MB (5.7%, F32 read twice per token), router 56.7 MB (1.4%), small F32 tensors 11.0 MB, attention BF16 tensors 7.7 MB.
- F32 -> bf16 at load (B): **3,864.3 MB** = -146.5 MB (-3.65%): controllers 227.4 -> 114.1 MB, router 56.7 -> 28.3 MB, small F32 11.0 -> 6.0 MB. Lossless (bit-exact bf16 values).
- lm_head trimmed to 200058 rows (C): **3,861.9 MB** = a further -2.4 MB (966 padded rows x 2448 B); total -3.71% vs A. The padded rows only cost bytes in llama.cpp; they carry no logits.
- At the measured 92.5 GB/s streaming ceiling C allows 24.0 tok/s (A: 23.1); the targets of CONTRACT.md need 44.4 GB/s (11.5 tok/s) and 65.6 GB/s (17 tok/s) of useful traffic
  on the C byte count. ARCH.md's "as shipped" bf16 checkpoint figure (5,497 MB) is 37% larger than A: the community GGUF stores attention, lm_head and the L20-23 experts as Q8_0
  (8.46 MB per expert instead of 8.21 MB MXFP8) and the L0-19 experts as NVFP4 (4.48 MB per expert, byte-identical to the checkpoint).
- Non-expert bytes N per token (what a speculative verify step reads once whatever T is): A 2,163.5 MB, B 2,016.9 MB, C 2,014.6 MB; expert bytes E(1) = 1,847.3 MB (docs/ROUTING.md).

## 4. Stock llama.cpp baseline (`tools/bench_llama.sh`, build-sys, llama.cpp @2a28df0f9)
Command per row: `OMP_WAIT_POLICY=active OMP_PLACES=cores OMP_PROC_BIND=close build-sys/bin/llama-bench -m <gguf> -p 0 -n 64 | -p 256 -n 0 -t <threads> -r 3 -o csv` (default `REPS=3`; the indicative run below used `REPS=2`)
(decode / prefill; one invocation per row, rows in `tools/bench_baseline.csv`). "Effective GB/s" = decode tok/s x 4.0108 GB (scenario A bytes), "% ceiling" is relative to 92.5 GB/s.
The host is shared: `load1` is the 1-minute loadavg before the run, `other CPUs busy` the average number of CPUs used by processes other than the benchmark during the run
(from /proc/stat), so a clean run shows load1 <= 2 and other CPUs busy ~0.

**Run conditions (read first).** The first, clean attempt (2026-11-21 21:05:46 - 21:15:46 UTC) waited the full 600 s for the 1-minute load average to fall to 2 or below and refused to start: the load stayed between 3.1 and 9.4 (samples 21:08-21:15) because other agents were running (a `llama-server -t 4`, 8-thread kernel/model comparisons) after my own oracle and trace runs. The numbers below were therefore taken at once with `MAX_LOAD=100 REPS=2 tools/bench_llama.sh` and are **INDICATIVE ONLY**: they include contention from those processes (see the load columns and `other CPUs busy`), and the run itself loads 8-16 cores, which disturbs other agents' timings. Re-run `tools/bench_llama.sh` (default MAX_LOAD 2, REPS 3) on an idle host for a clean baseline. The earlier matrix of ENGINE.md (tg32 5.36 / 5.71 / 4.80 tok/s at 8 / 12 / 16 threads, load 4.7) is of the same indicative kind.

| threads | decode tok/s (-p 0 -n 64) | effective GB/s | % of 92.5 GB/s | prefill tok/s (-p 256 -n 0) | load1 before (decode / prefill) | other CPUs busy (decode / prefill) | date |
|---|---|---|---|---|---|---|---|
| 8 | 5.26 +- 0.11 | 21.1 | 23% | 13.4 +- 0.0 | 2.54 / 4.40 | 0.5 / 3.2 | 2026-11-21T21:17:29 |
| 12 | 4.40 +- 0.16 | 17.7 | 19% | 17.1 +- 0.8 | 7.45 / 12.34 | 7.6 / 6.3 | 2026-11-21T21:19:04 |
| 16 | 3.99 +- 0.02 | 16.0 | 17% | 18.8 +- 0.1 | 15.52 / 18.40 | 7.3 / 7.8 | 2026-11-21T21:20:29 |

Reading the rows: only the 8-thread decode row had little foreign load (load1 2.5, other processes kept 0.5 CPUs busy): **5.26 +- 0.11 tok/s = 21.1 GB/s effective, 23% of the 92.5 GB/s ceiling**, in line with ENGINE.md's 5.2-5.7 tok/s. Every other row ran with 3-8 foreign CPUs busy (and the 16-thread run adds its own load), so the lower decode numbers at 12 and 16 threads and the prefill numbers (13.4 / 17.1 / 18.8 tok/s at 8 / 12 / 16 threads; ENGINE.md: 23 tok/s at 16 threads) most likely reflect contention, not thread scaling, and must not be used as a scaling result. Raw rows with the exact commands: `tools/bench_baseline.csv`.

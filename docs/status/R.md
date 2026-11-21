# R status (DSpark drafter specification + acceptance measurement)

Started 2026-11-21 ~21:43 UTC. Role: agent R. Deliverables: docs/DSPARK.md (spec + measured acceptance + speed-up estimates), tools/dspark/ (raw outputs, scripts).

## Plan
1. Build llama.cpp with the Inkling 2 DSpark drafter CPU-only in /data/inkling-mini/llama-unsloth (scripts tools/dspark/build_dspark.sh). The stock Unsloth tag b12588-mix-c0c3de8 has NO Inkling code (Unsloth merges ggml-org PR 32218 + others at CI time), so that directory is now upstream PR 32218 head 0a9105de (= upstream master + 1 commit "llama : implement Inkling2 DSpark block drafting", already containing inkling2_loop). /data/inkling-mini/llama.cpp is untouched (read only).
2. Drafter GGUF dspark-Inkling-2-Mini-Q8_0.gguf (402,278,560 B) downloaded once to /data/inkling-mini/models/.
3. Spec research: llama.cpp source read directly (dspark.cpp, dflash.cpp, speculative.cpp, sampling.cpp); SGLang / vLLM / paper+cards delegated to 3 read-only research subagents (outputs under tools/dspark/refs/).
4. Measurement: 8 prompts x draft n in {3,4,7} x {greedy, temp 1.0 (drafter-sampled + rejection, and drafter-argmax + match)}, 80 new tokens, per-step accepted counts logged by an instrumented llama-speculative-simple (tools/dspark/patch_speculative_simple.py).
5. docs/DSPARK.md with tables and ROUTING.md-based speed-up estimates; recommendation.

## Progress
- 21:44 drafter GGUF downloaded; GGUF header dump in tools/dspark/drafter_gguf_dump.txt (target_layers [5,23,41,48], block_size 8, markov_rank 256, sample_from_anchor False, has_confidence_head False, causal False, window 1024 pattern [T,F,T]).
- 21:51 build OK (llama-server, llama-cli, llama-speculative-simple). 21:53 smoke test OK (coherent text; greedy n=3, 25 tokens: 8 of 44 drafted tokens accepted in the very first sample, 2.8 tok/s under load 0.7-4).
- 21:56 acceptance matrix launched in the background (tools/dspark/run_matrix.sh, results in tools/dspark/raw/, marker raw/MATRIX_DONE).

## Findings so far (details go to docs/DSPARK.md)
- llama.cpp Inkling drafter: always runs the full 8-slot block [anchor, 7 x mask_embd]; draft length n only truncates the Markov chain (n_outputs = n); slot 0 is never sampled.
- The per-position head runs the target lm_head twice per drafted position (base + correction) in llama.cpp (2 x 492 MB Q8_0 per position); algebraically one pass over lm_head(h + correction) or a precomputed [vocab x 256] matrix (103 MB bf16) suffices. This matters for the bytes model: ROUTING.md assumes ONE lm_head pass per block.

## Update 22:07
- All three research subagents reported (SGLang, vLLM, paper+cards); llama.cpp PR 32218 source and vLLM drafter source read directly. The three implementations agree on the graph: fuse = softmax(fuse_weight rows) over the 4 taps + RMSNorm(dflare_norm); ctx K/V = Wk_ctx/Wv_ctx on the fused feature (k_norm_ctx + RoPE on K); block = [embed_norm(embed(anchor)), 7 x mask_embed], 3 pre-norm layers, bidirectional in the block, window 1024 before the block on layers 0 and 2; RNN head z=[s; markov_w1[prev]; hid; f_rnn] -> gate/cand/out; logits = 30*tanh((lm_head(hid)+lm_head(hidden_out(lat)))/30). Verification: n+1 rows, greedy match or rejection sampling (accept iff u*q<p), bonus token; no confidence head in the Mini checkpoint; verify width fixed.
- docs/DSPARK.md written (spec complete; results/speed/implication placeholders are filled by tools/dspark/assemble_dspark_md.py after the runs, implication by hand).
- Matrix running (greedy n=3 done; order greedy 3,7,4 then t1prob 3,7,4 then t1match 3,7); post_matrix.sh stops the runner at 22:31 UTC at the latest, then takes lock-protected speed runs (chat_answer, code_completion x n in 0,1,3,7) and assembles the doc.
- Finding: llama.cpp runs two full lm_head matvecs per chained position (drafting ~100 ms/step at n=3 here); folding lm_head x hidden_out into a [200058x256] matrix (vLLM/SGLang do this) costs 102 MB per chained position instead of 2 x 492 MB. ROUTING.md's Dr (1125.5 MB, independent of n) misses the n sequential passes: refined Dr(n) = 523.8 + 9.1 + 489.7 + 102.4 n MB.

## Update (final, agent R)
- Done: docs/DSPARK.md (spec of the drafter + verification, 3 independent implementations cross-checked, measured acceptance, speed-up estimates, recommendation), tools/dspark/ (scripts, prompts, raw step logs, refs/ sources).
- Measured (greedy, 8 prompts x 80 tokens, llama.cpp PR 32218 + Q4_K_XL target + Q8_0 drafter): mean tokens per verify step L = 2.37 +- 0.08 (n=3, T=4), 2.57 +- 0.10 (n=4, T=5), 2.77 +- 0.15 (n=7, T=8); acc@1 0.58-0.64. Per prompt: code completion 3.7/4.3/5.9, tool call 3.1/3.8/5.1, chat/prose/email 1.7-1.85 for every n. ROUTING break-even (refined Dr): L* = 2.05 / 2.24 / 2.75, so pooled speed-up estimate 1.16x / 1.15x / 1.00x (M), 1.22x / 1.21x / 1.06x (Q).
- Recommendation: build as an ADAPTIVE mode (T in {1, 4, 8} chosen from running acceptance), with the lm_head x hidden_out fold; fixed T=4 pooled +16-22% (loses 13-18% on chat/prose), fixed T=8 only pays on code/tool calls (~2x). llama.cpp's own implementation is slower than plain decoding (two lm_head matvecs per chained position).
- Background job tools/dspark/post_matrix.sh finishes at ~22:33 UTC (marker raw/ALL_DONE): it stops the temperature-1.0 matrix at 22:25, takes lock-protected speed runs (chat_answer, code_completion x n=0,3,7) and re-assembles docs/DSPARK.md (section 9.2/9.3 tables and the t1prob numbers update automatically; implication text is templated).
- Not measured (call/time budget): temperature 1.0 at n=4, drafter-argmax+match mode (t1match), full temperature-1.0 n=7 (partial), repeated seeds.

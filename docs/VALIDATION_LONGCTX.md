# Long-context parity vs llama.cpp (orchestrator, 2026-11-21 23:48-00:05 UTC)

Question (from agent P's report): on a 2200-token prompt made of repeated text, both the decode path and the batched prefill path scored cosine 0.977 / 0.965 against llama.cpp
(1305 tokens: 0.9994, 1500 tokens: 0.992), while all short prompts score >= 0.999. Is that a long-distance semantic bug (distances >= 1024 use the zero bias row on global layers,
the SWA ring wraps at 2048 positions) or numerical noise on a degenerate prompt?

Method: two natural wikitext-2 test regions (characters 150000 and 700000), tokenised with the server's tokenizer to exactly 2400 tokens each (`build-orch/longctx/lc1.txt`, `lc2.txt`),
which crosses the 1024-token window of the 20 SWA layers and the 2048-position wrap of the SWA ring. Reference: stock llama.cpp `tools/build/oracle_logits --all-logits` (whole prompt = one batch,
8 threads, `OMP_WAIT_POLICY=active`, 4-5 min per prompt). Engine: `build-orch/src/engine/test_prefill nll` = `engine_prefill_all` (chunk 256, fast attention), 6 threads, mean NLL of the next token
over all 2399 positions, argmax and per-position cosine against the llama.cpp logits. Host shared (load 13-15), numbers are numerics, not timings.

| prompt | positions | NLL engine | NLL llama.cpp | abs delta | argmax mismatches | worst per-position cosine |
|---|---|---|---|---|---|---|
| lc1 | 2399 | 2.03607 | 2.04003 | 0.00396 | 51 / 2400 (2.1%) | 0.890 |
| lc2 | 2399 | 2.03566 | 2.04041 | 0.00474 | 46 / 2400 (1.9%) | 0.901 |

Gate: |delta NLL| <= 0.01 nats (same as gate (d) of the 256-token chunks, where the deltas were 0.0016 / 0.0068 and 1-2% argmax mismatches): PASS on both.
Reading: over 2400 real tokens, 1376 of them beyond the 1024-distance horizon and 352 beyond the ring wrap, the engine's likelihood of the true text equals llama.cpp's within 0.005 nats (the engine's
is marginally LOWER). The ~2% argmax flips are near-ties (the same rate as at 256 tokens). The 0.97 cosine of the repeated-text prompt is therefore numerical sensitivity of a degenerate, extremely peaked
prompt, not a semantic difference; there is no evidence of a long-distance bug. Not covered: contexts beyond ~2700 tokens, tau > 1 (positions >= 8191).

Reproduce: `tools/build/oracle_logits -m GGUF --tokens-file build-orch/longctx/lc1.txt --all-logits --threads 8 --out build-orch/longctx/lc1` (1.9 GB logits per prompt), then
`INK_THREADS=6 build-orch/src/engine/test_prefill nll build-orch/longctx/lc1 build-orch/longctx/lc2` (data is not committed: build-* is ignored).

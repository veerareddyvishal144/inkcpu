# E status (oracle tools, routing study, benchmark harness)

Started 2026-11-21 20:46 UTC. Load while working: 1-min loadavg 0.4-2 at the start, 4-15 later (other agents: llama-server -t 4, kernel tests, my own 8-thread runs).

## Milestone 1 DONE (oracle_logits): data set ready for agent A
- Build: `cmake -S tools -B tools/build && cmake --build tools/build` (PATH=/usr/bin:/bin) -> `tools/build/oracle_logits`, `tools/build/oracle_greedy`.
- Data: `tools/oracle/<name>/{tokens.i32,logits_last.f32,top20.txt,meta.json,prompt.txt}` for s1_en_capital (13 tok, + logits_all.f32), s2_code_add (14),
  s3_en_story (15), m1_en_llm (43), m2_code_c (52), c1_chat_user (14, chat markers). Formats: `tools/README.md`.
- Facts A needs: logits width 200058 (padded rows already dropped, after the 30*tanh(x/30) softcap); this GGUF adds NO BOS/EOS (add_bos_token=false, no BOS id) so token 0
  is the first text token; special markers in `--prompt` text parse to single ids; whole prompt = one ubatch. Sanity: "...capital of Germany is" -> top-1 " Berlin" (25.89),
  chat prompt -> `<|content_thinking|>` (27.21).
- llama.cpp prints `special_eos_id is not in special_eog_ids` at load: it does not treat 200006 as EOG for this GGUF (servers stop on 200006 themselves).

## Milestone 2 DONE (oracle_greedy): `tools/oracle/greedy/<name>/` for s1, s2, m1, m2, c1: 64 greedy tokens each (prompt_tokens.i32, gen_tokens.i32, gen_text.txt, steps.txt with
top1/top2 margins). 8 threads, load 3-7: 5.1-5.5 tok/s decode (llama.cpp, OMP active), 1.2-4.1 s prompt.

## Milestone 3 DONE (routing study): docs/ROUTING.md, tools/route_stats.py, traces in tools/route_traces/ (wiki 4x512, code 1255, chat 895 tokens). Key numbers (mean of 3 data sets):
distinct experts per visit for T = 2/4/8 consecutive tokens: 12.7 / 19.7 / 29.8 (independent 15.5 / 29.1 / 51.6, ratios 0.82 / 0.68 / 0.58); same-token overlap between passes
S1&S2 4.7 of 8 (independent 0.5); top-16 experts take 50% of selections per layer (uniform 12.5%); verify-step expert bytes x1.59 / x2.47 / x3.72 of a one-token step.
Break-even (engine plan bytes, drafter 636 MB + lm_head pass): L*(T) = 1.57, 1.80, 1.99, 2.16, 2.32, 2.46, 2.59 for T = 2..8 (no acceptance data exists, none assumed).
llama.cpp patch: GGML_ROUTE_TRACE in ggml/src/ggml-cpu/ggml-cpu.c, built in build-nprof (build-sys untouched).

## Milestone 4: bytes per token from the GGUF tensor table: A as stored 4,010.8 MB, B (F32->bf16) 3,864.3 MB, C (+ lm_head 200058 rows) 3,861.9 MB (tools/gguf_bytes.py; docs/BASELINE.md section 2-3).


## Status
- [x] 1 oracle_logits + data + README
- [x] 2 oracle_greedy data
- [x] 3 routing study
- [x] 4 BASELINE bytes table
- [x] 5 bench_llama.sh run recorded in docs/BASELINE.md (INDICATIVE: host load 3-9, see BASELINE.md section 4)

## Milestone 5 DONE: tools/bench_llama.sh. The clean attempt waited 600 s (load stayed 3.1-9.4) and refused; the forced run (MAX_LOAD=100, REPS=2) is INDICATIVE ONLY:

| threads | decode tok/s (-p 0 -n 64) | effective GB/s | % of 92.5 GB/s | prefill tok/s (-p 256 -n 0) | load1 before (decode / prefill) | other CPUs busy (decode / prefill) | date |
|---|---|---|---|---|---|---|---|
| 8 | 5.26 +- 0.11 | 21.1 | 23% | 13.4 +- 0.0 | 2.54 / 4.40 | 0.5 / 3.2 | 2026-11-21T21:17:29 |
| 12 | 4.40 +- 0.16 | 17.7 | 19% | 17.1 +- 0.8 | 7.45 / 12.34 | 7.6 / 6.3 | 2026-11-21T21:19:04 |
| 16 | 3.99 +- 0.02 | 16.0 | 17% | 18.8 +- 0.1 | 15.52 / 18.40 | 7.3 / 7.8 | 2026-11-21T21:20:29 |

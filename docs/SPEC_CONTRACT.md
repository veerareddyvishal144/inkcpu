# Wave 3 contract: speculative decoding (S1, S2) and server integration (V)

Written by the orchestrator 2026-11-21. Read docs/CONTRACT.md first (rules still apply: no git by agents, no `pkill -f`, no personal data in external requests,
shared host => timings are indicative only). Facts and the drafter specification: docs/DSPARK.md (sections 2-8 are the spec, 9-10 the measured acceptance and the policy
recommendation). Engine facts: docs/ENGINE_DESIGN.md (decode, C1), docs/PREFILL_DESIGN.md (batched executor, P).

## Goal
Adaptive speculative decoding on this CPU: verify width T in {1, 4, 8} (T = drafted tokens + 1), chosen online from the running acceptance (DSPARK.md section 10).
Measured on this box (llama.cpp PR build, greedy): mean tokens per verify step L = 2.37 (T=4), 2.77 (T=8) pooled; code completion 3.7 / 5.9, tool calls 3.1 / 5.1, chat/prose 1.7-1.85 for every T.
Break-even L* about 2.0 (T=4) / 2.6-2.75 (T=8) with the refined drafter cost. Expected: ~1.2x pooled, ~1.5-2x on code and tool-call generation (the agentic use case), 1.0x on chat (policy falls back to plain decode).
The stock llama.cpp implementation is SLOWER than plain decode (2 lm_head matvecs per chained position); the gain exists only with the folded matrix M = lm_head . hidden_out (DSPARK.md section 8).

## Hard correctness property (S2 gate)
GREEDY speculation must produce EXACTLY the tokens of plain greedy decode (same engine, same prompt): the verify path must use the DECODE attention kernels
(INK_PF_ATTN=decode semantics: P's dense GEMMs are bit-identical to the decode GEMVs) so the logits of every verify row equal the plain-decode logits bit for bit, and rollback must restore the exact state.
Check on >= 8 prompts x 200 tokens for T in {4, 8} and for the adaptive policy; any mismatch is a bug, not noise.
Sampled mode (temperature > 0) must be distribution-preserving (Leviathan/Chen rejection sampling, DSPARK.md 6.3); test statistically on a small number of steps.

## Ownership (no agent edits a file it does not own; ask via docs/requests.md)
| agent | owns | may call (never edit) |
|---|---|---|
| S1 drafter | `include/ink/drafter.h` (extend only), `src/drafter/**` (own CMakeLists.txt, target `ink_drafter`), `tests/drafter/**`, `bench/bench_drafter.cpp`, `docs/DRAFTER.md`, `docs/status/S1.md` | kernels (gemv/gemm/misc), Engine public state, ThreadPool |
| S2 verify+policy | `include/ink/spec.h`, `src/spec/**` (target `ink_spec`), `tests/spec/**`, `bench/bench_spec.cpp`, `docs/SPEC_DESIGN.md`, `docs/status/S2.md`, AND the engine files: `engine.h`, `engine_priv.h`, `engine_decode.cpp`, `engine_prefill.cpp`, `engine_prefill.h`, `engine_load.cpp`, `src/capi/*`, `include/ink/ink_abi.h` (additions only; C1 and P are finished) | drafter.h (read only; S1 extends) |
| V server | `server/**`, `tests/server/**` (if any), `docs/status/V.md` | libink.so via the C ABI only |
The root `CMakeLists.txt` module loop is edited by the orchestrator (adds `drafter`, `spec`) before the agents start; each module has its own CMakeLists.txt.

## Interface S1 -> S2
`include/ink/drafter.h` (already written by the orchestrator; S1 extends if needed). Summary: `Drafter::create(Engine&, gguf, &err)`; per-sequence `state_new/free/reset/pos`;
`commit(st, taps, rows)` appends committed positions with their tap rows (F32 `[rows][4][2304]`); `draft(st, DraftRequest{anchor, n, sample, temperature, seed}, out_tokens, q_out)` -> n tokens
(and the dense q rows when sampling). All parallel work runs on the target's ThreadPool.
Taps: DSPARK.md 4.1: the stream entering visits 5, 23, 41 (running partial-sum of the current attention-residual block, NOT the depth-mixed input) and the final normalised hidden state (tap 48).
S2 captures them in the decode AND the batched executors (+ a callback/ring so the loop can commit exactly the accepted rows); S1 does not touch engine files.
S2 milestone 1 (target: first ~40 min): tap capture working in both executors + a tool that dumps the taps of a prompt to a file, so S1 can test against real taps. S1 milestone 1: drafter forward (block pass + chain) on the dumped taps.

## Interface S2 -> V (C ABI additions in `ink_abi.h`, S2 owns)
- `ink_spec_load(engine, drafter_gguf_path)`, `ink_spec_enable(engine, mode, ...)`.
- `ink_spec_step(engine, slot, request, result)` with a request carrying the anchor token, sampler params (temperature, top_k, top_p, min_p, seed; temperature 0 = greedy), a list of stop token ids and a max-emit budget;
  the result carries the emitted tokens (>= 1), counters (drafted, accepted, T used, verify/draft microseconds) and the NEXT anchor (the last emitted token, not yet processed by the target).
- If a stop token appears among the emitted tokens the engine state must end exactly as if the tokens BEFORE the stop token (and only those) had been processed (as plain decode does: the stop token is never fed).
- Plain decode steps (policy chose T=1) must keep the drafter's context in sync (taps are captured for every processed row; committed lazily in batches).
- Requests with penalties, logit_bias or grammar constraints cannot use speculation: the server falls back to plain `ink_forward` for those (V decides).

## CPU partition while the three agents run (spin-barrier pools convoy when they share CPUs: 4-5 tok/s instead of 13)
Even CPU ids are distinct physical cores (2k and 2k+1 are hyperthread siblings; core 0 runs at half speed, avoid it).
S1: `INK_CPUS=2,4,6,8` (4 threads). S2: `INK_CPUS=10,12,14,16,18,20` (6 threads). V: `INK_CPUS=22,24,26,28,30` (5 threads). Honour `INK_CPUS` / `EngineOptions::cpus` in every test and tool you run.
Timing numbers taken with fewer threads or next to other agents are for A/B comparisons inside your own session only. The orchestrator takes the final numbers on an idle host.
Do not start any engine/llama.cpp process before `docs/GO_RUN` exists (the orchestrator is taking clean benchmarks; until then read, design, write code; you may compile with `nice -n 19 ninja -j2`).

## Memory
Engine resident ~17 GB; drafter +0.4-0.7 GB (+0.1 GB folded matrix); each extra concurrent engine process doubles that: never run more than two at once per agent (the box has 125 GB, other tenants use ~10).

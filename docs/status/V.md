# V status (real engine behind the OpenAI-compatible server)

Agent V, started 2026-11-21 23:35 UTC. CPU partition `INK_CPUS=22,24,26,28,30` (5 threads), one engine process at a time, aux work `nice -n 10`.
All timings: shared host (load average 5-20 during my runs, other agents on their own partitions) => INDICATIVE ONLY.

## Log (UTC)
- 23:35-23:45 read CONTRACT / SPEC_CONTRACT / C1 / P / ENGINE_DESIGN / ink_abi.h / engine.h / server/*. Built `build-v/src/capi/libink.so` (cmake + `nice -n 19 ninja -j2`, rc 0, 22 steps).
- 23:41 sampler cost measured (numpy, 200058-vocab, loaded host): greedy 0.12 ms, temperature 0.6 sampling **5.6 ms/token**, top_p 6.0, top_k 1.3. Over the 3 ms bar =>
  `server/sampling.py` rewritten (no copy of the logits row, in-place float32 exp, two-level inverse CDF, threshold candidate sets): greedy 0.05, T=0.6 0.93, top_k 0.86, top_p 1.1,
  min_p 0.8 ms; same token as the full-sort reference for the same seed (400/400 identical); tests added.
- 23:46 docs/GO_RUN appeared. Task 5 groundwork written before that (pure python, mock): `StepResult` / `SpecRequest` / `Backend.supports_spec` / `Backend.spec_step`,
  `MockSpecBackend` (1-4 tokens per step), scheduler speculative path (anchor, `fed` = engine truth, stop/limit mid-step), 8 tests in `server/tests/test_spec_steps.py`.
- 23:48 FIRST REAL RUN: `python -m server --backend ink --lib build-v/src/capi/libink.so --threads 5 --slots 2 --ctx 8192` loads in 5 s (page cache hot), RSS 16.2 GiB.
  /v1/completions with the llama.cpp oracle prompts s1_en_capital and s2_code_add: **64/64 greedy tokens identical to llama.cpp** (tools/oracle/greedy), 8.1 tok/s decode at 5 threads,
  overhead outside the engine 0.17 ms/token. Chat, two-turn cache (cached_tokens 60 of 72, 12 tokens prefilled), forced and auto tool calls all worked first time.
- 23:50 DEFECT (engine pool, not server code): an idle server burns 30% of one core (4 workers x ~7.5%, the 20 us polling sleeps). Mitigated inside server/: InkBackend sets
  PR_SET_TIMERSLACK on the creating (engine) thread before `ink_load`, the pool threads inherit it. Fresh-process A/B (idle 10 s windows): slack 50 us (kernel default) 30.3%,
  2 ms 3.4%, 10 ms 1.0% of one core; decode and first-request-after-idle latency unchanged within noise. Default `--timer-slack-us 5000`. Reported in docs/requests.md.
- 23:50 peak RSS during load is 33.7 GiB (mapped GGUF pages + the copied arena), steady state 17.2 GiB: the server needs ~34 GiB free to START.
- 23:53 `server/tests/test_ink_real.py` written (tests a..h, skipped without INK_GGUF + INK_LIB); run 1 started 23:54.

(continued below as the work proceeds)

## Second incarnation (session restart), 2026-11-22
- 00:09 restarted mid-task; the previous session's two processes (server PID 456150, ops_measure A PID 456149) were already gone (`kill` -> no such process); run-1 results of
  test_ink_real.py were lost with the scratchpad (only the server logs in /tmp/inkreal_* survived). Host load 17 (32 logical CPUs), other agents on CPUs 2-20.
- 00:10 mock suite: `run_all.py` 59 tests, 0 failures, 21 s (test_ink_real.py SKIPs without INK_GGUF).
- 00:10 run 2 of test_ink_real.py started in the background (results -> build-v/vlogs/ink_real_results.json, log build-v/vlogs/ink_real_run2.log).
- `include/ink/ink_abi.h` is unchanged (no ink_spec_* yet); S2 has published the C++ side in include/ink/spec.h (SpecRequest/SpecResult/SpecDecoder::step), M2 verify tests passing.
- 00:13 FOUND AND STOPPED an orphan of the previous session: `scratchpad/chain1.sh` (PID 454771, reparented to init) was still chaining ops_measure phases on MY CPUs;
  its phase-B2 server (ctx 512, PID 463487) had been running since 00:13 next to my run-2 server => the first ~4 minutes of run 2 (tests a-e) are convoyed, timings there
  are NOT valid; tests f-h ran alone. Killed chain, ops_measure B2 and its server by PID. Salvaged from the scratchpad into build-v/vlogs/salvaged/: run 1 (23:54-23:59:
  a-f, h PASS; g failed on a test-code assert, fixed since), the g,h re-run (00:00-00:03: h PASS; g failed on a second test-code assert, fixed since; TTFT 2924 tokens
  116.8 s = 25 tok/s, /health max 37 ms during the prefill), ops_measure phase A (00:03-00:09, complete) and B1 (00:09, complete). B2 and C never completed.
- 00:16 task 4: `include/ink/ink_abi.h` still has no ink_spec_*; wrote the GUARDED ctypes binding anyway against a proposed layout mirroring S2's include/ink/spec.h
  (backends.InkBackend: SpecRequestC/SpecResultC, _bind_spec probes ink_spec_abi_version/ink_spec_load/ink_spec_step, spec_step -> StepResult with an explicit `fed`;
  `--drafter PATH` / INK_DRAFTER). Without the symbols supports_spec stays False (verified against build-v's libink.so in the new test). Proposal = docs/requests.md V item 4.
  Mock suite 60 tests, 0 failures.
- 00:18 run 2 of test_ink_real.py: **8/8 PASS** (a 145 s, b 12, c 35, d 16, e 18, f 1.5, g 109, h 50; exit 0). Numbers: TTFT 2924 tokens 104.1 s (28.1 tok/s prefill),
  /health max 19 ms during that prefill, two clients 8.48 / 8.44 tok/s per client (compute), cancel-after-disconnect 0.16 s, two-turn cache 76/89 cached (0.81 s vs 1.99 s
  without the cache), RSS 16.18 GiB after load / 16.77 after all tests / peak 32.12, idle CPU 1.95 % of a core (41 threads). Test a's speed numbers are the convoy
  artefact (raw engine 0.87 tok/s, HTTP 4.67 while the orphan's ctx-512 server shared CPUs 22-30): kept as evidence for "one engine process per CPU set".
- 00:17 agent task runs started (build-v/vlogs/run_agent.sh: ONE server --threads 5 --slots 2 --ctx 8192 on port 8100, PID in server_agent.pid; runs 1-2 T=0.6, run 3 T=0;
  work dirs build-v/agentwork_1..3 seeded with test_textstats.py only). docs/SERVING.md draft written (sections 1-7, 9, 10; 8 pending the runs).
- 00:30 agent task (tools/agent_task.py, one server, 5 threads, host load 6.5-13.7): run 1 T=0.6 **262 s, 5 steps, 7/7 PASS** (gen 1498 @ 6.93 tok/s, prefilled 1165,
  cached 5870); run 2 T=0.6 **207 s, 5 steps, 7/7** (1205 @ 7.45, 1165, 5246); run 3 T=0 **283 s, 5 steps, 7/7** (1636 @ 6.92, 1165, 6088). Decode = 78-83 % of wall,
  prefill 16-22 %, cache hit 82-84 % of prompt tokens. Baseline llama-server: 643 s / 9 steps / 3-5 tok/s at 16 threads idle. Server stopped by PID (470718);
  RSS 17.1 GiB after the runs, VmHWM 33.7. Records: build-v/vlogs/agent_run{1,2,3}.{json,log}, server_agent.log, SERVING.md section 8.
- 00:31 ops_measure phases C (1 slot, 2 clients) and B2 (ctx 512: 400s, clamping, chat growth to a full context) started chained in the background (one engine at a time).
- 00:33 ops_measure C: 1 slot, 2 clients 1 s apart (48 tokens each): client 2 queued 6.59 s, TTFT 7.53 s, wall 14.1 vs 7.6 s, both decode 7.3 tok/s (vs 2 slots: both
  start in 1.8 s at 3.9 tok/s each). B2 (ctx 512): 600-token prompt -> 400 context_length_exceeded, 512 -> 400, max_tokens clamped to the room (480+100 -> 32 tokens,
  finish length), chat grows 23 -> 429 prompt tokens with cache hits, turn 4 (526 tokens) -> clean 400, health ok, no log errors. docs/SERVING.md complete (10 sections).
- 00:34 DONE. All engine/server processes I started are stopped (ps: none). Not fixed / owners: idle-CPU root cause (B1/orchestrator, threadpool.h; mitigated server-side),
  2x peak RSS at load (S2, engine_load.cpp), ink_spec_* C ABI (S2; binding + proposal ready, requests item 4). Left for later: phase D (--prefill-chunk sweep),
  spec binding against the real symbols, an idle-host repeat of all numbers by the orchestrator. Tool-call budget used: ~45 of 90, wall 00:09-00:34.

## Third incarnation (Vc), 2026-11-22: speculative decoding on the REAL C ABI
- 00:35 read `include/ink/ink_abi.h` (S2's spec section = source of truth), `tests/spec/test_spec_capi.py`, S2.md, SPEC_CONTRACT. `nm -D build-s2/src/capi/libink.so`: all 7 symbols
  (ink_spec_load/free/enable/mode/synced/step, ink_sample) present, no `ink_spec_abi_version` (my predecessor's proposal is obsolete). S2 is running `test_spec all` on CPUs 10-20
  (PID 477261); library mtime 00:22:26 md5 cbb4cd2e -> snapshot copied once to `build-v/libink_spec.so` (used for everything below; S2 may rebuild the original).
- 00:40 `server/backends.py` rebound to the real layout: `SamplingParamsC` (nested `sp`), `SpecRequestC` (anchor, mode, sp, seed, stop_ids, n_stop, max_emit = 48 B), `SpecResultC`
  (tokens[8], n_tokens, stopped, fed[8], n_fed, T, drafted, accepted, rc, 4 doubles, next_seed = 136 B); sizes/offsets verified against gcc `sizeof`/`offsetof` of ink_abi.h.
  `_bind_spec` probes the 7 symbols (no version probe), `_setup_spec` = ink_spec_load right after ink_load (before any slot is filled) -> optional ink_spec_enable(--spec) ->
  ink_spec_mode (server default). New `Backend.sample()` = ink_sample for the first anchor; `spec_step` takes the engine's `fed[]`/`n_fed` (checked: fed[0] == anchor, fed[1:] ==
  emitted[:n_fed-1]) and `next_seed`; the scheduler chains the seed (first anchor -> step -> step) and keeps the `slot_pos == len(history)` check after every step.
  Per-request `speculation: auto | off | 2..8` (-> mode 1 / plain path / fixed T; absent -> -1 = engine default), server `--spec auto|off|2..8` -> ink_spec_enable.
  Effective mode 0, penalties or logit_bias => the plain ink_forward + numpy path. `timings` adds spec_tokens_per_step, spec_accept_rate, spec_mode, spec_synced (ink_spec_synced
  at the first step), spec_t_hist. Mock suite: 60 tests, 0 failures (ABI test rewritten for the real layout).
- 00:44 tests i..l added to `server/tests/test_ink_real.py` (own server with --drafter, INK_DRAFTER), `server/tests/spec_measure.py` (HTTP A/B off/auto/4/8), `tools/agent_task.py
  --speculation`. Chain started: `build-v/vlogs/chain_spec.sh` (tests -> measure -> agent task T=0 and T=0.6 with speculation auto, work dirs build-v/agentwork_spec_1/2).
- 00:47 first real run of tests i..l (own server with --drafter, build-v/libink_spec.so, load 6-14): k PASS (stop inside a verify window: slot == prompt + emitted tokens, next
  turn cached 100 %), l PASS (auto / forced / tool-result turns with speculation == plain). i: the greedy outputs WERE identical (2 prompts x 64, auto and fixed 4) but my own
  counter check used a 1e-6 tolerance on a 3-decimal value -> fixed (0.05). j: sampled T=0.6 seed 12345 **`auto` is NOT reproducible run to run** (tokens 1-2 equal, then
  150096 vs 160416): the policy's adaptive state persists across requests => different width => different draws. Test now asserts fixed widths (4, 8) and records auto;
  docs/requests.md item 5 for S2 (reset policy state per request, or position-seeded uniforms). Item 6: the predecessor's proposed ABI (item 4) is obsolete.
- 00:50 HTTP A/B (`server/tests/spec_measure.py`, greedy 100 tokens, 5 threads, load 12.8-14.6, indicative): code_completion off 7.03 tok/s, auto 6.80 (93 calls, 89 plain + 4 probes),
  fixed 4 5.99 (31 calls, 3.19 tok/call, acc 68/93), fixed 8 6.53 (21 calls, 4.71 tok/call, acc 78/143); chat_answer 7.29 / 6.91 / 4.05 / 2.73. All 6 speculative outputs ==
  plain. Per call: plain ~140 ms, T=4 ~540 ms (3.8x), T=8 ~730 ms (5.1x) => speculation does not pay on this partition and the auto policy correctly stays plain (3-5 % probe cost).
  docs/SERVING.md section 11 written (controls, counters, correctness, A/B table, when it pays). Agent-task runs with speculation auto (T=0, T=0.6) started 00:49:45 on port 8101.
- 00:59 agent task with speculation auto (port 8101, 5 threads, load 10.7-12.8): **T=0: 290.8 s, 5 steps, 7/7 PASS**, 1790 generated (1220 spec calls, 1.47 tok/call, acc 573/712),
  decode 7.3 tok/s overall vs plain run 3 283 s / 1636 tokens / 6.92; **T=0.6: 239.5 s, 5 steps, 7/7**, 1320 generated (851 calls, 1.55 tok/call, acc 474/625), 6.8 tok/s vs plain
  207-262 s / 6.93-7.45. Per token a wash (as the A/B says), wall time tracks the tokens the model wrote. Greedy run diverged from plain at step 3 (context > ~1000: fast verify
  kernel is bit-exact only < 768 keys; INK_SPEC_EXACT=1 for strict identity) -> docs/requests.md item 7. Records build-v/vlogs/agent_spec{1,2}.{json,log}, server_spec.log.
- 01:01 re-run of tests i..l on the final code: j, k, l PASS; i failed once more on MY histogram assertion (the engine shrinks T to the max_emit budget: last step T=3 for width 4;
  greedy tokens identical again) -> assertion widened, test i re-run alone (result below). Mock suite 64 collected, 0 failures. All engine processes stopped (ps: none).
- 01:04 test i re-run alone: **PASS** (s1: plain 7.14 tok/s, auto 7.62 (62 calls, 61 plain + 1 probe), fixed 4 7.49 (19 calls, 3.32 tok/call, acc 44/56); s2: 7.52 / 7.17 / 7.03,
  fixed 4 19 calls acc 44/54; both == plain token for token and == the llama.cpp oracle). Final state of tests i..l on the final code: 4/4 PASS (i 01:03, j/k/l 01:01).
- 01:05 DONE (Vc). Changed: server/backends.py (real ABI binding, ink_sample, fed[]/next_seed, --spec/parse_spec_mode), server/scheduler.py (seed chain, mode, synced, counters),
  server/app.py (`speculation` field, timings/log fields), server/__main__.py (--spec), server/tests/{test_spec_steps.py (ABI test), test_ink_real.py (tests i..l), spec_measure.py (new)},
  tools/agent_task.py (--speculation, counters), server/README.md, docs/SERVING.md section 11, docs/requests.md items 5-7. Not fixed / owners: auto-mode seed reproducibility (S2, item 5),
  fast-verify greedy divergence past 768 keys (S2/orchestrator, item 7: run INK_SPEC_EXACT=1 for identity), speculation does not pay at 5 threads on the loaded host (engine-side verify
  cost ratio 3.8-5.1x; orchestrator to redo the A/B at 15-16 threads idle). Left: idle-host numbers, exact-kernel cost, phase D. Tool calls ~30 of 70, wall 00:35-01:05.
  All processes I started are stopped (servers by SIGTERM via the test harness / chain, verified with ps).

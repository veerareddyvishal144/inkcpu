# Serving Inkling 2 Mini with inkcpu: operational review for long-running agentic use

Agent V, 2026-11-22. Server: `server/` (FastAPI, OpenAI-compatible) on `libink.so` (build-v, C1 decode executor + P batched prefill) with the Q4_K_XL GGUF.
EVERY timing below was taken on the shared 16-core Xeon 8171M host with other agents running (load average 8-21) on a 5-thread partition
(`INK_CPUS=22,24,26,28,30`): they are INDICATIVE; the orchestrator's idle-host numbers (13 tok/s decode at 16 threads) are the reference.
Sources: `server/tests/test_ink_real.py` (run 1 23:54, g/h re-run 00:00, run 2 00:10), `server/tests/ops_measure.py` phases A and B1 (00:03-00:09),
`tools/agent_task.py` (section 8). Raw JSON: `build-v/vlogs/` (`salvaged/ops_A.json`, `salvaged/ops_B1.json`, `salvaged/real_results.json`, `ink_real_results.json`, `agent_run*.json`).

## 1. Context and memory per slot
Engine geometry (`include/ink/engine.h`, `kernels_misc.h`, `engine_load.cpp`): 48 layer visits per token, 8 of them global attention (KV grows with the
context) and 40 sliding-window (SWA ring of `kSwaRingBlocks` = 32 blocks = 2048 positions). KV is fp16, 8 KV heads x 128 = 1024 values per tensor per position,
stored in 64-position blocks of 128 KiB per tensor.

| item | formula | value |
|---|---|---|
| KV per position, one visit | 1024 x 2 B x (K + V) | 4 KiB |
| global KV per slot (8 visits) | 32 KiB x tokens | 31.25 MiB per 1000 tokens, 256 MiB at 8192 |
| SWA KV per slot (40 visits, ring) | 160 KiB x min(tokens, 2048) | 156 MiB per 1000 tokens up to 2048, then constant 320 MiB |
| conv histories per slot | 48 x 3 x 6656 floats | 3.7 MiB |
| per-slot maximum at `--ctx 8192` | | ~0.57 GiB (mapped lazily, MAP_NORESERVE: only touched pages are resident) |
| per 1000 tokens | | 187 MiB while the slot holds < 2048 tokens, 31 MiB beyond |

Verified by RSS growth (phase A, prompts with no prefix hit): slot grows 300 -> 1000 tokens: +132.1 MiB measured vs 700 x 192 KiB = 131.3 MiB predicted (twice).
First 300-token fill of a fresh slot: +77.8 MiB (includes the lazily allocated prefill scratch), second slot +47.1 MiB.
Process RSS: 16.18 GiB right after load, 16.32 after the first request (prefill scratch), 16.7-16.8 GiB after the whole test suite with both slots used.
**Peak RSS during load is 32.1 GiB** (VmHWM: the mapped GGUF pages and the packed arena coexist until the copy ends; docs/requests.md V item 2): the
process needs ~34 GiB free to START, 17 GiB to run. Two servers loading at once need ~68 GiB.

Max context per slot = `--ctx` (default 8192; `max_ctx` of `ink_load`). The engine is verified up to ~2700 tokens (C1) / 1500 (P) against llama.cpp; positions >= 8191
enter the untested tau > 1 regime of the global layers, so **8192 is the recommended ceiling** (the test suite drives a 2924-token prompt and a 2x-over-context
rejection through the server; a full 8192 slot is memory-tested only).

## 2. When the context fills
- Engine (phase B1, ctx 512): a prefill or a step that would exceed the slot returns `-2` (`ContextFull`), the slot position is unchanged (400 stays 400, 512 stays 512),
  the slot stays usable (another 100 tokens appended fine), `ink_slot_reset` + new prefill works. No crash, no corruption.
- Server: a prompt with `len + 1 > ctx` is rejected at admission with **HTTP 400 `context_length_exceeded`** in < 5 s on both endpoints (test g: 9000-token raw
  prompt, 13600-token chat, exactly-8192 prompt); `max_tokens` is clamped to `ctx - prompt` (`app.clamp_max_tokens`), so a generation that reaches the end of the
  slot finishes with `finish_reason: "length"` instead of hitting `-2`; a multi-turn chat whose rendered prompt no longer fits gets the 400 and must summarise /
  truncate on the client side (the server never evicts or shifts context: append-only slots). The server stayed healthy after every rejection (200 on the next request).
- Verified end to end with a ctx-512 server (phase B2, 00:31): 600-token prompt -> 400 `context_length_exceeded` ("maximum context length is 512 tokens, but the
  prompt has 600 tokens"); exactly 512 -> 400; 480-token prompt with `max_tokens: 100` -> clamped, 32 tokens, `finish_reason: length`; 511 tokens -> 1 token, `length`.
  A chat grown turn by turn (23 -> 120 -> 334 -> 429 prompt tokens, cache hits 107 / 319 / 416) got its 400 at turn 4 (526 tokens); `/health` ok afterwards, no
  server-log errors, slots reusable.
- A `-2` that does slip through (should not) is a 500 for that request and the slot is reset (`SlotTable.release(healthy=False)`).

## 3. Client timeout, cancel, disconnect, restart (phase A, test e, test f)
| event | measured |
|---|---|
| streaming client disconnects mid-generation | slot free after 0.16 s (next decode step); `finish_reason: cancelled` in the request log |
| non-stream client read-timeout 2 s during decode | slot free 1.08 s after the timeout |
| client closes during a 3000-token prefill | slot free after 2.96 s (the 256-token chunk in flight completes: up to ~9.5 s at 5 threads); the slot keeps the prefilled prefix (256 tokens) for reuse |
| queue full (2 slots + `--max-queue 2`, 7 clients) | 4 admitted, 3 x **429** with `Retry-After` (seconds, from the EMA of request durations) and `code: queue_full`; `ink_rejected_requests_total` counts them |
| SIGTERM during generation | process exits 10.65 s later (uvicorn `--shutdown-grace 10`); the client sees an incomplete chunked body (`RemoteProtocolError`), no partial-JSON success |
| restart to healthy | 7.1 s with a hot page cache (5-7 s load); `/health` is 503 `loading` until then; a failed load exits with code 3 (`fail_fast`) so a supervisor restarts it |
Recommendation for agent clients: stream, set a read timeout >= 2 x the longest expected prefill chunk (20 s) and retry on 429 after `Retry-After`; never retry a
400 `context_length_exceeded` without shortening the prompt.

## 4. /metrics and the request log
`/metrics` (Prometheus text): `ink_requests_total`, `ink_rejected_requests_total`, `ink_tokens_total{kind=prompt|completion|cached|reasoning}`,
`ink_finished_total{reason=stop|length|tool_calls|cancelled|error}`, `ink_seconds_total{phase=prefill|decode_engine|decode_sampling|decode_other|queue}`,
`ink_slot_tokens{slot,busy}`, queue depth, request duration histogram, speculative counters, last prefill/decode tok/s. The phase split is the useful part for
capacity work: after phase A the server had spent 165 s in prefill, 114 s in the decode engine, 0.58 s sampling, 0.05 s other, 0.34 s queueing (prefill
dominates agentic traffic unless the cache hits). Per-request JSONL log (`--request-log`): ts, client, endpoint, model, status, finish_reason, prompt/cached/
completion/reasoning tokens, queue/prefill/decode/engine/sample/total ms, ttft_ms, decode_tps, sampling params, n_tools, spec_steps, errors (no content
unless `--log-bodies`). Missing (would help): per-slot cache-hit ratio and a gauge of the longest wait in the queue.

## 5. Two slots vs one: what breaks first
- Two concurrent decodes (phase A, 2 x 64 tokens): aggregate 7.07 tok/s = the single-stream speed; each client gets 3.86 / 3.89 tok/s, first token 1.8-1.9 s. The
  engine is single-sequence: slots are time-sliced at decode-step / prefill-chunk granularity, so 2 slots give fairness, not throughput.
- **What breaks first is the latency of the decoding client while the other slot prefills**: a 1000-token prefill next to a decode took 50.3 s (37.0 s alone) and the
  decoding stream saw gaps of up to 15.9 s (4 gaps > 1 s, median 0.13 s): the scheduler alternates one 256-token chunk (~9.5 s at 5 threads) with one decode
  step. With one slot (`--slots 1`, phase C, 00:30: two clients 1 s apart, 48 tokens each) the second client queues instead: queue 6.59 s, TTFT 7.53 s, request wall
  14.1 s vs 7.6 s for the first, but BOTH decode at the full 7.3 tok/s and neither sees a stall. So: 1 slot = predictable per-request speed, waiting is explicit
  (queue_ms, 429 when the queue is full); 2 slots = both start at once (first token 1.8-1.9 s) at half speed each, and a long prefill on one slot freezes the other
  for ~10 s per chunk. For a single agent loop use `--slots 1`; for two independent agents `--slots 2 --prefill-chunk 64`.
- Mitigation knob: `--prefill-chunk 64` (chunk ~2.4 s, also the cancel granularity) for interactive fairness; cost on prefill throughput not measured (phase D not run).
- Memory is not the limit: a second slot costs at most 0.57 GiB.
- Prefix-cache: with 2 slots a two-client agent workload keeps both conversations cached; a third conversation evicts the LRU idle slot (full re-prefill).

## 6. Prefill / TTFT (5 threads, loaded host, no prefix hit)
| prompt tokens | TTFT | prefill tok/s | source |
|---|---|---|---|
| 13 (raw engine, no server) | 1.04 s | 12.5 | run 1 (`raw_engine_prefill_13tok_s`) |
| 300 | 10.8 s / 11.1 s | 27.8 / 26.9 | phase A |
| 1000 | 37.0 s / 37.1 s | 27.0 / 27.0 | phase A |
| 2924 | 116.8 s / 104.1 s | 25.0 / 28.1 | g/h re-run 00:00 and run 2 00:15 (both alone on the partition; load average 15-21) |
`/health` stayed below 37 ms (run 2: 19 ms) during the 2924-token prefill (engine calls run on their own thread; chunks of 256 are visible in `/slots`). Prefill is ~4x decode
per token (27 vs 7 tok/s) because the batched executor amortises the weight stream over 256 rows; cache hits remove it: the two-turn test re-prefilled only the 13
new tokens of turn 2 (76 of 89 cached, 0.99 s instead of 2.63 s for the whole conversation without the cache).

## 7. Decode: HTTP vs raw engine, per-token overhead
| path | tok/s | source |
|---|---|---|
| raw libink through ctypes, greedy, 64 tokens, no server | 6.16 | run 1 (same minute as the HTTP number) |
| HTTP `/v1/completions` greedy 64 tokens | 6.42 | run 1 |
| HTTP greedy 96 tokens (phase A) | 7.07 (engine-only 7.08) | phase A |
| HTTP T=0.6 / top_k 40 / top_p 0.9 / rep-penalty 1.1 | 6.45 / 6.92 / 7.32 / 7.78 (engine-only 6.51 / 6.99 / 7.39 / 7.87) | phase A (host load drifts between rows) |
| streaming client, 96 events | 7.6 tok/s at the client, max gap 0.14 s | phase A |
| two clients interleaved, 24 tokens each (run 2, clean) | 8.48 / 8.44 per client (compute time; wall 7.28 s for both, engine 7.25 s) | run 2 |
| predecessor, 23:48, lighter load | 8.1 | status log |
| CONVOY: raw libink while ANOTHER engine process ran on the same 5 CPUs | **0.87** (prefill of 13 tokens 5.2 s instead of 1.0 s) | run 2 test a, accidental (section 10: one engine per CPU set) |
Per-token time outside the engine: sampling 0.15-0.16 ms greedy, 1.2-1.5 ms at T=0.6 (numpy two-level inverse-CDF sampler, 200k vocab); everything else
(scheduler, parser, SSE) 0.044-0.051 ms; the predecessor's 0.17 ms total for greedy is confirmed (0.20-0.24 ms here with sampling included). Against a 127-154
ms engine step this is 0.15 % (greedy) to 1 % (sampled): HTTP decode speed equals the raw engine's within the measurement noise.

## 8. Real agent task (tools/agent_task.py)
Task: implement `textstats.py` so that the 7 unit tests of `test_textstats.py` pass, with three tools (read_file / write_file / run_tests), max 12 steps, 3000 max
tokens per step, default reasoning effort. Each run in a FRESH work dir holding only the test file (`build-v/agentwork_1..3`). ONE server
(`--threads 5 --slots 2 --ctx 8192`, INK_CPUS=22,24,26,28,30) for all three runs; the harness sends `reasoning_content` back so the slot cache hits.
Host load average 6.5-13.7 during the runs (other agents' engines on CPUs 2-20).

**Run 1** (temperature 0.6, work dir `build-v/agentwork_1`, 5 steps, 262.0 s wall, tests PASS, 7 ran)

| step | wall s | prompt tokens | new (prefilled) | cached | generated | decode tok/s | prefill ms | ttft ms | tool |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 19 | 318 | 318 | 0 | 46 | 7.2 | 12377.7 | 12387.9 | read_file |
| 2 | 20 | 779 | 415 | 364 | 36 | 7.4 | 15236.8 | 15238.4 | run_tests |
| 3 | 190 | 1088 | 273 | 815 | 1242 | 6.9 | 10541.4 | 10543.1 | write_file |
| 4 | 4 | 2338 | 8 | 2330 | 23 | 6.5 | 841.5 | 843.3 | run_tests |
| 5 | 29 | 2512 | 151 | 2361 | 151 | 6.8 | 6460.2 | 6462.0 | stop |
| total | 262 | | 1165 | 5870 | 1498 | 6.93 (mean) | 45 s prefill | | |

**Run 2** (temperature 0.6, work dir `build-v/agentwork_2`, 5 steps, 207.2 s wall, tests PASS, 7 ran)

| step | wall s | prompt tokens | new (prefilled) | cached | generated | decode tok/s | prefill ms | ttft ms | tool |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 18 | 318 | 318 | 0 | 47 | 7.0 | 11792.5 | 11794.3 | read_file |
| 2 | 21 | 780 | 415 | 365 | 36 | 7.2 | 15866.3 | 15867.9 | run_tests |
| 3 | 134 | 1089 | 273 | 816 | 928 | 7.5 | 10187.3 | 10189.1 | write_file |
| 4 | 4 | 2025 | 8 | 2017 | 23 | 7.6 | 713.6 | 715.3 | run_tests |
| 5 | 29 | 2199 | 151 | 2048 | 171 | 7.4 | 5596.3 | 5597.9 | stop |
| total | 207 | | 1165 | 5246 | 1205 | 7.45 (mean) | 44 s prefill | | |

**Run 3** (temperature 0.0, work dir `build-v/agentwork_3`, 5 steps, 283.1 s wall, tests PASS, 7 ran)

| step | wall s | prompt tokens | new (prefilled) | cached | generated | decode tok/s | prefill ms | ttft ms | tool |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 19 | 318 | 318 | 0 | 46 | 6.5 | 12118.5 | 12195.7 | read_file |
| 2 | 21 | 779 | 415 | 364 | 36 | 6.8 | 15633.9 | 15634.6 | run_tests |
| 3 | 207 | 1088 | 273 | 815 | 1351 | 6.9 | 11717.4 | 11718.1 | write_file |
| 4 | 4 | 2447 | 8 | 2439 | 23 | 6.9 | 810.9 | 811.7 | run_tests |
| 5 | 31 | 2621 | 151 | 2470 | 180 | 7.1 | 6195.8 | 6196.7 | stop |
| total | 283 | | 1165 | 6088 | 1636 | 6.92 (mean) | 46 s prefill | | |

| run | T | steps | wall s | generated | prefilled | cached | decode tok/s | prefill s | decode s | tool time s | tests |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.6 | 5 | 262 | 1498 | 1165 | 5870 | 6.93 | 45 | 216 | 0 | PASS |
| 2 | 0.6 | 5 | 207 | 1205 | 1165 | 5246 | 7.45 | 44 | 162 | 1 | PASS |
| 3 | 0.0 | 5 | 283 | 1636 | 1165 | 6088 | 6.92 | 46 | 236 | 0 | PASS |

- **3 of 3 runs produced a module that passes all 7 tests, in 5 steps each** (read tests -> run tests -> write module -> run tests -> final answer; run 2 wrote the module
  at step 2). Wall 207-283 s; the model's own output (1205-1636 tokens incl. thinking) at 6.9-7.5 tok/s is 78-83 % of the wall time, prefill 16-22 % (1165 NEW tokens per
  run: the cache served 5246-6088 tokens, 82-84 % of all prompt tokens), tool execution and HTTP < 1 %.
- Baseline (orchestrator, llama-server `-t 16` on an IDLE host, older harness that dropped reasoning_content): 643 s, 9 steps, 3-5 tok/s decode, 7/7 tests.
  Normalised: our decode runs 6.9-7.5 tok/s at 5 threads on a loaded host vs 3-5 at 16 threads idle (1.4-2.5x per token); wall time 2.3-3.1x shorter, but the step
  count differs (5 vs 9: a different sampling trajectory, not a server property) and the baseline's per-step token counts are not on disk, so the fair statement is
  "per generated token 1.4-2.5x faster with a third of the threads; per task about 2.5x faster". The idle-host 16-thread number for this engine (13 tok/s) would
  roughly double our wall-time advantage again.
- Server RSS after the three runs: 17.1 GiB (16.2 at load; two slots holding ~2000-token histories), VmHWM 33.7 GiB (load peak). No errors, 429s or resets in the server log.

## 9. Idle CPU and peak RSS
Idle server: 2.0-2.2 % of one core (`--timer-slack-us 5000`, default) vs 30 % with the kernel's 50 us slack (docs/requests.md V item 1: the pool polls every 20 us);
41 threads in the process (5 engine threads + uvicorn/asyncio); run 1 2.0 %, g/h re-run 2.2 %, run 2 1.95 % (20 s windows after 30 s idle).
Peak RSS at load 32.1 GiB (VmHWM, all three runs), steady 16.2-16.8 GiB (section 1).

## 10. Recommended launch and limits
```
cd /data/inkling-mini/engine
INK_CPUS=2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 \      # distinct physical cores (even ids on this host; core 0 is half speed), nothing else pinned there
/data/engine/.venv/bin/python -m server --backend ink --model /data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
    --lib build/src/capi/libink.so --threads 15 --slots 2 --ctx 8192 --max-queue 8 --prefill-chunk 256 \
    --request-log /var/log/ink/requests.jsonl --port 8000              # add --drafter <dspark.gguf> once libink exports ink_spec_*
```
- RAM: 34 GiB free at start, 17 GiB + 0.6 GiB per slot steady; one engine process per CPU set (two on the same CPUs convoy to 4-5 tok/s).
- Limits: `--ctx 8192` (untested beyond), `--slots 1` for one agent loop (no prefill-induced decode stalls, explicit queueing), `--slots 2` for two independent agents, `--max-queue` so that
  queue x typical request time stays under the clients' timeout (429 is cheaper than a timeout), `--prefill-chunk 64` for interactive mixes.
- Supervise the process: exit code 3 = model load failed; SIGTERM drains open streams for `--shutdown-grace` seconds (10).
- Clients: send `reasoning_content` back in assistant messages (otherwise every turn is a full re-prefill), stream, honour `Retry-After`.

## 11. Speculative decoding (DSpark drafter, agent S2's `ink_spec_*` ABI) -- Vc, 2026-11-22
How it is wired: `--drafter PATH` (or `INK_DRAFTER`) makes `InkBackend.start()` call `ink_spec_load` right after `ink_load`, BEFORE any slot is
filled (the tap hook must see every token: a slot filled without it runs plain steps until it is reset; `timings.spec_synced` tells). Per request
the scheduler prefills with `ink_forward`, samples the first token with `ink_sample(logits, params, seed)` (the ANCHOR, emitted but not yet
processed), then loops `ink_spec_step(anchor, mode, params, seed, stop_ids, max_emit)`: the engine drafts, verifies, samples, applies the stop ids
and the `max_emit` budget (= remaining `max_tokens`, <= 8), rolls back and returns the emitted tokens, the tokens it fed (`fed[]`), counters and
`next_seed` (chained into the next step: a request is one deterministic chain). The slot history is `fed`, checked against `ink_slot_pos` after
every step (a mismatch is a 500 and the slot is reset; never happened in any run). Library: `build-v/libink_spec.so` = snapshot of
`build-s2/src/capi/libink.so` (00:22, md5 cbb4cd2e); verify attention kernel = the fast one (`INK_SPEC_EXACT` unset; bit-exact below 768 keys per S2).

Controls
| where | value | effect |
|---|---|---|
| server flag `--spec` | `auto` / `off` / `2..8` | engine default (`ink_spec_enable`) for requests without a `speculation` field; default = the engine's (env `INK_SPEC`, else auto) |
| server flag `--no-spec` | | never use the speculative path (no `ink_spec_step`, no `ink_sample`), even if a drafter is loaded |
| request field `speculation` | `"auto"` | the engine's policy picks the width per step (probes, backs off: see the numbers) |
| | `"off"` | plain decode for this request (`ink_forward` + the numpy sampler, exactly the pre-speculation path) |
| | `2..8` | fixed verify width T (rows incl. the anchor, so up to T-1 drafts accepted per step) |
| | absent | the server default above (`-1` to the engine) |
| automatic fallback | penalties, `logit_bias`, effective mode off | plain path; `n > 1`, `logprobs`, `response_format` are 400 as before |

Counters: `timings.spec_steps`, `spec_drafted`, `spec_accepted`, `spec_tokens_per_step` (emitted tokens per engine call; the first anchor is sampled
outside), `spec_accept_rate`, `spec_mode` (what was sent: 1 auto, 2..8 fixed, -1 server default), `spec_synced`, `spec_t_hist` (verify width ->
steps). The request log carries spec_steps / drafted / accepted / mode; `/metrics` sums them in `ink_spec_total{kind=steps|drafted|accepted|emitted}`.

Correctness (tests i..l of `server/tests/test_ink_real.py`, real engine + drafter, 00:44 and the re-run below): greedy output with `auto` and fixed 4 ==
plain greedy output token for token on s1_en_capital and s2_code_add x 64 tokens (and == llama.cpp's oracle, as before); all 6 greedy A/B runs below
`same_as_plain`; a stop token inside a verify window ends the turn cleanly and the slot holds exactly prompt + emitted tokens (the next turn of the chat
hit the cache for the whole first turn); tool calls (auto / forced / tool-result turn) identical to plain. Sampled (T=0.6, seed fixed): fixed widths are
reproducible run to run; **`auto` is not** (the policy's adaptive state persists across requests, so the width and the random draws differ:
docs/requests.md item 5 for S2; use a fixed width when `seed` must mean something).

HTTP A/B (`server/tests/spec_measure.py`, greedy, 100 new tokens, prompts of tools/dspark/prompts, 5 threads, host load 12.8-14.6, INDICATIVE):
| prompt | mode | tokens | tok/s | engine calls | ms per call | tokens per call | accepted/drafted | T histogram | == plain |
|---|---|---|---|---|---|---|---|---|---|
| code_completion | off (plain) | 100 | 7.03 | 100 | 142 | 1 | - | - | yes |
| code_completion | auto | 100 | 6.80 | 93 | 158 | 1.065 | 6/12 | {"1": 89, "4": 4} | yes |
| code_completion | fixed 4 | 100 | 5.99 | 31 | 538 | 3.194 | 68/93 | {"4": 31} | yes |
| code_completion | fixed 8 | 100 | 6.53 | 21 | 729 | 4.714 | 78/143 | {"4": 1, "8": 20} | yes |
| chat_answer | off (plain) | 90 | 7.29 | 90 | 137 | 1 | - | - | yes |
| chat_answer | auto | 90 | 6.91 | 87 | 150 | 1.034 | 3/9 | {"1": 84, "4": 3} | yes |
| chat_answer | fixed 4 | 90 | 4.05 | 49 | 453 | 1.837 | 41/147 | {"4": 49} | yes |
| chat_answer | fixed 8 | 90 | 2.73 | 48 | 688 | 1.875 | 43/336 | {"8": 48} | yes |
Reading: a plain step costs ~140 ms here; a fixed-4 step ~540 ms (3.8x) for 3.2 tokens on code (0.85x), a fixed-8 step ~730 ms (5.1x) for 4.7 tokens (0.92x);
on chat the acceptance collapses (1.8 tokens per call) and fixed widths halve the speed. The `auto` policy sees this and stays plain (89 of 93 steps at T=1,
4 probes of width 4) at a 3-5 % probe cost. **When it pays**: the verify of T rows must cost less than (tokens per call) plain steps; on this 5-thread loaded
partition the batched verify is 3.8-5.1x a decode step, so even 73 % acceptance on code does not win. That ratio is the lever (S2's bench_spec on an idle host
at 15-16 threads is the number to look at: decode is memory-bound there, so extra verify rows are cheaper); the server side adds nothing measurable
(0.2 ms per step). Recommendation today: `--drafter` + default `auto` is safe (identical greedy output, ~3 % cost on prose), `speculation: 4` for code-heavy
clients only once the idle-host A/B shows > 1.0x; leave sampled requests that need `seed` reproducibility at a fixed width or off.

Agent task (`tools/agent_task.py --speculation auto`, same harness and server geometry as section 8, fresh work dirs `build-v/agentwork_spec_1/2`, 5 threads, host
load 10.7-12.8, INDICATIVE; plain rows from section 8 for comparison):
| run | T | speculation | wall s | steps | generated | prefilled | cached | engine calls (spec) | tokens per call | accepted/drafted | decode tok/s overall | tests |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain run 3 | 0 | off | 283 | 5 | 1636 | 1165 | 6088 | - | - | - | 6.92 | 7/7 |
| spec 1 | 0 | auto | 291 | 5 | 1790 | 1167 | 6302 | 1220 | 1.47 | 573/712 (80 %) | 7.3 | 7/7 |
| plain run 1 | 0.6 | off | 262 | 5 | 1498 | 1165 | 5870 | - | - | - | 6.93 | 7/7 |
| plain run 2 | 0.6 | off | 207 | 5 | 1205 | 1165 | 5246 | - | - | - | 7.45 | 7/7 |
| spec 2 | 0.6 | auto | 240 | 5 | 1320 | 1167 | 5420 | 851 | 1.55 | 474/625 (76 %) | 6.8 | 7/7 |
Both speculative runs solved the task in 5 steps (7/7 tests), same trajectory shape as the plain runs (read -> run -> write 1000-1450 tokens -> run -> answer).
Per generated token the speculative runs decode at 6.8-7.3 tok/s vs 6.9-7.5 plain: a wash within the host-load noise, as the A/B predicts; wall time follows the
number of tokens the model chose to write (1320-1790), not the mode. In the long write step the policy did speculate (667-998 calls for 1015-1456 tokens,
acceptance 80-86 %) and still gained nothing, because a T=4 verify costs 3.8 plain steps here. Note on the greedy run: steps 1-2 reproduced the plain run
token counts exactly (46, 36), step 3 diverged (1456 vs 1351 tokens) once the context passed ~1000 tokens: the fast verify kernel is bit-exact only below
768 keys (S2), so beyond that a greedy argmax can flip. For strict plain == speculative identity at any length start the server with `INK_SPEC_EXACT=1`
(decode-kernel attention in the verify; its cost on this host is not measured).


## 12. Reasoning effort defaults for editor clients (orchestrator, 2026-11-22)

The model always opens with a thinking message unless told otherwise; its own effort 0 still thinks for ~150 tokens, so a 40-token
request (Caret/Cursor chat titles, Tab) returned an empty body. Two server changes:
- `--default-reasoning-effort <v>` (env `INK_DEFAULT_EFFORT`): effort applied when the request carries no `reasoning_effort`
  (editors send one only when their Thinking toggle is on). `tools/serve.sh` sets `none`.
- `reasoning_effort: "none"` (or 0) without tools now prefills `<|message_model|><|content_text|>` so no thinking segment is generated
  at all (title request: 1.1 s, 0 reasoning tokens). With tools declared the prefix is not used (it would stop the model from opening
  with a tool call) and effort 0 = short thinking (~40 tokens) applies instead.
Measured on a Caret agent turn: 7,215-token prompt (system prompt + 32 tool schemas), 0 cached tokens -> prefill 120.6 s at 12 threads
(60 tok/s), 59 output tokens in 5 s; total 125.6 s. The context was raised to 32k (`--ctx 32768`), threads to 16 and the prefill chunk
to 512 for this deployment; follow-up turns reuse the slot's prefix cache (`cached_tokens` in `usage`) if the client resends the same prefix.

## 13. Slot rewind and shared-prefix reuse (agent S3, 2026-11-22; details in docs/REWIND.md)

The engine now checkpoints every slot after each prefill chunk and at each prompt end (3.83 MB conv-history snapshots; the KV needs no copy while
at most 1025 tokens follow, see REWIND.md section 3) and can rewind a slot to a checkpoint (`ink_slot_rewind`). The server uses it when no slot's
history is a full prefix of the new prompt: the idle slot sharing the longest common prefix (>= `--prefix-reuse-min`, default 1024 tokens) is
rewound to the nearest restorable checkpoint <= the shared length and only the rest of the prompt is prefilled (`cached_tokens` reports the
reused length). Prompt ends get a "deep" checkpoint (adds a 170 MiB copy of the SWA window, so it survives a generation of any length), the point
where two conversations were seen to diverge gets one too, every generation end a shallow one. Effect on the Caret pattern (the client resends
the ~7,200-token prefix and re-renders the assistant turn without `reasoning_content`): the follow-up turn rewinds to the previous prompt end and
prefills only the re-rendered assistant turn + the new user message instead of the whole conversation; two threads with the same system prompt +
tool schemas share the prefix as well. Memory: <= ~370 MB per slot of checkpoint buffers (allocated on first use). Logits after a rewind are
bit-identical to a fresh slot fed the same tokens (tests/engine/test_rewind.cpp, test_ink_real.py m/n). Requires a libink.so with the
`ink_slot_*` checkpoint symbols (build-s3 or later); older builds keep the append-only policy. Measurements: REWIND.md section 7.

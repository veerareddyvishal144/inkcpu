# T status (wave 1: SMT topology, pool idle parking, server per-token overhead)

Owner: agent T. Owns `src/kernels_gemv/threadpool.cpp`, `include/ink/threadpool.h`, `tests/kernels_gemv/{test_threadpool,bench_pool}.cpp`,
`bench/bench_stream.cpp`, `server/*.py` (verified on TEST servers only, INK_PORT=8010), new `tools/*.sh`. Build dir `build-t`. Dev CPUs 29,31.
Time box: started 18:12 UTC (orchestrator restart 18:19; box ends 21:20 UTC).

## 18:20 UTC plan + first facts (loadavg 0.75, production active 0, free 36 GB / available 97 GB)
* Built `build-t` (cmake Release, ninja on CPUs 29,31): libink.so, bench_decode, bench_prefill, bench_stream, bench_pool, test_threadpool. Exit 0.
* Production (pid 1032707) idle CPU measured from /proc/<pid>/stat over 10 s: 7.1 % of one core (16-thread pool, timer slack 5 ms via prctl in
  the server; V measured 2 % with a 5-thread pool). Open item 2 of ENGINE.md confirmed as the baseline for task 2 (futex park).
* Production thread placement (read-only /proc): 15 pool workers pinned 1:1 to CPUs 4..30 and the engine thread on CPU 2; main thread + 4
  libuv workers restricted to the odd CPUs (server/__main__.py sched_setaffinity); BUT ~30 python threads with mask 0-31 (LWP ids below the
  pool's) float over all CPUs incl. the even ones. They were created BEFORE the affinity call, i.e. at `import numpy` (OpenBLAS worker pool,
  one per CPU). They use 0.0 % CPU (sleeping on condvars; the server never calls BLAS) so they do not steal issue slots today; cheap fix for
  task 4 = `OPENBLAS_NUM_THREADS=1` before numpy is imported + the affinity call before the import (server-only change, test server).
* Order of work: (1) SMT sweep script `tools/smt_sweep.sh` under the lock now (bench_decode ctx 16/1000 x 16/20/24/32 threads with INK_PROFILE=1,
  bench_prefill 256/512, bench_stream 16/24/32); (2) futex parking in the pool while the sweep runs (dev CPUs); (3) server overhead instrumentation
  on a test server (port 8010, CPUs 29,31); (4) server thread placement.

## 18:31 UTC: SMT decode sweep done, futex parking implemented and tested, stage 2 queued
### Task 1, decode (`tools/smt_sweep.sh`, build-t, lock, INK_PROFILE=1, 64 tokens x 2 reps; CPU list = 16 physical cores as default, then
odd siblings 3,5,..,17 for 20/24 threads, then the rest with CPU 1 last for 32). loadavg before each cell (1-min) rose 1.9 -> 9.4 (ctx 16) and
28.7 -> 25.5 (ctx 1000, inflated by the run's own 1000-token prefill-by-decode and the previous cells; the host was otherwise idle: active 0).
| ctx | 16 thr | 20 thr | 24 thr | 32 thr |
|---|---|---|---|---|
| 16 | 14.05 (reps 14.05 13.54) | 13.88 (13.79 13.88) | 14.28 (14.28 13.92) | 10.95 (10.95 10.41) |
| 1000 | 12.80 (12.77 12.80) | 13.57 (13.57 13.20) | 13.19 (13.19 12.73) | 11.68 (11.68 9.87) |
Per phase, ctx 16, ms/token (GB/s): proj 15.69 (65) / 15.44 / 14.78 (69) / 17.54 (58); experts gate_up 22.54 (55) / 21.50 / 21.22 (58) / 26.41 (47);
experts down 12.33 (50) / 12.17 / 12.25 / 15.90 (39); wo 6.41 / 6.26 / 6.50 / 8.61; lm_head 5.77 (85) / 5.62 / 5.53 (89) / 5.74; controllers 4.37 / 3.92 / 4.17 / 7.05;
serial mixes 3.40 / 3.51 / 3.77 / 8.18; router 0.84 / 1.05 / 0.99 / 2.03; attention 1.89 / 1.89 / 1.95 / 3.87; sum 73.4 / 71.6 / 71.3 / 95.5.
Reading: 8 SMT siblings (24 threads) buy 6 % on the ALU-bound gate_up phase and 6 % on proj, 4 % on lm_head, nothing on down; the serial phases
(+5-22 %: tid 0's sibling CPU 3 spins), router and attention lose a little; net +1.6 % tok/s at ctx 16 and +3 % at ctx 1000 = within the rep-to-rep
noise. 32 threads lose 22 %: every phase slows (serial phases 2.4x, controllers 1.6x, experts +17 %): with no free vCPU left every other runnable
thread on the box preempts a spinner and the barriers convoy (ENGINE_DESIGN.md 9 pathology), so the 32-cell says nothing about SMT itself.
Hypothesis status: confirmed weakly for gate_up only; the bandwidth-bound phases did not lose at 24 threads; the fixed/serial phases did.
A mixed layout (siblings only in the expert phases) could save at most ~2.5 ms/token (3 %) on gate_up+down IF the pause-spinning siblings cost
nothing elsewhere (they cost 5-22 % on the serial phases here) and would need all 32 vCPUs: not worth an engine change on this shared host -> I do
not implement run_chunks-with-participants; evidence goes to docs/requests.md for E1/orchestrator. Prefill + stream cells are queued (stage 2).
### Task 2, futex parking (`threadpool.cpp/.h`, default ON, `INK_POOL_PARK=0` = legacy 20 us polling)
* Design: after the 2 ms spin a worker does sleepers++ (seq_cst), re-checks the epoch, FUTEX_WAIT_PRIVATE on the epoch word (250 ms safety
  timeout, never a hang), sleepers--. The dispatcher stores the epoch seq_cst (one xchg instead of mov) and futex-wakes only if sleepers != 0:
  hot path = one extra L1 load, no syscall while tokens flow (wakes counter in bench_pool shows 0 during the latency loops).
* Idle CPU, 16-thread pool, process CPU over 10 s (getrusage), pool pinned to CPUs 29,31, default 50 us slack, loadavg 19 (E1's locked bench):
  legacy polling 86.35 % of one core, futex park 0.16 % (15 workers parked). Production today (5 ms slack): 7.1 % measured from /proc.
* First run() after an idle gap, 2 threads on CPUs 29,31, 200 samples, loadavg 17: legacy min 3 / med 53 / p90 84 us; park min 32 / med 58 / p90 114 us;
  park + 5 ms slack (server setting) med 60 / p90 153 us. Extra cost of parking at the median: ~5 us, at p90 ~30 us (requirement <= 100 us: met).
  16-thread numbers on the even CPUs are queued under the lock (stage 2).
* test_threadpool: all sections pass incl. the new park section (idle_spin 0, 300 dispatches after 0.2/2 ms gaps, n = 2/5/16, worst dispatch 3 ms
  on the oversubscribed dev CPUs, 0 lost wake-ups); legacy mode (INK_POOL_PARK=0) also passes. Hot-path A/B old vs new pool: running now (2 thr dev CPUs), 16 thr queued.
### Tasks 3/4 prepared
* `server/scheduler.py`: env `INK_STEP_TRACE=1` prints per request: engine / gap(ret->next call) / sample / emit / loop-GIL ms (mean, med, p90, max).
* `server/__main__.py`: sched_setaffinity now runs BEFORE numpy is imported and OPENBLAS/OMP/MKL_NUM_THREADS default to 1 (the production process has
  30 idle OpenBLAS threads with mask 0-31 created at import time; they sleep, so no measurable effect today, but they are what the "threads on CPUs
  2, 8, 12, 18, 26" observation was). Verification on the test server in stage 2 (`tools/step_overhead.sh`, port 8010, CPUs 29,31, lock held for its lifetime).

## 18:35 UTC: stage 2 running (lock), server changes written, mock suite green
* Hot-path A/B at 2 threads on CPUs 29,31 (3 alternations, loadavg 3.6): dispatch old 315/340/352 ns vs new 365/385/310; barrier old 258/259/249 vs
  new 264/252/254 -> within the +-40 ns run-to-run noise; the xchg costs at most ~20 ns on paper. 16-thread A/B on the even CPUs is in stage 2.
* Production /metrics (read-only; 6 chat requests, 752 completion tokens, spec auto, ~5.7k-token prompts): decode_engine 60.60 s (80.6 ms/token),
  decode_sampling 0.0123 s (16 us/token, ink_sample on the spec path), decode_other 0.0813 s (108 us/token): the server costs 0.15 % of the engine
  time in production today; last decode 12.45 tok/s at ~5.7k context (bench_decode 14.05 at ctx 16: context, not the server, explains the gap).
* server/*.py: INK_STEP_TRACE instrumentation (scheduler.py), affinity-before-numpy + OPENBLAS_NUM_THREADS=1 (__main__.py), opt-in
  `--sampler c` / INK_SAMPLER=c (backends.py binds ink_sample on its own; scheduler.py use_csampler; app.py config). Default behaviour unchanged.
  Mock suite (server/tests/run_all.py, CPUs 29,31): 72 tests, 0 failures, twice (after the trace and after the sampler flag).
* Queued: stage 2 (lock: prefill/stream SMT cells, 16-thread pool A/B + wake latency + idle CPU on the even CPUs, then the 2-thread test server
  overhead run) and stage 3 (16-thread test server on the even CPUs: numpy vs C sampler at full engine speed, then the C sampler on 29,31).

## 18:42 UTC: waiting for the lock (another agent's 16-thread job held the host at load 13-17 until 18:39); clean A/B builds
* Stage 2 (prefill/stream SMT cells, 16-thread pool A/B, wake, idle), stage 3 (full-speed test server, both samplers) and stage 4 (old vs new pool
  bench_decode + full test_threadpool on real cores) are queued on the lock; flock serialises them, each re-checks loadavg < 1.5 / active 0 / >= 40 GB.
* Confound found and removed: shared sources are being edited by K1/K3/E1 since 18:26 (gemv.cpp, attention.cpp, engine_load.cpp, engine_decode.cpp,
  ...), so a bench_decode relinked at 18:39 would mix their kernels with my pool. I now build BOTH pool variants from the same pristine
  `git archive HEAD` snapshot (build-t/head-old, build-t/head-new = HEAD + my 5 owned C++ files) on CPUs 29,31 and swap the binaries into the
  stage-4 paths atomically (mv). The libink.so used by the stage-2/3 TEST servers was relinked at 18:38 and may include E1's attention.cpp and K1's
  engine_load.cpp of that minute: irrelevant for the server-overhead numbers (absolute ms outside the engine) but noted.
* OpenBLAS mechanism confirmed: `import numpy` under a 2-CPU mask creates 2 threads, under the production's 32-CPU mask 32 (the 30 idle floaters),
  with OPENBLAS_NUM_THREADS=1 one. server/__main__.py now sets the affinity and the cap before numpy is imported. `python -m server --help` ok.
* New scheduler mock test: test_c_sampler_flag_uses_the_backend_sampler_on_the_plain_path_only_without_penalties (PASS; whole test_scheduler.py PASS).

## 18:46 UTC: lock queue; production overhead quantified from the request log
* My three stage scripts waited OUTSIDE the lock for loadavg < 1.5 and starved behind agents who take the lock immediately (E1 gates, K2 test_prefill,
  another 16-thread job at 18:33-18:39 and 18:43). Replaced by `tools/t_locked.sh`: take the lock first, wait inside it (<= 4 min) for the previous
  holder's load to decay below 1.5 (production active 0, >= 40 GB), run one < 10 min session; two sessions (A: SMT prefill/stream, old vs new pool
  bench_decode 16 thr, 16-thread pool A/B, wake/idle, test_threadpool on real cores; B: test servers 2/16 threads x numpy/C sampler). Queued behind
  E1 and K2 at 18:47. (Stopping my own scripts cost me one shell: the kill pattern matched my own command line, exit 144; no other process touched.)
* Clean A/B binaries: both pool variants built from the same `git archive HEAD` snapshot (27 objects each, 30 s per build on CPUs 29,31):
  build-t/bench_decode_oldpool (HEAD pool) vs build-t/src/engine/bench_decode (HEAD + my pool; contains the INK_POOL_PARK switch, the old one does not).
* Task 4 answer from production's own request log (run/requests.jsonl, read-only; 29 requests with >= 8 completion tokens, 4466 tokens, all on the
  spec path): engine 79.65 ms/token, sampling 0.011 ms, everything else 0.086 ms -> the server costs 0.12 % of the engine time. A 27-token-prompt
  request decoded at 14.14 tok/s = bench_decode's 14.05 (ctx 16, same day): no measurable API-vs-bench gap; the 11.7-12.6 tok/s of the other requests
  are 7.4-9.6k-token contexts (attention), not server cost. The 30 floating OpenBLAS threads sleep (0.0 % CPU) and are now prevented at start-up
  (affinity before the numpy import + OPENBLAS_NUM_THREADS=1); the engine thread (tid 0) is pinned to CPU 2 and the main/uvicorn threads stay on the odd CPUs.
* numpy Sampler cost on CPU 29 (random 200k logits, 200 draws): greedy 0.045 ms, T=0.6 0.92, T=1.0 0.89, top_k 40 0.86, top_p 0.9 1.17, min_p 0.79,
  repetition_penalty 1.1 1.16 ms/token; pieces of T=0.6: subtract/multiply/floor 0.38, exp 0.20, block sums 0.18, max 0.02. = 1.2-1.6 % of a 73 ms token
  at 16 threads for sampled requests, 0.06 % greedy. The opt-in C sampler (ink_sample) is measured in session B.

## 18:54 UTC: lock etiquette (orchestrator tip, please adopt): do NOT wait for the 1-minute loadavg to decay while holding the lock
The "loadavg < 1.5" rule means "nothing else is running". Once you hold the lock and the previous holder has exited, check instantaneous idleness:
two reads of /proc/stat 2 s apart with < 10 % non-idle CPU (or `ps -eo pcpu --sort=-pcpu | head -3` showing nothing of ours above a few percent),
production `"active":0`, >= 40 GB available, then start; still write the loadavg at the start of the run next to your numbers. Waiting for the
average to decay costs everyone ~2.5 min of lock time per session. Implemented in tools/t_locked.sh (quiet_inside) and tools/smt_sweep.sh (precheck):
`a=$(awk '/^cpu /{print $2+$3+$4+$7+$8+$9, $5+$6}' /proc/stat); sleep 2; b=$(...same...); busy% = 100*(nonidle_b-nonidle_a)/(total_b-total_a)`.

## 19:02 UTC: session A results (lock held 18:52-18:59; the run itself started 18:56:21 at loadavg 1.96 after the old 4-min decay wait capped out,
## instantaneous CPU was 0 % busy at 18:54; loadavg values next to the cells below are inflated by my own 20-32 spinning threads)
### Task 1 (prefill + stream), SMT list 2,4,..,30,0 then odd siblings 3..17 then 19..31,1 (`INK_PROFILE=1 bench_prefill --prompts 256,512 --reps 2`)
| prompt | 16 thr | 20 thr | 24 thr | 32 thr |
|---|---|---|---|---|
| 256 | 82.3 tok/s (81.6 82.3) | 72.3 (72.3 68.8) | 71.4 (71.4 70.9) | 74.4 (70.8 74.4) |
| 512 | 77.9 (77.9 70.5) | 71.1 (71.1 69.6) | 73.2 (71.1 73.2) | 73.7 (71.8 73.7) |
Per phase (256-token prompt, ms/token): gate_up GEMM 5.16 / 6.00 / 5.88 / 5.22; down GEMM 2.79 / 3.32 / 3.14 / 2.78; proj GEMM 2.03 / 2.47 / 2.35 / 2.17;
wo 0.82 / 0.95 / 0.97 / 0.89; TOTAL 12.13 / 14.53 / 14.09 / 13.44. Busy-time bound printed by the profiler: 32 threads could reach 92 tok/s if no
thread ever waited (actual 74.4): the extra threads add barrier/straggler waits, and the FMA units are already saturated by one thread per core
(32 threads = 16 threads on the GEMM phases, 20/24 threads are slower because the cores with two threads straggle). SMT does not help prefill.
bench_stream (3 GB, same list): free stream 80.7 GB/s at 16 threads, 83.0 at 24, 77.9 at 32; barrier-separated 3-96 MB phases: 70.7-83.1 GB/s at 16,
50.2-66.3 at 24 (+25-305 us per phase), 36.5-63.4 at 32 (+46-295 us). => more threads than cores hurt the phase-structured streaming badly.
VERDICT task 1: keep one thread per physical core (16). Decode +1.6 % at 24 threads is noise-level, prefill -5..-13 %, 32 threads lose everywhere.
No mixed-layout pool primitive is implemented (evidence: docs/requests.md T item 1).
### Task 2 gates (same session, 16 threads on the even CPUs)
* Decode, old pool vs futex-park pool, SAME pristine-HEAD sources, ctx 16, 64 tokens x 2 reps, alternating: old 14.09 (13.89 14.09) vs new 14.07
  (14.07 13.79); second pair old 12.65 (12.52 12.65) vs new 13.59 (12.86 13.59) (loadavg 23-26 from my own preceding 32-thread cells). No regression.
* Pool hot path (bench_pool, min/median ns, 3 alternations): dispatch old 1260/1483, 973/1358, 1028/1144 vs new 1148/1464, 1093/1300, 935/1139;
  barrier old 1039/1369, 928/1212, 949/1195 vs new 972/1174, 1045/1191, 1080/1205 -> equal within the +-10 % run-to-run noise (B1 measured 856/810
  min at load 10.8; the box was at load 14-22 here). 0-8 futex wakes during 220k back-to-back runs (a preempted worker lets the others park briefly).
* First run() after an idle gap, 16 threads, 300 samples: legacy polling (50 us slack) min 14 / median 81 / p90 1524 / max 7271 us; futex park
  min 64 / median 183 / p90 283 / max 9941 us. Median +102 us (the vCPUs go truly idle and need an IPI), p90 -1.2 ms. Production's 5 ms slack
  setting makes the legacy wake up to 5 ms, so parking is better than what runs today; the requirement (<= 100 us extra) is met at p90, at the
  median it is +102 us: borderline, once per request, 0.26 % of one token.
* Idle CPU, 16-thread pool on real cores, 10 s: legacy polling 120.1 % of one core (default slack) vs park 0.12 %.
* test_threadpool on real cores (default pinning): all ok in park mode and in legacy mode.

## 19:30 UTC: session B results (test servers on port 8010, lock held 19:21-19:28, INK_STEP_TRACE=1, libink = build-t 18:38 relink, INK_SPEC=off)
Per-token time OUTSIDE the engine (gap = ink_forward return -> next ink_forward call; mean over 46/94 steps; ms):
| server | request | engine ms/tok | gap | sample | emit | loop/GIL | overhead |
|---|---|---|---|---|---|---|---|
| 2 thr 29,31 numpy | greedy non-stream | 294.1 | 0.221 | 0.144 | 0.039 | 0.040 | 0.07 % |
| 2 thr numpy | greedy stream | 292.5 | 0.249 | 0.138 | 0.075 | 0.036 | 0.08 % |
| 2 thr numpy | T=0.6 stream | 292.8 | 1.235 | 1.143 | 0.069 | 0.039 | 0.41 % |
| 2 thr numpy | T=0.6 top_k 40 top_p 0.9 | 292.0 | 1.278 | 1.181 | 0.068 | 0.038 | 0.43 % |
| 2 thr numpy | T=0.6 repetition_penalty 1.1 | 291.3 | 1.536 | 1.440 | 0.069 | 0.039 | 0.52 % |
| 2 thr numpy | chat greedy stream | 294.0 | 0.258 | 0.140 | 0.080 | 0.038 | 0.09 % |
| 16 thr even CPUs numpy | greedy non-stream (96 tok) | 75.4 (med 68.7, max 266) | 0.210 | 0.138 | 0.038 | 0.036 | 0.28 % |
| 16 thr numpy | greedy stream | 84.5 | 0.355 | 0.211 | 0.098 | 0.046 | 0.42 % |
| 16 thr numpy | T=0.6 stream | 69.5 | 1.219 | 1.122 | 0.067 | 0.037 | 1.73 % |
| 16 thr numpy | T=0.6 top_k/top_p | 77.3 | 1.291 | 1.190 | 0.067 | 0.037 | 1.65 % |
| 16 thr numpy | T=0.6 penalty | 75.1 | 1.484 | 1.384 | 0.068 | 0.038 | 1.95 % |
| 16 thr numpy | chat greedy stream | 77.3 | 0.251 | 0.136 | 0.078 | 0.036 | 0.32 % |
| 16 thr C sampler (ink_sample) | greedy non-stream | 71.7 | 0.280 | 0.206 | 0.037 | 0.045 | 0.39 % |
| 16 thr C | T=0.6 stream | 73.4 | 1.877 | 1.775 | 0.070 | 0.037 | 2.53 % |
| 16 thr C | T=0.6 top_k/top_p | 73.1 | 1.686 | 1.573 | 0.075 | 0.037 | 2.28 % |
| 16 thr C | T=0.6 penalty (numpy fallback) | 75.6 | 1.488 | 1.392 | 0.069 | 0.040 | 1.95 % |
| 2 thr C | greedy / T=0.6 / top_k,p | 292.9 / 293.8 / 293.3 | 0.254 / 1.817 / 1.632 | 0.199 / 1.715 / 1.515 | | | |
Reading: greedy requests cost 0.21-0.36 ms/token of server time (0.3-0.5 % of a 71 ms token at 16 threads), sampled requests 1.2-1.5 ms (1.7-2.0 %), and
all of it is the sampler: emit (parser + stop strings + SSE hand-off) is 0.04-0.10 ms, the scheduler loop + GIL 0.036-0.046 ms with no outliers
(max 0.07 ms): no GIL stalls, no asyncio hand-off cost worth fixing, SSE batching would save < 0.05 ms. The greedy 0.14 ms is np.argmax over the
800 KB logits row that the engine's other cores just wrote (cross-core traffic; np.argmax on a warm row is 0.045 ms). The C sampler (S2's
ink_sample) is SLOWER than numpy here (T=0.6 1.78 vs 1.14 ms, greedy 0.21 vs 0.14): numpy stays the default; `--sampler c` remains opt-in only.
API speed vs bench: 16-thread greedy 96 tokens through HTTP = 13.37 tok/s (numpy run; engine mean 74.6 ms with preemption spikes to 266 ms,
loadavg 4-11 from other agents' dev work during the lock session) and 14.03 tok/s (C run, engine 71.0 ms) vs bench_decode 14.05-14.09 in session A:
the server's share is the 0.3 % above; the rest is host noise. Thread placement on the test servers: engine threads pinned 1:1, every other thread
masked to the non-engine CPUs, no thread with mask 0-31 any more (OpenBLAS fix verified). Idle test server: 0.2 % (2 threads) / 0.3 % (16 threads)
of one core with the parking pool (production today with polling + 5 ms slack: 7.1 %).

## FINAL (written 19:34 UTC; gate line amended when the last run finishes)
### Result
* SMT (task 1): one thread per physical core stays optimal. Decode ctx 16: 14.05 / 13.88 / 14.28 / 10.95 tok/s at 16 / 20 / 24 / 32 threads (ctx 1000:
  12.80 / 13.57 / 13.19 / 11.68); prefill 256: 82.3 / 72.3 / 71.4 / 74.4, 512: 77.9 / 71.1 / 73.2 / 73.7; stream phases 71-83 GB/s at 16 thr vs 50-66 at 24,
  37-63 at 32. Per phase at 24 thr: gate_up -6 %, proj -6 %, lm_head -4 %, down 0, serial/router/attention +5..22 %: net noise-level; 32 thr convoy.
  Hypothesis confirmed only weakly for gate_up; no mixed-layout pool primitive implemented (would be <= 3 % even if free elsewhere, needs all 32 vCPUs).
* Futex parking (task 2): idle 16-thread pool 86-120 % of one core (polling, 50 us slack) / 7.1 % (production's 5 ms slack) -> 0.12-0.16 %; test
  servers idle at 0.2-0.3 % of a core. Hot path unchanged within noise (dispatch/barrier min 935-1148 vs 973-1260 ns old at 16 thr; decode 14.07 vs
  14.09 tok/s same-source A/B). First dispatch after idle: median 183 us vs 81 us polling (p90 283 vs 1524 us) at 16 thr; 2 thr: 58 vs 53 us.
* Server overhead (task 3): 0.21-0.36 ms per greedy token (0.3-0.5 % of a 71 ms token), 1.2-1.5 ms per sampled token (1.7-2 %), all in the numpy
  sampler; emit 0.04-0.10 ms, scheduler loop + GIL 0.036-0.046 ms without outliers. Production request log: 0.12 % of engine time. Nothing cheap
  left to fix in Python; S2's C sampler is slower than numpy (request to K2). API tok/s == bench_decode tok/s within host noise (13.4-14.0 vs 14.05-14.09).
* Thread placement (task 4): the 30 floating threads in production are numpy/OpenBLAS workers created at import time (mask 0-31, asleep, 0 % CPU):
  no measurable decode effect; fixed at start-up anyway (affinity before the numpy import, OPENBLAS_NUM_THREADS=1): test servers have no 0-31 thread.
### What is wired in
* DEFAULT ON: futex parking in `ink::ThreadPool` (`src/kernels_gemv/threadpool.cpp`, `include/ink/threadpool.h`; `INK_POOL_PARK=0` = legacy polling);
  server start-up hygiene in `server/__main__.py` (affinity before numpy, BLAS thread caps). Behaviour-preserving instrumentation in `server/scheduler.py`
  (`INK_STEP_TRACE=1` prints a per-request step trace; off by default).
* OPT-IN: `python -m server --sampler c` / `INK_SAMPLER=c` (plain-path sampling via `ink_sample`; `server/backends.py` binds the symbol on its own,
  `server/app.py` ServerConfig.c_sampler, `server/scheduler.py` use_csampler). Default numpy, unchanged tokens for seeded requests.
* Files touched: src/kernels_gemv/threadpool.cpp, include/ink/threadpool.h, tests/kernels_gemv/test_threadpool.cpp (park section, INK_TEST_CPUS),
  tests/kernels_gemv/bench_pool.cpp (--cpus/--spin/--idle/--wake), server/scheduler.py, server/backends.py, server/app.py, server/__main__.py,
  server/tests/test_scheduler.py (+1 test), docs/requests.md (T items 1-7), docs/status/T.md; new tools/smt_sweep.sh, tools/step_overhead.sh,
  tools/t_locked.sh, tools/t_parse_sweep.py. Not touched: bench/bench_stream.cpp, engine/kernels, serve.sh, docs of other owners.
### Gates
* test_threadpool: all ok on the dev CPUs (INK_TEST_CPUS=29,31) and on real cores (default pinning), park and legacy modes; the park section proves
  workers park before every dispatch after 0.2/2 ms gaps and never need the 250 ms safety timeout (300 dispatches x n = 2/5/16).
* bench_pool old vs new, 2 thr dev CPUs (3 alternations) and 16 thr even CPUs (3 alternations): within +-10 % noise, no systematic change.
* bench_decode old pool vs new pool, SAME pristine-HEAD sources (build-t/head-old vs head-new), 16 thr ctx 16: 14.09 vs 14.07 tok/s (2nd pair
  12.65 vs 13.59, noisy): no regression. Numerics are untouched by construction (the pool change is idle-path only; same splits, same kernels).
* server mock suite (server/tests/run_all.py): 72 tests, 0 failures after each server edit; new test_c_sampler_flag... PASS; `python -m server --help` ok.
* Real-engine server tests a,b,d,h (server/tests/test_ink_real.py) with the clean HEAD+pool libink (build-t/head-new) on CPUs 29,31 under the lock,
  19:36-19:44: 4/4 PASS. a: greedy through the server == raw libink == llama.cpp oracle; b: chat fields; d: two-turn cache reuse; h: RSS 16.18 GiB
  after load, idle CPU 0.25 % of one core (V measured 2.0 % with the prctl, 30 % without), http overhead 0.191 ms/token greedy, sampling 0.147 ms.
  Raw JSON: build-t/real/ink_real_results.json, log build-t/real/test_ink_real.log. No test server left running (port 8010 stopped).
### Open
* The 24-thread decode cell (+1.6 %) deserves one idle-host repeat if the orchestrator wants the last 1-2 %; prefer a list without CPU 1 (slow core's sibling).
* Futex wake costs ~+100 us at the median after a true idle gap (IPI to halted vCPUs); fine for an LLM server (once per request), could be hidden by a
  pre-wake hint from the server when a request is admitted if ever needed.
* docs to update by their owners: KERNELS_GEMV.md 2, ENGINE_DESIGN.md 2 (idle behaviour), ENGINE.md open item 2 (done), SERVING.md 9 (idle CPU now
  0.2-0.3 % without the prctl; `--timer-slack-us` is harmless but no longer needed).
* K2: a vectorised C sampler would make `--sampler c` worth defaulting (1.7-2 % on sampled requests); E1: pool-side argmax (0.15 %).
* Orchestrator's final idle-host run: decode 16 thr ctx 16/1000 and prefill 256/512 with the new pool should reproduce 14.0-14.3 / 82 tok/s; also
  read `ps` idle CPU of the restarted production server (expect ~0.3 % of a core instead of 7 %).

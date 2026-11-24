# Optimisation wave 1 (2026-11-23): contract for agents K1, K2, K3, E1, T, R

Goal: squeeze more speed out of inkcpu on THIS host without changing what it computes (parity with llama.cpp on the same GGUF stays
the gate; bit-identity between prefill, verify and decode stays unless a change is explicitly flagged "numerics-deviating" and left OFF by default).
Baseline (idle host, 2026-11-22, `docs/ENGINE.md`): decode 13.9 tok/s @12 threads / 14.3 @16 (ctx 16), 13.2 / 14.0 (ctx 1000); prefill 256 tok
67.9 @12 / 82.7 @16; speculative auto 13.7 vs plain 13.9 (verify T=4 costs 2.6x a plain step). Per-phase decode profile: `ENGINE_DESIGN.md` 9.

Read first: `docs/CONTRACT.md` (machine facts, rules), `docs/ENGINE.md`, `docs/ENGINE_DESIGN.md` (esp. 5, 9, 10), the docs of your area, your
predecessor's `docs/status/*.md`. Everything in CONTRACT.md still applies (PATH=/usr/bin:/bin, no -march=native, no sudo/perf, /tmp noexec,
never kill processes you did not start, never `pkill -f`).

## Host sharing rules (strict)
* The PRODUCTION server (pid 1032707, `tools/serve.sh status`) is running on the even CPUs 2,4,...,30,0 with 16 threads and 18 GB RSS. It idles at
  ~0% CPU. Do NOT stop, restart or signal it, and do not touch /data/inkling-mini/models* or server/run state. Test servers: other INK_PORT, odd CPUs.
* Development / correctness runs: pin to YOUR odd CPUs (`INK_CPUS=...`, `taskset -c ...`) with that many threads, under `nice -n 10`:
  K1: 1,3,5,7   K2: 9,11,13,15   K3: 17,19,21,23   E1: 25,27   T: 29,31   (R: no CPU work).
* Timing runs: ONLY under `tools/with_bench_lock.sh <cmd>` (exclusive lock, 30 min wait), on the default CPU list (even CPUs), and only when
  `cat /proc/loadavg` 1-min < 1.5 and `tools/serve.sh status` shows `"active":0`. Keep each timing run < 10 min. Write the loadavg next to every
  number you report. Kernel microbenchmarks with synthetic weights on your odd CPUs do not need the lock but ARE perturbed by other agents: quote them as relative (A/B in the same window).
* Memory: a full-model process peaks at ~34 GB during load. EVERY process that loads the full GGUF (test_engine, test_prefill, bench_*, test servers)
  must run under `tools/with_bench_lock.sh` (also non-timing gates) so at most one full model besides production is resident. Check `free -g` (>= 40 GB available) first.
  Iterate with synthetic-weight kernel tests/benches; reserve full-model runs for gates and milestones.
* Build dir: `build-<agent>` (e.g. `cmake -S . -B build-k1 -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build-k1 <targets>`). Never build in build-orch.
* No git commits (the orchestrator commits). Edit only the files you own (below). If you need a change elsewhere, write it in `docs/requests.md`
  (append, with your agent name and UTC time) and continue with a local workaround. Re-read a shared file right before editing it (another agent may have edited another function).
* Status: `docs/status/<AGENT>.md`, append a dated entry at least every 20 minutes (what you tried, numbers with loadavg, what is next). Hard time box:
  3 hours of wall time from your start, then write the final section ("Result / What is wired in / Gates / Open") and return. Interim findings early beat a perfect report late.

## Ownership (wave 1)
| agent | owns (may edit) | must not edit |
|---|---|---|
| K1 | `src/kernels_gemv/gemv.cpp`, `gemv_ref.cpp`, `include/ink/kernels_gemv.h`, `include/ink/nvfp4_repack.h`, `tests/kernels_gemv/*`; the expert-tensor copy/repack hook in `src/engine/engine_load.cpp` (only the function(s) that copy expert tensors + the `LayerPacked` flag in `include/ink/engine.h`) | engine_decode.cpp, engine_prefill.cpp, kernels_gemm |
| K2 | NEW `src/kernels_gemv/gemv_multix.cpp` + `include/ink/kernels_gemv_multix.h` (+ tests/bench files with `multix` in the name), the MoE phases G/H of `src/engine/engine_prefill.cpp` (kernel choice per expert by pair count, verify path), `src/spec/*`, `bench/bench_spec.cpp`, `tests/spec/*` | gemv.cpp, gemm.cpp, engine_decode.cpp |
| K3 | `src/kernels_gemm/*`, `include/ink/kernels_gemm.h`, `tests/kernels_gemm/*` | engine_prefill.cpp (ask K2 via requests.md for wiring), gemv.cpp |
| E1 | `src/engine/engine_decode.cpp`, `src/engine/engine_priv.h`, `src/kernels_misc/*`, `include/ink/kernels_misc*.h`, `tests/kernels_misc/*`, `bench/bench_decode.cpp`; `Engine::alloc_slot` / KV allocation in `engine_load.cpp` (that function only) | gemv/gemm kernels, engine_prefill.cpp |
| T | `src/kernels_gemv/threadpool.cpp`, `include/ink/threadpool.h`, `tests/kernels_gemv/test_threadpool.cpp`, `bench_pool.cpp`, `bench/bench_stream.cpp`, `server/*.py` (test instances only), `tools/*.sh` (new scripts) | engine sources, kernels |
| R | `docs/RESEARCH_WAVE1.md` only | everything else |

## Shared decision already taken: NVFP4 scale re-encoding (`include/ink/nvfp4_repack.h`)
The engine's arena copy of the NVFP4 expert tensors will have its 4 scale bytes per super-block re-encoded at load (K1 does the loader hook) so that
16 scales decode with 3 vector uops instead of 16 table loads, bit-identical to `ue4m3_to_f32` (self-check in the header passes for all 256 bytes).
Kernels that read arena weights after the repack must use `nvfp4r_*` decoding (distinct entry points: `gemv_nvfp4r`, `gemm_moe_nvfp4r`, ...);
the GGUF-encoded kernels stay for tests/oracle paths. `LayerPacked` gets a flag (K1) telling the executors which encoding the arena holds; until K1's hook
lands, K2/K3 develop their `nvfp4r` kernels against synthetic re-encoded weights (`nvfp4r_repack_rows` on random UE4M3 data) and keep the old entry points working.

## Reporting format (end of your status file and your return message)
1. Result: measured numbers (before/after, same window, loadavg, thread count, context) for the metric you own.
2. What is wired in by default vs. behind a flag (env/Tune), and exact file list touched.
3. Gates run and their outcome (bit-identity vs reference, oracle/greedy/nll/determinism where applicable).
4. Open items, risks, and what the orchestrator should measure in the final idle-host run.

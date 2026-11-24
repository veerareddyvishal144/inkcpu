# Optimisation wave 2 (2026-11-24): agents L (session-lookup drafter) and P2 (int16 prefill GEMM)

Owner decision (2026-11-24): build both items 2 and 3 of docs/RESEARCH_WAVE2.md. Item 3 may change the float accumulation order of PROMPT
PROCESSING (prefill chunks) only; speculative verify and decode stay bit-identical to today, so speculation remains token-identical to plain decode.

All rules of docs/OPT_WAVE1.md apply (PATH=/usr/bin:/bin, no -march=native, own build dir, status file every 20 min, no git commits, never touch the
production server pid in run/server.pid or /data/inkling-mini/models*, never kill processes you did not start, never pkill -f). Updated lock etiquette: every
process that loads the full model runs under `tools/with_bench_lock.sh`; when you hold the lock, check INSTANTANEOUS idleness (two /proc/stat reads 2 s apart,
< 10 % non-idle) and start at once; never hold the lock while sleeping on the 1-minute loadavg; keep each locked run < 10 min; `free -g` >= 40 GB available first.
Baseline for both agents = `build-final` (commit 5be85af): decode 14.7-15.8 tok/s @16 threads, prefill 86.8 / 83.5 / 82.3 tok/s at 256 / 512 / 1024 tokens (chunk 256),
speculation 12 threads pooled plain 15.28 / T=4 13.28 / T=8 9.78 / auto 15.43, verify T=4 = 2.15x plain, T=8 = 3.43x, kVerifyMaxRows = 8, PrefillTune::multix = 8.
Production runs build-final through the `build` symlink; do not rebuild `build-final` or `build`.

| agent | owns (may edit) | dev CPUs | build dir | must not edit |
|---|---|---|---|---|
| L | `src/spec/*`, `include/ink/spec.h`, `src/capi/ink_capi.cpp` + `include/ink/ink_abi.h` (additive hooks only), `bench/bench_spec.cpp`, `tests/spec/*`, `server/backends.py` + `server/scheduler.py` (only if a new ABI call is needed), `docs/SPEC_DESIGN.md` | 1,3,5,7,9,11,13,15 | build-l | engine executors, kernels |
| P2 | `src/kernels_gemm/*`, `include/ink/kernels_gemm.h`, `tests/kernels_gemm/*`, the kernel selection in phases G/H of `src/engine/engine_prefill.cpp` (and a PrefillTune flag in `include/ink/engine_prefill.h`), `docs/KERNELS_GEMM.md` | 17,19,21,23,25,27,29,31 | build-p2 | spec, capi, decode executor, gemv kernels |

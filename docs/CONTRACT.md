# Contract for everyone working on inkcpu (read this first)

Goal: a single-model CPU inference engine for Inkling 2 Mini (arch `inkling2_loop`) that runs the community GGUF
`/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf` faster than tuned llama.cpp on THIS machine. No GPU, ever.
Targets (see ENGINE.md): decode >= 2x tuned llama.cpp (>= 11.5 tok/s, stretch 17), prefill >= 3x (>= 70 tok/s),
logits parity with llama.cpp on the same GGUF.

## Machine facts (measured)
- Azure VM, Xeon Platinum 8171M, 1 socket, 16 physical cores / 32 vCPUs; the two hyperthreads of physical core k are
  CPUs 2k and 2k+1. 1 NUMA node. 125 GB RAM. L3 35.8 MiB. Measured streaming read bandwidth: 92.5 GB/s at 16 threads
  (65 at 8, 82 at 32). One thread per physical core (even CPUs) is the intended layout.
- ISA: AVX-512 F/BW/DQ/VL, AVX2, FMA, F16C ONLY. NO AVX512-VNNI, NO AVX512-BF16, NO AMX, NO VBMI. Any of those = SIGILL.
  Build flags are set in the root CMakeLists (`-march=skylake-avx512`). Do not use `-march=native`.
- No sudo, `perf` is blocked (perf_event_paranoid=4). Use the sampling profiler `plan/tools/libprof.so` +
  `plan/tools/prof_report.py` (LD_PRELOAD, see plan/tools/prof.c) or manual timing.
- `/tmp` is noexec: build and run everything under `/data/inkling-mini/engine` (or its build dirs). Scratch: the session
  scratchpad is fine for text files only.
- Use the system toolchain: `export PATH=/usr/bin:/bin` before cmake/gcc (Homebrew's binutils/libgomp break linking).
  gcc 11.4, cmake 3.22, ninja 1.10 are installed. Python: use `/data/engine/.venv/bin/python` (numpy, fastapi, uvicorn,
  tokenizers, jinja2 present). Do NOT modify anything under /data/engine (read-only reference).
- The host is SHARED with other sessions and with the other agents building this engine in parallel. Check
  `cat /proc/loadavg` before any timing and say so in your report; timing numbers taken at load > 2 are indicative only.
  Never kill processes you did not start. Never use `pkill -f`.

## Reference material
- Model facts, tensor names, bytes/token: `/data/inkling-mini/plan/ARCH.md` (read sections 1-5 and S3, S4).
- llama.cpp checkout (our spec and oracle): `/data/inkling-mini/llama.cpp` @2a28df0f9 + local node-profiler patch in
  `ggml/src/ggml-cpu/ggml-cpu.c`. Stock baseline binaries: `build-sys/bin/` (DO NOT rebuild or modify build-sys).
  Profiler variant: `build-nprof/bin/` (GGML_NODE_PROF=1 prints per-op wall time). Model graph: `src/models/inkling2.cpp`,
  `src/models/inkling2-loop.cpp`, MoE/SiTU in `src/llama-graph.cpp`, CPU kernels `ggml/src/ggml-cpu/quants.c`,
  `ggml/src/ggml-cpu/arch/x86/*`.
- Phase 0 measurements: `docs/ENGINE.md`.

## Repository layout and ownership (edit only your own directory; ask via docs/requests.md for anything shared)
| path | owner | content |
|---|---|---|
| include/ink/ggml_types.h, include/ink/ink_abi.h, CMakeLists.txt, docs/ENGINE.md, docs/CONTRACT.md | orchestrator | shared contract |
| src/core/, include/ink/model.h, include/ink/gguf.h, docs/MODEL.md, tests/core/ | agent A | GGUF reader, config/weights, scalar reference forward |
| src/kernels_gemv/, include/ink/threadpool.h, include/ink/kernels_gemv.h, docs/KERNELS_GEMV.md, tests/kernels_gemv/ | agent B1 | thread pool, quantize, GEMV kernels |
| src/kernels_misc/, include/ink/kernels_misc.h, docs/KERNELS_MISC.md, tests/kernels_misc/ | agent B2 | vector math, norms, attention, conv, router kernels |
| tools/, docs/ROUTING.md, docs/BASELINE.md | agent E | oracle tools, routing trace, benchmark harness |
| server/ | agent D | Python OpenAI-compatible server |
| src/engine/, src/capi/ | later (executor) | optimized forward + C ABI |
Each module has its own `CMakeLists.txt` (the root adds it if present) building a static library
`ink_<module>` plus test executables registered with `add_test`. Tests are plain C++ `assert`-style executables returning
non-zero on failure (no external frameworks, no network).

## Conventions
- Weights are read in GGUF-native layouts (rows are contiguous along K; Q8_0 row = K/32 blocks of 34 bytes; NVFP4 row =
  K/64 blocks of 36 bytes). A kernel may require a load-time repack; if so provide `repack_*` and test it.
- Activation format for Q8_0 and NVFP4 weights is Q8_0 blocks of the F32 activation, exactly as ggml does it
  (`quantize_row_q8_0_ref`: d = amax/127, fp16 rounding of d, q = roundf(x/d)). This keeps logits close to llama.cpp.
- Kernels are single-threaded functions over a row range; the executor splits rows across the thread pool.
- Every optimized kernel ships with (a) a scalar reference implementation in the same library and (b) a randomised unit test
  comparing them over odd/edge shapes (tolerances stated in the test), and (c) a microbenchmark where it is
  bandwidth-bound (report achieved GB/s and the % of the 92.5 GB/s streaming ceiling).
- Do not run git commands that modify the repository: the orchestrator commits. Leave files in place.
- Progress protocol: write `docs/status/<your-role>.md` at the start (plan) and after each milestone (what works, numbers,
  open problems). The orchestrator reads these while you run. Your final reply must be <= 300 words and name the files that
  matter. If you are blocked on another module, say so in your status file and continue with what you can.
- Honest numbers only: report measured values with the load average at the time; mark estimates as estimates.

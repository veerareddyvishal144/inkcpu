# P status (batched prefill executor)

Started 2026-11-21 ~22:20 UTC. Budget 130 tool calls / ~130 min. Build: `export PATH=/usr/bin:/bin; cmake -S . -B build-p2 -G Ninja && ninja -C build-p2 test_prefill bench_prefill`
(root configure works now that src/kernels_gemm has sources; `build-p`, `build-p2` are mine). Design, numbers, limits: docs/PREFILL_DESIGN.md. Requests to C1: docs/requests.md (end of file).

## Files
`include/ink/engine_prefill.h`, `src/engine/engine_prefill.cpp` (+ block appended to `src/engine/CMakeLists.txt`, conditional on `TARGET ink_kernels_gemm`, adds engine_prefill.cpp to
ink_engine, links ink_kernels_gemm, builds test_prefill / bench_prefill), `tests/engine/test_prefill.cpp`, `bench/bench_prefill.cpp`, `docs/PREFILL_DESIGN.md`.
No edits to engine.h / engine_priv.h / engine_load.cpp / engine_decode.cpp (C1 had already made the SWA ring 32 blocks).

## Log
- [x] first correct chunk (~22:42): s1_en_capital, 12 threads, logits BIT-IDENTICAL to token-by-token decode in both attention modes, conv histories and all 48 x (K,V) fp16 caches bit-identical.
- [x] gate (a) oracle prompts (6 prompts 13-52 tokens + 300-token wikitext chunk): prefill logits == decode path BIT FOR BIT in both attention modes (maxabs 0); vs llama.cpp identical to decode
  (cos 0.9917-0.99998, same argmax 6/6, same top-10 overlap).
- [x] gate (b) state: prefill 43 tokens then 24 decode steps: 24/24 bit-identical, conv hist and KV fp16 bit-identical after prefill and after the decode steps (fast and decode-kernel attention).
  (700 / 1000-token variants: see PREFILL_DESIGN.md section 5.)
- [x] gate (c) chunk sizes 1, 2, 7, 64, 256 on the 300-token prompt: all BIT-IDENTICAL to the all-decode logits. Ring wrap (1305 and 1500 tokens): decode-kernel attention is bit-identical for
  chunk 64/256/1000; fast attention within rounding (cos 0.9997 / 0.995 vs decode kernels, and as close to llama.cpp as the decode kernels are: 0.99931 vs 0.99937 on 1305 tokens, 0.99550 vs 0.99243 on
  1500 tokens, same argmax).
- [!] 2200-token repeated-text prompt vs llama.cpp (`test_prefill long repeat2200`, past the 2048 ring wrap): decode path (decode-kernel mode, bit-identical to decode) cosine 0.977, fast prefill 0.965, argmax equal in all
  three; so the long-context deviation from llama.cpp is shared with the decode path (not prefill specific). Open for C1/B2: a REAL-text reference beyond 1305 tokens (only the repeated ring text was tried).
- [x] gate (d) NLL, 2 x 256 wikitext tokens, engine_prefill_all vs llama.cpp logits_all: |delta| 0.00155 and 0.00680 (limit 0.01), argmax mismatches 5/256, 3/256.
- [x] gate (e) determinism: 300-token prefill bit-identical across repeated runs and thread counts 16/12/8/5.
- [x] the fast attention kernel (`attn_prefill_group`) equals the decode kernel BIT FOR BIT up to 767 visible keys (`test_prefill split`: logits and the whole state); from 768 keys the decode path
  splits the keys of global layers (nsplit >= 2) and differs by rounding, amplified by Q8_0 re-quantisation / top-8 flips (768 tokens: cos 0.99996; 900: 0.99992; 1305: 0.9997; 1500: 0.995).
- [x] bench (tools/with_bench_lock.sh, best of 2, final code; run 1 at host load 14-16, run 2 at load 8-14 = indicative only; stock llama.cpp 19.0 tok/s at 12 threads, 23.5 at 16, 256-token prompt):
  | threads | 16 tok | 64 | 256 | 1024 (chunk 256) | 1024, chunk 512 | 1024, chunk 1024 |
  | 12 | 32.2 / 36.7 | 50.5 / 55.4 | 66.7 / 66.7 | 66.6 / 65.3 | 69.1 | 69.6 |
  | 16 | 44.3 / 43.7 | 68.4 / 67.5 | 83.6 / 78.9 | 78.5 / 75.4 | 84.1 | 87.2 |
  (run 1 / run 2). 70 tok/s target on 256 tokens: met at 16 threads, 5% short at 12 threads. Idle-host numbers would be higher (ESTIMATE 75-80 at 12 threads, 95-100 at 16).
- [x] profile 16 threads, 256 tokens, load 13: 83.9 tok/s, busy-time bound 87.1; MoE 66% (gate_up 25.2 / down 23.0 GMAC/s per busy thread), proj 16.9% (30.9), wo 6.1% (31.0), controllers 4.7% (7.2),
  everything else < 5% together. Same code at load 26 (12 threads): 31 tok/s, per-thread rates 17 / 14 / 12.5 GMAC/s. `bench_peak` at load 30: pure-ALU 5-uop loop 25 GMAC/s per thread (idle host ~54-61).

## Optimisations (before/after, host noise +-10% so only the structural ones count)
- BF16/F32 GEMMs (router, sink, dyn taps, controllers) in token groups of ~24 (activation group stays in L2): router 0.528 -> 0.300 ms/token at load 21-26.
- chunk 256 -> 512 on a 512-token prompt: MoE per-thread rate +6-8% (more pairs per expert), 31.3 -> 32.8 tok/s at load 26; 1024-token prompt at 16 threads 78.5 -> 84.0 tok/s.
- thread CPU-time accounting in the profile (busy ms per phase, GMAC/s per busy thread) to separate host preemption from kernel speed.

## Integration notes (for the orchestrator's Engine::forward wiring)
- `#include "ink/engine_prefill.h"`; `engine_prefill(e, slot, tokens, n, logits_last)` (n >= 1; use the decode path for n == 1 if preferred), `engine_prefill_all` (logits of every token),
  `engine_prefill_release(e)` from `~Engine` (scratch registry is keyed by `Engine*`; without the call the 150-600 MB arena lives until process exit).
- `ink_engine` needs `ink_kernels_gemm`: the CMake block in src/engine/CMakeLists.txt does it when the target exists (root build); src/engine/standalone/CMakeLists.txt has no gemm target
  (its configure still works, the prefill sources are simply not compiled there).
- Defaults (env, read once at first use; `prefill_tune()` to change in code): INK_PF_CHUNK=256 (max 1024; 512-1024 is +5-15% on prompts >= 512 tokens), INK_PF_PROJ=32, INK_PF_WO=16, INK_PF_GU=32,
  INK_PF_DN=32, INK_PF_TOK=4, INK_PF_ATTN_TQ=8, INK_PF_GROUP=0 (keep G's 64), INK_PF_ATTN=decode (validation: decode attention kernels, bit-identical to decode at any length), INK_PF_PROFILE=1.
- Scratch: allocated on the first call for the chunk size (one hugepage arena): ~150 MB at 256, ~300 MB at 512, ~600 MB at 1024 tokens; re-allocated if a larger chunk is requested later.
- The calling thread is pinned to cpus[0] like run_token; one dispatcher per pool (not concurrent with decode/prefill on the same engine); pool size <= 64 threads.
- Slot state after a prefill is exactly the decode state (S.pos, global n_blocks, SWA rings, conv histories); the SWA ring must have >= 1023 + chunk slots (32 blocks = chunk <= 1025); with a
  smaller ring the executor shrinks the chunk automatically (correct, slow).

## Open / not done
- Controllers (4.7%): gain/forget are K=64 BF16 GEMMs, hsum-dominated (7 GMAC/s per thread). A transposed reduction with the same tree would be bit-exact and ~2x faster; or a token-parallel fused item (no barriers).
- Fast attention is not chunk-size invariant bit-wise for odd chunk lengths (tile pairing starts at the chunk start); decode-kernel mode is invariant.
- Engine::forward is still the decode loop: see docs/requests.md item 1.

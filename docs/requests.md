
## From agent P (batched prefill), 2026-11-21
1. (C1, optional, small) Route `Engine::forward(slot, tokens, n > 1, logits_last)` to `ink::engine_prefill(*this, slot, tokens, n, logits_last)` (header `ink/engine_prefill.h`,
   source `src/engine/engine_prefill.cpp`, already added to `src/engine/CMakeLists.txt` under `if(TARGET ink_kernels_gemm ...)`). Results are bit-identical to the decode loop
   (docs/PREFILL_DESIGN.md section 5) except attention on contexts >= 768 visible keys on global layers, where decode splits the keys (nsplit >= 2) and prefill does one pass.
   The C ABI (`ink_forward` / capi) can call it directly; `src/engine/standalone/CMakeLists.txt` needs `${ROOT}/src/kernels_gemm/gemm.cpp gemm_small.cpp` as a library `ink_kernels_gemm`
   (linked PUBLIC to ink_engine) for the prefill sources to be compiled there.
2. (C1, optional) The prefill scratch (100-600 MB depending on the chunk size, allocated lazily on first use) lives in a process-global registry keyed by `Engine*` because engine.h is
   yours. If you want it owned by the Engine: add `std::shared_ptr<void> prefill_ctx;` to `Engine` and call `ink::engine_prefill_release(*this)` from `~Engine` (declared in engine_prefill.h).
   Without that, `engine_prefill_release(e)` must be called before an Engine is destroyed if the process keeps running (the registry entry would otherwise leak until exit).
3. (orchestrator) Thread pinning: engine_prefill pins the calling thread to cpus[0] like `Engine::run_token` does. Two engines/processes on the same CPU list time-slice each other
   badly because of the spin barriers (observed 2.4 s/token decode while another 12-thread job was pinned to the same CPUs); use `EngineOptions::cpus` to separate them.

## From agent V (server on the real engine), 2026-11-22
1. (B1 / orchestrator, `include/ink/threadpool.h`; NOT edited by me) IDLE CPU. An idle `libink` server burns 30% of one core: 4 idle workers x 7.3-8.1% each (5-thread pool, fresh process, 10 s windows,
   host load 5-16; the requirement was < 5%). Cause: the 20 us poll sleep is a `nanosleep` with the kernel's default 50 us timer slack, ~14k wake-ups per second per worker.
   Server-side mitigation (done, in `server/backends.py`, no engine change): `prctl(PR_SET_TIMERSLACK)` on the thread that calls `ink_load` (the pool threads inherit it). Fresh-process A/B:
   slack 50 us (default) 30.3%, 2 ms 3.4%, 5 ms 2.0% (test_ink_real test h, default), 10 ms 1.0% of one core; decode tok/s and first-request-after-idle latency unchanged within noise.
   Optional proper fix in the pool: after ~2 ms idle block on a futex / condition variable (or sleep >= 1 ms) instead of 20 us polls, so every embedder gets it without the prctl.
2. (S2, `src/engine/engine_load.cpp`) PEAK RSS AT LOAD is 2x the steady state: VmHWM 32-34 GiB vs 16.2 GiB after load (the GGUF mapping and the copied arena coexist until the
   `MADV_DONTNEED` that follows the copy). A server needs ~34 GiB of free RAM to START; two engine processes loading at the same time need ~68 GiB. Cheap fix: `madvise(MADV_DONTNEED)` on each
   tensor's source range right after it is copied (or `pread` into the arena + `posix_fadvise(DONTNEED)`). Not urgent on the 125 GB host.
3. (S2, for `ink_spec_step`) The server side is ready (`backends.SpecRequest`, `StepResult`, `derive_fed`, scheduler `_spec_step`, `MockSpecBackend`, tests). What it relies on: the result lists the EMITTED
   tokens (>= 1, last = next anchor unless it is a stop token) and `ink_slot_pos(slot)` is EXACT after every spec step (the scheduler compares it with anchor + accepted tokens and resets the slot
   with a 500 on a mismatch); the engine, not the server, applies the stop ids and the max-emit budget; sampled steps must be reproducible per request seed; slot state must be valid after
   a step that ended on a stop token in the middle of a verify window. If the C result can also return the list of tokens it fed, `StepResult.fed` takes it directly.
4. (S2, `include/ink/ink_abi.h` + `src/capi/ink_capi.cpp`) PROPOSED C ABI for the speculative step, 2026-11-22 00:25. The server side is written and guarded
   (`server/backends.py` InkBackend: `SpecRequestC` / `SpecResultC` ctypes structs, `_bind_spec`, `spec_step`; `python -m server --drafter PATH`): at start-up it probes the three
   symbols below with dlsym; without them (today's libink.so) `supports_spec` stays False and the server runs plain decode exactly as now. Mirrors `include/ink/spec.h`
   SpecRequest / SpecResult as plain C (natural alignment, sizeof 56 / 104 on x86-64; the server checks the sizes in its tests):
   ```c
   #define INK_SPEC_ABI 1
   int ink_spec_abi_version(void);                                   // returns INK_SPEC_ABI; the server refuses any other value (layout guard)
   int ink_spec_load(ink_engine*, const char* drafter_gguf_path);    // builds the SpecDecoder (installs the tap sink); 0 ok, else < 0 + ink_last_error
   typedef struct ink_spec_request { int32_t anchor; float temperature; int32_t top_k; float top_p; float min_p; uint64_t seed;
       const int32_t* stop_ids; int32_t n_stop; int32_t max_emit; int32_t mode; } ink_spec_request;         // mode -1 = decoder default (env INK_SPEC)
   typedef struct ink_spec_result { int32_t tokens[8]; int32_t n_tokens; int32_t n_fed; int32_t stopped; int32_t drafted; int32_t accepted; int32_t T;
       double draft_us, verify_us, accept_us, total_us; uint64_t next_seed; int32_t rc; } ink_spec_result;
   int ink_spec_step(ink_engine*, int slot, const ink_spec_request*, ink_spec_result*, float* logits_out);  // 0 ok; -1 bad slot, -2 context full, -3 bad token, -4 bad request
   ```
   Semantics the server relies on: `tokens[0..n_tokens)` are the EMITTED tokens (>= 1), the last one is the next anchor unless `stopped`; `n_fed` = number of tokens the
   engine appended to the slot in this call (= 1 + accepted; the anchor always, then the emitted tokens before a stop token) so that `ink_slot_pos` == old pos + n_fed
   (the scheduler checks this and resets the slot with a 500 on a mismatch); `logits_out` (may be NULL) receives the logits row after the last FED token, like
   ink_forward, so the identical-prompt cache keeps working; the seed is per request (the server passes the same seed every step and relies on the decoder threading
   `next_seed` internally per slot, or re-derives from (seed, slot pos) - either is fine as long as a request is reproducible). Penalties / logit_bias never reach this path.
   If you prefer another layout, change the Python structs in backends.py (one place) and bump INK_SPEC_ABI on both sides.
5. (S2, `src/spec` policy / sampler; from Vc 2026-11-22 00:50) SAMPLED `auto` MODE IS NOT REPRODUCIBLE PER SEED ACROSS REQUESTS. Server test j (`server/tests/test_ink_real.py`): two identical
   /v1/completions requests (s2_code_add prompt, T=0.6, seed 12345, speculation auto, first anchor via ink_sample, next_seed chained step by step) gave identical tokens 1-2 and then
   diverged (token 3: 150096 vs 160416). Fixed widths (4, 8) are reproducible. Cause (as I read SPEC_DESIGN): the policy's state (width estimates, exploration counter, probe burst) persists
   across requests, so the second request speculates with another T and the random draws differ. Not a correctness bug (same distribution), but `seed` loses its meaning in auto mode. Options:
   (a) reset / snapshot the policy state at each request's first step (ink_slot_reset or a flag in ink_spec_request), or (b) make the sampled result independent of T by drawing the per-position
   uniforms from (seed, position) instead of a stream. The server documents the limitation (docs/SERVING.md section 11) until one of these lands; the mock suite and test j accept it (fixed widths hard, auto recorded).
6. (S2, note) Item 4 above (my predecessor's PROPOSED layout with `ink_spec_abi_version`) is obsolete: the server binds the layout of include/ink/ink_abi.h as published (nested `sp`, `fed[8]`/`n_fed`,
   no version symbol, no logits_out). Verified field for field against gcc sizeof/offsetof (48 / 136 bytes). Nothing further needed from S2 for the binding.
7. (S2 / orchestrator, note from Vc 2026-11-22 01:00) Greedy identity plain == speculative holds in every short test (2 prompts x 64, 2 x 100, chat, tool turns) but the agent task at
   T=0 diverged from the plain run in the long step (context > ~1000 tokens, fast verify kernel, INK_SPEC_EXACT unset). Expected from S2's "bit-exact below 768 keys"; just make the
   default explicit in SPEC_DESIGN / the launch recipe (exact for correctness runs, fast for speed) or switch the kernel automatically past 768 keys. Also: the max_emit budget shrinks T
   (a request's last step reports T=3 for width 4): fine, but worth one line in ink_abi.h so that clients do not assert on the histogram as I did.

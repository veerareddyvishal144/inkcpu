
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

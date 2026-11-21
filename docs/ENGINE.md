# inkcpu: decision log and measurements

## 2026-11-21 Phase 0 (CPU-only profile of llama.cpp) and the go decision

Setup: stock llama.cpp mainline @2a28df0f9 (loop fix), CPU build with the system toolchain, GGUF `Inkling-2-Mini-UD-Q4_K_XL`
(17.63 GB: experts L0-19 NVFP4, L20-23 Q8_0, attention/embed/lm_head Q8_0, small tensors bf16/F32), quiet host (load < 1).

Baseline (tuned): decode 5.2-5.7 tok/s (best 5.71 at 12 threads with `OMP_WAIT_POLICY=active OMP_PLACES=cores
OMP_PROC_BIND=close`; `OMP_WAIT_POLICY=passive` collapses to 1-2 tok/s); prefill 23 tok/s at 16 threads.
Measured streaming read bandwidth of the box: 92.5 GB/s (16 threads). Bytes read per decoded token for this GGUF
~4.0 GB (estimate) => 17-20 tok/s ceiling at 70-80 GB/s; llama.cpp reaches ~23 GB/s effective (25%).

Where decode time goes (wall time per op as seen by thread 0, 33 tokens, 173 ms/token):
- MoE NVFP4 gate_up+down: 29% (effective ~32 GB/s). llama.cpp has NO x86 SIMD NVFP4 dot; only `ggml_vec_dot_nvfp4_q8_0_generic`.
- UNARY (tanh/sigmoid/softcap etc.): 14.8% = 25 ms/token (scalar libm), should be ~1-2 ms.
- Q8_0 attention matmuls: 19% (~41 GB/s effective); lm_head 3.9% (73 GB/s: fine).
- ~6000 tiny graph nodes per token (MUL/ADD/CONCAT/SCALE/CONT/RMS_NORM/...), each followed by a thread barrier:
  barrier waiting = 25.8% of wall time; tiny ops = ~18% on top of that.
- F32 matmuls (controllers, router; stored F32 in the GGUF although the original is bf16): ~4%.
Where prefill time goes (42.8 ms/token): NVFP4 MoE 55% (scalar kernel), UNARY 7%, Q8_0 attention ~11%, F32 matmuls 7%.

Gate: headroom is large (decode ~3.5x of bandwidth, prefill ~4-5x of int8 compute) and mostly structural (graph
fragmentation + missing SIMD kernels), so a purpose-built engine is justified. Decision: build `inkcpu`, CPU-only.
Plan: scalar reference (oracle-validated against llama.cpp on the same GGUF) -> AVX-512 kernels -> fused phase executor with
pinned spin-barrier thread pool -> prefill GEMM path -> speculative decoding if routing data says it pays -> server.
Targets: decode >= 2x tuned llama.cpp (>= 11.5 tok/s; stretch 3x = 17), prefill >= 3x (>= 70 tok/s), logits cosine >= 0.999
and greedy agreement vs llama.cpp on the same GGUF, resident memory < 25 GB.

Caveats: single measurements on a shared VM; baseline numbers are per `plan/raw/baseline_matrix.txt`; the 4.0 GB/token figure is
an estimate until E's exact table lands (docs/BASELINE.md).

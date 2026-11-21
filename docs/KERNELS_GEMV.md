# KERNELS_GEMV: thread pool, Q8_0 quantiser, GEMV kernels (agent B1)

Files: `include/ink/threadpool.h`, `include/ink/kernels_gemv.h`, `src/kernels_gemv/{threadpool,gemv,gemv_ref}.cpp`,
tests and benchmarks in `tests/kernels_gemv/`. Build: root CMake adds `src/kernels_gemv` (static lib `ink_kernels_gemv`,
tests `test_threadpool test_quantize test_gemv`, benches `bench_pool bench_gemv`). Library flags: root flags + `-ffp-contract=off`.

## 1. Host facts measured while building this (matter for every number below)
- The host is shared: during all measurements 1-min loadavg was 3 to 14 (other agents running 6-core jobs). Numbers are
  best-of-N windows, not idle-machine numbers. Loadavg is printed next to every bench row.
- **Physical core 0 (CPU 0 and CPU 1) runs at about half speed** (dependent-add chain: 1.2 GHz vs 2.5 GHz on every other
  CPU, stable over several minutes). With equal static row slices, the thread on CPU 0 is the straggler and the whole call
  runs at its pace. Remedies: dynamic chunking (`ThreadPool::run_chunks`, NVFP4 gate_up at 16 threads 41.2 -> 49.3 GB/s, load 3)
  and a CPU list that puts core 0 last: `{2,4,...,30,0}` (bench `--cpus`). The pool default stays even CPUs `0,2,...` as specified.
- THP is `madvise`: weights should live in `alloc_weight_buffer()` memory (2 MB aligned + MADV_HUGEPAGE; verified 2.4 GB in
  AnonHugePages). For a GGUF mmap, copy the tensors there at load.

## 2. ThreadPool (`ink::ThreadPool`)
```cpp
ink::ThreadPool pool(16);                      // caller = tid 0, workers pinned to CPUs 2,4,..,30 (cpus list configurable)
pool.run([&](int tid, int n) { ...; pool.barrier(); ... });          // all threads call barrier() equally often
pool.run_chunks(total_rows, chunk, [&](int tid, int64_t b, int64_t e) { gemv_multi_q8_0(mats, nm, k, b, e); });
```
Dispatch: epoch counter (release store) polled by workers; join: per-worker `done` flags polled by the caller; barrier:
per-thread arrive flags gathered by thread 0, one release flag. All flags are 64-byte padded monotonic counters, `_mm_pause`
spin, no futex/condvar in the hot path. Workers spin `idle_spin_us` (2 ms) after a job, then poll with 20 us sleeps
(first run after a long idle gap costs up to ~100 us). Default CPU list: 0,2,4,...,30 then 1,3,5,...
Latency (`bench_pool`, min / median over 11 rounds x 20000, load 10.8): run()+join empty body: 8 thr 527/547 ns, 12 thr 676/732 ns,
16 thr 856/1130 ns. barrier(): 8 thr 617/645 ns, 12 thr 605/657 ns, 16 thr 810/1020 ns. Targets (barrier < 2 us, dispatch < 5 us at 16 threads): met.

## 3. Kernels (API in kernels_gemv.h)
`y[r] = sum_k W[r][k]*x[k]`, single thread, row range, `row_stride` in bytes (expert slices, concatenations, padding). No alignment needs;
loads never leave a row (prefetch may). `y` is overwritten.

| function | weights | x | K constraint |
|---|---|---|---|
| `gemv_q8_0` | Q8_0 (34 B/32) | `block_q8_0[K/32]` from `quantize_row_q8_0` (|q| <= 127) | K % 32 |
| `gemv_nvfp4` | NVFP4 (36 B/64), per-expert F32 scale NOT applied | `block_q8_0[K/32]` | K % 64 |
| `gemv_bf16` | BF16 | `float[K]` | K % 16 |
| `gemv_f32` | F32 | `float[K]` | K % 16 |
| `gemv_multi_*` | list of `GemvMat{w,row_stride,x,y,nrows}` = one virtual matrix, thread computes virtual rows [vr0,vr1) | per-matrix x | as above |

- `quantize_row_q8_0`: AVX-512, bit-exact with ggml `quantize_row_q8_0_ref` (d = amax/127, id = 1/d, fp16 d, `roundf` = half away
  from zero, done with trunc + compare, not cvtps2dq). Verified bit-for-bit against ggml's own function (dlopen of
  `llama.cpp/build-sys/bin/libggml-base.so`) on 1971 vectors incl. exact .5 ties, zeros, denormals, 3e38.
- Q8_0: zmm = 2 blocks. `abs(w)` as unsigned operand, activation negated under the k-mask of w's sign bits, `maddubs` (|pair| <=
  2*128*127, no saturation), `madd` with ones, int->float, FMA with per-lane `dw*dx`. 16 float lanes, fixed reduction tree.
- NVFP4: nibbles broadcast to both ymm halves, upper half `srlv` 4, mask 0x0F, one `pshufb` with table `kvalue+12` (unsigned 0..24);
  activation permuted once per super-block (shared by the 2-4 rows); `maddubs`/`madd` give sum((kv+12)x), minus the row-independent
  C = sum(12x) gives the exact integer sum(kv*x). Sub-block scale `ue4m3(d)*dx` from a 256-entry float table (identical to
  `ue4m3_to_f32` for all byte values), expanded by `vpermps` to the 4 lanes of its sub-block.
- BF16: `cvtepu16_epi32` + `slli 16`, FMA, two accumulators; F32 plain FMA.
- 4 rows per group (then 2, 1). Prefetch: every row stream prefetches `g_gemv_prefetch_bytes` (512, T0) ahead; the first 512 B
  of each row of the NEXT group are requested while the current group runs. Tunables `g_gemv_prefetch_bytes/hint` (hint 0 T0, 1 T1,
  2 NTA, 3 none).
- **Determinism**: every row's result is independent of grouping, split and thread count (bitwise); `gemv_*_ref` reproduce the
  lane structure, `fmaf` and reduction tree, so kernel == ref bit-for-bit.

## 4. Tests (all pass; `ctest` in the build dir)
- `test_threadpool`: sizes 1,2,3,5,8,12,16: tid coverage, barrier ordering/visibility (3 phases x 600 runs, 20000 barriers in one run), reduction.
- `test_quantize`: SIMD vs scalar vs ggml, bit-exact.
- `test_gemv` (3980 cases): 4 formats x K in {16,32,48,64,96,160,352,1152,2304,3072,4608} (where legal) x rows {1..33, odd} x 3 row strides (tight, +pad, 2x+4) x
  5 value modes (normal, extreme +-128/max scales, zero scales, tiny/huge scales, unit). Checks: (a) kernel == scalar ref bitwise;
  (b) |kernel - double dequantised dot| <= 2e-4 * sum|w*x| (relative to the sum of absolute terms; a cancelling dot has no usable
  relative error); (c) unit-scale mode equals the exact integer dot; (d) results independent of splitting rows into calls, fused multi
  entry points equal single calls; (e) Q8_0 and NVFP4 vs ggml `ggml_vec_dot_{q8_0_q8_0,nvfp4_q8_0}_generic` within 3e-5 * sum|w*x|;
  prefetch knobs do not change results; garbage in row padding and guard bytes is ignored.

## 5. Bandwidth (`bench_gemv`, weight bytes / s, 2.5 GB rotating working set, hugepages, dynamic chunking, CPU list 2,4,..,30,0)
Best of 5 interleaved 0.12 s windows (kernel, stream-with-prefetch, stream-without, alternating). "stream" = a pure streaming-sum kernel reading the
same bytes with the same thread split and call size: the practical upper bound in the same window. Peak = 92.5 GB/s. **Load average rose from 8.5 to 14 during this
run**; the 16-thread rows of the 2-12 MB calls are visibly perturbed by preempted threads (stream shows it too).

| shape (per call) | 8 thr GB/s | 12 thr GB/s | 16 thr GB/s (% of 92.5) | stream @16 | kernel/stream @16 |
|---|---|---|---|---|---|
| Q8_0 3072x2304 (7.5 MB) | 39.8 | 49.4 | 52.6 (56.8%) | 68.4 | 0.77 |
| Q8_0 2304x3072 (7.5 MB) | 47.5 | 54.1 | 34.1 (36.8%, perturbed) | 37.1 | 0.92 |
| Q8_0 1024x2304 (2.5 MB) | 25.8 | 51.7 | 30.3 (32.8%, perturbed) | 33.9 | 0.89 |
| Q8_0 5120x2304 q\|k\|v fused (12.5 MB) | 37.7 | 57.6 | 34.7 (perturbed); 58.8 (63.6%) in an earlier run | 41.7 | 0.83 |
| **Q8_0 200058x2304 lm_head (490 MB)** | 53.9 | 74.9 | **87.4 (94.5%)** | 93.5 | 0.92 |
| **NVFP4 8 experts gate_up 8x2304x2304 (23.9 MB)** | 36.2 | 50.8 | **61.0 (66.0%)** | 84.6 | 0.72 |
| **NVFP4 8 experts down 8x2304x1152 (11.9 MB)** | 34.1 | 48.0 | **59.4 (64.2%)** | 78.0 | 0.76 |
| BF16 3072x2304 | 54.2 | 66.8 | 72.4 (78.2%) | 77.2 | 0.94 |
| F32 2304x2304 | 58.3 | 74.8 | 78.1 (84.4%) | 82.7 | 0.94 |

Verdict against the targets (>= 65 GB/s, 70% at 16 threads): Q8_0 met for the big matrix (lm_head 87.4 GB/s); NVFP4 NOT met in the measurements
(61 / 59 GB/s, 64-66%, while the stream ceiling in the same windows was 78-85 GB/s, so the machine had ~85% of 92.5 available). Small Q8_0 calls (2.5-12 MB) are limited
by per-call ramp-up and host noise: even the stream kernel only reaches 34-68 GB/s on them under this load.
What limits NVFP4 (evidence): (1) compute: single-thread, L2-resident (`--rows 64 --nm 1 --pool-gb 0.0003 --cpus 6`, load 10) the kernels sustain Q8_0 13.1,
NVFP4 7.4, BF16 31.5, F32 43.3 GB/s of weight bytes per thread; NVFP4 needs 61/16 = 3.8 GB/s per thread = 51% of its resident rate, so on a host whose
hyperthread siblings are busy (loadavg 8-14) the integer pipeline competes for ports and cannot keep 16 streams fully fed; Q8_0 needs 5.5 GB/s of 13.1 (42%) and does reach the ceiling.
(2) The core-0 straggler (half speed) is only partly hidden by chunking. NVFP4 costs ~25 instructions per 64 weights per row (scale decode ~8 of them); the
next step is trimming that (repack scales to fp16/fp32-ready form, or batch-decode scales per row) and re-measuring on a quiet host.
Prefetch sweep (NVFP4 gate_up, 16 thr, load 7.5-8.7): none 48.7; 512 B T0 62.9; 1024 56.0; 2048 54.3; 4096 52.1; T1 54.5/52.2 (1024/2048); **NTA 30.7/28.7 (do not use)**.
Static vs dynamic (NVFP4 gate_up, 16 thr, default even CPUs, load 3.1): 41.2 vs 49.3 GB/s. Raw output of the final matrix: `build-b1/bench_final_nocore0.txt` (not in git).

## 6. Recommended use
- Pool: 16 threads, one per physical core; on this host `cpus = {2,4,...,30,0}`.
- Split: `run_chunks(total_rows, chunk, ...)` with chunk = clamp(total/(6*nthreads), 16, 512) rounded to a multiple of 4 (the bench does this); static `split_rows(align=4)` is
  fine on a quiet host. Never split K. Call the kernels with the virtual-row range of the chunk.
- Fuse: q|k|v (shared x) in one `gemv_multi_q8_0` call; 8 experts as 8 `GemvMat` (gate_up shares x; down has one x per expert); one `run` per fused op, `barrier()` between dependent ops inside one `run`.
- Quantise x once per matvec (`quantize_row_q8_0`; cost not measured separately, estimated well under 1 us for K=2304) on one thread, then barrier, or on every thread into a private buffer.
- Apply the per-expert F32 scale (`ffn_*_exps.scale[e]`) outside the kernel (e.g. folded into the router weight).

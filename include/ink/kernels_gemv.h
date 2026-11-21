// GEMV kernels (batch-1 matvec, weights streamed once from DRAM). Owner: agent B1. See docs/KERNELS_GEMV.md.
//
// All kernels compute  y[r] = sum_k W[r][k] * x[k]  for rows r = 0..nrows-1 of ONE row range, single-threaded.
// The executor splits rows over the ThreadPool (split_rows) and quantises x once per matvec (quantize_row_q8_0).
// Row r of W starts at (const char*)w + r*row_stride (bytes); row_stride may exceed the row size (expert slices,
// concatenated matrices, padding). Alignment: none required. No kernel reads outside [row start, row start + row bytes)
// (prefetch hints may touch addresses beyond; they never fault).
//
// Formats (GGUF-native, see ggml_types.h):
//   Q8_0  : K % 32 == 0, row = K/32 block_q8_0 (34 B).        x = K/32 block_q8_0 (activation quantised with quantize_row_q8_0)
//   NVFP4 : K % 64 == 0, row = K/64 block_nvfp4 (36 B).       x = K/32 block_q8_0. Per-expert F32 scale NOT applied.
//   BF16  : K % 16 == 0, row = K uint16.                      x = K floats
//   F32   : K % 16 == 0, row = K floats.                      x = K floats
// x values of the Q8_0 activations must be in [-127,127] (true for quantize_row_q8_0 output); weights may be any int8.
//
// Results are deterministic and independent of how the rows are grouped or split over threads: the *_ref functions
// reproduce them bit-for-bit (same lane structure, fused multiply-adds, fixed reduction tree).
#pragma once
#include <cstddef>
#include <cstdint>

#include "ink/ggml_types.h"

namespace ink {

// ---- activation quantisation: identical algorithm to ggml quantize_row_q8_0_ref (d = amax/127, fp16 d, q = roundf(x/d)) ----
void quantize_row_q8_0_ref(const float* x, block_q8_0* y, int k);  // scalar
void quantize_row_q8_0(const float* x, block_q8_0* y, int k);      // AVX-512, bit-exact with the ref (k % 32 == 0)

// ---- row sizes in bytes ----
inline size_t row_bytes_q8_0(int k) { return (size_t)(k / QK8_0) * sizeof(block_q8_0); }
inline size_t row_bytes_nvfp4(int k) { return (size_t)(k / QK_NVFP4) * sizeof(block_nvfp4); }
inline size_t row_bytes_bf16(int k) { return (size_t)k * 2; }
inline size_t row_bytes_f32(int k) { return (size_t)k * 4; }

// ---- single-matrix row-range kernels (AVX-512 F/BW/DQ, no VNNI) ----
void gemv_q8_0(const void* w, size_t row_stride, const block_q8_0* x, int k, float* y, int nrows);
void gemv_nvfp4(const void* w, size_t row_stride, const block_q8_0* x, int k, float* y, int nrows);
void gemv_bf16(const void* w, size_t row_stride, const float* x, int k, float* y, int nrows);
void gemv_f32(const void* w, size_t row_stride, const float* x, int k, float* y, int nrows);

// ---- scalar references with the same accumulation structure (tests, oracle for the kernels) ----
void gemv_q8_0_ref(const void* w, size_t row_stride, const block_q8_0* x, int k, float* y, int nrows);
void gemv_nvfp4_ref(const void* w, size_t row_stride, const block_q8_0* x, int k, float* y, int nrows);
void gemv_bf16_ref(const void* w, size_t row_stride, const float* x, int k, float* y, int nrows);
void gemv_f32_ref(const void* w, size_t row_stride, const float* x, int k, float* y, int nrows);

// ---- fused multi-matrix variant ----
// A list of matrices is treated as ONE virtual matrix whose rows are the concatenation of the rows of m[0], m[1], ...
// (e.g. wq|wk|wv sharing one x, or 8 routed experts, each with its own x for the down projection).
// A thread computes virtual rows [vr0, vr1): the kernel walks the matrices, so one ThreadPool::run covers all of them
// and the row split stays balanced. x (per matrix) is block_q8_0* for q8_0/nvfp4 and float* for bf16/f32.
struct GemvMat {
    const void* w;       // first row of this matrix
    size_t row_stride;   // bytes between rows
    const void* x;       // activation for this matrix (may be the same pointer for all matrices)
    float* y;            // nrows outputs
    int nrows;
};
void gemv_multi_q8_0(const GemvMat* m, int nm, int k, int64_t vr0, int64_t vr1);
void gemv_multi_nvfp4(const GemvMat* m, int nm, int k, int64_t vr0, int64_t vr1);
void gemv_multi_bf16(const GemvMat* m, int nm, int k, int64_t vr0, int64_t vr1);
void gemv_multi_f32(const GemvMat* m, int nm, int k, int64_t vr0, int64_t vr1);

// ---- helpers ----
// Balanced contiguous split of n rows over nth threads; every slice start/length is a multiple of `align` except the
// tail (use align = 4 so the kernels run their 4-row path).
void split_rows(int64_t n, int nth, int tid, int64_t align, int64_t* begin, int64_t* end);

// 2 MB-aligned buffer with MADV_HUGEPAGE (THP is in "madvise" mode on this host): fewer TLB misses while streaming
// weights. Free with free_weight_buffer. hugepage=false gives a plain 4 KB-page allocation.
void* alloc_weight_buffer(size_t bytes, bool hugepage = true);
void free_weight_buffer(void* p);

// ---- tunables (read once per kernel call; for experiments, defaults are the tuned values) ----
extern int g_gemv_prefetch_bytes;  // software prefetch distance per row stream (0 = none). default 512
extern int g_gemv_prefetch_hint;   // 0 = T0, 1 = T1, 2 = NTA, 3 = no prefetch instructions at all

}  // namespace ink

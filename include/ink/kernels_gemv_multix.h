// Small-n multi-activation GEMV ("multix"): ONE weight stream of a few rows against N <= 8 activation vectors. Owner: agent K2 (wave 1).
//
// Purpose: the MoE expert matmuls of a speculative verify step (T = 4..8 rows) or of a short prefill chunk touch ~20-30 experts with only 1-2
// routed tokens each (docs/ROUTING.md). The batched GEMM (gemm_moe_*) unpacks every 4-row weight panel into a scratch (store-bound, ~7 cycles per
// row super-block) and that cost is never amortised over so few tokens; the decode GEMV streams the weights well but handles one activation per
// pass. These kernels stream each weight row exactly once (GEMV-style software prefetch, 8/4/2 row streams in flight), decode the weights in
// registers (nibbles -> |kv| table, sign mask, scale vector) ONCE per row super-block and apply them to all N activations held in registers.
//
//   y[j * ldy + r] = dot(W[r], x[j])   for rows r in [0, nrows) and activations j in [0, nx)
//
// Numerics: BIT-IDENTICAL per (row, activation) to gemv_q8_0 / gemv_nvfp4 (and therefore to gemm_q8_0 / gemm_nvfp4 / gemm_moe_*): the same exact
// integer sums of 4 products per 32-bit lane (|w| as unsigned maddubs operand, activation sign-transferred under the weight sign mask, madd with
// ones), the same per-lane float scale S_l = fl(dw * dx) (one float multiply of the same two floats), the same fmaf per lane, and the same hsum16
// reduction tree over the lanes in gemv.cpp's logical order (the kernels keep the GEMM's physical lane order so that the scale vector is a plain
// broadcast, and apply one vpermps before the reduction; no float operation differs). Same MXCSR dependence (FTZ/DAZ) as gemv_*.
//
// Formats (GGUF-native, see ggml_types.h; K % 64 == 0, K <= 8192):
//   gemv_q8_0_multix   : Q8_0 rows (34 B / 32 weights)         x[j] = block_q8_0[K/32] from quantize_row_q8_0 (|q| <= 127)
//   gemv_nvfp4_multix  : NVFP4 rows (36 B / 64 weights), GGUF UE4M3 scale bytes; the per-expert F32 scale is NOT applied (like gemv_nvfp4)
//   gemv_nvfp4r_multix : NVFP4 rows whose 4 scale bytes per super-block were re-encoded at load (include/ink/nvfp4_repack.h, LayerPacked::nvfp4r);
//                        bit-identical to gemv_nvfp4 on the original bytes (nvfp4r_decode == ue4m3_to_f32 for every byte value)
// x[j] are independent pointers (so the routed pairs of one expert can be addressed indirectly, no gather copy). nx > kMultixMaxX is legal: the
// weights are then streamed once per group of kMultixMaxX activations. Row r of W starts at (const char*)w + r * row_stride; no alignment needed;
// no kernel reads outside [row start, row start + row bytes) except prefetch hints (which never fault). y is overwritten (only the nx * nrows
// addressed elements; ldy may be padded).
//
// Tunables: the kernels read g_gemv_prefetch_bytes / g_gemv_prefetch_hint (kernels_gemv.h) like gemv_*. Thread-safe (a thread_local scratch of
// ~80 KB holds the lane-permuted activations of one call).
#pragma once
#include <cstddef>
#include <cstdint>

#include "ink/ggml_types.h"

namespace ink {

constexpr int kMultixMaxX = 8;

void gemv_q8_0_multix(const void* w, size_t row_stride, int nrows, const block_q8_0* const* x, int nx, int k, float* y, size_t ldy);
void gemv_nvfp4_multix(const void* w, size_t row_stride, int nrows, const block_q8_0* const* x, int nx, int k, float* y, size_t ldy);
void gemv_nvfp4r_multix(const void* w, size_t row_stride, int nrows, const block_q8_0* const* x, int nx, int k, float* y, size_t ldy);

// Rows per weight group the kernels use for nx activations (8 for nx <= 2, 4 for nx <= 4, else 2): informational, for tests / benchmarks.
int gemv_multix_rows_per_group(int nx);

}  // namespace ink

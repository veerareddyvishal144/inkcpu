// Batched GEMM kernels for prompt processing (prefill). Owner: agent G. See docs/KERNELS_GEMM.md.
//
// Every output element y[t][r] is BIT-IDENTICAL to ink::gemv_{q8_0,nvfp4,bf16,f32} applied to token t and row r (same integer
// partial sums, same 16-lane fmaf accumulation, same reduction tree), so prefill and decode agree exactly.
//
// All Q8_0 / NVFP4 entry points need K % 64 == 0 and K <= 8192. Weights are GGUF-native (no repack). All functions are
// single-threaded over (a row range x a token range); the caller splits rows (multiples of 4) and/or token ranges over the pool.
//
// Activations (Q8_0 and NVFP4 weights): quantise each token row once with quantize_row_q8_0 (block_q8_0[K/32]), then pack the
// nt rows into the lane-permuted GEMM layout with gemm_pack_act (one buffer per FORMAT: Q8_0 and NVFP4 weights need different
// permutations, so a matmul input shared by Q8_0 and NVFP4 weights is packed twice; q|k|v share one pack).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ink/ggml_types.h"
#include "ink/kernels_gemv.h"

namespace ink {

enum GemmFmt : int { GEMM_FMT_Q8 = 0, GEMM_FMT_NVFP4 = 1 };

// Non-owning view of a packed activation matrix (token t: int8 quants at xq + t*k, float scales at dx + t*dxn).
struct GemmAct {
    int fmt = 0;
    int nt = 0;
    int k = 0;
    int dxn = 0;                 // floats per token in dx (padded to a multiple of 16)
    const int8_t* xq = nullptr;  // nt * k bytes
    const float* dx = nullptr;   // nt * dxn floats
};

// Bytes of the buffer for nt tokens (allocate 64-byte aligned, e.g. alloc_weight_buffer(bytes, false) or std::aligned_alloc(64, ...)).
size_t gemm_act_bytes(int fmt, int nt, int k);
GemmAct gemm_act_view(int fmt, const void* buf, int nt, int k);
// Packs tokens [t0, t1) of x (x = nt rows of K/32 block_q8_0, row t at x + t*(k/32)) into buf. Disjoint token ranges may be
// packed from different threads concurrently (the scale padding is written by the token that owns the row).
void gemm_pack_act(int fmt, const block_q8_0* x, int t0, int t1, int nt, int k, void* buf);
// Convenience: quantize_row_q8_0 + pack for tokens [t0, t1) of the float matrix x (row t at x + t*ldx).
void gemm_quantize_pack_act(int fmt, const float* x, size_t ldx, int t0, int t1, int nt, int k, void* buf);

// y[t*ldy + r] = dot(W[r], x[t]) for r in [0, nrows) (W row r at (const char*)w + r*row_stride) and tokens t in [t0, t1) of `act`.
// y must be pre-offset by the caller for its row range. Weights are streamed once per group of g_gemm_group tokens.
void gemm_q8_0(const void* w, size_t row_stride, const GemmAct& act, int t0, int t1, int k, float* y, size_t ldy, int nrows);
// NVFP4 weights; the per-expert F32 scale is NOT applied (caller multiplies, as for gemv_nvfp4).
void gemm_nvfp4(const void* w, size_t row_stride, const GemmAct& act, int t0, int t1, int k, float* y, size_t ldy, int nrows);

// BF16 / F32 weights x float activations (x = nt rows of K floats, row t at x + t*ldx). K % 16 == 0. y[t*ldy + r].
void gemm_bf16(const void* w, size_t row_stride, const float* x, size_t ldx, int nt, int k, float* y, size_t ldy, int nrows);
void gemm_f32(const void* w, size_t row_stride, const float* x, size_t ldx, int nt, int k, float* y, size_t ldy, int nrows);

// ---------------------------------------------------------------- MoE grouped GEMM
// Routing plan: pairs (token, j) sorted by expert (stable in token order). Pair p of expert e lives at offset[e] <= p < offset[e+1].
struct MoeRoute {
    int n_experts = 0, n_tokens = 0, topk = 0;
    std::vector<int32_t> offset;      // [n_experts + 1]
    std::vector<int32_t> pair_token;  // [n_pairs] token index of pair p
    std::vector<int32_t> pair_slot;   // [n_pairs] original position token*topk + j (to scatter results back)
    int n_pairs() const { return (int)pair_token.size(); }
};
void moe_build_route(const int32_t* ids /*[n_tokens][topk]*/, int n_tokens, int topk, int n_experts, MoeRoute& r);

struct MoeItem {
    int32_t expert, row0, row1;  // rows [row0, row1) of the expert matrix, all pairs of that expert
};
// Work items = (expert with >= 1 pair) x (row chunks of chunk_rows, a multiple of 4), sorted by descending pair count (big experts
// first, for dynamic scheduling). Use pool.run_chunks(items.size(), 1, [&](int tid, int64_t b, int64_t e) { for i in [b,e): gemm_moe_*(..., items[i], ...) }).
void moe_work_items(const MoeRoute& r, int nrows, int chunk_rows, std::vector<MoeItem>& out);

// y[p*ldy + row] for every pair p of it.expert and row in [it.row0, it.row1). act_rows[p] = row of `act` used by pair p
// (e.g. r.pair_token.data() for gate_up where the activation is per token); nullptr = row p of act (activation per pair, e.g. down).
// Expert e's matrix starts at (const char*)w + e*expert_stride.
void gemm_moe_q8_0(const void* w, size_t expert_stride, size_t row_stride, const GemmAct& act, const int32_t* act_rows, const MoeRoute& r,
                   const MoeItem& it, int k, float* y, size_t ldy);
void gemm_moe_nvfp4(const void* w, size_t expert_stride, size_t row_stride, const GemmAct& act, const int32_t* act_rows, const MoeRoute& r,
                    const MoeItem& it, int k, float* y, size_t ldy);

// tunable: tokens per group (activations of a group stay L2-resident while all row panels stream over them). default 64.
extern int g_gemm_group;
extern int g_gemm_tile;      // tokens per register tile (1..5), default 4 (4 rows x tile tokens accumulators)
extern int g_gemm_prefetch;  // 1 = software-prefetch the next weight panel while computing (default), 0 = off

}  // namespace ink

// kernels_misc: vector math, norms, clips, short convolutions, attention, router, depth-mix kernels for inkcpu.
// Owner: agent B2.  Semantics are taken from llama.cpp @2a28df0f9 (src/models/inkling2*.cpp, src/llama-graph.cpp).
// Every function exists twice: ink::name (AVX-512, F/BW/DQ/VL+FMA+F16C only) and ink::ref::name (scalar, double
// internally where it matters).  All functions are single-threaded, allocation-free and deterministic: results depend only
// on the arguments (reduction orders are fixed by the lengths, never by thread count).  Array functions accept any length
// including 0 and tails; pointwise functions accept y == x (in place).  Full documentation: docs/KERNELS_MISC.md.
#pragma once
#include <cstddef>
#include <cstdint>
#include "ink/ggml_types.h"

namespace ink {

// ---------------------------------------------------------------- model constants (Inkling 2 Mini)
constexpr int kHidden = 2304;
constexpr int kHeadDim = 128;
constexpr int kNQHeads = 24;
constexpr int kNKvHeads = 8;
constexpr int kGqa = 3;                  // query heads per KV head (q head h uses kv head h / 3)
constexpr int kRelFeat = 16;             // d_rel
constexpr int kRelExtent = 1024;         // rel bias distances 0..1023
constexpr int kSwaWindow = 1024;         // visible distances 0..1023 on SWA layers
constexpr int kConvTaps = 4;             // short conv kernel size
constexpr int kKvBlock = 64;             // KV cache block = 64 positions
constexpr size_t kKvBlockElems = (size_t)kNKvHeads * kKvBlock * kHeadDim;  // fp16 elements per block per tensor (128 KiB)
constexpr int kRelTableLen = 64 + kRelExtent + 64;                          // padded row length of the repacked rel table

enum class Q8Mode {
    Ref,  // ggml quantize_row_q8_0_ref: id = 1/d (d = amax/127), q = roundf(x*id)       (what the CONTRACT names)
    X86   // ggml arch/x86 quantize_row_q8_0 (what llama.cpp RUNS on this CPU): id = 127/amax, q = round-half-even(x*id)
};

// ================================================================= 1. vector math (pointwise, in place ok)
// exp: valid for x in [-87.3, 88.7] (clamped outside).  max rel err ~2.5e-7.
void vexp(const float* x, float* y, size_t n);
void vtanh(const float* x, float* y, size_t n);      // max rel err ~3e-7 over all finite x
void vsigmoid(const float* x, float* y, size_t n);   // 1/(1+exp(-x)); max rel err ~3e-7 for x >= -87.3
// y = beta*tanh(g/beta)*sigmoid(g) * lin*tanh(u/lin)   (llama.cpp LLM_FFN_SITU; Inkling: beta 4, lin 25; lin <= 0: up unchanged)
void vsitu(const float* gate, const float* up, float* y, size_t n, float beta = 4.0f, float lin = 25.0f);
// same, gate and up stored back to back as produced by the fused gate_up GEMV: gu = [gate(n) | up(n)] (gate first, as ggml views it)
void vsitu_fused(const float* gu, float* y, size_t n, float beta = 4.0f, float lin = 25.0f);
// Fused SiTU + Q8_0 quantisation of the activation (input of the down GEMV).  n % 32 == 0.  Bit-identical to
// vsitu() followed by quantize_q8_0() with the same mode.
void vsitu_q8_0(const float* gate, const float* up, block_q8_0* out, size_t n, float beta = 4.0f, float lin = 25.0f,
                Q8Mode mode = Q8Mode::Ref);
// cap*tanh(x/cap): final-logit softcap (cap = 30), also usable in place on 201024 logits
void vsoftcap(const float* x, float* y, size_t n, float cap = 30.0f);

// ================================================================= 2. RMSNorm
// y = (x * s) * w, s = 1/sqrtf(sum(x^2)/n + eps), eps 1e-6.  w == nullptr: no weight.  x == y allowed.
void rmsnorm(const float* x, const float* w, float* y, size_t n, float eps = 1e-6f);
// Fused rmsnorm + Q8_0 quantisation of the result (n % 32 == 0).  Bit-identical to rmsnorm() followed by quantize_q8_0().
void rmsnorm_q8_0(const float* x, const float* w, block_q8_0* out, size_t n, float eps = 1e-6f, Q8Mode mode = Q8Mode::Ref);
// Thread-splittable form of the two functions above (bit-identical): s = rms_scale(x, n) once (serial, ~0.15 us for n = 2304),
// then any thread applies it to a range: rmsnorm_apply -> elements [i0, i1); rmsnorm_apply_q8_0 -> 32-element blocks [b0, b1).
float rms_scale(const float* x, size_t n, float eps = 1e-6f);
void rmsnorm_apply(const float* x, const float* w, float s, float* y, size_t i0, size_t i1);
void rmsnorm_apply_q8_0(const float* x, const float* w, float s, block_q8_0* out, size_t b0, size_t b1, Q8Mode mode = Q8Mode::Ref);
// PRIVATE bit-exact copy of the ggml Q8_0 activation quantiser (agent B1 owns the shared one).  n % 32 == 0.
void quantize_q8_0(const float* x, block_q8_0* out, size_t n, Q8Mode mode = Q8Mode::Ref);

// ================================================================= 3. L2 clip / V norm (per head rows of `dim`)
// llama.cpp clip(): x *= chan_scale (if given, stored back);  norm = sqrtf(sum x^2) + 1e-6;  x *= clamp(threshold/norm, 0, 1);
// then x *= post_scale (tau on global layers: applied AFTER the clip, as a separate multiply).  In place.
//   Q:  n_heads 24, dim 128, chan_scale = attn_q_clip [24][128], threshold 22.627
//   K:  n_heads 8,  dim 128, no chan_scale, threshold 22.627
//   R:  n_heads 24, dim 16,  chan_scale = attn_rel_clip [24][16], threshold 8.0
void l2clip_heads(float* x, const float* chan_scale, int n_heads, int dim, float threshold, float post_scale = 1.0f);
// V norm: per head RMS norm without gamma: y = x / sqrtf(mean(x^2) + eps) (eps 1e-6); x == y allowed.
void vnorm_heads(const float* x, float* y, int n_heads, int dim, float eps = 1e-6f);

// ================================================================= 4. short causal depthwise convolution (kernel 4)
// llama.cpp convolve(): out[c] = sum_{i=0..3} xin[t-3+i][c]*w[c][i]  (+ dynamic taps), y = (x + out) * 1/sqrt((dyn?2:1)*4+1).
// The history holds the previous 3 conv INPUTS (pre-conv values).  Optimised layouts (SoA over channels):
//   w_soa[tap][C], tap 0 = oldest (t-3) ... tap 3 = current;  hist[3][C]: row 0 = x[t-3], row 1 = x[t-2], row 2 = x[t-1].
// Dynamic taps (global-layer K/V): dyn[h*4 + lag], lag 0 = current token, 1 = t-1, 2 = t-2, 3 = t-3, applied to all
// head_dim channels of head h (raw layout of the attn_{k,v}_conv_proj matvec output).  nullptr = static only.
// All sconv functions work on the channel range [c0, c1) (c0 multiple of 16) so threads can split channels with no shared writes.
void sconv_repack_static(const float* w_ggml /*[C][4]*/, float* w_soa /*[4][C]*/, size_t C);
void sconv_step(const float* x, float* hist, const float* w_soa, float* y, size_t C, size_t c0, size_t c1,
                const float* dyn = nullptr, int head_dim = kHeadDim);   // y == x allowed
// n tokens (prefill): x,y [n][C] row-major (y must not alias x); hist updated to the last 3 inputs.  dyn [n][n_heads*4].
void sconv_seq(const float* x, float* hist, const float* w_soa, float* y, size_t n, size_t C, size_t c0, size_t c1,
               const float* dyn = nullptr, int head_dim = kHeadDim);

// ================================================================= 5. attention (24 q heads, 8 kv heads, head_dim 128)
// KV cache layout (per layer visit / cache slot), fp16, blocks of 64 positions, one allocation per tensor, ZERO-INITIALISED
// (or any finite fp16): block b (b = slot/64) starts at k + b*kKvBlockElems and holds
//   K: [kv head 8][d 128][key-in-block 64]   (K transposed: 16 keys = one 32-byte load per d)
//   V: [kv head 8][key-in-block 64][d 128]
// slot = pos for global layers (grow by appending zeroed blocks; n_blocks*64 > max pos); slot = pos % (n_blocks*64) for the SWA ring
// (n_blocks*64 >= 1024; a prefill chunk of n tokens needs n_blocks*64 >= 1023 + n so no needed key is overwritten).
struct KvLayer {
    uint16_t* k = nullptr;
    uint16_t* v = nullptr;
    uint32_t n_blocks = 0;
    bool ring = false;
    size_t bytes_per_tensor() const { return (size_t)n_blocks * kKvBlockElems * 2; }
};
// g0/g1: kv-head range, so threads can append disjoint head groups (no shared writes).
void kv_append(const KvLayer& L, int64_t pos, const float* k /*[8][128] final K (clipped, post-conv)*/, const float* v /*[8][128] final V (vnorm)*/,
               int g0 = 0, int g1 = kNKvHeads);
void kv_append_n(const KvLayer& L, int64_t pos0, int n, const float* k /*[n][1024]*/, const float* v /*[n][1024]*/);
// Rel bias table repack: rel_bias is the GGUF tensor [1024 distances][16 features] (feature fastest); table gets
// kRelFeat rows of kRelTableLen floats, row f = [64 zeros][bias reversed in distance: e = 1023 - d][64 zeros].
void attn_repack_rel_bias(const float* rel_bias, float* table /*[16][kRelTableLen]*/);

struct AttnQuery {
    const float* q;     // [24][128]  after q_clip scale + L2 clip (NOT yet multiplied by tau or 1/sqrt(128); done inside)
    const float* r;     // [24][16]   after rel clip (tau applied inside)
    const float* sink;  // [24]       attn_sink_proj output (the sink logit t); softmax denominator gets + exp(t)
    const float* gate;  // [24*128]   gate_attn pre-activation (sigmoid applied inside) or nullptr
    int64_t pos;        // position of the query token (the K/V of this token must already be appended)
    float tau;          // 1.0 on SWA layers; 1 + 0.1*ln(max(1,(pos+1)/8192)) on global layers
};
struct alignas(64) AttnPartial {  // one (kv head group, key split): running softmax state of the group's 3 q heads
    float m[4];
    float l[4];
    float acc[3 * kHeadDim];
};
// Decode, one query token.  Work item = (group g in 0..7, split s in 0..nsplit-1); items write only their own AttnPartial, so
// any thread may run any item; attn_decode_finish(g) merges the splits of group g in fixed order and writes out[g*384 .. +384).
// Results are bit-reproducible for a fixed nsplit regardless of which thread runs which item.  SWA layers: nsplit 1-2;
// global layers: nsplit ~ keys/8192.  Key ranges are split on 64-position block boundaries.
void attn_decode_task(const KvLayer& L, const float* rel_table, const AttnQuery& q, int g, int split, int nsplit, AttnPartial* out);
void attn_decode_finish(const AttnQuery& q, int g, const AttnPartial* parts /*[nsplit]*/, int nsplit, float* out /*[3072]*/);
// serial convenience: scratch must hold 8*nsplit partials (index g*nsplit+s)
void attn_decode(const KvLayer& L, const float* rel_table, const AttnQuery& q, int nsplit, AttnPartial* scratch, float* out /*[3072]*/);
// Heuristic from the measurements in docs/KERNELS_MISC.md: enough (group, split) items to occupy n_threads, one split per
// 8 threads' worth for a window-sized key range.  Results change bit-wise with nsplit (not with the thread count).
inline int attn_suggest_nsplit(int64_t n_keys, int n_threads) {
    if (n_keys <= kSwaWindow + kKvBlock || n_threads <= kNKvHeads) return 1;
    return (n_threads + kNKvHeads - 1) / kNKvHeads;
}

struct AttnPrefill {  // n query tokens at positions pos0 .. pos0+n-1; their K/V must already be appended
    const float* q;     // [n][3072]
    const float* r;     // [n][384]
    const float* sink;  // [n][24]
    const float* gate;  // [n][3072] or nullptr
    const float* tau;   // [n] or nullptr (=1)
    int64_t pos0;
    int n;
};
// queries [t0,t1) of the batch for kv group g; writes out[t][g*384 .. +384) of out [n][3072]; never builds an n x n matrix
// (keys are streamed in 1024-key chunks, 2 queries x 3 heads per tile).  Independent (g, token range) items: no shared writes.
void attn_prefill_group(const KvLayer& L, const float* rel_table, const AttnPrefill& P, int g, int t0, int t1, float* out);

// ================================================================= 6. router
// logits[128] (router matvec) -> probs = sigmoid; selection score = probs + bias; top-k by selection score; weights = probs[id],
// normalised: w = (p / ((float)sum_double(p) + eps)) * route_scale  (exactly the ggml op sequence; eps 1e-20, route_scale 8).
// ids come out in descending selection score; ties: LOWEST expert index first (stable).  llama.cpp uses std::sort (unstable)
// so only exact ties can differ in order/choice.  n_expert <= 128, k <= n_expert.  bias may be nullptr.
void router_topk(const float* logits, const float* bias, int n_expert, int k, float route_scale, float eps, int* ids, float* weights);
// selection only (scores any finite floats): ids[0..k) = indices of the k largest, descending, ties lowest index first
void router_select(const float* score, int n, int k, int* ids);

// ================================================================= 7. attention-residual depth mix and loop controller
// llama.cpp mix(): states s_0..s_{ns-1} (each [n]); normed_s = rmsnorm(s, eps 1e-6, no weight); for each query q:
//   w = softmax_s(scale * normed_s . query_q), out_q = sum_s w_s * normed_s      (values are the NORMED states)
// Fused: reads every state twice, never materialises the normed copies.  ns <= 8, nq <= 4.  scale = 1/sqrtf(2304).
// The softmax is permutation invariant; ggml lists the states as [newest blocks..., oldest, cur]; here any order, summed ascending.
void depth_mix(const float* const* states, int ns, const float* queries /*[nq][n]*/, int nq, float* const* outs, size_t n,
               float scale, float eps = 1e-6f);
// loop controller (llama.cpp control()): gate_in[0..n) = rmsnorm(cur), gate_in[n..2n) = rmsnorm(msg) (input of ctrl_down matvec)
void ctrl_gate_input(const float* cur, const float* msg, float* gate_in /*[2n]*/, size_t n, float eps = 1e-6f);
// out = (cur - tanh(forget_pre)*cur) + tanh(gain_pre)*msg   (gain_pre/forget_pre = outputs of the rank-64 up matvecs); out == cur ok
void ctrl_apply(const float* cur, const float* msg, const float* gain_pre, const float* forget_pre, float* out, size_t n);

// ================================================================= 8. converters and fused helpers
void bf16_to_f32(const uint16_t* x, float* y, size_t n);
void f16_to_f32(const uint16_t* x, float* y, size_t n);
void f32_to_f16(const float* x, uint16_t* y, size_t n);   // round to nearest even
void f32_to_bf16(const float* x, uint16_t* y, size_t n);  // round to nearest even
void vadd(const float* a, const float* b, float* y, size_t n);          // y = a + b            (residual add)
void vmul(const float* a, const float* b, float* y, size_t n);          // y = a * b
void vscale(const float* x, float s, float* y, size_t n);               // y = x * s
void vaxpy(float* y, float a, const float* x, size_t n);                // y += a * x
void vadd_scale(const float* a, const float* b, float s, float* y, size_t n);  // y = (a + b) * s   (conv-with-residual tail)
void vmul_sigmoid(const float* x, const float* g, float* y, size_t n);  // y = x * sigmoid(g)   (attention output gate)

// ================================================================= scalar references (same names; double accumulation)
namespace ref {
void vexp(const float* x, float* y, size_t n);
void vtanh(const float* x, float* y, size_t n);
void vsigmoid(const float* x, float* y, size_t n);
void vsitu(const float* gate, const float* up, float* y, size_t n, float beta = 4.0f, float lin = 25.0f);
void vsoftcap(const float* x, float* y, size_t n, float cap = 30.0f);
void rmsnorm(const float* x, const float* w, float* y, size_t n, float eps = 1e-6f);
void quantize_q8_0(const float* x, block_q8_0* out, size_t n, Q8Mode mode = Q8Mode::Ref);  // literal ggml scalar code
void l2clip_heads(float* x, const float* chan_scale, int n_heads, int dim, float threshold, float post_scale = 1.0f);
void vnorm_heads(const float* x, float* y, int n_heads, int dim, float eps = 1e-6f);
// ggml layouts: w [C][4] (index 3 = current), hist [C][3] (index 0 = oldest); dyn [n][n_heads*4]; hist updated
void sconv_seq(const float* x, float* hist, const float* w, float* y, size_t n, size_t C, const float* dyn = nullptr, int head_dim = kHeadDim);
// plain arrays: K,V [npos][8*128] (fp16-decoded values), rel_bias [1024][16] GGUF layout; swa: only distances < 1024 visible
void attn_decode(const float* K, const float* V, int64_t pos, bool swa, const float* rel_bias, const float* q, const float* r,
                 const float* sink, const float* gate, float tau, float* out /*[3072]*/);
void router_select(const float* score, int n, int k, int* ids);  // std::sort with ggml's comparator (descending)
void router_topk(const float* logits, const float* bias, int n_expert, int k, float route_scale, float eps, int* ids, float* weights);
void depth_mix(const float* const* states, int ns, const float* queries, int nq, float* const* outs, size_t n, float scale, float eps = 1e-6f);
void ctrl_gate_input(const float* cur, const float* msg, float* gate_in, size_t n, float eps = 1e-6f);
void ctrl_apply(const float* cur, const float* msg, const float* gain_pre, const float* forget_pre, float* out, size_t n);
}  // namespace ref

}  // namespace ink

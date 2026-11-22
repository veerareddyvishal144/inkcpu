// Scalar reference of the DSpark drafter, written from docs/DSPARK.md section 4 (double accumulation everywhere except the integer-dot primitive
// matvec ink::gemv_q8_0, which is bit-identical to its scalar reference gemv_q8_0_ref, tested by test_gemv). Activations of the Q8_0 matmuls are quantised
// exactly like the engine (B2 quantiser, Q8Mode::X86), so reference and drafter see the same quantisation points. The chain is UNFOLDED (llama.cpp style:
// two lm_head products per position), which makes the comparison with the folded drafter an end-to-end check of the fold. Owner: agent S1.
#pragma once
#include <vector>

#include "drafter_impl.h"

namespace ink {
namespace dr {
namespace rd {

// f_c(t) for one tap row (4 x 2304): softmax(fuse row c) weights, RMS with the consumer's gain
void fuse(const DrafterImpl& D, const float* taps, int c, std::vector<float>& f);
// ctx K (normed with k_ctx_norm, RoPE at pos; rounded through fp16 like the cache) and V (fp16-rounded) of one position for layer l; K, V: 1024 floats
void ctx_kv(const DrafterImpl& D, int l, const float* f, int64_t pos, std::vector<float>& K, std::vector<float>& V);

struct BlockOut {
    std::vector<float> X[NL];   // [8][2304] stream after each layer
    std::vector<float> hid;     // [8][2304] final normed (slots 1..n valid)
};
// ck/cv[l]: [npos][1024] ctx keys / values of ALL committed positions 0..p-1 (the SWA layers use the last 1024). anchor at position p.
void block_pass(const DrafterImpl& D, const std::vector<float> ck[NL], const std::vector<float> cv[NL], int64_t p, int32_t anchor, BlockOut* out);

struct ChainOut {
    std::vector<std::vector<float>> lat;     // [n][256]
    std::vector<std::vector<float>> logits;  // [n][n_vocab] softcapped, UNFOLDED (two lm_head products)
    std::vector<int32_t> tokens;             // argmax (or forced)
};
// f_rnn: 2304 floats; hid: [8][2304]. force: if non-null the chain feeds back these tokens instead of its argmax.
void chain(const DrafterImpl& D, const float* f_rnn, const float* hid, int32_t anchor, int n, const int32_t* force, ChainOut* out);

}  // namespace rd
}  // namespace dr
}  // namespace ink

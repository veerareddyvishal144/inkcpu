// Scalar float32 reference forward pass for Inkling 2 Mini (arch inkling2_loop). Owner: agent A (core).
//
// One token at a time, all state carried in RefState (per-visit KV cache / SWA ring, per-visit conv histories). Mirrors
// llama.cpp's graph (src/models/inkling2.cpp + inkling2-loop.cpp + build_moe_ffn) operation by operation; see docs/MODEL.md.
// Weights are dequantised on the fly (Q8_0, NVFP4, BF16, F16, F32). It is a correctness oracle, not a fast path.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "ink/gguf.h"
#include "ink/model.h"

namespace ink {

struct RefOptions {
    // true: quantise the F32 activations feeding Q8_0 / NVFP4 matmuls to Q8_0 blocks (exactly like ggml's vec_dot_type
    // path) and convert activations feeding BF16 matmuls to bf16, so logits track llama.cpp tightly.
    // false: float activations everywhere (weights are still dequantised from their stored format).
    bool act_quant_like_ggml = true;
    // Tri-state switches for the other ggml roundings (-1 = follow act_quant_like_ggml, 0 = off, 1 = on):
    int kv_f16 = -1;     // K and V are stored in the cache as F16 (llama.cpp default cache type)
    int q_f16 = -1;      // Q rounded to F16 before the K.Q dot. MEASURED: llama.cpp does NOT do this (default off even with act_quant_like_ggml)
    int bias_f16 = -1;   // relative bias (incl. the -sink term) is rounded to F16 (flash attention folds it into an F16 mask)
    int n_threads = 8;   // OpenMP threads for the matrix rows (keep <= 8 in tests: the host is shared)
    // Config override hook: > 0 replaces the sliding window of the SWA layers (cache ring size and mask). The relative
    // bias table is not touched (distances < window always index inside it).
    int sliding_window = 0;
    // Test-only: relative pseudo-random perturbation (+-noise) of the normed token embedding, used to measure how
    // sensitive the logits are to float-rounding-sized changes (0 = off).
    float debug_embed_noise = 0.0f;
};

struct RefVisitState {
    // Attention cache of this visit, K/V after conv + clip / RMS norm: [cell][kv_width]. SWA visits use a ring of at
    // most `window` cells, global visits grow without bound. cell_pos[i] = absolute position stored in cell i.
    std::vector<float>   k, v;
    std::vector<int32_t> cell_pos;
    int n_cells = 0;
    int ring_next = 0;
    // Conv histories = the last (n_conv-1) INPUTS of the conv, oldest first: [(n_conv-1)][channels]
    std::vector<float> hist_k, hist_v;       // kv_width channels each
    std::vector<float> hist_attn, hist_ffn;  // n_embd channels each (wo output / MoE output)
};

struct RefState {
    int pos = 0;                              // number of tokens processed so far (= position of the next token)
    std::vector<RefVisitState> visits;        // one per visit (48), indexed by Visit::index
    size_t bytes() const;
};

class RefModel {
public:
    // Reads config + weights from `g` (which must outlive the model). Throws std::runtime_error on mismatch.
    explicit RefModel(const GgufFile& g, const RefOptions& opt = RefOptions());

    const ModelConfig&        config() const { return cfg_; }
    const Weights&            weights() const { return w_; }
    const std::vector<Visit>& plan() const { return plan_; }
    const RefOptions&         options() const { return opt_; }
    int                       window() const { return window_; }

    RefState new_state() const;
    void     reset(RefState& st) const;       // drop all history (pos = 0, empty caches, zero conv histories)

    // Process ONE token at position st.pos and advance the state. If logits != nullptr it receives n_vocab floats
    // (30*tanh(x/30) already applied, padded vocabulary rows dropped); pass nullptr for prompt tokens to skip the lm head.
    void forward(RefState& st, int32_t token, float* logits) const;
    // Convenience: feed a whole prompt token by token and return the logits of the last token.
    std::vector<float> run(RefState& st, const std::vector<int32_t>& tokens) const;

    // Debug hook: called with llama.cpp's tensor names ("Qcur", "Kcur", "Vcur", "Rcur", "attn_out", "ffn_inp", "ffn_out",
    // "l_out", "result_norm") and the llama.cpp cache-slot number (`il`), for single-token runs.
    using TraceFn = std::function<void(const char* name, int il, const float* data, size_t n)>;
    void set_trace(TraceFn fn) { trace_ = std::move(fn); }

private:
    void visit(RefState& st, const Visit& vis, std::vector<float>& cur, std::vector<std::vector<float>>& blocks,
               const float* msgs, int pos, float tau) const;
    void moe(const LayerWeights& L, const float* x, float* out, int il) const;
    void trace(const char* name, int il, const float* d, size_t n) const { if (trace_) trace_(name, il, d, n); }

    RefOptions   opt_;
    ModelConfig  cfg_;
    Weights      w_;
    std::vector<Visit> plan_;
    int  window_ = 0;
    bool aq_ = true, kv16_ = true, q16_ = true, bias16_ = true;
    TraceFn trace_;
};

}  // namespace ink

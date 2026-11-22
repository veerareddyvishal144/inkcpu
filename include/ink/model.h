// ModelConfig / Weights / visit plan for Inkling 2 Mini. Owner: agent A (core); S4 added the second GGUF layout.
// Two GGUF layouts of the same weights are accepted (docs/ENGINE.md "GGUF layouts"):
//   arch `inkling2_loop` (interim): block_count = 24 physical layers, loop.{start,end,count}, blk.L.loop_msg_query + loop_*_ctrl_*
//   arch `inkling2` (llama.cpp canonical): block_count = 48 visits, num_loops, loop.core_{start,end}, per-visit arrays in
//     llama.cpp cache-slot order (physical layers first, then the extra core passes), global loop_res_query / loop_kv_res_query
// Both are parsed into the same ModelConfig / Weights (physical layers, per-layer [D,4] message queries), so nothing downstream
// sees the difference. Everything numeric comes from the GGUF KV table except the few rules that llama.cpp hard-codes (listed in
// ModelConfig::hard_coded below and in docs/MODEL.md). Weights are pointers into the GgufFile mmap (no copies), except the
// message-query blocks synthesised for the canonical layout (owned by Weights::owned_f32).
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ink/gguf.h"

namespace ink {

// One tensor inside the mmap. ne[0] is the contiguous dimension (K for a matrix whose rows are output channels).
struct TensorRef {
    const uint8_t* data = nullptr;
    int            type = -1;                 // ggml type id: T_F32, T_BF16, T_Q8_0, T_NVFP4, ...
    int            n_dims = 0;
    int64_t        ne[4] = {0, 0, 0, 0};
    size_t         nbytes = 0;

    bool    valid() const { return data != nullptr; }
    int64_t k() const { return ne[0]; }                        // reduction length of a matrix row
    int64_t rows() const { return ne[1]; }                     // output rows (per expert for 3D tensors)
    int64_t experts() const { return n_dims >= 3 ? ne[2] : 1; }
    size_t  row_bytes() const { return ggml_row_bytes(type, ne[0]); }
    size_t  expert_bytes() const { return row_bytes() * (size_t)ne[1]; }
    const uint8_t* row(int64_t r, int64_t expert = 0) const { return data + ((size_t)expert * ne[1] + r) * row_bytes(); }
    const float*   f32() const { return (const float*)data; }  // only valid if type == T_F32
};

struct LoopCtrl {            // rank-gate controller of one input stream (Q, K, V or FFN) of a core layer
    TensorRef down;          // [2*n_embd -> rank]  (F32 in the community GGUF)
    TensorRef gain;          // [rank -> n_embd]
    TensorRef forget;        // [rank -> n_embd]
};

struct LayerWeights {        // one PHYSICAL layer (shared by every pass that visits it)
    int  phys = -1;
    bool is_swa = false;     // sliding-window attention (window n_swa) vs global
    bool in_core = false;    // layer inside the looped core (has controllers and message queries)

    // attention projections
    TensorRef wq, wk, wv, wo;        // Q8_0 in the community GGUF
    TensorRef attn_gate;             // output gate projection [n_embd -> n_head*head_dim]
    TensorRef attn_q_clip;           // F32 [head_dim, n_head] per-channel Q scale
    TensorRef attn_sink_proj;        // BF16 [n_embd -> n_head] per-query sink logit
    TensorRef attn_rel;              // [n_embd -> n_rel*n_head] relative-bias features
    TensorRef attn_rel_bias;         // F32 [n_rel, extent] bias table, index = distance
    TensorRef attn_rel_clip;         // F32 [n_rel, n_head] per-channel scale of the features
    TensorRef attn_k_conv, attn_v_conv;       // F32 [n_conv, kv_width] static depthwise taps
    TensorRef attn_post_conv, ffn_post_conv;  // F32 [n_conv, n_embd] taps after wo / after the MoE
    // SWA layers only
    TensorRef attn_norm;             // F32 [n_embd]
    // global layers only
    TensorRef attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm;   // F32 [n_embd]
    TensorRef attn_k_conv_proj, attn_v_conv_proj;                   // BF16 [n_embd -> n_conv*n_head_kv] dynamic taps
    // attention-residual depth mixing
    TensorRef attn_res_query;        // F32 [n_embd] (SWA, 1 query) or [n_embd, 3] (global); absent in layer 0
    TensorRef ffn_res_query;         // F32 [n_embd]
    // MoE
    TensorRef ffn_norm;              // F32 [n_embd]
    TensorRef ffn_gate_inp;          // router [n_embd -> n_expert] (F32 in the GGUF)
    TensorRef exp_probs_b;           // F32 [n_expert] selection bias
    TensorRef ffn_gate_up_exps;      // [n_embd, 2*n_ff_exp, n_expert]: rows 0..n_ff-1 = gate, n_ff..2n_ff-1 = up
    TensorRef ffn_down_exps;         // [n_ff_exp, n_embd, n_expert]
    TensorRef ffn_gate_up_scale;     // F32 [n_expert] per-expert scale (NVFP4 layers only), applied to the whole gate_up result
    TensorRef ffn_down_scale;        // F32 [n_expert] (NVFP4 layers only)
    // loop (core layers only)
    TensorRef loop_msg_query;        // F32 [n_embd, 4]: queries of the Q, K, V, FFN messages
    LoopCtrl  ctrl[4];               // index 0 = Q, 1 = K, 2 = V, 3 = FFN
};

struct Weights {
    TensorRef tok_embd;              // [n_embd, n_vocab_padded] (Q8_0 in the GGUF)
    TensorRef tok_norm;              // F32 [n_embd] embed_norm
    TensorRef output;                // [n_embd, n_vocab_padded] lm head (Q8_0)
    TensorRef output_norm;           // F32 [n_embd]
    TensorRef output_res_query;      // F32 [n_embd]
    std::vector<LayerWeights> layers;   // physical layers 0..n_layer-1
    // storage behind the TensorRefs that do not point into the mmap (canonical layout: per-layer loop_msg_query blocks assembled
    // from loop_res_query / loop_kv_res_query); shared so copies of Weights keep the pointers valid
    std::vector<std::shared_ptr<std::vector<float>>> owned_f32;
};

struct ModelConfig {
    std::string arch;                           // "inkling2_loop" (interim layout) or "inkling2" (canonical layout)
    bool canonical_layout() const { return arch == "inkling2"; }
    // KV-derived (prefix = arch + ".")
    int   n_layer = 0, n_embd = 0, n_ff_exp = 0;   // n_layer = PHYSICAL layers (24) in both layouts
    int   n_visits = 0;                         // layer visits per token (48) = canonical block_count = n_slots()
    int   n_head = 0, n_head_kv = 0, head_dim = 0;
    float rms_eps = 0;
    int   n_expert = 0, n_expert_used = 0;
    float expert_weights_scale = 1.0f;          // route_scale (8.0)
    int   loop_start = 0, loop_end = 0, loop_count = 1, loop_gate_rank = 0;
    bool  loop_varying = false;
    int   n_vocab_padded = 0, n_vocab = 0;      // rows in embed/unembed / real vocabulary
    int   sliding_window = 0;                   // n_swa
    std::vector<bool> is_swa;                   // per physical layer
    int   n_conv = 0;                           // shortconv.l_cache (kernel size)
    float situ_beta = 0, situ_linear_beta = 0;  // 4.0, 25.0
    int   attn_res_block_size = 0;
    float final_logit_softcap = 0;              // 30.0 (0 = off)
    int   n_rel = 0, rel_extent = 0;            // relative_length (16), relative_extent (1024)
    float qk_clip = 0, rel_clip = 0;            // 22.627..., 8.0
    int   log_scale_floor = 0;
    float log_scale_alpha = 0;
    int   context_length = 0;

    // Numeric rules that are NOT in the KV table (hard-coded in llama.cpp inkling2.cpp / llama-graph.cpp, kept here as
    // named constants so nothing is silently baked into the forward code). See docs/MODEL.md.
    struct HardCoded {
        float expert_weights_norm_eps = 1e-20f;   // hparams.expert_weights_norm_eps: w / (sum(w) + eps)
        float clip_norm_eps = 1e-6f;              // clip(): norm = sqrt(sum x^2) + eps (added AFTER the sqrt)
        float mix_rms_eps = 1e-6f;                // depth mix: RMS norm of the states
        float ctrl_rms_eps = 1e-6f;               // loop controller: RMS norm of cur and message
        // mix softmax scale = 1/sqrtf(n_embd); attention scale = 1/sqrtf(head_dim); gating = sigmoid; no (1+w) norms
    } hard_coded;

    // derived
    int kv_width() const { return n_head_kv * head_dim; }
    int q_width() const { return n_head * head_dim; }
    int n_conv_channels() const { return 2 * kv_width() + 2 * n_embd; }   // K, V, wo-out, MoE-out conv histories
    int n_slots() const { return n_layer + (loop_count - 1) * (loop_end - loop_start); }   // 48

    // Reads every field from the GGUF; returns false and fills *err when a key is missing or inconsistent.
    static bool from_gguf(const GgufFile& g, ModelConfig* out, std::string* err);
};

// One of the 48 layer visits in execution order.
struct Visit {
    int  index = 0;        // execution order 0..47; = the KV / conv-state slot id used by the engine (ARCH.md section 1)
    int  phys = 0;         // physical layer whose weights are used
    int  pass = 0;         // 0 for prelude / postlude / first core pass, 1..n_passes-1 for the extra core passes
    int  kv_slot = 0;      // alias of index (the per-visit KV cache / SWA ring / conv-history slot)
    int  llama_slot = 0;   // llama.cpp cache-slot number: il for pass 0, prelude and postlude (il = phys), n_phys + (pass-1)*core_len + (phys-start) for extra passes
    bool in_core = false;
    bool is_swa = false;
    bool has_message = false;   // pass > 0: message queries + controllers are applied
};

// 48 visits: prelude L0..start-1, core passes 0..n_passes-1 over [start, end), postlude end..n_layer-1.
std::vector<Visit> visit_plan(const ModelConfig& cfg);

// Load every tensor of a physical layer / the model. Returns false (and *err) if a required tensor is missing or has
// an unexpected shape; the tensors that exist only for some layer kinds are validated per kind.
bool load_weights(const GgufFile& g, const ModelConfig& cfg, Weights* out, std::string* err);

}  // namespace ink

#include "ink/model.h"

#include <cstdio>
#include <cstring>

#include "ink/ggml_types.h"

namespace ink {

namespace {
struct Ctx {
    const GgufFile& g;
    std::string     p;   // arch prefix with the trailing dot
    std::string*    err;
    bool            ok = true;
    void fail(const std::string& m) {
        if (ok && err) *err = m;
        ok = false;
    }
    int geti(const char* k) {
        uint32_t v = 0;
        if (!g.get_u32(p + k, &v)) fail(std::string("missing/invalid KV ") + p + k);
        return (int)v;
    }
    float getf(const char* k) {
        float v = 0;
        if (!g.get_f32(p + k, &v)) fail(std::string("missing/invalid KV ") + p + k);
        return v;
    }
    int geti_opt(const char* k, int def) {
        uint32_t v = 0;
        return g.get_u32(p + k, &v) ? (int)v : def;
    }
    // a scalar or a per-layer array holding one repeated value
    int geti_uniform(const char* k, int n_layer) {
        uint32_t v = 0;
        if (g.get_u32(p + k, &v)) return (int)v;
        std::vector<int64_t> a;
        if (!g.get_arr_i64(p + k, &a) || (int)a.size() != n_layer || a.empty()) {
            fail(std::string("missing/invalid KV ") + p + k);
            return 0;
        }
        for (int64_t x : a)
            if (x != a[0]) fail(std::string("non-uniform per-layer array not supported: ") + p + k);
        return (int)a[0];
    }
};
}  // namespace

bool ModelConfig::from_gguf(const GgufFile& g, ModelConfig* out, std::string* err) {
    ModelConfig c;
    if (!g.get_string("general.architecture", &c.arch)) { if (err) *err = "missing general.architecture"; return false; }
    const bool canon = c.arch == "inkling2";
    if (c.arch != "inkling2_loop" && !canon) {
        if (err) *err = "unsupported architecture '" + c.arch + "' (need inkling2_loop or inkling2)";
        return false;
    }
    Ctx x{g, c.arch + ".", err};
    // Layer count and loop geometry. Interim layout: block_count = physical layers, loop.{start,end,count}. Canonical layout:
    // block_count = visits (48), num_loops, loop.core_{start,end}; physical layers = visits - (num_loops-1)*(core_end-core_start).
    if (canon) {
        c.n_visits = x.geti("block_count");
        c.loop_count = x.geti("num_loops");
        c.loop_start = x.geti("loop.core_start");
        c.loop_end = x.geti("loop.core_end");
        c.n_layer = c.n_visits - (c.loop_count - 1) * (c.loop_end - c.loop_start);
    } else {
        c.n_layer = x.geti("block_count");
        c.loop_start = x.geti("loop.start");
        c.loop_end = x.geti("loop.end");
        c.loop_count = x.geti("loop.count");
        g.get_bool(x.p + "loop.varying", &c.loop_varying);
        c.n_visits = c.n_layer + (c.loop_count - 1) * (c.loop_end - c.loop_start);
    }
    const int n_arr = canon ? c.n_visits : c.n_layer;   // length of the per-layer (interim) / per-visit (canonical) arrays
    c.n_embd = x.geti("embedding_length");
    c.n_ff_exp = x.geti("expert_feed_forward_length");
    c.n_head = x.geti_uniform("attention.head_count", n_arr);
    c.n_head_kv = x.geti_uniform("attention.head_count_kv", n_arr);
    c.head_dim = x.geti("attention.key_length");
    if (x.geti("attention.value_length") != c.head_dim) x.fail("key_length != value_length");
    c.rms_eps = x.getf("attention.layer_norm_rms_epsilon");
    c.n_expert = x.geti("expert_count");
    c.n_expert_used = x.geti("expert_used_count");
    g.get_f32(x.p + "expert_weights_scale", &c.expert_weights_scale);
    c.loop_gate_rank = x.geti("loop.gate_rank");
    c.n_vocab_padded = x.geti("vocab_size");
    c.n_vocab = x.geti("vocab_size_unpadded");
    c.sliding_window = x.geti("attention.sliding_window");
    std::vector<bool> swa_arr;
    if (!g.get_arr_bool(x.p + "attention.sliding_window_pattern", &swa_arr)) x.fail("missing attention.sliding_window_pattern");
    c.n_conv = x.geti("shortconv.l_cache");
    c.situ_beta = x.getf("activation.situ_beta");
    c.situ_linear_beta = x.getf("activation.situ_linear_beta");
    c.attn_res_block_size = x.geti("attn_res.block_size");
    c.final_logit_softcap = 0.0f;
    g.get_f32(x.p + "final_logit_softcapping", &c.final_logit_softcap);
    c.n_rel = x.geti("attention.relative_length");
    c.rel_extent = x.geti("attention.relative_extent");
    c.qk_clip = x.getf("attention.qk_clip");
    c.rel_clip = x.getf("attention.relative_clip");
    c.log_scale_floor = x.geti("attention.log_scale_floor");
    c.log_scale_alpha = x.getf("attention.log_scale_alpha");
    c.context_length = x.geti_opt("context_length", 0);
    if (!x.ok) return false;

    auto bad = [&](const std::string& m) { if (err) *err = "inconsistent config: " + m; return false; };
    if (c.loop_varying) return bad("variable loops (loop.varying) are not supported");
    if (c.loop_count < 2 || c.loop_start <= 0 || c.loop_start >= c.loop_end || c.loop_end > c.n_layer) return bad("loop range");
    if ((int)swa_arr.size() != n_arr) return bad("sliding_window_pattern length != block_count");
    if (canon) {
        // per-visit arrays are in llama.cpp cache-slot order: physical layers 0..n_layer-1 first, then the extra core passes
        // (slot n_layer + (pass-1)*core + (phys-start)); keep the physical part and check that the repeats agree with it
        c.is_swa.assign(swa_arr.begin(), swa_arr.begin() + c.n_layer);
        const int core = c.loop_end - c.loop_start;
        for (int v = c.n_layer; v < c.n_visits; ++v)
            if (swa_arr[v] != c.is_swa[c.loop_start + (v - c.n_layer) % core]) return bad("per-visit sliding_window_pattern does not repeat the core layers");
    } else {
        c.is_swa = swa_arr;
    }
    if (c.n_head % c.n_head_kv) return bad("n_head not a multiple of n_head_kv");
    if (c.loop_start % c.attn_res_block_size || c.loop_end % c.attn_res_block_size) return bad("loop range not block aligned");
    if (c.n_vocab > c.n_vocab_padded) return bad("vocab_size_unpadded > vocab_size");
    if (c.n_conv < 2 || c.n_conv > 64) return bad("shortconv.l_cache");
    if (c.n_expert_used > c.n_expert) return bad("expert_used_count > expert_count");
    *out = std::move(c);
    return true;
}

std::vector<Visit> visit_plan(const ModelConfig& c) {
    std::vector<Visit> v;
    const int core = c.loop_end - c.loop_start;
    auto add = [&](int phys, int pass, int llama_slot, bool in_core) {
        Visit t;
        t.index = (int)v.size();
        t.kv_slot = t.index;
        t.phys = phys;
        t.pass = pass;
        t.llama_slot = llama_slot;
        t.in_core = in_core;
        t.is_swa = c.is_swa[phys];
        t.has_message = pass > 0;
        v.push_back(t);
    };
    for (int l = 0; l < c.loop_start; ++l) add(l, 0, l, false);
    for (int p = 0; p < c.loop_count; ++p)
        for (int l = c.loop_start; l < c.loop_end; ++l)
            add(l, p, p == 0 ? l : c.n_layer + (p - 1) * core + (l - c.loop_start), true);
    for (int l = c.loop_end; l < c.n_layer; ++l) add(l, 0, l, false);
    return v;
}

// ---------------------------------------------------------------- weights
namespace {
struct Loader {
    const GgufFile& g;
    std::string*    err;
    bool            ok = true;
    TensorRef get(const std::string& name, bool required, int nd, int64_t n0, int64_t n1 = 0, int64_t n2 = 0) {
        TensorRef r;
        const GgufTensorInfo* t = g.tensor(name);
        if (!t) {
            if (required) fail("missing tensor " + name);
            return r;
        }
        const int64_t want[3] = {n0, n1, n2};
        bool shape_ok = t->n_dims == nd;
        for (int d = 0; d < nd && shape_ok; ++d) shape_ok = t->ne[d] == want[d];
        if (!shape_ok) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "tensor %s has shape [%lld,%lld,%lld] (%d dims), expected %d dims [%lld,%lld,%lld]", name.c_str(),
                          (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], t->n_dims, nd, (long long)n0, (long long)n1, (long long)n2);
            fail(buf);
            return r;
        }
        r.data = t->data;
        r.type = t->type;
        r.n_dims = t->n_dims;
        for (int d = 0; d < 4; ++d) r.ne[d] = t->ne[d];
        r.nbytes = t->nbytes;
        return r;
    }
    void fail(const std::string& m) {
        if (ok && err) *err = m;
        ok = false;
    }
};
}  // namespace

bool load_weights(const GgufFile& g, const ModelConfig& c, Weights* out, std::string* err) {
    Loader L{g, err};
    Weights w;
    const int D = c.n_embd, NR = c.n_rel, NC = c.n_conv, KV = c.kv_width(), QW = c.q_width();
    w.tok_embd = L.get("token_embd.weight", true, 2, D, c.n_vocab_padded);
    w.tok_norm = L.get("token_embd_norm.weight", true, 1, D);
    w.output = L.get("output.weight", true, 2, D, c.n_vocab_padded);
    w.output_norm = L.get("output_norm.weight", true, 1, D);
    w.output_res_query = L.get("output_res_query.weight", true, 1, D);

    w.layers.resize(c.n_layer);
    for (int i = 0; i < c.n_layer; ++i) {
        LayerWeights& l = w.layers[i];
        const std::string b = "blk." + std::to_string(i) + ".";
        l.phys = i;
        l.is_swa = c.is_swa[i];
        l.in_core = i >= c.loop_start && i < c.loop_end;
        l.wq = L.get(b + "attn_q.weight", true, 2, D, QW);
        l.wk = L.get(b + "attn_k.weight", true, 2, D, KV);
        l.wv = L.get(b + "attn_v.weight", true, 2, D, KV);
        l.wo = L.get(b + "attn_output.weight", true, 2, QW, D);
        l.attn_gate = L.get(b + "attn_gate.weight", true, 2, D, QW);
        l.attn_q_clip = L.get(b + "attn_q_clip.weight", true, 2, c.head_dim, c.n_head);
        l.attn_sink_proj = L.get(b + "attn_sink_proj.weight", true, 2, D, c.n_head);
        l.attn_rel = L.get(b + "attn_rel.weight", true, 2, D, (int64_t)NR * c.n_head);
        l.attn_rel_bias = L.get(b + "attn_rel_bias.weight", true, 2, NR, c.rel_extent);
        l.attn_rel_clip = L.get(b + "attn_rel_clip.weight", true, 2, NR, c.n_head);
        l.attn_k_conv = L.get(b + "attn_k_conv.weight", true, 2, NC, KV);
        l.attn_v_conv = L.get(b + "attn_v_conv.weight", true, 2, NC, KV);
        l.attn_post_conv = L.get(b + "attn_post_conv.weight", true, 2, NC, D);
        l.ffn_post_conv = L.get(b + "ffn_post_conv.weight", true, 2, NC, D);
        if (l.is_swa) {
            l.attn_norm = L.get(b + "attn_norm.weight", true, 1, D);
        } else {
            l.attn_q_inp_norm = L.get(b + "attn_q_inp_norm.weight", true, 1, D);
            l.attn_k_inp_norm = L.get(b + "attn_k_inp_norm.weight", true, 1, D);
            l.attn_v_inp_norm = L.get(b + "attn_v_inp_norm.weight", true, 1, D);
            l.attn_k_conv_proj = L.get(b + "attn_k_conv_proj.weight", true, 2, D, (int64_t)NC * c.n_head_kv);
            l.attn_v_conv_proj = L.get(b + "attn_v_conv_proj.weight", true, 2, D, (int64_t)NC * c.n_head_kv);
        }
        if (i > 0) {
            if (l.is_swa) l.attn_res_query = L.get(b + "attn_res_query.weight", true, 1, D);
            else          l.attn_res_query = L.get(b + "attn_res_query.weight", true, 2, D, 3);
        }
        l.ffn_res_query = L.get(b + "ffn_res_query.weight", true, 1, D);
        l.ffn_norm = L.get(b + "ffn_norm.weight", true, 1, D);
        l.ffn_gate_inp = L.get(b + "ffn_gate_inp.weight", true, 2, D, c.n_expert);
        l.exp_probs_b = L.get(b + "exp_probs_b.bias", false, 1, c.n_expert);
        l.ffn_gate_up_exps = L.get(b + "ffn_gate_up_exps.weight", true, 3, D, 2 * (int64_t)c.n_ff_exp, c.n_expert);
        l.ffn_down_exps = L.get(b + "ffn_down_exps.weight", true, 3, c.n_ff_exp, D, c.n_expert);
        l.ffn_gate_up_scale = L.get(b + "ffn_gate_up_exps.scale", false, 1, c.n_expert);
        l.ffn_down_scale = L.get(b + "ffn_down_exps.scale", false, 1, c.n_expert);
        if (l.in_core) {
            if (g.tensor(b + "loop_msg_query.weight")) {   // interim layout: per-layer [D, 4] block, rows Q, K, V, FFN
                l.loop_msg_query = L.get(b + "loop_msg_query.weight", true, 2, D, 4);
            } else {
                // canonical layout: two global tensors [D, 2*core] indexed by i = layer - loop_start:
                //   Q = loop_res_query[2i], FFN = loop_res_query[2i+1], K = loop_kv_res_query[2i], V = loop_kv_res_query[2i+1]
                // (verified bit-exact against the interim file's blk.L.loop_msg_query). Assemble the same [D, 4] block in owned memory.
                const int64_t ncore = c.loop_end - c.loop_start;
                const TensorRef rq = L.get("loop_res_query.weight", true, 2, D, 2 * ncore);
                const TensorRef kq = L.get("loop_kv_res_query.weight", true, 2, D, 2 * ncore);
                if (rq.valid() && kq.valid()) {
                    if (rq.type != T_F32 || kq.type != T_F32) {
                        L.fail("loop_res_query / loop_kv_res_query must be F32");
                    } else {
                        auto buf = std::make_shared<std::vector<float>>((size_t)4 * D);
                        const int64_t i2 = 2 * (i - c.loop_start);
                        std::memcpy(buf->data() + 0 * D, rq.row(i2), (size_t)D * 4);       // Q
                        std::memcpy(buf->data() + 1 * D, kq.row(i2), (size_t)D * 4);       // K
                        std::memcpy(buf->data() + 2 * D, kq.row(i2 + 1), (size_t)D * 4);   // V
                        std::memcpy(buf->data() + 3 * D, rq.row(i2 + 1), (size_t)D * 4);   // FFN
                        l.loop_msg_query.data = (const uint8_t*)buf->data();
                        l.loop_msg_query.type = T_F32;
                        l.loop_msg_query.n_dims = 2;
                        l.loop_msg_query.ne[0] = D;
                        l.loop_msg_query.ne[1] = 4;
                        l.loop_msg_query.nbytes = (size_t)4 * D * 4;
                        w.owned_f32.push_back(buf);
                    }
                }
            }
            // controllers: interim blk.L.loop_{q,k,v,ffn}_ctrl_{down,gain,forget}, canonical blk.L.loop_{attn,k,v,mlp}_{down,gain,forget}
            static const char* interim[4] = {"loop_q_ctrl_", "loop_k_ctrl_", "loop_v_ctrl_", "loop_ffn_ctrl_"};
            static const char* canonical[4] = {"loop_attn_", "loop_k_", "loop_v_", "loop_mlp_"};
            for (int j = 0; j < 4; ++j) {
                const std::string n = g.tensor(b + interim[j] + "down.weight") ? b + interim[j] : b + canonical[j];
                l.ctrl[j].down = L.get(n + "down.weight", true, 2, 2 * (int64_t)D, c.loop_gate_rank);
                l.ctrl[j].gain = L.get(n + "gain.weight", true, 2, c.loop_gate_rank, D);
                l.ctrl[j].forget = L.get(n + "forget.weight", true, 2, c.loop_gate_rank, D);
            }
        }
        if (!L.ok) return false;
    }
    if (!L.ok) return false;
    *out = std::move(w);
    return true;
}

}  // namespace ink

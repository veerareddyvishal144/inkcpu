// GGUF reader / ModelConfig / visit_plan / Weights checks on the real community GGUF (no inference).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "ink/ggml_types.h"
#include "ink/gguf.h"
#include "ink/model.h"

static int g_fail = 0;
#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
    } while (0)

static const char* gguf_path() {
    const char* e = std::getenv("INK_GGUF");
    return e ? e : "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf";
}

int main() {
    using namespace ink;
    auto t0 = std::chrono::steady_clock::now();
    std::string err;
    auto g = GgufFile::open(gguf_path(), &err);
    double open_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!g) { std::printf("FAIL open: %s\n", err.c_str()); return 1; }
    std::printf("open: %.2f ms, version %u, %zu tensors, %zu kv, data_offset %llu, file %zu bytes\n", open_ms, g->version(), g->tensors().size(),
                g->kvs().size(), (unsigned long long)g->data_offset(), g->file_size());
    CHECK(open_ms < 1000.0);   // mmap + header index only (measured ~10-15 ms warm)
    CHECK(g->version() == 3);
    CHECK(g->tensors().size() == 744);
    CHECK(g->kvs().size() == 63);
    CHECK(g->alignment() == 32);

    // every tensor: aligned, inside the file, size matches the type table; the tensors tile the data section exactly
    size_t total = 0;
    for (const auto& t : g->tensors()) {
        CHECK(t.offset % g->alignment() == 0);
        CHECK(t.data == g->base() + g->data_offset() + t.offset);
        CHECK(g->data_offset() + t.offset + t.nbytes <= g->file_size());
        total += (t.nbytes + g->alignment() - 1) / g->alignment() * g->alignment();
    }
    CHECK(g->data_offset() + total >= g->file_size() - g->alignment());

    // a few tensors: type + shape (ne[0] fastest)
    auto expect = [&](const char* name, int type, int nd, int64_t n0, int64_t n1, int64_t n2) {
        const GgufTensorInfo* t = g->tensor(name);
        if (!t) { std::printf("FAIL missing tensor %s\n", name); ++g_fail; return; }
        const bool ok = t->type == type && t->n_dims == nd && t->ne[0] == n0 && (nd < 2 || t->ne[1] == n1) && (nd < 3 || t->ne[2] == n2);
        if (!ok) std::printf("FAIL tensor %s: type %d dims %d [%lld,%lld,%lld]\n", name, t->type, t->n_dims, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
        CHECK(ok);
    };
    expect("token_embd.weight", T_Q8_0, 2, 2304, 201024, 0);
    expect("output.weight", T_Q8_0, 2, 2304, 201024, 0);
    expect("token_embd_norm.weight", T_F32, 1, 2304, 0, 0);
    expect("blk.0.attn_q.weight", T_Q8_0, 2, 2304, 3072, 0);
    expect("blk.0.attn_k.weight", T_Q8_0, 2, 2304, 1024, 0);
    expect("blk.0.attn_output.weight", T_Q8_0, 2, 3072, 2304, 0);
    expect("blk.0.attn_sink_proj.weight", T_BF16, 2, 2304, 24, 0);
    expect("blk.0.attn_rel_bias.weight", T_F32, 2, 16, 1024, 0);
    expect("blk.0.attn_k_conv.weight", T_F32, 2, 4, 1024, 0);
    expect("blk.0.ffn_gate_up_exps.weight", T_NVFP4, 3, 2304, 2304, 128);
    expect("blk.0.ffn_down_exps.weight", T_NVFP4, 3, 1152, 2304, 128);
    expect("blk.0.ffn_gate_up_exps.scale", T_F32, 1, 128, 0, 0);
    expect("blk.19.ffn_down_exps.scale", T_F32, 1, 128, 0, 0);
    expect("blk.20.ffn_gate_up_exps.weight", T_Q8_0, 3, 2304, 2304, 128);
    expect("blk.23.ffn_down_exps.weight", T_Q8_0, 3, 1152, 2304, 128);
    expect("blk.5.attn_k_conv_proj.weight", T_BF16, 2, 2304, 32, 0);
    expect("blk.5.attn_res_query.weight", T_F32, 2, 2304, 3, 0);
    expect("blk.6.loop_q_ctrl_down.weight", T_F32, 2, 4608, 64, 0);
    expect("blk.17.loop_ffn_ctrl_gain.weight", T_F32, 2, 64, 2304, 0);
    expect("blk.6.loop_msg_query.weight", T_F32, 2, 2304, 4, 0);
    CHECK(g->tensor("blk.0.attn_res_query.weight") == nullptr);   // layer 0 has no attention-residual query
    CHECK(g->tensor("blk.18.loop_msg_query.weight") == nullptr);  // only the core layers 6..17 have loop tensors
    CHECK(g->tensor("blk.20.ffn_gate_up_exps.scale") == nullptr); // no per-expert scale on the Q8_0 expert layers
    const GgufTensorInfo* eg = g->tensor("blk.0.ffn_gate_up_exps.weight");
    if (eg) CHECK(eg->nbytes == (size_t)128 * 2304 * (2304 / 64) * 36);

    // KV lookup incl. arrays
    std::string s;
    CHECK(g->get_string("general.architecture", &s) && s == "inkling2_loop");
    uint32_t u = 0;
    CHECK(g->get_u32("inkling2_loop.block_count", &u) && u == 24);
    CHECK(g->get_u32("inkling2_loop.expert_count", &u) && u == 128);
    CHECK(g->get_u32("inkling2_loop.vocab_size_unpadded", &u) && u == 200058);
    float f = 0;
    CHECK(g->get_f32("inkling2_loop.expert_weights_scale", &f) && f == 8.0f);
    CHECK(g->get_f32("inkling2_loop.attention.qk_clip", &f) && std::abs(f - 22.627417f) < 1e-4f);
    std::vector<bool> pat;
    CHECK(g->get_arr_bool("inkling2_loop.attention.sliding_window_pattern", &pat) && pat.size() == 24);
    for (int l = 0; l < 24 && pat.size() == 24; ++l) CHECK(pat[l] == !(l % 6 == 5));
    std::vector<int64_t> hk;
    CHECK(g->get_arr_i64("inkling2_loop.attention.head_count_kv", &hk) && hk.size() == 24 && hk[0] == 8);
    CHECK(g->arr_len("tokenizer.ggml.tokens") == 200058);
    CHECK(g->arr_len("tokenizer.ggml.merges") == 424413);
    bool b = true;
    CHECK(g->get_bool("inkling2_loop.loop.varying", &b) && !b);
    CHECK(!g->get_u32("inkling2_loop.no_such_key", &u));
    CHECK(g->kv("no.such.key") == nullptr);

    // ModelConfig + visit plan + weights
    ModelConfig c;
    CHECK(ModelConfig::from_gguf(*g, &c, &err));
    if (!err.empty()) std::printf("config error: %s\n", err.c_str());
    CHECK(c.n_layer == 24 && c.n_embd == 2304 && c.n_ff_exp == 1152 && c.n_head == 24 && c.n_head_kv == 8 && c.head_dim == 128);
    CHECK(c.n_expert == 128 && c.n_expert_used == 8 && c.expert_weights_scale == 8.0f);
    CHECK(c.loop_start == 6 && c.loop_end == 18 && c.loop_count == 3 && c.loop_gate_rank == 64);
    CHECK(c.n_vocab_padded == 201024 && c.n_vocab == 200058 && c.sliding_window == 1024 && c.n_conv == 4);
    CHECK(c.situ_beta == 4.0f && c.situ_linear_beta == 25.0f && c.final_logit_softcap == 30.0f);
    CHECK(c.n_rel == 16 && c.rel_extent == 1024 && c.log_scale_floor == 8192 && std::abs(c.log_scale_alpha - 0.1f) < 1e-6f);
    CHECK(c.n_slots() == 48 && c.n_conv_channels() == 6656 && c.attn_res_block_size == 6);

    auto plan = visit_plan(c);
    CHECK(plan.size() == 48);
    if (plan.size() == 48) {
        for (int i = 0; i < 48; ++i) {
            const Visit& v = plan[i];
            int phys = i < 6 ? i : i < 42 ? 6 + (i - 6) % 12 : i - 24;
            int pass = (i >= 6 && i < 42) ? (i - 6) / 12 : 0;
            CHECK(v.index == i && v.kv_slot == i && v.phys == phys && v.pass == pass);
            CHECK(v.is_swa == (phys % 6 != 5));
            CHECK(v.in_core == (i >= 6 && i < 42) && v.has_message == (pass > 0));
            // llama.cpp cache-slot numbering: physical index for prelude/pass 0/postlude, 24 + (pass-1)*12 + (l-6) for extra passes
            int ls = pass == 0 ? phys : 24 + (pass - 1) * 12 + (phys - 6);
            CHECK(v.llama_slot == ls);
        }
        CHECK(plan[18].phys == 6 && plan[18].pass == 1 && plan[18].llama_slot == 24);
        CHECK(plan[41].phys == 17 && plan[41].pass == 2 && plan[41].llama_slot == 47);
        CHECK(plan[42].phys == 18 && plan[47].phys == 23 && plan[47].llama_slot == 23);
    }

    Weights w;
    CHECK(load_weights(*g, c, &w, &err));
    if (!err.empty()) std::printf("weights error: %s\n", err.c_str());
    CHECK(w.layers.size() == 24);
    if (w.layers.size() == 24) {
        CHECK(w.layers[0].ffn_gate_up_exps.type == T_NVFP4 && w.layers[0].ffn_gate_up_scale.valid());
        CHECK(w.layers[20].ffn_gate_up_exps.type == T_Q8_0 && !w.layers[20].ffn_gate_up_scale.valid());
        CHECK(!w.layers[0].attn_res_query.valid() && w.layers[1].attn_res_query.valid() && w.layers[5].attn_res_query.ne[1] == 3);
        CHECK(w.layers[5].attn_k_conv_proj.valid() && !w.layers[4].attn_k_conv_proj.valid());
        CHECK(w.layers[6].in_core && w.layers[6].loop_msg_query.valid() && w.layers[6].ctrl[3].forget.valid());
        CHECK(!w.layers[5].in_core && !w.layers[18].loop_msg_query.valid());
        CHECK(w.layers[0].ffn_gate_up_exps.row_bytes() == (size_t)2304 / 64 * 36);
        // expert addressing: expert e, row r
        const auto& gu = w.layers[0].ffn_gate_up_exps;
        CHECK(gu.row(3, 2) == gu.data + ((size_t)2 * 2304 + 3) * gu.row_bytes());
    }
    // a corrupted path must fail cleanly
    std::string e2;
    CHECK(GgufFile::open("/nonexistent/file.gguf", &e2) == nullptr && !e2.empty());

    std::printf(g_fail ? "test_gguf: %d FAILURES\n" : "test_gguf: all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}

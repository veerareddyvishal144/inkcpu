// Cross-check of the semantics of kernels_misc against REAL ggml graph execution (build-sys libggml-cpu/base).  Agent B2.
// The graph snippets below are copied from src/models/inkling2.cpp (clip, mix, control, convolve) and src/llama-graph.cpp
// (soft_max with mask and sinks); they run on random data in ggml's CPU backend and are compared with ink:: kernels.
// Not part of the default build:
//   g++ -O2 -std=c++17 -march=skylake-avx512 -Iinclude -I/data/inkling-mini/llama.cpp/ggml/include tests/kernels_misc/xcheck_ggml_graph.cpp
//       build-b2s/kernels_misc/libink_kernels_misc.a -L/data/inkling-mini/llama.cpp/build-sys/bin -lggml-cpu -lggml-base
//       -Wl,-rpath,/data/inkling-mini/llama.cpp/build-sys/bin -o build-b2s/xcheck_ggml_graph
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "ggml-cpu.h"
#include "ggml.h"
#include "ink/kernels_misc.h"

static std::mt19937_64 rng(5);
static float nrand() { return std::normal_distribution<float>(0.0f, 1.0f)(rng); }
static int g_fail = 0;
static void report(const char* name, const float* a, const float* b, size_t n, double tol) {
    double mx = 0, mr = 0;
    for (size_t i = 0; i < n; ++i) { mx = std::max(mx, (double)std::fabs((double)a[i] - b[i])); mr = std::max(mr, (double)std::fabs(b[i])); }
    const double e = mx / std::max(mr, 1e-30);
    std::printf("  %-34s max|diff|/max|ggml| = %.3e  %s\n", name, e, e <= tol ? "ok" : "MISMATCH");
    if (e > tol) ++g_fail;
}
static ggml_tensor* rnd(ggml_context* c, int64_t a, int64_t b = 1, int64_t d = 1, float s = 1.0f) {
    ggml_tensor* t = ggml_new_tensor_3d(c, GGML_TYPE_F32, a, b, d);
    for (int64_t i = 0; i < ggml_nelements(t); ++i) ((float*)t->data)[i] = nrand() * s;
    return t;
}
static void run(ggml_context* c, ggml_tensor* out) {
    ggml_cgraph* gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(c, gf, 4);
}

int main() {
    ggml_init_params ip{(size_t)1 << 30, nullptr, false};
    ggml_context* ctx0 = ggml_init(ip);
    const int64_t n_embd = 2304, n_tokens = 5;

    // ---- clip(): Q-like [128, 24, T] with per-channel scale, and R-like [16, 24, T]
    for (int variant = 0; variant < 2; ++variant) {
        const int64_t dim = variant ? 16 : 128, nh = 24;
        const float thr = variant ? 8.0f : 22.627f;
        ggml_tensor* cur = rnd(ctx0, dim, nh, n_tokens, 6.0f);
        ggml_tensor* scale = rnd(ctx0, dim, nh, 1, 1.0f);
        for (int64_t i = 0; i < ggml_nelements(scale); ++i) ((float*)scale->data)[i] = 0.5f + std::fabs(((float*)scale->data)[i]);
        ggml_tensor* in = cur;
        // verbatim from inkling2::graph::clip
        cur = ggml_mul(ctx0, cur, scale);
        ggml_tensor* norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
        norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);
        ggml_tensor* factor = ggml_div(ctx0, ggml_fill(ctx0, norm, thr), norm);
        cur = ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
        run(ctx0, cur);
        std::vector<float> mine((float*)in->data, (float*)in->data + ggml_nelements(in));
        for (int64_t t = 0; t < n_tokens; ++t) ink::l2clip_heads(mine.data() + t * nh * dim, (float*)scale->data, (int)nh, (int)dim, thr);
        report(variant ? "clip R (16 x 24, thr 8)" : "clip Q (128 x 24, scale, thr 22.627)", mine.data(), (float*)cur->data, mine.size(), 2e-6);
    }

    // ---- mix(): depth attention over 3 earlier blocks + cur with 3 queries
    {
        const int nblk = 3, nq = 3;
        std::vector<ggml_tensor*> blocks;
        for (int b = 0; b < nblk; ++b) blocks.push_back(rnd(ctx0, n_embd, n_tokens, 1, std::pow(3.0f, (float)b)));
        ggml_tensor* curt = rnd(ctx0, n_embd, n_tokens, 1, 2.0f);
        ggml_tensor* query = rnd(ctx0, n_embd, nq, 1, 1.0f);
        // verbatim from inkling2::graph::mix
        ggml_tensor* states = ggml_reshape_3d(ctx0, curt, n_embd, 1, n_tokens);
        for (ggml_tensor* block : blocks) states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
        states = ggml_rms_norm(ctx0, states, 1e-6f);
        ggml_tensor* queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));
        ggml_tensor* scores = ggml_mul_mat(ctx0, states, queries);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f / sqrtf((float)n_embd), 0.0f);
        ggml_tensor* out = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
        run(ctx0, out);  // out: [n_embd, nq, n_tokens]
        std::vector<float> mine(n_embd * nq * n_tokens);
        for (int64_t t = 0; t < n_tokens; ++t) {
            std::vector<const float*> st;
            for (auto* b : blocks) st.push_back((float*)b->data + t * n_embd);
            st.push_back((float*)curt->data + t * n_embd);
            std::vector<float*> outs(nq);
            for (int q = 0; q < nq; ++q) outs[q] = mine.data() + (t * nq + q) * n_embd;
            ink::depth_mix(st.data(), (int)st.size(), (float*)query->data, nq, outs.data(), n_embd, 1.0f / sqrtf((float)n_embd));
        }
        report("depth mix (3 blocks + cur, 3 queries)", mine.data(), (float*)out->data, mine.size(), 2e-6);
    }

    // ---- control(): rank-64 gates, F32 weights
    {
        const int64_t rank = 64;
        ggml_tensor* cur = rnd(ctx0, n_embd, n_tokens), *msg = rnd(ctx0, n_embd, n_tokens, 1, 1.5f);
        ggml_tensor* down = rnd(ctx0, 2 * n_embd, rank, 1, 0.02f), *gain = rnd(ctx0, rank, n_embd, 1, 0.3f), *forget = rnd(ctx0, rank, n_embd, 1, 0.3f);
        // verbatim from inkling2::graph::control
        ggml_tensor* gate = ggml_concat(ctx0, ggml_rms_norm(ctx0, cur, 1e-6f), ggml_rms_norm(ctx0, msg, 1e-6f), 0);
        gate = ggml_mul_mat(ctx0, down, gate);
        ggml_tensor* g = ggml_tanh(ctx0, ggml_mul_mat(ctx0, gain, gate));
        ggml_tensor* f = ggml_tanh(ctx0, ggml_mul_mat(ctx0, forget, gate));
        ggml_tensor* out = ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, f, cur)), ggml_mul(ctx0, g, msg));
        run(ctx0, out);
        std::vector<float> mine(n_embd * n_tokens), gi(2 * n_embd), gate64(rank), gp(n_embd), fp(n_embd);
        for (int64_t t = 0; t < n_tokens; ++t) {
            ink::ctrl_gate_input((float*)cur->data + t * n_embd, (float*)msg->data + t * n_embd, gi.data(), n_embd);
            for (int64_t r = 0; r < rank; ++r) { double s = 0; for (int64_t i = 0; i < 2 * n_embd; ++i) s += (double)((float*)down->data)[r * 2 * n_embd + i] * gi[i]; gate64[r] = (float)s; }
            for (int64_t e = 0; e < n_embd; ++e) {
                double sg = 0, sf = 0;
                for (int64_t r = 0; r < rank; ++r) { sg += (double)((float*)gain->data)[e * rank + r] * gate64[r]; sf += (double)((float*)forget->data)[e * rank + r] * gate64[r]; }
                gp[e] = (float)sg; fp[e] = (float)sf;
            }
            ink::ctrl_apply((float*)cur->data + t * n_embd, (float*)msg->data + t * n_embd, gp.data(), fp.data(), mine.data() + t * n_embd, n_embd);
        }
        report("controller (gate_input+matvecs+apply)", mine.data(), (float*)out->data, mine.size(), 2e-6);
    }

    // ---- convolve(): static (K-like, C=1024) and dynamic (global-layer K, 8 heads x 128)
    for (int dyn = 0; dyn < 2; ++dyn) {
        const int64_t C = 1024, nconv = 4, nheads = 8;
        ggml_tensor* cur = rnd(ctx0, C, n_tokens), *weight = rnd(ctx0, nconv, C, 1, 0.6f), *history = rnd(ctx0, nconv - 1, C, 1);
        ggml_tensor* dynamic = dyn ? rnd(ctx0, nconv * nheads, n_tokens, 1, 0.5f) : nullptr;
        const bool is_dyn = dynamic != nullptr;
        const int64_t n_seq_tokens = n_tokens, n_seqs = 1;
        // verbatim from inkling2::graph::convolve (state store dropped, history is the whole tensor)
        ggml_tensor* past = history;
        ggml_tensor* x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, C, n_seq_tokens, n_seqs));
        x = ggml_concat(ctx0, past, x, 0);
        ggml_tensor* out = ggml_ssm_conv(ctx0, x, weight);
        out = ggml_reshape_2d(ctx0, out, C, n_tokens);
        if (dynamic) {
            const int64_t n_heads = dynamic->ne[0] / nconv;
            dynamic = ggml_reshape_3d(ctx0, dynamic, nconv, n_heads, n_tokens);
            for (int64_t lag = 0; lag < nconv; ++lag) {
                ggml_tensor* tap = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag * dynamic->nb[0]);
                ggml_tensor* window = ggml_view_3d(ctx0, x, n_seq_tokens, C, n_seqs, x->nb[1], x->nb[2], (nconv - 1 - lag) * x->nb[0]);
                window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
                window = ggml_reshape_3d(ctx0, window, C / n_heads, n_heads, n_tokens);
                ggml_tensor* term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), C, n_tokens);
                out = ggml_add(ctx0, out, term);
            }
        }
        out = ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f / sqrtf((is_dyn ? 2 : 1) * nconv + 1));
        run(ctx0, out);
        // mine: SoA
        std::vector<float> wsoa(4 * C), hs(3 * C), y(C * n_tokens);
        ink::sconv_repack_static((float*)weight->data, wsoa.data(), C);
        for (int64_t c = 0; c < C; ++c) for (int i = 0; i < 3; ++i) hs[i * C + c] = ((float*)history->data)[c * 3 + i];
        ink::sconv_seq((float*)cur->data, hs.data(), wsoa.data(), y.data(), n_tokens, C, 0, C, is_dyn ? (float*)dynamic->data : nullptr, 128);
        report(is_dyn ? "short conv, dynamic taps (K/V global)" : "short conv, static taps", y.data(), (float*)out->data, y.size(), 2e-6);
    }

    // ---- attention scores: GQA mapping of mul_mat, then soft_max_ext(scale, mask = bias) with sinks
    {
        const int64_t n_kv = 300, nh = 24, nkv = 8, hd = 128;
        ggml_tensor* k = rnd(ctx0, hd, n_kv, nkv, 0.8f), *q = rnd(ctx0, hd, 1, nh);
        ggml_tensor* bias = rnd(ctx0, n_kv, 1, nh, 2.0f), *sinks = rnd(ctx0, nh, 1, 1, 1.0f);
        ggml_tensor* kq = ggml_mul_mat(ctx0, k, q);
        ggml_tensor* sm = ggml_soft_max_ext(ctx0, kq, bias, 1.0f / sqrtf((float)hd), 0.0f);
        ggml_soft_max_add_sinks(sm, sinks);
        run(ctx0, sm);
        std::vector<float> mine(n_kv * nh);
        for (int64_t h = 0; h < nh; ++h) {
            std::vector<double> s(n_kv);
            double m = ((float*)sinks->data)[h];
            for (int64_t j = 0; j < n_kv; ++j) {
                double dot = 0;
                for (int64_t d = 0; d < hd; ++d) dot += (double)((float*)k->data)[(h / 3) * n_kv * hd + j * hd + d] * ((float*)q->data)[h * hd + d];
                s[j] = dot / std::sqrt((double)hd) + ((float*)bias->data)[h * n_kv + j];  // bias added AFTER the scale
                m = std::max(m, s[j]);
            }
            double den = std::exp(((float*)sinks->data)[h] - m);  // sink logit unscaled, in the denominator only
            for (int64_t j = 0; j < n_kv; ++j) den += std::exp(s[j] - m);
            for (int64_t j = 0; j < n_kv; ++j) mine[h * n_kv + j] = (float)(std::exp(s[j] - m) / den);
        }
        report("attn softmax: GQA h/3, bias after scale, sink", mine.data(), (float*)sm->data, mine.size(), 2e-6);
    }
    // ---- router (build_moe_ffn: sigmoid, selection bias, argsort_top_k, get_rows, normalise, route_scale), SiTU, final softcap
    {
        const int64_t E = 128, K = 8, T = 40;
        ggml_tensor* logits = rnd(ctx0, E, T, 1, 2.5f), *eb = rnd(ctx0, E, 1, 1, 0.1f);
        ggml_tensor* probs = ggml_sigmoid(ctx0, logits);
        ggml_tensor* selp = ggml_add(ctx0, probs, eb);
        ggml_tensor* ids = ggml_argsort_top_k(ctx0, selp, K);
        ggml_tensor* weights = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, E, T), ids);
        weights = ggml_reshape_2d(ctx0, weights, K, T);
        ggml_tensor* wsum = ggml_scale_bias(ctx0, ggml_sum_rows(ctx0, weights), 1.0f, 1e-20f);
        weights = ggml_scale(ctx0, ggml_reshape_3d(ctx0, ggml_div(ctx0, weights, wsum), 1, K, T), 8.0f);
        ggml_cgraph* gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, weights);
        ggml_build_forward_expand(gf, ids);
        ggml_graph_compute_with_ctx(ctx0, gf, 4);
        std::vector<float> wm(K * T);
        int bad_ids = 0;
        for (int64_t t = 0; t < T; ++t) {
            int mid[8];
            ink::router_topk((float*)logits->data + t * E, (float*)eb->data, (int)E, (int)K, 8.0f, 1e-20f, mid, wm.data() + t * K);
            // ids is a VIEW of the full argsort result: rows are nb[1] bytes apart (128 ints), not K ints
            for (int i = 0; i < K; ++i) bad_ids += mid[i] != *(int32_t*)((char*)ids->data + t * ids->nb[1] + i * ids->nb[0]);
        }
        report("router weights (ids differ: see below)", wm.data(), (float*)weights->data, wm.size(), 2e-6);
        std::printf("  router ids differing from ggml argsort_top_k: %d of %lld\n", bad_ids, (long long)(K * T));
        if (bad_ids) ++g_fail;
        // SiTU and softcap
        const int64_t n = 4096;
        ggml_tensor* g = rnd(ctx0, n, 1, 1, 6.0f), *u = rnd(ctx0, n, 1, 1, 30.0f), *x = rnd(ctx0, 201024 / 8, 1, 1, 40.0f);
        ggml_tensor* act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, g, 1.0f / 4.0f)), 4.0f);
        act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, g));
        ggml_tensor* upv = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, u, 1.0f / 25.0f)), 25.0f);
        ggml_tensor* si = ggml_mul(ctx0, act, upv);
        ggml_tensor* cap = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, x, 1.0f / 30.0f)), 30.0f);
        ggml_cgraph* gf2 = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf2, si);
        ggml_build_forward_expand(gf2, cap);
        ggml_graph_compute_with_ctx(ctx0, gf2, 4);
        std::vector<float> sm(n), cm(ggml_nelements(x));
        ink::vsitu((float*)g->data, (float*)u->data, sm.data(), n);
        ink::vsoftcap((float*)x->data, cm.data(), cm.size(), 30.0f);
        report("SiTU(4,25) vs ggml op chain", sm.data(), (float*)si->data, sm.size(), 2e-6);
        report("final softcap 30 vs ggml op chain", cm.data(), (float*)cap->data, cm.size(), 2e-6);
    }
    // ---- whole attention block of build_layer for T=3 tokens (rel-bias table, gather by distance, sink, mask, GQA, V sum, gate) vs
    //      ink::kv_append_n + attn_prefill_group.  K/V are fp16-rounded so the fp16 cache holds exactly the same values.
    for (int swa = 1; swa >= 0; --swa) {
        const int64_t T = 3, pos0 = 3000, n_kv = pos0 + T, nh = 24, nkv = 8, hd = 128, nrel = 16, extent = 1024;
        ggml_tensor* Q = rnd(ctx0, hd, nh, T), *R = rnd(ctx0, nrel, nh, T), *relb = rnd(ctx0, nrel, extent, 1, 0.5f);
        ggml_tensor* sinks = rnd(ctx0, nh, T), *gate = rnd(ctx0, nh * hd, T, 1, 2.0f);
        ggml_tensor* Kt = rnd(ctx0, hd, n_kv, nkv, 0.8f), *Vt = rnd(ctx0, n_kv, hd, nkv);  // Vt is V transposed per head, as in the non-FA path
        for (ggml_tensor* t : {Kt, Vt}) for (int64_t i = 0; i < ggml_nelements(t); ++i) ((float*)t->data)[i] = ink::fp16_to_f32(ink::f32_to_fp16(((float*)t->data)[i]));
        ggml_tensor* tau = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, T);
        for (int64_t t = 0; t < T; ++t) ((float*)tau->data)[t] = swa ? 1.0f : 1.1f + 0.013f * t;
        ggml_tensor* rel_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, T);
        ggml_tensor* kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, T);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < n_kv; ++j) {
                const int64_t dist = (pos0 + t) - j;  // llama_kv_cache::set_input_relative_pos
                ((int32_t*)rel_pos->data)[t * n_kv + j] = dist >= 0 && dist < extent ? (int32_t)dist : (int32_t)extent;
                const bool masked = dist < 0 || (swa && dist >= 1024);  // causal mask / sliding window mask
                ((float*)kq_mask->data)[t * n_kv + j] = masked ? -INFINITY : 0.0f;
            }
        ggml_tensor* Qs = Q, *Rs = R;
        if (!swa) { Qs = ggml_mul(ctx0, Q, tau); Rs = ggml_mul(ctx0, R, tau); }  // tau after the clip, on global layers only
        // verbatim from inkling2::graph::build_layer + build_attn (rel variant) + build_attn_mha (non-flash path)
        ggml_tensor* table = ggml_mul_mat(ctx0, ggml_pad(ctx0, relb, 0, 1, 0, 0), Rs);
        table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, nh, T));
        table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));
        ggml_tensor* kq_b = ggml_get_rows(ctx0, table, rel_pos);
        kq_b = ggml_reshape_4d(ctx0, kq_b, nh, n_kv, T, 1);
        kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
        ggml_tensor* zero_sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, nh), 0.0f);
        ggml_tensor* mask2 = ggml_add(ctx0, kq_b, kq_mask);
        ggml_tensor* q = ggml_permute(ctx0, Qs, 0, 2, 1, 3);
        ggml_tensor* kq = ggml_mul_mat(ctx0, Kt, q);
        kq = ggml_soft_max_ext(ctx0, kq, mask2, 1.0f / sqrtf((float)hd), 0.0f);
        ggml_soft_max_add_sinks(kq, zero_sinks);
        ggml_tensor* kqv = ggml_mul_mat(ctx0, Vt, kq);
        ggml_tensor* cur = ggml_cont_2d(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3), nh * hd, T);
        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        run(ctx0, cur);
        // mine
        const uint32_t nblk = swa ? 32 : 64;
        ink::KvLayer L;
        L.ring = swa != 0; L.n_blocks = nblk;
        L.k = (uint16_t*)aligned_alloc(64, L.bytes_per_tensor()); L.v = (uint16_t*)aligned_alloc(64, L.bytes_per_tensor());
        std::memset(L.k, 0, L.bytes_per_tensor()); std::memset(L.v, 0, L.bytes_per_tensor());
        std::vector<float> Kf(n_kv * 1024), Vf(n_kv * 1024);
        for (int64_t j = 0; j < n_kv; ++j)
            for (int64_t g = 0; g < nkv; ++g)
                for (int64_t d = 0; d < hd; ++d) {
                    Kf[j * 1024 + g * 128 + d] = ((float*)Kt->data)[d + hd * j + hd * n_kv * g];
                    Vf[j * 1024 + g * 128 + d] = ((float*)Vt->data)[j + n_kv * d + n_kv * hd * g];
                }
        ink::kv_append_n(L, 0, (int)n_kv, Kf.data(), Vf.data());
        std::vector<float> table_mine(16 * ink::kRelTableLen), rb(1024 * 16), out(T * 3072), tauv(T);
        std::memcpy(rb.data(), relb->data, rb.size() * 4);  // GGUF layout [1024][16] == ggml tensor {16, 1024}
        ink::attn_repack_rel_bias(rb.data(), table_mine.data());
        for (int64_t t = 0; t < T; ++t) tauv[t] = ((float*)tau->data)[t];
        ink::AttnPrefill P{(float*)Q->data, (float*)R->data, (float*)sinks->data, (float*)gate->data, swa ? nullptr : tauv.data(), pos0, (int)T};
        for (int g = 0; g < 8; ++g) ink::attn_prefill_group(L, table_mine.data(), P, g, 0, (int)T, out.data());
        report(swa ? "attention block, SWA ring (pos 3000..3002)" : "attention block, global, tau (pos 3000..3002)", out.data(), (float*)cur->data, out.size(), 1e-5);
        std::free(L.k); std::free(L.v);
    }
    ggml_free(ctx0);
    std::printf(g_fail ? "XCHECK FAILED (%d)\n" : "XCHECK OK\n", g_fail);
    return g_fail ? 1 : 0;
}

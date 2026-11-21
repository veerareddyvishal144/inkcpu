// Debug tool (agent A): run llama.cpp on a token file as ONE batch and dump selected intermediate tensors through the
// ggml eval callback (l_out, attn_out, ffn_inp, ffn_out, Qcur, Kcur, Vcur, Rcur, ffn_moe_topk, result_norm), one raw file per
// tensor: DIR/<name>.bin (+ DIR/manifest.txt: name type ne0 ne1 ne2 ne3 nb0 nb1 nb2 nb3). Used by compare_acts to localise
// reference-vs-llama.cpp differences layer by layer. Links the stock build-sys libs (never rebuilds them).
//   dump_llama_acts -m GGUF --tokens-file ids.txt --out DIR [--threads N] [--override-kv key=int:val]
#include "oracle_common.h"

#include "ggml-backend.h"

struct Cap {
    std::string out;
    FILE*       manifest = nullptr;
    int         n_dumped = 0;
};

static bool wanted(const char* nm) {
    static const char* pre[] = {"l_out", "attn_out", "ffn_inp", "ffn_out", "Qcur", "Kcur", "Vcur", "Rcur", "ffn_moe_topk", "result_norm", "ffn_moe_logits", "ffn_moe_weights_scaled", "ffn_moe_gate_up_scaled", "ffn_moe_situ", "ffn_moe_down_scaled", "ffn_moe_out", "norm", nullptr};
    for (int i = 0; pre[i]; ++i) {
        const size_t n = strlen(pre[i]);
        if (strncmp(nm, pre[i], n) == 0 && (nm[n] == '-' || nm[n] == 0)) return true;
    }
    return false;
}

static bool eval_cb(struct ggml_tensor* t, bool ask, void* ud) {
    Cap* c = (Cap*) ud;
    if (!wanted(t->name)) return ask ? false : true;
    if (ask) return true;
    const size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);
    oc::write_file(c->out + "/" + t->name + ".bin", buf.data(), nb);
    fprintf(c->manifest, "%s %d %lld %lld %lld %lld %zu %zu %zu %zu\n", t->name, (int) t->type, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
            (long long) t->ne[3], t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    ++c->n_dumped;
    return true;
}

int main(int argc, char** argv) {
    oc::Args a;
    if (!oc::parse_args(argc, argv, a, false)) {
        fprintf(stderr, "usage: dump_llama_acts -m GGUF --tokens-file ids.txt --out DIR [--threads N] [--override-kv k=int:v]\n");
        return 2;
    }
    llama_model* model = oc::load_model(a);
    if (!model) return 1;
    std::vector<llama_token> toks = oc::read_tokens_file(a.tokens_file);
    const int n = (int) toks.size();
    oc::mkdir_p(a.out);
    Cap cap;
    cap.out = a.out;
    cap.manifest = fopen((a.out + "/manifest.txt").c_str(), "w");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n;
    cp.n_batch = n;
    cp.n_ubatch = n;
    cp.n_threads = a.threads;
    cp.n_threads_batch = a.threads;
    cp.no_perf = true;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &cap;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = toks[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = i == n - 1;
    }
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    fclose(cap.manifest);
    fprintf(stderr, "dump_llama_acts: decode rc=%d, %d tensors dumped to %s\n", rc, cap.n_dumped, a.out.c_str());
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}

// oracle_logits: run llama.cpp on a prompt (one batch) and dump the logits. Owner: agent E.
//
//   oracle_logits -m GGUF (--tokens-file ids.txt | --prompt "text") [--ctx N] [--override-kv key=int:val ...]
//                 [--threads N] [--all-logits] --out DIR
//
// Writes DIR/tokens.i32, DIR/logits_last.f32, DIR/top20.txt, DIR/meta.json, DIR/prompt.txt (if --prompt),
// DIR/logits_all.f32 (if --all-logits). File formats: tools/README.md.
#include "oracle_common.h"

#include <chrono>

static void usage() {
    fprintf(stderr,
            "usage: oracle_logits -m GGUF (--tokens-file ids.txt | --prompt \"text\") [--ctx N]\n"
            "                     [--override-kv key=int:val ...] [--threads N] [--all-logits] --out DIR\n");
}

int main(int argc, char ** argv) {
    oc::Args a;
    if (!oc::parse_args(argc, argv, a, false)) {
        usage();
        return 2;
    }
    llama_model * model = oc::load_model(a);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks = a.tokens_file.empty() ? oc::tokenize(vocab, a.prompt) : oc::read_tokens_file(a.tokens_file);
    if (toks.empty()) {
        fprintf(stderr, "empty token sequence\n");
        return 1;
    }
    const int n = (int) toks.size();
    const int n_ctx = a.ctx > 0 ? a.ctx : n;
    if (n_ctx < n) {
        fprintf(stderr, "--ctx %d is smaller than the %d tokens\n", n_ctx, n);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n;    // the whole prompt is ONE ubatch: the logits come from a single forward pass
    cp.n_ubatch = n;
    cp.n_threads = a.threads;
    cp.n_threads_batch = a.threads;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = toks[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (a.all_logits || i == n - 1) ? 1 : 0;
    }
    b.n_tokens = n;

    const auto t0 = std::chrono::steady_clock::now();
    const int rc = llama_decode(ctx, b);
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed: %d\n", rc);
        return 1;
    }
    fprintf(stderr, "oracle_logits: %d tokens in %.2f s (%.1f tok/s, %d threads), n_vocab %d\n", n, dt, n / dt, a.threads, n_vocab);

    const float * last = llama_get_logits_ith(ctx, n - 1);
    if (!last) {
        fprintf(stderr, "no logits for the last position\n");
        return 1;
    }

    oc::mkdir_p(a.out);
    oc::write_file(a.out + "/tokens.i32", toks.data(), toks.size() * sizeof(int32_t));
    oc::write_file(a.out + "/logits_last.f32", last, (size_t) n_vocab * sizeof(float));
    if (a.have_prompt) oc::write_file(a.out + "/prompt.txt", a.prompt.data(), a.prompt.size());
    const std::string meta = oc::meta_json(a, vocab, n_vocab, n, n_ctx);
    oc::write_file(a.out + "/meta.json", meta.data(), meta.size());

    if (a.all_logits) {
        FILE * f = fopen((a.out + "/logits_all.f32").c_str(), "wb");
        for (int i = 0; i < n; i++) {
            const float * li = llama_get_logits_ith(ctx, i);
            if (!li || fwrite(li, sizeof(float), n_vocab, f) != (size_t) n_vocab) {
                fprintf(stderr, "failed to write logits of position %d\n", i);
                return 1;
            }
        }
        fclose(f);
    }

    std::vector<int> idx(n_vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(), [&](int x, int y) {
        return last[x] > last[y] || (last[x] == last[y] && x < y);
    });
    FILE * t = fopen((a.out + "/top20.txt").c_str(), "w");
    fprintf(t, "# rank token_id logit piece (escaped)   [logits_last.f32 of position %d, n_vocab %d]\n", n - 1, n_vocab);
    for (int r = 0; r < 20; r++) {
        fprintf(t, "%d %d %.6f %s\n", r, idx[r], last[idx[r]], oc::escape(oc::piece(vocab, idx[r])).c_str());
    }
    fclose(t);

    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

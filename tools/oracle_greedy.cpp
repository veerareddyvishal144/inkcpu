// oracle_greedy: generate N greedy (argmax) tokens from a prompt with llama.cpp. Owner: agent E.
//
//   oracle_greedy -m GGUF (--tokens-file ids.txt | --prompt "text") [-n 64] [--ctx N] [--threads N]
//                 [--override-kv key=int:val ...] --out DIR
//
// The prompt is decoded as ONE batch, then every generated token is fed back as a 1-token batch. EOS is not special:
// exactly N tokens are produced. Writes DIR/prompt_tokens.i32, DIR/gen_tokens.i32, DIR/gen_text.txt, DIR/steps.txt, DIR/meta.json.
#include "oracle_common.h"

#include <chrono>

static void usage() {
    fprintf(stderr,
            "usage: oracle_greedy -m GGUF (--tokens-file ids.txt | --prompt \"text\") [-n 64] [--ctx N]\n"
            "                     [--threads N] [--override-kv key=int:val ...] --out DIR\n");
}

int main(int argc, char ** argv) {
    oc::Args a;
    if (!oc::parse_args(argc, argv, a, true)) {
        usage();
        return 2;
    }
    llama_model * model = oc::load_model(a);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt = a.tokens_file.empty() ? oc::tokenize(vocab, a.prompt) : oc::read_tokens_file(a.tokens_file);
    if (prompt.empty()) {
        fprintf(stderr, "empty token sequence\n");
        return 1;
    }
    const int n_prompt = (int) prompt.size();
    const int n_ctx = a.ctx > 0 ? a.ctx : n_prompt + a.n_gen + 8;
    if (n_ctx < n_prompt + a.n_gen) {
        fprintf(stderr, "--ctx %d is smaller than prompt %d + n %d\n", n_ctx, n_prompt, a.n_gen);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_prompt;
    cp.n_ubatch = n_prompt;
    cp.n_threads = a.threads;
    cp.n_threads_batch = a.threads;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    std::vector<llama_token> gen;
    std::string steps = "# step token_id top1_logit top2_id top2_logit margin piece(escaped)\n";
    std::string text;

    const auto t0 = std::chrono::steady_clock::now();
    int rc = llama_decode(ctx, llama_batch_get_one(prompt.data(), n_prompt));
    const double t_prompt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (rc != 0) {
        fprintf(stderr, "prompt decode failed: %d\n", rc);
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (int s = 0; s < a.n_gen; s++) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (!lg) {
            fprintf(stderr, "no logits at step %d\n", s);
            return 1;
        }
        const int best = oc::argmax(lg, n_vocab);
        int second = best == 0 ? 1 : 0;
        for (int i = 0; i < n_vocab; i++) if (i != best && lg[i] > lg[second]) second = i;
        const std::string p = oc::piece(vocab, best);
        char line[512];
        snprintf(line, sizeof line, "%d %d %.6f %d %.6f %.6f ", s, best, lg[best], second, lg[second], lg[best] - lg[second]);
        steps += line + oc::escape(p) + "\n";
        text += p;
        gen.push_back(best);
        if (s + 1 < a.n_gen) {
            llama_token tok = best;
            rc = llama_decode(ctx, llama_batch_get_one(&tok, 1));
            if (rc != 0) {
                fprintf(stderr, "decode failed at step %d: %d\n", s, rc);
                return 1;
            }
        }
    }
    const double t_gen = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    fprintf(stderr, "oracle_greedy: prompt %d tokens in %.2f s; %d tokens in %.2f s = %.2f tok/s (%d threads)\n", n_prompt, t_prompt,
            a.n_gen, t_gen, a.n_gen / t_gen, a.threads);

    oc::mkdir_p(a.out);
    oc::write_file(a.out + "/prompt_tokens.i32", prompt.data(), prompt.size() * sizeof(int32_t));
    oc::write_file(a.out + "/gen_tokens.i32", gen.data(), gen.size() * sizeof(int32_t));
    oc::write_file(a.out + "/gen_text.txt", text.data(), text.size());
    oc::write_file(a.out + "/steps.txt", steps.data(), steps.size());
    if (a.have_prompt) oc::write_file(a.out + "/prompt.txt", a.prompt.data(), a.prompt.size());
    const std::string meta = oc::meta_json(a, vocab, n_vocab, n_prompt, n_ctx);
    oc::write_file(a.out + "/meta.json", meta.data(), meta.size());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

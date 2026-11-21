// Debug tool (agent A): compare RefModel intermediates with a dump_llama_acts directory, layer by layer, token by token.
//   compare_acts GGUF DUMPDIR tokens.i32
// Prints, per visit (execution order): relative L2 error of l_out (max over tokens), the first token whose error exceeds 1e-3,
// the number of tokens whose selected expert SET differs from llama.cpp, and the first such token.
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ink/gguf.h"
#include "ref_forward.h"
#include "test_util.h"

using namespace ink;

int main(int argc, char** argv) {
    if (argc < 4) { std::printf("usage: compare_acts GGUF DUMPDIR tokens.i32\n"); return 2; }
    std::string err;
    auto g = GgufFile::open(argv[1], &err);
    if (!g) { std::printf("open: %s\n", err.c_str()); return 1; }
    RefOptions opt;
    if (const char* e = std::getenv("INK_REF_AQ")) opt.act_quant_like_ggml = std::atoi(e) != 0;
    if (const char* e = std::getenv("INK_REF_KV16")) opt.kv_f16 = std::atoi(e);
    if (const char* e = std::getenv("INK_REF_Q16")) opt.q_f16 = std::atoi(e);
    if (const char* e = std::getenv("INK_REF_BIAS16")) opt.bias_f16 = std::atoi(e);
    RefModel m(*g, opt);
    const int D = m.config().n_embd;
    std::vector<int32_t> toks;
    tutil::read_raw(argv[3], &toks);
    const int T = (int)toks.size();

    // capture the reference traces: [name-il][token] -> vector<float>
    std::map<std::string, std::vector<std::vector<float>>> cap;
    int cur_tok = 0;
    m.set_trace([&](const char* name, int il, const float* d, size_t n) {
        std::string k = std::string(name) + "-" + std::to_string(il);
        auto& v = cap[k];
        if ((int)v.size() <= cur_tok) v.resize(cur_tok + 1);
        v[cur_tok].assign(d, d + n);
    });
    RefState st = m.new_state();
    std::vector<float> logits(m.config().n_vocab);
    for (cur_tok = 0; cur_tok < T; ++cur_tok) m.forward(st, toks[cur_tok], nullptr);

    std::string dir = argv[2];
    int bad_sets_total = 0;
    std::printf("%-6s %-5s %-4s %-12s %-10s %-12s %-12s\n", "visit", "phys", "il", "l_out relL2", "first>1e-3", "setMismatch", "firstMismatch");
    for (const Visit& v : m.plan()) {
        char fn[256];
        std::snprintf(fn, sizeof fn, "%s/l_out-%d.bin", dir.c_str(), v.llama_slot);
        std::vector<float> ll;
        if (!tutil::read_raw(fn, &ll) || ll.size() != (size_t)T * D) { std::printf("%-6d missing/incorrect %s (%zu floats)\n", v.index, fn, ll.size()); continue; }
        auto& mine = cap["l_out-" + std::to_string(v.llama_slot)];
        double worst = 0;
        int first = -1;
        for (int t = 0; t < T; ++t) {
            double nd = 0, nb = 0;
            for (int i = 0; i < D; ++i) { double d = mine[t][i] - ll[(size_t)t * D + i]; nd += d * d; nb += (double)ll[(size_t)t * D + i] * ll[(size_t)t * D + i]; }
            double rel = std::sqrt(nd / nb);
            worst = std::max(worst, rel);
            if (first < 0 && rel > 1e-3) first = t;
        }
        // expert sets
        std::snprintf(fn, sizeof fn, "%s/ffn_moe_topk-%d.bin", dir.c_str(), v.llama_slot);
        std::vector<int32_t> ids;
        int mism = 0, firstm = -1;
        if (tutil::read_raw(fn, &ids)) {
            // layout [n_expert_used][n_tokens] view with row stride n_expert (128 ints): read via stride 128
            const int U = m.config().n_expert_used, E = m.config().n_expert;
            auto& mt = cap["ffn_moe_topk-" + std::to_string(v.llama_slot)];
            for (int t = 0; t < T; ++t) {
                std::set<int> a, b;
                for (int u = 0; u < U; ++u) {
                    a.insert((int)mt[t][u]);
                    b.insert(ids[(size_t)t * E + u]);
                }
                if (a != b) { ++mism; if (firstm < 0) firstm = t; }
            }
        }
        bad_sets_total += mism;
        std::printf("%-6d %-5d %-4d %-12.3g %-10d %-12d %-12d\n", v.index, v.phys, v.llama_slot, worst, first, mism, firstm);
    }
    std::printf("total token/visit expert-set mismatches: %d\n", bad_sets_total);
    if (argc > 4) {   // detail for one llama slot: relative L2 of every captured tensor, per token
        const int il = std::atoi(argv[4]);
        const char* names[] = {"Qcur", "Kcur", "Vcur", "Rcur", "ffn_inp", "norm", "ffn_moe_logits", "ffn_moe_gate_up_scaled", "ffn_moe_situ", "ffn_moe_down_scaled", "ffn_out", "l_out"};
        std::printf("detail il=%d: relL2(ref vs llama.cpp) per token\n%-4s", il, "tok");
        for (auto n : names) std::printf(" %-10.10s", n + (strncmp(n, "ffn_moe_", 8) == 0 ? 8 : 0));
        std::printf("\n");
        for (int t = 0; t < T; ++t) {
            std::printf("%-4d", t);
            for (auto n : names) {
                char fn[256];
                std::snprintf(fn, sizeof fn, "%s/%s-%d.bin", dir.c_str(), n, il);
                std::vector<float> ll;
                auto& mine = cap[std::string(n) + "-" + std::to_string(il)];
                if (!tutil::read_raw(fn, &ll) || mine.size() <= (size_t)t) { std::printf(" %-10s", "n/a"); continue; }
                const size_t len = mine[t].size();
                double nd = 0, nb = 0, mx = 0;
                for (size_t i = 0; i < len; ++i) {
                    double d = mine[t][i] - ll[(size_t)t * len + i];
                    nd += d * d; nb += (double)ll[(size_t)t * len + i] * ll[(size_t)t * len + i];
                    mx = std::max(mx, std::fabs(d));
                }
                std::printf(" %-10.3g", std::sqrt(nd / nb));
            }
            std::printf("\n");
        }
    }
    if (argc > 5) {   // element view of Kcur
        const int il = std::atoi(argv[4]), t = std::atoi(argv[5]);
        char fn[256];
        std::snprintf(fn, sizeof fn, "%s/Kcur-%d.bin", dir.c_str(), il);
        std::vector<float> ll;
        auto& mine = cap["Kcur-" + std::to_string(il)];
        if (tutil::read_raw(fn, &ll) && mine.size() > (size_t)t) {
            const size_t len = mine[t].size();
            int nd = 0, shown = 0;
            for (size_t i = 0; i < len; ++i) {
                float a = mine[t][i], b = ll[(size_t)t * len + i];
                if (a != b) {
                    ++nd;
                    if (shown++ < 6) { int32_t ia, ib; std::memcpy(&ia, &a, 4); std::memcpy(&ib, &b, 4); std::printf("  Kcur[%zu] mine %.9g llama %.9g  ulp diff %d\n", i, a, b, ia - ib); }
                }
            }
            std::printf("Kcur token %d: %d of %zu elements differ\n", t, nd, len);
        }
    }
    if (argc > 5) {   // per-slot / per-row view of one MoE tensor at one token: argv[5] = token
        const int il = std::atoi(argv[4]), t = std::atoi(argv[5]);
        for (const char* n : {"ffn_moe_gate_up_scaled", "ffn_moe_down_scaled"}) {
            char fn[256];
            std::snprintf(fn, sizeof fn, "%s/%s-%d.bin", dir.c_str(), n, il);
            std::vector<float> ll;
            auto& mine = cap[std::string(n) + "-" + std::to_string(il)];
            if (!tutil::read_raw(fn, &ll) || mine.size() <= (size_t)t) continue;
            const size_t len = mine[t].size(), per = len / 8;
            std::printf("%s token %d: per slot relL2 / #differing rows / max|d|\n", n, t);
            for (int sl = 0; sl < 8; ++sl) {
                double nd = 0, nb = 0, mx = 0; int nrows = 0, firstrow = -1;
                for (size_t i = 0; i < per; ++i) {
                    double a = mine[t][sl * per + i], b = ll[(size_t)t * len + sl * per + i];
                    nd += (a - b) * (a - b); nb += b * b; mx = std::max(mx, std::fabs(a - b));
                    if (a != b) { ++nrows; if (firstrow < 0) firstrow = (int)i; }
                }
                std::printf("  slot %d: %.3g  %d/%zu  max %.3g (first diff row %d)  [expert %d]\n", sl, std::sqrt(nd / nb), nrows, per, mx, firstrow,
                            (int)cap["ffn_moe_topk-" + std::to_string(il)][t][sl]);
            }
        }
    }
    return 0;
}

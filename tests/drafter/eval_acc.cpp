// eval_acc: acceptance of the drafter on REAL taps (S2's dump_taps files) + fold-vs-unfold top-1 agreement on real hidden states. Owner: agent S1.
//   eval_acc [--threads 4] [--bf16] [--dir tests/spec/taps] [--prompts a,b|all] [--ref-every K] [--max-anchors N] [--sample-check]
// Simulates the greedy speculative loop at EVERY generated position p (anchor = tokens[p], committed rows 0..p-1, truth = tokens[p+1..]); a draft of n = 7 gives acc@1..7
// (the first j tokens of an n=7 draft equal an n=j draft: tested). L(n) = 1 + sum_{j<=n} acc@j. Full-length drafts only (p + 7 <= last position), like docs/DSPARK.md 9.1.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "light_target.h"
#include "ref_drafter.h"

using namespace ink;
using namespace ink::dr;

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct TapFile {
    int n = 0, n_prompt = 0;
    std::vector<int32_t> tokens, next;
    std::vector<float> taps;
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        unsigned char h[64];
        if (std::fread(h, 1, 64, f) != 64 || std::memcmp(h, "INKTAPS1", 8) != 0) { std::fclose(f); return false; }
        uint32_t u[7];
        std::memcpy(u, h + 8, sizeof(u));   // version, n_tokens, n_taps, tap_dim, n_vocab, flags, n_prompt
        n = (int)u[1];
        n_prompt = (int)u[6];
        if (u[2] != 4 || u[3] != 2304) { std::fclose(f); return false; }
        tokens.resize(n);
        next.resize(n);
        taps.resize((size_t)n * kTapFloats);
        bool ok = std::fread(tokens.data(), 4, n, f) == (size_t)n && std::fread(next.data(), 4, n, f) == (size_t)n &&
                  std::fread(taps.data(), 4, taps.size(), f) == taps.size();
        std::fclose(f);
        return ok;
    }
};

int main(int argc, char** argv) {
    int threads = 4, ref_every = 0, max_anchors = 1 << 30;
    bool bf16 = false, fold_compare = false;
    std::string dir = "tests/spec/taps", plist = "all";
    std::string target = "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf", drafter = "/data/inkling-mini/models/dspark-Inkling-2-Mini-Q8_0.gguf";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--bf16") bf16 = true;
        else if (a == "--fold-compare") fold_compare = true;
        else if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--prompts" && i + 1 < argc) plist = argv[++i];
        else if (a == "--ref-every" && i + 1 < argc) ref_every = std::atoi(argv[++i]);
        else if (a == "--max-anchors" && i + 1 < argc) max_anchors = std::atoi(argv[++i]);
    }
    std::vector<std::string> names;
    if (plist == "all") names = {"chat_answer", "code_completion", "prose_continuation", "reasoning", "tool_call", "long_context_qa", "chat_email", "chat_code"};
    else for (size_t s = 0; s <= plist.size();) { size_t e = plist.find(',', s); if (e == std::string::npos) e = plist.size(); names.push_back(plist.substr(s, e - s)); s = e + 1; }
    LightTarget lt;
    std::string err;
    if (!lt.init(target, threads, 4096, &err)) { std::fprintf(stderr, "light target: %s\n", err.c_str()); return 2; }
    DrafterOptions opt;
    opt.fold_q8 = !bf16;
    opt.max_states = 1;
    auto D = DrafterImpl::make(lt.t, drafter, opt, &err);
    if (!D) { std::fprintf(stderr, "create: %s\n", err.c_str()); return 2; }
    std::unique_ptr<DrafterImpl> D2;   // --fold-compare: the other storage format of M, driven with the first drafter's tokens
    if (fold_compare) {
        DrafterOptions o2 = opt;
        o2.fold_q8 = bf16;
        D2 = DrafterImpl::make(lt.t, drafter, o2, &err);
        if (!D2) { std::fprintf(stderr, "create 2: %s\n", err.c_str()); return 2; }
    }
    std::printf("eval_acc: M as %s%s, %d threads\n", bf16 ? "bf16" : "Q8_0", fold_compare ? " (+ the other format compared)" : "", threads);
    const int NV = lt.t.n_vocab;
    long pooled_anchor = 0, pooled_acc[8] = {};
    long pooled_steps[8] = {}, pooled_toks[8] = {};   // loop-simulated (per verify step, comparable with DSPARK.md 9.2): p += accepted + 1
    long fold_pos = 0, fold_agree = 0;
    double fold_maxerr = 0, fold_sumerr = 0;
    // fold-compare statistics: agreement of {A = first drafter, B = second drafter, U = unfolded Q8-activation reference} with the exact-activation reference E
    long cmp_pos = 0, agree_AE = 0, agree_BE = 0, agree_UE = 0, agree_AU = 0, agree_BU = 0;
    double err_AE = 0, err_BE = 0, err_UE = 0;
    double sum_AE = 0, sum_BE = 0, sum_UE = 0;
    std::vector<double> near_ties;
    std::printf("%-20s %8s  %6s %6s %6s %6s %6s %6s %6s   %6s %6s  %6s %6s   ms/draft\n", "prompt", "anchors", "acc@1", "acc@2", "acc@3", "acc@4", "acc@5", "acc@6", "acc@7", "L(n=3)", "L(n=7)", "Ls(3)", "Ls(7)");
    std::printf("(acc@k and L: every anchor of the greedy path; Ls(n) = mean tokens per verify step of the simulated loop over the same path, R's statistic)\n");
    for (const std::string& name : names) {
        TapFile T;
        if (!T.load(dir + "/" + name + ".taps")) { std::printf("%-20s (no dump)\n", name.c_str()); continue; }
        const int st = D->state_new();
        const int st2 = D2 ? D2->state_new() : -1;
        const int L = T.n_prompt;
        for (int off = 0; off < L; off += 256) {
            D->commit(st, T.taps.data() + (size_t)off * kTapFloats, std::min(256, L - off));
            if (D2) D2->commit(st2, T.taps.data() + (size_t)off * kTapFloats, std::min(256, L - off));
        }
        long anchors = 0, acc[8] = {};
        std::vector<int> ks;   // prefix-match length per anchor
        double tdraft = 0;
        int nref = 0;
        for (int p = L; p + 7 <= T.n - 1 && anchors < max_anchors; ++p) {
            DraftRequest rq;
            rq.anchor = T.tokens[p];
            rq.n = 7;
            int32_t d[8];
            const bool do_ref = ref_every > 0 && (p - L) % ref_every == 0;
            D->dbg.on = do_ref;
            const double t0 = now_s();
            D->draft(st, rq, d, nullptr);
            tdraft += now_s() - t0;
            D->dbg.on = false;
            int k = 0;
            while (k < 7 && d[k] == T.tokens[p + 1 + k]) ++k;
            for (int j = 1; j <= k; ++j) acc[j]++;
            ++anchors;
            ks.push_back(k);
            if (do_ref) {   // unfolded reference chain on the drafter's own hidden states, fed with the drafter's tokens
                rd::ChainOut co;
                rd::chain(*D, D->state(st).f_rnn, D->dbg.hid.data(), rq.anchor, 7, d, &co);
                std::vector<float> lg2;
                if (D2) {
                    D2->dbg.on = true;
                    D2->dbg.force_on = true;
                    for (int j = 0; j < 7; ++j) D2->dbg.force[j] = d[j];
                    int32_t d2[8];
                    D2->draft(st2, rq, d2, nullptr);
                    D2->dbg.on = false;
                    D2->dbg.force_on = false;
                    lg2 = D2->dbg.logits;
                }
                std::vector<float> ex(NV), hc(kD), corr(kD);
                for (int j = 0; j < 7; ++j) {
                    const float* ref_l = co.logits[j].data();
                    const float* got = D->dbg.logits.data() + (size_t)j * NV;
                    int b = 0;
                    for (int v = 1; v < NV; ++v) if (ref_l[v] > ref_l[b]) b = v;
                    float second = -1e30f;
                    for (int v = 0; v < NV; ++v) if (v != b && ref_l[v] > second) second = ref_l[v];
                    double me = 0;
                    for (int v = 0; v < NV; ++v) me = std::max(me, (double)std::fabs(ref_l[v] - got[v]));
                    fold_maxerr = std::max(fold_maxerr, me);
                    fold_sumerr += me;
                    ++fold_pos;
                    if (b == d[j]) ++fold_agree; else near_ties.push_back(ref_l[b] - second);
                    if (D2) {   // exact-activation reference: softcap(lm_head . (hid + corr)) in fp32, no activation quantisation
                        const float* hid = D->dbg.hid.data() + (size_t)(j + 1) * kD;
                        for (int k = 0; k < kD; ++k) {
                            double a = D->hout_b[k];
                            for (int i = 0; i < RANK; ++i) a += (double)bf16_to_f32(D->hout_w[(size_t)k * RANK + i]) * co.lat[j][i];
                            hc[k] = hid[k] + (float)a;
                        }
                        lt.pool->run_chunks(NV, 256, [&](int, int64_t r0, int64_t r1) {
                            for (int64_t v = r0; v < r1; ++v) {
                                const block_q8_0* bl = (const block_q8_0*)((const char*)lt.t.lm_head.w + (size_t)v * lt.t.lm_head.rs);
                                __m512 acc = _mm512_setzero_ps();
                                for (int blk = 0; blk < kD / 32; ++blk) {
                                    const __m512 dd = _mm512_set1_ps(fp16_to_f32(bl[blk].d));
                                    acc = _mm512_fmadd_ps(_mm512_mul_ps(dd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)bl[blk].qs)))), _mm512_loadu_ps(hc.data() + blk * 32), acc);
                                    acc = _mm512_fmadd_ps(_mm512_mul_ps(dd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(bl[blk].qs + 16))))), _mm512_loadu_ps(hc.data() + blk * 32 + 16), acc);
                                }
                                ex[v] = (float)(30.0 * std::tanh((double)_mm512_reduce_add_ps(acc) / 30.0));
                            }
                        });
                        auto am = [&](const float* x) { int bb = 0; for (int v = 1; v < NV; ++v) if (x[v] > x[bb]) bb = v; return bb; };
                        auto md = [&](const float* x, const float* y) { double m = 0; for (int v = 0; v < NV; ++v) m = std::max(m, (double)std::fabs(x[v] - y[v])); return m; };
                        const int e = am(ex.data()), a_ = am(got), b_ = am(lg2.data() + (size_t)j * NV), u_ = am(ref_l);
                        ++cmp_pos;
                        agree_AE += (a_ == e); agree_BE += (b_ == e); agree_UE += (u_ == e); agree_AU += (a_ == u_); agree_BU += (b_ == u_);
                        const double ea = md(got, ex.data()), eb = md(lg2.data() + (size_t)j * NV, ex.data()), eu = md(ref_l, ex.data());
                        err_AE = std::max(err_AE, ea); err_BE = std::max(err_BE, eb); err_UE = std::max(err_UE, eu);
                        sum_AE += ea; sum_BE += eb; sum_UE += eu;
                    }
                }
                ++nref;
            }
            D->commit(st, T.taps.data() + (size_t)p * kTapFloats, 1);
            if (D2) D2->commit(st2, T.taps.data() + (size_t)p * kTapFloats, 1);
        }
        if (D2) D2->state_free(st2);
        double L3 = 1, L7 = 1;
        for (int j = 1; j <= 3; ++j) L3 += (double)acc[j] / std::max(1L, anchors);
        for (int j = 1; j <= 7; ++j) L7 += (double)acc[j] / std::max(1L, anchors);
        std::printf("%-20s %8ld ", name.c_str(), anchors);
        for (int j = 1; j <= 7; ++j) std::printf(" %6.3f", (double)acc[j] / std::max(1L, anchors));
        auto lstep = [&](int n, long* steps_out, long* toks_out) {
            long steps = 0, toks = 0;
            for (size_t i = 0; i < ks.size();) { const int k = std::min(ks[i], n); ++steps; toks += k + 1; i += (size_t)k + 1; }
            *steps_out += steps; *toks_out += toks;
            return steps ? (double)toks / steps : 0.0;
        };
        long s3 = 0, t3 = 0, s7 = 0, t7 = 0;
        const double Ls3 = lstep(3, &s3, &t3), Ls7 = lstep(7, &s7, &t7);
        pooled_steps[3] += s3; pooled_toks[3] += t3; pooled_steps[7] += s7; pooled_toks[7] += t7;
        std::printf("   %6.2f %6.2f  %6.2f %6.2f   %6.1f\n", L3, L7, Ls3, Ls7, 1e3 * tdraft / std::max(1L, anchors));
        std::fflush(stdout);
        pooled_anchor += anchors;
        for (int j = 1; j <= 7; ++j) pooled_acc[j] += acc[j];
        D->state_free(st);
    }
    double L3 = 1, L7 = 1;
    for (int j = 1; j <= 3; ++j) L3 += (double)pooled_acc[j] / std::max(1L, pooled_anchor);
    for (int j = 1; j <= 7; ++j) L7 += (double)pooled_acc[j] / std::max(1L, pooled_anchor);
    std::printf("%-20s %8ld ", "POOLED", pooled_anchor);
    for (int j = 1; j <= 7; ++j) std::printf(" %6.3f", (double)pooled_acc[j] / std::max(1L, pooled_anchor));
    std::printf("   %6.2f %6.2f  %6.2f %6.2f\n", L3, L7, (double)pooled_toks[3] / std::max(1L, pooled_steps[3]), (double)pooled_toks[7] / std::max(1L, pooled_steps[7]));
    if (fold_pos) {
        std::sort(near_ties.begin(), near_ties.end());
        std::printf("fold check (M as %s vs unfolded two-lm_head reference, drafter's own tokens fed back): top-1 agreement %ld / %ld = %.3f %%, max |dlogit| %.4f, mean of per-step max %.4f\n",
                    bf16 ? "bf16" : "Q8_0", fold_agree, fold_pos, 100.0 * fold_agree / fold_pos, fold_maxerr, fold_sumerr / fold_pos);
        std::printf("  disagreements: %zu, their reference top1-top2 gaps: ", near_ties.size());
        for (size_t i = 0; i < near_ties.size() && i < 12; ++i) std::printf("%.4f ", near_ties[i]);
        std::printf("\n");
    }
    if (cmp_pos) {
        std::printf("fold compare over %ld positions (A = M as %s, B = M as %s, U = unfolded Q8-activation reference (llama.cpp style), E = exact-activation reference):\n", cmp_pos,
                    bf16 ? "bf16" : "Q8_0", bf16 ? "Q8_0" : "bf16");
        std::printf("  top-1 agreement with E: A %.3f %%  B %.3f %%  U %.3f %%;   with U: A %.3f %%  B %.3f %%\n", 100.0 * agree_AE / cmp_pos, 100.0 * agree_BE / cmp_pos,
                    100.0 * agree_UE / cmp_pos, 100.0 * agree_AU / cmp_pos, 100.0 * agree_BU / cmp_pos);
        std::printf("  max |dlogit| vs E: A %.4f (mean of step max %.4f)  B %.4f (%.4f)  U %.4f (%.4f)\n", err_AE, sum_AE / cmp_pos, err_BE, sum_BE / cmp_pos, err_UE, sum_UE / cmp_pos);
    }
    return 0;
}

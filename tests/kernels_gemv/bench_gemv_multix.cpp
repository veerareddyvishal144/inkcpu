// bench_gemv_multix (agent K2, wave 1): MoE-shaped A/B of the small-n multi-activation GEMV against the batched GEMM (gemm_moe_*) and the plain GEMV called
// once per activation, at 1 / 2 / 4 / 8 pairs per expert. Synthetic random weights, working set > L3 (streamed from DRAM like real expert weights), 32-row
// items like the prefill/verify executor (gate rows + up rows of one expert per item for gate_up, 32 down rows per item for down).
//   bench_gemv_multix [--threads 1] [--cpus 9,11,13,15] [--pairs 1,2,4,8] [--experts 48] [--reps 5] [--what gu|dn|both] [--fmt nvfp4|q8|both] [--pf 512] [--hint 0]
// Kernels are alternated inside each rep (same window A/B); the host is shared: quote RATIOS, the loadavg is printed per line.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/kernels_gemm.h"
#include "ink/kernels_gemv.h"
#include "ink/kernels_gemv_multix.h"
#include "ink/nvfp4_repack.h"
#include "ink/threadpool.h"
#include "testutil.h"

using namespace ink;

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

struct Setup {
    int fmt;            // GEMM_FMT_Q8 / GEMM_FMT_NVFP4
    int rows, k, E;     // rows per expert matrix (gate_up: 2304 = gate 1152 | up 1152; down: 2304), K, experts
    size_t rs, es;      // row stride, expert stride
    std::vector<uint8_t> w;     // E experts
    std::vector<uint8_t> wr;    // nvfp4r re-encoded copy (NVFP4 only)
    std::vector<block_q8_0> acts;   // pool of 64 activations
    int npool = 64;
    void* act_pack = nullptr;
    GemmAct act;
    bool gu;            // gate_up shape (items = 32 gate rows + 32 up rows) or down (32 rows)
};

void fill_random(Setup& S, tu::Rng& rng) {
    S.rs = S.fmt == GEMM_FMT_Q8 ? row_bytes_q8_0(S.k) : row_bytes_nvfp4(S.k);
    S.es = S.rs * (size_t)S.rows;
    S.w.resize(S.es * (size_t)S.E);
    for (size_t i = 0; i < S.w.size(); ++i) S.w[i] = (uint8_t)rng.u32();
    // sane scales: Q8 fp16 d in (0, 0.2]; NVFP4 UE4M3 bytes 0x28..0x48
    for (int e = 0; e < S.E; ++e)
        for (int r = 0; r < S.rows; ++r) {
            uint8_t* row = S.w.data() + (size_t)e * S.es + (size_t)r * S.rs;
            if (S.fmt == GEMM_FMT_Q8) {
                for (int b = 0; b < S.k / 32; ++b) {
                    const uint16_t d = f32_to_fp16(0.001f + 0.2f * rng.uni());
                    std::memcpy(row + (size_t)b * 34, &d, 2);
                }
            } else {
                for (int c = 0; c < S.k / 64; ++c)
                    for (int s = 0; s < 4; ++s) row[(size_t)c * 36 + s] = (uint8_t)rng.range(0x28, 0x48);
            }
        }
    if (S.fmt == GEMM_FMT_NVFP4) {
        S.wr = S.w;
        for (int e = 0; e < S.E; ++e) nvfp4r_repack_rows(S.wr.data() + (size_t)e * S.es, S.rs, S.rows, S.k / 64);
    }
    const int nb = S.k / 32;
    S.acts.resize((size_t)S.npool * nb);
    for (auto& b : S.acts) {
        b.d = f32_to_fp16(0.01f + 0.1f * rng.uni());
        for (int j = 0; j < 32; ++j) b.qs[j] = (int8_t)rng.range(-127, 127);
    }
    S.act_pack = std::aligned_alloc(64, (gemm_act_bytes(S.fmt, S.npool, S.k) + 63) & ~(size_t)63);
    gemm_pack_act(S.fmt, S.acts.data(), 0, S.npool, S.npool, S.k, S.act_pack);
    S.act = gemm_act_view(S.fmt, S.act_pack, S.npool, S.k);
}

// routing: expert e has `pairs` pairs using activations (e * pairs + q) % npool
void make_route(const Setup& S, int pairs, MoeRoute& r, std::vector<MoeItem>& items) {
    r.n_experts = S.E;
    r.n_tokens = S.npool;
    r.topk = pairs;
    r.offset.assign((size_t)S.E + 1, 0);
    r.pair_token.clear();
    r.pair_slot.clear();
    for (int e = 0; e < S.E; ++e) {
        r.offset[(size_t)e + 1] = r.offset[(size_t)e] + pairs;
        for (int q = 0; q < pairs; ++q) {
            r.pair_token.push_back((e * pairs + q) % S.npool);
            r.pair_slot.push_back(e * pairs + q);
        }
    }
    items.clear();
    const int half = S.gu ? S.rows / 2 : S.rows;
    for (int e = 0; e < S.E; ++e)
        for (int r0 = 0; r0 < half; r0 += 32) items.push_back(MoeItem{e, r0, std::min(half, r0 + 32)});
}

// one pass over all items with the given kernel; y[pair][rows]
enum Kern { K_GEMM = 0, K_MULTIX, K_MULTIX_R, K_GEMV, K_COUNT };
const char* kname[K_COUNT] = {"gemm_moe", "multix", "multix_r", "gemv xN"};

void run_items(const Setup& S, const MoeRoute& r, const std::vector<MoeItem>& items, int kern, ThreadPool& pool, float* y, size_t ldy, int pairs) {
    const int nb = S.k / 32;
    const bool nv = S.fmt == GEMM_FMT_NVFP4;
    const int half = S.gu ? S.rows / 2 : S.rows;
    pool.run_chunks((int64_t)items.size(), 1, [&](int, int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const MoeItem& it = items[(size_t)i];
            const int p0 = r.offset[(size_t)it.expert];
            const int nr = it.row1 - it.row0;
            const uint8_t* wbase = ((kern == K_MULTIX_R) ? S.wr.data() : S.w.data()) + (size_t)it.expert * S.es;
            auto run_rows = [&](int row0) {
                const uint8_t* w = wbase + (size_t)row0 * S.rs;
                float* yy = y + (size_t)p0 * ldy + row0;
                if (kern == K_GEMM) {
                    const MoeItem ii{it.expert, row0, row0 + nr};
                    // gemm_moe_* writes y[pair*ldy + row] with absolute indices
                    if (nv)
                        gemm_moe_nvfp4(S.w.data(), S.es, S.rs, S.act, r.pair_token.data(), r, ii, S.k, y, ldy);
                    else
                        gemm_moe_q8_0(S.w.data(), S.es, S.rs, S.act, r.pair_token.data(), r, ii, S.k, y, ldy);
                } else if (kern == K_GEMV) {
                    for (int q = 0; q < pairs; ++q) {
                        const block_q8_0* x = S.acts.data() + (size_t)r.pair_token[(size_t)(p0 + q)] * nb;
                        if (nv)
                            gemv_nvfp4(w, S.rs, x, S.k, yy + (size_t)q * ldy, nr);
                        else
                            gemv_q8_0(w, S.rs, x, S.k, yy + (size_t)q * ldy, nr);
                    }
                } else {
                    const block_q8_0* xs[kMultixMaxX];
                    for (int q = 0; q < pairs; ++q) xs[q] = S.acts.data() + (size_t)r.pair_token[(size_t)(p0 + q)] * nb;
                    if (!nv)
                        gemv_q8_0_multix(w, S.rs, nr, xs, pairs, S.k, yy, ldy);
                    else if (kern == K_MULTIX)
                        gemv_nvfp4_multix(w, S.rs, nr, xs, pairs, S.k, yy, ldy);
                    else
                        gemv_nvfp4r_multix(w, S.rs, nr, xs, pairs, S.k, yy, ldy);
                }
            };
            run_rows(it.row0);
            if (S.gu) run_rows(half + it.row0);   // the matching up rows
        }
    });
}
}  // namespace

int main(int argc, char** argv) {
    int threads = 1, reps = 5, E = 48, pf = 512, hint = 0;
    std::vector<int> cpus = {9, 11, 13, 15}, pairs = {1, 2, 4, 8};
    std::string what = "both", fmt = "both";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() { return i + 1 < argc ? argv[++i] : (const char*)""; };
        if (a == "--threads") threads = std::atoi(val());
        else if (a == "--cpus") cpus = parse_list(val());
        else if (a == "--pairs") pairs = parse_list(val());
        else if (a == "--experts") E = std::atoi(val());
        else if (a == "--reps") reps = std::atoi(val());
        else if (a == "--what") what = val();
        else if (a == "--fmt") fmt = val();
        else if (a == "--pf") pf = std::atoi(val());
        else if (a == "--hint") hint = std::atoi(val());
    }
    g_gemv_prefetch_bytes = pf;
    g_gemv_prefetch_hint = hint;
    ThreadPool pool(threads, cpus);
    tu::Rng rng(7);
    std::printf("bench_gemv_multix: threads %d, experts %d, reps %d, pf %d hint %d, loadavg %s\n", threads, E, reps, pf, hint, tu::loadavg().c_str());
    std::printf("%-6s %-5s %5s | %-9s %9s %9s %9s | %s\n", "fmt", "shape", "pairs", "kernel", "ms/pass", "GB/s", "GMAC/s", "ratio vs gemm_moe (same window), loadavg");
    for (int f = 0; f < 2; ++f) {
        if (fmt == "nvfp4" && f == 0) continue;
        if (fmt == "q8" && f == 1) continue;
        for (int shape = 0; shape < 2; ++shape) {
            if (what == "gu" && shape == 1) continue;
            if (what == "dn" && shape == 0) continue;
            Setup S;
            S.fmt = f == 0 ? GEMM_FMT_Q8 : GEMM_FMT_NVFP4;
            S.gu = shape == 0;
            S.rows = 2304;
            S.k = S.gu ? 2304 : 1152;
            S.E = E;
            fill_random(S, rng);
            const double mb_pass = (double)S.es * S.E / 1e6;
            for (int np : pairs) {
                if (np > kMultixMaxX) continue;
                MoeRoute r;
                std::vector<MoeItem> items;
                make_route(S, np, r, items);
                const size_t ldy = (size_t)S.rows;
                std::vector<float> y((size_t)r.n_pairs() * ldy, 0.f), yref;
                const double gmac = (double)S.rows * S.k * r.n_pairs() / 1e9;
                double best[K_COUNT];
                for (int kk = 0; kk < K_COUNT; ++kk) best[kk] = 1e30;
                const int nk = S.fmt == GEMM_FMT_NVFP4 ? K_COUNT : K_COUNT;   // multix_r on Q8 just runs the Q8 kernel again (harmless)
                for (int rep = 0; rep < reps; ++rep)
                    for (int kk = 0; kk < nk; ++kk) {
                        if (kk == K_MULTIX_R && S.fmt == GEMM_FMT_Q8) continue;
                        const double t0 = now_ms();
                        run_items(S, r, items, kk, pool, y.data(), ldy, np);
                        const double t = now_ms() - t0;
                        best[kk] = std::min(best[kk], t);
                        if (rep == 0) {
                            if (kk == K_GEMM) yref = y;
                            else if (std::memcmp(y.data(), yref.data(), y.size() * sizeof(float)) != 0) std::printf("  !! %s differs from gemm_moe (bits)\n", kname[kk]);
                        }
                    }
                for (int kk = 0; kk < K_COUNT; ++kk) {
                    if (kk == K_MULTIX_R && S.fmt == GEMM_FMT_Q8) continue;
                    std::printf("%-6s %-5s %5d | %-9s %9.2f %9.1f %9.1f | %.2fx  load %s\n", f == 0 ? "q8_0" : "nvfp4", S.gu ? "gu" : "dn", np, kname[kk], best[kk],
                                mb_pass / best[kk], gmac * 1000.0 / best[kk], best[K_GEMM] / best[kk], tu::loadavg().c_str());
                }
                std::fflush(stdout);
            }
            std::free(S.act_pack);
        }
    }
    return 0;
}

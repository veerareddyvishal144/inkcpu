// dump_taps (agent S2, milestone 1): the 4 DSpark tap vectors of every position of a token file + the target's greedy next token per position,
// so that S1 can test the drafter on real taps. Also the correctness gates of the tap capture (--check N):
//   (1) taps of the batched executor (engine_prefill_all, the sink fires per chunk) == taps of the decode executor (Engine::decode token by token), bit for bit,
//   (2) logits of position i: batched == decode (bit for bit; holds below 768 visible keys, or with INK_PF_ATTN=decode),
//   (3) adding the sink changes no logits: decode with and without tap_sink give identical logits bits.
// usage: dump_taps --tokens FILE.i32 [--out FILE.taps] [--check N] [--gen N] [--threads 6] [--model GGUF] [--max-ctx N]   (INK_CPUS selects the CPUs)
//   --gen N appends N tokens of the target's own greedy continuation (decode path) so that next[i] == tokens[i+1] on the generated tail (the true greedy path, for drafter acceptance tests)
// file format (little endian): 64-byte header { char magic[8] = "INKTAPS1"; u32 version = 1, n_tokens (prompt + generated), n_taps = 4, tap_dim = 2304, n_vocab, flags (bit0: decode-kernel
//   attention), n_prompt, pad... },
//   then i32 tokens[n], i32 next[n] (greedy argmax of the logits after processing position i), f32 taps[n][4][2304] (tap 0/1/2 = residual stream entering visit 5/23/41,
//   tap 3 = final normalised hidden state before the lm head). Position i's row is what the target computes when it processes token i.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/engine.h"
#include "ink/engine_prefill.h"

using namespace ink;

static std::vector<int32_t> read_i32(const std::string& p) {
    std::vector<int32_t> v;
    if (FILE* f = std::fopen(p.c_str(), "rb")) {
        int32_t x;
        while (std::fread(&x, 4, 1, f) == 1) v.push_back(x);
        std::fclose(f);
    }
    return v;
}
static int argmax(const float* x, size_t n) {
    size_t b = 0;
    for (size_t i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return (int)b;
}

int main(int argc, char** argv) {
    std::string model = "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf", tokf, outf;
    int threads = 6, check = 0, max_ctx = 0, gen = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() { return i + 1 < argc ? argv[++i] : (char*)""; };
        if (a == "--tokens") tokf = val();
        else if (a == "--out") outf = val();
        else if (a == "--check") check = std::atoi(val());
        else if (a == "--threads") threads = std::atoi(val());
        else if (a == "--model") model = val();
        else if (a == "--max-ctx") max_ctx = std::atoi(val());
        else if (a == "--gen") gen = std::atoi(val());
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    std::vector<int32_t> toks = read_i32(tokf);
    if (toks.empty()) {
        std::fprintf(stderr, "usage: dump_taps --tokens FILE.i32 [--out FILE.taps] [--check N] [--threads 6]\n");
        return 2;
    }
    const int n_prompt = (int)toks.size();
    int n = n_prompt;
    EngineOptions o;
    o.n_threads = threads;
    o.n_slots = 3;
    o.max_ctx = max_ctx > 0 ? max_ctx : std::max(4096, n_prompt + gen + 64);
    std::string err;
    auto E = Engine::create(model, o, &err);
    if (!E) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const int nv = E->n_vocab();
    std::fprintf(stderr, "loaded; %d tokens, %d threads\n", n, threads);
    std::vector<float> taps((size_t)(n_prompt + gen) * kEngineTapFloats);
    std::vector<int32_t> next(n_prompt + gen);
    // ---- batched pass with the sink
    E->tap_sink = [&](int slot, int64_t first_pos, int rows, const float* t) {
        (void)slot;
        std::memcpy(taps.data() + (size_t)first_pos * kEngineTapFloats, t, (size_t)rows * kEngineTapFloats * sizeof(float));
    };
    const int s0 = E->slot_new(), s1 = E->slot_new(), s2 = E->slot_new();
    const int chunk = std::max(1, std::min(256, prefill_tune().chunk));
    std::vector<float> lg((size_t)chunk * nv);
    const int keep_rows = std::min(n, std::max(check, 0));
    std::vector<float> lg_first((size_t)keep_rows * nv);   // logits of the first `check` positions (batched) for the comparison with decode
    const auto t0 = std::chrono::steady_clock::now();
    for (int off = 0; off < n; off += chunk) {
        const int m = std::min(chunk, n - off);
        const int rc = engine_prefill_all(*E, s0, toks.data() + off, m, lg.data());
        if (rc) {
            std::fprintf(stderr, "prefill failed: %d\n", rc);
            return 1;
        }
        for (int i = 0; i < m; ++i) {
            next[off + i] = argmax(lg.data() + (size_t)i * nv, nv);
            if (off + i < keep_rows) std::memcpy(lg_first.data() + (size_t)(off + i) * nv, lg.data() + (size_t)i * nv, (size_t)nv * 4);
        }
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "batched pass (logits + taps): %.1f s (%.1f tok/s)\n", sec, n / sec);
    if (gen > 0) {   // the target's own greedy continuation through the decode path (the sink fires per token): next[i] == tokens[i+1] from here on
        std::vector<float> l1((size_t)nv);
        const auto tg = std::chrono::steady_clock::now();
        for (int g = 0; g < gen; ++g) {
            const int32_t tok = next[n - 1];
            toks.push_back(tok);
            if (E->decode(s0, tok, l1.data())) {
                std::fprintf(stderr, "decode failed\n");
                return 1;
            }
            next[n] = argmax(l1.data(), nv);
            ++n;
        }
        const double sg = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
        std::fprintf(stderr, "generated %d tokens in %.1f s (%.2f tok/s)\n", gen, sg, gen / sg);
    }
    int rc_all = 0;
    if (check > 0) {
        const int N = keep_rows;
        std::vector<float> tapsd((size_t)N * kEngineTapFloats), l_on((size_t)nv), l_off((size_t)nv);
        std::vector<int> bad_rows;
        double maxd_tap = 0, maxd_logit = 0;
        int mism_logit = 0, mism_hook = 0, mism_tap = 0;
        E->tap_sink = [&](int slot, int64_t first_pos, int rows, const float* t) {
            (void)slot;
            std::memcpy(tapsd.data() + (size_t)first_pos * kEngineTapFloats, t, (size_t)rows * kEngineTapFloats * sizeof(float));
        };
        for (int i = 0; i < N; ++i) {   // decode path, sink ON (slot s1) and OFF (slot s2) in lockstep
            E->tap_sink ? (void)0 : (void)0;
            int rc = E->decode(s1, toks[i], l_on.data());
            Engine::TapSink saved = std::move(E->tap_sink);
            E->tap_sink = nullptr;
            rc |= E->decode(s2, toks[i], l_off.data());
            E->tap_sink = std::move(saved);
            if (rc) {
                std::fprintf(stderr, "decode failed\n");
                return 1;
            }
            if (std::memcmp(l_on.data(), l_off.data(), (size_t)nv * 4) != 0) ++mism_hook;
            if (std::memcmp(l_on.data(), lg_first.data() + (size_t)i * nv, (size_t)nv * 4) != 0) {
                ++mism_logit;
                for (int v = 0; v < nv; ++v) maxd_logit = std::max(maxd_logit, (double)std::fabs(l_on[v] - lg_first[(size_t)i * nv + v]));
            }
            if (std::memcmp(tapsd.data() + (size_t)i * kEngineTapFloats, taps.data() + (size_t)i * kEngineTapFloats, kEngineTapFloats * 4) != 0) {
                ++mism_tap;
                for (int k = 0; k < kEngineTapFloats; ++k)
                    maxd_tap = std::max(maxd_tap, (double)std::fabs(tapsd[(size_t)i * kEngineTapFloats + k] - taps[(size_t)i * kEngineTapFloats + k]));
            }
        }
        std::printf("check over %d positions: taps decode-vs-batched mismatching rows %d (max abs diff %.3g); logits decode-vs-batched mismatching rows %d (max abs diff %.3g); "
                    "logits with-sink vs without-sink mismatching rows %d\n", N, mism_tap, maxd_tap, mism_logit, maxd_logit, mism_hook);
        // sanity of the tap values themselves
        double s3 = 0, s0v = 0;
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < kD; ++k) {
                s0v += std::fabs(taps[(size_t)i * kEngineTapFloats + k]);
                s3 += std::fabs(taps[(size_t)i * kEngineTapFloats + 3 * kD + k]);
            }
        }
        std::printf("mean |tap0| %.4f  mean |tap3| %.4f (final normalised state)\n", s0v / ((double)N * kD), s3 / ((double)N * kD));
        rc_all = (mism_tap || mism_hook) ? 1 : 0;
        std::printf("%s\n", rc_all == 0 && mism_logit == 0 ? "CHECK PASS" : (rc_all == 0 ? "CHECK PASS (taps, hook) / logits differ batched-vs-decode (see numbers)" : "CHECK FAIL"));
    }
    if (!outf.empty()) {
        FILE* f = std::fopen(outf.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", outf.c_str());
            return 1;
        }
        char hdr[64] = {};
        std::memcpy(hdr, "INKTAPS1", 8);
        const uint32_t h[7] = {1u, (uint32_t)n, (uint32_t)kEngineTaps, (uint32_t)kD, (uint32_t)nv, prefill_tune().attn_decode_kernels ? 1u : 0u, (uint32_t)n_prompt};
        std::memcpy(hdr + 8, h, sizeof h);
        std::fwrite(hdr, 1, 64, f);
        std::fwrite(toks.data(), 4, (size_t)n, f);
        std::fwrite(next.data(), 4, (size_t)n, f);
        std::fwrite(taps.data(), 4, (size_t)n * kEngineTapFloats, f);
        std::fclose(f);
        std::fprintf(stderr, "wrote %s (%.1f MB)\n", outf.c_str(), (64.0 + 8.0 * n + 4.0 * (double)n * kEngineTapFloats) / 1e6);
    }
    return rc_all;
}

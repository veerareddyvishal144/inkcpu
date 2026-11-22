// Gates for slot checkpoints + rewind (agent S3, docs/REWIND.md). Usage: test_rewind <mode> [args]
//   quick                  (1) prefill A (1500 tokens, chunk 512), decode 50, rewind to the prompt end, prefill a different 40-token tail, decode 10:
//                              logits of every step and the slot state (conv histories, global KV < pos, SWA window) bit-identical to a fresh slot fed A + tail + 10;
//                          (2) rewind to a mid-prompt chunk checkpoint (1024) + tail + 10 decode vs fresh A[:1024] + tail + 10;
//                          (3) rewind refused for a shallow checkpoint with > 1025 tokens appended since; (4) the exact bound: 1025 appended = restorable
//                              and bit-exact, 1026 = refused; (5) deep checkpoint survives 1200 appended tokens across the 2048 ring wrap, bit-exact.
//   determinism <out.f32>  writes the final logits of scenario (1) (run with two INK_CPUS / INK_THREADS settings and cmp the files)
// env: INK_GGUF, INK_THREADS (default 3), INK_CPUS (pins the pool; e.g. 26,28,30)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ink/engine.h"
#include "ink/engine_prefill.h"

using namespace ink;

namespace {

const std::string kRoot = INK_SOURCE_DIR;
int g_fails = 0;

std::string gguf_path() {
    const char* e = std::getenv("INK_GGUF");
    return e ? e : "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf";
}
int env_threads() {
    const char* e = std::getenv("INK_THREADS");
    return e ? std::atoi(e) : 3;
}
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::string load1() {
    FILE* f = std::fopen("/proc/loadavg", "r");
    char b[64] = {0};
    if (f) {
        if (!std::fgets(b, sizeof b, f)) b[0] = 0;
        std::fclose(f);
    }
    char* sp = std::strchr(b, ' ');
    if (sp) *sp = 0;
    return b;
}
template <class T>
std::vector<T> read_file(const std::string& p) {
    std::vector<T> v;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)n / sizeof(T));
    if (std::fread(v.data(), sizeof(T), v.size(), f) != v.size()) v.clear();
    std::fclose(f);
    return v;
}
std::vector<int32_t> wiki_tokens(int n) {   // real text (C1's ring data), repeated if more are needed
    auto t = read_file<int32_t>(kRoot + "/build-c1b/data/ring2/tokens.i32");
    if (t.empty()) t = read_file<int32_t>(kRoot + "/build-c1b/data/ring/tokens.i32");
    if (t.empty()) t = read_file<int32_t>(kRoot + "/tests/spec/prompts/long_context_qa.i32");
    std::vector<int32_t> out;
    for (size_t i = 0; (int)out.size() < n && !t.empty(); ++i) out.push_back(t[i % t.size()]);
    return out;
}

std::unique_ptr<Engine> make_engine(int threads, int slots, int max_ctx) {
    EngineOptions o;
    o.n_threads = threads;
    o.n_slots = slots;
    o.max_ctx = max_ctx;
    if (const char* cl = std::getenv("INK_CPUS"))
        for (const char* q = cl; *q;) {
            o.cpus.push_back(std::atoi(q));
            while (*q && *q != ',') ++q;
            if (*q == ',') ++q;
        }
    std::string err;
    const double t0 = now_s();
    auto e = Engine::create(gguf_path(), o, &err);
    if (!e) {
        std::fprintf(stderr, "engine load failed: %s\n", err.c_str());
        std::exit(2);
    }
    std::printf("engine loaded in %.1f s, %d threads, load avg %s\n", now_s() - t0, threads, load1().c_str());
    return e;
}

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fails;
}

// Full comparison of what a later token can read: conv histories (memcmp), global KV cells of positions < pos, SWA ring cells inside the window of pos.
bool state_equal(const Engine& e, int a, int b, const char* label) {
    const SlotState &A = e.slots[a], &B = e.slots[b];
    bool ok = A.pos == B.pos;
    const bool hist_ok = std::memcmp(A.hist, B.hist, (size_t)kVisits * kHistPerVisit * 4) == 0;
    ok = ok && hist_ok;
    size_t kv_diff = 0, kv_n = 0;
    const int64_t P = A.pos;
    for (int v = 0; v < kVisits && A.pos == B.pos; ++v) {
        const KvLayer &La = A.kv[v], &Lb = B.kv[v];
        const int64_t lo = La.ring ? std::max<int64_t>(0, P - (kSwaWindow - 1)) : 0;
        for (int64_t p = lo; p < P; ++p) {
            const size_t ca = La.ring ? (size_t)(p % ((int64_t)La.n_blocks * kKvBlock)) : (size_t)p;
            const size_t cb = Lb.ring ? (size_t)(p % ((int64_t)Lb.n_blocks * kKvBlock)) : (size_t)p;
            const size_t ba = ca / kKvBlock, oa = ca % kKvBlock, bb = cb / kKvBlock, ob = cb % kKvBlock;
            for (int g = 0; g < kNKvHeads; ++g) {
                const size_t gs = (size_t)kKvBlock * kHeadDim;   // elements per head group inside a block
                const uint16_t* va = La.v + ba * kKvBlockElems + g * gs + oa * kHeadDim;
                const uint16_t* vb = Lb.v + bb * kKvBlockElems + g * gs + ob * kHeadDim;
                kv_diff += std::memcmp(va, vb, kHeadDim * 2) != 0;
                const uint16_t* ka = La.k + ba * kKvBlockElems + g * gs + oa;
                const uint16_t* kb = Lb.k + bb * kKvBlockElems + g * gs + ob;
                for (int d = 0; d < kHeadDim; ++d) kv_diff += ka[(size_t)d * kKvBlock] != kb[(size_t)d * kKvBlock];
                kv_n += kHeadDim + 1;
            }
        }
    }
    ok = ok && kv_diff == 0;
    std::printf("  %s state %-30s pos %lld/%lld hist %s, KV cells read by the next token: %zu of %zu differ\n", ok ? "PASS" : "FAIL", label,
                (long long)A.pos, (long long)B.pos, hist_ok ? "bit-identical" : "DIFFER", kv_diff, kv_n);
    if (!ok) ++g_fails;
    return ok;
}

struct Run {   // logits of the last prefilled token and of every decode step
    std::vector<std::vector<float>> logits;
};

std::vector<float> prefill(Engine& e, int slot, const std::vector<int32_t>& t, size_t off, size_t n) {
    std::vector<float> lg((size_t)e.n_vocab());
    const double t0 = now_s();
    const int rc = engine_prefill(e, slot, t.data() + off, (int)n, lg.data());
    if (rc) {
        std::printf("  prefill rc %d\n", rc);
        std::exit(3);
    }
    std::printf("    prefill %zu tokens -> pos %lld: %.1f s (%.1f tok/s, load %s)\n", n, (long long)e.slot_pos(slot), now_s() - t0, n / (now_s() - t0), load1().c_str());
    return lg;
}
void decode_n(Engine& e, int slot, const std::vector<int32_t>& t, size_t off, size_t n, Run* run) {
    for (size_t i = 0; i < n; ++i) {
        std::vector<float> lg((size_t)e.n_vocab());
        if (e.decode(slot, t[off + i], lg.data())) std::exit(3);
        if (run) run->logits.push_back(std::move(lg));
    }
}
bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0; }
bool runs_equal(const Run& a, const Run& b, const char* label) {
    bool ok = a.logits.size() == b.logits.size();
    size_t first_bad = 0;
    double worst = 1;
    for (size_t i = 0; ok && i < a.logits.size(); ++i)
        if (!bits_equal(a.logits[i], b.logits[i])) {
            ok = false;
            first_bad = i;
            double ab = 0, aa = 0, bb = 0;
            for (size_t j = 0; j < a.logits[i].size(); ++j) {
                ab += (double)a.logits[i][j] * b.logits[i][j];
                aa += (double)a.logits[i][j] * a.logits[i][j];
                bb += (double)b.logits[i][j] * b.logits[i][j];
            }
            worst = ab / std::sqrt(aa * bb + 1e-300);
        }
    if (ok)
        std::printf("  PASS %-40s %zu logits vectors BIT-IDENTICAL\n", label, a.logits.size());
    else
        std::printf("  FAIL %-40s first difference at step %zu (cosine %.8f)\n", label, first_bad, worst);
    if (!ok) ++g_fails;
    return ok;
}
std::vector<int64_t> ckpts(Engine& e, int slot) {
    std::vector<int64_t> v(64);
    const int n = e.slot_checkpoints(slot, v.data(), 64);
    v.resize((size_t)std::min(n, 64));
    std::string s;
    for (auto p : v) s += std::to_string(p) + " ";
    std::printf("    checkpoints of slot %d: [%s]\n", slot, s.c_str());
    return v;
}
bool has(const std::vector<int64_t>& v, int64_t p) { return std::find(v.begin(), v.end(), p) != v.end(); }

// scenario (1): prefill A, decode 50, rewind to |A|, tail + 10 decodes. Returns the run of the rewound slot; *ref = fresh slot run.
void scenario_prompt_end(Engine& e, int a, int b, const std::vector<int32_t>& W, Run* got, Run* ref) {
    const size_t nA = 1500, nTail = 40, nDec = 10;
    std::printf("[1] prefill A (%zu) + decode 50, rewind to %zu, tail %zu + decode %zu vs fresh\n", nA, nA, nTail, nDec);
    e.slot_reset(a);
    e.slot_reset(b);
    prefill(e, a, W, 0, nA);
    auto cp = ckpts(e, a);
    check(has(cp, 512) && has(cp, 1024) && has(cp, 1500) && cp.size() == 3, "automatic checkpoints at the chunk ends 512, 1024 and the prompt end 1500");
    decode_n(e, a, W, 1500, 50, nullptr);
    cp = ckpts(e, a);   // 512 died on the way: 1550 - 512 = 1038 > 1025 appended
    check(e.slot_pos(a) == 1550 && cp.size() == 2 && has(cp, 1024) && has(cp, 1500), "decode steps add no checkpoints (and 512 is dead: 1038 > 1025 appended)");
    const int64_t r = e.slot_rewind(a, (int64_t)nA);
    check(r == (int64_t)nA && e.slot_pos(a) == (int64_t)nA, "slot_rewind(1500) returns 1500 and sets pos");
    got->logits.push_back(prefill(e, a, W, 2000, nTail));
    decode_n(e, a, W, 2100, nDec, got);
    // reference: fresh slot
    ref->logits.push_back([&] { prefill(e, b, W, 0, nA); return prefill(e, b, W, 2000, nTail); }());
    decode_n(e, b, W, 2100, nDec, ref);
    runs_equal(*got, *ref, "rewind(prompt end) + tail + 10 decodes");
    state_equal(e, a, b, "after scenario (1)");
}

int mode_quick() {
    auto e = make_engine(env_threads(), 2, 4096);
    prefill_tune().chunk = 512;   // production value
    const auto W = wiki_tokens(3200);
    if (W.size() < 3200) {
        std::fprintf(stderr, "no token data\n");
        return 2;
    }
    const int a = e->slot_new(), b = e->slot_new();
    Run got1, ref1;
    scenario_prompt_end(*e, a, b, W, &got1, &ref1);

    // ---- (2) mid-prompt chunk checkpoint: slot a is at 1550 (hwm 1550); nearest checkpoint <= 1300 is 1024 (1550 - 1024 = 526 <= 1025)
    std::printf("[2] rewind(1300) -> 1024 (chunk checkpoint), tail 40 + decode 10 vs fresh A[:1024] + tail + 10\n");
    {
        const int64_t r = e->slot_rewind(a, 1300);
        check(r == 1024 && e->slot_pos(a) == 1024, "slot_rewind(1300) lands on the 1024 chunk checkpoint");
        check(!has(ckpts(*e, a), 1500), "checkpoints above the rewound position are dropped");
        Run got, ref;
        got.logits.push_back(prefill(*e, a, W, 3000, 40));
        decode_n(*e, a, W, 3100, 10, &got);
        e->slot_reset(b);
        prefill(*e, b, W, 0, 1024);
        ref.logits.push_back(prefill(*e, b, W, 3000, 40));
        decode_n(*e, b, W, 3100, 10, &ref);
        runs_equal(got, ref, "rewind(mid-prompt 1024) + tail + 10 decodes");
        state_equal(*e, a, b, "after scenario (2)");
    }
    // ---- (3) refusal: slot a hwm is 1550 (the abandoned continuation); checkpoint 512 has 1550 - 512 = 1038 > 1025 appended -> dead
    std::printf("[3] refusal: 512 is dead (hwm 1550 - 512 = 1038 > %lld)\n", (long long)Engine::rewind_max_append(kSwaRingBlocks));
    {
        check(!has(ckpts(*e, a), 512), "512 is not listed as restorable");
        check(e->slot_rewind(a, 512) == -1 && e->slot_pos(a) == 1074, "slot_rewind(512) is refused and leaves the slot unchanged");
        check(e->slot_rewind(a, 1023) == -1, "slot_rewind(1023) (no restorable checkpoint <= 1023) is refused");
    }
    // ---- (4) exact bound: checkpoint at 512, append until hwm - 512 == 1025 -> restorable and bit-exact; one more -> refused
    const int64_t bound = Engine::rewind_max_append(kSwaRingBlocks);
    std::printf("[4] exact bound %lld: prefill 600 (checkpoints 512, 600), append %lld more -> rewind(512) ok; append %lld -> refused\n", (long long)bound,
                (long long)(bound - 88), (long long)(bound - 87));
    {
        e->slot_reset(a);
        prefill(*e, a, W, 0, 600);
        prefill(*e, a, W, 600, (size_t)(512 + bound - 600));   // hwm = 512 + bound
        check(e->slot_pos(a) == 512 + bound, "hwm == 512 + bound");
        check(e->slot_rewind(a, 512) == 512, "rewind(512) at exactly the bound is accepted");
        Run got, ref;
        got.logits.push_back(prefill(*e, a, W, 2000, 40));
        decode_n(*e, a, W, 2100, 5, &got);
        e->slot_reset(b);
        prefill(*e, b, W, 0, 512);
        ref.logits.push_back(prefill(*e, b, W, 2000, 40));
        decode_n(*e, b, W, 2100, 5, &ref);
        runs_equal(got, ref, "rewind at exactly the bound + tail + 5 decodes");
        state_equal(*e, a, b, "after scenario (4a)");
        e->slot_reset(a);
        prefill(*e, a, W, 0, 600);
        prefill(*e, a, W, 600, (size_t)(512 + bound + 1 - 600));   // hwm = 512 + bound + 1
        check(e->slot_rewind(a, 512) == -1, "rewind(512) one past the bound is refused");
        check(e->slot_rewind(a, 599) == -1, "rewind(599): nothing restorable below 600");
        check(e->slot_rewind(a, 600) == 600 && e->slot_pos(a) == 600, "rewind(600) (bound - 87 appended) is accepted");
        got = Run();
        ref = Run();
        got.logits.push_back(prefill(*e, a, W, 2000, 40));
        decode_n(*e, a, W, 2100, 3, &got);
        e->slot_reset(b);
        prefill(*e, b, W, 0, 600);
        ref.logits.push_back(prefill(*e, b, W, 2000, 40));
        decode_n(*e, b, W, 2100, 3, &ref);
        runs_equal(got, ref, "rewind(600) + tail + 3 decodes");
    }
    // ---- (5) deep checkpoint: A (1500), deep at 1500, append 1200 (crosses the ring wrap at 2048), rewind(1500) -> same tail as (1): bit-identical to ref1
    std::printf("[5] deep checkpoint at 1500 survives 1200 appended tokens (ring wrap at 2048); shallow 1024 dies\n");
    {
        e->slot_reset(a);
        prefill(*e, a, W, 0, 1500);
        check(e->slot_checkpoint(a, true) == 0, "slot_checkpoint(deep) at 1500");
        prefill(*e, a, W, 1500, 1200);
        check(e->slot_pos(a) == 2700, "pos 2700 (wrapped the 2048 ring)");
        auto cp = ckpts(*e, a);
        check(has(cp, 1500) && !has(cp, 1024) && !has(cp, 512), "1500 (deep) restorable, 1024 / 512 (shallow, 1676 / 2188 appended) dead");
        check(e->slot_rewind(a, 1499) == -1, "rewind(1499): nothing restorable below the deep checkpoint");
        check(e->slot_rewind(a, 1500) == 1500 && e->slot_pos(a) == 1500, "rewind(1500) restores the deep checkpoint");
        Run got;
        got.logits.push_back(prefill(*e, a, W, 2000, 40));
        decode_n(*e, a, W, 2100, 10, &got);
        runs_equal(got, ref1, "deep rewind after 1200 tokens + tail + 10 decodes == fresh");
        e->slot_reset(b);
        prefill(*e, b, W, 0, 1500);
        prefill(*e, b, W, 2000, 40);
        decode_n(*e, b, W, 2100, 10, nullptr);
        state_equal(*e, a, b, "after scenario (5)");
        // a second rewind to the same deep checkpoint still works (the entry stays), and the deep store is bounded
        check(e->slot_rewind(a, 1500) == 1500, "the deep checkpoint can be used again");
        std::printf("    checkpoint memory held by slot a: %.1f MiB\n", slot_ckpt_bytes(e->slots[a]) / 1048576.0);
    }
    std::printf("%s: %d failures (load %s)\n", g_fails ? "FAIL" : "ALL PASS", g_fails, load1().c_str());
    return g_fails ? 1 : 0;
}

int mode_determinism(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: test_rewind determinism <out.f32>\n");
        return 2;
    }
    auto e = make_engine(env_threads(), 2, 4096);
    prefill_tune().chunk = 512;
    const auto W = wiki_tokens(3200);
    const int a = e->slot_new(), b = e->slot_new();
    Run got, ref;
    scenario_prompt_end(*e, a, b, W, &got, &ref);
    FILE* f = std::fopen(argv[2], "wb");
    for (const auto& v : got.logits) std::fwrite(v.data(), 4, v.size(), f);
    std::fclose(f);
    std::printf("wrote %zu logits vectors to %s; %s\n", got.logits.size(), argv[2], g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string m = argc > 1 ? argv[1] : "quick";
    if (m == "quick") return mode_quick();
    if (m == "determinism") return mode_determinism(argc, argv);
    std::fprintf(stderr, "unknown mode %s\n", m.c_str());
    return 2;
}

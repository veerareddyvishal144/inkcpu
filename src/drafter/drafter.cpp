// DSpark block drafter for Inkling 2 Mini (agent S1). Spec: docs/DSPARK.md 4-6; design and measurements: docs/DRAFTER.md.
// ONE ThreadPool::run per draft(): [ctx K/V flush rounds] -> embed -> 3 layers x (norm, QKV GEMM, q/k prep, attention, merge, o-proj, FFN) ->
// final norm -> ONE batched lm_head GEMM for hid[1..n] (+ joint_proj parts) -> n chain steps (joint matvec, Markov/RNN update, M matvec + softcap + argmax/sample).
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "drafter_impl.h"

namespace ink {

const char* DrafterProfile::name(int p) {
    static const char* n[P_COUNT] = {"ctx K/V flush", "embed + rope tables", "norm + QKV gemm + q/k prep", "attention (items + merge)", "o-proj + residual",
                                     "FFN (norm, gate/up, pack, down)", "final norm", "base lm_head gemm + joint parts", "chain: joint matvec", "chain: M matvec + softcap + argmax",
                                     "sampling post pass"};
    return p >= 0 && p < P_COUNT ? n[p] : "?";
}

namespace dr {

namespace {

constexpr int NPH = 1024;                  // dynamic-scheduling counters
constexpr int64_t SWA_TOKENS = 1024;

inline double uniform01(uint64_t seed, int j) {
    uint64_t x = seed + (uint64_t)(j + 1) * 0x9E3779B97F4A7C15ull;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

}  // namespace

int32_t sample_select(const ChunkRes* cres, int nchunks, const float* qrow, int nv, uint64_t seed, int j, double* M_out, double* Z_out) {
    double M = cres[0].val;
    for (int ch = 1; ch < nchunks; ++ch) M = std::max<double>(M, cres[ch].val);
    double Zs = 0;
    for (int ch = 0; ch < nchunks; ++ch) Zs += std::exp((double)cres[ch].val - M) * (double)cres[ch].sum;
    *M_out = M;
    *Z_out = Zs;
    const double target = uniform01(seed, j) * Zs;
    double cum = 0;
    for (int ch = 0; ch < nchunks; ++ch) {
        const double sc = std::exp((double)cres[ch].val - M);
        const double w = sc * (double)cres[ch].sum;
        if (cum + w > target || ch == nchunks - 1) {
            const int64_t r0 = (int64_t)ch * CH_ROWS;
            const int rows = (int)std::min<int64_t>(CH_ROWS, nv - r0);
            const double rem = (target - cum) / sc;
            double acc = 0;
            int pick = -1, lastpos = -1;
            for (int k = 0; k < rows; ++k) {
                const float ev = qrow[r0 + k];
                if (ev > 0.0f) lastpos = k;
                acc += ev;
                if (acc > rem && ev > 0.0f) { pick = k; break; }
            }
            if (pick < 0) pick = lastpos < 0 ? 0 : lastpos;
            return (int32_t)(r0 + pick);
        }
        cum += w;
    }
    return nv - 1;
}

namespace {

void* amalloc(size_t bytes) {
    void* p = std::aligned_alloc(64, (bytes + 63) / 64 * 64);
    if (p) std::memset(p, 0, (bytes + 63) / 64 * 64);
    return p;
}

}  // namespace

struct DrafterImpl::Scr {
    int nvp = 0, nchunks = 0, nv = 0;
    float *X, *hidf, *qkv, *attn_out, *gu_g, *gu_u, *dn_y, *cs_blk, *kraw, *vraw, *cs_ctx, *base, *Pj, *Pf, *upart, *qinternal;
    block_q8_0 *q8x, *q8o, *ffq, *q8ctx;
    void *act_in, *act_o, *act_dn[2], *act_ctx[NL];
    AttnPartial2* part;
    ChunkRes* cres;                           // [7][nchunks]
    double mj[8], zj[8];                      // per step: max of the chunk maxima, normaliser (sampled mode)
    struct alignas(64) Thr {
        float s[RANK];
        float x512[512];
        float u[JO];
        float lat[RANK];
        block_q8_0 latq[8];
        float z[CH_ROWS];
        float tmp[CH_ROWS];
    };
    Thr* thr;
    uint64_t ticks[DrafterProfile::P_COUNT] = {};
    uint64_t calls[DrafterProfile::P_COUNT] = {};
    double tsc_per_us = 2000.0;
    std::vector<void*> frees;
    template <class T>
    T* take(size_t n) {
        void* p = amalloc(n * sizeof(T));
        frees.push_back(p);
        return (T*)p;
    }
    ~Scr() {
        for (void* p : frees) std::free(p);
    }
};

struct DrafterImpl::Run {
    StateData* S = nullptr;
    const DraftRequest* rq = nullptr;
    int32_t* out_tokens = nullptr;
    float* q_out = nullptr;
    const float* src = nullptr;
    int src_rows = 0;
    int64_t src_pos0 = 0;
    int rounds = 0;
    int n = 0;
    bool sample = false;
    float inv_t = 1.0f;
    uint64_t seed = 0;
    int32_t anchor = 0;
    int64_t p = 0;
    bool dbg = false;
    uint64_t last = 0;
    int phases = 0;
};

// ------------------------------------------------------------------------------------------------------------------------------ load
std::unique_ptr<DrafterImpl> DrafterImpl::make(const DrafterTarget& t, const std::string& path, const DrafterOptions& opt, std::string* err) {
    auto fail = [&](const std::string& m) -> std::unique_ptr<DrafterImpl> {
        if (err) *err = "drafter: " + m;
        return nullptr;
    };
    if (!t.pool || t.lm_head.type != T_Q8_0 || t.lm_head.k != D || !t.tok_embd || !t.tok_norm) return fail("incomplete target description");
    if (t.pool->size() > MAXTH) return fail("too many threads");
    auto d = std::unique_ptr<DrafterImpl>(new DrafterImpl());
    d->tgt = t;
    d->opt = opt;
    std::string e2;
    d->gguf_ = GgufFile::open(path, &e2);
    if (!d->gguf_) return fail("cannot open " + path + ": " + e2);
    const GgufFile& g = *d->gguf_;
    std::string arch;
    if (!g.get_string("general.architecture", &arch) || arch != "dflash") return fail("not a dflash GGUF");
    uint32_t u = 0;
    if (!g.get_u32("dflash.block_size", &u) || u != BS) return fail("block_size != 8");
    if (!g.get_u32("dflash.dspark.markov_rank", &u) || u != RANK) return fail("markov_rank != 256");
    if (!g.get_u32("dflash.block_count", &u) || u != NL) return fail("block_count != 3");
    std::vector<int64_t> tl;
    if (!g.get_arr_i64("dflash.target_layers", &tl) || tl.size() != 4 || tl[0] != 5 || tl[1] != 23 || tl[2] != 41 || tl[3] != 48) return fail("target_layers != [5,23,41,48]");
    bool b = false;
    if (g.get_bool("dflash.attention.causal", &b) && b) return fail("causal drafter not supported");
    if (g.get_bool("dflash.sample_from_anchor", &b) && b) return fail("sample_from_anchor not supported");
    if (g.get_bool("dflash.has_confidence_head", &b) && b) return fail("confidence head not supported");
    float rope_base = 0, cap = 0, eps = 0;
    if (!g.get_f32("dflash.rope.freq_base", &rope_base) || rope_base != 1.0e6f) return fail("rope theta != 1e6");
    if (g.get_f32("dflash.final_logit_softcapping", &cap) && std::fabs(cap - t.softcap) > 1e-3f) return fail("softcap differs from the target's");
    if (g.get_f32("dflash.attention.layer_norm_rms_epsilon", &eps) && std::fabs(eps - t.rms_eps) > 1e-9f) return fail("rms eps differs from the target's");

    struct Pend { void** dst; const GgufTensorInfo* ti; size_t bytes; };
    std::vector<Pend> pend;
    std::string bad;
    auto find = [&](const std::string& name, int type, int64_t ne0, int64_t ne1) -> const GgufTensorInfo* {
        const GgufTensorInfo* ti = g.tensor(name);
        if (!ti) { bad = "missing tensor " + name; return nullptr; }
        if (ti->type != type || ti->ne[0] != ne0 || ti->ne[1] != ne1) { bad = "unexpected type/shape of " + name; return nullptr; }
        return ti;
    };
    auto mat = [&](Mat& m, const std::string& name, int64_t k, int64_t rows) {
        const GgufTensorInfo* ti = find(name, T_Q8_0, k, rows);
        if (!ti) return;
        m.type = T_Q8_0; m.rows = (int)rows; m.k = (int)k; m.rs = ti->row_bytes();
        pend.push_back({(void**)&m.w, ti, (size_t)rows * m.rs});
    };
    auto vecf = [&](const float** dst, const std::string& name, int64_t n) {
        const GgufTensorInfo* ti = find(name, T_F32, n, 1);
        if (ti) pend.push_back({(void**)dst, ti, (size_t)n * 4});
    };
    for (int l = 0; l < NL && bad.empty(); ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        LayerW& W = d->L[l];
        mat(W.wq, p + "attn_q.weight", D, QW);
        mat(W.wk, p + "attn_k.weight", D, KVW);
        mat(W.wv, p + "attn_v.weight", D, KVW);
        mat(W.wo, p + "attn_output.weight", QW, D);
        mat(W.wk_ctx, p + "attn_k_ctx.weight", D, KVW);
        mat(W.wv_ctx, p + "attn_v_ctx.weight", D, KVW);
        mat(W.wgate, p + "ffn_gate.weight", D, FF);
        mat(W.wup, p + "ffn_up.weight", D, FF);
        mat(W.wdown, p + "ffn_down.weight", FF, D);
        vecf(&W.attn_norm, p + "attn_norm.weight", D);
        vecf(&W.q_norm, p + "attn_q_norm.weight", HD);
        vecf(&W.k_norm, p + "attn_k_norm.weight", HD);
        vecf(&W.k_ctx_norm, p + "attn_k_ctx_norm.weight", HD);
        vecf(&W.dflare_norm, p + "dflare_norm.weight", D);
        vecf(&W.ffn_norm, p + "ffn_norm.weight", D);
    }
    vecf(&d->mask_embd, "mask_embd.weight", D);
    vecf(&d->out_norm, "output_norm.weight", D);
    vecf(&d->dspark_norm, "dspark_norm.weight", D);
    vecf(&d->joint_b, "joint_proj.bias", JO);
    vecf(&d->hout_b, "dspark_hidden_out.bias", D);
    const float* fusew = nullptr;
    if (const GgufTensorInfo* fz = find("dspark_fuse.weight", T_F32, 4, 4)) pend.push_back({(void**)&fusew, fz, 16 * sizeof(float)});
    {
        const GgufTensorInfo* ti = find("joint_proj.weight", T_BF16, JW, JO);
        if (ti) pend.push_back({(void**)&d->joint_w, ti, (size_t)JW * JO * 2});
        const GgufTensorInfo* hw = find("dspark_hidden_out.weight", T_BF16, RANK, D);
        if (hw) d->hout_w = (const uint16_t*)hw->data;
        const GgufTensorInfo* mw = find("markov_w1.weight", T_BF16, RANK, t.n_vocab_padded);
        if (mw) d->markov_w1 = (const uint16_t*)mw->data;
    }
    if (!bad.empty()) return fail(bad);
    if (t.n_vocab > t.n_vocab_padded || t.n_vocab % 6 != 0) { /* fold handles any row count; the 6-row tail path is slow but correct */ }

    // arena (hugepages) for everything that is streamed per draft: layer matrices, joint_proj, norms, M, m0
    auto au = [](size_t x) { return (x + 4095) / 4096 * 4096; };
    std::vector<size_t> offs;
    size_t total = 0;
    for (const Pend& p : pend) { offs.push_back(total); total += au(p.bytes); }
    const size_t m_rs = opt.fold_q8 ? row_bytes_q8_0(RANK) : row_bytes_bf16(RANK);
    const size_t m_off = total;
    total += au((size_t)t.n_vocab * m_rs);
    const size_t m0_off = total;
    total += au((size_t)t.n_vocab * 4);
    d->arena_ = alloc_weight_buffer(total, true);
    if (!d->arena_) return fail("cannot allocate the weight arena");
    d->arena_bytes_ = total;
    char* ar = (char*)d->arena_;
    for (size_t i = 0; i < pend.size(); ++i) *pend[i].dst = ar + offs[i];
    t.pool->run_chunks((int64_t)pend.size() * 64, 1, [&](int, int64_t b0, int64_t b1) {   // parallel copy, 64 pieces per tensor
        for (int64_t c = b0; c < b1; ++c) {
            const size_t i = (size_t)c / 64, k = (size_t)c % 64;
            const size_t per = (pend[i].bytes + 63) / 64, lo = std::min(pend[i].bytes, k * per), hi = std::min(pend[i].bytes, lo + per);
            std::memcpy(ar + offs[i] + lo, pend[i].ti->data + lo, hi - lo);
        }
    });
    for (int a = 0; a < NFUSE; ++a) {   // alpha = softmax over the taps of each fuse row
        float mx = -1e30f;
        for (int k = 0; k < NFUSE; ++k) mx = std::max(mx, fusew[a * 4 + k]);
        double s = 0;
        for (int k = 0; k < NFUSE; ++k) s += std::exp((double)fusew[a * 4 + k] - mx);
        for (int k = 0; k < NFUSE; ++k) d->alpha[a][k] = (float)(std::exp((double)fusew[a * 4 + k] - mx) / s);
    }
    d->fold.q8 = opt.fold_q8;
    d->fold.data = ar + m_off;
    d->fold.rs = m_rs;
    d->fold.m0 = (const float*)(ar + m0_off);
    d->fold.bytes = (size_t)t.n_vocab * m_rs;
    d->weight_bytes_ = total;

    // fold M = lm_head . hidden_out, m0 = lm_head . bias: cache key = content hash of lm_head, hidden_out, bias
    {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t key = hash_bytes_parallel(*t.pool, t.lm_head.w, (size_t)t.n_vocab * t.lm_head.rs);
        key ^= hash_bytes_parallel(*t.pool, d->hout_w, (size_t)D * RANK * 2) * 0x9E3779B97F4A7C15ull;
        key ^= hash_bytes_parallel(*t.pool, d->hout_b, (size_t)D * 4) * 0xC2B2AE3D27D4EB4Full;
        key ^= (uint64_t)t.n_vocab * 1000003ull + (opt.fold_q8 ? 7 : 11);
        char hex[32];
        std::snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)key);
        std::string dir = opt.cache_dir;
        if (dir.empty()) { dir = path; const size_t sl = dir.find_last_of('/'); dir = sl == std::string::npos ? "." : dir.substr(0, sl); }
        d->fold_path = opt.cache_dir == "-" ? std::string() : dir + "/dspark-fold-" + (opt.fold_q8 ? "q8-" : "bf16-") + hex + ".bin";
        bool have = !d->fold_path.empty() && fold_cache_load(d->fold_path, key, opt.fold_q8, t.n_vocab, ar + m_off, d->fold.bytes, (float*)(ar + m0_off));
        d->fold_from_cache = have;
        if (!have) {
            fold_compute(*t.pool, t.lm_head, t.n_vocab, d->hout_w, d->hout_b, opt.fold_q8, ar + m_off, m_rs, (float*)(ar + m0_off));
            if (!d->fold_path.empty() && !fold_cache_store(d->fold_path, key, opt.fold_q8, t.n_vocab, ar + m_off, d->fold.bytes, (const float*)(ar + m0_off)))
                std::fprintf(stderr, "drafter: warning: cannot write the fold cache %s\n", d->fold_path.c_str());
        }
        d->fold_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    // scratch
    d->scr_.reset(new Scr());
    Scr& Z = *d->scr_;
    Z.nv = t.n_vocab;
    Z.nvp = (t.n_vocab + 15) / 16 * 16;
    Z.nchunks = (t.n_vocab + CH_ROWS - 1) / CH_ROWS;
    Z.X = Z.take<float>((size_t)BS * D);
    Z.hidf = Z.take<float>((size_t)BS * D);
    Z.qkv = Z.take<float>((size_t)BS * 5120);
    Z.attn_out = Z.take<float>((size_t)BS * QW);
    Z.gu_g = Z.take<float>((size_t)BS * FF);
    Z.gu_u = Z.take<float>((size_t)BS * FF);
    Z.dn_y = Z.take<float>((size_t)2 * BS * D);
    Z.cs_blk = Z.take<float>((size_t)BS * 128);
    Z.kraw = Z.take<float>((size_t)NL * RC * KVW);
    Z.vraw = Z.take<float>((size_t)NL * RC * KVW);
    Z.cs_ctx = Z.take<float>((size_t)RC * 128);
    Z.base = Z.take<float>((size_t)BS * Z.nvp);
    Z.Pj = Z.take<float>((size_t)BS * JO);
    Z.Pf = Z.take<float>(JO);
    Z.upart = Z.take<float>(JO);
    Z.qinternal = Z.take<float>((size_t)7 * Z.nvp);
    Z.q8x = Z.take<block_q8_0>((size_t)BS * (D / 32));
    Z.q8o = Z.take<block_q8_0>((size_t)BS * (QW / 32));
    Z.ffq = Z.take<block_q8_0>((size_t)2 * BS * (FF / 64));
    Z.q8ctx = Z.take<block_q8_0>((size_t)NL * RC * (D / 32));
    Z.act_in = Z.take<char>(gemm_act_bytes(GEMM_FMT_Q8, BS, D));
    Z.act_o = Z.take<char>(gemm_act_bytes(GEMM_FMT_Q8, BS, QW));
    for (int h = 0; h < 2; ++h) Z.act_dn[h] = Z.take<char>(gemm_act_bytes(GEMM_FMT_Q8, BS, FF / 2));
    for (int l = 0; l < NL; ++l) Z.act_ctx[l] = Z.take<char>(gemm_act_bytes(GEMM_FMT_Q8, RC, D));
    Z.part = (AttnPartial2*)Z.take<char>(sizeof(AttnPartial2) * NKV * NSPLIT_MAX);
    Z.cres = (ChunkRes*)Z.take<char>(sizeof(ChunkRes) * 7 * (size_t)Z.nchunks);
    Z.thr = (Scr::Thr*)Z.take<char>(sizeof(Scr::Thr) * MAXTH);
    d->sched_.c.reset(new Sched::C[NPH]);
    for (int i = 0; i < NPH; ++i) d->sched_.c[i].v.store(0);
    d->sched_.n = NPH;
    {   // tsc calibration
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t c0 = __rdtsc();
        while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < 3000.0) {}
        const uint64_t c1 = __rdtsc();
        Z.tsc_per_us = (double)(c1 - c0) / std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }
    int ns = opt.max_states > 0 ? opt.max_states : std::max(1, t.n_slots);
    d->states_.resize(ns);
    return d;
}

DrafterImpl::~DrafterImpl() {
    for (auto& S : states_) free_state(S);
    if (arena_) free_weight_buffer(arena_);
}

// ------------------------------------------------------------------------------------------------------------------------------ states
bool DrafterImpl::alloc_state(StateData& S, std::string*) {
    for (int l = 0; l < NL; ++l) {
        const bool ring = (l != 1);
        const uint32_t nb = ring ? WIN_BLOCKS : (uint32_t)((tgt.max_ctx + 63) / 64 + 1);
        const size_t bytes = (size_t)nb * kKvBlockElems * 2;
        void* k = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        void* v = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (k == MAP_FAILED || v == MAP_FAILED) return false;
        S.kv[l].k = (uint16_t*)k; S.kv[l].v = (uint16_t*)v; S.kv[l].n_blocks = nb; S.kv[l].ring = ring; S.kv_bytes[l] = bytes;
    }
    S.pend = (float*)amalloc((size_t)PEND_CAP * kTapFloats * sizeof(float));
    return S.pend != nullptr;
}

void DrafterImpl::free_state(StateData& S) {
    for (int l = 0; l < NL; ++l)
        if (S.kv[l].k) {
            munmap(S.kv[l].k, S.kv_bytes[l]);
            munmap(S.kv[l].v, S.kv_bytes[l]);
            S.kv[l] = KvLayer();
        }
    std::free(S.pend);
    S.pend = nullptr;
    S.used = false;
}

int DrafterImpl::state_new() {
    for (size_t i = 0; i < states_.size(); ++i)
        if (!states_[i].used) {
            StateData& S = states_[i];
            if (!S.pend && !alloc_state(S, nullptr)) return -1;
            S.used = true;
            state_reset((int)i);
            return (int)i;
        }
    return -1;
}

void DrafterImpl::state_free(int st) {
    if (!ok(st)) return;
    free_state(states_[st]);
}

void DrafterImpl::state_reset(int st) {
    if (!ok(st)) return;
    StateData& S = states_[st];
    S.pos = S.done = 0;
    S.have_rnn = false;
    for (int l = 0; l < NL; ++l) {   // zero the caches (anonymous mappings: DONTNEED gives zero pages back on touch)
        madvise(S.kv[l].k, S.kv_bytes[l], MADV_DONTNEED);
        madvise(S.kv[l].v, S.kv_bytes[l], MADV_DONTNEED);
    }
}

size_t DrafterImpl::state_bytes(int st) const {
    if (!ok(st)) return 0;
    const StateData& S = states_[st];
    size_t b = (size_t)PEND_CAP * kTapFloats * 4;
    for (int l = 0; l < NL; ++l) {
        const size_t used_blocks = S.kv[l].ring ? std::min<size_t>(S.kv[l].n_blocks, (size_t)((S.pos + 63) / 64)) : (size_t)((S.pos + 63) / 64);
        b += 2 * used_blocks * kKvBlockElems * 2;
    }
    return b;
}

// ------------------------------------------------------------------------------------------------------------------------------ commit
void DrafterImpl::fuse_row(const float* taps, int c, float* out) const {
    alignas(64) float mix[D];
    const float a0 = alpha[c][0], a1 = alpha[c][1], a2 = alpha[c][2], a3 = alpha[c][3];
    const float *t0 = taps, *t1 = taps + D, *t2 = taps + 2 * D, *t3 = taps + 3 * D;
    for (int i = 0; i < D; i += 16) {
        __m512 s = _mm512_mul_ps(_mm512_set1_ps(a0), _mm512_loadu_ps(t0 + i));
        s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_set1_ps(a1), _mm512_loadu_ps(t1 + i)));
        s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_set1_ps(a2), _mm512_loadu_ps(t2 + i)));
        s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_set1_ps(a3), _mm512_loadu_ps(t3 + i)));
        _mm512_storeu_ps(mix + i, s);
    }
    rmsnorm(mix, c < 3 ? L[c].dflare_norm : dspark_norm, out, D, tgt.rms_eps);
}

int DrafterImpl::commit(int st, const float* taps, int rows) {
    if (!ok(st) || rows < 0 || (rows > 0 && !taps)) return -1;
    if (rows == 0) return 0;
    StateData& S = states_[st];
    if (S.pos + rows > tgt.max_ctx) return -2;
    fuse_row(taps + (size_t)(rows - 1) * kTapFloats, 3, S.f_rnn);   // f_rnn = f_3 of the last committed position
    S.have_rnn = true;
    int64_t pending = S.pos - S.done;
    if (pending + rows > PEND_CAP && pending > 0) {
        run_pool(S, nullptr, nullptr, nullptr, S.pend, (int)pending, S.done);
        S.done = S.pos;
        pending = 0;
    }
    if (rows <= PEND_CAP - pending) {
        std::memcpy(S.pend + (size_t)pending * kTapFloats, taps, (size_t)rows * kTapFloats * sizeof(float));
        S.pos += rows;
        return 0;
    }
    for (int off = 0; off < rows; off += 4 * RC) {   // large commit (prompt): straight from the caller's buffer, 256 rows per pool run
        const int n = std::min(4 * RC, rows - off);
        run_pool(S, nullptr, nullptr, nullptr, taps + (size_t)off * kTapFloats, n, S.pos + off);
    }
    S.pos += rows;
    S.done = S.pos;
    return 0;
}

int DrafterImpl::flush(int st) {
    if (!ok(st)) return -1;
    StateData& S = states_[st];
    const int64_t pending = S.pos - S.done;
    if (pending > 0) {
        run_pool(S, nullptr, nullptr, nullptr, S.pend, (int)pending, S.done);
        S.done = S.pos;
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------------------------------------ draft
int DrafterImpl::draft(int st, const DraftRequest& rq, int32_t* out_tokens, float* q_out) {
    if (!ok(st) || !out_tokens || rq.n < 1 || rq.n > kMaxDraft) return -1;
    StateData& S = states_[st];
    if (S.pos <= 0 || !S.have_rnn) return -3;   // no context: nothing to draft from
    if (rq.anchor < 0 || rq.anchor >= tgt.n_vocab_padded) return -4;
    DraftRequest r = rq;
    bool onehot = false;
    if (r.sample && !(r.temperature > 0.0f)) { r.sample = false; onehot = q_out != nullptr; }   // temperature 0 = greedy
    const int64_t pending = S.pos - S.done;
    run_pool(S, &r, out_tokens, q_out, S.pend, (int)pending, S.done);
    S.done = S.pos;
    if (onehot) {
        for (int j = 0; j < r.n; ++j) {
            std::memset(q_out + (size_t)j * tgt.n_vocab, 0, (size_t)tgt.n_vocab * sizeof(float));
            q_out[(size_t)j * tgt.n_vocab + out_tokens[j]] = 1.0f;
        }
    }
    return 0;
}

int DrafterImpl::run_pool(StateData& S, const DraftRequest* rq, int32_t* out_tokens, float* q_out, const float* src, int src_rows, int64_t src_pos0) {
    if (tgt.cpu0 >= 0 && std::this_thread::get_id() != pinned_) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(tgt.cpu0, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        pinned_ = std::this_thread::get_id();
    }
    Run R;
    R.S = &S; R.rq = rq; R.out_tokens = out_tokens; R.q_out = q_out; R.src = src; R.src_rows = src_rows; R.src_pos0 = src_pos0;
    R.rounds = (src_rows + RC - 1) / RC;
    R.dbg = dbg.on && rq;
    if (rq) {
        R.n = rq->n; R.sample = rq->sample; R.inv_t = rq->sample ? 1.0f / rq->temperature : 1.0f; R.seed = rq->seed; R.anchor = rq->anchor; R.p = S.pos;
        if (R.sample && !q_out) R.q_out = scr_->qinternal;   // rows are indexed with the stride nvp when internal, see body()
    }
    const int use = std::min(sched_.used + 8, NPH);
    for (int i = 0; i < use; ++i) sched_.c[i].v.store(0, std::memory_order_relaxed);
    const auto t0 = std::chrono::steady_clock::now();
    tgt.pool->run([&](int tid, int nth) { body(R, tid, nth); });
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    sched_.used = std::max(sched_.used, R.phases);
    if (opt.profile) {
        Scr& Z = *scr_;
        for (int p = 0; p < DrafterProfile::P_COUNT; ++p) {
            prof_.ms[p] += (double)Z.ticks[p] / Z.tsc_per_us / 1000.0;
            prof_.calls[p] += Z.calls[p];
            Z.ticks[p] = 0;
            Z.calls[p] = 0;
        }
        prof_.total_ms += wall;
        if (rq) { prof_.drafts++; prof_.drafted_tokens += rq->n; }
        // analytical traffic (bytes) and MACs per phase: weights streamed + KV read + fold rows; GB/s and GMAC/s in profile_report()
        using DP = DrafterProfile;
        auto rsq = [](int k) { return (double)row_bytes_q8_0(k); };
        double mb[DP::P_COUNT] = {}, mac[DP::P_COUNT] = {};
        const int64_t end_pos = src_pos0 + src_rows;
        for (int round = 0; round < R.rounds; ++round) {
            const int r0 = round * RC, nr = std::min(RC, src_rows - r0);
            const int64_t pos0 = src_pos0 + r0;
            mb[DP::P_FLUSH] += (double)nr * kTapFloats * 4;
            for (int l = 0; l < NL; ++l)
                if (l == 1 || pos0 + nr > end_pos - SWA_TOKENS) { mb[DP::P_FLUSH] += 2.0 * KVW * rsq(D); mac[DP::P_FLUSH] += 2.0 * nr * KVW * D; }
        }
        if (rq) {
            const int n = rq->n;
            for (int l = 0; l < NL; ++l) {
                const int rows = (l == NL - 1) ? n : BS;
                const int64_t nvalid = (l == 1) ? R.p : std::min<int64_t>(R.p, SWA_TOKENS);
                mb[DP::P_QKV] += (QW + 2 * KVW) * rsq(D);
                mac[DP::P_QKV] += (double)rows * QW * D + 2.0 * BS * KVW * D;
                mb[DP::P_ATTN] += (double)nvalid * NKV * HD * 2 * 2;
                mac[DP::P_ATTN] += (double)(nvalid + BS) * 3 * rows * NKV * HD * 2;
                mb[DP::P_WO] += D * rsq(QW);
                mac[DP::P_WO] += (double)rows * D * QW;
                mb[DP::P_FFN] += 2.0 * FF * rsq(D) + D * rsq(FF);
                mac[DP::P_FFN] += 3.0 * rows * FF * D;
            }
            mb[DP::P_BASE] += (double)tgt.n_vocab * tgt.lm_head.rs + (double)JO * (JW - 512) * 2;
            mac[DP::P_BASE] += (double)n * tgt.n_vocab * D + (double)JO * (JW - 512) * (n + 1);
            mb[DP::P_CHAIN_U] += (double)n * JO * 512 * 2;
            mac[DP::P_CHAIN_U] += (double)n * JO * 512;
            mb[DP::P_CHAIN_M] += (double)n * ((double)fold.bytes + 2.0 * tgt.n_vocab * 4 + (rq->sample ? 4.0 * tgt.n_vocab : 0.0));
            mac[DP::P_CHAIN_M] += (double)n * tgt.n_vocab * RANK;
            if (rq->sample) mb[DP::P_POST] += (double)n * tgt.n_vocab * 8;
        }
        for (int p = 0; p < DP::P_COUNT; ++p) { prof_.mb[p] += mb[p] / 1e6; prof_.gmac[p] += mac[p] / 1e9; }
    }
    return 0;
}

void DrafterImpl::profile_report(FILE* f) const {
    const double nd = (double)std::max<uint64_t>(1, prof_.drafts);
    std::fprintf(f, "drafter profile: %llu drafts, %llu tokens, mean wall %.2f ms per run (GB/s and GMAC/s: analytical bytes / MACs over thread-0 phase time)\n",
                 (unsigned long long)prof_.drafts, (unsigned long long)prof_.drafted_tokens, prof_.total_ms / nd);
    std::fprintf(f, "  %-40s %9s %6s %9s %7s %7s  %s\n", "phase", "ms/draft", "%", "MB/draft", "GB/s", "GMAC/s", "barriers");
    double sum_ms = 0, sum_mb = 0, sum_g = 0;
    for (int p = 0; p < DrafterProfile::P_COUNT; ++p)
        if (prof_.calls[p]) {
            const double ms = prof_.ms[p], gbs = ms > 0 ? prof_.mb[p] / ms : 0.0, gm = ms > 0 ? prof_.gmac[p] / (ms * 1e-3) : 0.0;
            std::fprintf(f, "  %-40s %9.3f %5.1f%% %9.1f %7.1f %7.1f  (%llu)\n", DrafterProfile::name(p), ms / nd, 100.0 * ms / std::max(1e-9, prof_.total_ms), prof_.mb[p] / nd, gbs, gm,
                         (unsigned long long)prof_.calls[p]);
            sum_ms += ms; sum_mb += prof_.mb[p]; sum_g += prof_.gmac[p];
        }
    std::fprintf(f, "  %-40s %9.3f %5.1f%% %9.1f %7.1f %7.1f\n", "sum of phases", sum_ms / nd, 100.0 * sum_ms / std::max(1e-9, prof_.total_ms), sum_mb / nd,
                 sum_ms > 0 ? sum_mb / sum_ms : 0.0, sum_ms > 0 ? sum_g / (sum_ms * 1e-3) : 0.0);
}

// ------------------------------------------------------------------------------------------------------------------------------ the pool body
void DrafterImpl::body(Run& R, int tid, int nth) {
    Scr& Z = *scr_;
    ThreadPool& pool = *tgt.pool;
    StateData& S = *R.S;
    const float eps = tgt.rms_eps;
    int ph = 0;
    const bool prof = opt.profile;
    if (tid == 0) R.last = __rdtsc();
    auto mark = [&](int p) {
        if (prof && tid == 0) {
            const uint64_t now = __rdtsc();
            Z.ticks[p] += now - R.last;
            Z.calls[p]++;
            R.last = now;
        }
    };
    auto BAR = [&](int p) { pool.barrier(); mark(p); };
    auto dyn = [&](int64_t total, int64_t chunk, auto&& f) {
        std::atomic<int64_t>& a = sched_.c[ph++].v;
        for (;;) {
            const int64_t b = a.fetch_add(1, std::memory_order_relaxed) * chunk;
            if (b >= total) break;
            f(b, std::min<int64_t>(b + chunk, total));
        }
    };
    using DP = DrafterProfile;

    // ------------------------------------------------------------------------------------------------ 1. ctx K/V flush rounds
    for (int round = 0; round < R.rounds; ++round) {
        const int r0 = round * RC;
        const int nr = std::min(RC, R.src_rows - r0);
        const int64_t pos0 = R.src_pos0 + r0, end_pos = R.src_pos0 + R.src_rows;
        bool need[NL];
        for (int l = 0; l < NL; ++l) need[l] = (l == 1) || (pos0 + nr > end_pos - SWA_TOKENS);
        dyn((int64_t)nr * 4, 1, [&](int64_t b, int64_t e) {
            for (int64_t u = b; u < e; ++u) {
                if (u < (int64_t)nr * 3) {
                    const int row = (int)(u / 3), c = (int)(u % 3);
                    if (!need[c]) continue;
                    alignas(64) float mix[D];
                    const float* taps = R.src + (size_t)(r0 + row) * kTapFloats;
                    const float a0 = alpha[c][0], a1 = alpha[c][1], a2 = alpha[c][2], a3 = alpha[c][3];
                    for (int i = 0; i < D; i += 16) {
                        __m512 s = _mm512_mul_ps(_mm512_set1_ps(a0), _mm512_loadu_ps(taps + i));
                        s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_set1_ps(a1), _mm512_loadu_ps(taps + D + i)));
                        s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_set1_ps(a2), _mm512_loadu_ps(taps + 2 * D + i)));
                        s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_set1_ps(a3), _mm512_loadu_ps(taps + 3 * D + i)));
                        _mm512_storeu_ps(mix + i, s);
                    }
                    block_q8_0* q8 = Z.q8ctx + (size_t)c * RC * (D / 32);
                    rmsnorm_q8_0(mix, L[c].dflare_norm, q8 + (size_t)row * (D / 32), D, eps, Q8Mode::X86);
                    gemm_pack_act(GEMM_FMT_Q8, q8, row, row + 1, nr, D, Z.act_ctx[c]);
                } else {
                    const int row = (int)(u - (int64_t)nr * 3);
                    rope_table(pos0 + row, Z.cs_ctx + (size_t)row * 128);
                }
            }
        });
        BAR(DP::P_FLUSH);
        dyn(NL * 2 * 32, 1, [&](int64_t b, int64_t e) {
            for (int64_t u = b; u < e; ++u) {
                const int l = (int)(u / 64), kv = (int)((u / 32) & 1), ch = (int)(u % 32);
                if (!need[l]) continue;
                const Mat& W = kv ? L[l].wv_ctx : L[l].wk_ctx;
                float* y = (kv ? Z.vraw : Z.kraw) + (size_t)l * RC * KVW + ch * 32;
                const GemmAct act = gemm_act_view(GEMM_FMT_Q8, Z.act_ctx[l], nr, D);
                gemm_q8_0((const char*)W.w + (size_t)ch * 32 * W.rs, W.rs, act, 0, nr, D, y, KVW, 32);
            }
        });
        BAR(DP::P_FLUSH);
        dyn(NL * NKV, 1, [&](int64_t b, int64_t e) {
            for (int64_t u = b; u < e; ++u) {
                const int l = (int)(u / NKV), g = (int)(u % NKV);
                if (!need[l]) continue;
                for (int row = 0; row < nr; ++row) {
                    float* k = Z.kraw + ((size_t)l * RC + row) * KVW + g * HD;
                    rmsnorm(k, L[l].k_ctx_norm, k, HD, eps);
                    rope_head(k, Z.cs_ctx + (size_t)row * 128);
                }
                for (int row = 0; row < nr; ++row) {
                    const int64_t pos = pos0 + row;
                    if (l != 1 && pos < end_pos - SWA_TOKENS) continue;   // outside the window of every future query
                    kv_append(S.kv[l], pos, Z.kraw + ((size_t)l * RC + row) * KVW, Z.vraw + ((size_t)l * RC + row) * KVW, g, g + 1);
                }
            }
        });
        BAR(DP::P_FLUSH);
    }
    if (!R.rq) {
        if (tid == 0) R.phases = ph;
        return;
    }

    // ------------------------------------------------------------------------------------------------ 2. embed
    const int n = R.n;
    const int64_t p = R.p;
    if (tid == 0) {
        alignas(64) float emb[D];
        const block_q8_0* row = (const block_q8_0*)(tgt.tok_embd + (size_t)R.anchor * tgt.tok_embd_rs);
        for (int b = 0; b < D / 32; ++b) {
            const float d = fp16_to_f32(row[b].d);
            for (int j = 0; j < 32; ++j) emb[b * 32 + j] = d * (float)row[b].qs[j];
        }
        rmsnorm(emb, tgt.tok_norm, Z.X, D, eps);
    }
    for (int s = 1 + tid; s < BS; s += nth) std::memcpy(Z.X + (size_t)s * D, mask_embd, D * sizeof(float));
    for (int s = tid; s < BS; s += nth) rope_table(p + s, Z.cs_blk + (size_t)s * 128);
    BAR(DP::P_EMBED);

    // ------------------------------------------------------------------------------------------------ 3. the three layers
    const float qscale = 1.0f / std::sqrt((float)HD);
    for (int l = 0; l < NL; ++l) {
        const LayerW& W = L[l];
        const int s0 = (l == NL - 1) ? 1 : 0, s1 = (l == NL - 1) ? n + 1 : BS;   // slots that need q / o / FFN (the last layer: only the drafted ones)
        const int64_t nvalid = (l == 1) ? p : std::min<int64_t>(p, SWA_TOKENS);
        for (int s = tid; s < BS; s += nth) {
            rmsnorm_q8_0(Z.X + (size_t)s * D, W.attn_norm, Z.q8x + (size_t)s * (D / 32), D, eps, Q8Mode::X86);
            gemm_pack_act(GEMM_FMT_Q8, Z.q8x, s, s + 1, BS, D, Z.act_in);
        }
        BAR(DP::P_QKV);
        {
            const GemmAct a_in = gemm_act_view(GEMM_FMT_Q8, Z.act_in, BS, D);
            dyn(160, 1, [&](int64_t b, int64_t e) {
                for (int64_t u = b; u < e; ++u) {
                    if (u < 96)
                        gemm_q8_0((const char*)W.wq.w + (size_t)u * 32 * W.wq.rs, W.wq.rs, a_in, s0, s1, D, Z.qkv + u * 32, 5120, 32);
                    else if (u < 128)
                        gemm_q8_0((const char*)W.wk.w + (size_t)(u - 96) * 32 * W.wk.rs, W.wk.rs, a_in, 0, BS, D, Z.qkv + 3072 + (u - 96) * 32, 5120, 32);
                    else
                        gemm_q8_0((const char*)W.wv.w + (size_t)(u - 128) * 32 * W.wv.rs, W.wv.rs, a_in, 0, BS, D, Z.qkv + 4096 + (u - 128) * 32, 5120, 32);
                }
            });
        }
        BAR(DP::P_QKV);
        dyn(BS * NKV, 1, [&](int64_t b, int64_t e) {
            for (int64_t u = b; u < e; ++u) {
                const int s = (int)(u / NKV), g = (int)(u % NKV);
                float* row = Z.qkv + (size_t)s * 5120;
                const float* cs = Z.cs_blk + (size_t)s * 128;
                if (s >= s0 && s < s1)
                    for (int hh = 0; hh < 3; ++hh) {
                        float* qh = row + (3 * g + hh) * HD;
                        rmsnorm(qh, W.q_norm, qh, HD, eps);
                        rope_head(qh, cs);
                        vscale(qh, qscale, qh, HD);
                    }
                float* kh = row + 3072 + g * HD;
                rmsnorm(kh, W.k_norm, kh, HD, eps);
                rope_head(kh, cs);
            }
        });
        BAR(DP::P_QKV);
        const int nb = (int)((nvalid + 63) / 64);
        const int nsp = attn_nsplit(nb);
        const int nq = 3 * (s1 - s0);
        dyn((int64_t)NKV * nsp, 1, [&](int64_t b, int64_t e) {
            for (int64_t u = b; u < e; ++u) {
                const int g = (int)(u / nsp), sp = (int)(u % nsp);
                alignas(64) float qg[24 * HD];
                for (int s = s0; s < s1; ++s)
                    for (int hh = 0; hh < 3; ++hh) std::memcpy(qg + ((size_t)(s - s0) * 3 + hh) * HD, Z.qkv + (size_t)s * 5120 + (3 * g + hh) * HD, HD * sizeof(float));
                attn_ctx_part(S.kv[l], g, nb * sp / nsp, nb * (sp + 1) / nsp, nvalid, qg, nq, Z.part + g * NSPLIT_MAX + sp);
            }
        });
        BAR(DP::P_ATTN);
        dyn(s1 - s0, 1, [&](int64_t b, int64_t e) {
            for (int64_t ui = b; ui < e; ++ui) {
                const int s = s0 + (int)ui;
                float* orow = Z.attn_out + (size_t)s * QW;
                for (int g = 0; g < NKV; ++g)
                    for (int hh = 0; hh < 3; ++hh)
                        attn_finish_query(Z.part + g * NSPLIT_MAX, nsp, (s - s0) * 3 + hh, Z.qkv + (size_t)s * 5120 + (3 * g + hh) * HD, Z.qkv + 3072 + g * HD,
                                          Z.qkv + 4096 + g * HD, 5120, BS, orow + (3 * g + hh) * HD);
                quantize_q8_0(orow, Z.q8o + (size_t)s * (QW / 32), QW, Q8Mode::X86);
                gemm_pack_act(GEMM_FMT_Q8, Z.q8o, s, s + 1, BS, QW, Z.act_o);
            }
        });
        BAR(DP::P_ATTN);
        {
            const GemmAct a_o = gemm_act_view(GEMM_FMT_Q8, Z.act_o, BS, QW);
            dyn(D / 32, 1, [&](int64_t b, int64_t e) {
                for (int64_t u = b; u < e; ++u) {
                    gemm_q8_0((const char*)W.wo.w + (size_t)u * 32 * W.wo.rs, W.wo.rs, a_o, s0, s1, QW, Z.dn_y + u * 32, D, 32);
                    for (int s = s0; s < s1; ++s) vadd(Z.X + (size_t)s * D + u * 32, Z.dn_y + (size_t)s * D + u * 32, Z.X + (size_t)s * D + u * 32, 32);
                }
            });
        }
        BAR(DP::P_WO);
        for (int s = s0 + tid; s < s1; s += nth) {
            rmsnorm_q8_0(Z.X + (size_t)s * D, W.ffn_norm, Z.q8x + (size_t)s * (D / 32), D, eps, Q8Mode::X86);
            gemm_pack_act(GEMM_FMT_Q8, Z.q8x, s, s + 1, BS, D, Z.act_in);
        }
        BAR(DP::P_FFN);
        {
            const GemmAct a_in = gemm_act_view(GEMM_FMT_Q8, Z.act_in, BS, D);
            dyn(FF / 64, 1, [&](int64_t b, int64_t e) {
                for (int64_t u = b; u < e; ++u) {
                    const size_t r0 = (size_t)u * 64;
                    gemm_q8_0((const char*)W.wgate.w + r0 * W.wgate.rs, W.wgate.rs, a_in, s0, s1, D, Z.gu_g + r0, FF, 64);
                    gemm_q8_0((const char*)W.wup.w + r0 * W.wup.rs, W.wup.rs, a_in, s0, s1, D, Z.gu_u + r0, FF, 64);
                    const int half = r0 >= (size_t)FF / 2 ? 1 : 0;
                    const size_t blk0 = (r0 - (size_t)half * (FF / 2)) / 32;
                    for (int s = s0; s < s1; ++s) {
                        alignas(64) float x[64];
                        for (int i = 0; i < 64; i += 16) {
                            const __m512 gg = _mm512_loadu_ps(Z.gu_g + (size_t)s * FF + r0 + i), uu = _mm512_loadu_ps(Z.gu_u + (size_t)s * FF + r0 + i);
                            _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_mul_ps(gg, simd::sigmoid16(gg)), uu));
                        }
                        quantize_q8_0(x, Z.ffq + ((size_t)half * BS + s) * (FF / 64) + blk0, 64, Q8Mode::X86);
                    }
                }
            });
        }
        BAR(DP::P_FFN);
        for (int it = tid; it < 2 * (s1 - s0); it += nth) {
            const int half = it & 1, s = s0 + (it >> 1);
            gemm_pack_act(GEMM_FMT_Q8, Z.ffq + (size_t)half * BS * (FF / 64), s, s + 1, BS, FF / 2, Z.act_dn[half]);
        }
        BAR(DP::P_FFN);
        {
            const GemmAct a0 = gemm_act_view(GEMM_FMT_Q8, Z.act_dn[0], BS, FF / 2), a1 = gemm_act_view(GEMM_FMT_Q8, Z.act_dn[1], BS, FF / 2);
            const size_t half_bytes = (size_t)(FF / 2 / 32) * sizeof(block_q8_0);
            dyn(D / 32, 1, [&](int64_t b, int64_t e) {
                for (int64_t u = b; u < e; ++u) {
                    const char* w = (const char*)W.wdown.w + (size_t)u * 32 * W.wdown.rs;
                    gemm_q8_0(w, W.wdown.rs, a0, s0, s1, FF / 2, Z.dn_y + u * 32, D, 32);
                    gemm_q8_0(w + half_bytes, W.wdown.rs, a1, s0, s1, FF / 2, Z.dn_y + (size_t)BS * D + u * 32, D, 32);
                    for (int s = s0; s < s1; ++s) {
                        float* x = Z.X + (size_t)s * D + u * 32;
                        for (int i = 0; i < 32; i += 16)
                            _mm512_storeu_ps(x + i, _mm512_add_ps(_mm512_loadu_ps(x + i), _mm512_add_ps(_mm512_loadu_ps(Z.dn_y + (size_t)s * D + u * 32 + i),
                                                                                                          _mm512_loadu_ps(Z.dn_y + (size_t)(BS + s) * D + u * 32 + i))));
                    }
                }
            });
        }
        BAR(DP::P_FFN);
        if (R.dbg && tid == 0) {
            dbg.x[l].assign(Z.X, Z.X + (size_t)BS * D);
        }
    }

    // ------------------------------------------------------------------------------------------------ 4. final norm, base logits, joint parts
    for (int s = 1 + tid; s <= n; s += nth) {
        rmsnorm(Z.X + (size_t)s * D, out_norm, Z.hidf + (size_t)s * D, D, eps);
        quantize_q8_0(Z.hidf + (size_t)s * D, Z.q8x + (size_t)s * (D / 32), D, Q8Mode::X86);
        gemm_pack_act(GEMM_FMT_Q8, Z.q8x, s, s + 1, BS, D, Z.act_in);
    }
    BAR(DP::P_FINAL);
    {
        const GemmAct a_lm = gemm_act_view(GEMM_FMT_Q8, Z.act_in, BS, D);
        const int nv = Z.nv;
        const int nch64 = (nv + 63) / 64;
        dyn(JO / 16 + nch64, 1, [&](int64_t b, int64_t e) {
            for (int64_t u = b; u < e; ++u) {
                if (u < JO / 16) {
                    const int r0 = (int)u * 16;
                    gemv_bf16((const char*)joint_w + ((size_t)r0 * JW + 2816) * 2, (size_t)JW * 2, S.f_rnn, D, Z.Pf + r0, 16);
                    for (int i = 0; i < 16; ++i) Z.Pf[r0 + i] += joint_b[r0 + i];
                    gemm_bf16((const char*)joint_w + ((size_t)r0 * JW + 512) * 2, (size_t)JW * 2, Z.hidf + D, D, n, D, Z.Pj + JO + r0, JO, 16);
                    for (int t = 0; t < n; ++t)
                        for (int i = 0; i < 16; ++i) Z.Pj[(size_t)(1 + t) * JO + r0 + i] += Z.Pf[r0 + i];
                } else {
                    const int64_t r0 = (u - JO / 16) * 64;
                    const int rows = (int)std::min<int64_t>(64, nv - r0);
                    gemm_q8_0((const char*)tgt.lm_head.w + (size_t)r0 * tgt.lm_head.rs, tgt.lm_head.rs, a_lm, 1, n + 1, D, Z.base + r0, Z.nvp, rows);
                }
            }
        });
    }
    BAR(DP::P_BASE);
    if (R.dbg && tid == 0) {
        dbg.hid.assign(Z.hidf, Z.hidf + (size_t)BS * D);
        dbg.base.assign((size_t)7 * Z.nv, 0.0f);
        for (int j = 1; j <= n; ++j) std::memcpy(dbg.base.data() + (size_t)(j - 1) * Z.nv, Z.base + (size_t)j * Z.nvp, (size_t)Z.nv * 4);
        dbg.lat.assign((size_t)7 * RANK, 0.0f);
        dbg.logits.assign((size_t)7 * Z.nv, 0.0f);
    }

    // ------------------------------------------------------------------------------------------------ 5. the Markov / RNN chain
    Scr::Thr& T = Z.thr[tid];
    std::memset(T.s, 0, sizeof(T.s));
    int32_t prev = R.anchor;
    const int nv = Z.nv, nvp = Z.nvp, nchunks = Z.nchunks;
    for (int j = 1; j <= n; ++j) {
        std::memcpy(T.x512, T.s, RANK * sizeof(float));
        bf16_to_f32(markov_w1 + (size_t)prev * RANK, T.x512 + RANK, RANK);
        dyn(JO / 16, 1, [&](int64_t b, int64_t e) {
            for (int64_t u = b; u < e; ++u)
                gemv_bf16((const char*)joint_w + (size_t)u * 16 * JW * 2, (size_t)JW * 2, T.x512, 512, Z.upart + u * 16, 16);
        });
        BAR(DP::P_CHAIN_U);
        for (int i = 0; i < JO; i += 16) _mm512_storeu_ps(T.u + i, _mm512_add_ps(_mm512_loadu_ps(Z.upart + i), _mm512_loadu_ps(Z.Pj + (size_t)j * JO + i)));
        {
            alignas(64) float gate[RANK], cand[RANK];
            vsigmoid(T.u, gate, RANK);
            vtanh(T.u + RANK, cand, RANK);
            for (int i = 0; i < RANK; i += 16) {
                const __m512 c = _mm512_loadu_ps(cand + i), s = _mm512_loadu_ps(T.s + i), gt = _mm512_loadu_ps(gate + i);
                _mm512_storeu_ps(T.s + i, _mm512_add_ps(c, _mm512_mul_ps(gt, _mm512_sub_ps(s, c))));
            }
            vtanh(T.u + 2 * RANK, T.lat, RANK);
            if (fold.q8) quantize_q8_0(T.lat, T.latq, RANK, Q8Mode::X86);
        }
        if (R.dbg && tid == 0) std::memcpy(dbg.lat.data() + (size_t)(j - 1) * RANK, T.lat, RANK * sizeof(float));
        ChunkRes* cres = Z.cres + (size_t)(j - 1) * nchunks;
        float* qrow = R.sample ? (R.q_out == Z.qinternal ? Z.qinternal + (size_t)(j - 1) * nvp : R.q_out + (size_t)(j - 1) * nv) : nullptr;
        dyn(nchunks, 1, [&](int64_t b, int64_t e) {
            for (int64_t ch = b; ch < e; ++ch) {
                const int64_t r0 = ch * CH_ROWS;
                const int rows = (int)std::min<int64_t>(CH_ROWS, nv - r0);
                float* z = T.z;
                if (fold.q8)
                    gemv_q8_0((const char*)fold.data + (size_t)r0 * fold.rs, fold.rs, T.latq, RANK, z, rows);
                else
                    gemv_bf16((const char*)fold.data + (size_t)r0 * fold.rs, fold.rs, T.lat, RANK, z, rows);
                const float* bj = Z.base + (size_t)j * nvp + r0;
                const float* m0 = fold.m0 + r0;
                int i = 0;
                for (; i + 16 <= rows; i += 16)
                    _mm512_storeu_ps(z + i, _mm512_add_ps(_mm512_add_ps(_mm512_loadu_ps(z + i), _mm512_loadu_ps(bj + i)), _mm512_loadu_ps(m0 + i)));
                for (; i < rows; ++i) z[i] = (z[i] + bj[i]) + m0[i];
                vsoftcap(z, z, rows, tgt.softcap);
                if (R.dbg) std::memcpy(dbg.logits.data() + (size_t)(j - 1) * nv + r0, z, (size_t)rows * 4);
                if (!R.sample) {
                    float best = z[0];
                    int bi = 0;
                    for (int k = 1; k < rows; ++k)
                        if (z[k] > best) { best = z[k]; bi = k; }
                    cres[ch].val = best;
                    cres[ch].idx = (int32_t)(r0 + bi);
                } else {
                    float mx = z[0] * R.inv_t;
                    for (int k = 0; k < rows; ++k) { z[k] *= R.inv_t; mx = std::max(mx, z[k]); }
                    for (int k = 0; k < rows; ++k) T.tmp[k] = z[k] - mx;
                    vexp(T.tmp, qrow + r0, (size_t)rows);
                    float sum = 0;
                    for (int k = 0; k < rows; ++k) sum += qrow[r0 + k];
                    cres[ch].val = mx;
                    cres[ch].sum = sum;
                    cres[ch].idx = (int32_t)r0;
                }
            }
        });
        BAR(DP::P_CHAIN_M);
        int32_t d = 0;
        if (!R.sample) {
            float best = cres[0].val;
            d = cres[0].idx;
            for (int ch = 1; ch < nchunks; ++ch)
                if (cres[ch].val > best) { best = cres[ch].val; d = cres[ch].idx; }
        } else {
            double M = 0, Zs = 0;
            d = sample_select(cres, nchunks, qrow, nv, R.seed, j, &M, &Zs);
            if (tid == 0) { Z.mj[j] = M; Z.zj[j] = Zs; }
        }
        if (dbg.force_on) d = dbg.force[j - 1];
        if (tid == 0) R.out_tokens[j - 1] = d;
        prev = d;
    }
    if (R.sample) {
        BAR(DP::P_POST);   // everyone has finished reading the unnormalised rows
        const bool internal = R.q_out == Z.qinternal;
        dyn((int64_t)n * nchunks, 4, [&](int64_t b, int64_t e) {
            for (int64_t u = b; u < e; ++u) {
                const int j = (int)(u / nchunks) + 1, ch = (int)(u % nchunks);
                const ChunkRes& c = Z.cres[(size_t)(j - 1) * nchunks + ch];
                const float f = (float)(std::exp((double)c.val - Z.mj[j]) / Z.zj[j]);
                const int64_t r0 = (int64_t)ch * CH_ROWS;
                const int rows = (int)std::min<int64_t>(CH_ROWS, nv - r0);
                float* q = (internal ? Z.qinternal + (size_t)(j - 1) * nvp : R.q_out + (size_t)(j - 1) * nv) + r0;
                vscale(q, f, q, (size_t)rows);
            }
        });
        mark(DP::P_POST);
    }
    if (tid == 0) R.phases = ph;
}

}  // namespace dr

// ------------------------------------------------------------------------------------------------------------------------------ public factory
std::unique_ptr<Drafter> Drafter::create(Engine& target, const std::string& drafter_gguf, std::string* err) {
    return create(target, drafter_gguf, DrafterOptions(), err);
}

std::unique_ptr<Drafter> Drafter::create(Engine& E, const std::string& drafter_gguf, const DrafterOptions& opt, std::string* err) {
    dr::DrafterTarget t;
    t.pool = E.pool.get();
    const std::vector<int> cpus = E.opt.cpus.empty() ? engine_default_cpus(E.opt.n_threads) : E.opt.cpus;
    t.cpu0 = cpus.empty() ? -1 : cpus[0];
    t.lm_head = E.lm_head;
    t.tok_embd = E.tok_embd;
    t.tok_embd_rs = E.tok_embd_rs;
    t.tok_norm = E.tok_norm;
    t.n_vocab = E.cfg.n_vocab;
    t.n_vocab_padded = E.cfg.n_vocab_padded;
    t.rms_eps = E.cfg.rms_eps;
    t.softcap = E.cfg.final_logit_softcap;
    t.max_ctx = E.opt.max_ctx;
    t.n_slots = E.opt.n_slots;
    std::unique_ptr<dr::DrafterImpl> d = dr::DrafterImpl::make(t, drafter_gguf, opt, err);
    return std::unique_ptr<Drafter>(d.release());
}

}  // namespace ink

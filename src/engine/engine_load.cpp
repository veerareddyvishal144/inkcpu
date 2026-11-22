// Engine::create (weight packing into hugepage memory), slots, profile report. Owner: agent C1. See docs/ENGINE_DESIGN.md.
#include <sys/mman.h>
#include <unistd.h>
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "engine_priv.h"
#ifdef INK_HAVE_PREFILL
#include "ink/engine_prefill.h"
#endif

namespace ink {

const char* const kProfNames[PH_COUNT] = {"embed",
                                          "attn-in mix+norm (serial)",
                                          "loop controllers",
                                          "proj q,k,v,gate,rel,sink",
                                          "conv/clip/kv + attention",
                                          "wo + post-conv + residual",
                                          "ffn mix+norm (serial)",
                                          "router matvec",
                                          "experts gate_up + SiTU",
                                          "experts down + post-conv",
                                          "final mix+norm",
                                          "lm_head + softcap"};

std::vector<int> engine_default_cpus(int n) {
    std::vector<int> c;
    if (const char* e = std::getenv("INK_CPUS")) {   // explicit CPU list (e.g. to stay off the CPUs other tenants pin to), wraps around
        std::vector<int> l;
        for (const char* p = e; *p;) {
            l.push_back(std::atoi(p));
            while (*p && *p != ',') ++p;
            if (*p == ',') ++p;
        }
        if (!l.empty()) {
            for (int i = 0; i < n; ++i) c.push_back(l[(size_t)i % l.size()]);
            return c;
        }
    }
    const int ncpu = std::max(2, (int)std::thread::hardware_concurrency());
    if (n <= 15) {
        for (int i = 0; i < n; ++i) c.push_back((2 * (i + 1)) % ncpu);
    } else {
        for (int i = 1; i <= 15; ++i) c.push_back((2 * i) % ncpu);
        c.push_back(0);                               // physical core 0 last: it runs at half speed on the dev host
        for (int i = 0; (int)c.size() < n; ++i) c.push_back((2 * i + 1) % ncpu);
    }
    return c;
}

namespace {

size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

bool f32_is_bf16_exact(const TensorRef& t) {
    const uint32_t* p = (const uint32_t*)t.data;
    const size_t n = t.nbytes / 4;
    for (size_t i = 0; i < n; ++i)
        if (p[i] & 0xFFFFu) return false;
    return true;
}

double calibrate_tsc_per_us() {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const uint64_t c0 = __rdtsc();
    while (clk::now() - t0 < std::chrono::milliseconds(3)) {
    }
    const uint64_t c1 = __rdtsc();
    const double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    return (double)(c1 - c0) / us;
}

}  // namespace

std::unique_ptr<Engine> Engine::create(const std::string& path, const EngineOptions& o, std::string* err) {
    using clk = std::chrono::steady_clock;
    const auto t_start = clk::now();
    auto fail = [&](const std::string& m) -> std::unique_ptr<Engine> {
        if (err) *err = m;
        return nullptr;
    };
    std::unique_ptr<Engine> e(new Engine());
    e->opt = o;
    if (e->opt.n_threads < 1) e->opt.n_threads = 1;
    if (e->opt.n_slots < 1) e->opt.n_slots = 1;
    if (e->opt.max_ctx < 64) e->opt.max_ctx = 64;
    const char* pe = std::getenv("INK_PROFILE");
    e->prof_on = o.profile || (pe && std::atoi(pe) != 0);
    auto envi = [](const char* n, int d) { const char* v = std::getenv(n); return v && *v ? std::atoi(v) : d; };
    e->tune.chunk_proj = envi("INK_CHUNK_PROJ", e->tune.chunk_proj);
    e->tune.chunk_wo = envi("INK_CHUNK_WO", e->tune.chunk_wo);
    e->tune.chunk_down = envi("INK_CHUNK_DOWN", e->tune.chunk_down);
    e->tune.chunk_lm = envi("INK_CHUNK_LM", e->tune.chunk_lm);
    e->tune.chunk_router = envi("INK_CHUNK_ROUTER", e->tune.chunk_router);
    e->tune.chunk_ctrl = envi("INK_CHUNK_CTRL", e->tune.chunk_ctrl);
    e->tune.split_keys = envi("INK_SPLIT_KEYS", e->tune.split_keys);
    e->tune.swa_blocks = std::max(16, envi("INK_SWA_BLOCKS", e->tune.swa_blocks));
    g_gemv_prefetch_bytes = envi("INK_PF", g_gemv_prefetch_bytes);     // B1 tunables (experiments)
    g_gemv_prefetch_hint = envi("INK_PFH", g_gemv_prefetch_hint);
    e->prof.tsc_per_us = calibrate_tsc_per_us();

    std::string gerr;
    e->gguf = GgufFile::open(path, &gerr);
    if (!e->gguf) return fail("gguf: " + gerr);
    if (!ModelConfig::from_gguf(*e->gguf, &e->cfg, &gerr)) return fail("config: " + gerr);
    Weights w;
    if (!load_weights(*e->gguf, e->cfg, &w, &gerr)) return fail("weights: " + gerr);
    e->plan = visit_plan(e->cfg);
    const ModelConfig& c = e->cfg;
    if (c.n_layer != 24 || c.n_embd != kD || c.n_head != kNQHeads || c.n_head_kv != kNKvHeads || c.head_dim != kHeadDim ||
        c.n_ff_exp != kFF || c.n_expert != kNExp || c.n_expert_used != kTopK || c.n_conv != kConvTaps || c.n_rel != kRelFeat ||
        c.rel_extent != kRelExtent || c.sliding_window != kSwaWindow || c.loop_gate_rank != kCtrlRank || c.loop_count != 3 ||
        c.loop_start != 6 || c.loop_end != 18 || c.attn_res_block_size != 6 || (int)e->plan.size() != kVisits)
        return fail("unsupported model geometry (the executor is specialised for Inkling 2 Mini)");

    e->pool.reset(new ThreadPool(e->opt.n_threads, e->opt.cpus.empty() ? engine_default_cpus(e->opt.n_threads) : e->opt.cpus));

    // ---------------------------------------------------------------- plan the arena (big streamed tensors, layer order)
    struct Pend {
        const void** slot;
        const uint8_t* src;
        size_t bytes;
    };
    std::vector<Pend> pend;
    auto big = [&](const void** slot, const TensorRef& t, size_t bytes = 0) { pend.push_back({slot, t.data, bytes ? bytes : t.nbytes}); };
    auto small = [&](size_t bytes) -> void* {
        void* p = std::aligned_alloc(64, align_up(std::max<size_t>(bytes, 64), 64));
        e->small_.push_back(p);
        return p;
    };
    std::string bad;
    auto need = [&](const TensorRef& t, int type, const char* name) {
        if (!t.valid() || t.type != type) {
            if (bad.empty()) bad = std::string("tensor ") + name + " missing or unexpected type";
            return false;
        }
        return true;
    };
    auto f32copy = [&](const TensorRef& t, const char* name) -> const float* {
        if (!need(t, T_F32, name)) return nullptr;
        float* p = (float*)small(t.nbytes);
        std::memcpy(p, t.data, t.nbytes);
        return p;
    };
    auto q8mat = [&](Mat& m, const TensorRef& t, const char* name) {
        if (!need(t, T_Q8_0, name)) return;
        m.type = T_Q8_0;
        m.k = (int)t.ne[0];
        m.rows = (int)t.ne[1];
        m.rs = t.row_bytes();
        big(&m.w, t);
    };
    auto bf16mat = [&](Mat& m, const TensorRef& t, const char* name) {
        if (!need(t, T_BF16, name)) return;
        m.type = T_BF16;
        m.k = (int)t.ne[0];
        m.rows = (int)t.ne[1];
        m.rs = (size_t)m.k * 2;
        void* p = small(t.nbytes);
        std::memcpy(p, t.data, t.nbytes);
        m.w = p;
    };
    // F32 matrix (router, controllers): BF16 when EVERY value is exactly representable (checked), else an F32 copy
    auto lowp = [&](Mat& m, const TensorRef& t, bool as_bf16, const char* name) {
        if (!need(t, T_F32, name)) return;
        m.k = (int)t.ne[0];
        m.rows = (int)t.ne[1];
        const size_t n = t.nbytes / 4;
        if (as_bf16) {
            uint16_t* p = (uint16_t*)small(n * 2);
            const uint32_t* s = (const uint32_t*)t.data;
            for (size_t i = 0; i < n; ++i) p[i] = (uint16_t)(s[i] >> 16);
            m.type = T_BF16;
            m.rs = (size_t)m.k * 2;
            m.w = p;
        } else {
            float* p = (float*)small(t.nbytes);
            std::memcpy(p, t.data, t.nbytes);
            m.type = T_F32;
            m.rs = (size_t)m.k * 4;
            m.w = p;
        }
    };
    bool router_bf16 = true, ctrl_bf16 = true;
    for (int i = 0; i < c.n_layer; ++i) {
        const LayerWeights& lw = w.layers[i];
        if (lw.ffn_gate_inp.valid() && !f32_is_bf16_exact(lw.ffn_gate_inp)) router_bf16 = false;
        if (lw.in_core)
            for (int j = 0; j < 4; ++j)
                for (const TensorRef* t : {&lw.ctrl[j].down, &lw.ctrl[j].gain, &lw.ctrl[j].forget})
                    if (t->valid() && !f32_is_bf16_exact(*t)) ctrl_bf16 = false;
    }

    e->layers.assign(c.n_layer, LayerPacked());
    for (int i = 0; i < c.n_layer; ++i) {
        const LayerWeights& lw = w.layers[i];
        LayerPacked& p = e->layers[i];
        p.phys = i;
        p.swa = lw.is_swa;
        p.global = !lw.is_swa;
        p.core = lw.in_core;
        q8mat(p.wq, lw.wq, "attn_q");
        q8mat(p.wk, lw.wk, "attn_k");
        q8mat(p.wv, lw.wv, "attn_v");
        q8mat(p.wgate, lw.attn_gate, "attn_gate");
        q8mat(p.wrel, lw.attn_rel, "attn_rel");
        q8mat(p.wo, lw.wo, "attn_output");
        bf16mat(p.sink, lw.attn_sink_proj, "attn_sink_proj");
        if (p.global) {
            bf16mat(p.kproj, lw.attn_k_conv_proj, "attn_k_conv_proj");
            bf16mat(p.vproj, lw.attn_v_conv_proj, "attn_v_conv_proj");
            p.inp_norm[0] = f32copy(lw.attn_q_inp_norm, "attn_q_inp_norm");
            p.inp_norm[1] = f32copy(lw.attn_k_inp_norm, "attn_k_inp_norm");
            p.inp_norm[2] = f32copy(lw.attn_v_inp_norm, "attn_v_inp_norm");
        } else {
            p.attn_norm = f32copy(lw.attn_norm, "attn_norm");
        }
        p.q_clip = f32copy(lw.attn_q_clip, "attn_q_clip");
        p.rel_clip = f32copy(lw.attn_rel_clip, "attn_rel_clip");
        if (need(lw.attn_rel_bias, T_F32, "attn_rel_bias")) {
            float* tab = (float*)small((size_t)kRelFeat * kRelTableLen * 4);
            attn_repack_rel_bias(lw.attn_rel_bias.f32(), tab);
            p.rel_table = tab;
        }
        auto conv = [&](const float*& dst, const TensorRef& t, size_t C, const char* name) {
            if (!need(t, T_F32, name)) return;
            float* soa = (float*)small(4 * C * 4);
            sconv_repack_static(t.f32(), soa, C);
            dst = soa;
        };
        conv(p.kconv, lw.attn_k_conv, kKVW, "attn_k_conv");
        conv(p.vconv, lw.attn_v_conv, kKVW, "attn_v_conv");
        conv(p.aconv, lw.attn_post_conv, kD, "attn_post_conv");
        conv(p.fconv, lw.ffn_post_conv, kD, "ffn_post_conv");
        if (i > 0) p.attn_res_query = f32copy(lw.attn_res_query, "attn_res_query");
        p.ffn_res_query = f32copy(lw.ffn_res_query, "ffn_res_query");
        p.ffn_norm = f32copy(lw.ffn_norm, "ffn_norm");
        lowp(p.router, lw.ffn_gate_inp, router_bf16, "ffn_gate_inp");
        p.exp_bias = f32copy(lw.exp_probs_b, "exp_probs_b");
        if (lw.ffn_gate_up_exps.valid() && lw.ffn_down_exps.valid() && lw.ffn_gate_up_exps.type == lw.ffn_down_exps.type &&
            (lw.ffn_gate_up_exps.type == T_NVFP4 || lw.ffn_gate_up_exps.type == T_Q8_0)) {
            p.expert_type = lw.ffn_gate_up_exps.type;
            p.gu_rs = lw.ffn_gate_up_exps.row_bytes();
            p.gu_eb = p.gu_rs * (size_t)lw.ffn_gate_up_exps.ne[1];
            p.dn_rs = lw.ffn_down_exps.row_bytes();
            p.dn_eb = p.dn_rs * (size_t)lw.ffn_down_exps.ne[1];
            big(&p.gu, lw.ffn_gate_up_exps);
            big(&p.dn, lw.ffn_down_exps);
            if (lw.ffn_gate_up_exps.ne[1] != 2 * kFF || lw.ffn_down_exps.ne[1] != kD || lw.ffn_gate_up_exps.ne[0] != kD ||
                lw.ffn_down_exps.ne[0] != kFF)
                bad = "expert tensor shape";
            if (p.expert_type == T_NVFP4) {
                p.gu_scale = f32copy(lw.ffn_gate_up_scale, "ffn_gate_up_exps.scale");
                p.dn_scale = f32copy(lw.ffn_down_scale, "ffn_down_exps.scale");
            }
        } else if (bad.empty()) {
            bad = "expert tensors missing or unsupported type in layer " + std::to_string(i);
        }
        if (lw.in_core) {
            p.msg_query = f32copy(lw.loop_msg_query, "loop_msg_query");
            for (int j = 0; j < 4; ++j) {
                lowp(p.ctrl_down[j], lw.ctrl[j].down, ctrl_bf16, "ctrl_down");
                lowp(p.ctrl_gain[j], lw.ctrl[j].gain, ctrl_bf16, "ctrl_gain");
                lowp(p.ctrl_forget[j], lw.ctrl[j].forget, ctrl_bf16, "ctrl_forget");
            }
        }
    }
    e->tok_norm = f32copy(w.tok_norm, "token_embd_norm");
    e->out_norm = f32copy(w.output_norm, "output_norm");
    e->out_res_query = f32copy(w.output_res_query, "output_res_query");
    if (!need(w.tok_embd, T_Q8_0, "token_embd") || !need(w.output, T_Q8_0, "output")) {
    } else {
        e->tok_embd = w.tok_embd.data;
        e->tok_embd_rs = w.tok_embd.row_bytes();
        e->lm_head.type = T_Q8_0;
        e->lm_head.k = (int)w.output.ne[0];
        e->lm_head.rows = c.n_vocab;
        e->lm_head.rs = w.output.row_bytes();
        big(&e->lm_head.w, w.output, (size_t)c.n_vocab * e->lm_head.rs);
    }
    if (!bad.empty()) return fail(bad);

    // ---------------------------------------------------------------- allocate the arena and copy in parallel
    std::vector<size_t> offs;
    size_t total = 0;
    for (const Pend& p : pend) {
        offs.push_back(total);
        total += align_up(p.bytes, 4096);
    }
    e->arena_ = alloc_weight_buffer(total, true);
    if (!e->arena_) return fail("cannot allocate " + std::to_string(total >> 20) + " MiB for the packed weights");
    e->weight_bytes = total;
    struct Piece {
        uint8_t* dst;
        const uint8_t* src;
        size_t n;
    };
    std::vector<Piece> pieces;
    const size_t kPiece = (size_t)4 << 20;
    for (size_t i = 0; i < pend.size(); ++i) {
        uint8_t* dst = (uint8_t*)e->arena_ + offs[i];
        *pend[i].slot = dst;
        for (size_t o = 0; o < pend[i].bytes; o += kPiece) pieces.push_back({dst + o, pend[i].src + o, std::min(kPiece, pend[i].bytes - o)});
    }
    std::atomic<size_t> next{0};
    e->pool->run([&](int tid, int nth) {
        for (;;) {
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= pieces.size()) break;
            std::memcpy(pieces[i].dst, pieces[i].src, pieces[i].n);
            // drop the copied source pages right away (whole pages inside the piece; the edges go with the file-wide call below), so the mapping and the arena never
            // coexist in full: peak RSS ~ arena + one piece per thread instead of 2x the weights (docs/requests.md, V item 2)
            const uintptr_t a = ((uintptr_t)pieces[i].src + 4095) & ~(uintptr_t)4095, b = ((uintptr_t)pieces[i].src + pieces[i].n) & ~(uintptr_t)4095;
            if (b > a) madvise((void*)a, (size_t)(b - a), MADV_DONTNEED);
        }
    });
    // the page cache keeps the file; drop this process's page-table references to it (only token_embd rows are read later)
    madvise(const_cast<uint8_t*>(e->gguf->base()), e->gguf->file_size(), MADV_DONTNEED);

    // ---------------------------------------------------------------- scratch, counters, slots
    e->scr_ = (Scratch*)std::aligned_alloc(64, align_up(sizeof(Scratch), 64));
    std::memset((void*)e->scr_, 0, sizeof(Scratch));
    e->ctr_ = (PhaseCounter*)std::aligned_alloc(64, sizeof(PhaseCounter) * kMaxPhases);
    std::memset((void*)e->ctr_, 0, sizeof(PhaseCounter) * kMaxPhases);
    e->slots.resize(e->opt.n_slots);
    for (auto& s : e->slots) {
        std::string serr;
        if (!e->alloc_slot(s, &serr)) return fail("slot allocation: " + serr);
    }
    e->load_seconds = std::chrono::duration<double>(clk::now() - t_start).count();
    return e;
}

Engine::~Engine() {
#ifdef INK_HAVE_PREFILL
    engine_prefill_release(*this);   // the scratch of the batched executor lives in a registry keyed by Engine*
#endif
    pool.reset();
    for (auto& s : slots) free_slot(s);
    if (arena_) free_weight_buffer(arena_);
    for (void* p : small_) std::free(p);
    std::free((void*)scr_);
    std::free((void*)ctr_);
}

void Engine::set_threads(int n) {
    opt.n_threads = std::max(1, n);
    pool.reset();
    pool.reset(new ThreadPool(opt.n_threads, engine_default_cpus(opt.n_threads)));
    pinned_thread_ = std::thread::id();
}

// ---------------------------------------------------------------------------------------------------- slots
bool Engine::alloc_slot(SlotState& s, std::string* err) {
    const size_t blk_bytes = kKvBlockElems * 2;
    const size_t cap_blocks = ((size_t)opt.max_ctx + kKvBlock - 1) / kKvBlock + 1;
    for (int v = 0; v < kVisits; ++v) {
        const bool swa = layers[plan[v].phys].swa;
        const size_t nb = swa ? (size_t)tune.swa_blocks : cap_blocks;
        const size_t bytes = nb * blk_bytes;
        void* k = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        void* vv = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (k == MAP_FAILED || vv == MAP_FAILED) {
            if (err) *err = "mmap failed for the KV cache";
            return false;
        }
        s.kv[v].k = (uint16_t*)k;
        s.kv[v].v = (uint16_t*)vv;
        s.kv[v].ring = swa;
        s.kv[v].n_blocks = swa ? (uint32_t)tune.swa_blocks : 1;
        s.kv_map_bytes[v] = bytes;
    }
    s.hist = (float*)std::aligned_alloc(64, align_up(kVisits * kHistPerVisit * 4, 64));
    std::memset(s.hist, 0, kVisits * kHistPerVisit * 4);
    s.pos = 0;
    s.used = false;
    return true;
}

void Engine::free_slot(SlotState& s) {
    for (int v = 0; v < kVisits; ++v) {
        if (s.kv[v].k) munmap(s.kv[v].k, s.kv_map_bytes[v]);
        if (s.kv[v].v) munmap(s.kv[v].v, s.kv_map_bytes[v]);
        s.kv[v].k = s.kv[v].v = nullptr;
    }
    std::free(s.hist);
    s.hist = nullptr;
}

// blocks of a visit that may hold data: global = grown extent, ring = written blocks (all once the ring has wrapped)
static size_t used_blocks(const SlotState& s, int v) {
    const KvLayer& L = s.kv[v];
    if (!L.ring) return L.n_blocks;
    return std::min<size_t>(L.n_blocks, (size_t)((s.pos + kKvBlock - 1) / kKvBlock));
}

void Engine::slot_reset(int slot) {
    if (slot < 0 || slot >= (int)slots.size()) return;
    SlotState& s = slots[slot];
    const size_t blk_bytes = kKvBlockElems * 2;
    for (int v = 0; v < kVisits; ++v) {
        const size_t used = used_blocks(s, v);
        std::memset(s.kv[v].k, 0, used * blk_bytes);
        std::memset(s.kv[v].v, 0, used * blk_bytes);
        if (!s.kv[v].ring) s.kv[v].n_blocks = 1;
    }
    std::memset(s.hist, 0, kVisits * kHistPerVisit * 4);
    s.pos = 0;
}

int Engine::slot_new() {
    for (size_t i = 0; i < slots.size(); ++i)
        if (!slots[i].used) {
            slot_reset((int)i);
            slots[i].used = true;
            return (int)i;
        }
    return -1;
}

void Engine::slot_free(int slot) {
    if (slot >= 0 && slot < (int)slots.size()) slots[slot].used = false;
}

int64_t Engine::slot_pos(int slot) const { return (slot >= 0 && slot < (int)slots.size()) ? slots[slot].pos : -1; }

bool Engine::slot_copy(int src, int dst) {
    if (src < 0 || dst < 0 || src >= (int)slots.size() || dst >= (int)slots.size() || src == dst) return false;
    SlotState& a = slots[src];
    SlotState& b = slots[dst];
    const size_t blk_bytes = kKvBlockElems * 2;
    for (int v = 0; v < kVisits; ++v) {
        const size_t na = used_blocks(a, v), nb_used = used_blocks(b, v);
        if (!a.kv[v].ring && na > b.kv_map_bytes[v] / blk_bytes) return false;
        if (nb_used > na) {   // clear what the destination used beyond the source's extent
            std::memset(b.kv[v].k + na * kKvBlockElems, 0, (nb_used - na) * blk_bytes);
            std::memset(b.kv[v].v + na * kKvBlockElems, 0, (nb_used - na) * blk_bytes);
        }
        if (!a.kv[v].ring) b.kv[v].n_blocks = a.kv[v].n_blocks;
        std::memcpy(b.kv[v].k, a.kv[v].k, na * blk_bytes);
        std::memcpy(b.kv[v].v, a.kv[v].v, na * blk_bytes);
    }
    std::memcpy(b.hist, a.hist, kVisits * kHistPerVisit * 4);
    b.pos = a.pos;
    return true;
}

// ---------------------------------------------------------------------------------------------------- profile
void Engine::profile_reset() {
    const double t = prof.tsc_per_us;
    prof = EngineProfile();
    prof.tsc_per_us = t;
}

void Engine::profile_report(FILE* f) const {
    if (!prof.tokens) {
        std::fprintf(f, "profile: no tokens recorded\n");
        return;
    }
    const double tpu = prof.tsc_per_us;
    double sum = 0;
    for (int p = 0; p < PH_COUNT; ++p) sum += (double)prof.ticks[p];
    const double nt = (double)prof.tokens;
    std::fprintf(f, "profile over %llu tokens (thread 0 rdtsc, phase = time between barrier completions; includes straggler wait)\n",
                 (unsigned long long)prof.tokens);
    std::fprintf(f, "%-30s %9s %6s %9s %10s %8s\n", "phase", "ms/token", "%", "calls/tok", "MB/token", "GB/s");
    for (int p = 0; p < PH_COUNT; ++p) {
        const double ms = (double)prof.ticks[p] / tpu / 1000.0 / nt;
        const double mb = (double)prof.bytes[p] / 1e6 / nt;
        std::fprintf(f, "%-30s %9.3f %5.1f%% %9.1f %10.2f %8.1f\n", kProfNames[p], ms, 100.0 * (double)prof.ticks[p] / std::max(sum, 1.0),
                     (double)prof.calls[p] / nt, mb, mb > 0 && ms > 0 ? mb / ms : 0.0);
    }
    std::fprintf(f, "%-30s %9.3f  (%.2f tok/s if serialized at this speed)\n", "sum of phases", sum / tpu / 1000.0 / nt,
                 1e6 / (sum / tpu / nt));
}

}  // namespace ink

// inkcpu decode engine: packed weights, per-slot state, single-sequence decode executor. Owner: agent C1.
// See docs/ENGINE_DESIGN.md. The per-visit weight descriptors (LayerPacked) and the per-slot state (SlotState) are plain
// structs so a batch-n (prefill) visit executor can live next to the decode one and reuse them.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ink/ggml_types.h"
#include "ink/gguf.h"
#include "ink/kernels_gemv.h"
#include "ink/kernels_misc.h"
#include "ink/model.h"
#include "ink/threadpool.h"

namespace ink {

// Geometry the executor is specialised for (checked at load; see Engine::create).
constexpr int kD = kHidden;                       // 2304
constexpr int kQW = kNQHeads * kHeadDim;          // 3072
constexpr int kKVW = kNKvHeads * kHeadDim;        // 1024
constexpr int kRelW = kNQHeads * kRelFeat;        // 384
constexpr int kFF = 1152;                         // expert ff
constexpr int kNExp = 128, kTopK = 8;
constexpr int kCtrlRank = 64;
constexpr int kVisits = 48;
constexpr int kMaxSplit = 16;                     // max key splits of one attention group
constexpr int kSwaRingBlocks = 32;                // SWA KV ring: 32 blocks = 2048 positions (a prefill chunk of n tokens needs >= 1023 + n)
constexpr int kMaxBlocks = 5;                     // attention-residual block states kept (4 pushed + cur)
// conv histories per visit, SoA [3][C] each, order: K (1024), V (1024), wo-out (2304), moe-out (2304)
constexpr size_t kHistPerVisit = 3 * (2 * (size_t)kKVW + 2 * (size_t)kD);
// ---- DSpark taps (docs/DSPARK.md 4.1, docs/SPEC_DESIGN.md): per processed row 4 vectors of 2304 floats, layout [tap][2304], F32:
// tap 0/1/2 = the residual stream `cur` ENTERING visit 5 / 23 / 41 (running partial sum of the current attention-residual block, before any depth mix),
// tap 3 = the final normalised hidden state (output depth mix + output_norm), the input of the lm_head quantiser.
constexpr int kEngineTaps = 4;
constexpr int kEngineTapFloats = kEngineTaps * kD;
constexpr int kTapVisit[3] = {5, 23, 41};
constexpr int kVerifyMaxRows = 8;                 // widest verify step (T = anchor + 7 drafts)

struct EngineOptions {
    int n_threads = 12;
    int n_slots = 1;
    int max_ctx = 4096;
    std::vector<int> cpus;     // empty: {2,4,..,2n} for n <= 15, {2,4,..,30,0} for 16 (core 0 is half speed on the dev host), then odd CPUs
    bool profile = false;      // also enabled by env INK_PROFILE=1
};

// One streamed weight matrix, rows contiguous along k. type = T_Q8_0 / T_NVFP4 / T_BF16 / T_F32.
struct Mat {
    const void* w = nullptr;
    int type = -1;
    int rows = 0, k = 0;
    size_t rs = 0;                         // row stride in bytes
    size_t bytes() const { return (size_t)rows * rs; }
};

// Everything a visit of one PHYSICAL layer reads (shared by the passes that visit it).
struct LayerPacked {
    int phys = 0;
    bool swa = false, global = false, core = false;
    Mat wq, wk, wv, wgate, wrel, wo;       // Q8_0
    Mat sink, kproj, vproj;                // BF16 (kproj / vproj: global layers only)
    const float* q_clip = nullptr;         // [24*128]
    const float* rel_clip = nullptr;       // [24*16]
    const float* rel_table = nullptr;      // [16][kRelTableLen] (attn_repack_rel_bias)
    const float* kconv = nullptr;          // SoA taps [4][1024] (sconv_repack_static)
    const float* vconv = nullptr;          // [4][1024]
    const float* aconv = nullptr;          // [4][2304] after wo
    const float* fconv = nullptr;          // [4][2304] after the MoE
    const float* attn_norm = nullptr;      // SWA layers
    const float* inp_norm[3] = {nullptr, nullptr, nullptr};   // global layers: q, k, v
    const float* attn_res_query = nullptr; // [1][2304] (SWA) / [3][2304] (global); null in layer 0
    const float* ffn_res_query = nullptr;
    const float* ffn_norm = nullptr;
    Mat router;                            // BF16 (exact conversion) or F32, [128][2304]
    const float* exp_bias = nullptr;       // [128]
    int expert_type = 0;                   // T_NVFP4 (layers 0-19) or T_Q8_0 (20-23)
    const void* gu = nullptr;              // [128 experts][2304 rows][k=2304]
    size_t gu_rs = 0, gu_eb = 0;           // row stride, expert stride (bytes)
    const void* dn = nullptr;              // [128][2304 rows][k=1152]
    size_t dn_rs = 0, dn_eb = 0;
    const float* gu_scale = nullptr;       // [128] (NVFP4 layers only)
    const float* dn_scale = nullptr;
    const float* msg_query = nullptr;      // [4][2304] (core layers)
    Mat ctrl_down[4], ctrl_gain[4], ctrl_forget[4];   // core layers: Q, K, V, FFN
};

// Per-sequence state of one slot.
struct SlotState {
    bool used = false;
    int64_t pos = 0;                       // tokens in the slot (a batch executor must advance it and, for the 8 global visits,
                                           // keep kv[v].n_blocks >= (last position / 64 + 1) before it appends/attends)
    KvLayer kv[kVisits];                   // per visit (SWA: ring of kSwaRingBlocks blocks; global: blocks grow up to max_ctx)
    size_t kv_map_bytes[kVisits] = {};     // mapped bytes per K / V tensor
    float* hist = nullptr;                 // kVisits * kHistPerVisit floats
    int64_t hwm = 0;                       // highest pos reached since the reset (a rewind / verify rollback does not lower it): ring cells of
                                           // positions < hwm hold data; the restorability of a shallow checkpoint is judged against it (docs/REWIND.md)
    struct CkptStore* ckpt = nullptr;      // slot checkpoints (engine_rewind.cpp), allocated on first use, dropped by slot_reset
};

enum ProfPhase {
    PH_EMBED = 0, PH_MIX, PH_CTRL, PH_PROJ, PH_ATTN, PH_WO, PH_FFNMIX, PH_ROUTER, PH_GATEUP, PH_DOWN, PH_FINAL, PH_LMHEAD, PH_COUNT
};
extern const char* const kProfNames[PH_COUNT];

struct EngineProfile {
    uint64_t ticks[PH_COUNT] = {};
    uint64_t calls[PH_COUNT] = {};
    uint64_t bytes[PH_COUNT] = {};
    uint64_t tokens = 0;
    uint64_t total_ticks = 0;
    double tsc_per_us = 0;
};

// CPU list used for n threads (see EngineOptions::cpus)
std::vector<int> engine_default_cpus(int n);
// checkpoint store helpers (engine_rewind.cpp): drop all checkpoints of a slot / release the store / bytes it holds
void slot_ckpt_clear(SlotState& s);
void slot_ckpt_free(SlotState& s);
size_t slot_ckpt_bytes(const SlotState& s);

struct Scratch;                            // decode scratch (engine_decode.cpp)
struct PhaseCounter;

struct Tune {                              // env-overridable tunables (read at create)
    int chunk_proj = 32, chunk_wo = 32, chunk_down = 32, chunk_lm = 512, chunk_router = 8, chunk_ctrl = 64;
    int split_keys = 384;                  // attention key split granule: nsplit = clamp(n_keys / split_keys, 1, kMaxSplit)
    int swa_blocks = kSwaRingBlocks;       // SWA ring size in 64-position blocks (env INK_SWA_BLOCKS, tests only; >= 16)
    int ckpt_max = 8;                      // checkpoints kept per slot (env INK_CKPT_MAX; 0 = no checkpoints at all, rewind always fails)
    int ckpt_deep = 2;                     // of which deep ones (with the SWA window copy, ~170 MiB each at 32 ring blocks; env INK_CKPT_DEEP)
};

class Engine {
public:
    // Loads the GGUF, packs every streamed tensor into hugepage memory, builds the thread pool. nullptr + *err on failure.
    static std::unique_ptr<Engine> create(const std::string& gguf_path, const EngineOptions& opt, std::string* err);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // ---- slots ----
    int slot_new();                                  // -1 if none free
    void slot_free(int slot);
    void slot_reset(int slot);
    int64_t slot_pos(int slot) const;
    bool slot_copy(int src, int dst);                // copies the whole sequence state (same engine; the destination's checkpoints are dropped)

    // ---- checkpoints and rewind (agent S3, docs/REWIND.md). Checkpoints are taken automatically after every prefill chunk and at the end of
    // every forward() of >= 2 tokens (shallow: pos + conv histories, 3.83 MB); decode() never checkpoints. deep = also copies the SWA window
    // (restorable no matter how many tokens follow); a shallow one is restorable only while hwm - pos <= rewind_max_append (1025).
    int slot_checkpoint(int slot, bool deep = false);          // 0 ok, -1 bad slot, -5 out of memory
    int64_t slot_rewind(int slot, int64_t pos);                // rewinds to the nearest restorable checkpoint <= pos and returns it; -1 = none.
                                                               // Afterwards decode/forward continue from there; checkpoints above it are dropped.
    int slot_checkpoints(int slot, int64_t* out, int max) const;   // restorable positions, ascending; returns their count (writes min(count, max))
    static constexpr int64_t rewind_max_append(int swa_blocks) { return (int64_t)swa_blocks * kKvBlock - (kSwaWindow - 1); }   // 1025 for 32 blocks

    // ---- decode: append ONE token; logits (n_vocab floats, softcap applied) or nullptr to skip the lm head. 0 = ok ----
    int decode(int slot, int32_t token, float* logits);
    // n tokens (decode path looped for now; a batched prefill executor can replace it); logits of the last token only
    int forward(int slot, const int32_t* tokens, int n, float* logits_last);

    int n_vocab() const { return cfg.n_vocab; }
    void set_threads(int n_threads);                 // rebuilds the pool (results are bit-identical for any count)

    // ---- profiling (INK_PROFILE=1 or opt.profile) ----
    bool profiling() const { return prof_on; }
    void profile_reset();
    void profile_report(FILE* f) const;
    double last_total_us = 0;       // wall time of the last decode() (one token)
    double last_forward_us = 0;     // wall time of the last forward() (all its tokens)

    // ---- debug trace (single-token tests): called by thread 0 after the named point of each visit ----
    // names: "Qcur","Kcur","Vcur","Rcur" (post clip), "attn_out" (post conv), "ffn_inp", "l_out"; il = llama slot of the visit
    using TraceFn = std::function<void(const char* name, int visit_index, const float* data, size_t n)>;
    void set_trace(TraceFn fn) { trace = std::move(fn); }

    // ---- public state (so a batch executor can reuse it) ----
    EngineOptions opt;
    Tune tune;
    ModelConfig cfg;
    std::vector<Visit> plan;
    std::unique_ptr<GgufFile> gguf;
    std::vector<LayerPacked> layers;                 // physical layers
    Mat lm_head;                                     // Q8_0 [200058][2304]
    const uint8_t* tok_embd = nullptr;               // Q8_0 rows in the GGUF mapping (one row read per token)
    size_t tok_embd_rs = 0;
    const float* tok_norm = nullptr;
    const float* out_norm = nullptr;
    const float* out_res_query = nullptr;
    std::unique_ptr<ThreadPool> pool;
    std::vector<SlotState> slots;
    double load_seconds = 0;
    size_t weight_bytes = 0;
    EngineProfile prof;
    bool prof_on = false;
    TraceFn trace;

    // ---- tap sink (speculative decoding). Null = off: the executors then do no tap work at all (a pointer test).
    // Called after every token processed by decode() (rows = 1) and after every chunk of a batched prefill (rows <= chunk); `taps` = [rows][kEngineTapFloats],
    // valid only during the call; first_pos = position of the first row (the slot position BEFORE the call). Never called by engine_verify (the caller
    // commits the rows it keeps).
    using TapSink = std::function<void(int slot, int64_t first_pos, int rows, const float* taps)>;
    TapSink tap_sink;
    const float* last_taps() const;                  // taps of the last decode() token (valid when tap_sink was set)

private:
    Engine() = default;
    bool alloc_slot(SlotState& s, std::string* err);
    void free_slot(SlotState& s);
    void run_token(SlotState& S, int32_t token, float* logits);

    void* arena_ = nullptr;
    std::vector<void*> small_;
    Scratch* scr_ = nullptr;
    PhaseCounter* ctr_ = nullptr;
    std::thread::id pinned_thread_;
    friend struct EngineLoader;
};

}  // namespace ink

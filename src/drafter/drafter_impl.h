// Internal header of the DSpark drafter (agent S1): shared by drafter.cpp, attn.cpp, fold.cpp and the tests. Not part of the public API
// (include/ink/drafter.h is). Design: docs/DRAFTER.md. Spec: docs/DSPARK.md sections 4-5.
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ink/drafter.h"
#include "ink/engine.h"
#include "ink/kernels_gemm.h"
#include "ink/kernels_gemv.h"
#include "ink/kernels_misc.h"
#include "ink/kernels_misc_simd.h"

namespace ink {
namespace dr {

constexpr int D = kD;                 // 2304
constexpr int HD = 128;               // head dim
constexpr int NQH = 24, NKV = 8;      // query / kv heads
constexpr int QW = 3072, KVW = 1024;
constexpr int FF = 9216;
constexpr int NL = 3;                 // drafter layers
constexpr int BS = 8;                 // block slots (slot 0 = anchor, 1..7 = mask)
constexpr int RANK = 256;             // Markov rank
constexpr int JW = 5120, JO = 768;    // joint_proj: [768 rows][5120 cols] = [s 256 | prev 256 | hid 2304 | f_rnn 2304]
constexpr int RC = 64;                // ctx rows per flush round
constexpr int MAXTH = 64;
constexpr int NSPLIT_MAX = 16;        // key splits of one attention group
constexpr int CH_ROWS = 512;          // rows per chain chunk (M matvec)
constexpr int WIN_BLOCKS = 16;        // SWA ring: 16 blocks of 64 = the 1024 positions before the anchor
constexpr int PEND_CAP = 64;          // pending (lazily committed) rows per state
constexpr int NFUSE = 4;              // taps / consumers

// what the drafter borrows from the target engine (a test can fill it without loading the 17 GB engine)
struct DrafterTarget {
    ThreadPool* pool = nullptr;
    int cpu0 = -1;                    // pin the calling thread (tid 0) here on first use; -1 = do not pin
    Mat lm_head;                      // Q8_0 [n_vocab][2304]
    const uint8_t* tok_embd = nullptr;
    size_t tok_embd_rs = 0;
    const float* tok_norm = nullptr;
    int n_vocab = 0, n_vocab_padded = 0;
    float rms_eps = 1e-6f, softcap = 30.0f;
    int max_ctx = 4096, n_slots = 1;
};

struct LayerW {
    Mat wq, wk, wv, wo, wk_ctx, wv_ctx, wgate, wup, wdown;   // Q8_0 (rows = outputs, k = inputs)
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *k_ctx_norm = nullptr, *dflare_norm = nullptr, *ffn_norm = nullptr;
};

struct FoldM {
    bool q8 = true;
    const void* data = nullptr;       // [n_vocab][256] Q8_0 (rs 272) or bf16 (rs 512)
    size_t rs = 0;
    const float* m0 = nullptr;        // [n_vocab]
    size_t bytes = 0;
};

// fold.cpp ------------------------------------------------------------------------------------------------------------------------------------
// M[v][i] = sum_k lm_head[v][k] * hidden_out[k][i]  (hout_w: bf16 [2304][256] row-major, k = row), m0[v] = sum_k lm_head[v][k] * hout_b[k].
// fp32 accumulation, rows split over the pool. Output: Q8_0 blocks (quantize_row_q8_0_ref) or bf16, plus m0 (f32).
void fold_compute(ThreadPool& pool, const Mat& lm_head, int n_vocab, const uint16_t* hout_w, const float* hout_b, bool q8, void* out_M, size_t out_rs,
                  float* out_m0);
// 64-bit content hash of a byte range (parallel, independent of the thread count)
uint64_t hash_bytes_parallel(ThreadPool& pool, const void* p, size_t n);
bool fold_cache_load(const std::string& path, uint64_t key, bool q8, int n_vocab, void* M, size_t M_bytes, float* m0);
bool fold_cache_store(const std::string& path, uint64_t key, bool q8, int n_vocab, const void* M, size_t M_bytes, const float* m0);

// attn.cpp ------------------------------------------------------------------------------------------------------------------------------------
struct alignas(64) AttnPartial2 {     // running softmax state of up to 24 queries of one kv group over a key range
    float m[24];
    float l[24];
    float acc[24 * HD];
};
inline int attn_nsplit(int nblocks) { return nblocks <= 8 ? 1 : (nblocks / 8 > NSPLIT_MAX ? NSPLIT_MAX : nblocks / 8); }
// cache blocks [b0, b1) of one layer cache for kv group g; nvalid = number of live key slots (slots 0..nvalid-1); q = [nq][128] (pre-scaled by 1/sqrt(128)),
// nq in {3,6,...,24}. Block keys are NOT included (see attn_finish_query).
void attn_ctx_part(const KvLayer& L, int g, int b0, int b1, int64_t nvalid, const float* q, int nq, AttnPartial2* out);
// Final output of ONE query i of group g: merges `nparts` ctx partials (parts[sp].m/l/acc at query index i; a partial with l == 0 is skipped) and the
// nkeys (<= 8) block keys (kb/vb: row k at kb + k*kstride, 128 floats each). out: 128 floats.
void attn_finish_query(const AttnPartial2* parts, int nparts, int i, const float* q, const float* kb, const float* vb, size_t kstride, int nkeys, float* out);

// small vector helpers shared with the tests ---------------------------------------------------------------------------------------------------
void rope_table(int64_t pos, float* cs /*[128]: cos[64], sin[64]*/);
void rope_head(float* x /*[128]*/, const float* cs);

struct Sched {                        // dynamic-scheduling counters (one per phase, identical numbering on every thread)
    struct alignas(64) C { std::atomic<int64_t> v; };
    std::unique_ptr<C[]> c;
    int n = 0;
    int used = 0;
};

struct StateData {
    bool used = false;
    int64_t pos = 0;                  // committed positions
    int64_t done = 0;                 // positions whose ctx K/V are in the caches (pos - done = pending rows)
    KvLayer kv[NL];
    size_t kv_bytes[NL] = {};         // mapped bytes per tensor
    float* pend = nullptr;            // [PEND_CAP][4*D] pending taps
    alignas(64) float f_rnn[D];
    bool have_rnn = false;
};

struct ChunkRes {                     // per chain chunk (CH_ROWS vocabulary rows): greedy: best value / index; sampled: chunk max (of logit/T), sum of exp(x - max)
    float val;
    int32_t idx;
    float sum;
    float pad;
};
// Inverse-CDF draw from the chunked unnormalised probabilities (qrow[v] = exp(x_v - chunk max)); deterministic in (seed, j). Returns the token; M / Z = global max / normaliser.
int32_t sample_select(const ChunkRes* cres, int nchunks, const float* qrow, int nv, uint64_t seed, int j, double* M_out, double* Z_out);

struct Debug {                        // captured by thread 0 of the next draft() when on (tests)
    bool on = false;
    bool force_on = false;            // teacher forcing: the chain feeds back force[j-1] instead of its own choice
    int32_t force[8] = {};
    std::vector<float> x[NL];         // X[8][D] after each layer
    std::vector<float> hid;           // hidf[8][D]
    std::vector<float> lat;           // [7][256] latents of the chain
    std::vector<float> logits;        // [7][n_vocab] softcapped logits of each chain step (before sampling)
    std::vector<float> base;          // [7][n_vocab] base logits lm_head . hid[j]
    std::vector<float> fused[NL];     // unused
};

class DrafterImpl final : public Drafter {
public:
    static std::unique_ptr<DrafterImpl> make(const DrafterTarget& t, const std::string& gguf, const DrafterOptions& opt, std::string* err);
    ~DrafterImpl() override;

    int state_new() override;
    void state_free(int st) override;
    void state_reset(int st) override;
    int64_t state_pos(int st) const override { return ok(st) ? states_[st].pos : -1; }
    int commit(int st, const float* taps, int rows) override;
    int draft(int st, const DraftRequest& rq, int32_t* out_tokens, float* q_out) override;
    int64_t state_pending(int st) const override { return ok(st) ? states_[st].pos - states_[st].done : 0; }
    int flush(int st) override;
    size_t state_bytes(int st) const override;
    size_t weight_bytes() const override { return weight_bytes_; }
    const DrafterProfile* profile() const override { return &prof_; }
    void profile_reset() override { prof_ = DrafterProfile(); }
    void profile_report(FILE* f) const override;

    // ---- public state for the tests ----
    DrafterTarget tgt;
    DrafterOptions opt;
    LayerW L[NL];
    float alpha[NFUSE][NFUSE];        // softmax(fuse_weight rows)
    const float *mask_embd = nullptr, *out_norm = nullptr, *dspark_norm = nullptr, *joint_b = nullptr, *hout_b = nullptr;
    const uint16_t *joint_w = nullptr, *hout_w = nullptr, *markov_w1 = nullptr;
    FoldM fold;
    Debug dbg;
    StateData& state(int st) { return states_[st]; }
    bool ok(int st) const { return st >= 0 && st < (int)states_.size() && states_[st].used; }
    // the fold cache key / path actually used (informational)
    std::string fold_path;
    bool fold_from_cache = false;
    double fold_seconds = 0;
    // computes f_c(t) (c = 0..3) for one tap row (4 x 2304 floats); used by commit() and the tests
    void fuse_row(const float* taps, int c, float* f_out) const;

private:
    DrafterImpl() = default;
    struct Run;                       // per-call context of the pool body (drafter.cpp)
    struct Scr;                       // scratch (drafter.cpp)
    int run_pool(StateData& S, const DraftRequest* rq, int32_t* out_tokens, float* q_out, const float* src, int src_rows, int64_t src_pos0);
    void body(Run& R, int tid, int nth);
    bool alloc_state(StateData& S, std::string* err);
    void free_state(StateData& S);
    std::vector<StateData> states_;
    std::unique_ptr<Scr> scr_;
    Sched sched_;
    std::unique_ptr<GgufFile> gguf_;
    void* arena_ = nullptr;
    size_t arena_bytes_ = 0;
    size_t weight_bytes_ = 0;
    void* fold_buf_ = nullptr;
    float* m0_buf_ = nullptr;
    DrafterProfile prof_;
    std::thread::id pinned_;
};

}  // namespace dr
}  // namespace ink

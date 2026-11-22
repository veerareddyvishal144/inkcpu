// A "light target" for the drafter tests / bench / tools: the pieces of the target GGUF the drafter borrows (lm_head copied into a hugepage arena,
// token_embd + norm in the mapping) plus a ThreadPool, WITHOUT loading the 17 GB engine. Honours INK_CPUS like the engine. Owner: agent S1.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "drafter_impl.h"
#include "ink/model.h"

namespace ink {
namespace dr {

struct LightTarget {
    std::unique_ptr<GgufFile> g;
    ModelConfig cfg;
    std::unique_ptr<ThreadPool> pool;
    DrafterTarget t;
    void* lm_buf = nullptr;
    ~LightTarget() {
        pool.reset();
        if (lm_buf) free_weight_buffer(lm_buf);
    }
    bool init(const std::string& path, int nthreads, int max_ctx, std::string* err) {
        g = GgufFile::open(path, err);
        if (!g) return false;
        if (!ModelConfig::from_gguf(*g, &cfg, err)) return false;
        const GgufTensorInfo* out = g->tensor("output.weight");
        const GgufTensorInfo* emb = g->tensor("token_embd.weight");
        const GgufTensorInfo* nrm = g->tensor("token_embd_norm.weight");
        if (!out || !emb || !nrm || out->type != T_Q8_0 || emb->type != T_Q8_0) {
            if (err) *err = "target tensors missing";
            return false;
        }
        const std::vector<int> cpus = engine_default_cpus(nthreads);
        pool.reset(new ThreadPool(nthreads, cpus));
        t.pool = pool.get();
        t.cpu0 = cpus[0];
        const size_t rs = out->row_bytes();
        lm_buf = alloc_weight_buffer((size_t)cfg.n_vocab * rs, true);
        std::memcpy(lm_buf, out->data, (size_t)cfg.n_vocab * rs);
        t.lm_head.w = lm_buf;
        t.lm_head.type = T_Q8_0;
        t.lm_head.rows = cfg.n_vocab;
        t.lm_head.k = (int)out->ne[0];
        t.lm_head.rs = rs;
        t.tok_embd = emb->data;
        t.tok_embd_rs = emb->row_bytes();
        t.tok_norm = (const float*)nrm->data;
        t.n_vocab = cfg.n_vocab;
        t.n_vocab_padded = cfg.n_vocab_padded;
        t.rms_eps = cfg.rms_eps;
        t.softcap = cfg.final_logit_softcap;
        t.max_ctx = max_ctx;
        t.n_slots = 1;
        return true;
    }
};

}  // namespace dr
}  // namespace ink

// Slot checkpoints and rewind (agent S3). Design, proof of the restorability bound and memory: docs/REWIND.md.
//
// A checkpoint of a slot at position P is {P, copy of the 48 conv histories (3.83 MB)} ("shallow") plus, for a "deep" one, a copy of the
// SWA ring blocks that hold the attention window of P (positions P-1023..P-1 of the 40 SWA visits, <= 17 blocks x 128 KiB x 2 tensors
// x 40 visits = 170 MiB). The KV of the 8 global visits never needs a copy: rows >= P are junk after a rewind and are overwritten before
// they can be read. A shallow checkpoint is restorable while the ring cells of P-1023..P-1 are intact, i.e. while hwm - P <= ring -
// (window - 1) = 1025 for the 32-block ring (hwm = highest position ever written since the reset; a rewind does not lower it).
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ink/engine.h"

namespace ink {

namespace {

struct CkptEntry {
    int64_t pos = -1;            // -1 = empty
    bool deep = false;
    uint64_t seq = 0;            // creation / last use order (eviction picks the smallest)
    float* hist = nullptr;       // kVisits * kHistPerVisit floats (allocated once, reused)
    uint16_t* swa = nullptr;     // deep: [swa visit][K, V][blk][kKvBlockElems] fp16, swa_nblk ring blocks starting at ring block swa_blk0
    int swa_blk0 = 0, swa_nblk = 0;
};

constexpr size_t kHistBytes = (size_t)kVisits * kHistPerVisit * sizeof(float);
constexpr size_t kBlkBytes = kKvBlockElems * sizeof(uint16_t);

}  // namespace

struct CkptStore {
    std::vector<CkptEntry> e;
    std::vector<uint16_t*> swa_pool;   // free deep buffers (at most tune.ckpt_deep are ever allocated)
    std::vector<int> swa_visits;       // visit indices whose KV is a ring
    int ring_blocks = 0;               // blocks of the ring (all SWA visits share it)
    int blk_cap = 0;                   // ring blocks a deep buffer can hold (blocks covering kSwaWindow - 1 positions, <= ring_blocks)
    uint64_t seq = 0;
    size_t deep_bytes() const { return (size_t)swa_visits.size() * 2 * (size_t)blk_cap * kBlkBytes; }
};

namespace {

CkptStore& ensure_store(const Engine& E, SlotState& S) {
    if (S.ckpt) return *S.ckpt;
    auto* st = new CkptStore();
    st->e.resize((size_t)std::max(1, E.tune.ckpt_max));
    for (int v = 0; v < kVisits; ++v)
        if (S.kv[v].ring) {
            st->swa_visits.push_back(v);
            st->ring_blocks = (int)S.kv[v].n_blocks;
        }
    // the window of P is kSwaWindow - 1 positions; they start anywhere inside a block, so they span at most ceil((W-1)/64) + 1 blocks
    st->blk_cap = std::min(st->ring_blocks, (kSwaWindow - 1 + kKvBlock - 1) / kKvBlock + 1);
    S.ckpt = st;
    return *st;
}

float* hist_buf(CkptEntry& e) {
    if (!e.hist) e.hist = (float*)std::aligned_alloc(64, (kHistBytes + 63) / 64 * 64);
    return e.hist;
}

void release_deep(CkptStore& st, CkptEntry& e) {
    if (e.swa) st.swa_pool.push_back(e.swa);
    e.swa = nullptr;
    e.deep = false;
    e.swa_nblk = 0;
}

void drop(CkptStore& st, CkptEntry& e) {
    release_deep(st, e);
    e.pos = -1;
}

int deep_count(const CkptStore& st) {
    int n = 0;
    for (const auto& e : st.e) n += e.pos >= 0 && e.deep;
    return n;
}

// Ring blocks that hold positions max(0, P - (kSwaWindow - 1)) .. P - 1. Returns the count (0 when P <= 0) and the first block.
int window_blocks(const CkptStore& st, int64_t P, int* blk0) {
    const int64_t R = (int64_t)st.ring_blocks * kKvBlock;
    const int64_t a = std::max<int64_t>(0, P - (kSwaWindow - 1));
    if (P <= 0 || P - a <= 0 || R <= 0) {
        *blk0 = 0;
        return 0;
    }
    const int b0 = (int)((a % R) / kKvBlock), b1 = (int)(((P - 1) % R) / kKvBlock);
    *blk0 = b0;
    return ((b1 - b0) % st.ring_blocks + st.ring_blocks) % st.ring_blocks + 1;
}

void save_deep(const CkptStore& st, const SlotState& S, CkptEntry& e) {
    e.swa_nblk = window_blocks(st, e.pos, &e.swa_blk0);
    uint16_t* dst = e.swa;
    for (int v : st.swa_visits)
        for (int t = 0; t < 2; ++t) {
            const uint16_t* base = t ? S.kv[v].v : S.kv[v].k;
            for (int i = 0; i < e.swa_nblk; ++i)
                std::memcpy(dst + (size_t)i * kKvBlockElems, base + (size_t)((e.swa_blk0 + i) % st.ring_blocks) * kKvBlockElems, kBlkBytes);
            dst += (size_t)st.blk_cap * kKvBlockElems;
        }
}

void restore_deep(const CkptStore& st, SlotState& S, const CkptEntry& e) {
    const uint16_t* src = e.swa;
    for (int v : st.swa_visits)
        for (int t = 0; t < 2; ++t) {
            uint16_t* base = t ? S.kv[v].v : S.kv[v].k;
            for (int i = 0; i < e.swa_nblk; ++i)
                std::memcpy(base + (size_t)((e.swa_blk0 + i) % st.ring_blocks) * kKvBlockElems, src + (size_t)i * kKvBlockElems, kBlkBytes);
            src += (size_t)st.blk_cap * kKvBlockElems;
        }
}

int64_t shallow_bound(const CkptStore& st) { return Engine::rewind_max_append(st.ring_blocks); }

bool restorable(const CkptStore& st, const SlotState& S, const CkptEntry& e) {
    return e.pos >= 0 && (e.deep || std::max(S.hwm, S.pos) - e.pos <= shallow_bound(st));
}

// Eviction order: empty, dead shallow (never restorable again), shallow by age, deep by age.
CkptEntry& victim(CkptStore& st, const SlotState& S) {
    CkptEntry* best = nullptr;
    auto rank = [&](const CkptEntry& e) -> int {
        if (e.pos < 0) return 0;
        if (!e.deep && !restorable(st, S, e)) return 1;
        return e.deep ? 3 : 2;
    };
    for (auto& e : st.e)
        if (!best || rank(e) < rank(*best) || (rank(e) == rank(*best) && e.seq < best->seq)) best = &e;
    return *best;
}

}  // namespace

void slot_ckpt_clear(SlotState& s) {
    if (!s.ckpt) return;
    for (auto& e : s.ckpt->e) drop(*s.ckpt, e);
}

void slot_ckpt_free(SlotState& s) {
    if (!s.ckpt) return;
    for (auto& e : s.ckpt->e) {
        std::free(e.hist);
        std::free(e.swa);
    }
    for (uint16_t* p : s.ckpt->swa_pool) std::free(p);
    delete s.ckpt;
    s.ckpt = nullptr;
}

size_t slot_ckpt_bytes(const SlotState& s) {
    if (!s.ckpt) return 0;
    size_t b = s.ckpt->swa_pool.size() * s.ckpt->deep_bytes();
    for (const auto& e : s.ckpt->e) b += (e.hist ? kHistBytes : 0) + (e.swa ? s.ckpt->deep_bytes() : 0);
    return b;
}

int Engine::slot_checkpoint(int slot, bool deep) {
    if (slot < 0 || slot >= (int)slots.size() || !slots[slot].used) return -1;
    SlotState& S = slots[slot];
    if (tune.ckpt_max <= 0 || S.pos <= 0) return 0;   // nothing to save (position 0 = slot_reset)
    S.hwm = std::max(S.hwm, S.pos);
    CkptStore& st = ensure_store(*this, S);
    if (deep && tune.ckpt_deep <= 0) deep = false;
    CkptEntry* e = nullptr;
    for (auto& x : st.e)
        if (x.pos == S.pos) e = &x;
    if (e && (!deep || e->deep)) {   // already there (same prefix: an entry at the live position is always current)
        e->seq = ++st.seq;
        return 0;
    }
    if (!e) {
        e = &victim(st, S);
        drop(st, *e);
        std::memcpy(hist_buf(*e), S.hist, kHistBytes);
        e->pos = S.pos;
    }
    if (deep) {   // deep budget: demote the oldest deep entry (it stays as a shallow checkpoint)
        while (deep_count(st) >= tune.ckpt_deep) {
            CkptEntry* old = nullptr;
            for (auto& x : st.e)
                if (x.pos >= 0 && x.deep && (!old || x.seq < old->seq)) old = &x;
            if (!old) break;
            release_deep(st, *old);
        }
        if (st.swa_pool.empty()) {
            void* p = std::aligned_alloc(64, (st.deep_bytes() + 63) / 64 * 64);
            if (!p) return -5;
            st.swa_pool.push_back((uint16_t*)p);
        }
        e->swa = st.swa_pool.back();
        st.swa_pool.pop_back();
        e->deep = true;
        save_deep(st, S, *e);
    }
    e->seq = ++st.seq;
    return 0;
}

int64_t Engine::slot_rewind(int slot, int64_t pos) {
    if (slot < 0 || slot >= (int)slots.size() || !slots[slot].used) return -1;
    SlotState& S = slots[slot];
    S.hwm = std::max(S.hwm, S.pos);
    if (!S.ckpt) return -1;
    CkptStore& st = *S.ckpt;
    CkptEntry* best = nullptr;
    for (auto& e : st.e)
        if (e.pos >= 0 && e.pos <= pos && restorable(st, S, e) && (!best || e.pos > best->pos)) best = &e;
    if (!best) return -1;
    if (best->pos != S.pos) {
        std::memcpy(S.hist, best->hist, kHistBytes);
        if (best->deep) restore_deep(st, S, *best);
        S.pos = best->pos;   // global KV rows >= pos are junk and get overwritten before any read (n_blocks stays grown, like after a verify rollback)
    }
    for (auto& e : st.e)
        if (e.pos > best->pos) drop(st, e);   // they describe the abandoned continuation
    best->seq = ++st.seq;
    return best->pos;
}

int Engine::slot_checkpoints(int slot, int64_t* out, int max) const {
    if (slot < 0 || slot >= (int)slots.size() || !slots[slot].used || !slots[slot].ckpt) return 0;
    const SlotState& S = slots[slot];
    std::vector<int64_t> v;
    for (const auto& e : S.ckpt->e)
        if (restorable(*S.ckpt, S, e)) v.push_back(e.pos);
    std::sort(v.begin(), v.end());
    for (int i = 0; i < (int)v.size() && i < max && out; ++i) out[i] = v[(size_t)i];
    return (int)v.size();
}

}  // namespace ink

// Private definitions shared by engine_load.cpp and engine_decode.cpp (decode scratch, phase counters).
#pragma once
#include "ink/engine.h"

namespace ink {

constexpr int kMaxPhases = 4096;                 // dynamic-scheduling counters per token (phases are numbered identically on every thread)
struct PhaseCounter {
    alignas(64) std::atomic<int64_t> v;
    char pad[64 - sizeof(std::atomic<int64_t>)];
};

// Per-token scratch of the decode executor (one per engine; the executor runs one token at a time).
struct Scratch {
    alignas(64) float cur[2][kD];             // residual stream, ping-pong
    alignas(64) float blocks[kMaxBlocks][kD]; // attention-residual block starts
    alignas(64) float prelude[kD];
    alignas(64) float bank_b[kD];             // state entering layer 12 of the previous pass (message source)
    alignas(64) float bank_out[kD];           // output of layer 17 of the previous pass
    alignas(64) float msgs[4][kD];
    alignas(64) float mix[3][kD];
    alignas(64) float cin[4][kD];             // controller outputs (3 attention inputs, 1 ffn)
    alignas(64) float inx[3][kD];
    alignas(64) float inbf[3][kD];
    alignas(64) block_q8_0 inq[3][kD / 32];
    alignas(64) float q[kQW];
    alignas(64) float kp[kKVW];
    alignas(64) float vp[kKVW];
    alignas(64) float rel[kRelW];
    alignas(64) float gate[kQW];
    alignas(64) float sink[32];
    alignas(64) float kdyn[32];
    alignas(64) float vdyn[32];
    alignas(64) float ao[kQW];
    alignas(64) block_q8_0 aoq[kQW / 32];
    alignas(64) AttnPartial parts[8 * kMaxSplit];
    alignas(64) float wo_out[kD];
    alignas(64) float attn_out[kD];
    alignas(64) float fmix[kD];
    alignas(64) float xffn[kD];
    alignas(64) block_q8_0 xq[kD / 32];
    alignas(64) float rlogits[kNExp];
    alignas(64) block_q8_0 actq[kTopK][kFF / 32];
    alignas(64) float moe_out[kD];
    alignas(64) float ctrl_in[4][2 * kD];
    alignas(64) float ctrl_g[4][kCtrlRank];
    alignas(64) float ctrl_gain[4][kD];
    alignas(64) float ctrl_forget[4][kD];
    alignas(64) float fin_mix[kD];
    alignas(64) float fin_x[kD];
    alignas(64) block_q8_0 fin_q[kD / 32];
    alignas(64) float taps[kEngineTapFloats];   // taps of the last token (written only when a tap sink / tap capture is requested)
    int used_phases = 0;                      // dynamic-scheduling counters used by the previous token (reset only those)
};

std::vector<int> engine_default_cpus(int n);

}  // namespace ink

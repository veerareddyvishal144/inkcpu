// Block formats shared by every module. Bit-compatible with ggml (llama.cpp @2a28df0f9) so the engine can read the
// community GGUF directly and kernels can be tested against llama.cpp. Owner: orchestrator (request changes in docs/requests.md).
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace ink {

// ggml tensor type ids that occur in the Inkling 2 Mini GGUF
enum GgmlType : int { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_BF16 = 30, T_NVFP4 = 40 };

// ---- Q8_0: 32 int8 values with one fp16 scale. value = d * qs[i]. Also the ACTIVATION format for Q8_0/NVFP4 matmuls. ----
constexpr int QK8_0 = 32;
struct block_q8_0 {
    uint16_t d;           // IEEE fp16 scale
    int8_t   qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// ---- NVFP4: 64 weights = 4 sub-blocks of 16, each with an unsigned E4M3 scale; weights are E2M1 nibbles. 36 bytes. ----
// Sub-block s (0..3) owns bytes qs[8*s .. 8*s+7]; within it element j (0..7) is the LOW nibble of byte j and
// element 8+j is the HIGH nibble of byte j. Weight value = kvalues_fp4[nibble] * ue4m3_to_f32(d[s]).
// ue4m3_to_f32 already contains the factor 0.5 (kvalues_fp4 is 2*E2M1). The per-expert F32 scale tensor
// (ffn_gate_up_exps.scale[e], ffn_down_exps.scale[e]) is applied by the caller on top.
constexpr int QK_NVFP4 = 64;
constexpr int QK_NVFP4_SUB = 16;
struct block_nvfp4 {
    uint8_t d[QK_NVFP4 / QK_NVFP4_SUB];
    uint8_t qs[QK_NVFP4 / 2];
};
static_assert(sizeof(block_nvfp4) == 36, "block_nvfp4 must be 36 bytes");

constexpr int8_t kvalues_fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// UE4M3: unsigned, 4 exponent bits (bias 7), 3 mantissa bits; 0x00 and 0x7F decode to 0. Returns value * 0.5.
inline float ue4m3_to_f32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF;
    int man = x & 0x7;
    float raw = (exp == 0) ? ldexpf((float)man, -9) : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return raw * 0.5f;
}

inline float    fp16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t f32_to_fp16(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
inline float    bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline uint16_t f32_to_bf16(float f) {  // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

}  // namespace ink

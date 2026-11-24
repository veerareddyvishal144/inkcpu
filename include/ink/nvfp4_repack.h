// NVFP4 sub-block scale re-encoding ("nvfp4r") shared by the decode GEMV (K1), the small-n multi-activation GEMV (K2) and the
// prefill/verify GEMM unpack (K3). Orchestrator, 2026-11-23. Owner after this commit: agent K1 (post changes in docs/requests.md).
//
// Problem: the GGUF stores each 64-weight NVFP4 super-block as 4 UE4M3 scale bytes + 32 nibble bytes (36 B). Decoding UE4M3 needs a
// 256-entry table lookup per scale (4 scalar loads + 3 vinsertps + vpermps per row super-block in gemv.cpp, 4 scalar stores in the GEMM
// unpack). All 256 UE4M3 byte values decode (ue4m3_to_f32, factor 0.5 included) to 0 or to m * 2^e with m in {1, 1.125, ..., 1.875}
// (3 mantissa bits) and e in [-10, 7] (18 exponents): the denormal codes 1..7 are 2^-10, 2^-9, 1.5*2^-9, 2^-8, 1.25*2^-8, 1.5*2^-8,
// 1.75*2^-8; the normal codes are (1 + man/8) * 2^(exp-8), exp = 1..15, excluding 0x7F (= 0) and ignoring the sign bit.
//
// Re-encoding (done once at load, IN PLACE on the 4 scale bytes of every super-block of the engine's arena copy; the 32 nibble bytes are
// untouched, the block stays 36 B):  code = 0 for value 0, else code = (e5 << 3) | man3 with e5 = e + 11 (1..18), man3 = (m - 1) * 8.
// Decoding is then pure integer + one exact power-of-two multiply, bit-identical to ue4m3_to_f32:
//      value(code) = bit_cast<float>(uint32_t(code) << 20) * 0x1p116f            (code 0 -> +0.0f)
// because code << 20 puts e5 into the fp32 exponent field (2^(e5-127), always a NORMAL float for e5 >= 1) and man3 into the top 3 mantissa
// bits; * 2^116 restores 2^(e5-11) = 2^e exactly (no rounding, no denormals, no overflow: results lie in [2^-10, 224]).
// Vector form for 16 scales (= 4 rows x 4 sub-blocks, or 16 consecutive sub-blocks):
//      __m512 s = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(bytes16), 20)), _mm512_set1_ps(0x1p116f));
// 3 uops (p5 + p0 + p05) instead of 16 table loads; multiplying by the activation scale dx afterwards is the same float product as before
// (same two float operands), so every kernel built on it stays bit-identical to its scalar reference and to the table-based kernels.
//
// Rules: (1) the GGUF file and the mmap are never modified; only the engine's arena copy is re-encoded. (2) A kernel that reads
// re-encoded scales must be a distinct entry point or take the table/mode explicitly (`nvfp4r_*`), never silently assume one encoding.
// (3) ink::nvfp4r_scale_tab[256] (below) is the table form for code paths that still want a lookup (same floats as the vector form).
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "ink/ggml_types.h"

namespace ink {

constexpr float kNvfp4rPow = 0x1p116f;

// Scalar decode of a re-encoded scale byte (bit-identical to ue4m3_to_f32 of the original byte).
inline float nvfp4r_decode(uint8_t code) {
    uint32_t u = (uint32_t)code << 20;
    float f;
    std::memcpy(&f, &u, 4);
    return f * kNvfp4rPow;
}

// Original UE4M3 byte -> re-encoded byte. Total function on all 256 values (sign bit ignored, 0x00/0x7F/0xFF... -> 0).
inline uint8_t nvfp4r_encode(uint8_t ue4m3) {
    const float v = ue4m3_to_f32(ue4m3);
    if (v == 0.0f) return 0;
    int e;
    const float m2 = frexpf(v, &e);          // v = m2 * 2^e, m2 in [0.5, 1)
    const float m = m2 * 2.0f;               // m in [1, 2)
    const int e1 = e - 1;                    // v = m * 2^e1, e1 in [-10, 7]
    const int man3 = (int)((m - 1.0f) * 8.0f);
    const int e5 = e1 + 11;                  // 1..18
    return (uint8_t)((e5 << 3) | man3);
}

// 16 re-encoded scale bytes (already in a register, e.g. gathered with vmovd + vpinsrd from 4 rows) -> 16 floats. 3 uops.
inline __m512 nvfp4r_decode16v(__m128i b) {
    return _mm512_mul_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(b), 20)), _mm512_set1_ps(kNvfp4rPow));
}

// 16 re-encoded scale bytes -> 16 floats (see the header comment).
inline __m512 nvfp4r_decode16(const void* bytes16) {
    return nvfp4r_decode16v(_mm_loadu_si128((const __m128i*)bytes16));
}

// Byte table UE4M3 -> re-encoded code (built once, thread-safe), for the load-time repack of billions of scale bytes:
// nvfp4r_encode is a few ns per byte (frexpf), the table is one load.
inline const uint8_t* nvfp4r_encode_table() {
    static const struct Tab {
        uint8_t t[256];
        Tab() { for (int i = 0; i < 256; ++i) t[i] = nvfp4r_encode((uint8_t)i); }
    } tab;
    return tab.t;
}

// Re-encode the 4 scale bytes of `nblocks` CONSECUTIVE 36-byte super-blocks starting at p (dense tensors: rows and experts
// contiguous; any block-aligned byte range of such a tensor can be repacked independently, e.g. the pieces of a parallel copy).
inline void nvfp4r_repack_blocks(uint8_t* p, size_t nblocks) {
    const uint8_t* t = nvfp4r_encode_table();
    for (size_t i = 0; i < nblocks; ++i, p += sizeof(block_nvfp4)) {
        p[0] = t[p[0]]; p[1] = t[p[1]]; p[2] = t[p[2]]; p[3] = t[p[3]];
    }
}

// Re-encode the 4 scale bytes of every super-block of an NVFP4 row range in place. `w` points at the first block of row 0, rows are
// `row_stride` bytes apart, each row has `nsb` super-blocks (K / 64). Idempotence is NOT guaranteed (a code is also a valid UE4M3 byte):
// call it exactly once per tensor and record that in the owning descriptor (e.g. LayerPacked::nvfp4r = true).
inline void nvfp4r_repack_rows(uint8_t* w, size_t row_stride, int64_t nrows, int nsb) {
    for (int64_t r = 0; r < nrows; ++r) nvfp4r_repack_blocks(w + (size_t)r * row_stride, (size_t)nsb);
}

// Self-check: every byte value round-trips bit-exactly. Returns 0 on success, else 1 + the first failing byte value.
inline int nvfp4r_selfcheck() {
    for (int b = 0; b < 256; ++b) {
        const float want = ue4m3_to_f32((uint8_t)b);
        const uint8_t c = nvfp4r_encode((uint8_t)b);
        const float got = nvfp4r_decode(c);
        uint32_t uw, ug;
        std::memcpy(&uw, &want, 4); std::memcpy(&ug, &got, 4);
        if (uw != ug) return 1 + b;
        alignas(16) uint8_t v[16]; for (int i = 0; i < 16; ++i) v[i] = c;
        alignas(64) float out[16]; _mm512_store_ps(out, nvfp4r_decode16(v));
        std::memcpy(&ug, &out[7], 4);
        if (uw != ug) return 1 + b;
    }
    return 0;
}

}  // namespace ink

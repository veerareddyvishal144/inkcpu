// Scalar reference implementations. They reproduce the lane structure, fused multiply-adds and reduction tree of the
// AVX-512 kernels in gemv.cpp, so the kernels must match them bit-for-bit (integer parts are exact; every float op is
// the same IEEE operation in the same order). Built with -ffp-contract=off. Owner: agent B1.
#include <cmath>

#include "ink/kernels_gemv.h"
#include "ink/nvfp4_repack.h"

namespace ink {

namespace {
// fixed reduction tree: lanes j+8, then j+4, then j+2, then j+1
inline float hsum16_ref(const float a[16]) {
    float t[8], u[4];
    for (int j = 0; j < 8; ++j) t[j] = a[j] + a[j + 8];
    for (int j = 0; j < 4; ++j) u[j] = t[j] + t[j + 4];
    const float v0 = u[0] + u[2];
    const float v1 = u[1] + u[3];
    return v0 + v1;
}
inline const char* rowp(const void* w, size_t rs, int r) { return (const char*)w + (size_t)r * rs; }
}  // namespace

void quantize_row_q8_0_ref(const float* x, block_q8_0* y, int k) {
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            const float v = std::fabs(x[i * QK8_0 + j]);
            amax = amax > v ? amax : v;
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = f32_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j) y[i].qs[j] = (int8_t)std::roundf(x[i * QK8_0 + j] * id);
    }
}

void gemv_q8_0_ref(const void* w, size_t rs, const block_q8_0* x, int k, float* y, int nrows) {
    const int nb = k / QK8_0;
    for (int r = 0; r < nrows; ++r) {
        const block_q8_0* wb = (const block_q8_0*)rowp(w, rs, r);
        float acc[16] = {0};
        for (int b = 0; b < nb; ++b) {
            const int half = b & 1;
            const float s = fp16_to_f32(wb[b].d) * fp16_to_f32(x[b].d);
            for (int j = 0; j < 8; ++j) {
                int I = 0;
                for (int t = 0; t < 4; ++t) I += (int)wb[b].qs[4 * j + t] * (int)x[b].qs[4 * j + t];
                acc[8 * half + j] = std::fmaf((float)I, s, acc[8 * half + j]);
            }
        }
        y[r] = hsum16_ref(acc);
    }
}

void gemv_nvfp4_ref(const void* w, size_t rs, const block_q8_0* x, int k, float* y, int nrows) {
    const int nsb = k / QK_NVFP4;
    for (int r = 0; r < nrows; ++r) {
        const block_nvfp4* wb = (const block_nvfp4*)rowp(w, rs, r);
        float acc[16] = {0};
        for (int sb = 0; sb < nsb; ++sb) {
            const block_q8_0* xb = x + 2 * sb;
            const float dxs[2] = {fp16_to_f32(xb[0].d), fp16_to_f32(xb[1].d)};
            float sc[4];
            for (int s = 0; s < 4; ++s) sc[s] = ue4m3_to_f32(wb[sb].d[s]) * dxs[s >> 1];
            for (int l = 0; l < 16; ++l) {
                int I = 0;
                for (int tt = 0; tt < 4; ++tt) {
                    const int b = 4 * l + tt;     // byte position in the shuffled nibble vector
                    const int t = b >> 3, j = b & 7;
                    const int s = t & 3;          // sub-block
                    const bool hi = t >= 4;
                    const uint8_t q = wb[sb].qs[8 * s + j];
                    const int nib = hi ? (q >> 4) : (q & 15);
                    const int e = 16 * s + (hi ? 8 : 0) + j;  // element within the 64-element super-block
                    I += (int)kvalues_fp4[nib] * (int)xb[e >> 5].qs[e & 31];
                }
                acc[l] = std::fmaf((float)I, sc[(l & 7) >> 1], acc[l]);
            }
        }
        y[r] = hsum16_ref(acc);
    }
}

// re-encoded scales: identical to gemv_nvfp4_ref except for the scale decode (nvfp4r_decode == ue4m3_to_f32 of the original byte)
void gemv_nvfp4r_ref(const void* w, size_t rs, const block_q8_0* x, int k, float* y, int nrows) {
    const int nsb = k / QK_NVFP4;
    for (int r = 0; r < nrows; ++r) {
        const block_nvfp4* wb = (const block_nvfp4*)rowp(w, rs, r);
        float acc[16] = {0};
        for (int sb = 0; sb < nsb; ++sb) {
            const block_q8_0* xb = x + 2 * sb;
            const float dxs[2] = {fp16_to_f32(xb[0].d), fp16_to_f32(xb[1].d)};
            float sc[4];
            for (int s = 0; s < 4; ++s) sc[s] = nvfp4r_decode(wb[sb].d[s]) * dxs[s >> 1];
            for (int l = 0; l < 16; ++l) {
                int I = 0;
                for (int tt = 0; tt < 4; ++tt) {
                    const int b = 4 * l + tt;
                    const int t = b >> 3, j = b & 7;
                    const int s = t & 3;
                    const bool hi = t >= 4;
                    const uint8_t q = wb[sb].qs[8 * s + j];
                    const int nib = hi ? (q >> 4) : (q & 15);
                    const int e = 16 * s + (hi ? 8 : 0) + j;
                    I += (int)kvalues_fp4[nib] * (int)xb[e >> 5].qs[e & 31];
                }
                acc[l] = std::fmaf((float)I, sc[(l & 7) >> 1], acc[l]);
            }
        }
        y[r] = hsum16_ref(acc);
    }
}

void gemv_bf16_ref(const void* w, size_t rs, const float* x, int k, float* y, int nrows) {
    for (int r = 0; r < nrows; ++r) {
        const uint16_t* wr = (const uint16_t*)rowp(w, rs, r);
        float a0[16] = {0}, a1[16] = {0};
        int i = 0;
        for (; i + 32 <= k; i += 32)
            for (int l = 0; l < 16; ++l) {
                a0[l] = std::fmaf(bf16_to_f32(wr[i + l]), x[i + l], a0[l]);
                a1[l] = std::fmaf(bf16_to_f32(wr[i + 16 + l]), x[i + 16 + l], a1[l]);
            }
        if (i < k)
            for (int l = 0; l < 16; ++l) a0[l] = std::fmaf(bf16_to_f32(wr[i + l]), x[i + l], a0[l]);
        float s[16];
        for (int l = 0; l < 16; ++l) s[l] = a0[l] + a1[l];
        y[r] = hsum16_ref(s);
    }
}

void gemv_f32_ref(const void* w, size_t rs, const float* x, int k, float* y, int nrows) {
    for (int r = 0; r < nrows; ++r) {
        const float* wr = (const float*)rowp(w, rs, r);
        float a0[16] = {0}, a1[16] = {0};
        int i = 0;
        for (; i + 32 <= k; i += 32)
            for (int l = 0; l < 16; ++l) {
                a0[l] = std::fmaf(wr[i + l], x[i + l], a0[l]);
                a1[l] = std::fmaf(wr[i + 16 + l], x[i + 16 + l], a1[l]);
            }
        if (i < k)
            for (int l = 0; l < 16; ++l) a0[l] = std::fmaf(wr[i + l], x[i + l], a0[l]);
        float s[16];
        for (int l = 0; l < 16; ++l) s[l] = a0[l] + a1[l];
        y[r] = hsum16_ref(s);
    }
}

}  // namespace ink

// Shared helpers of the core tests (header only).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace tutil {

inline const char* gguf_path() {
    const char* e = std::getenv("INK_GGUF");
    return e ? e : "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf";
}

template <class T>
inline bool read_raw(const std::string& path, std::vector<T>* out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out->resize((size_t)n / sizeof(T));
    size_t got = std::fread(out->data(), sizeof(T), out->size(), f);
    std::fclose(f);
    return got == out->size();
}

inline std::vector<int> topk(const float* x, int n, int k) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    idx.resize(k);
    return idx;
}

struct Cmp {
    int argmax_a = -1, argmax_b = -1;
    int overlap10 = 0;
    double cosine = 0, max_abs = 0, rel_l2 = 0, top1_gap = 0;
};

inline Cmp compare(const float* a, const float* b, int n) {
    Cmp c;
    double dot = 0, na = 0, nb = 0, nd = 0;
    for (int i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
        double d = (double)a[i] - b[i];
        nd += d * d;
        c.max_abs = std::max(c.max_abs, std::fabs(d));
    }
    c.cosine = dot / std::sqrt(na * nb);
    c.rel_l2 = std::sqrt(nd / nb);
    auto ta = topk(a, n, 10), tb = topk(b, n, 10);
    c.argmax_a = ta[0];
    c.argmax_b = tb[0];
    for (int x : ta)
        for (int y : tb)
            if (x == y) { ++c.overlap10; break; }
    return c;
}

inline double loadavg1() {
    double l = -1;
    FILE* f = std::fopen("/proc/loadavg", "r");
    if (f) { if (std::fscanf(f, "%lf", &l) != 1) l = -1; std::fclose(f); }
    return l;
}

}  // namespace tutil

#include "ink/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

namespace ink {

// ---------------------------------------------------------------- ggml type table (types that can occur in GGUFs we read)
namespace {
struct TypeInfo { int id; const char* name; int64_t blck; size_t bytes; };
const TypeInfo kTypes[] = {
    {0, "F32", 1, 4},      {1, "F16", 1, 2},      {2, "Q4_0", 32, 18},   {3, "Q4_1", 32, 20},   {6, "Q5_0", 32, 22},
    {7, "Q5_1", 32, 24},   {8, "Q8_0", 32, 34},   {9, "Q8_1", 32, 36},   {10, "Q2_K", 256, 84}, {11, "Q3_K", 256, 110},
    {12, "Q4_K", 256, 144},{13, "Q5_K", 256, 176},{14, "Q6_K", 256, 210},{15, "Q8_K", 256, 292},
    {24, "I8", 1, 1},      {25, "I16", 1, 2},     {26, "I32", 1, 4},     {27, "I64", 1, 8},     {28, "F64", 1, 8},
    {30, "BF16", 1, 2},    {39, "MXFP4", 32, 17}, {40, "NVFP4", 64, 36},
};
const TypeInfo* find_type(int t) {
    for (const auto& ti : kTypes) if (ti.id == t) return &ti;
    return nullptr;
}

size_t scalar_size(uint32_t t) {
    switch (t) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0;
    }
}

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <class T> T get() {
        T v{};
        if ((size_t)(end - p) < sizeof(T)) { ok = false; return v; }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    bool skip(size_t n) {
        if ((size_t)(end - p) < n) { ok = false; return false; }
        p += n;
        return true;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (!ok || (uint64_t)(end - p) < n) { ok = false; return {}; }
        std::string s((const char*)p, (size_t)n);
        p += n;
        return s;
    }
    void skip_str() {
        uint64_t n = get<uint64_t>();
        if (!ok || (uint64_t)(end - p) < n) { ok = false; return; }
        p += n;
    }
};

template <class T> T load(const uint8_t* p) { T v; std::memcpy(&v, p, sizeof(T)); return v; }

bool as_i64(GgufValueType t, const uint8_t* p, int64_t* out) {
    switch (t) {
        case GGUF_U8: *out = load<uint8_t>(p); return true;
        case GGUF_I8: *out = load<int8_t>(p); return true;
        case GGUF_U16: *out = load<uint16_t>(p); return true;
        case GGUF_I16: *out = load<int16_t>(p); return true;
        case GGUF_U32: *out = load<uint32_t>(p); return true;
        case GGUF_I32: *out = load<int32_t>(p); return true;
        case GGUF_U64: *out = (int64_t)load<uint64_t>(p); return true;
        case GGUF_I64: *out = load<int64_t>(p); return true;
        case GGUF_BOOL: *out = load<uint8_t>(p) != 0; return true;
        default: return false;
    }
}
bool as_f64(GgufValueType t, const uint8_t* p, double* out) {
    if (t == GGUF_F32) { *out = load<float>(p); return true; }
    if (t == GGUF_F64) { *out = load<double>(p); return true; }
    int64_t i;
    if (as_i64(t, p, &i)) { *out = (double)i; return true; }
    return false;
}
}  // namespace

int64_t     ggml_blck_size(int type) { const TypeInfo* t = find_type(type); return t ? t->blck : 0; }
size_t      ggml_type_bytes(int type) { const TypeInfo* t = find_type(type); return t ? t->bytes : 0; }
const char* ggml_type_name(int type) { const TypeInfo* t = find_type(type); return t ? t->name : "?"; }

// ---------------------------------------------------------------- open
std::unique_ptr<GgufFile> GgufFile::open(const std::string& path, std::string* err) {
    auto fail = [&](const std::string& m) -> std::unique_ptr<GgufFile> {
        if (err) *err = m;
        return nullptr;
    };
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return fail("cannot open " + path);
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 32) { ::close(fd); return fail("cannot stat / file too small: " + path); }
    void* m = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) return fail("mmap failed for " + path);

    std::unique_ptr<GgufFile> f(new GgufFile());
    f->map_ = m;
    f->size_ = (size_t)st.st_size;
    f->base_ = (const uint8_t*)m;

    Reader r{f->base_, f->base_ + f->size_};
    char magic[4];
    std::memcpy(magic, r.p, 4);
    r.p += 4;
    if (std::memcmp(magic, "GGUF", 4) != 0) return fail("bad magic (not a GGUF file)");
    f->version_ = r.get<uint32_t>();
    if (f->version_ != 3) return fail("unsupported GGUF version " + std::to_string(f->version_) + " (need 3)");
    const uint64_t n_tensors = r.get<uint64_t>();
    const uint64_t n_kv = r.get<uint64_t>();
    if (!r.ok || n_tensors > (1u << 24) || n_kv > (1u << 24)) return fail("corrupt header counts");

    f->kvs_.reserve(n_kv);
    for (uint64_t i = 0; i < n_kv; ++i) {
        GgufKV kv;
        kv.key = r.str();
        kv.type = (GgufValueType)r.get<uint32_t>();
        if (!r.ok) return fail("truncated KV table");
        if (kv.type == GGUF_ARRAY) {
            kv.elem_type = (GgufValueType)r.get<uint32_t>();
            kv.count = r.get<uint64_t>();
            kv.data = r.p;
            if (!r.ok) return fail("truncated KV array header: " + kv.key);
            if (kv.elem_type == GGUF_STRING) {
                for (uint64_t j = 0; j < kv.count && r.ok; ++j) r.skip_str();
            } else if (kv.elem_type == GGUF_ARRAY) {
                return fail("nested arrays unsupported: " + kv.key);
            } else {
                size_t es = scalar_size(kv.elem_type);
                if (!es) return fail("bad array element type: " + kv.key);
                r.skip(es * kv.count);
            }
        } else if (kv.type == GGUF_STRING) {
            kv.elem_type = GGUF_STRING;
            kv.data = r.p;
            r.skip_str();
        } else {
            size_t es = scalar_size(kv.type);
            if (!es) return fail("bad value type: " + kv.key);
            kv.elem_type = kv.type;
            kv.data = r.p;
            r.skip(es);
        }
        if (!r.ok) return fail("truncated KV payload: " + kv.key);
        f->kv_index_[kv.key] = f->kvs_.size();
        f->kvs_.push_back(std::move(kv));
    }

    uint32_t al = 0;
    if (f->get_u32("general.alignment", &al) && al) f->alignment_ = al;

    f->tensors_.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensorInfo t;
        t.name = r.str();
        t.n_dims = (int)r.get<uint32_t>();
        if (!r.ok || t.n_dims < 1 || t.n_dims > 4) return fail("bad tensor dims");
        for (int d = 0; d < t.n_dims; ++d) t.ne[d] = (int64_t)r.get<uint64_t>();
        t.type = (int)r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        if (!r.ok) return fail("truncated tensor table");
        f->tensor_index_[t.name] = f->tensors_.size();
        f->tensors_.push_back(std::move(t));
    }
    const uint64_t header_end = (uint64_t)(r.p - f->base_);
    f->data_offset_ = (header_end + f->alignment_ - 1) / f->alignment_ * f->alignment_;
    for (auto& t : f->tensors_) {
        const int64_t b = ggml_blck_size(t.type);
        if (!b) return fail("unsupported ggml type " + std::to_string(t.type) + " in tensor " + t.name);
        if (t.ne[0] % b) return fail("row length not a multiple of the block size: " + t.name);
        t.nbytes = (size_t)(t.nelements() / b) * ggml_type_bytes(t.type);
        if (f->data_offset_ + t.offset + t.nbytes > f->size_) return fail("tensor data out of file bounds: " + t.name);
        t.data = f->base_ + f->data_offset_ + t.offset;
    }
    return f;
}

GgufFile::~GgufFile() {
    if (map_) munmap(map_, size_);
}

const GgufTensorInfo* GgufFile::tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}
const GgufKV* GgufFile::kv(const std::string& key) const {
    auto it = kv_index_.find(key);
    return it == kv_index_.end() ? nullptr : &kvs_[it->second];
}

bool GgufFile::get_i64(const std::string& key, int64_t* out) const {
    const GgufKV* k = kv(key);
    if (!k || k->type == GGUF_ARRAY || k->type == GGUF_STRING) return false;
    return as_i64(k->type, k->data, out);
}
bool GgufFile::get_u32(const std::string& key, uint32_t* out) const {
    int64_t v;
    if (!get_i64(key, &v) || v < 0 || v > 0xFFFFFFFFll) return false;
    *out = (uint32_t)v;
    return true;
}
bool GgufFile::get_f32(const std::string& key, float* out) const {
    const GgufKV* k = kv(key);
    if (!k || k->type == GGUF_ARRAY || k->type == GGUF_STRING) return false;
    double d;
    if (!as_f64(k->type, k->data, &d)) return false;
    *out = (float)d;
    return true;
}
bool GgufFile::get_bool(const std::string& key, bool* out) const {
    int64_t v;
    if (!get_i64(key, &v)) return false;
    *out = v != 0;
    return true;
}
bool GgufFile::get_string(const std::string& key, std::string* out) const {
    const GgufKV* k = kv(key);
    if (!k || k->type != GGUF_STRING) return false;
    uint64_t n = load<uint64_t>(k->data);
    out->assign((const char*)k->data + 8, (size_t)n);
    return true;
}
uint64_t GgufFile::arr_len(const std::string& key) const {
    const GgufKV* k = kv(key);
    return (k && k->type == GGUF_ARRAY) ? k->count : 0;
}
bool GgufFile::get_arr_i64(const std::string& key, std::vector<int64_t>* out) const {
    const GgufKV* k = kv(key);
    if (!k || k->type != GGUF_ARRAY || k->elem_type == GGUF_STRING || k->elem_type == GGUF_F32 || k->elem_type == GGUF_F64) return false;
    const size_t es = scalar_size(k->elem_type);
    out->resize(k->count);
    for (uint64_t i = 0; i < k->count; ++i) as_i64(k->elem_type, k->data + i * es, &(*out)[i]);
    return true;
}
bool GgufFile::get_arr_f32(const std::string& key, std::vector<float>* out) const {
    const GgufKV* k = kv(key);
    if (!k || k->type != GGUF_ARRAY || k->elem_type == GGUF_STRING) return false;
    const size_t es = scalar_size(k->elem_type);
    out->resize(k->count);
    for (uint64_t i = 0; i < k->count; ++i) {
        double d;
        as_f64(k->elem_type, k->data + i * es, &d);
        (*out)[i] = (float)d;
    }
    return true;
}
bool GgufFile::get_arr_bool(const std::string& key, std::vector<bool>* out) const {
    std::vector<int64_t> v;
    if (!get_arr_i64(key, &v)) return false;
    out->assign(v.size(), false);
    for (size_t i = 0; i < v.size(); ++i) (*out)[i] = v[i] != 0;
    return true;
}
bool GgufFile::get_arr_strings(const std::string& key, std::vector<std::string>* out) const {
    const GgufKV* k = kv(key);
    if (!k || k->type != GGUF_ARRAY || k->elem_type != GGUF_STRING) return false;
    out->clear();
    out->reserve(k->count);
    const uint8_t* p = k->data;
    for (uint64_t i = 0; i < k->count; ++i) {
        uint64_t n = load<uint64_t>(p);
        out->emplace_back((const char*)p + 8, (size_t)n);
        p += 8 + n;
    }
    return true;
}

}  // namespace ink

// Minimal mmap-based GGUF v3 reader. Owner: agent A (core).
//
// Opening is O(header): the file is mmap'ed read-only (no copies, no page-in of tensor data); the KV table and tensor
// table are indexed once (string arrays such as the tokenizer are skipped, not materialised). Tensor data pointers point
// straight into the mapping. The GgufFile must outlive everything that holds a tensor pointer.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ink {

enum GgufValueType : uint32_t {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4, GGUF_I32 = 5, GGUF_F32 = 6, GGUF_BOOL = 7,
    GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_U64 = 10, GGUF_I64 = 11, GGUF_F64 = 12,
};

// ggml type helpers (block size in elements, bytes per block). Unknown types return 0.
int64_t     ggml_blck_size(int type);
size_t      ggml_type_bytes(int type);   // bytes per block
const char* ggml_type_name(int type);
inline size_t ggml_row_bytes(int type, int64_t ne0) {
    const int64_t b = ggml_blck_size(type);
    return b ? (size_t)(ne0 / b) * ggml_type_bytes(type) : 0;
}

struct GgufTensorInfo {
    std::string    name;
    int            type = 0;                  // ggml type id (see include/ink/ggml_types.h)
    int            n_dims = 0;
    int64_t        ne[4] = {1, 1, 1, 1};      // ne[0] is the contiguous (fastest) dimension, like ggml
    uint64_t       offset = 0;                // from the start of the data section
    const uint8_t* data = nullptr;            // pointer into the mmap
    size_t         nbytes = 0;
    int64_t nelements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
    size_t  row_bytes() const { return ggml_row_bytes(type, ne[0]); }
};

struct GgufKV {
    std::string    key;
    GgufValueType  type = GGUF_U8;            // GGUF_ARRAY for arrays
    GgufValueType  elem_type = GGUF_U8;       // element type of arrays (== type for scalars)
    uint64_t       count = 1;                 // array length (1 for scalars)
    const uint8_t* data = nullptr;            // raw payload: scalar value / first array element / string u64 length prefix
};

class GgufFile {
public:
    // Returns nullptr on failure and fills *err.
    static std::unique_ptr<GgufFile> open(const std::string& path, std::string* err = nullptr);
    ~GgufFile();
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    uint32_t       version() const { return version_; }
    uint64_t       alignment() const { return alignment_; }
    uint64_t       data_offset() const { return data_offset_; }
    size_t         file_size() const { return size_; }
    const uint8_t* base() const { return base_; }

    const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }
    const GgufTensorInfo*              tensor(const std::string& name) const;   // nullptr if absent
    const std::vector<GgufKV>&         kvs() const { return kvs_; }
    const GgufKV*                      kv(const std::string& key) const;        // nullptr if absent

    // Scalar getters accept any integer (or float) KV type that converts without loss of meaning; return false if absent.
    bool get_u32(const std::string& key, uint32_t* out) const;
    bool get_i64(const std::string& key, int64_t* out) const;
    bool get_f32(const std::string& key, float* out) const;
    bool get_bool(const std::string& key, bool* out) const;
    bool get_string(const std::string& key, std::string* out) const;
    // Array getters (numeric / bool arrays); false if absent or not an array of a convertible type.
    bool get_arr_i64(const std::string& key, std::vector<int64_t>* out) const;
    bool get_arr_f32(const std::string& key, std::vector<float>* out) const;
    bool get_arr_bool(const std::string& key, std::vector<bool>* out) const;
    // Copies the strings (can be large for the tokenizer arrays, only call on demand).
    bool get_arr_strings(const std::string& key, std::vector<std::string>* out) const;
    uint64_t arr_len(const std::string& key) const;                             // 0 if absent / not an array

private:
    GgufFile() = default;
    void*          map_ = nullptr;
    size_t         size_ = 0;
    const uint8_t* base_ = nullptr;
    uint32_t       version_ = 0;
    uint64_t       alignment_ = 32;
    uint64_t       data_offset_ = 0;
    std::vector<GgufKV>                          kvs_;
    std::vector<GgufTensorInfo>                  tensors_;
    std::unordered_map<std::string, size_t>      kv_index_;
    std::unordered_map<std::string, size_t>      tensor_index_;
};

}  // namespace ink

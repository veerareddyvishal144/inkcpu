// Shared helpers of the llama.cpp oracle tools (oracle_logits, oracle_greedy).
// Owner: agent E. Links against the stock build-sys libs; nothing here touches ggml internals.
#pragma once
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#ifndef ORACLE_LLAMA_HASH
#define ORACLE_LLAMA_HASH "unknown"
#endif

namespace oc {

inline bool g_verbose = false;  // ORACLE_VERBOSE=1 keeps llama.cpp's INFO log

inline void log_cb(enum ggml_log_level level, const char * text, void *) {
    static bool show = true;
    if (level != GGML_LOG_LEVEL_CONT) {
        show = g_verbose || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR;
    }
    if (show) {
        fputs(text, stderr);
    }
}

struct Args {
    std::string model;
    std::string tokens_file;
    std::string prompt;
    bool have_prompt = false;
    std::string out;
    int  ctx = 0;          // 0 = automatic (just big enough)
    int  threads = 8;
    int  n_gen = 64;       // oracle_greedy only
    bool all_logits = false;  // oracle_logits only
    std::vector<std::string> overrides;
};

inline bool parse_args(int argc, char ** argv, Args & a, bool greedy) {
    for (int i = 1; i < argc; i++) {
        const std::string s = argv[i];
        auto need = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", what);
                exit(2);
            }
            return argv[++i];
        };
        if (s == "-m" || s == "--model")            a.model = need("-m");
        else if (s == "--tokens-file")              a.tokens_file = need("--tokens-file");
        else if (s == "--prompt")                   { a.prompt = need("--prompt"); a.have_prompt = true; }
        else if (s == "--ctx")                      a.ctx = atoi(need("--ctx"));
        else if (s == "--threads" || s == "-t")     a.threads = atoi(need("--threads"));
        else if (s == "--override-kv")              a.overrides.push_back(need("--override-kv"));
        else if (s == "--out")                      a.out = need("--out");
        else if (s == "--all-logits" && !greedy)    a.all_logits = true;
        else if ((s == "-n" || s == "--n-predict") && greedy) a.n_gen = atoi(need("-n"));
        else {
            fprintf(stderr, "unknown argument: %s\n", s.c_str());
            return false;
        }
    }
    if (a.model.empty() || a.out.empty()) return false;
    if (a.tokens_file.empty() == !a.have_prompt) {
        fprintf(stderr, "give exactly one of --tokens-file / --prompt\n");
        return false;
    }
    return true;
}

// "key=int:123", "key=float:1.5", "key=bool:true", "key=str:text"  (same grammar as llama.cpp --override-kv)
inline bool parse_override(const std::string & s, llama_model_kv_override & kvo) {
    const size_t eq = s.find('=');
    if (eq == std::string::npos) return false;
    const std::string key = s.substr(0, eq), rest = s.substr(eq + 1);
    const size_t col = rest.find(':');
    if (col == std::string::npos || key.size() >= sizeof(kvo.key)) return false;
    const std::string type = rest.substr(0, col), val = rest.substr(col + 1);
    memset(&kvo, 0, sizeof(kvo));
    memcpy(kvo.key, key.c_str(), key.size());
    if (type == "int")        { kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;   kvo.val_i64 = strtoll(val.c_str(), nullptr, 0); }
    else if (type == "float") { kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT; kvo.val_f64 = strtod(val.c_str(), nullptr); }
    else if (type == "bool")  { kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;  kvo.val_bool = (val == "true" || val == "1"); }
    else if (type == "str")   { kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;   strncpy(kvo.val_str, val.c_str(), sizeof(kvo.val_str) - 1); }
    else return false;
    return true;
}

inline llama_model * load_model(const Args & a) {
    g_verbose = getenv("ORACLE_VERBOSE") != nullptr;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    static std::vector<llama_model_kv_override> kvo;  // must outlive the load call; list ends with an empty key
    for (const auto & o : a.overrides) {
        llama_model_kv_override k;
        if (!parse_override(o, k)) {
            fprintf(stderr, "bad --override-kv '%s' (want key=int|float|bool|str:value)\n", o.c_str());
            exit(2);
        }
        kvo.push_back(k);
    }
    llama_model_params mp = llama_model_default_params();
    if (!kvo.empty()) {
        llama_model_kv_override end;
        memset(&end, 0, sizeof(end));
        kvo.push_back(end);
        mp.kv_overrides = kvo.data();
    }
    llama_model * m = llama_model_load_from_file(a.model.c_str(), mp);
    if (!m) fprintf(stderr, "failed to load %s\n", a.model.c_str());
    return m;
}

inline std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    // add_special = true: BOS/EOS are added only if the GGUF says so (this GGUF: add_bos_token=false, add_eos_token=false,
    //                     so nothing is added); parse_special = true: "<|message_user|>" etc. become single special tokens.
    int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, true);
    if (n <= 0) return {};
    std::vector<llama_token> t(n);
    const int r = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), t.data(), (int32_t) t.size(), true, true);
    if (r < 0) return {};
    t.resize(r);
    return t;
}

inline std::vector<llama_token> read_tokens_file(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    for (char & c : s) if (c == ',' || c == '[' || c == ']') c = ' ';
    std::istringstream is(s);
    std::vector<llama_token> t;
    long long v;
    while (is >> v) t.push_back((llama_token) v);
    return t;
}

inline std::string piece(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::string s(-n, '\0');
        n = llama_token_to_piece(vocab, tok, &s[0], (int32_t) s.size(), 0, true);
        return n > 0 ? s.substr(0, n) : std::string();
    }
    return std::string(buf, n);
}

inline std::string escape(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c == '\\') o += "\\\\";
        else if (c < 0x20 || c == 0x7f) { char b[8]; snprintf(b, sizeof b, "\\x%02x", c); o += b; }
        else o += (char) c;
    }
    return o;
}

inline std::string json_escape(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char) c;
    }
    return o;
}

inline void mkdir_p(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); i++) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) mkdir(cur.c_str(), 0755);
    }
}

inline void write_file(const std::string & path, const void * data, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f || (n && fwrite(data, 1, n, f) != n)) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    fclose(f);
}

inline int argmax(const float * x, int n) {  // lowest id wins ties (same as llama.cpp's greedy sampler)
    int b = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

inline std::string meta_json(const Args & a, const llama_vocab * vocab, int n_vocab, int n_tokens, int n_ctx) {
    std::ostringstream o;
    o << "{\n"
      << "  \"model\": \"" << json_escape(a.model) << "\",\n"
      << "  \"llama_cpp_commit\": \"" << ORACLE_LLAMA_HASH << "\",\n"
      << "  \"n_vocab_logits\": " << n_vocab << ",\n"
      << "  \"n_tokens\": " << n_tokens << ",\n"
      << "  \"n_ctx\": " << n_ctx << ",\n"
      << "  \"threads\": " << a.threads << ",\n"
      << "  \"input\": \"" << (a.tokens_file.empty() ? "prompt (llama_tokenize add_special=true parse_special=true)" : "tokens-file (ids used verbatim)") << "\",\n"
      << "  \"vocab_add_bos\": " << (llama_vocab_get_add_bos(vocab) ? "true" : "false") << ",\n"
      << "  \"vocab_add_eos\": " << (llama_vocab_get_add_eos(vocab) ? "true" : "false") << ",\n"
      << "  \"bos_id\": " << llama_vocab_bos(vocab) << ",\n"
      << "  \"eos_id\": " << llama_vocab_eos(vocab) << ",\n"
      << "  \"override_kv\": [";
    for (size_t i = 0; i < a.overrides.size(); i++) o << (i ? ", " : "") << "\"" << json_escape(a.overrides[i]) << "\"";
    o << "]\n}\n";
    return o.str();
}

}  // namespace oc

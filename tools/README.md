# tools/ (agent E)

Oracle tools built against the STOCK llama.cpp libs (`/data/inkling-mini/llama.cpp/build-sys/bin`, @2a28df0f9; linked only, never
rebuilt). Everything below is plain text or headerless little-endian raw binary (numpy: `np.fromfile(path, '<f4')`). The logits come from llama.cpp's
CPU kernels at 8 threads. Measured on `s1_en_capital`: a rerun at 8 threads (with a no-op `--override-kv tokenizer.ggml.add_bos_token=bool:false`, which also
exercises that path) and a run at 4 threads are bit-identical to the stored `logits_last.f32` (max |diff| 0), so the result does not depend on the thread count here;
a different CPU/ISA (other kernels) is not covered.
`tools/build/` and `tools/build-nprof/` are build outputs (candidates for .gitignore).

## Build

```
export PATH=/usr/bin:/bin
cmake -S /data/inkling-mini/engine/tools -B /data/inkling-mini/engine/tools/build && cmake --build /data/inkling-mini/engine/tools/build -j4
```
Binaries: `tools/build/oracle_logits`, `tools/build/oracle_greedy`. The CMake file also works when the root CMakeLists adds `tools/`.
`ORACLE_VERBOSE=1` turns llama.cpp's INFO log back on. Expected warning at load: `special_eos_id is not in special_eog_ids`
(llama.cpp does not list token 200006 as end-of-generation for this GGUF: servers must stop on 200006 themselves).

## oracle_logits

```
oracle_logits -m GGUF (--tokens-file ids.txt | --prompt "text") [--ctx N] [--override-kv key=int:val ...] [--threads N]
              [--all-logits] --out DIR
```
- The whole prompt is decoded as ONE batch/ubatch (`n_batch = n_ubatch = n_tokens`), default llama.cpp context settings otherwise
  (F16 KV, flash-attn auto, full-size SWA cache), CPU only. `--ctx` defaults to n_tokens (llama.cpp pads it internally).
- `--override-kv` uses the llama.cpp grammar `key=int|float|bool|str:value` (repeatable) and is applied at model load.
- Tokenisation with `--prompt`: `llama_tokenize(vocab, text, add_special=true, parse_special=true)`.
  This GGUF has `tokenizer.ggml.add_bos_token=false`, `add_eos_token=false` and NO BOS id (`bos_id` -1), so NOTHING is added: token 0 is the
  first text token. `parse_special=true` turns marker strings in the text (`<|message_user|>`, `<|content_text|>`, `<|end_message|>`,
  `<|message_model|>` ...) into their single special-token ids (no leading space is inserted). The chat prompt in the data set is the user
  turn from ARCH.md section 4 followed by the generation marker: `<|message_user|><|content_text|>...<|end_message|><|message_model|>`
  (no system/effort message, no tools).
- `--tokens-file`: whitespace/comma separated decimal ids, used VERBATIM (nothing added, no tokenizer involved).

Output files in DIR:

| file | format |
|---|---|
| `tokens.i32` | little-endian int32 x n_tokens, no header: the exact token ids that were evaluated |
| `logits_last.f32` | little-endian float32 x 200058 (`n_vocab_logits` in meta.json): logits of the LAST position as llama.cpp returns them (after the final `30*tanh(x/30)` softcap, padded vocabulary rows 200058..201023 already dropped by the graph) |
| `logits_all.f32` | only with `--all-logits`: float32, row-major `[n_tokens][200058]`, row i = logits after consuming token i |
| `top20.txt` | text, one line per rank: `rank token_id logit piece` (piece escaped: `\n`, `\t`, `\\`, `\xNN`; ties broken by lower id) |
| `meta.json` | model path, llama.cpp commit, n_tokens, n_ctx, threads, tokenizer flags, bos/eos ids, kv overrides |
| `prompt.txt` | the exact prompt bytes (only when `--prompt` was used) |

## oracle_greedy

```
oracle_greedy -m GGUF (--tokens-file ids.txt | --prompt "text") [-n 64] [--ctx N] [--threads N] [--override-kv ...] --out DIR
```
Prompt as one batch, then `-n` greedy (argmax, lowest id on ties) tokens fed back one per decode. EOS is NOT special: exactly N tokens.
Files: `prompt_tokens.i32` / `gen_tokens.i32` (int32, no header), `gen_text.txt` (pieces concatenated, special tokens rendered),
`steps.txt` (`step token_id top1_logit top2_id top2_logit margin piece`; a small margin marks a near-tie where another engine may
legitimately diverge), `prompt.txt`, `meta.json`.

## Data set (`make_oracle.sh logits|greedy|all`, 8 threads, regenerates everything below)

`tools/oracle/<name>/` (logits) and `tools/oracle/greedy/<name>/` (5 prompts x 64 tokens: s1, s2, m1, m2, c1):

| name | tokens | content |
|---|---|---|
| `s1_en_capital` | 13 | `The capital of France is Paris, and the capital of Germany is` (+ `logits_all.f32`, 10.4 MB, the only copy of per-position logits) |
| `s2_code_add` | 14 | Python `def add(a, b):` two lines |
| `s3_en_story` | 15 | `Once upon a time, in a small village by the sea, there lived` |
| `m1_en_llm` | 43 | two English sentences about language models |
| `m2_code_c` | 52 | a C function `sum` (indentation, braces) |
| `c1_chat_user` | 14 | chat-wrapped user turn + `<|message_model|>`; top-1 is `<|content_thinking|>` (27.2 vs 3.8) |

The prompt texts are in the `prompt.txt` of each directory and in `make_oracle.sh`.

## Routing study (docs/ROUTING.md)

- Trace patch (local, in the llama.cpp checkout): `GGML_ROUTE_TRACE=<file>` in `ggml/src/ggml-cpu/ggml-cpu.c`, built into `build-nprof`
  (`cmake --build build-nprof --target llama-perplexity`; `build-sys` is untouched). Record format: `int32 header[8] = {0x52545243, fwd, visit,
  layer, n_tokens, n_expert_used(8), 0, 0}` + `int32 ids[n_tokens][8]`, one record per `gate_up_exps` MUL_MAT_ID (48 per forward).
- `route_traces/{wiki,code,chat}.rtrace`: traces (wiki = 4 x 512 tokens via llama-perplexity, code = 1255 tokens, chat = 895 tokens, both through
  `oracle_logits` linked to build-nprof: `cmake -S tools -B tools/build-nprof -DINK_LLAMA_BIN=<llama.cpp>/build-nprof/bin`); inputs in `route_data/`.
  Warm-up forwards (1 token) are in the files and skipped by `--min-tokens`.
- `route_stats.py wiki=route_traces/wiki.rtrace code=... chat=... --out route_traces/route_stats_output.md` (interpreter `/data/engine/.venv/bin/python`).

## Bytes per token and baseline (docs/BASELINE.md)

- `gguf_bytes.py [--gguf PATH] [--json tools/bytes_per_token.json] [--markdown FILE]`: tensor-table accounting (python, gguf-py from the checkout).
- `bench_llama.sh [OUT.csv]`: stock `build-sys/bin/llama-bench` decode (`-p 0 -n 64`) and prefill (`-p 256 -n 0`) for threads 8, 12, 16 with
  `OMP_WAIT_POLICY=active OMP_PLACES=cores OMP_PROC_BIND=close`, `-r 3`; waits up to 10 min for the 1-min load average to be <= 2 (`MAX_LOAD`, `WAIT_MAX_S`) and
  otherwise exits 3 without running. One CSV row per invocation (`bench_baseline.csv`): date, load before/after, load of other processes, llama.cpp commit, test,
  threads, tokens/s, stddev, reps, command.

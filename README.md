# inkcpu

Run **Inkling 2 Mini** (Thinking Machines' 26B mixture-of-experts model) on an ordinary x86 server with **no GPU** — 2.8–3.9× faster than llama.cpp on the same machine, with an OpenAI-compatible API you can point your editor or agent at.

One model, one CPU class, built from scratch: GGUF reader, AVX-512 kernels, a decode executor with one thread-pool run per token, batched prefill, DSpark speculative decoding, slot checkpoint/rewind for prompt reuse, and a FastAPI server. Everything is measured against stock llama.cpp on the same GGUF and the same host.

## How fast

Xeon Platinum 8171M (16 cores, AVX-512, no VNNI/AMX), 125 GB RAM, no GPU. Same GGUF (Unsloth `Q4_K_XL`, 17.6 GB), idle host, 3 runs.

| | tuned llama.cpp | **inkcpu** | speed-up |
|---|---|---|---|
| Generation (decode), 12 threads | 5.5 tok/s | **15.3 tok/s** | 2.8× |
| Generation, 16 threads | 4.7 tok/s | **14.7–15.8 tok/s** | 3.1–3.4× |
| Generation at 8k / 16k tokens of context, 16 threads | n/a | **13.8 / 13.0 tok/s** | |
| Prompt processing (prefill), 256 tokens, 16 threads | 22 tok/s | **87 tok/s** | 3.9× |
| Coding-agent task through the API (same 7-test task) | 643 s | **207–291 s** | ~2.5× |
| Follow-up turn in an editor thread (7k-token re-rendered prompt) | full re-process | **1.8 s** (prefix rewind) | 68× |
| Memory | 17.6 GB | 17 GB resident (34 GB peak while loading) | |

Outputs match llama.cpp: identical greedy text on the test prompts, next-token log-likelihood within 0.005 nats on 2,400-token real text. Numbers from the 2026-11-23 idle-host run (`tools/final_bench_wave1.sh`); run-to-run spread on this VM is about 5 %. Details, per-phase profiles, what limits the speed and what is left to try: [`docs/ENGINE.md`](docs/ENGINE.md), [`docs/RESEARCH_WAVE2.md`](docs/RESEARCH_WAVE2.md).

What a CPU still can't hide: every *new* prompt token costs ~12–15 ms, so a 10k-token tool result takes ~2 minutes to read. Repeated prefixes (system prompts, conversation history) are nearly free thanks to the slot cache and rewind.

## Requirements

- Linux x86-64 with **AVX-512 F/BW/DQ/VL** (Skylake-SP / Cascade Lake or newer; AMD Zen 4 also qualifies). No GPU needed.
- 40 GB RAM free to load (17 GB steady), 20 GB disk for the model files.
- GCC 11+, CMake 3.20+, Ninja; Python 3.10+.

## Install

```bash
git clone https://github.com/veerareddyvishal144/inkcpu.git && cd inkcpu
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build ink
python3 -m venv .venv && .venv/bin/pip install -r server/requirements.txt
```

## Run

```bash
tools/serve.sh start        # first run downloads the model (17.6 GB) + drafter (0.4 GB) from Hugging Face, then serves on 127.0.0.1:8000
tools/serve.sh status       # health, memory
tools/serve.sh logs
tools/serve.sh stop
```
Set `HF_TOKEN` before the first start to avoid Hugging Face's unauthenticated-download throttling. Tunables are environment variables (`INK_THREADS`, `INK_CTX`, `INK_SLOTS`, `INK_SPEC`, `INK_MODEL_DIR`, …) — see the header of [`tools/serve.sh`](tools/serve.sh).

Ask it something:
```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "inkling-2-mini", "max_tokens": 400, "temperature": 0, "reasoning_effort": "low",
  "messages": [{"role":"user","content":"Write a Python function that parses ISO-8601 dates."}]
}'
```
The model thinks before answering (`reasoning_content`); `reasoning_effort` takes `none | minimal | low | medium | high` or a number 0–0.99. Requests that omit it get `none` (direct answer) by default — change with `INK_DEFAULT_EFFORT`.

**From your laptop:** the server listens on localhost only. Tunnel it: `ssh -N -L 8000:127.0.0.1:8000 user@server`, then use `http://localhost:8000/v1` in any OpenAI-compatible client (Cursor, Caret, Continue, aider, the OpenAI SDK — model id `inkling-2-mini`). Tools, streaming, `reasoning_content`, prefix caching and `/metrics` are supported; embeddings, vision and logprobs are not.

## Verify

```bash
ninja -C build test_engine test_prefill test_spec
INK_THREADS=12 build/src/engine/test_engine oracle          # logits vs llama.cpp on the bundled oracle prompts
INK_THREADS=12 build/src/engine/test_prefill quick          # batched prefill == token-by-token decode, bit for bit
INK_THREADS=12 build/src/spec/test_spec quick               # speculative decoding == plain decoding, token for token
.venv/bin/python server/tests/run_all.py                    # 72 server tests (no model needed)
tools/with_bench_lock.sh build/src/engine/bench_decode --threads 12,16 --ctx 16,1000 --n 64 --reps 3
```

## Layout

| path | what |
|---|---|
| `src/kernels_*` | AVX-512 kernels: Q8_0 / NVFP4 / BF16 matvec and GEMM, attention with learned relative bias, short convolutions, norms, router |
| `src/engine` | decode executor, batched prefill, slot checkpoints and rewind, hugepage weight arena |
| `src/drafter`, `src/spec` | DSpark block drafter; verify, exact rollback, rejection sampler, adaptive speculation policy |
| `src/capi` | `libink.so` — the C ABI (`include/ink/ink_abi.h`) |
| `server/` | OpenAI-compatible server (FastAPI + ctypes) |
| `tests/`, `bench/`, `tools/` | gates against llama.cpp oracles, benchmarks, model download, serve script |
| `docs/` | design notes and measurements: `ENGINE.md` (results + decision log), `MODEL.md` (the forward pass), `KERNELS_*.md`, `PREFILL_DESIGN.md`, `DSPARK.md`, `SPEC_DESIGN.md`, `REWIND.md`, `SERVING.md`; `docs/status/` has the build log of every agent that worked on it |

Both GGUF layouts Unsloth has published for this model load (`inkling2_loop` and the canonical `inkling2`; identical weights, verified bit-identical logits). The model is Apache-2.0 (Thinking Machines); quantised files by Unsloth.

## Limits

Single-box personal server: one engine, ~14 tok/s total, two concurrent conversations by default. Speculative decoding is on in `auto` mode but barely pays on this CPU (it needs VNNI/AMX to win); it never costs more than ~4%. Contexts were validated against llama.cpp up to ~2.7k tokens; longer contexts run (tested to 36k) but slow down to ~11 tok/s at 28k.

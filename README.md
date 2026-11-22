# inkcpu: a CPU-only inference engine for Inkling 2 Mini

One model (Thinking Machines' Inkling 2 Mini, 26.4B looped MoE, Unsloth `Q4_K_XL` GGUF), one machine class (x86 with AVX-512 F/BW/DQ/VL, no GPU),
built to beat a tuned llama.cpp on the same box. Everything is measured against stock llama.cpp on the same GGUF on the same host; the claim is
"N x faster than tuned llama.cpp on this CPU", nothing broader. Results: `docs/ENGINE.md` (decision log + final numbers).

## What is here
| path | what |
|---|---|
| `include/ink/`, `src/core/` | GGUF reader, model config, scalar reference forward (`docs/MODEL.md`) |
| `src/kernels_gemv/`, `src/kernels_gemm/`, `src/kernels_misc/` | AVX-512 kernels: Q8_0 / NVFP4 / BF16 matvec + GEMM (bit-identical to the matvec), attention with learned relative bias, short convs, norms, router (`docs/KERNELS_*.md`) |
| `src/engine/` | decode executor (one thread-pool run per token, `docs/ENGINE_DESIGN.md`) and batched prefill (`docs/PREFILL_DESIGN.md`); hugepage weight arena |
| `src/drafter/` | DSpark block drafter for speculative decoding (`docs/DRAFTER.md`, spec in `docs/DSPARK.md`) |
| `src/spec/` | verify + exact rollback, rejection sampler, adaptive width policy (`docs/SPEC_DESIGN.md`) |
| `src/capi/` | `libink.so`, the C ABI (`include/ink/ink_abi.h`: `ink_load`, `ink_forward`, `ink_spec_*`, `ink_sample`) |
| `server/` | OpenAI-compatible server (FastAPI, ctypes over `libink.so`): chat/completions, tools, streaming, slot cache, speculation (`server/README.md`, `docs/SERVING.md`) |
| `tests/`, `bench/`, `tools/` | gates against llama.cpp oracles, benchmarks, oracle generators, DSpark acceptance study |
| `docs/status/` | per-agent build logs; `docs/CONTRACT.md` / `docs/SPEC_CONTRACT.md` are the rules the agents worked under |

## Build
```
export PATH=/usr/bin:/bin:$PATH      # system gcc + binutils (Homebrew's toolchain mislinks libgomp)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build ink test_engine test_prefill bench_decode bench_prefill bench_spec
```
Flags are `-O3 -march=skylake-avx512` (no VNNI/BF16/AMX: this host faults on them).

## Run
```
PY=/data/engine/.venv/bin/python
MODEL=/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf
DRAFTER=/data/inkling-mini/models/dspark-Inkling-2-Mini-Q8_0.gguf
INK_CPUS=2,4,6,8,10,12,14,16,18,20,22,24 $PY -m server --backend ink --model $MODEL --lib build/src/capi/libink.so \
    --threads 12 --slots 2 --ctx 8192 --drafter $DRAFTER --spec auto --port 8000
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Say hi."}],"max_tokens":64,"temperature":0}'
```
One pool thread per physical core (`INK_CPUS` = even CPU ids on this box; core 0 is slow). Two engine processes must never share CPUs (spin barriers convoy).
Steady RSS ~17 GB (+0.4 GB drafter), peak ~34 GB during load. `tools/serve.sh start|stop|status|logs` runs it as a managed localhost-only process (SSH tunnel from a laptop: `ssh -N -L 8000:127.0.0.1:8000 <vm>`); `--default-reasoning-effort none` answers without thinking when a client sends no `reasoning_effort` (editor titles/Tab). `docs/SERVING.md` has limits, TTFT, slot/KV memory and the agent-task results.

## Validate
```
INK_THREADS=12 build/src/engine/test_engine oracle|greedy|determinism      # vs llama.cpp oracles in tools/oracle
INK_THREADS=12 build/src/engine/test_prefill quick|nll <dir>                # prefill == decode bit for bit; NLL parity on wikitext-2
INK_THREADS=12 build/src/spec/test_spec all                                 # greedy speculation == plain decode, token for token and state for state
$PY server/tests/run_all.py                                                 # 64 server tests (mock); INK_GGUF+INK_LIB(+INK_DRAFTER) enable the real-engine tests
tools/with_bench_lock.sh build/src/engine/bench_decode --threads 12,16 --ctx 16,1000 --n 64 --reps 3
```
Numerics: parity with llama.cpp is gated on argmax / top-10 / NLL within 0.01 nats, not on tight cosine (quantisation flips are the noise floor, `docs/ENGINE.md`).

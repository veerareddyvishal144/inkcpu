# inkcpu serving layer (OpenAI-compatible server for Inkling 2 Mini)

Python 3.12, FastAPI. Use the engine venv (do not modify it): `/data/engine/.venv/bin/python`.
All commands run from `/data/inkling-mini/engine`.

## Run
```
cd /data/inkling-mini/engine
PY=/data/engine/.venv/bin/python
$PY -m server --backend mock --port 8000                       # development, no model
$PY -m server --backend llama --llama-url http://127.0.0.1:8001 # testing against a llama.cpp server (slow, n_probs)
# real engine on a CPU partition (verified 2026-11-21: 64/64 greedy tokens equal to llama.cpp, see docs/SERVING.md):
INK_CPUS=22,24,26,28,30 $PY -m server --backend ink --model /data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
     --lib build-v/src/capi/libink.so --threads 5 --slots 2 --ctx 8192 --port 8000
```
`--backend ink` loads `libink.so` through ctypes (`include/ink/ink_abi.h`). Search order: `--lib`, `$INK_LIB`, `build/libink.so`,
`build/src/capi/libink.so`, ...; if it is missing the server exits with an error that lists the searched paths.
`ink_load` and every other engine call run in ONE dedicated thread (ABI requirement); `/health` returns 503 `loading` until
the model is loaded (or 503 `error` with the load error). Loading takes ~5 s with a hot page cache; the PEAK RSS during load is
~2x the steady state (mapped GGUF + copied arena: 32-34 GiB), steady state 16.2 GiB + ~0.6 GiB KV per full 8192-token slot.
CPUs: the engine pins its pool to `$INK_CPUS` (one CPU per thread, `--threads` defaults to the length of that list; a mismatch prints
a warning). Never run two engine processes on the same CPUs (spin barriers convoy: 4-5 tok/s instead of 13). The server moves its own
Python/HTTP threads off the engine CPUs (`--no-keep-off-engine-cpus` to disable) and sets the engine threads' timer slack
(`--timer-slack-us`, default 5000) so that an IDLE server costs ~2% of a core instead of ~30% (the pool polls every 20 us).

Environment: `INK_API_KEY` (if set, `/v1/*` needs `Authorization: Bearer <key>` or `x-api-key`; `/health` and `/metrics` stay
open), `INK_REQUEST_LOG` (JSONL path; `--request-log`; empty disables), `INK_LIB`.
Other flags: `--max-queue N` (waiting requests beyond the busy slots, default 8), `--default-max-tokens`, `--prefill-chunk`
(tokens per prefill call, default 256, the cancellation / fairness granule), `--log-bodies` (also log prompts and outputs),
`--drafter PATH` / `INK_DRAFTER` (DSpark drafter GGUF: loads the drafter right after the model, `ink_spec_load`, and enables speculative decoding;
a libink.so without the `ink_spec_*` symbols prints a warning and runs plain decode), `--spec auto|off|2..8` (engine default for requests that do not
send `speculation`; default: the engine's, env `INK_SPEC`, else auto), `--no-spec` (never use the speculative path at all), `--timer-slack-us`,
`--no-keep-off-engine-cpus`. Speculative decoding: docs/SERVING.md section 11.

## Endpoints
| endpoint | notes |
|---|---|
| `POST /v1/chat/completions` | stream and non-stream; `tools`, `tool_choice` (`none`, `auto`, `required` = auto, named function = forced call header), `stop`, `max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`/`repeat_penalty`, `reasoning_effort` (number in [0, 0.99] or none/minimal/low/medium/high/xhigh/max; default 0.9), `stream_options.include_usage`, non-standard `speculation` (`"auto"`, `"off"` or a verify width 2..8; absent = server default) |
| `POST /v1/completions` | raw prompt (string with special tokens parsed, or token ids), same sampling params; `skip_special_tokens` (default true) |
| `GET /v1/models`, `GET /health`, `GET /metrics`, `GET /slots` | `/metrics` is Prometheus text (requests, tokens, queue depth, durations, wall seconds per phase: prefill / decode engine / sampling / other, finish reasons, per-slot KV fill, speculative counters, last prefill/decode tok/s); `/slots` lists tokens held per slot (no content) |

Responses carry `usage` (with `prompt_tokens_details.cached_tokens` = tokens served from the slot cache and
`completion_tokens_details.reasoning_tokens`) and a non-standard `timings` object with llama.cpp's field names (`cache_n` reused,
`prompt_n` NEW prompt tokens, `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_second`) plus `queue_ms`,
`ttft_ms`, `total_ms`, `engine_ms` (time inside the engine during decode), `sample_ms` and, when the request ran speculatively, `spec_steps`,
`spec_drafted`, `spec_accepted`, `spec_tokens_per_step`, `spec_accept_rate`, `spec_mode`, `spec_synced`, `spec_t_hist`; `prompt_ms` / `predicted_ms` are COMPUTE time of
this request, not wall time (with 2 requests interleaved the wall time is longer: `total_ms`). `return_token_ids: true` (non-stream) adds
`prompt_token_ids` and `choices[0].token_ids` (used by the real-engine tests). Reasoning goes to `reasoning_content`
(message and stream delta), tool calls to `tool_calls` (streaming: first delta has id/name and empty arguments, the arguments
follow in one delta when the call is complete). Not supported (400): `n > 1`, `logprobs`, `response_format` other than text, images/audio.
Sampling defaults are the checkpoint's generation_config (temperature 1.0, top_p 1, top_k 0); `temperature: 0` is greedy.
Stop token is 200006 only; `<|end_message|>` (200010) just closes a message segment.

## Back pressure
`n_slots` requests run concurrently (round-robin at prefill-chunk / decode-step granularity); up to `--max-queue` more wait FIFO.
Beyond that the server answers `429` with `Retry-After` (seconds, estimated from recent request durations) and an OpenAI-style error
body (`code: queue_full`). A client disconnect cancels the generation and frees the slot.

## Slot policy (append-only prefix reuse)
The engine can append tokens to a slot or reset it, never rewind. A slot's history is exactly the tokens fed to the engine
(prompt + generated tokens; the final sampled token, i.e. the stop token, is never fed). For each request:
1. an idle slot whose whole history is a prefix of the new prompt continues (longest history, then most recent): only the new
   tokens are prefilled; an identical prompt reuses the cached last logits and prefills nothing;
2. otherwise an empty slot, else the least recently used idle slot is reset and the whole prompt is prefilled.
Multi-turn chats hit the cache only if the re-rendered prompt starts with the old history token for token. The model's thinking is
part of the history, so clients must send `reasoning_content` back in assistant messages (the template renders it); a client that
drops it gets a reset (full prefill). Tool-call turns and forced tool calls round-trip exactly.

## Files
`tokenizer.py` (HF tokenizers from the checkpoint's tokenizer.json, UTF-8 safe streaming decode), `chat_template.py` (the model
card's Jinja template, HF-identical environment), `parser.py` (state machine for reasoning / content / tool calls, token- and
text-mode, `StopMatcher`), `sampling.py` (numpy sampler), `backends.py` (Mock, LlamaServer, Ink), `scheduler.py` (engine thread,
slot table), `app.py`, `metrics.py`, `assets/` (tokenizer.json 26 MB, chat_template.jinja, configs; from
`https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/<file>`, re-downloaded automatically if missing).

## Tests
Mock backend, no network, about 20 s (`test_ink_real.py` is collected too and reports SKIP):
```
/data/engine/.venv/bin/python server/tests/run_all.py            # everything
/data/engine/.venv/bin/python server/tests/test_parser.py        # one file
```
Real engine (`test_ink_real.py`: greedy parity with raw libink and llama.cpp, chat / stream / tools / two-turn cache / 2 clients /
429 / 3000-token prefill / ctx overflow / RSS and idle CPU; about 8 minutes, starts its own server subprocess; `INK_REAL_ONLY=g,h`
selects tests; `ops_measure.py A|B1|B2|C` produces the numbers of docs/SERVING.md):
```
INK_GGUF=/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf INK_LIB=$PWD/build-v/src/capi/libink.so INK_CPUS=22,24,26,28,30 \
  /data/engine/.venv/bin/python server/tests/test_ink_real.py
```
Speculative steps (agent S2's C ABI in `include/ink/ink_abi.h`, bound in `InkBackend`: `ink_spec_load/enable/mode/synced/step`, `ink_sample`):
`Backend.supports_spec` / `sample(logits, SpecRequest, seed)` (first anchor) / `spec_step(slot, anchor, SpecRequest) -> StepResult` return >= 1 emitted
tokens per call plus the engine's `fed` list and `next_seed`; `scheduler.Sequence._spec_step` drops tokens after a stop token / stop string / max_tokens,
chains the seed and keeps the slot history equal to what the engine really processed (`StepResult.fed`, cross-checked with `slot_pos` after every step).
`MockSpecBackend` emits 1-4 tokens per step (`tests/test_spec_steps.py`). Requests with penalties / `logit_bias`, and requests whose effective mode is off,
use plain decode. Real-engine tests i..l (`INK_DRAFTER` set): greedy spec == plain token for token, sampled reproducibility, stop inside a verify window +
cache hit on the next turn, tool calls; `tests/spec_measure.py` is the HTTP A/B (off / auto / 4 / 8).
Parity fixtures were captured from `llama-server --jinja` (port 8001, GGUF Q4_K_XL) by `tests/capture_parity.py` (17 tokenizer strings,
18 conversations: identical text and token ids) and `tests/capture_real.py` (real greedy model outputs for the parser tests).

## Known limits
- Special-token strings inside user content (e.g. a literal `<|end_message|>`) are tokenized as special tokens, exactly as HF and
  llama.cpp do; there is no injection guard (`InkTokenizer.encode(..., parse_special=False)` exists for building one).
- Tool-call arguments stream as one delta at the end of the call (the JSON wrapper is parsed whole); `tool_choice: required` is not enforced (no grammar).
- Prefix reuse compares TOKENS: if the model sampled a non-canonical token split, the re-rendered (re-tokenized) assistant message differs from the slot history and the next turn resets the slot.
- A named `tool_choice` prefills `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":` (no thinking) and ends the turn when that call closes: with the real model (llama.cpp backend) it otherwise went on with a thinking message.
- The llama backend recovers logits from `n_probs` (about 1-3 s per token); use it for parity tests only.
- The engine was verified up to ~2700 tokens (C1) and the batched prefill up to 1500 tokens against the decode path (P); positions >= 8191 (tau > 1)
  are untested. Prefill chunks are 256 tokens: at 5 threads a chunk takes ~9 s, which is also the longest a cancelled prefill keeps running.

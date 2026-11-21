# inkcpu serving layer (OpenAI-compatible server for Inkling 2 Mini)

Python 3.12, FastAPI. Use the engine venv (do not modify it): `/data/engine/.venv/bin/python`.
All commands run from `/data/inkling-mini/engine`.

## Run
```
cd /data/inkling-mini/engine
PY=/data/engine/.venv/bin/python
$PY -m server --backend mock --port 8000                       # development, no model
$PY -m server --backend llama --llama-url http://127.0.0.1:8001 # testing against a llama.cpp server (slow, n_probs)
$PY -m server --backend ink --model /data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
     --threads 16 --slots 2 --ctx 8192 --port 8000              # real engine (needs libink.so)
```
`--backend ink` loads `libink.so` through ctypes (`include/ink/ink_abi.h`). Search order: `--lib`, `$INK_LIB`, `build/libink.so`,
`build/src/capi/libink.so`, ...; if it is missing the server exits with an error that lists the searched paths.
`ink_load` and every other engine call run in ONE dedicated thread (ABI requirement); `/health` returns 503 `loading` until
the model is loaded (or 503 `error` with the load error).

Environment: `INK_API_KEY` (if set, `/v1/*` needs `Authorization: Bearer <key>` or `x-api-key`; `/health` and `/metrics` stay
open), `INK_REQUEST_LOG` (JSONL path; `--request-log`; empty disables), `INK_LIB`.
Other flags: `--max-queue N` (waiting requests beyond the busy slots, default 8), `--default-max-tokens`, `--prefill-chunk`
(tokens per prefill call, default 256, the cancellation / fairness granule), `--log-bodies` (also log prompts and outputs).

## Endpoints
| endpoint | notes |
|---|---|
| `POST /v1/chat/completions` | stream and non-stream; `tools`, `tool_choice` (`none`, `auto`, `required` = auto, named function = forced call header), `stop`, `max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`/`repeat_penalty`, `reasoning_effort` (number in [0, 0.99] or none/minimal/low/medium/high/xhigh/max; default 0.9), `stream_options.include_usage` |
| `POST /v1/completions` | raw prompt (string with special tokens parsed, or token ids), same sampling params; `skip_special_tokens` (default true) |
| `GET /v1/models`, `GET /health`, `GET /metrics` | `/metrics` is Prometheus text (requests, tokens, queue depth, durations, last prefill/decode tok/s) |

Responses carry `usage` (with `prompt_tokens_details.cached_tokens` = tokens served from the slot cache and
`completion_tokens_details.reasoning_tokens`) and a non-standard `timings` object. Reasoning goes to `reasoning_content`
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

## Tests (mock backend, no network, about 15 s)
```
/data/engine/.venv/bin/python server/tests/run_all.py            # everything
/data/engine/.venv/bin/python server/tests/test_parser.py        # one file
```
Parity fixtures were captured from `llama-server --jinja` (port 8001, GGUF Q4_K_XL) by `tests/capture_parity.py` (17 tokenizer strings,
18 conversations: identical text and token ids) and `tests/capture_real.py` (real greedy model outputs for the parser tests).

## Known limits
- Special-token strings inside user content (e.g. a literal `<|end_message|>`) are tokenized as special tokens, exactly as HF and
  llama.cpp do; there is no injection guard (`InkTokenizer.encode(..., parse_special=False)` exists for building one).
- Tool-call arguments stream as one delta at the end of the call (the JSON wrapper is parsed whole); `tool_choice: required` is not enforced (no grammar).
- Prefix reuse compares TOKENS: if the model sampled a non-canonical token split, the re-rendered (re-tokenized) assistant message differs from the slot history and the next turn resets the slot.
- A named `tool_choice` prefills `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":` (no thinking) and ends the turn when that call closes: with the real model (llama.cpp backend) it otherwise went on with a thinking message.
- InkBackend is untested until libink.so exists; the llama backend recovers logits from `n_probs` (about 1-3 s per token).

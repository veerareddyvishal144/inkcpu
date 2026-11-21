# Agent D status (serving layer: server/)

## Plan (written at start, 2026-11-21)
1. assets: tokenizer.json + chat_template.jinja (+ generation_config/tokenizer_config) from HF thinkingmachines/Inkling-2-Mini.
2. server/tokenizer.py (tokenizers lib, special-token map, stop ids) + parity vs llama-server /tokenize.
3. server/chat_template.py (jinja render, tools, reasoning effort, thinking) + parity vs llama-server /apply-template.
4. server/parser.py (final + incremental streaming parser) + tests with captured real outputs.
5. server/sampling.py + slots.py (prefix reuse) + backends.py (Mock, LlamaServer, Ink ctypes) + app.py (FastAPI).
6. tests under server/tests (mock only, < 1 min, no network), README.md.
Constraints: llama-server only on port 8001 with PID file, short generations, CPU kept low.

## Milestones
(none yet)

### Milestone 1 (2026-11-21): tokenizer + chat template parity DONE
- server/assets: tokenizer.json (26 MB), chat_template.jinja, generation_config.json, tokenizer_config.json, config.json from HF.
- server/tokenizer.py (HF tokenizers; UTF-8 safe byte table; stop id 200006 only; parse_special on/off).
- server/chat_template.py (HF-identical jinja env incl. `generation` tag, tojson override; OpenAI normalisation).
- Parity vs llama-server --jinja (port 8001, pid file in scratchpad): 17/17 tokenize strings (special and plain) and 18/18
  conversations (plain, system, multi-turn, reasoning history, tools, tool result by name/id, parallel calls, unicode, parts,
  reasoning_effort variants) identical in text AND token ids. Fixtures: server/tests/fixtures/parity.json (+ capture_parity.py).
- 5 real model outputs captured (greedy, effort 0, short; tool cases use a forced one-line thinking prefix):
  server/tests/fixtures/real_outputs.json (+ capture_real.py). llama-server generation speed ~3 tok/s at -t 4.
- Template facts: no generation prompt (model emits <|message_model|> itself); first generated token is 200001.

### Milestone 2 (2026-11-21): server complete against MockBackend, 38 tests pass (about 12 s, no network)
- server/{parser,sampling,backends,scheduler,metrics,app,__main__}.py + README.md written; tests in server/tests (run_all.py).
- Tests found and fixed: top_p/min_p ignored for V <= 4096, parser `finished` flag semantics, capacity race (sequence in no deque
  while stepping let a third request past the 429 limit), ink-style scripted mock.
- Real-model e2e through LlamaServerBackend (forced tool call, tool-result turn with reasoning stream): see below if finished.
- InkBackend (ctypes) written per ink_abi.h, untested: libink.so does not exist yet.

### Milestone 3 (2026-11-21): real-model end-to-end through the server, DONE (final)
- python -m server --backend llama (llama-server :8001, real Inkling GGUF) -> forced tool call returned a correct tool_calls entry
  (get_weather {"city":"Paris"}); a tool-result turn streamed real reasoning_content deltas. llama backend speed is 0.4 tok/s
  (full-vocab n_probs JSON per token, load average 5-8): only for parity testing.
- Real finding: after a FORCED call header (no thinking) the model continued with a new thinking message instead of the stop token;
  fixed by ending the turn when the forced call closes (GenRequest.stop_after_tool; unit-tested).
- llama-server (pid file in scratchpad) and the e2e server were stopped by PID. Final test run: 38 tests, 0 failures, 11.9 s, mock only.
- Open: InkBackend needs libink.so (INK_LIB or build dirs); token-level prefix matching resets on non-canonical sampled splits.
Files: server/{tokenizer,chat_template,parser,sampling,backends,scheduler,metrics,app,__main__}.py, server/README.md,
server/tests/{run_all,helpers,test_*,capture_*,parity_cases}.py, server/tests/fixtures/{parity,real_outputs}.json, server/assets/.

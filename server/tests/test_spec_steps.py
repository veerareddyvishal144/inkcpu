"""Speculative-step plumbing (agent S2's ink_spec_* later): a backend may emit MORE THAN ONE token per scheduler step.
MockSpecBackend emits 1-4 tokens per step; the output, the token counts and the slot history must equal plain decode."""
import queue
import threading

import httpx

import helpers
from helpers import LiveServer, tok

from server.app import ServerConfig, create_app
from server.backends import BackendError, MockBackend, MockSpecBackend, StepResult
from server.sampling import SamplingParams
from server.scheduler import GenRequest, Scheduler

T = tok()
SCRIPT = T.encode("The quick brown fox jumps over the lazy dog. " * 3)
PROMPT = list(range(1000, 1040))


def _sched(be, chunk=16, **kw):
    s = Scheduler(be, T, max_queue=2, prefill_chunk=chunk, **kw)
    s.start()
    return s


def _run(s, prompt=PROMPT, max_tokens=200, stop=(), temperature=0.0, seed=1, mode="raw", **spk):
    q = queue.Queue()
    s.submit(GenRequest(id="t", prompt_ids=list(prompt), sampling=SamplingParams(seed=seed, temperature=temperature, **spk),
                        max_tokens=max_tokens, sink=q.put, mode=mode, stop=list(stop)))
    while True:
        it = q.get(timeout=20)
        if it[0] in ("done", "error"):
            return it


def _histories(s):
    return [list(x.tokens) for x in s.table.slots.values() if x.tokens]


def _plain(**kw):
    be = MockBackend(n_slots=1, max_ctx=4096, responder=lambda p: SCRIPT)
    s = _sched(be)
    try:
        it = _run(s, **kw)
        assert it[0] == "done", it
        return it[1], _histories(s)[0]
    finally:
        s.shutdown()


def _spec(pattern, overshoot=0, **kw):
    be = MockSpecBackend(n_slots=1, max_ctx=4096, responder=lambda p: SCRIPT, emit_pattern=pattern, overshoot=overshoot)
    s = _sched(be)
    try:
        it = _run(s, **kw)
        assert it[0] == "done", it
        hist = _histories(s)[0]
        assert be.slot_pos(0) == len(hist)                      # the slot history is exactly what the engine processed
        return it[1], hist, be
    finally:
        s.shutdown()


def test_emitting_1_to_4_tokens_per_step_equals_plain_decode():
    ref, ref_hist = _plain()
    assert ref.token_ids == SCRIPT and ref_hist == PROMPT + SCRIPT and ref.finish_reason == "stop"
    for pattern in ((1,), (4,), (1, 3, 2, 4), (2,), (4, 1), (3, 1, 1)):
        fin, hist, be = _spec(pattern)
        assert fin.token_ids == ref.token_ids and fin.message.content == ref.message.content, pattern
        assert fin.completion_tokens == ref.completion_tokens and fin.finish_reason == "stop", pattern
        assert hist == ref_hist, pattern
        assert fin.spec_steps > 0 and fin.spec_steps <= ref.completion_tokens
        if max(pattern) > 1:
            assert fin.spec_steps < ref.completion_tokens, pattern  # several tokens per engine call
        assert sum(len(e) for _, _, e, _ in be.spec_log) >= fin.completion_tokens - 1


def test_tokens_after_the_stop_token_are_dropped_and_never_enter_the_history():
    ref, ref_hist = _plain()
    for pattern in ((4,), (3,), (2,)):
        fin, hist, be = _spec(pattern, overshoot=3)             # a misbehaving engine keeps emitting after the stop token
        assert any(e[-1] == 7 for _, _, e, _ in be.spec_log)    # the overshoot really happened
        assert fin.token_ids == SCRIPT and 7 not in fin.token_ids and fin.finish_reason == "stop"
        assert hist == ref_hist == PROMPT + SCRIPT              # the stop token (200006) was never fed
        assert 200006 not in hist


def test_max_tokens_inside_a_step_is_respected_exactly():
    for n in (1, 2, 5, 7, 8, 13):
        ref, ref_hist = _plain(max_tokens=n)
        for pattern in ((4,), (1, 3, 2, 4), (3,)):
            fin, hist, be = _spec(pattern, max_tokens=n)
            assert fin.finish_reason == "length" and fin.completion_tokens == n == ref.completion_tokens, (n, pattern)
            assert fin.token_ids == SCRIPT[:n] and hist == ref_hist == PROMPT + SCRIPT[:n - 1], (n, pattern)
            assert all(len(e) <= n for _, _, e, _ in be.spec_log)   # max_emit was passed on (never asked for more than needed)


def test_stop_string_inside_a_step_truncates_the_output_but_the_history_is_engine_truth():
    ref, ref_hist = _plain(stop=["lazy"])
    assert ref.message.content.endswith("over the ") and ref.finish_reason == "stop"
    for pattern in ((4,), (3,), (1, 4)):
        fin, hist, be = _spec(pattern, stop=["lazy"])
        assert fin.message.content == ref.message.content and fin.finish_reason == "stop", pattern
        assert hist[:len(ref_hist)] == ref_hist and len(hist) >= len(ref_hist)   # the engine may have processed a few tokens more
        assert be.slot_pos(0) == len(hist)
    # such a slot is not a prefix of the re-rendered next prompt: the scheduler resets it instead of reusing a wrong cache
    be = MockSpecBackend(n_slots=1, max_ctx=4096, responder=lambda p: SCRIPT, emit_pattern=(4,))
    s = _sched(be)
    try:
        it = _run(s, stop=["lazy"])
        hist = _histories(s)[0]
        nxt = PROMPT + T.encode(it[1].message.content)
        it2 = _run(s, prompt=nxt + [5])
        assert it2[0] == "done" and it2[1].cached_tokens in (0, len(hist) if hist == nxt else 0) or hist[:len(nxt)] == nxt
        assert be.slot_pos(0) == len(_histories(s)[0])
    finally:
        s.shutdown()


def test_sampled_speculation_is_reproducible_with_a_seed():
    a, _, _ = _spec((1, 3, 2, 4), temperature=0.8, seed=11, top_k=5)
    b, _, _ = _spec((1, 3, 2, 4), temperature=0.8, seed=11, top_k=5)
    assert a.token_ids == b.token_ids and a.completion_tokens > 0


def test_requests_with_penalties_never_use_the_speculative_path():
    be = MockSpecBackend(n_slots=1, max_ctx=4096, responder=lambda p: SCRIPT, emit_pattern=(4,))
    s = _sched(be)
    try:
        it = _run(s, frequency_penalty=0.1)
        assert it[0] == "done" and it[1].spec_steps == 0 and not [c for c in be.calls if c[0] == "spec"]
        assert it[1].completion_tokens > 0
        it = _run(s, prompt=PROMPT + [3])
        assert it[1].spec_steps > 0
    finally:
        s.shutdown()
    be = MockSpecBackend(n_slots=1, max_ctx=4096, responder=lambda p: SCRIPT, emit_pattern=(4,))
    s = _sched(be, spec=False)                                  # --no-spec
    try:
        it = _run(s)
        assert it[1].spec_steps == 0 and not [c for c in be.calls if c[0] == "spec"]
    finally:
        s.shutdown()


class _Liar(MockSpecBackend):
    def spec_step(self, slot, anchor, req):
        r = super().spec_step(slot, anchor, req)
        return StepResult(r.tokens, r.fed[:-1] if len(r.fed) > 1 else [], None)   # claims fewer processed tokens than it did


def test_a_backend_whose_fed_list_disagrees_with_slot_pos_is_an_error_and_the_slot_is_reset():
    be = _Liar(n_slots=1, max_ctx=4096, responder=lambda p: SCRIPT, emit_pattern=(3,))
    s = _sched(be)
    try:
        it = _run(s)
        assert it[0] == "error" and "out of sync" in it[1] and it[2] == 500, it
        assert be.slot_pos(0) == 0 and not _histories(s)        # unhealthy slot: reset, cache dropped
        be.emit_pattern = (1,)
    finally:
        s.shutdown()


def test_http_chat_with_a_multitoken_backend_matches_plain_decode():
    T_, SP, END, STOP = T, "<|message_model|>", "<|end_message|>", "<|content_model_end_sampling|>"
    script = T_.encode(SP + "<|content_thinking|>Let me think." + END + SP + "<|content_text|>pong, pong, pong" + END + STOP, parse_special=True)
    out = {}
    for name, be in (("plain", MockBackend(n_slots=1, max_ctx=4096, responder=lambda p: script)),
                     ("spec", MockSpecBackend(n_slots=1, max_ctx=4096, responder=lambda p: script, emit_pattern=(1, 4, 2, 3)))):
        live = LiveServer(create_app(be, ServerConfig(), T_))
        try:
            c = httpx.Client(base_url=live.url, timeout=30)
            body = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0, "return_token_ids": True}
            d = c.post("/v1/chat/completions", json=body).json()
            chunks = []
            with c.stream("POST", "/v1/chat/completions", json=dict(body, stream=True, stream_options={"include_usage": True})) as r:
                for line in r.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunks.append(line[6:])
            import json as _json
            objs = [_json.loads(x) for x in chunks]
            text = "".join(o["choices"][0]["delta"].get("content", "") for o in objs if o["choices"])
            reasoning = "".join(o["choices"][0]["delta"].get("reasoning_content", "") for o in objs if o["choices"])
            out[name] = (d["choices"][0]["message"], d["usage"], d["choices"][0]["token_ids"], text, reasoning, objs[-1]["usage"])
            if name == "spec":
                assert d["timings"]["spec_steps"] > 0 and "cache_n" in d["timings"]
                assert "ink_spec_total" in c.get("/metrics").text
            c.close()
        finally:
            live.stop()
    assert out["plain"][0] == out["spec"][0] and out["plain"][2] == out["spec"][2]
    assert out["plain"][3] == out["spec"][3] == "pong, pong, pong" and out["plain"][4] == out["spec"][4] == "Let me think."
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert out["plain"][1][k] == out["spec"][1][k] == out["spec"][5][k]


if __name__ == "__main__":
    raise SystemExit(helpers.run_module(globals()))

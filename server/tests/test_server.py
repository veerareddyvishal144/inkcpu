"""HTTP-level tests: uvicorn in a thread + MockBackend with a scripted responder (no network, no model)."""
import json
import os
import tempfile
import threading
import time

import httpx

import helpers
from helpers import LiveServer, tok

from server.app import ServerConfig, create_app
from server.backends import MockBackend

T = tok()
SP, END, STOP = "<|message_model|>", "<|end_message|>", "<|content_model_end_sampling|>"
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "weather",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
SCRIPTS = {
    "plain": SP + "<|content_thinking|>Let me think." + END + SP + "<|content_text|>pong" + END + STOP,
    "tool": SP + "<|content_thinking|>Need weather." + END + SP + 'get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}' + END + STOP,
    "after_tool": SP + "<|content_text|>It is 18C." + END + STOP,
    "stoptest": SP + "<|content_text|>Hello STOP world" + END + STOP,
    "long": SP + "<|content_text|>" + "word " * 300 + END + STOP,
    "forced": '{"city":"Paris"}}' + END + SP + "<|content_thinking|>I already called it." + END + STOP,
}


def responder(prompt_ids):
    text = T.decode(prompt_ids)
    if text.endswith('"args":'):
        key = "forced"
    elif "<|message_tool|>" in text:
        key = "after_tool"
    elif "STOPTEST" in text:
        key = "stoptest"
    elif "LONG" in text:
        key = "long"
    elif "weather" in text and "tool_declare" in text:
        key = "tool"
    else:
        key = "plain"
    return T.encode(SCRIPTS[key], parse_special=True)


class Srv:
    def __init__(self, **kw):
        be_kw = {k: kw.pop(k) for k in list(kw) if k in ("n_slots", "max_ctx", "delay")}
        self.be = MockBackend(responder=responder, **be_kw)
        self.app = create_app(self.be, ServerConfig(**kw), T)
        self.live = LiveServer(self.app)
        self.c = httpx.Client(base_url=self.live.url, timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.c.close()
        self.live.stop()

    def chat(self, content="hello", **kw):
        body = {"messages": [{"role": "user", "content": content}]}
        body.update(kw)
        return self.c.post("/v1/chat/completions", json=body, headers=kw.pop("headers", None) or {})

    def stream(self, body, path="/v1/chat/completions"):
        chunks = []
        with self.c.stream("POST", path, json=dict(body, stream=True)) as r:
            assert r.status_code == 200, r.read()
            for line in r.iter_lines():
                if line.startswith("data: "):
                    chunks.append(line[6:])
        return chunks


def test_chat_nonstream_usage_and_prefix_reuse_over_http():
    with Srv(n_slots=2) as s:
        d = s.chat("hello", temperature=0).json()
        m = d["choices"][0]["message"]
        assert m["role"] == "assistant" and m["content"] == "pong" and m["reasoning_content"] == "Let me think."
        assert d["choices"][0]["finish_reason"] == "stop" and d["object"] == "chat.completion"
        u = d["usage"]
        assert u["completion_tokens"] == len(T.encode(SCRIPTS["plain"])) - 1  # the stop token is not counted
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] and u["prompt_tokens_details"]["cached_tokens"] == 0
        assert u["completion_tokens_details"]["reasoning_tokens"] > 0
        r2 = s.c.post("/v1/chat/completions", json={"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "pong", "reasoning_content": "Let me think."},
            {"role": "user", "content": "again"}]}).json()
        # history = prompt + generated tokens (the stop token is never fed) -> next turn continues the slot
        assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
        r3 = s.c.post("/v1/chat/completions", json={"messages": [  # reasoning dropped by the client: cache miss (documented)
            {"role": "user", "content": "hello"}, {"role": "assistant", "content": "pong"},
            {"role": "user", "content": "again"}]}).json()
        assert r3["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def test_tool_call_nonstream_and_roundtrip():
    with Srv() as s:
        d = s.chat("what is the weather?", tools=[WEATHER]).json()
        ch = d["choices"][0]
        tc = ch["message"]["tool_calls"][0]
        assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] is None
        assert tc["function"]["name"] == "get_weather" and json.loads(tc["function"]["arguments"]) == {"city": "Paris"}
        assert tc["id"].startswith("call_") and tc["type"] == "function" and ch["message"]["reasoning_content"] == "Need weather."
        d2 = s.c.post("/v1/chat/completions", json={"tools": [WEATHER], "messages": [
            {"role": "user", "content": "what is the weather?"},
            {"role": "assistant", "content": None, "tool_calls": [tc], "reasoning_content": "Need weather."},
            {"role": "tool", "tool_call_id": tc["id"], "content": "18C"}]}).json()
        assert d2["choices"][0]["message"]["content"] == "It is 18C." and d2["choices"][0]["finish_reason"] == "stop"
        assert d2["usage"]["prompt_tokens_details"]["cached_tokens"] > 0   # continued the slot of the first call


def test_forced_tool_choice_and_none():
    with Srv() as s:
        d = s.chat("anything", tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "get_weather"}}).json()
        tc = d["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather" and tc["function"]["arguments"] == '{"city":"Paris"}'
        assert d["choices"][0]["finish_reason"] == "tool_calls"
        # the turn ends right after the forced call, even if the model would continue (it did with the real model)
        assert d["usage"]["completion_tokens"] == len(T.encode('{"city":"Paris"}}' + END)) and "reasoning_content" not in d["choices"][0]["message"]
        d = s.chat("what is the weather?", tools=[WEATHER], tool_choice="none").json()  # tools hidden: plain answer
        assert d["choices"][0]["message"]["content"] == "pong"
        assert s.chat("x", tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "nope"}}).status_code == 400


def test_streaming_order_and_equivalence():
    with Srv() as s:
        chunks = s.stream({"messages": [{"role": "user", "content": "hello"}], "stream_options": {"include_usage": True}})
        assert chunks[-1] == "[DONE]"
        objs = [json.loads(c) for c in chunks[:-1]]
        assert objs[0]["choices"][0]["delta"].get("role") == "assistant"
        kinds = []
        for o in objs[1:-1]:
            dl = o["choices"][0]["delta"]
            kinds += [k for k in ("reasoning_content", "content") if k in dl]
        assert kinds.index("content") > max(i for i, k in enumerate(kinds) if k == "reasoning_content")  # reasoning first
        assert objs[-2]["choices"][0]["finish_reason"] == "stop" and objs[-1]["choices"] == [] and objs[-1]["usage"]["completion_tokens"] > 0
        text = "".join(o["choices"][0]["delta"].get("content", "") for o in objs[1:-2] if o["choices"])
        reasoning = "".join(o["choices"][0]["delta"].get("reasoning_content", "") for o in objs[1:-2] if o["choices"])
        assert text == "pong" and reasoning == "Let me think."
        assert all("<|" not in json.dumps(o) for o in objs)
        # tool call stream: id+name first (arguments ""), then the arguments, then finish_reason tool_calls
        chunks = s.stream({"messages": [{"role": "user", "content": "weather please"}], "tools": [WEATHER]})
        objs = [json.loads(c) for c in chunks[:-1]]
        tcs = [o["choices"][0]["delta"]["tool_calls"][0] for o in objs if o["choices"] and "tool_calls" in o["choices"][0]["delta"]]
        assert tcs[0]["id"].startswith("call_") and tcs[0]["function"] == {"name": "get_weather", "arguments": ""} and tcs[0]["index"] == 0
        assert "".join(t["function"].get("arguments", "") for t in tcs[1:]) == '{"city":"Paris"}'
        assert objs[-1]["choices"][0]["finish_reason"] == "tool_calls" and chunks[-1] == "[DONE]"


def test_stop_strings_length_and_completions():
    with Srv() as s:
        d = s.chat("STOPTEST", stop=["STOP"]).json()
        assert d["choices"][0]["message"]["content"] == "Hello " and d["choices"][0]["finish_reason"] == "stop"
        chunks = s.stream({"messages": [{"role": "user", "content": "STOPTEST"}], "stop": ["STOP"]})
        pieces = [json.loads(c)["choices"][0]["delta"].get("content", "") for c in chunks[:-1]]
        assert "".join(pieces) == "Hello " and not any("ST" in p for p in pieces)  # a partial stop string is never emitted
        d = s.chat("LONG", max_tokens=10).json()
        assert d["choices"][0]["finish_reason"] == "length" and d["usage"]["completion_tokens"] == 10
        d = s.c.post("/v1/completions", json={"prompt": "abc LONG", "max_tokens": 6}).json()
        assert d["object"] == "text_completion" and d["choices"][0]["finish_reason"] == "length" and d["usage"]["completion_tokens"] == 6
        assert "<|" not in d["choices"][0]["text"]
        chunks = s.stream({"prompt": [1, 2, 3, 4], "max_tokens": 5}, "/v1/completions")
        assert chunks[-1] == "[DONE]" and json.loads(chunks[-2])["choices"][0]["finish_reason"] in ("length", "stop")


def test_validation_errors():
    with Srv(max_ctx=64) as s:
        for body in ({"messages": []}, {"messages": [{"role": "user", "content": "x"}], "temperature": "hot"},
                     {"messages": [{"role": "user", "content": "x"}], "n": 2}, {"messages": [{"role": "user", "content": "x"}], "top_p": 0},
                     {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}},
                     {"messages": [{"role": "user", "content": "x" * 400}]},
                     {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": 5}):
            r = s.c.post("/v1/chat/completions", json=body)
            assert r.status_code == 400 and "error" in r.json() and r.json()["error"]["message"], (body, r.text)
        assert s.c.post("/v1/chat/completions", content=b"not json").status_code == 400


def test_api_key_models_health_metrics_and_log():
    log = os.path.join(tempfile.mkdtemp(), "req.jsonl")
    with Srv(api_key="sekret", request_log=log) as s:
        assert s.c.get("/health").status_code == 200
        assert s.chat().status_code == 401 and s.c.get("/v1/models").status_code == 401
        h = {"Authorization": "Bearer sekret"}
        assert s.c.get("/v1/models", headers=h).json()["data"][0]["id"] == "inkling-2-mini"
        r = s.c.post("/v1/chat/completions", headers=h, json={"messages": [{"role": "user", "content": "hello"}]})
        assert r.status_code == 200
        m = s.c.get("/metrics").text
        assert 'ink_requests_total{endpoint="/v1/chat/completions",status="200"} 1' in m and "ink_tokens_total" in m
        assert "ink_request_duration_seconds_count" in m and "ink_queue_depth 0" in m
        t0 = time.time()
        lines = []
        while time.time() - t0 < 3:
            lines = [json.loads(x) for x in open(log).read().splitlines() if x.strip()]
            if any(x.get("status") == 200 and x.get("completion_tokens") for x in lines) and any(x.get("status") == 401 for x in lines):
                break
            time.sleep(0.05)
        done = [x for x in lines if x.get("completion_tokens")]
        assert done and done[0]["endpoint"] == "/v1/chat/completions" and done[0]["finish_reason"] == "stop" and "ts" in done[0]
        assert any(x.get("status") == 401 for x in lines)
        assert "request" not in done[0]  # bodies are not logged by default


def test_429_with_retry_after_when_queue_is_full():
    with Srv(n_slots=1, max_queue=1, delay=0.1) as s:
        res = []
        def fire(i):
            r = s.chat("hello %d" % i, max_tokens=4)
            res.append((r.status_code, r.headers.get("retry-after"), r.json()))
        ths = [threading.Thread(target=fire, args=(i,)) for i in range(4)]
        for t in ths:
            t.start(); time.sleep(0.03)
        for t in ths:
            t.join(30)
        codes = sorted(c for c, _, _ in res)
        assert codes == [200, 200, 429, 429], codes
        for c, ra, body in res:
            if c == 429:
                assert ra is not None and int(ra) >= 1 and body["error"]["code"] == "queue_full"
        m = s.c.get("/metrics").text
        assert "ink_rejected_requests_total 2" in m and 'status="429"} 2' in m
        assert s.chat("after the burst", max_tokens=2).status_code == 200  # capacity is back


def test_client_disconnect_frees_the_slot():
    with Srv(n_slots=1, delay=0.03) as s:
        with s.c.stream("POST", "/v1/chat/completions", json={"messages": [{"role": "user", "content": "LONG"}], "stream": True,
                                                                "max_tokens": 1000}) as r:
            it = r.iter_lines()
            for _ in range(4):
                next(it)
        t0 = time.time()
        while time.time() - t0 < 5:
            h = s.c.get("/health").json()
            if h["active"] == 0:
                break
            time.sleep(0.05)
        assert h["active"] == 0, h  # generation was cancelled long before 1000 tokens (30 s) elapsed
        assert s.chat("hello", max_tokens=2).status_code == 200


if __name__ == "__main__":
    raise SystemExit(helpers.run_module(globals()))

"""Real-engine tests: `python -m server --backend ink` (libink.so + the Q4_K_XL GGUF) behind the OpenAI-compatible API.

Skipped unless INK_GGUF and INK_LIB are set. Run (one engine process at a time, own CPU partition; takes ~6-8 minutes):
    INK_GGUF=/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf INK_LIB=/data/inkling-mini/engine/build-v/src/capi/libink.so \
    INK_CPUS=22,24,26,28,30 /data/engine/.venv/bin/python server/tests/test_ink_real.py
The tests start ONE server subprocess (2 slots, ctx 8192, --max-queue 2, request log) and share it; names a..h fix the order.
Tests m, n (agent S3: slot rewind / shared-prefix reuse, docs/REWIND.md) need a libink.so with ink_slot_rewind; ~8 min on 3 threads.
Tests i..l need INK_DRAFTER (DSpark drafter GGUF) and a libink.so that exports ink_spec_*: they stop the plain server and start one
with --drafter (one engine process per CPU set), e.g. INK_REAL_ONLY=i,j,k,l INK_DRAFTER=/data/inkling-mini/models/dspark-Inkling-2-Mini-Q8_0.gguf.
Numbers (TTFT, tok/s, RSS, idle CPU) are written to $INK_REAL_RESULTS (default: <tmp>/ink_real_results.json). Timings on the
shared host are indicative only.
"""
import atexit
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import helpers
from helpers import ENGINE, tok

GGUF = os.environ.get("INK_GGUF", "")
LIB = os.environ.get("INK_LIB", "")
DRAFTER = os.environ.get("INK_DRAFTER", "")
_SP = None
ORACLE = os.path.join(ENGINE, "tools", "oracle", "greedy")
RESULTS = {}
_S = None


def _need():
    if not (GGUF and LIB):
        raise helpers.Skip("INK_GGUF / INK_LIB not set")
    os.environ.setdefault("INK_CPUS", "22,24,26,28,30")


def rec(key, value):
    RESULTS[key] = value
    path = os.environ.get("INK_REAL_RESULTS") or os.path.join(tempfile.gettempdir(), "ink_real_results.json")
    try:
        with open(path, "w") as f:
            json.dump(RESULTS, f, indent=1, sort_keys=True)
    except OSError:
        pass


class Server:
    """The server subprocess (python -m server --backend ink ...)."""

    def __init__(self, ctx=8192, slots=2, max_queue=2, extra=()):
        import httpx
        self.httpx = httpx
        self.tmp = tempfile.mkdtemp(prefix="inkreal_")
        self.log_path = os.path.join(self.tmp, "server.log")
        self.req_log = os.path.join(self.tmp, "req.jsonl")
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        cpus = [c for c in os.environ["INK_CPUS"].split(",") if c]
        cmd = [sys.executable, "-m", "server", "--backend", "ink", "--model", GGUF, "--lib", LIB, "--threads", str(len(cpus)),
               "--slots", str(slots), "--ctx", str(ctx), "--max-queue", str(max_queue), "--port", str(self.port),
               "--request-log", self.req_log] + list(extra)
        self.logf = open(self.log_path, "w")
        self.t_start = time.time()
        self.proc = subprocess.Popen(cmd, cwd=ENGINE, env=dict(os.environ), stdout=self.logf, stderr=subprocess.STDOUT)
        self.pid = self.proc.pid
        atexit.register(self.stop)
        self.url = "http://127.0.0.1:%d" % self.port
        deadline = time.time() + 240
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited (%s): %s" % (self.proc.returncode, self.tail()))
            try:
                r = httpx.get(self.url + "/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("server did not become healthy: " + self.tail())
        self.load_s = time.time() - self.t_start
        self.c = httpx.Client(base_url=self.url, timeout=900)

    def tail(self, n=1500):
        try:
            return open(self.log_path).read()[-n:]
        except OSError:
            return ""

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(40)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)

    # ---- /proc helpers
    def status(self, key):
        for line in open("/proc/%d/status" % self.pid):
            if line.startswith(key + ":"):
                return int(line.split()[1])
        return 0

    def cpu_ticks(self):
        f = open("/proc/%d/stat" % self.pid).read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])

    def thread_ticks(self):
        out = {}
        for t in os.listdir("/proc/%d/task" % self.pid):
            try:
                raw = open("/proc/%d/task/%s/stat" % (self.pid, t)).read()
            except OSError:
                continue
            name = raw[raw.index("(") + 1:raw.rindex(")")]
            f = raw.rsplit(")", 1)[1].split()
            out[t] = (name, int(f[11]) + int(f[12]))
        return out

    # ---- API helpers
    def post(self, path, body, **kw):
        t0 = time.time()
        r = self.c.post(path, json=body, **kw)
        r.wall = time.time() - t0
        return r

    def health(self):
        return self.c.get("/health").json()

    def wait_idle(self, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = self.health()
            if h["active"] == 0 and h["queued"] == 0:
                return h
            time.sleep(0.1)
        raise AssertionError("server not idle: %s" % self.health())

    def stream(self, body, path="/v1/chat/completions", max_events=None):
        """POST a streaming request; returns (status, [parsed SSE objects], first_event_s)."""
        objs, t0, first = [], time.time(), None
        with self.c.stream("POST", path, json=dict(body, stream=True)) as r:
            if r.status_code != 200:
                r.read()
                return r.status_code, r.json(), None
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    if first is None:
                        first = time.time() - t0
                    objs.append(json.loads(line[6:]))
                    if max_events and len(objs) >= max_events:
                        break
        return 200, objs, first


def server():
    global _S
    _need()
    if _S is None:
        _S = Server()
        rec("server_load_s", round(_S.load_s, 1))
        rec("rss_after_load_gib", round(_S.status("VmRSS") / 1048576.0, 2))
    return _S


def spec_server():
    """Server with the drafter loaded (speculative decoding). The plain shared server is stopped first: one engine per CPU set."""
    global _S, _SP
    _need()
    if not DRAFTER:
        raise helpers.Skip("INK_DRAFTER not set")
    if _SP is None:
        if _S is not None:
            _S.stop()
            _S = None
        _SP = Server(extra=["--drafter", DRAFTER])
        tail = _SP.tail(4000)
        assert "without speculative decoding" not in tail, tail
        rec("spec_server_load_s", round(_SP.load_s, 1))
        rec("spec_rss_after_load_gib", round(_SP.status("VmRSS") / 1048576.0, 2))
        rec("spec_exact_verify_env", os.environ.get("INK_SPEC_EXACT", "(unset: fast verify kernel)"))
    return _SP


def _oracle(name):
    import numpy as np
    d = os.path.join(ORACLE, name)
    return (np.fromfile(os.path.join(d, "prompt_tokens.i32"), dtype=np.int32).tolist(),
            np.fromfile(os.path.join(d, "gen_tokens.i32"), dtype=np.int32).tolist(), open(os.path.join(d, "prompt.txt")).read())


def _raw_greedy(prompt, n):
    """Greedy generation straight through libink with plain ctypes (independent of server/backends.py), in a short-lived thread
    (the engine pins its calling thread to cpus[0]; the test thread must not inherit that)."""
    import numpy as np
    out = {}

    def run():
        try:
            lib = ctypes.CDLL(LIB)
            c_int, c_void_p = ctypes.c_int, ctypes.c_void_p
            p32, pf = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
            lib.ink_load.restype = c_void_p
            lib.ink_load.argtypes = [ctypes.c_char_p, c_int, c_int, c_int]
            lib.ink_n_vocab.argtypes = [c_void_p]
            lib.ink_slot_new.argtypes = [c_void_p]
            lib.ink_forward.argtypes = [c_void_p, c_int, p32, c_int, pf]
            lib.ink_free.argtypes = [c_void_p]
            lib.ink_last_error.restype = ctypes.c_char_p
            n_threads = len([c for c in os.environ["INK_CPUS"].split(",") if c])
            h = lib.ink_load(GGUF.encode(), n_threads, 1, 2048)
            assert h, lib.ink_last_error()
            slot = lib.ink_slot_new(h)
            buf = np.zeros(lib.ink_n_vocab(h), np.float32)
            arr = np.asarray(prompt, np.int32)
            t0 = time.perf_counter()
            assert lib.ink_forward(h, slot, arr.ctypes.data_as(p32), len(arr), buf.ctypes.data_as(pf)) == 0
            out["prefill_s"] = time.perf_counter() - t0
            toks, t1 = [], time.perf_counter()
            for i in range(n):
                t = int(np.argmax(buf))
                toks.append(t)
                if i + 1 < n:
                    one = np.array([t], np.int32)
                    assert lib.ink_forward(h, slot, one.ctypes.data_as(p32), 1, buf.ctypes.data_as(pf)) == 0
            out["decode_tps"] = (n - 1) / (time.perf_counter() - t1)
            out["tokens"] = toks
            lib.ink_free(h)
        except BaseException as e:  # noqa: BLE001
            out["error"] = repr(e)

    th = threading.Thread(target=run)
    th.start()
    th.join(900)
    assert "error" not in out and "tokens" in out, out
    return out


def _complete(s, prompt, n, **kw):
    r = s.post("/v1/completions", dict({"prompt": prompt, "max_tokens": n, "temperature": 0, "return_token_ids": True}, **kw))
    assert r.status_code == 200, r.text
    return r.json()


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                           "required": ["city"]}}}


# ---------------------------------------------------------------------------------------------------------------- (a)
def test_a_greedy_through_server_equals_raw_libink_and_llamacpp():
    _need()
    p_ids, ref, p_text = _oracle("s1_en_capital")
    raw = _raw_greedy(p_ids, 64)                                 # BEFORE the server exists: one engine at a time
    rec("raw_engine_decode_tps_5threads", round(raw["decode_tps"], 2))
    rec("raw_engine_prefill_13tok_s", round(raw["prefill_s"], 3))
    s = server()
    d = _complete(s, p_ids, 24)
    got = d["choices"][0]["token_ids"]
    assert got == raw["tokens"][:24], (got, raw["tokens"][:24])   # server path == direct ctypes path, token for token
    assert got == ref[:24], (got, ref[:24])                       # == llama.cpp's greedy continuation
    d64 = _complete(s, p_ids, 64)
    assert d64["choices"][0]["token_ids"] == raw["tokens"] == ref, "64 greedy tokens must equal llama.cpp (C1 gate b: s1 64/64)"
    # the prompt given as TEXT (special tokens parsed) must tokenize to the llama.cpp ids and give the same continuation
    p2_ids, ref2, p2_text = _oracle("c1_chat_user")
    d2 = s.post("/v1/completions", {"prompt": p2_text, "max_tokens": 24, "temperature": 0, "return_token_ids": True}).json()
    assert d2["prompt_token_ids"] == p2_ids and d2["choices"][0]["token_ids"] == ref2[:24]
    t = d64["timings"]
    rec("http_decode_tps_64tok_5threads", t["predicted_per_second"])
    rec("http_overhead_ms_per_token_greedy", round((t["predicted_ms"] - t["engine_ms"]) / t["predicted_n"], 3))
    rec("sampling_ms_per_token_greedy", round(t["sample_ms"] / t["predicted_n"], 3))


# ---------------------------------------------------------------------------------------------------------------- (b)
def test_b_chat_nonstream_and_stream_fields():
    s = server()
    body = {"messages": [{"role": "user", "content": "Reply with just the word: pong"}], "temperature": 0, "reasoning_effort": 0,
            "max_tokens": 120, "return_token_ids": True}
    r = s.post("/v1/chat/completions", body)
    assert r.status_code == 200, r.text
    d = r.json()
    ch, m, u, tm = d["choices"][0], d["choices"][0]["message"], d["usage"], d["timings"]
    assert d["object"] == "chat.completion" and ch["finish_reason"] == "stop" and m["role"] == "assistant"
    assert m["content"].strip().lower().startswith("pong") and m["reasoning_content"]
    assert u["completion_tokens"] == len(ch["token_ids"]) and u["completion_tokens_details"]["reasoning_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] and u["prompt_tokens_details"]["cached_tokens"] == 0
    for k in ("queue_ms", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second", "ttft_ms", "cache_n", "engine_ms", "sample_ms"):
        assert k in tm, k
    assert tm["predicted_n"] == u["completion_tokens"] and tm["prompt_n"] == u["prompt_tokens"] and tm["predicted_per_second"] > 1.0
    assert "<|" not in json.dumps(m)
    st, objs, first = s.stream(dict(body, stream_options={"include_usage": True}))
    assert st == 200 and objs[0]["choices"][0]["delta"].get("role") == "assistant"
    kinds, text, reasoning = [], "", ""
    for o in objs[1:]:
        if not o["choices"]:
            continue
        dl = o["choices"][0]["delta"]
        for k in ("reasoning_content", "content"):
            if dl.get(k):
                kinds.append(k)
        text += dl.get("content") or ""
        reasoning += dl.get("reasoning_content") or ""
    assert kinds and kinds[0] == "reasoning_content" and kinds.index("content") > max(i for i, k in enumerate(kinds) if k == "reasoning_content")
    assert text == m["content"] and reasoning == m["reasoning_content"], "greedy stream must equal the non-stream answer"
    assert objs[-2]["choices"][0]["finish_reason"] == "stop" and objs[-1]["choices"] == []
    assert objs[-1]["usage"]["completion_tokens"] == u["completion_tokens"] and "<|" not in json.dumps(objs)
    rec("stream_role_chunk_s", round(first, 3))


# ---------------------------------------------------------------------------------------------------------------- (c)
def test_c_forced_and_auto_tool_calls_and_tool_result_turn():
    s = server()
    q = [{"role": "user", "content": "What is the weather in Paris?"}]
    r = s.post("/v1/chat/completions", {"messages": q, "tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                                        "temperature": 0, "reasoning_effort": 0, "max_tokens": 100})
    assert r.status_code == 200, r.text
    ch = r.json()["choices"][0]
    tc = ch["message"]["tool_calls"][0]
    assert ch["finish_reason"] == "tool_calls" and tc["function"]["name"] == "get_weather" and json.loads(tc["function"]["arguments"]) == {"city": "Paris"}
    # auto: the model decides to call the tool; then the tool result goes back and it answers
    r = s.post("/v1/chat/completions", {"messages": q, "tools": [WEATHER], "temperature": 0, "reasoning_effort": 0, "max_tokens": 300})
    d = r.json()
    ch = d["choices"][0]
    assert ch["finish_reason"] == "tool_calls", ch
    tc = ch["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather" and json.loads(tc["function"]["arguments"]) == {"city": "Paris"} and tc["id"].startswith("call_")
    asst = {"role": "assistant", "content": None, "tool_calls": [tc]}
    if ch["message"].get("reasoning_content"):
        asst["reasoning_content"] = ch["message"]["reasoning_content"]
    r2 = s.post("/v1/chat/completions", {"messages": q + [asst, {"role": "tool", "tool_call_id": tc["id"], "content": "18C and sunny"}],
                                         "tools": [WEATHER], "temperature": 0, "reasoning_effort": 0, "max_tokens": 200})
    d2 = r2.json()
    c2 = d2["choices"][0]
    assert r2.status_code == 200 and c2["finish_reason"] == "stop" and "18" in (c2["message"]["content"] or ""), c2
    cached = d2["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached == d["usage"]["prompt_tokens"] + d["usage"]["completion_tokens"], (cached, d["usage"])   # the tool turn round-trips exactly
    # streaming auto tool call: id+name first with empty arguments, the arguments in later deltas
    st, objs, _ = s.stream({"messages": q, "tools": [WEATHER], "temperature": 0, "reasoning_effort": 0, "max_tokens": 300})
    tcs = [o["choices"][0]["delta"]["tool_calls"][0] for o in objs if o["choices"] and "tool_calls" in o["choices"][0]["delta"]]
    assert tcs[0]["function"] == {"name": "get_weather", "arguments": ""} and tcs[0]["id"].startswith("call_")
    assert json.loads("".join(t["function"].get("arguments", "") for t in tcs[1:])) == {"city": "Paris"}
    assert objs[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------------------------------------------- (d)
def test_d_two_turn_conversation_reuses_the_cache_and_prefills_only_new_tokens():
    s = server()
    s.wait_idle()
    msgs = [{"role": "user", "content": "Name one prime number between 10 and 20. Answer with the number only."}]
    d1 = s.post("/v1/chat/completions", {"messages": msgs, "temperature": 0, "reasoning_effort": 0, "max_tokens": 120}).json()
    m1 = d1["choices"][0]["message"]
    assert d1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    asst = {"role": "assistant", "content": m1["content"]}
    if m1.get("reasoning_content"):
        asst["reasoning_content"] = m1["reasoning_content"]
    follow = {"role": "user", "content": "Now name a different one. Number only."}
    d2 = s.post("/v1/chat/completions", {"messages": msgs + [asst, follow], "temperature": 0, "reasoning_effort": 0, "max_tokens": 120}).json()
    u1, u2, t1, t2 = d1["usage"], d2["usage"], d1["timings"], d2["timings"]
    cached = u2["prompt_tokens_details"]["cached_tokens"]
    assert cached == u1["prompt_tokens"] + u1["completion_tokens"] > 0, (cached, u1)
    assert t2["cache_n"] == cached and t2["prompt_n"] == u2["prompt_tokens"] - cached and 0 < t2["prompt_n"] < 40
    # a client that drops reasoning_content gets a full prefill of the whole conversation (documented cache miss): slower
    d3 = s.post("/v1/chat/completions", {"messages": msgs + [{"role": "assistant", "content": m1["content"]}, follow], "temperature": 0,
                                         "reasoning_effort": 0, "max_tokens": 4}).json()
    assert d3["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and d3["timings"]["prompt_n"] == d3["usage"]["prompt_tokens"]
    assert t2["prompt_ms"] < d3["timings"]["prompt_ms"], (t2["prompt_ms"], d3["timings"]["prompt_ms"])
    rec("two_turn", {"turn1_prompt_tokens": u1["prompt_tokens"], "turn2_prompt_tokens": u2["prompt_tokens"], "turn2_cached": cached,
                     "turn2_prefill_ms": t2["prompt_ms"], "same_conversation_without_cache_prefill_ms": d3["timings"]["prompt_ms"]})


# ---------------------------------------------------------------------------------------------------------------- (e)
def test_e_two_concurrent_clients_are_serialised_and_disconnect_frees_the_slot():
    s = server()
    s.wait_idle()
    pa, _, _ = _oracle("s1_en_capital")
    pb, _, _ = _oracle("s2_code_add")
    solo = {k: _complete(s, p, 24)["choices"][0]["token_ids"] for k, p in (("a", pa), ("b", pb))}
    s.wait_idle()
    res, seen_active = {}, []

    def go(k, p):
        t0 = time.time()
        d = _complete(s, p, 24)
        res[k] = (d, time.time() - t0)

    ths = [threading.Thread(target=go, args=(k, p)) for k, p in (("a", pa + [0] * 0), ("b", pb))]
    t0 = time.time()
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        seen_active.append(s.health()["active"])
        time.sleep(0.05)
    wall = time.time() - t0
    for t in ths:
        t.join()
    assert max(seen_active) == 2, "both requests must have been active at the same time: %s" % seen_active[:20]
    for k in ("a", "b"):
        assert res[k][0]["choices"][0]["token_ids"] == solo[k], k      # interleaving the slots does not change a single token
    eng = sum(res[k][0]["timings"]["engine_ms"] + res[k][0]["timings"]["prompt_ms"] for k in res) / 1000.0
    assert eng <= wall * 1.03, "engine time of the two requests (%.2fs) exceeds the wall time (%.2fs): calls overlapped" % (eng, wall)
    rec("two_clients_24tok", {"wall_s": round(wall, 2), "engine_s_sum": round(eng, 2),
                              "per_client_tps": [res[k][0]["timings"]["predicted_per_second"] for k in ("a", "b")]})
    # a client that disconnects mid-stream frees its slot (generation is cancelled at the next step)
    s.wait_idle()
    t0 = time.time()
    st, objs, _ = s.stream({"messages": [{"role": "user", "content": "Write a long story about a dragon."}], "temperature": 0.8,
                            "max_tokens": 400}, max_events=6)
    assert st == 200 and len(objs) >= 3
    t_close = time.time()
    h = None
    while time.time() - t_close < 20:
        h = s.health()
        if h["active"] == 0:
            break
        time.sleep(0.05)
    assert h["active"] == 0, "slot not freed after disconnect: %s" % h
    rec("cancel_latency_s_after_disconnect", round(time.time() - t_close, 2))
    assert time.time() - t0 < 60
    r = s.post("/v1/completions", {"prompt": pa, "max_tokens": 4, "temperature": 0})
    assert r.status_code == 200
    log = [json.loads(x) for x in open(s.req_log) if x.strip()]
    assert any(x.get("finish_reason") == "cancelled" for x in log), "cancelled generation missing from the request log"


# ---------------------------------------------------------------------------------------------------------------- (f)
def test_f_429_when_the_queue_is_full():
    s = server()
    s.wait_idle()
    n = 7                                                      # 2 slots + --max-queue 2 = 4 admitted, the rest must get 429
    out, release, lock = [], threading.Event(), threading.Lock()

    def fire(i):
        import httpx
        with httpx.Client(base_url=s.url, timeout=300) as c:
            with c.stream("POST", "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Count to 100 slowly. %d" % i}],
                                                                  "stream": True, "max_tokens": 300, "temperature": 0.7}) as r:
                with lock:
                    out.append((r.status_code, r.headers.get("retry-after"), r.read() if r.status_code != 200 else b""))
                if r.status_code == 200:
                    release.wait(120)

    ths = []
    for i in range(n):
        th = threading.Thread(target=fire, args=(i,))
        th.start()
        ths.append(th)
        time.sleep(0.15)
    t0 = time.time()
    while len(out) < n and time.time() - t0 < 60:
        time.sleep(0.1)
    codes = sorted(c for c, _, _ in out)
    h = s.health()
    release.set()
    for th in ths:
        th.join(60)
    assert codes == [200] * 4 + [429] * 3, (codes, h)
    for c, ra, body in out:
        if c == 429:
            j = json.loads(body)
            assert ra is not None and int(ra) >= 1 and j["error"]["code"] == "queue_full" and j["error"]["type"] == "rate_limit_error"
    s.wait_idle(120)                                            # all four admitted streams were cancelled by the closed connections
    assert "ink_rejected_requests_total 3" in s.c.get("/metrics").text
    assert s.post("/v1/completions", {"prompt": "hello", "max_tokens": 2}).status_code == 200


# ---------------------------------------------------------------------------------------------------------------- (g)
def test_g_long_prompt_is_prefilled_in_chunks_without_stalling_health_and_ctx_overflow_is_400():
    s = server()
    s.wait_idle()
    T = tok()
    para = ("The committee reviewed the quarterly logistics report and noted that shipments from the northern depot arrived "
            "three days late because of storms, while the southern depot met every target. ")
    n_one = len(T.encode(para))
    reps = int(3000 / n_one) + 1
    text = para * reps
    ids = T.encode(text)
    n_prompt = len(ids)
    assert 2900 <= n_prompt <= 3300, n_prompt
    lat, snaps, stop = [], [], threading.Event()

    def watch():
        import httpx
        with httpx.Client(base_url=s.url, timeout=30) as c:
            while not stop.is_set():
                t0 = time.time()
                c.get("/health")
                lat.append(time.time() - t0)
                snaps.append([x["n_tokens"] for x in c.get("/slots").json()["slots"]])
                time.sleep(0.25)

    w = threading.Thread(target=watch)
    w.start()
    t0 = time.time()
    st, objs, first = s.stream({"prompt": ids, "max_tokens": 4, "temperature": 0}, "/v1/completions")
    total = time.time() - t0
    stop.set()
    w.join()
    assert st == 200 and objs[-1]["choices"] == [] or objs[-1]["choices"][0]["finish_reason"] in ("length", "stop")
    assert max(lat) < 1.5, "/health stalled during prefill: max %.2fs" % max(lat)
    mids = sorted({x for sn in snaps for x in sn if 100 < x < n_prompt})     # (the other slot still holds a few tokens of earlier tests)
    assert len(mids) >= 3 and all(x % 256 == 0 for x in mids), "prefill progress should be visible in 256-token chunks: %s" % mids
    rec("ttft_3000tok_prompt_s", round(first, 2))
    rec("health_max_latency_s_during_3000tok_prefill", round(max(lat), 3))
    rec("prompt_tokens_long", n_prompt)
    rec("prefill_tokens_per_s_3000", round(n_prompt / first, 1))
    # a prompt that exceeds the context is a clean 400, immediately, and the server stays usable
    big = T.encode(para * 400)
    assert len(big) > 9000, len(big)
    t0 = time.time()
    r = s.post("/v1/completions", {"prompt": big[:9000], "max_tokens": 4})
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded" and time.time() - t0 < 5, r.text
    r = s.post("/v1/chat/completions", {"messages": [{"role": "user", "content": para * 400}], "max_tokens": 4})
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded", r.text
    r = s.post("/v1/completions", {"prompt": big[:8192], "max_tokens": 1})       # exactly the context: no room for one token
    assert r.status_code == 400, r.text
    assert s.post("/v1/completions", {"prompt": "hello", "max_tokens": 2}).status_code == 200


# ---------------------------------------------------------------------------------------------------------------- (h)
def test_h_resident_memory_and_idle_cpu():
    s = server()
    s.wait_idle()
    rss = s.status("VmRSS") / 1048576.0
    hwm = s.status("VmHWM") / 1048576.0
    rec("rss_gib_after_all_tests", round(rss, 2))
    rec("rss_peak_gib_incl_load", round(hwm, 2))   # the load peak (mapped GGUF pages + the copied arena) is ~2x the steady state
    assert rss < 25.0, rss
    time.sleep(30)                                              # settle: the pool spins 2 ms after a job, then polls with 20 us sleeps
    w = 20.0
    t1, th1 = s.cpu_ticks(), s.thread_ticks()
    time.sleep(w)
    t2, th2 = s.cpu_ticks(), s.thread_ticks()
    tick = os.sysconf("SC_CLK_TCK")
    pct = (t2 - t1) / tick / w * 100.0
    per = sorted(((th2[k][1] - th1.get(k, ("", 0))[1]) / tick / w * 100.0, th2[k][0]) for k in th2)[::-1]
    rec("idle_cpu_percent_of_one_core", round(pct, 2))
    rec("idle_cpu_top_threads", [(round(p, 2), n) for p, n in per[:8]])
    rec("n_threads", len(th2))
    assert pct < 5.0, "idle CPU %.1f%% of one core (limit 5): top threads %s" % (pct, per[:8])


# ---------------------------------------------------------------------------------------------------------------- (i)
def test_i_greedy_speculation_equals_plain_decode_token_for_token():
    """The hard property: greedy output with speculation (auto and fixed width 4) == plain greedy output, 2 prompts x 64 tokens;
    `timings` carries the speculation counters."""
    s = spec_server()
    s.wait_idle()
    table = {}
    for name in ("s1_en_capital", "s2_code_add"):
        p_ids, ref, _ = _oracle(name)
        plain = _complete(s, p_ids, 64, speculation="off")
        want = plain["choices"][0]["token_ids"]
        assert "spec_steps" not in plain["timings"] and len(want) == 64, plain["timings"]
        row = {"plain_tps": plain["timings"]["predicted_per_second"], "equals_llamacpp_oracle": want == ref[:64]}
        for mode in ("auto", 4):
            d = _complete(s, p_ids, 64, speculation=mode)
            got, t = d["choices"][0]["token_ids"], d["timings"]
            assert got == want, (name, mode, got, want)
            for k in ("spec_steps", "spec_drafted", "spec_accepted", "spec_tokens_per_step", "spec_mode", "spec_synced", "spec_t_hist"):
                assert k in t, (k, t)
            assert 1 <= t["spec_steps"] <= 63 and t["spec_drafted"] >= t["spec_accepted"] >= 0 and t["spec_tokens_per_step"] >= 1.0, t
            assert t["spec_synced"] is True and t["spec_mode"] == (1 if mode == "auto" else 4), t
            assert abs(t["spec_tokens_per_step"] * t["spec_steps"] - 63) < 0.05, t          # 63 tokens came out of spec steps (the first anchor is sampled; 3-decimal rounding)
            if mode == 4:
                assert t["spec_drafted"] > 0 and set(t["spec_t_hist"]) <= {"1", "2", "3", "4"}, t   # the engine shrinks T to the remaining max_emit budget
            row["%s" % mode] = {"tps": t["predicted_per_second"], "steps": t["spec_steps"], "drafted": t["spec_drafted"], "accepted": t["spec_accepted"],
                                "tokens_per_step": t["spec_tokens_per_step"], "t_hist": t["spec_t_hist"]}
        table[name] = row
    rec("spec_greedy_64", table)


# ---------------------------------------------------------------------------------------------------------------- (j)
def test_j_sampled_speculation_is_reproducible_run_to_run():
    """Sampled (T=0.6, fixed seed): a FIXED verify width is a deterministic chain (first anchor via ink_sample, next_seed per step) ->
    identical tokens run to run (hard). `auto` is only recorded: the policy's adaptive state (exploration / probe history) persists across
    requests, so the width and with it the random draws can differ between two identical requests (seen 2026-11-22; docs/requests.md)."""
    s = spec_server()
    s.wait_idle()
    p_ids, _, _ = _oracle("s2_code_add")
    out = {}
    for mode in (4, 8, "auto"):
        a = _complete(s, p_ids, 48, temperature=0.6, seed=12345, speculation=mode)
        b = _complete(s, p_ids, 48, temperature=0.6, seed=12345, speculation=mode)
        ta, tb = a["choices"][0]["token_ids"], b["choices"][0]["token_ids"]
        same = ta == tb and a["timings"]["spec_steps"] == b["timings"]["spec_steps"]
        assert a["timings"]["spec_steps"] >= 1 and len(ta) >= 1, a["timings"]
        if mode != "auto":
            assert same, (mode, ta, tb, a["timings"], b["timings"])
        out[str(mode)] = {"reproducible": same, "n": len(ta), "steps": a["timings"]["spec_steps"], "accepted": a["timings"]["spec_accepted"],
                          "drafted": a["timings"]["spec_drafted"], "tps": a["timings"]["predicted_per_second"], "t_hist": [a["timings"]["spec_t_hist"], b["timings"]["spec_t_hist"]]}
    c = _complete(s, p_ids, 48, temperature=0.6, seed=54321, speculation=4)["choices"][0]["token_ids"]
    out["other_seed_differs"] = c != _complete(s, p_ids, 48, temperature=0.6, seed=12345, speculation=4)["choices"][0]["token_ids"]
    rec("spec_sampled_T0.6", out)


# ---------------------------------------------------------------------------------------------------------------- (k)
def test_k_stop_token_ends_the_turn_cleanly_and_the_slot_history_equals_the_fed_tokens():
    s = spec_server()
    s.wait_idle()
    msgs = [{"role": "user", "content": "Reply with just the word: pong"}]
    body = {"messages": msgs, "temperature": 0, "reasoning_effort": 0, "max_tokens": 120, "speculation": 8, "return_token_ids": True}
    r = s.post("/v1/chat/completions", body)
    assert r.status_code == 200, r.text
    d1 = r.json()
    ch, m, t = d1["choices"][0], d1["choices"][0]["message"], d1["timings"]
    assert ch["finish_reason"] == "stop" and (m["content"] or "").strip().lower().startswith("pong") and "<|" not in json.dumps(m), m
    assert t["spec_steps"] >= 1 and t["spec_synced"] is True, t
    hist = d1["usage"]["prompt_tokens"] + d1["usage"]["completion_tokens"]       # prompt + emitted tokens; the stop token is never fed
    assert d1["usage"]["completion_tokens"] == len(ch["token_ids"]) and 200006 not in ch["token_ids"]
    slots = s.c.get("/slots").json()["slots"]
    assert any(x["n_tokens"] == hist and not x["busy"] for x in slots), (hist, slots)
    plain = s.post("/v1/chat/completions", dict(body, speculation="off")).json()
    assert plain["choices"][0]["token_ids"] == ch["token_ids"], "stop handling inside a verify window changed the greedy output"
    # the next turn on that history hits the cache exactly (engine slot == server history == what the client sends back)
    asst = {"role": "assistant", "content": m["content"]}
    if m.get("reasoning_content"):
        asst["reasoning_content"] = m["reasoning_content"]
    d2 = s.post("/v1/chat/completions", {"messages": msgs + [asst, {"role": "user", "content": "Again, the same word only."}], "temperature": 0,
                                         "reasoning_effort": 0, "max_tokens": 60, "speculation": "auto"}).json()
    assert d2["usage"]["prompt_tokens_details"]["cached_tokens"] == hist, (d2["usage"], hist)
    assert d2["choices"][0]["finish_reason"] == "stop" and d2["timings"]["cache_n"] == hist and 0 < d2["timings"]["prompt_n"] < 40, d2["timings"]
    rec("spec_stop_turn", {"turn1_tokens": d1["usage"]["completion_tokens"], "steps": t["spec_steps"], "t_hist": t["spec_t_hist"], "accepted": t["spec_accepted"],
                           "drafted": t["spec_drafted"], "turn2_cached": hist, "turn2_prefilled": d2["timings"]["prompt_n"], "turn2_spec_steps": d2["timings"].get("spec_steps")})


# ---------------------------------------------------------------------------------------------------------------- (l)
def test_l_tool_call_turn_with_speculation_produces_a_valid_tool_call():
    s = spec_server()
    s.wait_idle()
    q = [{"role": "user", "content": "What is the weather in Paris?"}]
    base = {"messages": q, "tools": [WEATHER], "temperature": 0, "reasoning_effort": 0, "max_tokens": 300}
    ref = s.post("/v1/chat/completions", dict(base, speculation="off")).json()["choices"][0]
    assert ref["finish_reason"] == "tool_calls", ref
    out = {}
    for mode in ("auto", 8):
        r = s.post("/v1/chat/completions", dict(base, speculation=mode))
        assert r.status_code == 200, r.text
        d = r.json()
        ch = d["choices"][0]
        assert ch["finish_reason"] == "tool_calls", ch
        tc = ch["message"]["tool_calls"][0]
        assert tc["type"] == "function" and tc["id"].startswith("call_") and tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}, tc
        assert tc["function"] == ref["message"]["tool_calls"][0]["function"] and ch["message"].get("content") == ref["message"].get("content")
        assert d["timings"]["spec_steps"] >= 1, d["timings"]
        out[str(mode)] = {"steps": d["timings"]["spec_steps"], "accepted": d["timings"]["spec_accepted"], "drafted": d["timings"]["spec_drafted"],
                          "tokens": d["usage"]["completion_tokens"], "tps": d["timings"]["predicted_per_second"]}
    # forced call with speculation, then the tool result turn
    r = s.post("/v1/chat/completions", dict(base, tool_choice={"type": "function", "function": {"name": "get_weather"}}, max_tokens=100, speculation="auto"))
    ch = r.json()["choices"][0]
    assert ch["finish_reason"] == "tool_calls" and json.loads(ch["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}, ch
    tc = ch["message"]["tool_calls"][0]
    r2 = s.post("/v1/chat/completions", dict(base, messages=q + [{"role": "assistant", "content": None, "tool_calls": [tc]},
                                                                  {"role": "tool", "tool_call_id": tc["id"], "content": "18C and sunny"}], max_tokens=200, speculation="auto"))
    c2 = r2.json()["choices"][0]
    assert r2.status_code == 200 and c2["finish_reason"] == "stop" and "18" in (c2["message"]["content"] or ""), c2
    rec("spec_tool_call", out)


# ---------------------------------------------------------------------------------------------------------------- (m, n) S3: slot rewind
def _big_system(n_lines=62):
    words = ["apples", "bridges", "copper", "dolphins", "engines", "falcons", "granite", "harbors", "islands", "jasmine", "kettles", "lanterns"]
    lines = ["You are a terse assistant. Follow every rule below exactly."]
    for i in range(n_lines):
        lines.append("Rule %d: when the user mentions %s, answer in at most %d words, prefer the word %s, and never mention %s." %
                     (i + 1, words[i % 12], 5 + i % 7, words[(i * 5) % 12], words[(i * 7 + 3) % 12]))
    return "\n".join(lines)


def _local_ids(messages, effort):
    """The prompt ids the server builds for /v1/chat/completions without tools (app.py): template + the no-think header for effort 0."""
    from server import chat_template as ct
    t = tok()
    ids = t.encode(ct.get_template().render_openai(messages, None, effort), parse_special=True)
    if effort == 0:
        ids = ids + t.encode("<|message_model|><|content_text|>", parse_special=True)
    return ids


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _flush_slots(s):
    """Fill both slots with unrelated short prompts so the next request has no prefix to reuse (cold prefill)."""
    for i in range(2):
        r = s.post("/v1/completions", {"prompt": "Unrelated prompt number %d. Count to three:" % i, "max_tokens": 3, "temperature": 0})
        assert r.status_code == 200, r.text


def test_m_shared_system_prefix_is_reused_by_rewind_and_the_output_equals_a_cold_run():
    """Three conversations sharing a ~2k-token system prompt: the 2nd rewinds the 1st one's slot to a checkpoint <= the shared length
    (cached_tokens within one prefill chunk of the LCP), the 3rd hits the checkpoint left at the divergence point exactly, and the
    greedy output of a rewound request equals a cold run of the same request."""
    s = server()
    s.wait_idle()
    sysmsg = _big_system()
    qs = ["Q1: reply with the single word apple.", "Q2: reply with the single word banana.", "Q3: reply with the single word cherry."]
    msgs = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": q}] for q in qs]
    ids = [_local_ids(m, 0) for m in msgs]
    lcp12, lcp23 = _lcp(ids[0], ids[1]), _lcp(ids[1], ids[2])
    assert lcp12 >= 2000 and lcp23 == lcp12, (lcp12, lcp23, len(ids[0]))

    def ask(i):
        r = s.post("/v1/chat/completions", {"messages": msgs[i], "temperature": 0, "reasoning_effort": 0, "max_tokens": 12})
        assert r.status_code == 200, r.text
        return r.json()

    d1 = ask(0)
    u1, t1 = d1["usage"], d1["timings"]
    exact = u1["prompt_tokens"] == len(ids[0])           # local rendering == server rendering (template parity): exact LCP assertions
    assert u1["prompt_tokens"] >= 2000, u1
    d2 = ask(1)
    u2, t2 = d2["usage"], d2["timings"]
    c2 = u2["prompt_tokens_details"]["cached_tokens"]
    assert c2 >= 1024 and t2["cache_n"] == c2 and t2["prompt_n"] == u2["prompt_tokens"] - c2, (c2, u2, t2)
    if exact:
        assert lcp12 - 256 <= c2 <= lcp12, (c2, lcp12)   # nearest chunk checkpoint (--prefill-chunk 256) below the LCP
    d3 = ask(2)
    u3, t3 = d3["usage"], d3["timings"]
    c3 = u3["prompt_tokens_details"]["cached_tokens"]
    assert c3 >= c2 and c3 >= 2000, (c3, c2)
    if exact:
        assert c3 == lcp23, (c3, lcp23)                   # the divergence point of request 2 left a checkpoint exactly at the LCP
    _flush_slots(s)
    d2c = ask(1)                                          # cold run of request 2
    assert d2c["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, d2c["usage"]
    assert d2c["choices"][0]["message"]["content"] == d2["choices"][0]["message"]["content"], (d2c["choices"][0]["message"], d2["choices"][0]["message"])
    assert d2c["usage"]["completion_tokens"] == u2["completion_tokens"]
    assert t2["prompt_ms"] < 0.5 * d2c["timings"]["prompt_ms"], (t2["prompt_ms"], d2c["timings"]["prompt_ms"])
    rec("shared_prefix_rewind", {"prompt_tokens": u1["prompt_tokens"], "lcp": lcp12, "exact_template_parity": exact,
                                 "req2_cached": c2, "req2_prefill_ms": t2["prompt_ms"], "req2_new_tokens": t2["prompt_n"],
                                 "req3_cached": c3, "req3_prefill_ms": t3["prompt_ms"],
                                 "cold_prefill_ms": d2c["timings"]["prompt_ms"], "first_cold_prefill_ms": t1["prompt_ms"],
                                 "cold_prefill_tps": d2c["timings"]["prompt_per_second"]})


def test_n_caret_pattern_rerendered_assistant_turn_rewinds_to_the_prompt_end():
    """The Caret / Cursor case: turn 2 resends the conversation with the assistant message re-rendered WITHOUT its reasoning_content, so the
    slot history is not a prefix of the new prompt. The server rewinds to the deep checkpoint at the end of turn 1's prompt:
    cached_tokens == turn 1's prompt_tokens, only the re-rendered assistant turn + the new user turn are prefilled, output == cold run."""
    s = server()
    s.wait_idle()
    msgs = [{"role": "system", "content": _big_system()}, {"role": "user", "content": "Explain in one sentence why the sky is blue."}]
    body = {"messages": msgs, "temperature": 0, "reasoning_effort": "low", "max_tokens": 40}
    r1 = s.post("/v1/chat/completions", body)
    assert r1.status_code == 200, r1.text
    d1 = r1.json()
    m1, u1 = d1["choices"][0]["message"], d1["usage"]
    assert u1["completion_tokens"] > 0
    asst = {"role": "assistant", "content": m1.get("content") or ""}          # reasoning_content dropped (what Caret sends back)
    follow = msgs + [asst, {"role": "user", "content": "Now one sentence about sunsets."}]
    r2 = s.post("/v1/chat/completions", dict(body, messages=follow))
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    u2, t2 = d2["usage"], d2["timings"]
    c2 = u2["prompt_tokens_details"]["cached_tokens"]
    lcp_local = _lcp(_local_ids(msgs, "low"), _local_ids(follow, "low"))   # how much of turn 1's prompt the re-rendered prompt really shares
    if lcp_local >= u1["prompt_tokens"]:
        assert c2 == u1["prompt_tokens"], (c2, u1["prompt_tokens"], lcp_local)   # the prompt-end checkpoint of turn 1, whatever was generated after it
    else:   # the template renders the assistant turn with a header that differs from the generation prompt: the LCP is inside turn 1's prompt
        assert lcp_local - 256 <= c2 <= lcp_local and c2 >= 1024, (c2, lcp_local, u1["prompt_tokens"])
    assert t2["prompt_n"] == u2["prompt_tokens"] - c2 < 300, t2
    _flush_slots(s)
    r2c = s.post("/v1/chat/completions", dict(body, messages=follow))
    d2c = r2c.json()
    assert d2c["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, d2c["usage"]
    mc, mw = d2c["choices"][0]["message"], d2["choices"][0]["message"]
    assert (mc.get("content"), mc.get("reasoning_content")) == (mw.get("content"), mw.get("reasoning_content")), (mc, mw)
    assert t2["prompt_ms"] < 0.5 * d2c["timings"]["prompt_ms"], (t2["prompt_ms"], d2c["timings"]["prompt_ms"])
    rec("caret_pattern", {"turn1_prompt_tokens": u1["prompt_tokens"], "turn1_completion_tokens": u1["completion_tokens"], "lcp_local": lcp_local,
                          "turn2_prompt_tokens": u2["prompt_tokens"], "turn2_cached": c2, "turn2_new_tokens": t2["prompt_n"],
                          "turn2_prefill_ms_with_rewind": t2["prompt_ms"], "turn2_prefill_ms_cold": d2c["timings"]["prompt_ms"],
                          "turn2_ttft_ms_with_rewind": t2["ttft_ms"], "turn2_ttft_ms_cold": d2c["timings"]["ttft_ms"]})


# ---------------------------------------------------------------------------------------------------------------- (o) S3: rewind + speculation
def test_o_rewound_slot_reports_unsynced_speculation_and_still_decodes_the_plain_output():
    """With the drafter loaded: a rewound slot has a stale drafter context, so ink_spec_synced is 0 and the speculative steps run plain
    (spec_synced False, every step T = 1) while the output equals the non-speculative output; a reset slot is synced again."""
    s = spec_server()
    s.wait_idle()
    sysmsg = _big_system()
    def ask(q, **kw):
        r = s.post("/v1/chat/completions", dict({"messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}],
                                                 "temperature": 0, "reasoning_effort": 0, "max_tokens": 16}, **kw))
        assert r.status_code == 200, r.text
        return r.json()
    d1 = ask("Q1: reply with the single word apple.", speculation=4)
    t1 = d1["timings"]
    assert d1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and t1.get("spec_synced") is True, t1   # fresh slot: drafter in sync
    d2 = ask("Q2: reply with the single word banana.", speculation=4)     # shares the system prompt: rewind -> drafter stale
    u2, t2 = d2["usage"], d2["timings"]
    assert u2["prompt_tokens_details"]["cached_tokens"] >= 1024, u2
    assert t2.get("spec_synced") is False and t2["spec_steps"] >= 1 and set(t2["spec_t_hist"]) == {"1"} and t2["spec_drafted"] == 0, t2
    d2p = ask("Q2: reply with the single word banana.", speculation="off")   # identical prompt: cached logits, plain decode on the same slot
    assert d2p["choices"][0]["message"]["content"] == d2["choices"][0]["message"]["content"], (d2p["choices"][0]["message"], d2["choices"][0]["message"])
    r3 = s.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Name a colour. One word."}], "temperature": 0,
                                         "reasoning_effort": 0, "max_tokens": 8, "speculation": 4})   # no system prompt = no shared prefix: LRU reset -> drafter resynced
    assert r3.status_code == 200, r3.text
    d3 = r3.json()
    assert d3["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and d3["timings"].get("spec_synced") is True, d3["timings"]
    rec("rewind_spec", {"req2_cached": u2["prompt_tokens_details"]["cached_tokens"], "req2_spec_synced": t2["spec_synced"], "req2_t_hist": t2["spec_t_hist"],
                        "req3_spec_synced": d3["timings"]["spec_synced"]})


if __name__ == "__main__":
    only = [x for x in os.environ.get("INK_REAL_ONLY", "").split(",") if x]       # e.g. INK_REAL_ONLY=g,h  (tests a..o)
    if only:
        for k in [k for k in list(globals()) if k.startswith("test_") and k[5] not in only]:
            del globals()[k]
    rc = helpers.run_module(globals())
    for srv in (_S, _SP):
        if srv is not None:
            srv.stop()
            print("server log tail:\n" + srv.tail(600))
    raise SystemExit(rc)

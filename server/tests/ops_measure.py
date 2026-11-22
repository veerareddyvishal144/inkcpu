"""Operational measurements of the real engine behind the server (feeds docs/SERVING.md). NOT a pass/fail test.

    INK_GGUF=... INK_LIB=... INK_CPUS=22,24,26,28,30 /data/engine/.venv/bin/python server/tests/ops_measure.py A|B1|B2|C [out.json]

D   prefill speed for --prefill-chunk 64 / 128 / 256 / 512 (one server restart each).
A   server ctx 8192, 2 slots: RSS / KV growth, TTFT for 300/1000 token prompts, decode tok/s (HTTP vs engine-only, sampler cost for
    several sampling settings), 2 concurrent streams, prefill stalls the other stream, client timeout / disconnect / cancel
    during prefill, SIGTERM during generation and restart time.
B1  engine only (no HTTP), ctx 512: what the engine does when a forward would exceed the slot context.
B2  server ctx 512: clean 400s, max_tokens clamped to the room that is left, multi-turn growth until the context is full.
C   server with ONE slot: a second client waits in the queue (compare with the 2-slot concurrency of A).
Timings on the shared host: indicative only.
"""
import json
import os
import signal
import sys
import threading
import time

import helpers
from helpers import tok

import test_ink_real as R

BASE = ("The committee reviewed the quarterly logistics report and noted that shipments from the northern depot arrived three days "
        "late because of storms, while the southern depot met every target. ")
P13 = [371, 151451, 149879, 151194, 149884, 152717, 11, 149880, 149876, 151451, 149879, 153067, 149884]


def ids_of_len(n, salt):
    return tok().encode(("[%s] " % salt) + BASE * (n // 20 + 2))[:n]


def rss(s):
    return round(s.status("VmRSS") / 1048576.0, 3)


def complete(s, prompt, n, **kw):
    r = s.post("/v1/completions", dict({"prompt": prompt, "max_tokens": n, "temperature": 0}, **kw))
    assert r.status_code == 200, r.text
    return r.json()


def stream_ts(s, body, path="/v1/completions"):
    """Streaming request; returns (status, list of (t_since_start, event) for events that carry text, t_first_text, t_total)."""
    out, t0, first = [], time.time(), None
    with s.c.stream("POST", path, json=dict(body, stream=True)) as r:
        if r.status_code != 200:
            r.read()
            return r.status_code, r.json(), None, time.time() - t0
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                o = json.loads(line[6:])
                ch = (o.get("choices") or [{}])[0]
                txt = ch.get("text") if "text" in ch else (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content")
                if txt:
                    t = time.time() - t0
                    first = first if first is not None else t
                    out.append(t)
    return 200, out, first, time.time() - t0


def wait_active0(s, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if s.health()["active"] == 0:
            return time.time() - t0
        time.sleep(0.02)
    return None


def phase_A(res):
    s = R.Server(ctx=8192, slots=2, max_queue=8)
    res["server_load_s"], res["rss_after_load_gib"] = round(s.load_s, 1), rss(s)
    res["vm_hwm_gib_incl_load"] = round(s.status("VmHWM") / 1048576.0, 2)
    complete(s, ids_of_len(40, "warm"), 2)                                    # allocates the prefill scratch
    res["rss_after_warmup_gib"] = rss(s)
    # ---- TTFT and RSS growth (prompts are unique: no prefix hits)
    res["ttft"], res["rss_trace"] = [], []
    for n in (300, 300, 1000, 1000):
        r0 = rss(s)
        d = complete(s, ids_of_len(n, "ttft%d_%d" % (n, len(res["ttft"]))), 1)
        tm = d["timings"]
        s.wait_idle()
        res["ttft"].append({"prompt_tokens": n, "ttft_ms": tm["ttft_ms"], "prefill_ms": tm["prompt_ms"], "prefill_tok_per_s": tm["prompt_per_second"],
                            "wall_ms": tm["total_ms"]})
        res["rss_trace"].append({"after_prompt_tokens": n, "rss_gib": rss(s), "delta_mib": round((rss(s) - r0) * 1024, 1),
                                 "slot_tokens": [x["n_tokens"] for x in s.c.get("/slots").json()["slots"]]})
    # ---- decode speed: HTTP vs engine only, sampler cost
    res["decode"] = {}
    for i, (name, kw) in enumerate((("greedy", dict(temperature=0)), ("T0.6", dict(temperature=0.6, seed=1)),
                                    ("T0.6_top_k40", dict(temperature=0.6, top_k=40, seed=1)), ("T0.6_top_p0.9", dict(temperature=0.6, top_p=0.9, seed=1)),
                                    ("T0.6_rep_penalty1.1", dict(temperature=0.6, repetition_penalty=1.1, seed=1)))):
        d = complete(s, P13 + [i + 100], 96, **kw)
        tm, n = d["timings"], d["usage"]["completion_tokens"]
        res["decode"][name] = {"tokens": n, "http_decode_tok_s": tm["predicted_per_second"], "engine_only_tok_s": round(n / (tm["engine_ms"] / 1000.0), 2),
                               "engine_ms_per_tok": round(tm["engine_ms"] / n, 1), "sample_ms_per_tok": round(tm["sample_ms"] / n, 3),
                               "other_ms_per_tok": round((tm["predicted_ms"] - tm["engine_ms"] - tm["sample_ms"]) / n, 3)}
    st, ts, first, total = stream_ts(s, {"prompt": P13 + [200], "max_tokens": 96, "temperature": 0})
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    res["client_stream_96tok"] = {"events": len(ts), "client_tok_s": round((len(ts) - 1) / (ts[-1] - ts[0]), 2), "max_gap_s": round(max(gaps), 3)}
    # ---- two concurrent streams (2 slots)
    s.wait_idle()
    outs = {}

    def go(k):
        outs[k] = stream_ts(s, {"prompt": P13 + [300 + k], "max_tokens": 64, "temperature": 0})
    ths = [threading.Thread(target=go, args=(k,)) for k in (0, 1)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    ev = [len(outs[k][1]) for k in (0, 1)]
    res["two_streams_64tok"] = {"wall_s": round(wall, 2), "tokens": ev, "aggregate_tok_s": round(sum(ev) / wall, 2),
                                "per_stream_client_tok_s": [round((ev[k] - 1) / (outs[k][1][-1] - outs[k][1][0]), 2) for k in (0, 1)],
                                "first_token_s": [round(outs[k][2], 2) for k in (0, 1)]}
    # ---- a long prefill of one client stalls the decode of the other (granule = one prefill chunk of 256 tokens)
    s.wait_idle()
    box = {}

    def decoder():
        box["a"] = stream_ts(s, {"prompt": P13 + [400], "max_tokens": 80, "temperature": 0})
    th = threading.Thread(target=decoder)
    th.start()
    time.sleep(4.0)
    t0 = time.time()
    d = complete(s, ids_of_len(1000, "stall"), 1)
    res["prefill_1000_while_other_decodes"] = {"wall_s": round(time.time() - t0, 2), "prefill_ms": d["timings"]["prompt_ms"]}
    th.join()
    ts = box["a"][1]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    res["decode_stream_gaps_during_other_prefill"] = {"max_gap_s": round(max(gaps), 2), "gaps_over_1s": sum(1 for g in gaps if g > 1.0),
                                                      "median_gap_s": round(sorted(gaps)[len(gaps) // 2], 3)}
    # ---- client timeout (non-stream, read timeout 2 s), disconnect, cancel during a long prefill
    import httpx
    s.wait_idle()
    t_ex = None
    try:
        with httpx.Client(base_url=s.url, timeout=2.0) as c:
            c.post("/v1/completions", json={"prompt": P13 + [500], "max_tokens": 200, "temperature": 0})
    except httpx.TimeoutException:
        t_ex = time.time()
    res["client_timeout_2s_nonstream_slot_free_after_s"] = round(wait_active0(s, 30) or -1, 2) if t_ex else "no timeout?"
    s.wait_idle()
    big = ids_of_len(3000, "cancelprefill")
    box = {}

    def long_prefill():
        try:
            with httpx.Client(base_url=s.url, timeout=300) as c:
                with c.stream("POST", "/v1/completions", json={"prompt": big, "max_tokens": 4, "stream": True}) as r:
                    box["t"] = time.time()
                    time.sleep(6.0)                                      # close the connection in the middle of the prefill
        except Exception as e:  # noqa: BLE001
            box["err"] = repr(e)
    th = threading.Thread(target=long_prefill)
    th.start()
    th.join()
    t_close = time.time()
    res["cancel_during_3000tok_prefill_slot_free_after_s"] = round(wait_active0(s, 60) or -1, 2)
    res["slots_after_cancelled_prefill"] = [x["n_tokens"] for x in s.c.get("/slots").json()["slots"]]
    # ---- metrics snapshot + log sample
    res["metrics_excerpt"] = [l for l in s.c.get("/metrics").text.splitlines() if l.startswith(("ink_seconds_total", "ink_finished_total", "ink_slot_tokens", "ink_tokens_total"))]
    log = [json.loads(x) for x in open(s.req_log) if x.strip()]
    res["request_log_fields"] = sorted(log[-1].keys()) if log else []
    # ---- SIGTERM during generation, then restart
    box = {}

    def gen():
        try:
            box["r"] = stream_ts(s, {"prompt": P13 + [600], "max_tokens": 300, "temperature": 0})
        except Exception as e:  # noqa: BLE001
            box["err"] = repr(e)
    th = threading.Thread(target=gen)
    th.start()
    time.sleep(3.0)
    t0 = time.time()
    s.proc.send_signal(signal.SIGTERM)
    try:
        s.proc.wait(60)
        res["sigterm_mid_generation_exit_s"] = round(time.time() - t0, 2)
        res["sigterm_exit_code"] = s.proc.returncode
    except Exception:  # noqa: BLE001
        s.proc.kill()
        res["sigterm_mid_generation_exit_s"] = "timeout (killed)"
    th.join(30)
    res["client_saw_after_sigterm"] = box.get("err") or "stream ended with %d tokens" % len(box.get("r", (0, []))[1])
    s2 = R.Server(ctx=8192, slots=2, max_queue=8)
    res["restart_to_healthy_s"] = round(s2.load_s, 1)
    s2.stop()


def phase_B1(res):
    from server.backends import ContextFull, InkBackend
    out = {}

    def run():
        try:
            be = InkBackend(R.GGUF, R.LIB, len([c for c in os.environ["INK_CPUS"].split(",") if c]), 1, 512)
            be.start()
            sl = be.slot_new()
            be.prefill(sl, ids_of_len(400, "ctx"))
            out["pos_after_400"] = be.slot_pos(sl)
            try:
                be.prefill(sl, ids_of_len(200, "x"))
                out["overflow_prefill"] = "NO ERROR"
            except ContextFull as e:
                out["overflow_prefill"] = "ContextFull: %s" % e
            out["pos_after_failed_prefill"] = be.slot_pos(sl)
            be.prefill(sl, ids_of_len(100, "y"))
            out["pos_after_100_more"] = be.slot_pos(sl)
            be.prefill(sl, ids_of_len(12, "z"))
            out["pos_at_ctx"] = be.slot_pos(sl)
            try:
                be.step(sl, 5)
                out["step_at_full_ctx"] = "NO ERROR"
            except ContextFull as e:
                out["step_at_full_ctx"] = "ContextFull: %s" % e
            out["pos_after_failed_step"] = be.slot_pos(sl)
            be.slot_reset(sl)
            be.prefill(sl, ids_of_len(50, "again"))
            out["usable_after_reset_pos"] = be.slot_pos(sl)
            be.close()
        except BaseException as e:  # noqa: BLE001
            out["error"] = repr(e)
    th = threading.Thread(target=run)
    th.start()
    th.join(600)
    res.update(out)


def phase_B2(res):
    s = R.Server(ctx=512, slots=2, max_queue=4)
    T = tok()
    r = s.post("/v1/completions", {"prompt": ids_of_len(600, "big"), "max_tokens": 4})
    res["prompt_600_ctx512"] = (r.status_code, r.json()["error"]["code"], r.json()["error"]["message"])
    r = s.post("/v1/completions", {"prompt": ids_of_len(512, "exact"), "max_tokens": 1})
    res["prompt_512_ctx512"] = (r.status_code, r.json()["error"]["code"])
    d = complete(s, ids_of_len(480, "room"), 100)
    res["prompt_480_max_tokens_100"] = {"finish_reason": d["choices"][0]["finish_reason"], "completion_tokens": d["usage"]["completion_tokens"],
                                        "prompt_tokens": d["usage"]["prompt_tokens"]}
    d = complete(s, ids_of_len(511, "room1"), 50)
    res["prompt_511_max_tokens_50"] = {"finish_reason": d["choices"][0]["finish_reason"], "completion_tokens": d["usage"]["completion_tokens"]}
    # a chat that grows turn by turn until the context is full
    msgs, turns = [{"role": "user", "content": "Tell me a fact about the number %d." % 7}], 0
    final = None
    for turn in range(12):
        r = s.post("/v1/chat/completions", {"messages": msgs, "temperature": 0.7, "reasoning_effort": 0, "max_tokens": 200})
        if r.status_code != 200:
            final = (turn, r.status_code, r.json()["error"]["code"], r.json()["error"]["message"])
            break
        d = r.json()
        m = d["choices"][0]["message"]
        turns += 1
        a = {"role": "assistant", "content": m.get("content")}
        if m.get("reasoning_content"):
            a["reasoning_content"] = m["reasoning_content"]
        msgs += [a, {"role": "user", "content": "Another fact about a different number, please."}]
        res.setdefault("chat_growth", []).append({"turn": turn, "prompt_tokens": d["usage"]["prompt_tokens"], "completion_tokens": d["usage"]["completion_tokens"],
                                                  "finish": d["choices"][0]["finish_reason"], "cached": d["usage"]["prompt_tokens_details"]["cached_tokens"]})
    res["chat_growth_end"] = final
    res["health_after"] = s.health()
    res["server_log_errors"] = [l for l in s.tail(6000).splitlines() if "Traceback" in l or "Error" in l][:5]
    s.stop()


def phase_C(res):
    s = R.Server(ctx=8192, slots=1, max_queue=4)
    complete(s, ids_of_len(40, "warm"), 2)
    outs = {}

    def go(k, delay):
        time.sleep(delay)
        t0 = time.time()
        d = complete(s, P13 + [700 + k], 48)
        outs[k] = (d["timings"], time.time() - t0)
    ths = [threading.Thread(target=go, args=(0, 0.0)), threading.Thread(target=go, args=(1, 1.0))]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    res["one_slot_two_clients_48tok"] = {"wall_s": round(time.time() - t0, 2),
                                         "client0": {"queue_ms": outs[0][0]["queue_ms"], "ttft_ms": outs[0][0]["ttft_ms"], "request_wall_s": round(outs[0][1], 2),
                                                     "decode_tok_s": outs[0][0]["predicted_per_second"]},
                                         "client1": {"queue_ms": outs[1][0]["queue_ms"], "ttft_ms": outs[1][0]["ttft_ms"], "request_wall_s": round(outs[1][1], 2),
                                                     "decode_tok_s": outs[1][0]["predicted_per_second"]}}
    s.stop()


def phase_D(res):
    """Prefill speed and the time one chunk takes (= how long a cancelled prefill keeps running, = how long the other slot's decode
    stalls) for several --prefill-chunk values; a fresh 700-token prompt each (warm-up first)."""
    res["chunks"] = {}
    for chunk in (64, 128, 256, 512):
        s = R.Server(ctx=8192, slots=2, max_queue=4, extra=["--prefill-chunk", str(chunk)])
        complete(s, ids_of_len(40, "warm"), 2)
        d = complete(s, ids_of_len(700, "chunk%d" % chunk), 1)
        tm = d["timings"]
        res["chunks"][chunk] = {"prefill_ms": tm["prompt_ms"], "tok_per_s": tm["prompt_per_second"], "approx_chunk_s": round(chunk / tm["prompt_per_second"], 2),
                                "loadavg": open("/proc/loadavg").read().split()[0]}
        s.stop()


if __name__ == "__main__":
    R._need()
    phase = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops_%s.json" % phase)
    res = {"phase": phase, "start_utc": time.strftime("%H:%M:%S", time.gmtime()), "loadavg_start": open("/proc/loadavg").read().split()[:3]}
    try:
        {"A": phase_A, "B1": phase_B1, "B2": phase_B2, "C": phase_C, "D": phase_D}[phase](res)
    finally:
        res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))

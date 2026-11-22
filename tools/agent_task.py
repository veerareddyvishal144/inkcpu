#!/usr/bin/env python3
"""Agentic coding task against any OpenAI-compatible server (llama-server earlier, the inkcpu server now).

The model gets three tools (read_file, write_file, run_tests) in a FRESH copy of the work dir (only the test file, not the earlier
solution) and must implement textstats.py so that test_textstats.py passes.  Prints one line per step (wall time, prompt tokens that
had to be prefilled, tokens served from the cache, generated tokens and their tokens/s as reported by the server) and the final test run.

    /data/engine/.venv/bin/python tools/agent_task.py --base-url http://127.0.0.1:8100 --temperature 0.6 --out run1.json

Timing fields: the inkcpu server and llama-server both return a non-standard `timings` object (prompt_n = NEW prompt tokens,
cache_n / usage.prompt_tokens_details.cached_tokens = reused, predicted_n, predicted_per_second). `--drop-reasoning` reproduces the
older harness, which sent assistant messages back WITHOUT reasoning_content (the inkcpu server then cannot reuse its slot cache).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8001")
ap.add_argument("--model", default="inkling-2-mini")
ap.add_argument("--seed-dir", default="/data/inkling-mini/agent-work", help="directory holding the task's test file")
ap.add_argument("--seed-files", nargs="*", default=["test_textstats.py"], help="files copied into the fresh work dir (NOT the solution)")
ap.add_argument("--work-dir", default=None, help="fresh work dir (default: a new temp dir; its parent is $TMPDIR)")
ap.add_argument("--temperature", type=float, default=0.6)
ap.add_argument("--max-tokens", type=int, default=3000)
ap.add_argument("--max-steps", type=int, default=12)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--reasoning-effort", default=None, help="passed through as reasoning_effort when given")
ap.add_argument("--drop-reasoning", action="store_true", help="do not send reasoning_content back (old harness behaviour)")
ap.add_argument("--speculation", default=None, help="inkcpu server: speculation field per request (auto | off | 2..8); default: not sent (server default)")
ap.add_argument("--timeout", type=float, default=3000)
ap.add_argument("--out", default=None, help="write a JSON record of the run here")
args = ap.parse_args()

if args.work_dir:
    W = os.path.realpath(args.work_dir)
    os.makedirs(W, exist_ok=True)
else:
    W = os.path.realpath(tempfile.mkdtemp(prefix="agentwork_"))
for f in args.seed_files:
    shutil.copy(os.path.join(args.seed_dir, f), os.path.join(W, f))

TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": list(p)}}} for n, d, p in [
    ("read_file", "Read a file in the work dir", {"path": {"type": "string"}}),
    ("write_file", "Write a file in the work dir", {"path": {"type": "string"}, "content": {"type": "string"}}),
    ("run_tests", "Run python3 -m unittest -v in the work dir", {})]]


def tool(n, a):
    try:
        if n == "run_tests":
            r = subprocess.run(["python3", "-I", "-m", "unittest", "-v"], cwd=W, capture_output=True, text=True, timeout=60,
                               env={"PATH": "/usr/bin:/bin", "PYTHONPATH": W})
            return (r.stdout + r.stderr)[-2500:]
        p = os.path.realpath(os.path.join(W, a["path"]))
        if not p.startswith(W + "/"):
            return "error: path outside work dir"
        if n == "read_file":
            return open(p).read()[:6000]
        if n == "write_file":
            open(p, "w").write(a["content"])
            return "ok"
    except Exception as e:
        return f"error: {e}"
    return "error: unknown tool"


msgs = [{"role": "system", "content": "You are a coding agent. Use the tools. Finish when all tests pass."},
        {"role": "user", "content": "Implement textstats.py in the work dir so that test_textstats.py passes. Read the tests first, write the module, run the tests, and fix failures."}]
record = {"base_url": args.base_url, "temperature": args.temperature, "work_dir": W, "drop_reasoning": args.drop_reasoning, "speculation": args.speculation, "steps": []}
T0 = time.time()
for step in range(1, args.max_steps + 1):
    t = time.time()
    body = {"model": args.model, "messages": msgs, "tools": TOOLS, "max_tokens": args.max_tokens, "temperature": args.temperature}
    if args.seed is not None:
        body["seed"] = args.seed + step
    if args.speculation is not None:
        body["speculation"] = int(args.speculation) if args.speculation.isdigit() else args.speculation
    if args.reasoning_effort is not None:
        body["reasoning_effort"] = args.reasoning_effort
    req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=args.timeout))
    m = d["choices"][0]["message"]
    tm = d.get("timings") or {}
    us = d.get("usage") or {}
    cached = tm.get("cache_n")
    if cached is None:
        cached = (us.get("prompt_tokens_details") or {}).get("cached_tokens")
    gen = tm.get("predicted_n", us.get("completion_tokens"))
    tps = tm.get("predicted_per_second") or 0.0
    calls = m.get("tool_calls") or []
    wall = time.time() - t
    prompt_new = tm.get("prompt_n")
    spec = ""
    if tm.get("spec_steps"):
        spec = f" spec_steps={tm['spec_steps']} acc={tm.get('spec_accepted')}/{tm.get('spec_drafted')} tok/step={tm.get('spec_tokens_per_step')}"
    print(f"[step {step}] {wall:.0f}s prompt_new={prompt_new} cached={cached} gen={gen}@{tps:.1f} tok/s prefill_ms={tm.get('prompt_ms')} "
          f"ttft_ms={tm.get('ttft_ms')} calls={[c['function']['name'] for c in calls]}{spec}", flush=True)
    record["steps"].append({"step": step, "wall_s": round(wall, 2), "prompt_tokens": us.get("prompt_tokens"), "prompt_new": prompt_new,
                            "cached": cached, "gen": gen, "decode_tps": tps, "prefill_ms": tm.get("prompt_ms"), "ttft_ms": tm.get("ttft_ms"),
                            "engine_ms": tm.get("engine_ms"), "sample_ms": tm.get("sample_ms"), "spec_steps": tm.get("spec_steps"),
                            "spec_drafted": tm.get("spec_drafted"), "spec_accepted": tm.get("spec_accepted"), "spec_t_hist": tm.get("spec_t_hist"),
                            "calls": [c["function"]["name"] for c in calls], "finish": d["choices"][0].get("finish_reason")})
    keep = ("role", "content", "tool_calls") if args.drop_reasoning else ("role", "content", "tool_calls", "reasoning_content")
    msgs.append({k: v for k, v in m.items() if k in keep})
    if not calls:
        print("FINAL:", (m.get("content") or "")[:400], flush=True)
        record["final"] = m.get("content")
        break
    for c in calls:
        try:
            a = json.loads(c["function"]["arguments"] or "{}")
        except Exception:
            a = {}
        out = tool(c["function"]["name"], a)
        print(f"   -> {c['function']['name']}: {out[:150]!r}", flush=True)
        msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
total = time.time() - T0
print("TOTAL", f"{total:.0f}s", flush=True)
final = tool("run_tests", {})
print(final[-400:], flush=True)
ok = bool(re.search(r"^OK\s*$", final, re.M))
mm = re.search(r"Ran (\d+) tests?", final)
record.update(total_s=round(total, 1), tests_ok=ok, tests_ran=int(mm.group(1)) if mm else None, n_steps=len(record["steps"]),
              gen_tokens=sum(s["gen"] or 0 for s in record["steps"]), prefill_new_tokens=sum(s["prompt_new"] or 0 for s in record["steps"]),
              cached_tokens=sum(s["cached"] or 0 for s in record["steps"]), spec_steps=sum(s["spec_steps"] or 0 for s in record["steps"]),
              spec_drafted=sum(s["spec_drafted"] or 0 for s in record["steps"]), spec_accepted=sum(s["spec_accepted"] or 0 for s in record["steps"]))
record["spec_tokens_per_step"] = round(record["gen_tokens"] / record["spec_steps"], 2) if record["spec_steps"] else None
print("RESULT", json.dumps({k: record[k] for k in ("total_s", "tests_ok", "tests_ran", "n_steps", "gen_tokens", "prefill_new_tokens", "cached_tokens",
                                                   "spec_steps", "spec_drafted", "spec_accepted", "spec_tokens_per_step")}), flush=True)
if args.out:
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
sys.exit(0 if ok else 1)

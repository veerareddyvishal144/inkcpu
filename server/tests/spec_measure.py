#!/usr/bin/env python
"""Speculative decoding A/B through HTTP: plain (speculation off) vs auto vs fixed verify width 4 vs 8, greedy, N new tokens, on
tools/dspark/prompts/code_completion.txt and chat_answer.txt. One server subprocess with --drafter (test_ink_real.Server).
    INK_GGUF=... INK_LIB=... INK_DRAFTER=... INK_CPUS=22,24,26,28,30 /data/engine/.venv/bin/python server/tests/spec_measure.py [--n 100] [--out F.json]
Shared-host numbers are indicative only (load average is recorded)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_ink_real as R  # noqa: E402

n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100
out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
R._need()
assert R.DRAFTER, "INK_DRAFTER not set"
s = R.Server(extra=["--drafter", R.DRAFTER])
print("server up in %.1f s, RSS %.2f GiB, INK_SPEC_EXACT=%s" % (s.load_s, s.status("VmRSS") / 1048576.0, os.environ.get("INK_SPEC_EXACT", "(unset)")), flush=True)
res = {"n": n, "load_start": open("/proc/loadavg").read().split()[:3], "prompts": {}}
for name in ("code_completion", "chat_answer"):
    text = open(os.path.join(R.ENGINE, "tools", "dspark", "prompts", name + ".txt")).read()
    rows, want = {}, None
    for mode in ("off", "auto", 4, 8):
        s.wait_idle()
        d = R._complete(s, text, n, speculation=mode)
        t, toks = d["timings"], d["choices"][0]["token_ids"]
        if want is None:
            want = toks
        rows[str(mode)] = {"tps": t["predicted_per_second"], "predicted_ms": t["predicted_ms"], "engine_ms": t["engine_ms"], "n": t["predicted_n"],
                           "prompt_n": t["prompt_n"], "steps": t.get("spec_steps"), "drafted": t.get("spec_drafted"), "accepted": t.get("spec_accepted"),
                           "tok_per_step": t.get("spec_tokens_per_step"), "t_hist": t.get("spec_t_hist"), "same_as_plain": toks == want,
                           "finish": d["choices"][0]["finish_reason"], "load": open("/proc/loadavg").read().split()[0]}
        print("%-16s %-5s %6.2f tok/s  n=%3d steps=%-4s acc=%s/%s tok/step=%s T=%s same_as_plain=%s load=%s" % (
            name, mode, t["predicted_per_second"], t["predicted_n"], t.get("spec_steps"), t.get("spec_accepted"), t.get("spec_drafted"),
            t.get("spec_tokens_per_step"), t.get("spec_t_hist"), toks == want, rows[str(mode)]["load"]), flush=True)
    res["prompts"][name] = rows
res["load_end"] = open("/proc/loadavg").read().split()[:3]
s.stop()
if out_path:
    json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res["prompts"], indent=None)[:3000])

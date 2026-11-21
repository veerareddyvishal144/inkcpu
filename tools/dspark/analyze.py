#!/usr/bin/env python3
"""Summarise tools/dspark/raw/*.steps into markdown tables (acceptance per position, mean accepted length, bytes-model speed-ups).
Usage: /usr/bin/python3 -I analyze.py [raw_dir]  -> prints markdown, writes raw/summary.md"""
import os, re, sys, glob, math, collections

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
STEP = re.compile(r"STEP new=(\d) n_draft=(\d+) n_acc=(\d+) t_draft_us=(\d+) t_target_us=(\d+) t_proc_us=(\d+) rej=(\d)")
DEC = re.compile(r"decoded\s+(\d+) tokens in\s+([\d.]+) seconds, speed:\s+([\d.]+) t/s")
ENC = re.compile(r"encoded\s+(\d+) tokens")

# ROUTING.md (agent E) numbers, MB
E = {1: 1847.3, 2: 2928.1, 3: 3803.1, 4: 4553.8, 5: 5218.7, 6: 5819.7, 7: 6369.6, 8: 6876.3}
N_MB, B1 = 2014.6, 3861.9
LAYERS_MB, SMALL_HEAD_MB, LMHEAD_MB, M_BF16_MB = 523.8, 9.1, 489.7, 200058 * 256 * 2 / 1e6

def dr_routing(n):  # ROUTING.md: whole drafter 635.8 MB + one lm_head pass
    return 635.8 + LMHEAD_MB

def dr_refined(n):  # layers + joint/hidden_out + batched base lm_head pass + one pass over a precomputed [vocab x 256] bf16 matrix per chained position
    return LAYERS_MB + SMALL_HEAD_MB + LMHEAD_MB + n * M_BF16_MB

def dr_llamacpp(n):  # llama.cpp: two lm_head passes (base + correction, Q8_0 201024 rows = 492 MB) per chained position
    return LAYERS_MB + SMALL_HEAD_MB + 2 * n * 492.0

def dr_q8(n):  # refined, drafter layers stored Q8_0 (523.8 x 17/32 = 278.3 MB) and the folded matrix Q8_0 (200058 x 256 x 1.0625 = 54.4 MB)
    return LAYERS_MB * 17 / 32 + SMALL_HEAD_MB + LMHEAD_MB + n * 200058 * 256 * 1.0625 / 1e6

def speedup(L, n, dr):
    T = n + 1
    S = N_MB + E[T] + dr(n)
    return L * B1 / S, S

runs = collections.defaultdict(dict)  # (mode, n) -> prompt -> dict
for f in sorted(glob.glob(os.path.join(RAW, "*__n*.steps"))):
    base = os.path.basename(f)[:-6]
    prompt, mode, nn = base.split("__")
    n = int(nn[1:])
    rows = [STEP.match(l) for l in open(f)]
    rows = [tuple(int(x) for x in m.groups()) for m in rows if m]
    logf = f[:-6] + ".log"
    txt = open(logf, errors="replace").read() if os.path.exists(logf) else ""
    dec = DEC.search(txt); enc = ENC.search(txt)
    lb = re.search(r"loadavg_before (\S+ \S+ \S+)", txt)
    done = "n_predict" in txt
    runs[(mode, n)][prompt] = dict(rows=rows, tps=float(dec.group(3)) if dec else None, ntok=int(dec.group(1)) if dec else None,
                                  nprompt=int(enc.group(1)) if enc else None, load=lb.group(1) if lb else "?", done=done)

def stats(rows, n):
    full = [r for r in rows if r[0] == 1 and r[1] == n]
    k = [r[2] for r in full]
    steps = len(k)
    if steps == 0:
        return None
    L = sum(x + 1 for x in k) / steps
    sd = math.sqrt(sum((x + 1 - L) ** 2 for x in k) / max(steps - 1, 1))
    pos = [sum(1 for x in k if x >= j) / steps for j in range(1, n + 1)]
    cond = [pos[0]] + [(pos[j] / pos[j - 1]) if pos[j - 1] > 0 else float("nan") for j in range(1, n)]
    tdraft = sum(r[3] for r in full) / steps / 1000
    ttgt = sum(r[4] for r in full) / steps / 1000
    tproc = sum(r[5] for r in full) / steps / 1000
    return dict(steps=steps, L=L, se=sd / math.sqrt(steps), pos=pos, cond=cond, excl=len(rows) - steps, tdraft=tdraft, ttgt=ttgt, tproc=tproc,
                hist=collections.Counter(k))

out = []
P = out.append
MODES = [("greedy", "greedy (temp 0), drafter argmax, target argmax match"),
         ("t1prob", "temp 1.0, drafter SAMPLES, target verifies by rejection sampling (`--spec-draft-sampling probabilistic`)"),
         ("t1match", "temp 1.0, drafter argmax, target samples and must match the draft")]
P("## Pooled acceptance (all prompts, full-length drafts only; L = mean tokens per verify step = accepted drafts + 1)\n")
P("| mode | draft n (T=n+1) | prompts done | steps | **L +- SE** | accepted drafts/step | acc@1 | acc@2 | acc@3 | acc@4 | acc@5 | acc@6 | acc@7 |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
pooled = {}
for mode, desc in MODES:
    for n in (3, 4, 7):
        d = runs.get((mode, n))
        if not d:
            continue
        allrows = [r for p in d.values() if p["done"] for r in p["rows"]]
        s = stats(allrows, n)
        if not s:
            continue
        pooled[(mode, n)] = s
        accs = " | ".join(("%.2f" % s["pos"][j]) if j < n else "-" for j in range(7))
        P("| %s | %d (T=%d) | %d/8 | %d | **%.2f +- %.2f** | %.2f | %s |" % (mode, n, n + 1, sum(1 for p in d.values() if p["done"]), s["steps"], s["L"], s["se"], s["L"] - 1, accs))
P("\nacc@j = fraction of verify steps in which draft position j was accepted (cumulative: positions 1..j all accepted).\n")
P("## Conditional acceptance P(position j accepted | positions < j accepted)\n")
P("| mode | n | c@1 | c@2 | c@3 | c@4 | c@5 | c@6 | c@7 |")
P("|---|---|---|---|---|---|---|---|---|")
for (mode, n), s in sorted(pooled.items()):
    P("| %s | %d | %s |" % (mode, n, " | ".join(("%.2f" % s["cond"][j]) if j < n else "-" for j in range(7))))
P("\n## Per prompt: L (steps) \n")
prompts = ["chat_answer", "code_completion", "prose_continuation", "reasoning", "tool_call", "long_context_qa", "chat_email", "chat_code"]
cfgs = sorted(runs.keys())
P("| prompt | " + " | ".join("%s n=%d" % c for c in cfgs) + " |")
P("|---|" + "---|" * len(cfgs))
for p in prompts:
    cells = []
    for c in cfgs:
        d = runs[c].get(p)
        s = stats(d["rows"], c[1]) if d and d["done"] else None
        cells.append("%.2f (%d)" % (s["L"], s["steps"]) if s else "-")
    P("| %s | %s |" % (p, " | ".join(cells)))
P("\n## Bytes-model speed-up estimates (ROUTING.md, mean expert-union bytes; decode bandwidth-bound; speed-up = L * B1 / S(T))\n")
P("S(T) = N + E(T) + Dr: N = 2014.6 MB, B1 = 3861.9 MB/token, E(T) from ROUTING.md (d). Dr variants: (R) ROUTING.md 635.8 MB drafter + 1 lm_head pass = 1125.5 MB; "
  "(M) refined = 523.8 layers + 9.1 head + 489.7 batched base lm_head pass + n x 102.4 MB (one pass over a precomputed bf16 [200058 x 256] matrix lm_head x hidden_out per chained position); "
  "(C) llama.cpp as implemented = 523.8 + 9.1 + 2 x n x 492 MB (two lm_head passes per position).\n")
P("(Q) = (M) with the drafter layers and the folded matrix stored Q8_0 (278.3 MB layers + 54.4 MB per chained position; Unsloth reports the same acceptance for its Q8_0 drafter).\n")
P("| mode | n | T | L measured | L* (R) | L* (M) | L* (Q) | speed-up (R) | speed-up (M) | speed-up (Q) | speed-up (C) | S(T) (M) MB |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for (mode, n), s in sorted(pooled.items()):
    sR, SR = speedup(s["L"], n, dr_routing); sM, SM = speedup(s["L"], n, dr_refined); sC, SC = speedup(s["L"], n, dr_llamacpp)
    sQ, SQ = speedup(s["L"], n, dr_q8)
    P("| %s | %d | %d | %.2f | %.2f | %.2f | %.2f | %.2fx | %.2fx | %.2fx | %.2fx | %.0f |" % (mode, n, n + 1, s["L"], SR / B1, SM / B1, SQ / B1, sR, sM, sQ, sC, SM))
P("\n## llama.cpp reference timing (this machine, 8 threads, shared host: indicative; tokens/s from the log's `decoded` line)\n")
P("| mode | n | mean ms per step: drafter | target verify (T tokens) | feature injection | tokens/s (mean over prompts) | loadavg before (min..max 1-min) |")
P("|---|---|---|---|---|---|---|")
for (mode, n), s in sorted(pooled.items()):
    d = runs[(mode, n)]
    tps = [p["tps"] for p in d.values() if p["done"] and p["tps"]]
    loads = [float(p["load"].split()[0]) for p in d.values() if p["done"] and p["load"] != "?"]
    P("| %s | %d | %.0f | %.0f | %.1f | %.2f | %.1f..%.1f |" % (mode, n, s["tdraft"], s["ttgt"], s["tproc"], sum(tps) / max(len(tps), 1), min(loads or [0]), max(loads or [0])))
P("\n## Histogram of accepted drafts per step (pooled; count of steps with k accepted)\n")
for (mode, n), s in sorted(pooled.items()):
    P("- %s n=%d: %s" % (mode, n, ", ".join("%d:%d" % (k, s["hist"][k]) for k in range(n + 1))))
txt = "\n".join(out)
open(os.path.join(RAW, "summary.md"), "w").write(txt + "\n")
print(txt)

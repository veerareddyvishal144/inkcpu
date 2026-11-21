#!/usr/bin/env python3
"""Fill docs/DSPARK.md placeholders (@@HEADLINE@@, @@RESULTS@@, @@SPEED@@) from tools/dspark/raw. Template copy: tools/dspark/DSPARK.template.md
(created on first run). @@IMPLICATION@@ is left for a manual edit. Run once at the end: /usr/bin/python3 -I assemble_dspark_md.py"""
import os, re, glob, runpy, io, contextlib, shutil
D = "/data/inkling-mini/engine/tools/dspark"
DOC = "/data/inkling-mini/engine/docs/DSPARK.md"
TPL = D + "/DSPARK.template.md"
if not os.path.exists(TPL):
    shutil.copy(DOC, TPL)
tpl = open(TPL).read()
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(D + "/analyze.py", run_name="analysis")
summary = re.sub(r"^## ", "#### ", open(D + "/raw/summary.md").read(), flags=re.M)
pooled, speedup = g["pooled"], g["speedup"]
dr_r, dr_m, dr_q, dr_c = g["dr_routing"], g["dr_refined"], g["dr_q8"], g["dr_llamacpp"]
B1, E, N_MB, STEP, DEC, ENC = g["B1"], g["E"], g["N_MB"], g["STEP"], g["DEC"], g["ENC"]
# prompt sizes
pr = ["| prompt | tokens in prompt (llama.cpp `encoded`) |", "|---|---|"]
for p in ["chat_answer", "code_completion", "prose_continuation", "reasoning", "tool_call", "long_context_qa", "chat_email", "chat_code"]:
    f = "%s/raw/%s__greedy__n3.log" % (D, p)
    m = ENC.search(open(f, errors="replace").read()) if os.path.exists(f) else None
    pr.append("| %s | %s |" % (p, m.group(1) if m else "-"))
# break-even table
be = ["#### Break-even table (bytes model of docs/ROUTING.md, no measurement)", "",
      "| n | T=n+1 | S(T) R MB | L* (R) | S(T) M MB | L* (M) | S(T) Q MB | L* (Q) | max speed-up at L=T (M) |", "|---|---|---|---|---|---|---|---|---|"]
for n in range(1, 8):
    T = n + 1
    sR, sM, sQ = (N_MB + E[T] + f(n) for f in (dr_r, dr_m, dr_q))
    be.append("| %d | %d | %.0f | %.2f | %.0f | %.2f | %.0f | %.2f | %.2fx |" % (n, T, sR, sR / B1, sM, sM / B1, sQ, sQ / B1, T * B1 / sM))
res = "\n".join(pr) + "\n\n" + summary + "\n\n" + "\n".join(be) + "\n"
# headline
hl = ["**Headline** ([M]: 8 prompts, <= 80 new tokens each, Q4_K_XL target + Q8_0 drafter, llama.cpp PR 32218; L = mean tokens per verify step incl. bonus; "
      "L* = break-even from docs/ROUTING.md bytes model with the refined drafter cost (M); speed-up = L x B1 / S(T)):", ""]
best = None
for (mode, n), s in sorted(pooled.items()):
    sM, SM = speedup(s["L"], n, dr_m); sQ, SQ = speedup(s["L"], n, dr_q); sR, SR = speedup(s["L"], n, dr_r)
    hl.append("- %s, n=%d (T=%d): L = **%.2f +- %.2f** (acc@1 %.2f), break-even L* = %.2f (M) / %.2f (R); estimated speed-up %.2fx (M), %.2fx (Q), %.2fx (R)" %
              (mode, n, n + 1, s["L"], s["se"], s["pos"][0], SM / B1, SR / B1, sM, sQ, sR))
    if best is None or sM > best[0]:
        best = (sM, mode, n)
if best:
    hl.append("- best bytes-model speed-up (M) among measured configs: %.2fx (%s, n=%d). Recommendation and caveats: section 10." % best)
else:
    hl.append("- no completed configuration yet.")
headline = "\n".join(hl)
# speed table
sp = ["| prompt | n (T=n+1) | tokens/s (80 new tokens) | speed vs n=0 | L | ms per step: drafter | ms per step: target verify | loadavg before / after |", "|---|---|---|---|---|---|---|---|"]
base = {}
rowsp = []
for f in sorted(glob.glob(D + "/raw/speed/*__lock__n*.log")):
    b = os.path.basename(f)[:-4]
    prompt, _, nn = b.split("__"); n = int(nn[1:])
    txt = open(f, errors="replace").read()
    dec = DEC.search(txt)
    rows = [tuple(int(x) for x in m.groups()) for m in map(STEP.match, open(f[:-4] + ".steps")) if m] if os.path.exists(f[:-4] + ".steps") else []
    full = [r for r in rows if r[0] == 1 and r[1] == n]
    L = sum(r[2] + 1 for r in full) / len(full) if full else float("nan")
    td = sum(r[3] for r in full) / len(full) / 1000 if full else float("nan")
    tt = sum(r[4] for r in full) / len(full) / 1000 if full else float("nan")
    lb = re.search(r"loadavg_before (\S+)", txt); la = re.search(r"loadavg_after (\S+)", txt)
    tps = float(dec.group(3)) if dec else float("nan")
    if n == 0:
        base[prompt] = tps
    rowsp.append((prompt, n, tps, L, td, tt, lb.group(1) if lb else "?", la.group(1) if la else "?"))
for prompt, n, tps, L, td, tt, lb, la in rowsp:
    ratio = tps / base[prompt] if prompt in base and base[prompt] == base[prompt] else float("nan")
    sp.append("| %s | %d (T=%d) | %.2f | %.2fx | %.2f | %.0f | %.0f | %s / %s |" % (prompt, n, n + 1, tps, ratio, L, td, tt, lb, la))
speed = ("Greedy, 8 threads, taken under `tools/with_bench_lock.sh` after the acceptance matrix finished (my own matrix runs were not under the lock; other agents' load is in the last column). "
         "n=0 is plain decoding through the same binary (no draft). `ms per step: target verify` is the time of the target forward over T tokens, `drafter` the time of the drafting call (the ctx injection is ~2 ms).\n\n"
         + "\n".join(sp) + "\n") if rowsp else "(speed runs not available)\n"
impl = open(D + "/implication.md").read() if os.path.exists(D + "/implication.md") else "(to be written)"
tok = {}
for (mode_, n_), s_ in pooled.items():
    k_ = "%s_%d" % (mode_, n_)
    tok["L_" + k_] = "%.2f" % s_["L"]; tok["SE_" + k_] = "%.2f" % s_["se"]; tok["A1_" + k_] = "%.2f" % s_["pos"][0]; tok["STEPS_" + k_] = str(s_["steps"])
    for nm_, f_ in (("SPM", dr_m), ("SPQ", dr_q), ("SPR", dr_r), ("SPC", dr_c)):
        tok["%s_%s" % (nm_, k_)] = "%.2f" % speedup(s_["L"], n_, f_)[0]
    tok["TD_" + k_] = "%.0f" % s_["tdraft"]; tok["TT_" + k_] = "%.0f" % s_["ttgt"]
for n_ in range(1, 8):
    for nm_, f_ in (("BEM", dr_m), ("BEQ", dr_q), ("BER", dr_r)):
        tok["%s_%d" % (nm_, n_)] = "%.2f" % ((N_MB + E[n_ + 1] + f_(n_)) / B1)
runs_, stats_ = g["runs"], g["stats"]
PR_ = ["chat_answer", "prose_continuation", "chat_email", "reasoning", "long_context_qa", "chat_code", "tool_call", "code_completion"]
tab = ["| prompt (text type) | L n=3 | L n=4 | L n=7 | speed-up (M) n=3 | speed-up (M) n=4 | speed-up (M) n=7 | best choice (M) |", "|---|---|---|---|---|---|---|---|"]
macro = {3: [], 4: [], 7: []}; oracle = []; oracle_q = []
for p_ in PR_:
    Ls = {}; sps = {}; spq = {}
    for n_ in (3, 4, 7):
        d_ = runs_.get(("greedy", n_), {}).get(p_)
        s_ = stats_(d_["rows"], n_) if d_ and d_["done"] else None
        if s_:
            Ls[n_] = s_["L"]; sps[n_] = speedup(s_["L"], n_, dr_m)[0]; spq[n_] = speedup(s_["L"], n_, dr_q)[0]; macro[n_].append(sps[n_])
    best_ = max([(1.0, "plain")] + [(v, "n=%d" % k) for k, v in sps.items()])
    oracle.append(best_[0]); oracle_q.append(max([1.0] + list(spq.values())))
    tab.append("| %s | %s | %s | %s | %s | %s | %s | %s (%.2fx) |" % ((p_,) + tuple("%.2f" % Ls[n] if n in Ls else "-" for n in (3, 4, 7)) + tuple("%.2fx" % sps[n] if n in sps else "-" for n in (3, 4, 7)) + (best_[1], best_[0])))
tok["REGIME_TABLE"] = "\n".join(tab)
for n_ in (3, 4, 7):
    tok["MACRO_N%d" % n_] = "%.2f" % (sum(macro[n_]) / max(len(macro[n_]), 1))
tok["MACRO_ORACLE_M"] = "%.2f" % (sum(oracle) / len(oracle)); tok["MACRO_ORACLE_Q"] = "%.2f" % (sum(oracle_q) / len(oracle_q))
impl = re.sub(r"\{\{(\w+)\}\}", lambda m_: tok.get(m_.group(1), "n/a"), impl)
out = tpl.replace("@@IMPLICATION@@", impl).replace("@@HEADLINE@@", headline).replace("@@RESULTS@@", res).replace("@@SPEED@@", speed)
open(DOC, "w").write(out)
open(D + "/raw/speed_summary.md", "w").write(speed)
print("assembled", len(out), "bytes; configs:", sorted(pooled.keys()))

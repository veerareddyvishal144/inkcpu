#!/usr/bin/env python
"""Summarise agent T's sweep logs (smt_sweep / t_stage2): bench_decode / bench_prefill result rows and the per-phase profile tables side by side.
usage: python tools/t_parse_sweep.py LOG [LOG...]"""
import re
import sys

for path in sys.argv[1:]:
    log = open(path).read()
    print("=== %s" % path)
    # bench_decode rows:   ctx threads tok/s ms/tok reps.. load/load split
    for m in re.finditer(r"^\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d. ]+?)\s+([\d.]+)/([\d.]+)\s+split_keys", log, re.M):
        print("decode  ctx %5s thr %2s  %6s tok/s  reps %-14s load %s->%s" % (m.group(1), m.group(2), m.group(3), m.group(5).strip(), m.group(6), m.group(7)))
    # bench_prefill rows:  threads prompt best_s tok/s reps load/load
    for m in re.finditer(r"^(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d. ]+?)\s+([\d.]+)/([\d.]+)\s*$", log, re.M):
        print("prefill thr %2s prompt %4s  %6s tok/s  reps %-14s load %s->%s" % (m.group(1), m.group(2), m.group(4), m.group(5).strip(), m.group(6), m.group(7)))
    # bench_stream
    for m in re.finditer(r"^threads\s+(\d+): free stream ([\d.]+) GB/s \| phase size -> GB/s.*?:(.*)$", log, re.M):
        print("stream  thr %2s  free %s GB/s  phases%s" % (m.group(1), m.group(2), m.group(3)))
    # profile tables (decode: "--- profile ctx C, T threads ---"; prefill: whatever header precedes a "phase ... ms" table)
    blocks = re.split(r"--- profile ctx (\d+), (\d+) threads ---\n", log)
    tables = {}
    for i in range(1, len(blocks), 3):
        ctx, th, body = int(blocks[i]), int(blocks[i + 1]), blocks[i + 2]
        rows = {}
        for line in body.splitlines():
            m = re.match(r"(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                rows[m.group(1).strip()] = (float(m.group(2)), float(m.group(6)))
            m2 = re.match(r"sum of phases\s+([\d.]+)", line)
            if m2:
                rows["sum of phases"] = (float(m2.group(1)), 0.0)
            if re.match(r"^\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+", line):
                break
        tables[(ctx, th)] = rows
    for ctx in sorted({c for c, _ in tables}):
        ths = sorted(t for c, t in tables if c == ctx)
        base = tables[(ctx, ths[0])]
        print("\n  decode profile ctx %d: ms/token (GB/s) at %s threads" % (ctx, ths))
        for ph in base:
            cells = []
            for t in ths:
                ms, gbs = tables[(ctx, t)].get(ph, (float("nan"), 0))
                cells.append("%8.2f (%3.0f)" % (ms, gbs) if gbs else "%8.2f      " % ms)
            print("  %-30s%s" % (ph[:30], "".join("%15s" % c for c in cells)))
    # generic prefill profile tables: print raw blocks that contain 'ms' tables after a bench_prefill header
    pf = [m.start() for m in re.finditer(r"^bench_prefill:", log, re.M)]
    if pf:
        seg = log[pf[0]:]
        tabs = re.findall(r"((?:^[^\n]*phase[^\n]*\n)(?:^[^\n]+\n){3,40})", seg, re.M)
        print("\n  prefill profile tables found: %d (printed raw below)" % len(tabs))
        for t in tabs:
            print(t.rstrip())
            print("  ---")

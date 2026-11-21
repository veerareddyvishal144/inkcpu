#!/bin/bash
# Reproducible baseline runs of the STOCK llama-bench (build-sys), one CSV row per invocation (agent E).
#   tools/bench_llama.sh [OUT.csv]            default OUT: tools/bench_baseline.csv (rows are appended)
# decode:  llama-bench -p 0 -n 64     prefill: llama-bench -p 256 -n 0     for each thread count in THREADS
# env: MODEL, LLAMA_DIR, BENCH, THREADS ("8 12 16"), REPS (3), MAX_LOAD (2), WAIT_MAX_S (600)
# Refuses to start a run while the 1-minute load average in /proc/loadavg exceeds MAX_LOAD: it waits up to WAIT_MAX_S seconds,
# then exits with status 3 and writes no row. Raising MAX_LOAD makes numbers "indicative only"; the CSV records the load anyway.
# CSV columns: date, load1_before, load1_after (1-min loadavg; the after value includes the benchmark's own threads),
#   other_cpus_busy (average number of CPUs kept busy by OTHER processes during the run = (/proc/stat busy time - benchmark CPU time)/wall),
#   llama_cpp_commit, test, threads, tokens_per_s, stddev, reps, command
set -uo pipefail
export PATH=/usr/bin:/bin
HERE=$(cd "$(dirname "$0")" && pwd)
LLAMA_DIR=${LLAMA_DIR:-/data/inkling-mini/llama.cpp}
BENCH=${BENCH:-$LLAMA_DIR/build-sys/bin/llama-bench}
MODEL=${MODEL:-/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf}
THREADS=${THREADS:-"8 12 16"}
REPS=${REPS:-3}
MAX_LOAD=${MAX_LOAD:-2}
WAIT_MAX_S=${WAIT_MAX_S:-600}
OUT=${1:-$HERE/bench_baseline.csv}
PY=${PY:-/data/engine/.venv/bin/python}
export OMP_WAIT_POLICY=active OMP_PLACES=cores OMP_PROC_BIND=close
HASH=$(git -C "$LLAMA_DIR" rev-parse --short=9 HEAD 2>/dev/null || echo unknown)

load1() { cut -d' ' -f1 /proc/loadavg; }
too_busy() { awk -v a="$(load1)" -v b="$MAX_LOAD" 'BEGIN { exit !(a > b) }'; }
wait_for_idle() {
  local waited=0
  while too_busy; do
    if [ "$waited" -ge "$WAIT_MAX_S" ]; then
      echo "bench_llama: refusing to start: 1-min load $(load1) > $MAX_LOAD after waiting ${WAIT_MAX_S}s" >&2
      exit 3
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

read -r -d '' ROW_PY <<'EOF'
import csv, datetime, io, os, subprocess, sys, time
out, commit, test, threads, reps, cmdline = sys.argv[1:7]
def load1(): return open('/proc/loadavg').read().split()[0]
def busy():
    v = list(map(int, open('/proc/stat').readline().split()[1:9]))   # user nice system idle iowait irq softirq steal
    return sum(v) - v[3] - v[4]
hz = os.sysconf('SC_CLK_TCK')
lb, b0, t0, c0 = load1(), busy(), time.time(), os.times()
p = subprocess.run(cmdline, shell=True, capture_output=True, text=True)
c1, t1, b1, la = os.times(), time.time(), busy(), load1()
mine = (c1.children_user - c0.children_user) + (c1.children_system - c0.children_system)
other = max(0.0, ((b1 - b0) / hz - mine) / max(t1 - t0, 1e-9))
rows = list(csv.DictReader(io.StringIO(p.stdout)))
if p.returncode != 0 or not rows:
    sys.stderr.write(p.stderr[-2000:] + '\nllama-bench failed (rc %d)\n' % p.returncode)
    sys.exit(1)
r = rows[-1]
date = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
new = not os.path.exists(out)
with open(out, 'a', newline='') as f:
    w = csv.writer(f)
    if new:
        w.writerow(['date', 'load1_before', 'load1_after', 'other_cpus_busy', 'llama_cpp_commit', 'test', 'threads', 'tokens_per_s', 'stddev', 'reps', 'command'])
    w.writerow([date, lb, la, '%.1f' % other, commit, test, r['n_threads'], r['avg_ts'], r['stddev_ts'], reps, cmdline])
print('%s threads=%s: %.2f +- %.2f tok/s (load1 %s -> %s, other CPUs busy %.1f, %.0f s)' % (test, r['n_threads'], float(r['avg_ts']), float(r['stddev_ts']), lb, la, other, t1 - t0), file=sys.stderr)
EOF

for t in $THREADS; do
  for test in decode prefill; do
    if [ "$test" = decode ]; then args="-p 0 -n 64"; else args="-p 256 -n 0"; fi
    wait_for_idle
    cmd="OMP_WAIT_POLICY=active OMP_PLACES=cores OMP_PROC_BIND=close $BENCH -m $MODEL $args -t $t -r $REPS -o csv"
    "$PY" -c "$ROW_PY" "$OUT" "$HASH" "$test" "$t" "$REPS" "$cmd" || exit 1
  done
done
echo "bench_llama: rows appended to $OUT" >&2

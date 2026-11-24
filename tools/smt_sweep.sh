#!/bin/bash
# SMT sweep (agent T, wave 1): decode / prefill / stream with 16, 20, 24, 32 threads where threads > 16 add SMT siblings.
# CPU order: the 16 physical cores as the engine's default (2,4,..,30 then the half-speed core 0), then the odd siblings of
# cores 1..8 (20 and 24 threads), then the remaining siblings with core 0's sibling (CPU 1) last (32 threads).
# Every full-model run takes the shared lock (tools/with_bench_lock.sh) and checks loadavg < 1.5, production active 0, >= 40 GB available.
# usage: tools/smt_sweep.sh [build dir] [out log]      (defaults: build-t, <build>/smt_sweep_<UTC>.log)
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
B=${1:-build-t}; OUT=${2:-$B/smt_sweep_$(date -u +%H%M%S).log}
export INK_CPUS=${INK_CPUS:-2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,1}
THREADS=${THREADS:-16,20,24,32}
precheck() {   # instantaneous idleness (two /proc/stat reads 2 s apart < 10 % busy), production active 0, >= 40 GB; prints the loadavg for the record
  local a b busy act avail l1
  a=$(awk '/^cpu /{print $2+$3+$4+$7+$8+$9, $5+$6}' /proc/stat); sleep 2; b=$(awk '/^cpu /{print $2+$3+$4+$7+$8+$9, $5+$6}' /proc/stat)
  busy=$(echo "$a $b" | awk '{t=($3+$4)-($1+$2); if (t<=0) t=1; printf "%d", 100*(($3-$1))/t}')
  l1=$(cut -d' ' -f1 /proc/loadavg); act=$(curl -s 127.0.0.1:8000/health | sed -n 's/.*"active":\([0-9]*\).*/\1/p'); avail=$(free -g | awk '/Mem:/{print $7}')
  echo "[$(date -u +%H:%M:%S)] precheck cpu busy ${busy}% loadavg $l1 active ${act:-?} avail ${avail} GB"
  [ "$busy" -lt 10 ] || { echo "host busy: waiting 30 s"; sleep 30; precheck; return; }
  [ "${act:-0}" = "0" ] || { echo "production active $act: waiting 30 s"; sleep 30; precheck; return; }
  [ "$avail" -ge 40 ] || { echo "only $avail GB available"; exit 3; }
}
run() { echo; echo "### $(date -u +%H:%M:%S) $*"; echo "busy threads (>1% cpu) before: $(ps -eLo psr,pcpu,comm --sort=-pcpu | awk 'NR>1 && $2>1.0 {printf "cpu%s:%s(%s) ", $1, $3, $2}')"; tools/with_bench_lock.sh "$@"; echo "[$(date -u +%H:%M:%S)] loadavg after: $(cut -d' ' -f1-3 /proc/loadavg)"; }
{
  echo "smt_sweep $(date -u) build $B INK_CPUS=$INK_CPUS threads $THREADS"
  precheck; run env INK_PROFILE=1 "$B/src/engine/bench_decode" --threads "$THREADS" --ctx 16,1000 --n 64 --reps 2
  precheck; run env INK_PROFILE=1 "$B/src/engine/bench_prefill" --threads "$THREADS" --prompts 256,512 --reps 2
  precheck; run "$B/src/engine/bench_stream" "${STREAM_THREADS:-16,24,32}" 3.0
  echo "done $(date -u)"
} 2>&1 | tee "$OUT"

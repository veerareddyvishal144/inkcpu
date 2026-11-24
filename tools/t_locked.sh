#!/bin/bash
# Agent T, wave 1: lock-first timing sessions (each < 10 min). The lock is taken FIRST, then the session waits (inside the lock, <= 4 min) for
# the previous holder's load to decay below 1.5 with production idle and >= 40 GB available, then runs its batch. Two sessions, lock released between.
#   A: bench_prefill + bench_stream SMT cells, old-pool vs futex-park bench_decode (16 thr, ctx 16), 16-thread pool A/B, wake/idle, test_threadpool
#   B: test-server per-token overhead (2 thr numpy, 16 thr numpy, 16 thr C sampler, 2 thr C sampler) via tools/step_overhead.sh's inner function
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
B=build-t; OUT=$B/t_locked_$(date -u +%H%M%S).log
SMT=2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,1
EVEN=2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,0
quiet_inside() {   # instantaneous idleness (orchestrator 18:53): two /proc/stat reads 2 s apart, < 10 % of all CPUs busy; production idle; >= 40 GB
  local l1 act av busy a b
  for i in $(seq 1 40); do
    a=$(awk '/^cpu /{print $2+$3+$4+$7+$8+$9, $5+$6}' /proc/stat); sleep 2; b=$(awk '/^cpu /{print $2+$3+$4+$7+$8+$9, $5+$6}' /proc/stat)
    busy=$(echo "$a $b" | awk '{t=($3+$4)-($1+$2); if (t<=0) t=1; printf "%d", 100*(($3-$1))/t}')
    l1=$(cut -d' ' -f1 /proc/loadavg); act=$(curl -s 127.0.0.1:8000/health | sed -n 's/.*"active":\([0-9]*\).*/\1/p'); av=$(free -g | awk '/Mem:/{print $7}')
    [ "$busy" -lt 10 ] && [ "${act:-0}" = "0" ] && [ "$av" -ge 40 ] && { echo "[$(date -u +%H:%M:%S)] quiet inside the lock: cpu busy ${busy}% over 2 s, loadavg $l1 (1-min average still decaying), active $act, avail $av GB"; return 0; }
    echo "[$(date -u +%H:%M:%S)] lock held, host busy ${busy}% / active ${act:-?} / avail $av GB, loadavg $l1"; sleep 3
  done; echo "WARNING: host not idle after ~3 min, running anyway (numbers indicative)"; }
sessionA() {
  quiet_inside
  echo "### $(date -u +%H:%M:%S) A1 bench_prefill SMT list, threads 16,20,24,32, prompts 256,512 (loadavg $(cut -d' ' -f1 /proc/loadavg))"
  INK_CPUS=$SMT INK_PROFILE=1 $B/src/engine/bench_prefill --threads 16,20,24,32 --prompts 256,512 --reps 2
  echo "### $(date -u +%H:%M:%S) A2 bench_stream SMT list 16,24,32 (loadavg $(cut -d' ' -f1 /proc/loadavg))"
  INK_CPUS=$SMT $B/src/engine/bench_stream 16,24,32 3.0
  for r in 1 2; do
    echo "### $(date -u +%H:%M:%S) A3 OLD pool bench_decode 16 thr ctx 16 (loadavg $(cut -d' ' -f1 /proc/loadavg))"; $B/bench_decode_oldpool --threads 16 --ctx 16 --n 64 --reps 2 | grep -E '^ +16 '
    echo "### $(date -u +%H:%M:%S) A3 NEW pool bench_decode 16 thr ctx 16 (loadavg $(cut -d' ' -f1 /proc/loadavg))"; $B/src/engine/bench_decode --threads 16 --ctx 16 --n 64 --reps 2 | grep -E '^ +16 '
  done
  echo "### $(date -u +%H:%M:%S) A4 pool hot path, 16 threads on $EVEN: old vs new, 3 alternations"
  for r in 1 2 3; do echo -n "old: "; $B/oldpool/bench_pool_oldhot --cpus $EVEN 16 | tail -1; echo -n "new: "; $B/src/kernels_gemv/bench_pool --cpus $EVEN 16 | tail -1; done
  echo "### $(date -u +%H:%M:%S) A5 first run() after idle, 16 threads: legacy polling vs park"
  INK_POOL_PARK=0 $B/src/kernels_gemv/bench_pool --cpus $EVEN --wake 300 16 | tail -1; $B/src/kernels_gemv/bench_pool --cpus $EVEN --wake 300 16 | tail -1
  echo "### $(date -u +%H:%M:%S) A6 idle CPU 10 s, 16 threads: legacy vs park"
  INK_POOL_PARK=0 $B/src/kernels_gemv/bench_pool --cpus $EVEN --idle 10 16 | tail -1; $B/src/kernels_gemv/bench_pool --cpus $EVEN --idle 10 16 | tail -1
  echo "### $(date -u +%H:%M:%S) A7 test_threadpool on real cores (default pinning), park + legacy"
  $B/src/kernels_gemv/test_threadpool | tail -4; INK_POOL_PARK=0 $B/src/kernels_gemv/test_threadpool | tail -1
  echo "[$(date -u +%H:%M:%S)] session A done, loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
}
sessionB() {
  quiet_inside
  . <(sed -n '/^inner() {/,/^}/p' tools/step_overhead.sh)       # reuse the measurement function (needs INK_* env + PY + NT)
  PY=/data/engine/.venv/bin/python
  run_server() { # cpus sampler ntok
    local cpus=$1; NT=$(echo "$cpus" | tr ',' '\n' | wc -l)
    export INK_PORT=8010 INK_CPUS=$cpus INK_THREADS=$NT INK_LIB=$ROOT/$B/src/capi/libink.so INK_SPEC=off INK_STEP_TRACE=1 INK_CTX=4096 INK_SLOTS=1 INK_NO_DOWNLOAD=1 INK_SAMPLER=$2 NTOK=$3 PORT=8010 PY NT
    echo "### $(date -u +%H:%M:%S) B test server: $NT threads on $cpus, sampler $2, $3 tokens (loadavg $(cut -d' ' -f1 /proc/loadavg))"
    inner
  }
  run_server 29,31 numpy 48
  run_server $EVEN numpy 96
  run_server $EVEN c 96
  run_server 29,31 c 48
  echo "[$(date -u +%H:%M:%S)] session B done, loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
}
export -f quiet_inside sessionA sessionB; export B SMT EVEN ROOT
WHICH=${1:-AB}
{
  echo "t_locked start $(date -u) sessions $WHICH loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  case "$WHICH" in *A*) tools/with_bench_lock.sh bash -c sessionA; echo "--- lock released after A ---";; esac
  case "$WHICH" in *B*) tools/with_bench_lock.sh bash -c sessionB;; esac
  echo "done $(date -u)"
} 2>&1 | tee "$OUT"

#!/bin/bash
# Final idle-host A/B for optimisation wave 1: old build vs new build in the same window.
#   tools/with_bench_lock.sh tools/final_bench_wave1.sh <old_build_dir> <new_build_dir> <out_dir> [quick]
# Requires: nothing else of ours running (checked: instantaneous CPU idle via /proc/stat), production server idle.
set -u
OLD=$1; NEW=$2; OUT=$3; QUICK=${4:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
mkdir -p "$OUT"
export INK_PREFILL_MIN=${INK_PREFILL_MIN:-0}
idle_pct() { read -r _ a b c d rest < /proc/stat; sleep 2; read -r _ a2 b2 c2 d2 rest < /proc/stat
  local busy=$(( (a2-a)+(b2-b)+(c2-c) )) idle=$(( d2-d )); echo $(( 100*idle/(busy+idle) )); }
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do p=$(idle_pct); [ "$p" -ge 90 ] && break; echo "host busy (idle $p%), waiting"; sleep 10; done
act=$(curl -s 127.0.0.1:8000/health | grep -o '"active":[0-9]*' | cut -d: -f2)
echo "start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg) idle ${p}% server_active ${act:-?}" | tee "$OUT/00_conditions.txt"
run() { local name=$1; shift; echo "== $name  ($(date -u +%T), loadavg $(cut -d' ' -f1 /proc/loadavg))" | tee -a "$OUT/00_log.txt"; "$@" > "$OUT/$name.txt" 2>&1; echo "   rc=$? $(grep -E 'tok/s' "$OUT/$name.txt" | tail -3 | tr '\n' ';')" | tee -a "$OUT/00_log.txt"; }
REPS=3; [ -n "$QUICK" ] && REPS=2
# decode, short and 1k context (both builds, alternating so the window is shared)
for rep in A B; do
  run "decode_old_ctx16_1000_$rep" env INK_PROFILE=1 "$OLD/src/engine/bench_decode" --threads 16 --ctx 16,1000 --n 64 --reps $REPS
  run "decode_new_ctx16_1000_$rep" env INK_PROFILE=1 "$NEW/src/engine/bench_decode" --threads 16 --ctx 16,1000 --n 64 --reps $REPS
done
run "decode_new_12thr"             env INK_PROFILE=1 "$NEW/src/engine/bench_decode" --threads 12 --ctx 16 --n 64 --reps $REPS
# long context (new build only has the prefill-built context option; old build at 8000 would take 10 min by decode)
run "decode_new_ctx8000"           env INK_PROFILE=1 "$NEW/src/engine/bench_decode" --threads 16 --ctx 8000 --n 48 --reps 2 --prefill 1
[ -z "$QUICK" ] && run "decode_new_ctx16000" env INK_PROFILE=1 "$NEW/src/engine/bench_decode" --threads 16 --ctx 16000 --n 48 --reps 2 --prefill 1
# prefill
run "prefill_old"                  "$OLD/src/engine/bench_prefill" --threads 16 --prompts 256,512,1024 --reps 2
run "prefill_new"                  "$NEW/src/engine/bench_prefill" --threads 16 --prompts 256,512,1024 --reps 2
run "prefill_new_chunk512"         "$NEW/src/engine/bench_prefill" --threads 16 --prompts 1024 --chunk 512 --reps 2
# speculation
run "spec_new"                     env INK_THREADS=12 "$NEW/src/spec/bench_spec" --modes plain,4,8,auto --n 120
[ -z "$QUICK" ] && run "spec_old"  env INK_THREADS=12 "$OLD/src/spec/bench_spec" --modes plain,4,auto --n 120
echo "end $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/00_conditions.txt"

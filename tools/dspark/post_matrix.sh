#!/bin/bash
# Waits for the matrix (or the deadline, then stops the runner after its current run), then takes lock-protected speed runs and assembles docs/DSPARK.md.
cd /data/inkling-mini/engine/tools/dspark || exit 1
DEADLINE=${DEADLINE:-22:31:00}
mkdir -p raw/speed
while [ ! -f raw/MATRIX_DONE ]; do
  if [[ "$(date -u +%T)" > "$DEADLINE" ]]; then
    P=$(pgrep -x run_matrix.sh | head -1)
    if [ -n "$P" ] && [ "$(readlink /proc/$P/cwd)" = "/data/inkling-mini/engine/tools/dspark" ]; then
      C=$(pgrep -P "$P" | head -1)
      kill "$P"
      while [ -n "$C" ] && kill -0 "$C" 2>/dev/null; do sleep 2; done
    fi
    break
  fi
  sleep 10
done
echo "matrix_end $(date -u +%T)" > raw/MATRIX_STOPPED
BIN=/data/inkling-mini/llama-unsloth/build/bin/llama-speculative-simple
TGT=/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf
DRF=/data/inkling-mini/models/dspark-Inkling-2-Mini-Q8_0.gguf
for p in chat_answer code_completion; do
  for n in 0 3 7; do
    cfg=raw/speed/${p}__lock__n${n}
    echo "loadavg_before $(cut -d' ' -f1 /proc/loadavg) start $(date -u +%T)" > $cfg.log
    DSPARK_STEPLOG=$cfg.steps /data/inkling-mini/engine/tools/with_bench_lock.sh nice -n 15 $BIN -m $TGT -md $DRF --spec-type draft-dspark \
      --spec-draft-n-max $n --spec-draft-sampling greedy -t 8 -td 8 -c 2048 -n 80 --temp 0 --top-k 1 --seed 1234 -f prompts/$p.txt >> $cfg.log 2>&1
    echo "loadavg_after $(cut -d' ' -f1 /proc/loadavg) end $(date -u +%T)" >> $cfg.log
  done
done
date -u +%T > raw/SPEED_DONE
/usr/bin/python3 -I assemble_dspark_md.py > raw/assemble.out 2>&1
date -u +%T > raw/ALL_DONE

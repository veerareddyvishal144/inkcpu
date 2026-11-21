#!/bin/bash
# DSpark acceptance matrix on the Inkling 2 Mini Q4_K_XL target with Unsloth's Q8_0 drafter (llama.cpp + ggml-org PR 32218, instrumented speculative-simple).
# Acceptance numbers do not depend on speed, so these runs do NOT take the bench lock; tokens/s in the logs are indicative only (loadavg recorded per run).
# Usage: run_matrix.sh   (writes raw/<prompt>__<mode>__n<N>.{log,steps} and raw/runs.csv, then raw/MATRIX_DONE)
cd /data/inkling-mini/engine/tools/dspark || exit 1
BIN=/data/inkling-mini/llama-unsloth/build/bin/llama-speculative-simple
TGT=/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf
DRF=/data/inkling-mini/models/dspark-Inkling-2-Mini-Q8_0.gguf
NPRED=${NPRED:-80}
PROMPTS="chat_answer code_completion prose_continuation reasoning tool_call long_context_qa chat_email chat_code"
mkdir -p raw
rm -f raw/MATRIX_DONE
# mode: greedy = temp 0; t1prob = temp 1.0, drafter samples, target verifies by rejection sampling; t1match = temp 1.0, drafter argmax, target samples and must match
run_cfg () {  # mode n
  local mode=$1 n=$2 samp spec
  case $mode in
    greedy) samp="--temp 0 --top-k 1"; spec="--spec-draft-sampling greedy";;
    t1prob) samp="--temp 1.0 --top-k 0 --top-p 1.0 --min-p 0.0"; spec="--spec-draft-sampling probabilistic";;
    t1match) samp="--temp 1.0 --top-k 0 --top-p 1.0 --min-p 0.0"; spec="--spec-draft-sampling greedy";;
  esac
  for p in $PROMPTS; do
    cfg=${p}__${mode}__n${n}
    if [ -s raw/$cfg.steps ] && grep -q 'n_predict' raw/$cfg.log 2>/dev/null; then continue; fi
    echo "loadavg_before $(cut -d' ' -f1-3 /proc/loadavg) start $(date -u +%T)" > raw/$cfg.log
    DSPARK_STEPLOG=raw/$cfg.steps nice -n 15 $BIN -m $TGT -md $DRF --spec-type draft-dspark --spec-draft-n-max $n $spec \
        -t 8 -td 8 -c 2048 -n $NPRED $samp --seed 1234 -f prompts/$p.txt >> raw/$cfg.log 2>&1
    echo "loadavg_after $(cut -d' ' -f1-3 /proc/loadavg) end $(date -u +%T) exit=$?" >> raw/$cfg.log
    echo "$cfg,$(date -u +%T),$(cut -d' ' -f1-3 /proc/loadavg)" >> raw/runs.csv
  done
}
for cfg in "greedy 3" "greedy 7" "greedy 4" "t1prob 3" "t1prob 7" "t1prob 4" "t1match 3" "t1match 7"; do
  run_cfg $cfg
done
date -u +%T > raw/MATRIX_DONE

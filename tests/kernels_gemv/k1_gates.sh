#!/bin/bash
# K1 correctness gates (full model, run under tools/with_bench_lock.sh): oracle / greedy in both INK_NVFP4R modes, determinism with the repack on,
# decode-path logits dump in both modes + cmp. Usage: tests/kernels_gemv/k1_gates.sh <build dir> <out dir>
set -u
B=${1:-build-k1}; OUT=${2:-$B/k1_gates}
mkdir -p "$OUT"
export INK_THREADS=${INK_THREADS:-16} INK_PREFILL_MIN=0
PY=/data/engine/.venv/bin/python
echo "== $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg) threads $INK_THREADS"
for m in 0 1; do
  echo "== oracle INK_NVFP4R=$m ($(date -u +%H:%M:%S), loadavg $(cut -d' ' -f1 /proc/loadavg))"
  INK_NVFP4R=$m $B/src/engine/test_engine oracle 2>&1 | grep -v '^$' | tee "$OUT/oracle_$m.txt" | tail -8
done
for m in 0 1; do
  echo "== greedy INK_NVFP4R=$m ($(date -u +%H:%M:%S), loadavg $(cut -d' ' -f1 /proc/loadavg))"
  INK_NVFP4R=$m $B/src/engine/test_engine greedy 2>&1 | tee "$OUT/greedy_$m.txt" | tail -8
done
echo "== determinism INK_NVFP4R=1 ($(date -u +%H:%M:%S), loadavg $(cut -d' ' -f1 /proc/loadavg))"
INK_NVFP4R=1 $B/src/engine/test_engine determinism 2>&1 | tee "$OUT/determinism_1.txt" | tail -8
for m in 0 1; do
  echo "== logits dump INK_NVFP4R=$m ($(date -u +%H:%M:%S), loadavg $(cut -d' ' -f1 /proc/loadavg))"
  INK_NVFP4R=$m $PY tests/kernels_gemv/dump_logits_decode.py $B/src/capi/libink.so "$OUT/logits_$m" 2>&1 | tee "$OUT/dump_$m.txt" | tail -9
done
echo "== cmp of the decode-path logits, INK_NVFP4R=0 vs 1"
fail=0
for f in "$OUT"/logits_0/*.f32 "$OUT"/logits_0/*.i32; do
  n=$(basename "$f")
  if cmp -s "$f" "$OUT/logits_1/$n"; then echo "  identical: $n ($(stat -c %s "$f") bytes)"; else echo "  DIFFERENT: $n"; fail=1; fi
done
echo "== oracle tables identical: $(cmp -s "$OUT/oracle_0.txt" "$OUT/oracle_1.txt" && echo yes || echo NO)"
echo "== greedy outputs identical: $(cmp -s "$OUT/greedy_0.txt" "$OUT/greedy_1.txt" && echo yes || echo NO)"
echo "== done $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg) cmp_fail=$fail"

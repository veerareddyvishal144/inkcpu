#!/bin/bash
# K1 timing session (run under tools/with_bench_lock.sh, idle host: loadavg < 1.5, production active 0). Same window A/B:
#   bench_decode INK_PROFILE=1, 16 threads, ctx 16, 64 tokens x 2 reps, INK_NVFP4R=0 then 1 (then 0 again to bracket the drift)
#   bench_gemv 16 threads, default CPU list, nvfp4 vs nvfp4r gate_up / down shapes with the stream ceiling
# Usage: tests/kernels_gemv/k1_timing.sh <build dir> <out dir>
set -u
B=${1:-build-k1}; OUT=${2:-$B/k1_timing}
mkdir -p "$OUT"
export INK_PREFILL_MIN=0
echo "== start $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
for m in 0 1 0b; do
  mm=${m%b}
  echo "== bench_decode INK_NVFP4R=$mm ($(date -u +%H:%M:%S), loadavg $(cut -d' ' -f1-3 /proc/loadavg))"
  # --prefill 0: the context is built through decode() (the batched prefill GEMM still expects GGUF-encoded scales); --dump: logits of the 64 timed tokens (rep 1)
  INK_NVFP4R=$mm INK_PROFILE=1 $B/src/engine/bench_decode --threads 16 --ctx 16 --n 64 --reps 2 --prefill 0 --dump "$OUT/dump_$m" 2>&1 | tee "$OUT/bench_decode_$m.txt" | grep -E 'tok/s|experts|sum of phases|proj|lm_head|load' | head -14
done
echo "== cmp of the 64 timed-token logits (ctx 16, 16 threads): INK_NVFP4R=0 vs 1: $(cmp -s "$OUT"/dump_0_ctx16_t16.f32 "$OUT"/dump_1_ctx16_t16.f32 && echo IDENTICAL || echo DIFFERENT) ($(stat -c %s "$OUT"/dump_0_ctx16_t16.f32 2>/dev/null) bytes)"
# integration check for the orchestrator: context built by the BATCHED prefill (K2/K3 nvfp4r paths in engine_prefill.cpp) with INK_NVFP4R=1, then the same 64 decode tokens
echo "== bench_decode INK_NVFP4R=1 --prefill 1 (K2/K3 nvfp4r prefill path) ($(date -u +%H:%M:%S), loadavg $(cut -d' ' -f1-3 /proc/loadavg))"
INK_NVFP4R=1 INK_PREFILL_MIN=3 $B/src/engine/bench_decode --threads 16 --ctx 16 --n 64 --reps 1 --prefill 1 --dump "$OUT/dump_1p" 2>&1 | tee "$OUT/bench_decode_1p.txt" | grep -E 'tok/s|WARNING|prefill' | head -6
echo "== cmp prefill-built context (INK_NVFP4R=1, --prefill 1) vs decode-built (INK_NVFP4R=0): $(cmp -s "$OUT"/dump_0_ctx16_t16.f32 "$OUT"/dump_1p_ctx16_t16.f32 && echo IDENTICAL || echo DIFFERENT)"
echo "== for reference: prefill-built context with INK_NVFP4R=0 vs decode-built INK_NVFP4R=0 (batched prefill is documented bit-identical below 768 keys)"
INK_NVFP4R=0 INK_PREFILL_MIN=3 $B/src/engine/bench_decode --threads 16 --ctx 16 --n 64 --reps 1 --prefill 1 --dump "$OUT/dump_0p" 2>&1 | grep -E 'tok/s' | head -2
echo "== cmp prefill-built (INK_NVFP4R=0) vs decode-built (INK_NVFP4R=0): $(cmp -s "$OUT"/dump_0_ctx16_t16.f32 "$OUT"/dump_0p_ctx16_t16.f32 && echo IDENTICAL || echo DIFFERENT)"
echo "== experiment: INK_NVFP4R=1 INK_CHUNK_DOWN=16 (E1's tunable; finer phase-H chunks) ($(date -u +%H:%M:%S), loadavg $(cut -d' ' -f1-3 /proc/loadavg))"
INK_NVFP4R=1 INK_CHUNK_DOWN=16 INK_PROFILE=1 $B/src/engine/bench_decode --threads 16 --ctx 16 --n 64 --reps 2 --prefill 0 2>&1 | tee "$OUT/bench_decode_1_cd16.txt" | grep -E '^ +16 +16 |experts|sum of phases' | head -6
echo "== bench_gemv 16 threads ($(date -u +%H:%M:%S), loadavg $(cut -d' ' -f1-3 /proc/loadavg))"
$B/src/kernels_gemv/bench_gemv --threads 16 --shapes "8x" --reps 3 --time 0.25 2>&1 | tee "$OUT/bench_gemv.txt" | grep -v '^#'
echo "== bench_gemv 16 threads, --chunk 32 (the engine's expert call size) ($(date -u +%H:%M:%S), loadavg $(cut -d' ' -f1-3 /proc/loadavg))"
$B/src/kernels_gemv/bench_gemv --threads 16 --shapes "8x" --reps 3 --time 0.25 --chunk 32 2>&1 | tee "$OUT/bench_gemv_chunk32.txt" | grep -v '^#'
echo "== done $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"

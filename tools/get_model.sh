#!/bin/bash
# Fetch the model files the engine needs (skips files already present; resumes partial downloads).
#   tools/get_model.sh [--check]      --check: only print what would be downloaded (remote sizes), no transfer
# Files (Hugging Face, Unsloth's GGUF repo):
#   Inkling-2-Mini-UD-Q4_K_XL.gguf   ~17.6 GB  target weights (the only quant the engine's kernels are built for). Upstream re-uploaded it
#                                     2026-11-22 in llama.cpp's canonical `inkling2` layout (48-visit block_count, loop_res_query); the engine
#                                     loads both that and the interim `inkling2_loop` layout (docs/ENGINE.md "GGUF layouts")
#   dspark-Inkling-2-Mini-Q8_0.gguf   402 MB  DSpark drafter (speculative decoding; optional: without it run serve.sh with INK_SPEC=off)
# Optional: HF_TOKEN for authenticated (faster, rate-limit-free) downloads; INK_MODEL_DIR (default /data/inkling-mini/models).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR=${INK_MODEL_DIR:-$( [ -d /data/inkling-mini/models ] && echo /data/inkling-mini/models || echo "$ROOT/models" )}
REPO=https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main
FILES="Inkling-2-Mini-UD-Q4_K_XL.gguf dspark-Inkling-2-Mini-Q8_0.gguf"
CHECK=${1:-}
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
mkdir -p "$DIR"
for f in $FILES; do
  url=$REPO/$f; dst=$DIR/$f
  remote=$(curl -sIL "${AUTH[@]}" "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  if [ -s "$dst" ] && head -c 4 "$dst" | grep -q '^GGUF'; then
    echo "present: $dst ($(stat -c %s "$dst") bytes; upstream now $remote)"; continue
  fi
  echo "need:    $f  ($remote bytes from $url)"
  [ "$CHECK" = "--check" ] && continue
  curl -L -C - --fail --retry 5 --retry-delay 10 "${AUTH[@]}" -o "$dst.part" "$url"
  head -c 4 "$dst.part" | grep -q '^GGUF' || { echo "ERROR: $f is not a GGUF file (bad download or auth page)"; exit 1; }
  mv "$dst.part" "$dst"; echo "done:    $dst ($(stat -c %s "$dst") bytes)"
done

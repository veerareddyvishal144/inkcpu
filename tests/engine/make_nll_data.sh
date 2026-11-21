#!/bin/bash
# Builds the llama.cpp-side data for gates (c) and (e) with tools/build/oracle_logits (stock llama.cpp, whole prompt = one batch).
#   nll1/ nll2/ : exactly 256 wikitext-2 test tokens each, logits_all.f32 (205 MB each)   -> test_engine nll build-c1b/data/nll1 build-c1b/data/nll2
#   ring2/      : > 2100 tokens (wraps the 2048-position SWA ring), logits_last.f32 only
#   ring/       : > 1030 tokens of wikitext-2 test text, logits_last.f32 only             -> test_engine slots build-c1b/data/ring
# Usage: tests/engine/make_nll_data.sh OUTDIR   (run with nice -n 10; ~3-4 min)
set -e
OUT=${1:-/data/inkling-mini/engine/build-c1b/data}
BIN=/data/inkling-mini/engine/tools/build/oracle_logits
GGUF=/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf
RAW=/data/inkling-mini/wiki/wikitext-2-raw/wiki.test.raw
mkdir -p "$OUT"
chunk() {  # name offset nchars
  local name=$1 off=$2 n=$3
  local text; text=$(tail -c +$((off + 1)) "$RAW" | head -c "$n")
  "$BIN" -m "$GGUF" --prompt "$text" --threads 8 --out "$OUT/${name}_tok" > "$OUT/${name}_tok.log" 2>&1
  /data/engine/.venv/bin/python - "$OUT/${name}_tok/tokens.i32" "$OUT/${name}.txt" <<'PY'
import sys, numpy as np
t = np.fromfile(sys.argv[1], '<i4')
print("tokenised", len(t), "tokens", file=sys.stderr)
open(sys.argv[2], 'w').write(' '.join(str(int(x)) for x in t[:256]))
PY
  "$BIN" -m "$GGUF" --tokens-file "$OUT/${name}.txt" --all-logits --threads 8 --out "$OUT/$name" > "$OUT/${name}.log" 2>&1
}
chunk nll1 0 1500
chunk nll2 600000 1500
text=$(tail -c +300001 "$RAW" | head -c 5600)
"$BIN" -m "$GGUF" --prompt "$text" --threads 8 --out "$OUT/ring" > "$OUT/ring.log" 2>&1
/data/engine/.venv/bin/python -c "import numpy as np; print('ring tokens', len(np.fromfile('$OUT/ring/tokens.i32','<i4')))"
text=$(tail -c +900001 "$RAW" | head -c 11000)
"$BIN" -m "$GGUF" --prompt "$text" --threads 8 --out "$OUT/ring2" > "$OUT/ring2.log" 2>&1
/data/engine/.venv/bin/python -c "import numpy as np; print('ring2 tokens', len(np.fromfile('$OUT/ring2/tokens.i32','<i4')))"

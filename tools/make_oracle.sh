#!/bin/bash
# Regenerates the oracle data set under tools/oracle/ with the stock llama.cpp build (see tools/README.md).
#   tools/make_oracle.sh logits    # 6 prompts -> tools/oracle/<name>/ (oracle_logits)
#   tools/make_oracle.sh greedy    # 5 prompts x 64 tokens -> tools/oracle/greedy/<name>/ (oracle_greedy)
#   tools/make_oracle.sh all
# env: MODEL, THREADS (default 8), BIN (tools build dir), OUT
set -euo pipefail
export PATH=/usr/bin:/bin
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf}
THREADS=${THREADS:-8}
BIN=${BIN:-$HERE/build}
OUT=${OUT:-$HERE/oracle}
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-active}

declare -A P
P[s1_en_capital]='The capital of France is Paris, and the capital of Germany is'
P[s2_code_add]=$'def add(a, b):\n    return a + b'
P[s3_en_story]='Once upon a time, in a small village by the sea, there lived'
P[m1_en_llm]='Large language models are trained on vast amounts of text and learn the statistical patterns of language. When asked a question, they generate an answer one token at a time, each token depending on everything that came before it.'
P[m2_code_c]=$'int sum(const int *a, int n) {\n    int s = 0;\n    for (int i = 0; i < n; i++) s += a[i];\n    return s;\n}\n'
P[c1_chat_user]='<|message_user|><|content_text|>Explain in two sentences why the sky is blue.<|end_message|><|message_model|>'
LOGIT_SET=(s1_en_capital s2_code_add s3_en_story m1_en_llm m2_code_c c1_chat_user)
GREEDY_SET=(s1_en_capital s2_code_add m1_en_llm m2_code_c c1_chat_user)

what=${1:-all}
echo "load before: $(cut -d' ' -f1-3 /proc/loadavg)  threads $THREADS" >&2
if [[ $what == logits || $what == all ]]; then
  for k in "${LOGIT_SET[@]}"; do
    extra=()
    [[ $k == s1_en_capital ]] && extra=(--all-logits)   # per-position logits for ONE short prompt (10 MB)
    "$BIN/oracle_logits" -m "$MODEL" --threads "$THREADS" --prompt "${P[$k]}" "${extra[@]}" --out "$OUT/$k"
  done
fi
if [[ $what == greedy || $what == all ]]; then
  for k in "${GREEDY_SET[@]}"; do
    "$BIN/oracle_greedy" -m "$MODEL" --threads "$THREADS" -n 64 --prompt "${P[$k]}" --out "$OUT/greedy/$k"
  done
fi
echo "load after: $(cut -d' ' -f1-3 /proc/loadavg)" >&2

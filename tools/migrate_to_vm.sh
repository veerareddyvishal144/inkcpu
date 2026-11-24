#!/bin/bash
# Bring up inkcpu on a fresh Ubuntu VM from this machine:  tools/migrate_to_vm.sh <host-or-ip> [threads]
# 1) rsync the repo (sources, docs, tools, server; no build dirs) and the two model files, 2) install build deps, 3) build,
# 4) memory-bandwidth spot test, 5) unit tests + full-model gates, 6) decode/prefill/spec benchmark, 7) start the server on 127.0.0.1:8000.
set -euo pipefail
H=${1:?usage: tools/migrate_to_vm.sh <host> [threads]}; T=${2:-auto}
SRC=/data/inkling-mini/engine; MF=/data/inkling-mini/models_fresh; R=azureuser@$H
SSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 $R"
echo "== 1/7 copy repo + models to $H ($(date -u +%T))"
$SSH "mkdir -p /data/inkling-mini/engine /data/inkling-mini/models_fresh" 2>/dev/null || $SSH "sudo mkdir -p /data/inkling-mini && sudo chown -R azureuser /data && mkdir -p /data/inkling-mini/engine /data/inkling-mini/models_fresh"
rsync -a --info=progress2 --exclude 'build*' --exclude 'run/' --exclude '.venv' --exclude 'tests/spec/taps/*.taps' --exclude '__pycache__' $SRC/ $R:/data/inkling-mini/engine/
rsync -a --info=progress2 $MF/Inkling-2-Mini-UD-Q4_K_XL.gguf $MF/dspark-Inkling-2-Mini-Q8_0.gguf $R:/data/inkling-mini/models_fresh/
$SSH "ln -sfn /data/inkling-mini/models_fresh /data/inkling-mini/models"   # the benches/tests default to the models/ path
echo "== 2/7 deps ($(date -u +%T))"
$SSH 'sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake ninja-build python3-venv python3-pip git > /dev/null && lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Flags" | cut -c1-160 && free -g | sed -n 2p'
echo "== 3/7 build ($(date -u +%T))"
$SSH 'cd /data/inkling-mini/engine && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release > build.cfg.log 2>&1 && ninja -C build > build.log 2>&1 && tail -1 build.log && python3 -m venv .venv && .venv/bin/pip install -q -r server/requirements.txt && echo venv ok'
NT=$T; if [ "$T" = auto ]; then NT=$($SSH 'lscpu -p=CORE | grep -v "^#" | sort -u | wc -l'); fi; echo "physical cores on $H: $NT"
echo "== 4/7 memory bandwidth spot test ($(date -u +%T))"
$SSH "cd /data/inkling-mini/engine && build/src/engine/bench_stream $NT 3.0 2>&1 | tail -6"
echo "== 5/7 unit tests + full-model gates ($(date -u +%T))"
$SSH 'cd /data/inkling-mini/engine && for t in build/src/kernels_gemv/test_gemv build/src/kernels_gemv/test_gemv_multix build/src/kernels_gemm/test_gemm build/src/kernels_misc/test_kernels_misc build/src/kernels_gemv/test_threadpool; do $t > /dev/null 2>&1 && echo "PASS $(basename $t)" || echo "FAIL $(basename $t)"; done'
$SSH "cd /data/inkling-mini/engine && export INK_GGUF=/data/inkling-mini/models_fresh/Inkling-2-Mini-UD-Q4_K_XL.gguf INK_THREADS=$NT && build/src/engine/test_engine determinism 2>&1 | tail -2 && build/src/engine/test_prefill quick 2>&1 | grep -E 'argmax|BIT-IDENTICAL' | head -3 && build/src/spec/test_verify quick 2>&1 | tail -1"
echo "== 6/7 benchmark ($(date -u +%T))"
$SSH "cd /data/inkling-mini/engine && export INK_GGUF=/data/inkling-mini/models_fresh/Inkling-2-Mini-UD-Q4_K_XL.gguf && INK_PROFILE=1 build/src/engine/bench_decode --threads $NT --ctx 16,1000 --n 64 --reps 3 2>&1 | grep -E '^\s+(16|1000)\s' && build/src/engine/bench_prefill --threads $NT --prompts 256,512,1024 --chunk 512 --reps 2 2>&1 | grep -A3 '^threads' && INK_THREADS=$NT build/src/spec/bench_spec --modes plain,auto --n 120 --prompts code_completion,chat_code,lookup_docstring 2>&1 | grep -A4 '^pooled'"
echo "== 7/7 start the server ($(date -u +%T))"
$SSH "cd /data/inkling-mini/engine && INK_MODEL_DIR=/data/inkling-mini/models_fresh INK_THREADS=$NT INK_CPUS=\$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 | paste -sd,) bash tools/serve.sh start && sleep 8 && bash tools/serve.sh status | head -2"
echo "== done ($(date -u +%T)). From your Mac: ssh -N -L 8000:127.0.0.1:8000 azureuser@$H"

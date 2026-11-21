#!/bin/bash
# Build llama.cpp with the Inkling 2 DSpark drafter (upstream ggml-org PR #32218, head 0a9105de) in /data/inkling-mini/llama-unsloth.
# Why not the stock Unsloth tag: unslothai/llama.cpp tag b12588-mix-c0c3de8 is a base tree that has NO Inkling code; Unsloth's release
# merges pins (incl. ggml-org #32218 commit 78d7794) at CI time. So this directory (originally the stock-tag clone) is switched to the
# upstream PR head, which is upstream master + one DSpark commit and already contains inkling2_loop. Nothing under /data/inkling-mini/llama.cpp is touched.
set -e
export PATH=/usr/bin:/bin
cd /data/inkling-mini/llama-unsloth
git fetch --depth 30 https://github.com/ggml-org/llama.cpp pull/32218/head 2>&1 | tail -3
git checkout --detach FETCH_HEAD 2>&1 | tail -3
git log --oneline -3
git show --stat --format='%H %an %cd %s' HEAD | tail -30
git fetch --depth 2 https://github.com/ggml-org/llama.cpp 78d779424d83e17a450807fe42e28445510134b4 2>&1 | tail -2 || echo "unsloth-pinned commit fetch failed"
nice -n 15 cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_MTMD=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON 2>&1 | tail -n 8
nice -n 15 cmake --build build -j 8 --target llama-server llama-cli llama-speculative-simple 2>&1 | tail -n 60
echo BUILD_DONE
ls -la build/bin | head -30

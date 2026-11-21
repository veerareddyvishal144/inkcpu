#!/bin/bash
# Clone and build the Unsloth llama.cpp fork (tag b12588-mix-c0c3de8) CPU-only into /data/inkling-mini/llama-unsloth.
# Does not touch /data/inkling-mini/llama.cpp or its build dirs. Flags mirror llama.cpp/build-sys (GGML_NATIVE=ON, OpenMP on).
set -e
export PATH=/usr/bin:/bin
cd /data/inkling-mini
if [ ! -d llama-unsloth/.git ]; then
  git clone --depth 1 --branch b12588-mix-c0c3de8 https://github.com/unslothai/llama.cpp llama-unsloth
fi
cd llama-unsloth
git log -1 --format='HEAD %H %cd %s'
nice -n 15 cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
nice -n 15 cmake --build build -j 8 --target llama-server llama-cli llama-speculative-simple 2>&1 | tail -n 400
echo BUILD_EXIT=$?
ls -la build/bin | head -40

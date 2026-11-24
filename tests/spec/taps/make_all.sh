#!/bin/bash
# Regenerates the tap files of the 8 DSpark prompts (prompt + 160 greedy-continuation tokens each): ~4 min on 6 threads. Generated data: do not commit.
cd "$(dirname "$0")/../../.."
for p in chat_answer chat_code chat_email code_completion long_context_qa prose_continuation reasoning tool_call; do
  INK_CPUS=${INK_CPUS:-10,12,14,16,18,20} nice -n 5 build-s2-b/spec/dump_taps --tokens tests/spec/prompts/$p.i32 --gen 160 --threads 6 --out tests/spec/taps/$p.taps || exit 1
done
echo ALL_DONE > tests/spec/taps/DONE

# Agent A (core) status: DONE (2026-11-21)

Deliverables (all in place, built and run; standalone build: `export PATH=/usr/bin:/bin; cmake -S src/core/standalone -B build-core -G Ninja; cmake --build build-core`)
- include/ink/gguf.h + src/core/gguf.cpp: mmap GGUF v3 reader, opens the 17.6 GB file in ~4 ms (63 KV, 744 tensors, arrays incl. tokenizer skipped not copied).
- include/ink/model.h + src/core/model.cpp: ModelConfig (all hyper-parameters from KV; hard-coded rules named in ModelConfig::hard_coded), Weights (TensorRef = ptr/type/ne for every tensor incl. `.scale`, loop queries, controllers, heads), visit_plan() (48 visits: phys, pass, kv_slot, llama_slot).
- src/core/ref_forward.{h,cpp}: RefModel/RefState scalar reference, one token at a time, OpenMP over rows (<= 8 threads), options act_quant_like_ggml (default true), kv_f16 / q_f16 / bias_f16 tri-states, sliding_window override hook, trace hook with llama.cpp tensor names.
- docs/MODEL.md: exact per-visit forward pass, tensors, state, numerics, oddities, validation numbers, gaps.
- tests/core: test_gguf (all checks pass), test_ref_state (all pass: determinism bit-identical, reset, fork+continue, position/cache accounting, SWA ring), test_ref_vs_oracle, plus debug tools dump_llama_acts (llama.cpp eval-callback dump) and compare_acts (layer-by-layer compare).

## Measured vs llama.cpp (oracle = agent E data, whole prompt one batch; load avg 3-6; ~1 s/token at 8 threads)
| prompt | tok | argmax | top10 | cosine | rel L2 |
| s1_en_capital | 13 | same | 10 | 0.99987 | 0.0164 |
| s2_code_add | 14 | same | 10 | 0.99994 | 0.0114 |
| s3_en_story | 15 | same | 10 | 0.99914 | 0.0414 |
| c1_chat_user | 14 | same | 9 | 0.99989 | 0.0278 |
| m1_en_llm | 43 | same | 10 | 0.99885 | 0.0479 |
| m2_code_c | 52 | same | 9 | 0.99406 | 0.109 |
Target cosine >= 0.9999 reached on 1 of 6 (s2). Same argmax and top-10 overlap >= 9 on 6/6. Test hard bar is cosine >= 0.999 (default 4 prompts pass; the 43/52-token extended prompts FAIL it).
Why: layers 0-5 are bit-identical to llama.cpp until the first discrete flip (Q8_0 re-quantisation / F16 rounding / top-8 expert choice triggered by a 1e-7 input change); a relative 1e-7 embedding perturbation alone moves the cosine by ~2e-4. See MODEL.md section 9.

## Open problems / not validated
- Window-16 + 48-token comparison is impossible: llama.cpp refuses `--override-kv inkling2_loop.attention.sliding_window=int:16` (attn_rel_bias shape check 16x16 vs 16x1024). Window logic only checked internally (ring size, newest cells, window>prompt == default) and by reading llama.cpp. Window 1024 wrap-around, tau>1 (pos>=8191), >1024 distances, decode path/split-KV not covered by any oracle.
- Global-layer dynamic-tap Kcur/Vcur differ 5e-8 (not bit-exact); controller F32 dot order unverified; passes 1-2 validated end-to-end only.
- Oracle was delivered by agent E in time; I did not write my own oracle (only dump_llama_acts for debugging).

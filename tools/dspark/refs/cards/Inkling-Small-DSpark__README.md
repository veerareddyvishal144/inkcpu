---
library_name: transformers
pipeline_tag: text-generation
tags:
- speculative-decoding
- dspark
- dflash
- specforge
- sglang
inference: false
---

# Inkling-Small DSpark speculator

## Overview

A DSpark speculator for the Inkling-Small NVFP4 target, enabling faster inference through speculative decoding. DSpark extends the DFlash parallel-draft backbone with a Markov logit-bias head and a per-position confidence head. This checkpoint was trained with [SpecForge](https://github.com/sgl-project/SpecForge/) using hidden states from a live SGLang target engine.

## Model Specifications

- **Base model:** `thinkingmachines/Inkling-Small-NVFP4`; the exact matching target checkpoint and tokenizer are required.
- **Format:** Safetensors (single-file BF16, 1.39B trainable parameters; draft weights only).
- **Draft:** 6 layers (Qwen3-style GQA), hidden 4096, 32 heads / 8 KV heads, head_dim 128, FFN 12288, rope_theta 8000000, `block_size=7`.
- **RoPE:** YaRN, factor 128, `original_max_position_embeddings=8192`, `max_position_embeddings=1048576` (matches the target addressable range); config carries both `rope_scaling` and `rope_parameters` schemas.
- **Vocabulary:** 200,058 tokenizer entries and 201,024 padded weight rows; `mask_token_id=200064`.
- **DSpark heads:** Markov rank 256 (vanilla) and confidence head (with-Markov).
- **Aux hidden-state layers:** `[1, 6, 12, 17, 23, 28, 34, 39]`.
- **Trained context:** 8,192 native; long-context adaptation with sequences up to 65,536.
- **Target weights:** target embedding and unembedding weights are not included in this checkpoint.

## Evaluation Results

Acceptance length over the full DeepSpec workload datasets. DSPARK block size 7, thinking effort 0.99, temperature 0 and temperature 1.0 / top-p 0.95:

| Dataset | n | acc_len @ T=0 | acc_len @ T=1 |
|:--|--:|--:|--:|
| GSM8K | 1,319 | 5.4254 | 5.2700 |
| MATH500 | 500 | 4.8009 | 4.6787 |
| MBPP | 257 | 4.1515 | 4.0105 |
| HumanEval | 164 | 3.9911 | 3.9084 |
| MT-Bench | 80 | 3.7309 | 3.6076 |
| LBPP | 162 | 3.6420 | 3.5093 |
| AIME24 | 30 | 3.5456 | 3.3864 |
| AIME25 | 30 | 3.4172 | 3.3192 |
| LiveCodeBench | 1,055 | 3.4154 | 3.3139 |
| Alpaca | 52,002 | 3.3504 | 3.2087 |
| Arena-Hard-v2 | 750 | 3.2091 | 3.0265 |
| SWE-Bench | 300 | 3.1914 | 2.9975 |
| **Mean (12)** | | **3.8226** | **3.6864** |

RULER V2 at the 1M input configuration (50 prompts per partition, actual prompts 1,022,335–1,045,000 tokens):

| Partition | acc_len @ T=0 | acc_len @ T=1 |
|:--|--:|--:|
| MK (multi-key) | 4.2665 | 3.9417 |
| MV (multi-value) | 4.2324 | 3.9666 |
| QA | 3.4938 | 3.2632 |

Long-context acceptance is flat on held-out 16K–64K agentic trajectories (67 rows, temperature 0): 3.592 (8–16K), 3.560 (16–32K), 3.534 (32K+).

## Serving with SGLang

Requires a SGLang build with DSpark support

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
sglang serve --trust-remote-code \
  --model-path thinkingmachines/Inkling-Small-NVFP4 --tp 8 \
  --quantization modelopt_fp4 --attention-backend fa4 --page-size 128 \
  --fp4-gemm-backend flashinfer_trtllm --moe-runner-backend flashinfer_trtllm_routed \
  --enable-torch-symm-mem --mamba-radix-cache-strategy extra_buffer \
  --mem-fraction-static 0.60 --swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1 \
  --max-running-requests 68 --reasoning-parser inkling --tool-call-parser inkling \
  --skip-server-warmup --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Inkling-Small-DSpark \
  --speculative-draft-model-quantization unquant \
  --speculative-dspark-block-size 7 \
  --chunked-prefill-size 8192 --cuda-graph-max-bs-prefill 8192 \
  --disable-flashinfer-autotune --host 0.0.0.0 --port 30000
```

## Training Details

- **Framework:** SpecForge online distillation with hidden states captured from a frozen Inkling-Small target served by a colocated per-node TP4 SGLang engine; KV injection of fused target features into every draft layer with block-local bidirectional attention.
- **Loss:** `0.1 CE + 0.9 L1 distillation + 1.0 confidence BCE`, 512 sampled anchors per sequence, `block_size=7`, within-block decay gamma `4`.
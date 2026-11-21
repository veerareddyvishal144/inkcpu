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

# Inkling DSpark speculator

## Overview

A DSpark speculator for the Inkling NVFP4 target, enabling faster inference through speculative decoding. DSpark extends the DFlash parallel-draft backbone with a Markov logit-bias head and a per-position confidence head. This checkpoint was trained with [SpecForge](https://github.com/sgl-project/SpecForge/) using hidden states from
a live SGLang target engine.

**Preview:** This is a preview checkpoint. Official training and evaluation are in progress.

## Model Specifications

- **Base model:** Inkling NVFP4; the exact matching target checkpoint and tokenizer are required.
- **Format:** Safetensors (single-file BF16, 1.93B trainable parameters + embedding + lm_head).
- **Draft:** 5 layers (Qwen3-style GQA), hidden 6144, 64 heads / 16 KV heads, head_dim 64, FFN 12288, rope_theta 8000000, `block_size=7`.
- **Vocabulary:** 200,058 tokenizer entries and 201,024 padded weight rows; `mask_token_id=200064`.
- **DSpark heads:** Markov rank 256 (vanilla) and confidence head (with-Markov).
- **Aux hidden-state layers:** `[5, 17, 35, 47, 59]`.
- **Trained context:** sequence length 4096.
- **Target weights:** target embedding and unembedding weights are not included in this checkpoint.

## Evaluation Results

Acceptance length (`acc_len`) at temperature 0:

| Dataset | acc_len |
|:--|--:|
| GSM8K | 4.7585 |
| MATH500 | 4.1830 |
| HumanEval | 4.0504 |
| MBPP | 4.0577 |
| AIME25 | 3.4986 |
| MT-Bench | 3.2215 |
| LiveCodeBench | 3.1384 |
| Alpaca | 3.0479 |
| Arena-Hard-v2 | 3.0280 |
| **Mean** | **3.6649** |

## Serving with SGLang

Requires a SGLang build (`docker pull lmsysorg/sglang:dev-cu13-inkling-dspark`) with DSpark support:

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
python -m sglang.launch_server \
  --trust-remote-code \
  --model-path thinkingmachines/Inkling-NVFP4 \
  --tp 8 \
  --quantization modelopt_fp4 \
  --attention-backend fa4 \
  --page-size 128 \
  --fp4-gemm-backend flashinfer_trtllm \
  --moe-runner-backend flashinfer_trtllm_routed \
  --enable-torch-symm-mem \
  --mamba-radix-cache-strategy extra_buffer \
  --mem-fraction-static 0.68 \
  --swa-full-tokens-ratio 0.1 \
  --mamba-full-memory-ratio 0.1 \
  --max-running-requests 68 \
  --reasoning-parser inkling \
  --tool-call-parser inkling \
  --skip-server-warmup \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Inkling-DSpark-Preview \
  --speculative-draft-model-quantization unquant \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs-prefill 8192 \
  --disable-flashinfer-autotune \
  --host 0.0.0.0 \
  --port 30000
```

## Training Details

- **Framework:** SpecForge online distillation with hidden states captured from the frozen Inkling target.
- **Data:** 400K Inkling regenerations from `open-perfectblend`.
- **Schedule:** up to 10 epochs, AdamW, peak learning rate `6e-4`, cosine decay, 4% warmup, and gradient clipping at 1.0.
- **Loss:** `0.1 CE + 0.9 L1 distillation + 1.0 confidence BCE`, with 512 sampled anchors per sequence and `block_size=7`.

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

**Preview:** This is a research preview. Official training and evaluation are in progress.

## Model Specifications

- **Base model:** `thinkingmachines/Inkling-Small-NVFP4`; the exact matching target checkpoint and tokenizer are required.
- **Format:** Safetensors (single-file BF16, 0.90B trainable parameters; draft weights only).
- **Draft:** 5 layers (Qwen3-style GQA), hidden 4096, 64 heads / 16 KV heads, head_dim 64, FFN 8192, rope_theta 8000000, `block_size=15`.
- **RoPE:** YaRN, factor 128, `original_max_position_embeddings=8192`, `max_position_embeddings=1048576` (matches the target addressable range); config carries both `rope_scaling` and `rope_parameters` schemas.
- **Vocabulary:** 200,058 tokenizer entries and 201,024 padded weight rows; `mask_token_id=200064`.
- **DSpark heads:** Markov rank 256 (vanilla) and confidence head (with-Markov).
- **Aux hidden-state layers:** `[5, 11, 23, 29, 35]`.
- **Trained context:** 8,192 native; long-context adaptation with sequences up to 65,536.
- **Target weights:** target embedding and unembedding weights are not included in this checkpoint.

## Evaluation Results

Acceptance length (`acc_len`) at temperature 0 (128 prompts per task, DSPARK block size 7, thinking effort 0.99):

| Dataset | acc_len |
|:--|--:|
| GSM8K | 4.7871 |
| MATH500 | 4.1426 |
| MBPP | 3.4394 |
| HumanEval | 3.3491 |
| MT-Bench | 3.1140 |
| LiveCodeBench | 2.9288 |
| AIME25 | 2.8938 |
| Alpaca | 2.7823 |
| Arena-Hard-v2 | 2.6982 |
| **Mean** | **3.3484** |

## Serving with SGLang

Requires a SGLang build with DSpark support

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
sglang serve --trust-remote-code \
  --model-path thinkingmachines/Inkling-Small-NVFP4 --tp 8 \
  --quantization modelopt_fp4 --attention-backend fa4 --page-size 128 \
  --fp4-gemm-backend flashinfer_trtllm --moe-runner-backend flashinfer_trtllm_routed \
  --enable-torch-symm-mem --mamba-radix-cache-strategy extra_buffer \
  --mem-fraction-static 0.68 --swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1 \
  --max-running-requests 68 --reasoning-parser inkling --tool-call-parser inkling \
  --skip-server-warmup --speculative-algorithm DSPARK \
  --speculative-draft-model-path skx618/InkLing-Small-DSpark-Preview \
  --speculative-draft-model-quantization unquant \
  --speculative-dspark-block-size 7 \
  --chunked-prefill-size 8192 --cuda-graph-max-bs-prefill 8192 \
  --disable-flashinfer-autotune --host 0.0.0.0 --port 30000
```

## Training Details

- **Framework:** SpecForge online distillation with hidden states captured from a frozen Inkling-Small target served by a colocated per-node TP4 SGLang engine.
- **Stage 1 (original RoPE, 8K):** 444,264 Inkling-Small self-regenerations from `open-perfectblend` (temperature 1.0, top-p 0.95, reasoning effort 0.99), 2 epochs, AdamW, global batch 512, gradient clipping 1.0; peak LR `6e-4` (cosine, 4% warmup).
- **Stage 2 (YaRN-128 adaptation):** weights-only warm start from stage 1; 250,000 chat rows (192,055 previously unseen regenerations + 57,945 replayed) plus 1,092 real 16K-64K agentic trajectories upsampled x8; `max_length=65536`; constant LR `5e-4` with a 64-step re-warm.
- **Loss:** `0.1 CE + 0.9 L1 distillation + 1.0 confidence BCE`, 512 sampled anchors per sequence, `block_size=15`, within-block decay gamma `28/3`.

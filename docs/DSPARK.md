# DSpark drafter of Inkling 2 Mini: specification, measured acceptance, CPU speed-up estimate

Agent R, 2026-11-21. Scope: (1) a spec precise enough to implement the drafter and its verification without reading code; (2) acceptance statistics measured on THIS machine with real text;
(3) what that means for speculative decoding on this CPU (bytes model of docs/ROUTING.md). Raw data: `tools/dspark/` (scripts, prompts, `raw/*.steps` per-step logs, `raw/*.log`, `refs/` downloaded sources).

**Headline** ([M]: 8 prompts, <= 80 new tokens each, Q4_K_XL target + Q8_0 drafter, llama.cpp PR 32218; L = mean tokens per verify step incl. bonus; L* = break-even from docs/ROUTING.md bytes model with the refined drafter cost (M); speed-up = L x B1 / S(T)):

- greedy, n=3 (T=4): L = **2.37 +- 0.08** (acc@1 0.64), break-even L* = 2.05 (M) / 1.99 (R); estimated speed-up 1.16x (M), 1.22x (Q), 1.19x (R)
- greedy, n=4 (T=5): L = **2.57 +- 0.10** (acc@1 0.62), break-even L* = 2.24 (M) / 2.16 (R); estimated speed-up 1.15x (M), 1.21x (Q), 1.19x (R)
- greedy, n=7 (T=8): L = **2.77 +- 0.15** (acc@1 0.58), break-even L* = 2.75 (M) / 2.59 (R); estimated speed-up 1.00x (M), 1.06x (Q), 1.07x (R)
- t1prob, n=3 (T=4): L = **2.51 +- 0.10** (acc@1 0.69), break-even L* = 2.05 (M) / 1.99 (R); estimated speed-up 1.23x (M), 1.29x (Q), 1.26x (R)
- best bytes-model speed-up (M) among measured configs: 1.23x (t1prob, n=3). Recommendation and caveats: section 10.

## 1. Sources and confidence tags

Every claim below carries a tag. A claim is "verified" when at least two independent implementations (or the checkpoint itself) agree; single-source claims say so; inferred ones are marked **UNVERIFIED**.

| tag | source | what was read |
|---|---|---|
| [L] | llama.cpp upstream PR ggml-org#32218 "feat: Inkling 2 DSpark support", head `0a9105de` (Piotr Wilkin, 2026-11-21; Unsloth's release b12588-mix-c0c3de8 merges the earlier commit `78d7794` of the same PR, https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8). Local tree `/data/inkling-mini/llama-unsloth` (= upstream master + that commit; it already contains `inkling2_loop`). | `src/models/dspark.cpp`, `src/models/dflash.cpp`, `src/models/inkling2-loop.cpp`, `common/speculative.cpp` (`common_speculative_impl_inkling2_dspark`), `common/sampling.cpp` (`..._rejection`), `examples/speculative-simple`. Read directly. |
| [V] | vLLM main `c32251b5` (drafter file added by commit `abb8194dc35e`, "[Model] Add Inkling 2 Small for Blackwell (#67325)"; ships after v0.34.0; recipe says min vLLM 0.35.0). https://vllm.ai/blog/2026-11-12-inkling2-small, https://vllm.ai/blog/2026-11-20-inkling2-mini | `vllm/models/inkling2/nvidia/dspark.py`, `.../ops/dspark.py`, `configs.py`, `v1/worker/gpu/spec_decode/{dspark,dflash}/speculator.py`, `rejection_sampler*.py`, `config/speculative.py`, recipe yaml. Drafter file read directly; the rest through a research subagent (cited lines checked where marked). Copies in `tools/dspark/refs/vllm/`. |
| [S] | SGLang main `a945a721` (2026-11-21), PR #51926 (Inkling-2) on top of the DSPARK work of https://lmsys.org/blog/2026-07-06-dspark-sglang | `srt/models/inkling2_mtp.py`, `srt/models/inkling2.py`, `kernels/ops/speculative/dspark/*`, `srt/speculative/dspark_components/*`. Via a research subagent, spot-checked by me (block inputs, accept kernels). Copies in `tools/dspark/refs/sglang/`. |
| [P] | paper arXiv 2607.05147, https://huggingface.co/papers/2607.05147 (DeepSeek-AI + PKU, v1, 2026-07-06) | full text (`refs/paper/*.txt`) read by a research subagent. The paper describes the generic DSpark (Markov / RNN / confidence heads); it never uses the Mini tensor names. |
| [C] | original checkpoint https://huggingface.co/thinkingmachines/Inkling-2-Mini : `config.json` key `mtp_config`, `mtp.safetensors` header | `plan/raw/hf/repo/config.json`, `plan/raw/hf/headers/mtp.safetensors.json` (tensor names, shapes, dtypes from the header only; no weights downloaded). |
| [G] | Unsloth drafter GGUF https://huggingface.co/unsloth/Inkling-2-Mini-GGUF `dspark-Inkling-2-Mini-Q8_0.gguf` (402,278,560 B) | header dump `tools/dspark/drafter_gguf_dump.txt`. |
| [RA] | RadixArk DSpark cards for Inkling Small (https://huggingface.co/RadixArk/Inkling-Small-DSpark and two Preview cards) | READMEs and config.json (`refs/cards/`). Different drafters (Small); used only for context. |
| [M] | measurements of this document | section 9. |

Not found anywhere: an accepted-length number for Inkling 2 Mini itself (none in the TML announcement, the Mini README, the vLLM/SGLang Mini docs, or the Unsloth card). The `draft_sample_method` key is NOT in the Mini `mtp_config` (15 keys, listed in 2.2); it is a vLLM `SpeculativeConfig` field (section 6.4).

## 2. Constants and checkpoint

### 2.1 Constants (all [C]/[G], agreed by [L][V][S])
H = 2304 (hidden), 24 query heads, 8 KV heads (GQA group 3), head dim d = 128, FFN F = 9216 (dense SwiGLU), 3 drafter layers (layers 0 and 2 sliding-window, layer 1 global), window W = 1024,
RoPE NEOX (rotate-half, pair (i, i+64)), theta 1e6, rotary dim 128, no scaling, RMSNorm eps 1e-6 (weight multiply only), block size B = 8 (slot 0 = anchor, slots 1..7 = mask), gamma = B-1 = 7 drafted tokens,
taps `target_layer_ids` K = [5, 23, 41, 48], Markov rank r = 256, vocab 200058 real / 201024 padded rows, final logit softcap c = 30 (the target's), non-causal attention, no confidence head.
`mtp_config` keys (config.json): hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, max_position_embeddings (1048576), torch_dtype, q_bias=false, o_bias=false,
rms_norm_eps, rope_theta, local_layer_ids [0,2], sliding_window_size 1024, block_size 8, markov_rank 256, markov_head_type "rnn_hidden", target_layer_ids [5,23,41,48]. No other key.

### 2.2 Tensors of `mtp.safetensors` (54 tensors, all BF16, 635,795,744 B) and the Unsloth GGUF names
Shapes are PyTorch [out, in] ([C] header); the GGUF stores the same matrices with ne = [in, out]. The drafter file has NO embedding, NO lm_head and NO embed_norm: it borrows the target's (`embed_norm(embed(token))` and the lm_head) [L][V][S][P].

| HF name (`layers.N.` = per layer, N = 0..2) | GGUF name | shape | MB (bf16) | role |
|---|---|---|---|---|
| `layers.N.input_layernorm.weight` | `blk.N.attn_norm` | [2304] | | pre-attention RMSNorm of the block stream |
| `layers.N.self_attn.q_proj / k_proj / v_proj` | `attn_q / attn_k / attn_v` | [3072,2304] / [1024,2304] / [1024,2304] | 14.2 / 4.7 / 4.7 | block-slot projections |
| `layers.N.self_attn.q_norm / k_norm` | `attn_q_norm / attn_k_norm` | [128] each | | per-head RMSNorm of q and of block k |
| `layers.N.self_attn.o_proj` | `attn_output` | [2304,3072] | 14.2 | |
| `layers.N.self_attn.k_proj_ctx / v_proj_ctx` | `attn_k_ctx / attn_v_ctx` | [1024,2304] each | 4.7 each | K/V of the fused target context (no Q path) |
| `layers.N.self_attn.k_norm_ctx` | `attn_k_ctx_norm` | [128] | | per-head RMSNorm of ctx k |
| `layers.N.dflare_norm.weight` | `blk.N.dflare_norm` | [2304] | | gain of the RMSNorm applied to the layer-N fused context |
| `layers.N.post_attention_layernorm.weight` | `blk.N.ffn_norm` | [2304] | | pre-FFN RMSNorm |
| `layers.N.mlp.gate_proj / up_proj / down_proj` | `ffn_gate / ffn_up / ffn_down` | [9216,2304] / [9216,2304] / [2304,9216] | 42.5 each | SwiGLU |
| `fuse_weight` | `dspark_fuse.weight` | [4,4] | | rows = consumers (layer 0, 1, 2, Markov head), columns = taps; softmax over taps |
| `mask_embed` | `mask_embd.weight` | [2304] | | embedding of mask slots 1..7 (used raw) |
| `norm.weight` | `output_norm.weight` | [2304] | | final RMSNorm of the drafter stack |
| `markov_head.dflare_norm.weight` | `dspark_norm.weight` | [2304] | | gain for the Markov-head context feature (fuse row 3) |
| `markov_head.markov_w1.weight` | `markov_w1.weight` | [201024,256] | 102.9 | embedding table of the previous token (row gather) |
| `markov_head.rnn_proj.weight / .bias` | `joint_proj.weight / .bias` | [768,5120] / [768] | 7.9 | one projection split into gate / candidate / output |
| `markov_head.hidden_out.weight / .bias` | `dspark_hidden_out.weight / .bias` | [2304,256] / [2304] | 1.2 | latent (256) -> hidden space (2304) |

Per layer 174.6 MB (87.3 M params), 3 layers 523.8 MB, Markov head 112.0 MB (of which only 7 rows x 512 B of `markov_w1` are touched per cycle). Unsloth's Q8_0 GGUF quantises the 27 projections (0.40 GB total), keeps `markov_w1`, `hidden_out`, `joint_proj` BF16 and all norms/biases/`fuse`/`mask` F32 [G]; "accepts at the same rate" per Unsloth (no numbers given).
GGUF metadata [G]: `dflash.block_size=8`, `dflash.target_layers=[5,23,41,48]`, `dflash.dspark.markov_rank=256`, `dflash.sample_from_anchor=False`, `dflash.has_confidence_head=False`, `dflash.attention.causal=False`,
`dflash.attention.sliding_window=1024`, `dflash.attention.sliding_window_pattern=[True,False,True]`, `dflash.final_logit_softcapping=30.0`, `tokenizer.ggml.mask_token_id=0` (unused by the Inkling graph: mask slots take `mask_embd`).

## 3. One decode cycle (what runs when)

Let `p` be the position of the **anchor** `a`: the last token sampled by the target (the "bonus"/corrected token of the previous cycle; after prefill, the first sampled token). The target has NOT processed `a` yet: its KV/conv state is valid for positions < p. The drafter holds context K/V for positions < p and the vector `f_rnn` (section 4.1) of position p-1.

1. **Draft** (section 4): one drafter pass over the 8-slot block `[a, mask x7]` at positions p..p+7, then a sequential Markov/RNN chain gives d_1..d_n (n <= 7).
2. **Verify**: the target runs ONE forward over the n+1 tokens `[a, d_1..d_n]` at positions p..p+n ("T = n+1" of docs/ROUTING.md), producing logits for every row and the 4 tap vectors for every row (section 6.1).
3. **Accept** (section 6): k accepted drafts (0..n) plus one extra token b (corrected or bonus), k+1 tokens emitted; b is the next anchor at position p' = p+k+1. Target KV (and short-conv state) for positions > p+k is discarded.
4. **Commit to the drafter**: for the rows 0..k (positions p..p+k) compute the fused context and write ctx K/V; set `f_rnn` from row k. The 8 block K/V entries written in step 1 are scratch and are overwritten/dropped.
Prefill: the same commit step runs over all prompt positions (the target processes the prompt except its last token in llama.cpp [L]; SGLang/vLLM inject every prefill row [S][V]). A request with no cached drafter context cannot draft ([L] skips drafting when `context_pos != pos0-1`).

## 4. Drafter forward, precisely

Notation: `RMS(x; g)` = x / sqrt(mean(x^2) + 1e-6) * g over the last axis; `softmax_K` over the 4 taps; all weights as in 2.2. Verified identical in [L][V][S] unless noted.

### 4.1 Fused target context (computed once per committed position t, from its 4 tap vectors tap_0..tap_3, each 2304)
```
alpha[c, :] = softmax(fuse_weight[c, :])                       # c = 0,1,2 (layers), 3 (Markov head); fp32
mix_c(t)    = sum_k alpha[c,k] * tap_k(t)                       # no per-tap norm, no fc/projection
f_c(t)      = RMS(mix_c(t); gain_c)   gain_c = layers.c.dflare_norm (c=0..2), markov_head.dflare_norm (c=3)
K_ctx_l(t)  = RoPE_neox( RMS_head( Wk_ctx_l . f_l(t) ; k_norm_ctx_l ), position t )   # [8 heads, 128]; RMS_head over each head's 128
V_ctx_l(t)  = Wv_ctx_l . f_l(t)                                                          # [8, 128], neither normed nor rotated
f_rnn       = f_3(p-1)   # only for the LAST committed position; one 2304 vector per cycle
```
Taps (target side) are the tensors fed to the target's visit `v` ([L] `res->t_layer_inp[visit] = cur` before `build_layer`; [S] `state.partial` read before layer i runs; [V] `stack[num_blocks].clone()` before visit v): the running partial-sum stream of the current attention-residual block as it enters visits 5, 23 and 41
(visit numbering: prelude layers 0-5 = visits 0-5, core pass 1 = 6-17, pass 2 = 18-29, pass 3 = 30-41, postlude = 42-47; so 5 = prelude L5, 23 = pass 2 core layer 11, 41 = pass 3 core layer 17), plus tap 48 = the final normalised hidden state (after the output depth-mix and `output_norm`, before lm_head).
Not the depth-mixed layer input. Order of the 4 taps along the concatenated feature axis = order of `target_layer_ids`. Dtype: F32 in [L], bf16 in [V][S].
The drafter requires the default 3 passes of the target loop ([V] blog, [S] cookbook).

### 4.2 Block pass (once per cycle; 8 slots, positions p+j, j = 0..7)
```
X[0]   = RMS(E[a]; 1) * embed_norm_gain         # the TARGET's embedding table and embed_norm, applied to the anchor token
X[1..7]= mask_embed                              # raw, identical for all 7 slots (the input token ids of mask slots are ignored)
for l in 0,1,2:
    h  = RMS(X; input_layernorm_l)
    q  = h Wq^T -> [8,24,128];  k = h Wk^T -> [8,8,128];  v = h Wv^T -> [8,8,128]
    q  = RMS_head(q; q_norm);   k = RMS_head(k; k_norm)                 # per head, weight [128]
    q,k = RoPE_neox(q,k at positions p..p+7)
    keys   = [ K_ctx_l(visible positions) ; k ]   values = [ V_ctx_l(visible) ; v ]      # block keys use k_norm, ctx keys use k_norm_ctx
    attn   = softmax( q . keys^T / sqrt(128) ) . values      # GQA: query head i uses kv head i // 3; NO causal mask: every slot sees all 8 slots
    X      = X + attn_flat Wo^T                              # attn_flat [8, 3072]
    h2     = RMS(X; post_attention_layernorm_l)
    X      = X + ( silu(h2 Wgate^T) * (h2 Wup^T) ) Wdown^T
hid[j] = RMS(X[j]; norm)   for j = 1..7        # slot 0's output is unused (sample_from_anchor = false)
```
Visibility of context keys: layers 0 and 2: ctx positions [p-1024, p-1] (the 1024 positions before the block) + all 8 block slots; layer 1: all ctx positions 0..p-1 + the block.
Same window for all 8 queries (anchored at the block start, not per query): [V] `window = sliding_window_size + block`, "every query of a block sees the whole block and, for local layers, the `window` positions before the block" (tests/models/inkling2/test_dspark.py:96-99); [L] `n_swa = 1024+1` with `block_uniform` (the anchor's mask row serves the whole block); [S] draft window `sliding_window_size + 1`. No attention sinks, relative bias, softcap, log-scaling, bias or gate in the drafter (unlike the target) [V][L]. Attention scale 1/sqrt(128).
The block's own K/V are written to the layer cache at positions p..p+7 before attention ([V] dspark.py:212-216, [L] build_attn with cache); they are scratch (section 5).

### 4.3 Markov / RNN chain (sequential over the n = 1..7 drafted positions)
```
s      = 0 (256);  prev = a                                   # prev_1 = the anchor token
for j = 1..n:
    z      = [ s (256) ; markov_w1[prev] (256) ; hid[j] (2304) ; f_rnn (2304) ]      # 5120, this order
    u      = rnn_proj z + b                                    # 768 = [gate_raw | cand_raw | out_raw], 256 each
    g      = sigmoid(u[0:256]);  c = tanh(u[256:512])
    s      = g * s + (1 - g) * c                               # == cand + gate*(state - cand) in [L]
    lat    = tanh(u[512:768])
    corr   = hidden_out lat + b_out                            # 2304
    logit[v] = 30 * tanh( ( lm_head[v] . hid[j]  +  lm_head[v] . corr ) / 30 )      for v < 200058 (padded rows dropped)
    d_j    = argmax_v logit[v]            (greedy; ties -> lowest id [L])   or   a sample from softmax(logit / T)   (probabilistic, 6.4)
    prev   = d_j
```
The lm_head is the TARGET's (borrowed): [L] reads its Q8_0 copy in the GGUF, [V][S] the bf16 one. The softcap is the target's `final_logit_softcapping` (30). `lm_head . corr` is linear, so
`lm_head . corr = M lat + m0` with `M = lm_head . hidden_out.weight` ([200058,256]) and `m0 = lm_head . hidden_out.bias`: [V] folds this once at load time in fp64 and stores bf16 (`fold`, dspark.py:347-361), [S] likewise (`_fold_weight`, `_fold_bias`); [L] does NOT fold and runs two lm_head matvecs per position (section 8).
Two more vLLM batching facts that hold mathematically: the `[hid ; f_rnn]` columns of `rnn_proj` (4608 of the 5120) do not depend on the chain, so all n positions are projected in one batched pass; only the `[s ; markov_w1[prev]]` columns (512) are sequential. Base logits `lm_head . hid[j]` for all j are one batched lm_head pass [V] dspark.py:544-547.
Numerics: [V] computes `rnn_proj` in fp32 and rounds `raw` to bf16 before the gates; [S] rounds after every op; [L] runs the chain in F32 with F32-accumulated lm_head products. Exact bitwise agreement across implementations is neither expected nor needed.

### 4.4 Chain semantics that matter
* The chain feeds back the token it actually selected (argmax, or the sampled token in probabilistic mode) [L][V][S].
* The drafter always runs the full 8-slot block; the draft length n only truncates the chain ([L]: `n_outputs = n`; [S][V]: always n = 7, vLLM raises if `num_speculative_tokens != 7`, dspark.py:467-472). Truncating therefore gives the same d_1..d_n as the first n tokens of a 7-token draft (block attention sees all 8 slots either way) - **UNVERIFIED numerically**, true by construction of [L] (the block graph does not depend on n).
* `mask_token_id=0` in the GGUF is irrelevant for Inkling (mask slots use `mask_embed`); RadixArk's Small drafters use a real mask token instead [RA].

## 5. Drafter state across cycles
* Per layer a KV cache of ctx K/V for committed positions (K and V: 8 heads x 128 = 1024 elements each, 4 KB per token per layer in bf16, 12 KB per token for 3 layers). Layers 0 and 2 need only the last 1024 committed positions (+ block); layer 1 keeps all committed positions (global).
* Written (a) at prefill for every prompt position, (b) after each verify for positions p..p+k only ([S] `store_prefix(commit_lens rows)`; [V] "rejected suffix rows write no KV"; [L] injects all verified rows and then `seq_rm` drops positions > p+k, equivalent).
* The block's K/V (positions p..p+7) are scratch: [L] `llama_memory_seq_rm(ctx_dft, seq, ckpt.pos_max+1, -1)` after drafting and again after acceptance; [V][S] overwrite them (no explicit rollback; **UNVERIFIED** by tests).
* One 2304-vector `f_rnn` (position p-1), recomputed from row k of the verify forward after every cycle ([S] `commit_lens - 1`; [L] ring `context_embd`, `context_pos`).
* No recurrent state survives a cycle: the Markov state `s` restarts at 0 for each block.

## 6. Verification and acceptance

### 6.1 What the target must provide
For every row of the verify forward (T = n+1 rows: `a, d_1..d_n`): the logits (to verify and to draw the extra token) and the 4 tap vectors (4 x 2304). Only rows 0..k are used afterwards (k known after acceptance), so an engine may compute the fused context lazily for those rows.
The target's verify forward is an ordinary batched forward with positions p..p+n.

### 6.2 Greedy (temperature 0) [L][V][S agree]
Row j (0 <= j <= n) holds the target distribution of the token at position p+j+1. Accept d_{j+1} iff it equals argmax of row j, for j = 0,1,...; stop at the first mismatch or after n. With k accepted, the extra token is argmax of row k (the target's own choice at the mismatch, or the bonus after n accepted). k+1 tokens are emitted. Equals plain greedy decoding up to numerical ties.

### 6.3 Sampling (temperature > 0)
Exact speculative sampling (Leviathan/Chen), lossless w.r.t. the target sampler:
```
for j = 0..n-1:   p = target distribution of row j (after the request's temperature/top-k/top-p/min-p),  q = distribution d_{j+1} was drawn from
    accept d_{j+1} iff u * q(d) < p(d), u ~ U(0,1)            # = accept with probability min(1, p/q)   [S blockwise_rejection.py, V rejection_sampler_utils.py:714-716, L sampling.cpp:803]
    on reject: emit one token from norm(max(p - q, 0)) and stop [S Gumbel-max over max(p-q,0); V log(p-q) via log1p(-q/p); L sampling.cpp:814-841]
if all n accepted: emit one token from p of row n (the bonus)
```
* Greedy drafts (argmax) have q = one-hot: accept with probability p(d); on reject the residual is p with d removed. llama.cpp's default path ("match") samples the target at each row and accepts iff the sample equals the draft; it has the same acceptance probability p(d) and the same residual law, so it is equivalent [L sampling.cpp:698-730, M].
* Probabilistic drafts sample d from q = softmax(draft logits / T) over the full vocabulary (no top-k/top-p on q) and the ratio test uses the dense q: [L] `--spec-draft-sampling probabilistic` (default greedy); [V] `draft_sample_method="probabilistic"` (default "greedy"; the Mini recipe sets probabilistic and says "drafts are sampled from the draft distribution, as in the reference"); [S] drafts are sampled whenever the request has temperature > 0 and top_k > 1, env `SGLANG_DSPARK_USE_GREEDY_DRAFT` forces argmax. The DSpark paper evaluates at T = 1 with sampled (probabilistic) drafts [P].
* Draw coupling: the chain feeds the SAMPLED token back [L][S][V].

### 6.4 `draft_sample_method`
Not in the checkpoint's `mtp_config` (2.1), not read by SGLang (inert attribute), not read by llama.cpp (flag `--spec-draft-sampling`). It is a vLLM `SpeculativeConfig` field: `greedy` (default; q treated as one-hot) or `probabilistic` [V config/speculative.py:700-706, rejection_sampler_utils.py:198-201].

### 6.5 Rollback
Target KV: positions > p+k are dropped. Target short-conv state: [V] stores each conv stream's INPUT at every absolute position (ops/sconv.py:3-24), i.e. a per-position cache and no recurrent state, so rejected positions are simply overwritten (inferred, **UNVERIFIED**); [L] keeps bounded recurrent-state snapshots (`llama_n_rs_seq > 0`, `COMMON_CONTEXT_SEQ_RM_TYPE_RS`), which is why the example needs no checkpoint/replay ([M]: no "will use checkpoints" message). An engine must make the conv history restorable to the state after row k (e.g. keep the last T+3 conv inputs).
Drafter: nothing to roll back except scratch K/V (section 5).

### 6.6 Block verification variants
[V] `rejection_sample_method` = standard | synthetic | block | fly and [S] `--speculative-use-block-verification` exist but are off by default; not analysed here.

## 7. Confidence and length scheduling

* The DSpark paper adds a confidence head c_k = sigmoid(w^T [h_k ; W1[x_{k-1}]]) (Eq. 7), calibrated post hoc (sequential temperature scaling), and a serving-time scheduler that admits (request, position) slots by cumulative survival probability under a profiled step-cost table (Alg. 1). Its Table 1 runs WITHOUT the scheduler; production (DeepSeek-V4) uses it with gamma = 5 [P Sec. 3.2, 4.2, 5.2].
* **Inkling 2 Mini ships no confidence head**: no confidence tensor in the 54 [C]; GGUF `has_confidence_head=False`; llama.cpp throws unless `--spec-draft-p-min 0` [L speculative.cpp:1545]; SGLang's loader raises on unknown tensors and sets `confidence_head = None` [S inkling2_mtp.py:130,369-372]; vLLM asserts `not enable_adaptive_verification` for Inkling2 [V speculator.py:20-21].
* Therefore at inference the verify width is fixed: [S] all 8 rows (`SGLANG_RAGGED_VERIFY_MODE=static`, default; compact/cap-accept need an SPS table and an online estimator, both off by default), [V] exactly 7 drafts, [L] default `--spec-draft-n-max 3`, any n in 1..7 (Unsloth: "values up to 7 are accepted: longer drafts land more tokens per step but spend more draft compute on tokens the target rejects").
* Any adaptive length for this engine would be an engine policy (section 10), not part of the model.

## 8. Cost model for the engine

Bytes read per cycle (decode is bandwidth-bound here, docs/ROUTING.md; MB = 1e6 B; engine-plan numbers N = 2014.6 MB, E(T) from ROUTING.md table (d), B1 = 3861.9 MB per plain token):
`S(T) = N + E(T) + Dr(n)` with T = n+1. Components of Dr for an implementation that exploits the algebra of 4.3:

| part | bytes | note |
|---|---|---|
| 3 drafter layers, bf16 | 523.8 MB | q,k,v,o + ctx k,v + SwiGLU; the ctx projections (28 MB) are applied to the k+1 newly committed rows, the rest to the 8 block rows; Q8_0 would be 278 MB |
| `rnn_proj` + `hidden_out` | 9.1 MB | fits in L3, count once |
| base logits `lm_head . hid[1..n]` | 489.7 MB | ONE batched pass (Q8_0 lm_head, 200058 rows) |
| per chained position: `M lat + m0` | 102.4 MB bf16 (54.4 MB Q8_0) | `M = lm_head . hidden_out`, resident +102 MB; n sequential passes |
| `markov_w1` | ~0 | n gathered rows of 512 B |
| global-layer ctx K/V | 4 KB x ctx_len | layer 1 reads all committed K/V per cycle (8 MB at 2K, 134 MB at 32K); SWA layers <= 8.4 MB |

So `Dr(n) = 523.8 + 9.1 + 489.7 + 102.4 n` MB (variant M) = 1329.8 MB at n = 3, 1739.4 MB at n = 7. docs/ROUTING.md assumed 635.8 + 489.7 = 1125.5 MB independent of n (variant R); the `markov_w1` table is never streamed but the n sequential passes over `M` were missing. A top-K candidate restriction of the correction (vLLM has `dspark_draft_topk` for Qwen3 DSpark, not Inkling) could remove most of the `M` term; it is an approximation and **UNVERIFIED** for this checkpoint.
**llama.cpp as built does none of this**: `inkling2_head_step` runs `lm_head . hid[j]` and `lm_head . corr` as two separate full lm_head matvecs per position (2 x 492 MB Q8_0), so `Dr_llama(n) = 523.8 + 9.1 + 984 n` MB (3.5 GB at n = 3, 7.4 GB at n = 7, more than one plain target token). Measured drafting time in [M] confirms it is slow.
The ROUTING.md caveat (verify of rejected tokens may touch more experts than the measured true continuations) still applies.

## 9. Measurement

### 9.1 Method
* Target `/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf` (community mixed GGUF: NVFP4/MXFP8 experts, Q8_0 attention/embed/lm_head), drafter `dspark-Inkling-2-Mini-Q8_0.gguf`, llama.cpp at PR #32218 head `0a9105de` built CPU-only (`-march=native` GGML_NATIVE, OpenMP, Ninja, system gcc 11.4) in `/data/inkling-mini/llama-unsloth`. Binary: `llama-speculative-simple`, instrumented by `tools/dspark/patch_speculative_simple.py`
  (`DSPARK_STEPLOG`: one line per verify step with n_draft, accepted drafts, times; rejection-sampling verification when `--spec-draft-sampling probabilistic` and temp > 0; stop on EOS id 200006, which the GGUF does not mark as EOG).
* 8 prompts (`tools/dspark/prompts/`, built with the chat markers of ARCH.md section 4 / the jinja template, `tools/dspark/make_prompts.py`): chat answer (sky), code completion (raw Python), prose continuation (raw), reasoning-mode turn (thinking prefilled, effort 0.9), tool-call turn (tool_declare + weather question), ~600-token-context question (wikitext passage + "summarise"), chat email, chat code. 80 new tokens each (fewer if EOS). Seed 1234. `-t 8 -td 8 -c 2048`, `nice -n 15`.
* Modes: `greedy` (temp 0); `t1prob` (temp 1.0, top-k off, top-p 1, min-p 0; drafter samples; target verifies by rejection sampling); `t1match` (temp 1.0, drafter argmax, target samples and must match). Draft lengths n = 3, 4, 7 (T = 4, 5, 8).
* Statistic definitions: one **step** = one verify forward of a fresh draft. Only steps with the full requested draft length are counted (the last 1-3 steps of a run have shorter drafts because of the token budget). **L** = mean(accepted drafts + 1) = mean tokens per verify step (includes the bonus/corrected token; this is the "mean accepted length" compared with ROUTING.md's L*). acc@j = fraction of steps in which draft positions 1..j were all accepted; c@j = acc@j / acc@(j-1).
* Acceptance does not depend on speed; these runs did not take the bench lock (other agents were benchmarking in parallel; the per-run load average is in the logs and in the tables). Speed rows in 9.3 were taken under `tools/with_bench_lock.sh`.
* Caveats: 8 prompts x ~80 tokens per config is a small sample (SEs are given); the Q4_K_XL target and Q8_0 lm_head differ from the bf16 targets of [V][S][P][RA]; no bit-level comparison with a reference implementation was possible (no GPU; the HF reference code is not public here), but the generated text is coherent and per-position acceptance decays smoothly.

### 9.2 Acceptance results
| prompt | tokens in prompt (llama.cpp `encoded`) |
|---|---|
| chat_answer | 34 |
| code_completion | 105 |
| prose_continuation | 74 |
| reasoning | 80 |
| tool_call | 153 |
| long_context_qa | 715 |
| chat_email | 36 |
| chat_code | 44 |

#### Pooled acceptance (all prompts, full-length drafts only; L = mean tokens per verify step = accepted drafts + 1)

| mode | draft n (T=n+1) | prompts done | steps | **L +- SE** | accepted drafts/step | acc@1 | acc@2 | acc@3 | acc@4 | acc@5 | acc@6 | acc@7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 3 (T=4) | 8/8 | 260 | **2.37 +- 0.08** | 1.37 | 0.64 | 0.41 | 0.32 | - | - | - | - |
| greedy | 4 (T=5) | 8/8 | 238 | **2.57 +- 0.10** | 1.57 | 0.62 | 0.41 | 0.31 | 0.24 | - | - | - |
| greedy | 7 (T=8) | 8/8 | 214 | **2.77 +- 0.15** | 1.77 | 0.58 | 0.37 | 0.28 | 0.22 | 0.14 | 0.10 | 0.07 |
| t1prob | 3 (T=4) | 5/8 | 155 | **2.51 +- 0.10** | 1.51 | 0.69 | 0.47 | 0.35 | - | - | - | - |

acc@j = fraction of verify steps in which draft position j was accepted (cumulative: positions 1..j all accepted).

#### Conditional acceptance P(position j accepted | positions < j accepted)

| mode | n | c@1 | c@2 | c@3 | c@4 | c@5 | c@6 | c@7 |
|---|---|---|---|---|---|---|---|---|
| greedy | 3 | 0.64 | 0.64 | 0.77 | - | - | - | - |
| greedy | 4 | 0.62 | 0.66 | 0.75 | 0.77 | - | - | - |
| greedy | 7 | 0.58 | 0.63 | 0.76 | 0.78 | 0.62 | 0.76 | 0.73 |
| t1prob | 3 | 0.69 | 0.68 | 0.74 | - | - | - | - |

#### Per prompt: L (steps) 

| prompt | greedy n=3 | greedy n=4 | greedy n=7 | t1prob n=3 |
|---|---|---|---|---|
| chat_answer | 1.77 (44) | 1.81 (43) | 1.79 (42) | 2.05 (38) |
| code_completion | 3.67 (21) | 4.33 (18) | 5.92 (13) | 3.50 (22) |
| prose_continuation | 1.67 (46) | 1.71 (45) | 1.66 (44) | 1.75 (44) |
| reasoning | 2.85 (27) | 3.17 (24) | 3.48 (21) | 2.89 (27) |
| tool_call | 3.13 (23) | 3.79 (19) | 5.14 (14) | 3.29 (24) |
| long_context_qa | 2.72 (29) | 2.85 (27) | 3.17 (23) | - |
| chat_email | 1.73 (45) | 1.85 (41) | 1.85 (40) | - |
| chat_code | 3.12 (25) | 3.71 (21) | 4.41 (17) | - |

#### Bytes-model speed-up estimates (ROUTING.md, mean expert-union bytes; decode bandwidth-bound; speed-up = L * B1 / S(T))

S(T) = N + E(T) + Dr: N = 2014.6 MB, B1 = 3861.9 MB/token, E(T) from ROUTING.md (d). Dr variants: (R) ROUTING.md 635.8 MB drafter + 1 lm_head pass = 1125.5 MB; (M) refined = 523.8 layers + 9.1 head + 489.7 batched base lm_head pass + n x 102.4 MB (one pass over a precomputed bf16 [200058 x 256] matrix lm_head x hidden_out per chained position); (C) llama.cpp as implemented = 523.8 + 9.1 + 2 x n x 492 MB (two lm_head passes per position).

(Q) = (M) with the drafter layers and the folded matrix stored Q8_0 (278.3 MB layers + 54.4 MB per chained position; Unsloth reports the same acceptance for its Q8_0 drafter).

| mode | n | T | L measured | L* (R) | L* (M) | L* (Q) | speed-up (R) | speed-up (M) | speed-up (Q) | speed-up (C) | S(T) (M) MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 3 | 4 | 2.37 | 1.99 | 2.05 | 1.94 | 1.19x | 1.16x | 1.22x | 0.91x | 7898 |
| greedy | 4 | 5 | 2.57 | 2.16 | 2.24 | 2.13 | 1.19x | 1.15x | 1.21x | 0.85x | 8666 |
| greedy | 7 | 8 | 2.77 | 2.59 | 2.75 | 2.60 | 1.07x | 1.00x | 1.06x | 0.65x | 10631 |
| t1prob | 3 | 4 | 2.51 | 1.99 | 2.05 | 1.94 | 1.26x | 1.23x | 1.29x | 0.96x | 7898 |

#### llama.cpp reference timing (this machine, 8 threads, shared host: indicative; tokens/s from the log's `decoded` line)

| mode | n | mean ms per step: drafter | target verify (T tokens) | feature injection | tokens/s (mean over prompts) | loadavg before (min..max 1-min) |
|---|---|---|---|---|---|---|
| greedy | 3 | 111 | 515 | 2.3 | 4.19 | 2.7..14.3 |
| greedy | 4 | 167 | 779 | 3.4 | 3.15 | 13.5..19.0 |
| greedy | 7 | 225 | 862 | 3.8 | 3.19 | 6.9..17.1 |
| t1prob | 3 | 159 | 572 | 2.9 | 3.29 | 13.9..15.7 |

#### Histogram of accepted drafts per step (pooled; count of steps with k accepted)

- greedy n=3: 0:93, 1:60, 2:25, 3:82
- greedy n=4: 0:90, 1:51, 2:24, 3:17, 4:56
- greedy n=7: 0:89, 1:46, 2:19, 3:13, 4:18, 5:7, 6:6, 7:16
- t1prob n=3: 0:48, 1:34, 2:19, 3:54


#### Break-even table (bytes model of docs/ROUTING.md, no measurement)

| n | T=n+1 | S(T) R MB | L* (R) | S(T) M MB | L* (M) | S(T) Q MB | L* (Q) | max speed-up at L=T (M) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 6068 | 1.57 | 6068 | 1.57 | 5774 | 1.50 | 1.27x |
| 2 | 3 | 6943 | 1.80 | 7045 | 1.82 | 6704 | 1.74 | 1.64x |
| 3 | 4 | 7694 | 1.99 | 7898 | 2.05 | 7509 | 1.94 | 1.96x |
| 4 | 5 | 8359 | 2.16 | 8666 | 2.24 | 8228 | 2.13 | 2.23x |
| 5 | 6 | 8960 | 2.32 | 9369 | 2.43 | 8883 | 2.30 | 2.47x |
| 6 | 7 | 9510 | 2.46 | 10021 | 2.59 | 9488 | 2.46 | 2.70x |
| 7 | 8 | 10016 | 2.59 | 10631 | 2.75 | 10049 | 2.60 | 2.91x |


### 9.3 Speed of this llama.cpp build (reference only)
(speed runs not available)


## 10. Implication for this CPU and recommendation
**What was measured** (section 9.2; llama.cpp PR 32218, Q4_K_XL target, Q8_0 drafter, 8 prompts, greedy). Mean tokens per verify step L = **2.37 +- 0.08** at n = 3 (T = 4, 260 steps), **2.57** at n = 4 (T = 5), **2.77 +- 0.15** at n = 7 (T = 8). Cumulative per-position acceptance falls fast (acc@1 about 0.64, acc@2 about 0.4, acc@3 about 0.3, acc@7 about 0.07), but conditional acceptance is 0.6-0.8 at every position: once the first tokens are right the rest usually are. At temperature 1.0 with sampled drafts and rejection sampling (the paper / vLLM-recipe mode) L = 2.51 at n = 3 and n/a at n = 7 (n/a = not finished when this was assembled; the tables in 9.2 are authoritative).
**There are two regimes**, and the pooled mean describes neither (per-prompt L, steps 13-46 per cell, so +-0.3 noise per cell):

| prompt (text type) | L n=3 | L n=4 | L n=7 | speed-up (M) n=3 | speed-up (M) n=4 | speed-up (M) n=7 | best choice (M) |
|---|---|---|---|---|---|---|---|
| chat_answer | 1.77 | 1.81 | 1.79 | 0.87x | 0.81x | 0.65x | plain (1.00x) |
| prose_continuation | 1.67 | 1.71 | 1.66 | 0.82x | 0.76x | 0.60x | plain (1.00x) |
| chat_email | 1.73 | 1.85 | 1.85 | 0.85x | 0.83x | 0.67x | plain (1.00x) |
| reasoning | 2.85 | 3.17 | 3.48 | 1.39x | 1.41x | 1.26x | n=4 (1.41x) |
| long_context_qa | 2.72 | 2.85 | 3.17 | 1.33x | 1.27x | 1.15x | n=3 (1.33x) |
| chat_code | 3.12 | 3.71 | 4.41 | 1.53x | 1.66x | 1.60x | n=4 (1.66x) |
| tool_call | 3.13 | 3.79 | 5.14 | 1.53x | 1.69x | 1.87x | n=7 (1.87x) |
| code_completion | 3.67 | 4.33 | 5.92 | 1.79x | 1.93x | 2.15x | n=7 (2.15x) |

Structured, predictable text (code, tool calls, code in chat) accepts most of the block and keeps improving with n (L up to 5-6 of 8); reasoning and long-context QA sit at L = 2.7-3.5; ordinary conversational text (chat answer, prose, email) saturates at L = 1.7-1.85 for every n because acceptance after the first one or two tokens adds almost nothing (n = 7 gives the same L as n = 3 there).

**Bytes-model speed-up, pooled** (docs/ROUTING.md expert-union bytes; speed-up = L x B1 / S(T); decode assumed bandwidth-bound; T = n+1; R = ROUTING.md drafter cost 1125.5 MB, M = refined cost with the lm_head fold (section 8), Q = M with Q8_0 drafter and folded matrix; M/Q are the honest columns, R omits the n sequential Markov passes):

| n (T) | measured L (greedy) | break-even L*: R / M / Q | speed-up: R / M / Q |
|---|---|---|---|
| 3 (4) | 2.37 | 1.99 / 2.05 / 1.94 | 1.19x / 1.16x / 1.22x |
| 4 (5) | 2.57 | 2.16 / 2.24 / 2.13 | 1.19x / 1.15x / 1.21x |
| 7 (8) | 2.77 | 2.59 / 2.75 / 2.60 | 1.07x / 1.00x / 1.06x |

Macro mean over the 8 prompts (each prompt weighted equally, M model): fixed n = 3 1.26x, fixed n = 4 1.29x, fixed n = 7 1.25x; best of {plain, n=3, n=4, n=7} chosen per prompt (an upper bound for any online policy) **1.43x (M) / 1.49x (Q)**.

**Recommendation.**
1. **Build it, as an adaptive mode and only with the lm_head fold; do not build always-on speculation.** A fixed T = 4 averages a modest gain (1.16x M, 1.22x Q pooled) because it wins 1.3-1.8x on structured text and reasoning and loses 13-18% on plain conversation; fixed T = 8 (1.00x pooled) only pays on code and tool calls (about 2x) and loses up to 40% on conversation. The per-prompt best choice reaches 1.43x (M).
2. **Policy**: verify width T in {1 (plain), 4, 8}. Keep a running estimate of cumulative acceptance per draft position (EWMA of acc@j over the last ~16 steps), pick the T that maximises L(T)/S(T) with S from the bytes table (section 8), fall back to plain decoding while L(4) < about 2.0 (L*(4) = 2.05 M / 1.94 Q), re-probe every ~32 tokens. T = 2 (needs L > 1.57, c@1 about 0.6 gives about 1.6), T = 5, 6, 7 add nothing over {4, 8}. The model has no confidence head (section 7), so this is engine policy.
3. **Prerequisites**: the fold M = lm_head x hidden_out (+102 MB resident, 102 MB read per chained position; Q8_0 54 MB) and a bf16 or Q8_0 drafter (Unsloth reports the same acceptance for Q8_0); tap capture of 3 visit inputs + the final state for all T rows; rejection sampler for T > 0; target conv-state restore to row k.
4. **Do not use this llama.cpp build as the speed baseline.** It runs the target lm_head twice per chained position (variant C: 0.91x at n = 3 and 0.65x at n = 7 in the bytes model) and its T = 4 verify step takes about 515 ms against roughly 190-200 ms for one token (drafting alone 111 ms per step); section 9.3 has the measured tokens/s. The gains above exist only for an engine that implements the fold and keeps the verify step near the ROUTING.md bytes (1.7x one token at T = 4).
5. **Risks that could change the verdict**: (a) the T = 4 / 8 verify must stay bandwidth-bound on AVX-512 without VNNI (ROUTING caveat 2); (b) ROUTING.md measured expert overlap on true continuations, rejected drafts are off-distribution and under independent routing S(4) grows by about 2.2 GB (3.64x instead of 2.47x the one-token expert bytes), which would turn the pooled T = 4 estimate into a loss of about 9%; (c) real agentic traffic is more structured than this mix (Inkling Small deployments report 3.3-3.5 tokens per step at 7 drafts and temperature 1.0 on 32K-256K agentic sessions, vLLM blog; RadixArk's Small drafter 3.7-3.8 on benchmarks), which favours T = 8; (d) this used a Q4_K_XL target and a Q8_0 lm_head, not the bf16 target the drafter was trained on; (e) 8 prompts x 80 tokens is a small sample.


## 11. Differences between the implementations, open items

| item | llama.cpp [L] | vLLM [V] | SGLang [S] |
|---|---|---|---|
| draft length | 1..7 (default 3), block always 8 slots | exactly 7 | 7 (verify window 8) |
| lm_head for the drafter | target's Q8_0 copy, 2 matvecs per chain step | bf16, folded `M`, one batched base pass | bf16, folded `M` |
| default draft sampling | greedy (`--spec-draft-sampling`) | `draft_sample_method=greedy`, Mini recipe sets probabilistic | sampled when T>0 unless forced greedy |
| verify at T>0 | match (greedy draft) or rejection (probabilistic) | rejection (standard) | blockwise rejection (no top-k/top-p) else classic |
| tap dtype | F32 | bf16 | bf16 |
| conv-state rollback | recurrent-state snapshots | per-position input cache (inferred) | UNVERIFIED |
| adaptive verify | none (needs `--spec-draft-p-min 0`) | asserts off for Inkling2 | static (all 8) |

UNVERIFIED items: (1) the target conv-state rollback in [S]; (2) numerical equivalence of the first n tokens of a 7-token draft with an n-token chain in [V][S] (true by construction in [L]); (3) whether HF's `hidden_states[i]` used in training equals the taps above (all three runtimes agree with each other, none shows the training code); (4) the paper's Table 1 numbers are for Qwen3/Gemma4 drafters, not Mini; (5) the speed-up estimates are bandwidth-bound estimates, not measurements of an engine that does not exist yet; (6) acceptance of the real bf16 target may differ from the Q4_K_XL GGUF used here.

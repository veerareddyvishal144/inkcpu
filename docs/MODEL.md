# MODEL.md: exact forward pass of Inkling 2 Mini (arch `inkling2_loop`) as run by llama.cpp @2a28df0f9

Owner: agent A (core). Written 2026-11-21 from `llama.cpp/src/models/inkling2.cpp`, `inkling2-loop.cpp`, `src/llama-graph.cpp`
(`build_moe_ffn`, `build_attn*`), ggml CPU kernels, and MEASURED against llama.cpp tensors (see section 9). The scalar reference that
implements everything below is `src/core/ref_forward.cpp`; when this text and that file disagree, the file is right (it is the thing
that was validated). Notation: `D=2304`, `H=24` Q heads, `G=8` KV heads, `HD=128`, `QW=3072`, `KVW=1024`, `F=1152`, `E=128`, `U=8`.
All activations are float32 unless said otherwise. `x @ W` means "one dot product per output row, rows contiguous along K".

## 0. Status in one paragraph

Against llama.cpp on the same GGUF (six prompts, 13-52 tokens, whole prompt as one batch): same argmax on 6/6, top-10 overlap 9-10 on
6/6, cosine 0.99987 / 0.99994 / 0.99914 / 0.99989 (the four 13-15 token prompts) and 0.99885 / 0.99406 (43 / 52 tokens). The target
cosine >= 0.9999 is reached on 1 of 6 prompts. This is NOT a structural error: layers 0-5 are bit-identical to llama.cpp (relative L2 = 0.0
on Q, R, attention output, router, gate_up, SiTU, down for every token before the first discrete flip) and the rest of the gap is the
amplification of tiny float differences by the discrete decisions of the pipeline (Q8_0 re-quantisation of activations, F16 rounding, top-8
expert choice); details and the measured noise floor in section 9. An executor that quantises activations the same way will sit in the
same noise floor against llama.cpp, so use "same argmax, top-10 overlap, cosine >= 0.999" as the parity gate, not 0.9999.

## 1. Constants (all from the GGUF KV, prefix `inkling2_loop.`; ModelConfig::from_gguf)

| name | value | KV |
|---|---|---|
| layers / hidden / expert ff | 24 / 2304 / 1152 | block_count, embedding_length, expert_feed_forward_length |
| heads / kv heads / head dim | 24 / 8 / 128 (GQA group 3: head h uses kv head h/3) | attention.head_count(_kv), key_length |
| experts / used / route scale | 128 / 8 / 8.0 | expert_count, expert_used_count, expert_weights_scale |
| SiTU betas | 4.0 (gate), 25.0 (up) | activation.situ_beta, situ_linear_beta |
| rms eps | 1e-6 (float 9.99999997e-07) | attention.layer_norm_rms_epsilon |
| loop | core layers [6,18), 3 passes, gate rank 64, not varying | loop.start/end/count/gate_rank/varying |
| conv kernel | 4 (history 3) | shortconv.l_cache |
| attention-residual block | 6 layers | attn_res.block_size |
| SWA window / pattern | 1024; SWA on all layers except 5, 11, 17, 23 (global) | attention.sliding_window(_pattern) |
| relative bias | 16 features, table of 1024 distances | attention.relative_length / relative_extent |
| clips | qk 22.627416610717773, rel 8.0 | attention.qk_clip / relative_clip |
| log scale | floor 8192, alpha 0.1 | attention.log_scale_floor / alpha |
| logit softcap | 30.0 | final_logit_softcapping |
| vocab | 201024 rows (padded), 200058 real | vocab_size, vocab_size_unpadded |
| rope | NONE (positions only enter through the relative bias and tau) | rope.dimension_count 0 |

Rules that are NOT in the KV (hard-coded in llama.cpp; `ModelConfig::hard_coded`): route-weight normaliser eps 1e-20; clip() adds 1e-6 to
the L2 norm AFTER the sqrt; the RMS eps inside depth-mix and loop controllers is 1e-6; sigmoid router gating; mix softmax scale
`1/sqrtf(2304)`; attention scale `1/sqrtf(128)`; norm weights are plain multipliers (no `1+w`).

## 2. Tensors (names, GGUF shapes ne[0] fastest, types in the community UD-Q4_K_XL GGUF)

744 tensors, data section at 11,980,000 bytes, 32-byte alignment. Type ids: 0 F32, 8 Q8_0, 30 BF16, 40 NVFP4.

Model level: `token_embd.weight` Q8_0 [2304,201024]; `token_embd_norm.weight` F32 [2304]; `output.weight` Q8_0 [2304,201024];
`output_norm.weight` F32 [2304]; `output_res_query.weight` F32 [2304].

Per physical layer `blk.L.` (L = 0..23):

| tensor | type | shape | notes |
|---|---|---|---|
| attn_q / attn_k / attn_v | Q8_0 | [2304,3072] / [2304,1024] / [2304,1024] | |
| attn_output | Q8_0 | [3072,2304] | wo |
| attn_gate | Q8_0 | [2304,3072] | output gate, sigmoid |
| attn_rel | Q8_0 | [2304,384] | 24 heads x 16 features, index h*16+f |
| attn_sink_proj | BF16 | [2304,24] | per-query sink logit t |
| attn_q_clip | F32 | [128,24] | per-channel Q scale, index h*128+d |
| attn_rel_bias | F32 | [16,1024] | bias table, element (f,d) at d*16+f; same shape on SWA and global layers |
| attn_rel_clip | F32 | [16,24] | per-feature R scale, index h*16+f |
| attn_k_conv / attn_v_conv | F32 | [4,1024] | tap j of channel c at c*4+j (j=0 multiplies the OLDEST input) |
| attn_post_conv / ffn_post_conv | F32 | [4,2304] | conv after wo / after the MoE |
| attn_norm | F32 | [2304] | SWA layers only |
| attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm | F32 | [2304] | global layers (5,11,17,23) only |
| attn_k_conv_proj, attn_v_conv_proj | BF16 | [2304,32] | global layers only: dynamic taps, index head*4+lag |
| attn_res_query | F32 | [2304] (SWA, 1 query) or [2304,3] (global, queries Q,K,V) | ABSENT in layer 0 |
| ffn_res_query | F32 | [2304] | |
| ffn_norm | F32 | [2304] | |
| ffn_gate_inp | F32 | [2304,128] | router (bf16 in the original checkpoint) |
| exp_probs_b.bias | F32 | [128] | selection bias |
| ffn_gate_up_exps | NVFP4 (L0-19) / Q8_0 (L20-23) | [2304,2304,128] | rows 0..1151 gate, 1152..2303 up; row = (expert*2304 + r) |
| ffn_down_exps | NVFP4 / Q8_0 | [1152,2304,128] | |
| ffn_gate_up_exps.scale, ffn_down_exps.scale | F32 | [128] | per-expert scale, NVFP4 layers (0-19) only, shared by all passes |
| loop_msg_query | F32 | [2304,4] | layers 6-17: queries for the Q,K,V,FFN messages |
| loop_{q,k,v,ffn}_ctrl_{down,gain,forget} | F32 | [4608,64], [64,2304], [64,2304] | layers 6-17 only (bf16 in the checkpoint) |

Block formats: Q8_0 = 34 B per 32 (fp16 d, int8 q), row = K/32 blocks. NVFP4 = 36 B per 64 (4 x UE4M3 sub-block scales d[4], 32 bytes
of E2M1 nibbles; sub-block s owns bytes 8s..8s+7, element j<8 = LOW nibble of byte j, element 8+j = HIGH nibble); value =
`kvalues_fp4[nibble] * ue4m3(d[s])` with `kvalues_fp4 = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}` and `ue4m3` already containing the
factor 0.5 (include/ink/ggml_types.h); the per-expert F32 `.scale[e]` multiplies the finished dot product. UE4M3 bytes 0x00 and 0x7F decode
to 0, the sign bit (0x80) is ignored.

## 3. Visit plan (visit_plan(), 48 visits)

Execution order: L0..L5 (prelude, once), then 3 passes over L6..L17, then L18..L23 (postlude, once).
`Visit.index` 0..47 is the KV/conv-state slot id. `Visit.llama_slot` is llama.cpp's slot number (il in tensor names such as `l_out-24`):
`il = phys` for prelude, postlude and pass 0; `il = 24 + (pass-1)*12 + (phys-6)` for passes 1, 2 (visits 18-29 -> il 24-35, 30-41 -> 36-47).
Weights are shared: a pass-p visit of layer l uses `blk.l.*` (all of it, incl. expert scales); only KV cache, conv histories and the
message/controller inputs differ per visit. Controllers and message queries are used in passes 1 and 2 (0-based) only.

## 4. State

Per sequence, per visit (48 of them):
* attention cache: K and V post conv/clip/norm, `KVW` floats per cached token, stored ROUNDED TO F16 (llama.cpp default cache type).
  SWA visits (40): ring of the last `window` (1024) tokens; global visits (8): all tokens. Each cell remembers its position.
* four conv histories, each the last 3 INPUTS of that conv (oldest first), zero at sequence start: K (1024 ch), V (1024), wo-output (2304),
  MoE-output (2304) = 6656 channels x 3 floats = 79.9 KB per visit = 3.83 MB per sequence (fp32).
Per token (transient, not state): `cur` (D), `blocks` (list of D-vectors, one per attention-residual block start), `prelude` output and its
blocks, `bank`/`bank_out` of the previous pass. Total KV bytes per token: 8 global visits x 2048 values.

## 5. The forward pass (pseudocode; one token at position `pos`)

```
tau = 1 + 0.1 * logf(max(1, (pos + 1) * (1/8192)))                      # float ops in this order; = 1.0 for pos < 8191
cur = dequant_q8_0(token_embd[tok]); cur = rms_norm(cur) * token_embd_norm
blocks = []
for visit in L0..L5:   visit(cur, blocks, msgs=none)
prelude = cur; prelude_blocks = blocks
for pass in 0..2:
    if pass > 0:
        bank = blocks[len(prelude_blocks)+1 :]          # blocks created after the first core block of the previous pass = [state entering layer 12]
        bank_out = cur                                  # output of layer 17 of the previous pass
        cur = prelude; blocks = prelude_blocks          # EVERY extra pass restarts from the prelude output
    for ip in 6..17:
        msgs = none if pass == 0 else depth_mix(states=[reversed(bank)..., bank_out], queries=blk.ip.loop_msg_query (4))   # per layer queries, same states
        visit(L=ip, cur, blocks, msgs)
for visit in L18..L23: visit(cur, blocks, msgs=none)
m  = depth_mix(states=[reversed(blocks)..., cur], queries=output_res_query (1))
xn = rms_norm(m) * output_norm
logit[v] = output.row(v) @ q8(xn)  for v in 0..200057          # padded rows 200058..201023 are never computed
logit = tanhf(logit * (1.0f/30)) * 30
```

### 5.1 visit(ip, cur, blocks, msgs)  (L = blk.ip)

```
inpSA = cur
# attention inputs
if ip > 0:   M = depth_mix(states=[reversed(blocks)..., cur], queries=L.attn_res_query, nq = SWA ? 1 : 3)
             in[j] = M[SWA ? 0 : j]  for j = 0,1,2        # global layers: separate mix per Q, K, V
else:        in[0] = in[1] = in[2] = cur                   # layer 0 has no attn_res_query
if msgs:     in[j] = control(in[j], msgs[j], L.ctrl[j])    for j = 0,1,2   # msgs[0..2] = Q,K,V messages, msgs[3] = FFN message
norms:       SWA & msgs : in[j] = rms_norm(in[j]) * attn_norm  (three separate norms)
             SWA        : in[0] = rms_norm(in[0]) * attn_norm; in[1] = in[2] = in[0]
             global     : in[0] *= rms scale * attn_q_inp_norm; in[1] ... attn_k_inp_norm; in[2] ... attn_v_inp_norm
# projections (q8() = quantise the activation to Q8_0 blocks, section 6)
Q = attn_q @ q8(in[0]);  Kp = attn_k @ q8(in[1]);  Vp = attn_v @ q8(in[2]);  R = attn_rel @ q8(in[0])
t = attn_sink_proj @ bf16(in[0]);  Gt = attn_gate @ q8(in[0])
global only:  dk = attn_k_conv_proj @ bf16(in[1]);  dv = attn_v_conv_proj @ bf16(in[2])        # 32 values each
K = convolve(Kp, attn_k_conv, dk, hist_k);  V = convolve(Vp, attn_v_conv, dv, hist_v)           # SWA: dk, dv absent
Q = clip(Q as [24][128], scale = attn_q_clip, thr = 22.6274)            # per head
K = clip(K as [8][128],  no scale,            thr = 22.6274)            # per kv head
V = rms_norm(V as [8][128])                                             # eps 1e-6, no weight
R = clip(R as [24][16],  scale = attn_rel_clip, thr = 8.0)              # per head
global: Q *= tau; R *= tau
cache[pos] = (f16(K), f16(V))                                           # post-clip K, post-norm V, rounded to F16
# attention, per head h, kv head g = h/3, over cached cells j with dist = pos - pos_j:  SWA: 0 <= dist < window;  global: all
b_j  = (dist < 1024 ? dot16(attn_rel_bias[dist][0..15], R[h]) : 0) - t[h]        # sink folded into the bias for EVERY key
b_j  = f16(b_j)                                                          # flash attention folds the bias into an F16 mask
s_j  = (Q[h] . K_j[g]) * (1/sqrtf(128)) + b_j                            # Q is NOT rounded to F16, K_j is the F16 cache value
p    = softmax over {s_j} U {0}                                          # the extra zero logit is the sink: exp(0-M) joins the denominator, no V
ao_h = sum_j p_j * V_j[g]
ao   = concat_h(ao_h) * sigmoid(Gt)                                      # elementwise, QW wide
y    = attn_output @ q8(ao)
attn_out = convolve(y, attn_post_conv, none, hist_attn)
# residual stream
if ip % 6 == 0:  blocks.push(inpSA); cur = attn_out                      # a new block starts: the old stream is stored, NOT added back
else:            cur = inpSA + attn_out
ffn_inp = cur
# MoE
m = depth_mix(states=[reversed(blocks)..., ffn_inp], queries=L.ffn_res_query, nq = 1)     # blocks already contains the push above
if msgs: m = control(m, msgs[3], L.ctrl[3])
x = rms_norm(m) * ffn_norm
moe_out  = moe(x)
ffn_out  = convolve(moe_out, ffn_post_conv, none, hist_ffn)
cur = ffn_inp + ffn_out
```

### 5.2 Helpers

```
rms_norm(v):        scale = 1.0f / sqrtf((float)(sum_double(v[i]*v[i]) / n) + eps);   y = v * scale     # weight multiply is a SEPARATE (x*scale)*w
clip(rows, scale, thr):  row *= scale (if any); n = sqrtf((float)sum_double(row^2)); n = n*1.0f + 1e-6f; f = clamp(thr / n, 0, 1); row *= f
depth_mix(S, Qs, nq):    S_s = rms_norm(state_s) (no weight, eps 1e-6);  score[s][q] = dot(S_s, Qs[q]) * (1/sqrtf(2304));
                         w[.][q] = softmax_s(score)  (float expf of score-max, DOUBLE sum, inv = (float)(1.0/sum), w = e*inv)
                         out[q] = sum_s w[s][q] * S_s     # of the NORMALISED states, not the raw ones
control(cur, msg, c):    g = c.down @ [rms_norm(cur), rms_norm(msg)];  gain = tanhf(c.gain @ g);  forget = tanhf(c.forget @ g)
                         out = (cur - forget*cur) + gain*msg
convolve(in, w, dyn, hist):   x_j = hist[j] (j<3), x_3 = in;  s = x_0*w_0; s += x_1*w_1; s += x_2*w_2; s += x_3*w_3    # NO fma, tap 0 = oldest
                         if dyn: for lag in 0..3: s = s + x_{3-lag} * dyn[head(c)][lag]      # lag 0 = newest; head(c) = c/128; dyn index head*4+lag
                         out = (in + s) * (1/sqrtf(dyn ? 9 : 5));  hist = [x_1, x_2, in]
moe(x):  logit = ffn_gate_inp @ x (F32);  p = sigmoid(logit);  sel = p + exp_probs_b;  top = 8 largest sel (descending; ties -> lower index)
         w_i = p[top_i];  w_i = w_i / ((float)sum_double(w) + 1e-20f) * 8.0f                        # unbiased probabilities, renormalised
         xq = q8(x);  for e in top:  gu = ffn_gate_up_exps[e] @ xq (2304 rows) * gate_up_scale[e] (NVFP4 layers);  g = gu[0:1152]; u = gu[1152:2304]
         a_j = (tanhf(g_j*(1/4))*4 * sigmoid(g_j)) * (tanhf(u_j*(1/25))*25)                       # SiTU, float
         d_e = ffn_down_exps[e] @ q8(a) * down_scale[e];  d_e *= w_e
         out = d_top0 + d_top1 + ... + d_top7                                                       # sequential float adds in top-k order
sigmoid(x) = 1.0f / (1.0f + expf(-x))
```

## 6. Quantisation of activations and exact dot kernels (what "tracks llama.cpp" means)

llama.cpp quantises the F32 activation of every matmul to the weight type's `vec_dot_type`:
* Q8_0 or NVFP4 weights -> activation as Q8_0 blocks (32 values): `amax = max|x|; d = amax/127 (stored fp16); id = 127/amax;
  q = rint(x*id)` (round half to even; this is the x86 AVX2 `quantize_row_q8_0`, NOT `quantize_row_q8_0_ref`, which uses `1/d` and `roundf`;
  they differ only at exact ties). Applies to: wq, wk, wv, attn_rel, attn_gate, attn_output, expert gate_up (input x),
  expert down (input = SiTU output, one quantisation per selected expert), lm head.
* BF16 weights (attn_sink_proj, attn_*_conv_proj) -> activation rounded to bf16 (round-to-nearest-even).
* F32 weights (router, controllers, rel bias, depth-mix queries) -> float activations, no quantisation.
Bit-exact summation orders reproduced in `ref_forward.cpp` (needed because the quantisers and F16 roundings turn 1-ulp differences into
visible jumps, section 9):
* Q8_0 x Q8_0: 8 float lanes, lane l = int sum of elements 4l..4l+3 of the block, `acc[l] = fmaf(d_w*d_a, (float)sum_l, acc[l])` per block,
  then `hsum_float_8` = `((a4+a0)+(a6+a2)) + ((a5+a1)+(a7+a3))`.
* NVFP4 x Q8_0: same 8 lanes; per 64-block two FMAs, first sub-blocks (0,1) against Q8_0 block 2b, then (2,3) against block 2b+1; lanes 0-3 =
  first sub-block of the pair (scale `ue4m3(d)*d_a`), lanes 4-7 = second; lane = 4 consecutive elements of the sub-block in natural order.
* BF16 (sink, dyn taps): two 16-lane accumulators over 32-element steps, products and adds NOT fused, each reduced with the
  `_mm512_reduce_add_ps` tree (`v[i]+v[i+8]`, `+4`, `+2`, `+1`), both added in DOUBLE, result cast to float.
* F32 router (measured exact at token 0 of s1; controllers: UNVERIFIED, same order assumed): llamafile tinyBLAS order = one 16-lane FMA chain over K/16 steps, then the same
  reduce tree. F32 depth-mix scores use `ggml_vec_dot_f32` order = four 16-lane FMA accumulators over 64-element steps, combined 0+2, 1+3, then
  0+1, then the reduce tree (measured: router matches tinyBLAS, mix scores match vec_dot; do not unify them).
* depth_mix output (reduction over the few states): float FMA chain in the order [newest-block-first ..., cur], starting from 0.
* ssm_conv and the dynamic taps: plain float multiply and add, NOT contracted into FMA (the file is built with `-ffp-contract=off`).
* Router sum of the 8 weights, rms_norm sums, clip sums: DOUBLE accumulation of float products, then cast.

## 7. Numerically sensitive details (checklist for the executor)

1. Q is NOT rounded to F16 before the K.Q dot (measured, contradicts the one_chunk flash-attention source: do not "fix"); K and V ARE rounded to F16 (cache);
   the relative bias INCLUDING the sink term is rounded to F16 (the sink logit t is subtracted from every key's bias before the rounding).
2. The sink is an implicit extra softmax logit of 0 after the subtraction of t: probabilities are exp(s_j - t)/(1 + sum exp(s_k - t)) with s_j from the dot + table.
3. Bias table index = distance (0 = the token itself). Distances >= 1024 (global layers only) use the zero row (bias = 0, minus t). R is scaled by tau on global layers.
4. clip adds 1e-6 to the norm after the sqrt; V is RMS-normalised AFTER the conv and has no weight; K/Q are clipped AFTER the conv; the cache holds the post-clip K.
5. Q8_0 quantisation uses id = 127/amax and half-even rounding; d is rounded to fp16 for the dequantisation but id is computed from the unrounded amax.
6. The depth mix averages the RMS-NORMALISED states; raw magnitudes survive only in `cur`/`blocks`. All softmax scales are 1/sqrtf(2304) (mix) and 1/sqrtf(128) (attention).
7. Residual: layers with phys % 6 == 0 (0, 6, 12, 18) push inpSA to `blocks` and set cur = attn_out; all others add. The FFN mix sees the block list AFTER that push.
8. Each extra pass restarts from the PRELUDE output with the prelude blocks only; the message states are the previous pass's blocks after its first core block plus its final `cur`.
9. Expert scale tensors multiply the whole gate_up result (gate and up), before SiTU; the down scale multiplies before the route weight.
10. Router: selection uses sigmoid + bias, weights use the unbiased sigmoid, renormalised over the 8 and multiplied by 8.
11. Conv tap order: static tap j=0 hits the oldest input; dynamic lag 0 hits the newest. Histories store conv INPUTS. Static scale 1/sqrtf(5), dynamic 1/sqrtf(9).
12. Final: tanhf(x*(1/30))*30 on the 200058 real rows only.
13. Token ids come straight from the tokenizer: this GGUF adds no BOS/EOS; embeddings go through token_embd_norm.

## 8. Things in llama.cpp that look odd (flagged, all reproduced as is)

* FMA contraction differs per op: `ssm_conv` is uncontracted, the AVX2 dot kernels use explicit FMA, FA accumulation uses `ggml_vec_mad` (FMA). Mixed conventions.
* F32 mat-muls take different kernels (tinyBLAS vs ggml_vec_dot_f32) depending on shape: router (128 rows) vs depth-mix scores (2-5 rows).
* The F16 mask trick makes the relative bias coarse (11 mantissa bits) and the sink term is subtracted before that rounding: large sink logits lose resolution.
* `GGML_FA_TILE_Q` source suggests Q is F16-rounded for n < 64 query rows; the measured oracle shows it is not (also for n = 2). The oracle binary may differ from the checkout.
* Layer 0 has no `attn_res_query`; first-in-block layers drop the residual; `ggml_argsort_top_k` tie order is unspecified.
* The `.scale` tensors only exist for NVFP4 layers; layers 20-23 are Q8_0 (the original checkpoint is MXFP8 there, which this GGUF converted).
* `expert_weights_norm_eps` (1e-20) is lost in float for any realistic sum; harmless.
* Long contexts: decode with n_kv >= 512 uses the split-KV flash-attention path (different summation order); not exercised here.

## 9. Validation (measured, load average 3-6 on a shared host; the reference runs ~1 s/token with 8 threads)

Oracle = `tools/oracle/*` (agent E): whole prompt as ONE batch, default llama.cpp context (F16 KV, flash-attn auto), 8 threads.
`build-core/core/test_ref_vs_oracle` (INK_ORACLE_EXTENDED=1 INK_ORACLE_ALL=1), final code:

| prompt | tokens | argmax (llama / ref) | top-10 overlap | cosine | max abs diff | rel L2 |
|---|---|---|---|---|---|---|
| s1_en_capital | 13 | 154328 / 154328 | 10 | 0.9998656 | 0.249 | 0.0164 |
| s2_code_add | 14 | 68014 / 68014 | 10 | 0.9999352 | 0.274 | 0.0114 |
| s3_en_story | 15 | 149878 / 149878 | 10 | 0.9991443 | 0.911 | 0.0414 |
| c1_chat_user | 14 | 200008 / 200008 | 9 | 0.9998934 | 1.484 | 0.0278 |
| m1_en_llm | 43 | 151787 / 151787 | 10 | 0.9988543 | 0.934 | 0.0479 |
| m2_code_c | 52 | 407 / 407 | 9 | 0.9940577 | 2.645 | 0.109 |

Per position on s1 (logits_all.f32): argmax matches at all 13 positions, worst cosine 0.99945 (position 8).

Layer-by-layer (tests/core/dump_llama_acts.cpp + compare_acts.cpp: llama.cpp eval callback dumps `Qcur, Kcur, Vcur, Rcur, ffn_inp, ffn_out,
l_out, ffn_moe_*, norm, ffn_moe_topk` per slot; compared with the reference traces): with the arithmetic of section 6, visits 0-5 are
bit-identical (relative L2 exactly 0 for Qcur, Rcur, ffn_inp, ffn_moe_gate_up_scaled, ffn_moe_situ, ffn_moe_down_scaled; Kcur/Vcur 5e-8 on the
global layer 5 whose dynamic-tap arithmetic is not fully bit-matched) for every token up to the first discrete flip (s1: tokens 0-4 are exact through layer 5; at
token 5 the input of layer 4 already differs by ~1e-3, i.e. a flip happened in layers 0-3 at token 5, and everything after it differs). A flip is a Q8_0 re-quantisation of an activation element, an F16 rounding, or a different top-8 set after a 1e-7 input
change; one flipped element of the MoE input changes all expert outputs by ~4e-4 relative. Noise-floor measurement (before the exactness work,
with all roundings emulated): perturbing the token embedding by a RELATIVE 1e-7 moves the s1 cosine to the oracle from 0.99986 to 0.99964, i.e. any two
non-bit-identical implementations of this network are about 1e-4 apart in (1 - cosine) on 13 tokens and further apart on longer prompts. Expert
sets differ in 82 of 624 token/visit pairs on s1, all downstream of the first flip.

Sensitivity of the s1 cosine (before the exactness work) to the emulated roundings: all on 0.99986; KV not rounded 0.99934; bias not rounded 0.99936;
Q rounded to F16 0.99986 -> but layer-0 attention error 4e-4 (so Q rounding is wrong at the tensor level); float activations (act_quant_like_ggml=false) 0.99984.

NOT validated / gaps (be explicit): (a) sliding window 1024 and the ring wrap-around: every oracle prompt is <= 52 tokens. The requested window-16 comparison
is IMPOSSIBLE with this GGUF: `oracle_logits --override-kv inkling2_loop.attention.sliding_window=int:16` fails at load with
`check_tensor_dims: tensor 'blk.0.attn_rel_bias.weight' has wrong shape; expected 16, 16, got 16, 1024` (llama.cpp sizes the SWA bias table by n_swa). The
reference supports the override (`RefOptions::sliding_window`, tested for ring size, newest-cell retention and "window > prompt == default" in
test_ref_state); the semantics follow llama.cpp's `is_masked_swa` (masked iff pos_q - pos_k >= window) and the relative-pos input (index = distance if < extent). (b) tau > 1
(pos >= 8191) and distances >= 1024 on global layers (zero bias row). (c) The single-token decode path of llama.cpp (the oracle is a batch; split-KV attention for
n_kv >= 512). (d) Passes 1 and 2 are validated end-to-end only (controllers/messages: their F32 dot order is unverified at tensor level). (e) The non-act-quant mode
is a convenience, not tuned. (f) KV memory beyond a few hundred tokens (state vectors are float, 4x the F16 cache).

## 10. Notes for the executor author

* Per visit the matmuls are (K -> N, type): wq 2304->3072, wk 2304->1024, wv 2304->1024, wo 3072->2304, attn_gate 2304->3072, attn_rel 2304->384 (all Q8_0);
  sink 2304->24 (BF16); global: 2 x 2304->32 (BF16); router 2304->128 (F32); experts: 8 x (2304->2304 gate_up, 1152->2304 down) NVFP4 or Q8_0;
  passes 1,2: 4 controllers (4608->64, 64->2304 twice, F32). Head: 2304->200058 Q8_0 once per token. The three activation vectors in[0..2] are identical on SWA layers
  without messages (quantise once). Everything else (norms, clips, mixes, convs, softmax) is O(D) per visit.
* The depth mix touches at most 5 states; the message mix of a pass uses the same states for all 12 layers (only the 4 queries differ): the normalised states can be shared.
* Parity target against llama.cpp: argmax + top-10 + cosine >= 0.999 per prompt; do not expect 0.9999 unless you copy the dot kernels of section 6 (then layers
  0-5 become bit-exact, the rest still depends on flips).
* Reproduce: `export PATH=/usr/bin:/bin; cmake -S src/core/standalone -B build-core -G Ninja; cmake --build build-core;`
  `build-core/core/test_gguf; OMP_NUM_THREADS=8 build-core/core/test_ref_state; INK_ORACLE_EXTENDED=1 INK_ORACLE_ALL=1 build-core/core/test_ref_vs_oracle`;
  layer-by-layer: `build-core/core/dump_llama_acts -m GGUF --tokens-file ids.txt --out DIR` then `compare_acts GGUF DIR tokens.i32 [llama_slot [token]]`.

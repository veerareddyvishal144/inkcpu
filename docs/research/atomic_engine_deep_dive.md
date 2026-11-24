# Atomic Engine deep dive (branch `codex/expert-weight-source`, e110381) for inkcpu

Research agent, 2026-11-24. Read-only review of /data/inkling-mini/research/atomic-engine-ews (single squashed commit, 2026-11-23). Builds on
`docs/research/atomic_engine_firstpass.md` (orchestrator notes on `main`). Nothing was built or run. Question: what in this fork makes OUR engine
(CPU-only, RAM-resident, 1x Xeon 8171M, decode ~16 tok/s bandwidth-bound, prefill ~115 tok/s ALU-bound, DSpark + session-lookup speculation) faster or better.

**Headline.** The branch is a Metal + SSD-streaming fork of llama.cpp whose Inkling work is (a) the upstream DSpark graph with support for a second GGUF
serialisation of the same drafter, (b) a very conservative prompt-lookup drafter (32-token exact suffix), (c) seven fused Metal kernels for the
Inkling target, (d) a resident-bank / expert-cache budget for streaming the MoE from SSD, and (e) one benchmark whose decode numbers are measured
on a fully repetitive fixture (100 % draft acceptance). Our engine already has the equivalent of (a), a richer version of (b), and the CPU analogue
of most of (c). The things worth taking are small: two verification checks on the AtomicChat drafter (per-slot mask embedding, W2 = our fold),
one scheduling/kernel idea for small verify widths (dequantise-once multi-row expert GEMV), two negative results on draft-length policies that
confirm our EWMA design, and the fixture itself as a copy-mode benchmark. Nothing here changes the picture of RESEARCH_WAVE2.md.

---

## 1. The "Markov context" head (priority 1)

### 1.1 What the AtomicChat GGUF is, as the code reads it
`src/models/inkling2-dspark.cpp:12-54` (`load_hparams_inkling2`) and `:56-119` (`load_tensors_inkling2`). The fork supports two serialisations
of the same 3-block DSpark drafter, selected by the presence of `markov_w2.weight` (`projected` flag, line 68; detection also in
`src/models/dflash.cpp:42-44`: `dspark_hidden_out.weight` OR {`markov_ctx_mix.weight`, `blk.0.ctx_mix.weight`, `markov_w2.bias`, `token_embd_norm.weight`}).

| role | Unsloth GGUF (what we run) | AtomicChat GGUF (`projected`) | code |
|---|---|---|---|
| tap mixing for drafter layer il | `softmax(dspark_fuse[il, :])` over 4 taps, then RMS with `blk.il.dflare_norm` | `blk.il.ctx_mix [4]` used RAW (plain `ggml_mul_mat`, NO softmax), then RMS with `blk.il.ctx_norm` | `:197-209` |
| tap mixing for the Markov head (`f_rnn`) | `softmax(dspark_fuse[3, :])`, RMS with `dspark_norm` | `markov_ctx_mix [4]` raw, RMS with `markov_ctx_norm [2304]` | `:198-202`, `:238` |
| vocab bias of chained position j | `lm_head . (hidden_out . lat + b)` (two lm_head matvecs; or F16 `dspark_folded_output` sidecar built by `tools/expert-source/fold-inkling-dspark.py`) | `markov_w2 [256 x 200058] . lat + markov_w2.bias [200058]` (one 256-wide matvec over the TRUE vocab) | `:137-155` |
| W1 (prev-token table) | `markov_w1 [201024 padded x 256]` | `markov_w1 [200058 x 256]` (unpadded; `n_vocab_padded := 200058`, line 20) | `:15-24`, `:81` |
| mask slots | one `mask_embd [2304]` repeated for slots 1..7 | `mask_embd [2304 x 7]`: a DISTINCT vector per mask slot (`ggml_view_2d(mask, n_embd, block-1)`) | `:70`, `:263-270` |
| anchor embedding norm | target's `token_embd_norm` | drafter's own `token_embd_norm.weight` copy | `:71`, `:258-259` |
| SWA window | `n_swa += 1` (anchor row counted) | `n_swa` as stored (file already includes the anchor) | `:49-50` |

Head math per chained position j (both formats, `build_inkling2_head`, `:123-163`), with `r = 256`, `H = 2304`, `V = 200058`:
```
z      = [ s ; W1[prev] ; hid_j ; f_rnn ]                      # 2r + 2H = 5120
u      = joint_proj . z + b_joint                              # 3r = 768 = [gate | cand | out]
g      = sigmoid(u[0:r]);  c = tanh(u[r:2r]);  s' = c + g * (s - c)          # == g*s + (1-g)*c
lat    = tanh(u[2r:3r])
bias_v = W2[v] . lat + b2[v]                                   # AtomicChat: W2 = markov_w2, b2 = markov_w2.bias
       = lm_head[v] . (hidden_out . lat + b_hidden)            # Unsloth: two lm_head products (or the F16 folded sidecar)
logit_v = 30 * tanh( (base_v + bias_v) / 30 ),  base_v = lm_head[v] . hid_j
d_j    = argmax_v logit_v  (first maximum: ggml_pool_1d max + ggml_step + arange tie-break, lines 323-327; host-side sampler is temp 0, speculative.cpp:1441-1450)
```
So `markov_w2` is exactly our fold `M = lm_head . hidden_out` and `markov_w2.bias` is our `m0 = lm_head . hidden_out.bias` (docs/DRAFTER.md section 2,
fold.cpp), stored BF16 over the unpadded vocab (102.4 MB, same as our `--bf16` variant; we default to Q8_0, 54.4 MB).

**Is it retrained or a re-export?** The repo contains no converter for this format (`conversion/inkling2.py:188` skips `markov_head`; the C++ header
comment, line 13-14, attributes it to "PR #32218 at e4b1cb5318f1", i.e. a later revision of the same upstream PR we built from at `0a9105de`).
Evidence that it is a re-export of the same checkpoint rather than new weights: (1) tensor count 58 = 54 - {hidden_out.w, hidden_out.b, fuse, dspark_norm, mask_embd}
+ {markov_w2.w, markov_w2.b, markov_ctx_mix, markov_ctx_norm, 3 x ctx_mix, token_embd_norm, mask_embd[2304x7]}; the per-layer `dflare_norm` are
renamed `ctx_norm`; (2) the raw-mix graph is equivalent to the softmax graph iff `ctx_mix = softmax(fuse row)`, which a converter can precompute;
(3) README.md:117-125: on the handover fixture (DSpark 4/32, 14 GiB) AtomicChat averaged 72.34 tok/s vs 70.74 for the target-folded Unsloth
draft, and 66.25 vs 67.34 on archive 7/32 at 8 GiB: a 2 % difference either way, i.e. the same acceptance (a retrained head with better
acceptance would separate more). Greedy output IDs are identical by construction (verification), so "both drafts produced identical IDs" proves nothing.
UNVERIFIED without the weights; see check C1 below.

Two details that differ from our implementation and ARE worth a check (both cheap, header/weights only, no engine change):
* **C1 (W2 vs fold).** Dequantise `markov_w2` from the AtomicChat file and compare row-wise with our `M` (bf16 fold through the Q8_0 lm_head): if
  cosine ~ 1 and `markov_ctx_mix == softmax(fuse_weight[3])`, the AtomicChat drafter is our drafter and there is nothing to learn from it. If W2
  differs materially, it was folded through the bf16 HF lm_head (closer to training) and we could switch our fold source; acceptance effect at most
  the 1.6 % of positions where our Q8_0 chain disagrees with the exact reference (DRAFTER.md section 4).
* **C2 (per-slot mask embedding).** `mask_embd` is `[2304, 7]` in the AtomicChat file. The HF checkpoint has one `mask_embed [2304]` (DSPARK.md 2.2) and we
  broadcast it. If the 7 columns are NOT identical, upstream's later revision models a per-slot mask (e.g. mask + learned slot offset) and our block
  input differs from the reference for slots 2..7, which would lower acceptance at positions >= 2 (our acc@2/acc@1 ratio is 0.64-0.73). One `gguf-py`
  header+tensor read answers it. Effect if real: unknown, potentially several points of acc@j>=2; effort 1 hour to check, 1 day to adopt.

### 1.2 Cost of the head in their engine vs ours
Their graph runs the chain in-graph with `n` sequential `build_inkling2_head` calls (`:320-328`), each a `[256 x 200058]` BF16 matvec (102 MB) + a
200058-wide add; on Metal the add is dispatched with the row-broadcast binary kernel when the node carries the `GGML_METAL_INKLING_VOCAB_ADD` hint
(`ggml-metal-device.cpp:2160-2162`, env `LLAMA_EXPERIMENTAL_INKLING_W2_PARALLEL_ADD`, `inkling2-dspark.cpp:312-318`): a Metal dispatch-shape tweak,
no measured number in the repo, nothing for a CPU. The base lm_head over all `block` rows is one batched matmul (`:293`). They do NOT split
`joint_proj` into chain-independent and chain-dependent columns (we do, DRAFTER.md phase 8/9: 4608 of 5120 columns batched). No draft-time
measurements for Inkling exist in the docs (only end-to-end tok/s).

### 1.3 Measured acceptance / speed data for Inkling in the repo
All from README.md:85-157 and `docs/atomic-engine-inkling-low-bit-metal.json` (M4 Max 48 GB, 4000-token prompt, 64 greedy tokens, Q4_0 KV):
* **Archive fixture = copy mode.** `tools/expert-source/bench-inkling-metal.py:64-71`: the prompt is `docs/atomic-engine-qwen38-context.txt` repeated
  twice and cropped to 4000 tokens, so the continuation is a verbatim copy of text already in context. Acceptance was **55/55 at n=7 (8 rounds for 64
  tokens) and 47/47 at n=3** at every budget (`draft_n_accepted == draft_n` in every JSON row). All decode numbers in the matrix (Atomic 68-109 tok/s,
  Strata 113) are therefore pure copy-mode numbers: tokens per round = 8 (n=7) or 4 (n=3). They say nothing about DSpark quality.
* **Handover fixture** (`tools/expert-source/inkling-handover-4000.json`: a briefing task over a repetitive batch log): README.md:150-152, folded
  draft, IQ2_M, 14 GiB, DSpark 7/32: **47 of 105 proposals accepted in 16 rounds** = 2.94 accepted per round, **L = 4.0 tokens per verify at n=7**,
  53.9 tok/s (vs 105.5 on the copy fixture). Our pooled greedy L at n=7 is 2.77 (DSPARK.md 9.2); this prompt is structured/repetitive, in line with
  our code/tool-call rows (4.4-5.9). At 4/32 (n=4) the same prompt gave 72.34 tok/s vs 53.9 at 7/32, i.e. on their hardware n=4 beat n=7 by 34 %
  on a prompt with L(7) = 4.0: consistent with our finding that T=8 only pays on code.
* Default profiles chosen by them: n=7 for IQ2_M (cheapest verify), **n=3 for IQ3_S and Q4_K** (`bench-inkling-metal.py` DEFAULT_DEPTHS; release.md:36-50).
* No per-position acceptance, no temperature>0 data, no KL/perplexity for any Inkling quant or KV type anywhere in docs/ (grep `kld|perplex` finds only
  Qwen/LFM docs).

### 1.4 DFlash depth / confidence / feedback experiments (Flash Next, NOT Inkling)
`docs/atomic-engine-flash-next-dflash-*.json` are Qwen3.8-Flash-Next with a DFlash2 drafter (selector lattice, not DSpark), 8 GiB SSD-streamed on M4 Max,
912-token handover prompt, 64 tokens, five alternating pairs each. Still informative as policy evidence (acceptance fraction ~0.53 at depth 3, 1.58 accepted/round):

| experiment | candidate | control | result |
|---|---|---|---|
| depth2-vs3 | depth 2: 14.63 tok/s, 35/56 accepted | depth 3: 15.18, 38/72 | depth 3 wins by 3.8 % |
| depth3-vs4 | depth 4: 13.74, 42/83 | depth 3: 15.41, 38/72 | depth 4 loses 11 % (verify grows faster than acceptance) |
| confidence (gate the 3rd proposal on selector conditional prob >= 0.5 / 0.625 / 0.75) | 14.78 / 14.42 / 14.42 | 14.80 / 14.83 / 14.73 | no gain at any threshold (saves 3-5 rows per request, same accepted count) |
| feedback (depth 3 -> 2 after a zero-accept round, back to 3 after a full-accept round) | 13.72, 35/72 | 14.89, 38/72 | **reactive depth policy loses 8 %** |
| prime-wave / ready8 / next-wave / whole-expert (SSD read scheduling) | +0.5 % / +7 % / +8.7 % / -3 % | | SSD-overlap only |

Transfer: the "feedback" result is the one to remember. A one-step reactive width policy was worse than a fixed width; our policy (SPEC_DESIGN.md 7)
uses per-position EWMA acceptance + measured costs with hysteresis, which is the principled version. Confidence gating without a trained confidence
head (they used the selector's conditional probability) gave nothing: do not build a heuristic confidence gate for DSpark either (Inkling ships no head).

---

## 2. Prompt lookup (priority 2)
`common/speculative.cpp:1455-1476` (`draft_from_prompt`), enabled by `LLAMA_EXPERIMENTAL_INKLING_PROMPT_LOOKUP=1`, only for the Inkling DSpark
drafter with one sequence (`:1400-1404`), called before drafting (`:1519`):
* match: the last **32** tokens of the committed history (31 + `id_last`) must match EXACTLY an earlier occurrence; search backwards from the end
  (most recent occurrence wins) within the last **8192** tokens; greedy only (`dp.temp > 0` disables), no `result_q`.
* proposal: `count = min(dp.n_max, params.n_max)` = the DSpark depth (7 / 4 / 3), the tokens following the match; if found, DSpark is skipped for the round.
* no acceptance tracking, no length adaptation, no arbitration, no probe; nothing is fed back.
* no acceptance data for it in docs (README.md:56-60 only describes it).
Compared with ours (SPEC_DESIGN.md 11): we fire at >= 8-token matches with per-match-length-bucket acceptance EWMAs, width 2..8 chosen by predicted
speed-up, probing, and arbitration against DSpark's predicted speed-up; measured +9.3 % pooled over max(plain, dspark) on copy-heavy prompts and
0 mismatches. Theirs is a strict subset. Only idea to note: their 32-token key means they never fire on short coincidences, which is roughly what our
[32+) bucket does with 160/161 and 238/238 accepted rows (L.md numbers in SPEC_DESIGN 11.5). **Nothing to adopt.**

---

## 3. Numerics they ship (priority 3)
* **Q4_0 KV for Inkling** (`-ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0`, README.md:43, release.md:47; also for the drafter's KV). No quality measurement in
  the repo. Their only correctness gate is "Atomic's 64 output IDs == stock llama.cpp's 64 IDs with the SAME Q4_0 KV" on a copy fixture (bench script
  `matches_stock`), which cannot detect KV damage. The fused prefill attention kernel hard-codes Q4_0 KV (`fa_aux.metal:600-611`, `block_q4_0`
  dequant in `inkling_stage_kv`), so the choice is a Metal implementation convenience. For us this remains the RESEARCH_WAVE2 item 9 precision trade-off
  (Q8_0 V first), owner's decision; no new evidence here. Note Strata used FP16 KV and was fastest, so KV type is not where their speed comes from.
* **Fused prefill attention** (`LLAMA_EXPERIMENTAL_INKLING_PREFILL_ATTN`, `inkling2-loop.cpp:121-143`, `llama-graph.cpp:3953-3961`,
  `fa_aux.metal:613-690`): one Metal kernel for prompt blocks of 128..1024 tokens (multiple of 16), 16 queries x 64-key tiles, Q4_0 KV staged to
  threadgroup memory, rel-bias table gathered by cached distance map, per-query sink shift, online softmax, skipping fully masked tiles. Measured
  effect (JSON `comparisons.fused_attention_off`): TTFT 8.21 s vs 10.91 s at 8 GiB (-25 %), 7.24 vs 10.27 s at 12 GiB; decode unchanged. It fuses
  exactly what our CPU prefill attention already fuses (rel-bias table + sinks + mask in one pass, ENGINE_DESIGN.md "fused rel-bias table + 16-key vector
  path"); our prefill is ALU-bound in the MoE GEMMs, not attention. **No transfer.**
* `kernel_inkling_rel_mask` (`fa_aux.metal:581-595`): builds `kq_b = table[distance] - sink + causal` directly instead of permute/get_rows/permute. We already
  read the repacked rel-bias table in-kernel. No transfer.
* **Chat template / parser** (`common/parsers/inkling.cpp:1-165`, `common/chat.cpp:1468-1471`): PEG parser for `<|message_model|>` blocks with
  `<|content_thinking|>` / `<|content_text|>` / `NAME<|content_invoke_tool_json|>{"name":..,"args":..}` closed by `<|end_message|>`; media-part splitting;
  lazy grammar trigger `<|message_model|>([^<])` for tool calls; continuation handling for prefilled reasoning. This is upstream llama.cpp (PR 25731) and
  matches ARCH.md section 4 as used by our prompts; nothing Inkling-specific added by the fork. Worth a 10-minute diff against our server's template
  for the continuation case (`generation_prompt = HEADER + THINK + reasoning [+ END + HEADER + TEXT + content]`, lines 81-96) only if we support
  `continue_final_message`.
* Rollback: `llama-context.cpp:197-202`: Inkling DSpark targets use recurrent-state snapshots (`n_rs_seq < 8`) and allow `ubatch` up to 1024 for prefill.
  Equivalent to our conv-history record (SPEC_DESIGN.md 4). No transfer.

---

## 4. Expert executor / store / budget: scheduling ideas for a RAM-resident CPU MoE (priority 4)
What the code is: `src/llama-expert-executor.cpp` (2074 lines) streams expert projections from the GGUF on SSD into a budgeted cache and runs them on
Metal via custom ops; `src/llama-expert-store.cpp` is the cache (LRU or 8-class decayed-frequency "class-LRU" from upstream PR 31233);
`common/atomic-budget.*` is the memory ledger; `src/llama-inkling-core-banks.h` keeps whole physical layers resident ("banks"), repeated-core first.

| idea | where | what | applies to inkcpu? |
|---|---|---|---|
| resident core first | `llama-inkling-core-banks.h:28-60`, `:130-145` ("Prioritize the repeated core, then the suffix and finally the prefix") | with a partial budget, keep the 12 physical core layers (visited 3x per token) resident, stream the 12 outer layers | no: everything is RAM-resident for us. The ordering fact is already in our bytes model (ROUTING.md) |
| `inkling-loop-age` eviction policy | `llama-expert-store.cpp:176-181`, `:400-404`, `:1301-1310` | class-LRU with half-life 512 instead of 128 because core layers advance their clock 3x per token; age normalised by sqrt(clock) for 4-row verifies | no (no cache to evict). Confirms: expert reuse across the three passes is weak enough that they needed a longer half-life, in line with RESEARCH_WAVE2 "ruled out: expert caching" |
| route index | `set_route_index` `:1894-1914`, docs/atomic-engine-route-index.md | per projection, linked lists of (row, slot) per expert + ascending active-expert list, built once per projection, 22.5 KB | we already group verify/prefill rows by expert (GEMM executor). no |
| early reads / next-layer preview / prime wave | `set_early_read` `:1915-1935`, `set_preview` `:2021-2074`, docs/atomic-engine-early-reads.md, atomic-engine-preview.md | overlap SSD reads of the next projection's experts with the current GPU group; a shadow-router prediction of next-layer experts failed their numerical gate and is off by default; early reads: +8.7 % in one five-pair run, "not established" in another | no (no I/O to overlap; our DRAM streaming already runs at 52-68 of 80 GB/s and the next visit's experts are unknown until its router runs). Their failed shadow-router trial is one more data point against routing prediction |
| grouped short-batch expert GEMV | `ggml-metal-inkling-groups.cpp.inc:1-62`, `mul_mm.metal:1022-1120` | for verify batches of n = 2..5 rows: a tiny plan kernel maps each (row, slot) route to a leader per expert (up to 5 rows packed in 6-bit fields), then one dispatch per (expert row-block, route) where the leader dequantises the expert's weight row ONCE and keeps 1..5 register accumulators, one per row sharing that expert (`inkling_group_dot<count>`), writing each row's output slot | **maybe**: this is the "multix GEMV" of RESEARCH_WAVE2 item 12 applied to verify. Our verify uses the batched prefill GEMM executor (dequantise once per tile). Worth checking that at T = 2..5 the GEMM tile is not the loser (token-count-aware N tiles are listed as a known waste in RESEARCH_WAVE2 item 4); a count-specialised multi-row NVFP4 GEMV (1..5 activation rows per expert, weights read once) is the natural small-T kernel. Note they EXCLUDE Q4_K gate/up from it ("preserve greedy verification on handover", line 22): accumulation order changed outputs |
| fused FFN custom op | `build_inkling_ffn` `:1278-1306`, `llama-graph.cpp:2627-2637` | gate_up + SiTU + down for the routed experts in one custom op, only in the streamed Metal path (env-gated, `LLAMA_EXPERIMENTAL_INKLING_EXPERT_FFN`) | we already run gate/up -> SiTU -> down per expert tile; our remaining item is the down-phase call structure (RESEARCH_WAVE2 item 8), not fusion |
| compact decode scheduler | `llama-context.cpp:932`, `:961-962`, `:2499-2505`, `:3009-3016` | reserve a 64-row decode graph after a wide prompt graph, re-reserve on the next wide batch | Metal scheduler allocation detail. no |
| budget ledger / pressure monitor | `common/atomic-budget.cpp`, `llama-context-atomic.cpp:297-330` | live memory budget changes (20 -> 8 GiB) | no |

Bottom line for priority 4: the only transferable item is the dequantise-once multi-row GEMV for T <= 5 verify rows. It is worth a measurement: our
measured verify(T)/plain is 1.45 / 2.15 / 3.43 at T = 2 / 4 / 8, while the total-bytes floor (ROUTING.md expert-union bytes + shared tensors once,
i.e. DSPARK.md break-even table S(T) minus the 1125 MB drafter term, over B1 = 3862 MB) is 1.28 / 1.70 / 2.30. Verify therefore runs 13 % / 26 % / 49 %
above its bandwidth floor at every width, so a kernel that decodes each expert's weights once and applies them to the 1..5 rows routed to it (instead of
GEMM tiles sized for 9+ rows) could recover part of the gap at small T, where most of our auto-policy steps live.

---

## 5. The M4 Max benchmark: Atomic 77 vs Strata 113 (priority 5)
README.md:85-101 and the JSON. Facts that explain the table:
1. **Copy fixture, 100 % acceptance** (section 1.3). Tokens per verify round = n + 1 exactly. Decode tok/s = 64 / (rounds x (draft + verify)).
2. **Atomic Q4_K ran DSpark 3/32 (n = 3, 4 tokens per round); IQ2_M ran 7/32 (8 per round).** IQ2_M at 12-20 GiB reaches 105-109 tok/s, i.e. the same
   level as Strata's 113. The Q4_K column (41-77 tok/s) is low mostly because of half the tokens per round plus 2x the expert bytes per verify, and
   because the grouped-expert Metal kernel is disabled for Q4_K gate/up (`ggml-metal-inkling-groups.cpp.inc:22`), so Q4_K verify runs the stock
   `mul_mat_id` path. The budget rows (41 -> 77 tok/s from 8 to 20 GiB) are SSD streaming (`expert_logical_read_bytes` 84 GB at 8 GiB -> 0 at 18 GiB for IQ3_S).
3. **Strata** (`strata-inkling` 0.1.65, "native SQ-10 pack, DSpark and FP16 KV") is a separate binary; the repo contains nothing about SQ-10 (grep finds only the
   table), its draft depth, or its kernels. Its 4.21 s TTFT vs 7-10 s is a prefill-kernel difference on Metal (they run 1024-token ubatches through ggml).
   With 100 % acceptance, 113 tok/s = 64 tokens in 0.57 s; at depth 7 that is 8 rounds of ~70 ms; at depth 3, 16 rounds of ~35 ms.
4. Stock llama.cpp (33 tok/s, no DSpark) is plain decode: 30 ms/token for ~3.9 GB of expert+dense bytes at IQ2_M/Q4_K on 546 GB/s = bandwidth-bound like us.
**Transfer to x86: none.** The ratios are speculation depth on a copy task plus Metal kernel coverage. The useful artefact is the fixture idea: a doubled
document is a clean copy-mode benchmark where our lookup drafter should hit ~8 tokens per verify (T = 8 at 100 % acceptance = 3.43x plain = ~55 tok/s
on our box by the verify table; DSpark alone would give the same L but pays 30 % draft cost). Worth adding as `bench_spec` prompt `lookup_copy` for
regression tracking (effort: 30 minutes).

---

## 6. Other Inkling-specific code (priority 6)
* `inkling2-loop.cpp:82-118`: four env-gated fused Metal kernels on the <= 8-token decode/verify graph: `control` (`unary.metal:3-13`: `(x - tanh(forget)*x) + tanh(gain)*msg`
  in one pass), `clip` (`fa_aux.metal:5`: per-row L2 clip with optional per-channel scale), `conv_residual` (`conv.metal:3-30`: 4-tap short conv + dynamic
  per-head taps + residual + scale in one pass), `rel_mask` (section 3). Plus `situ` (`unary.metal:15-34`: `beta*tanh(gate/beta) * sigmoid(gate) * [linear*tanh(up/linear)]`
  fused with the gate_up split). All are elementwise fusions of ggml op chains. Our engine already fuses these on the CPU (ENGINE_DESIGN.md: controllers "gain+forget rows
  fused with `ctrl_apply`", clips, short conv, SiTU, depth mix in agent B2's kernels; controller matrices stored BF16). **No transfer**; our ~7 ms fixed
  per-token cost is barriers (528) and serial depth mixes, for which the fork has no idea (its `mix` is plain ggml concat/rms_norm/mul_mat/soft_max, `inkling2.cpp:181-196`).
* `llama-graph.cpp:1906-1908`, `:2627-2637`: route expert ops to the Metal "window" backend for prompts >= 512 rows. Metal plumbing.
* `llama-batch.cpp:1214-1223`: `n_embd_state` = tap width for the DSpark drafter (per-token `f_rnn` injected as batch state). Same as our `commit` API.
* `llama-context.cpp:1080-1089`: reserve the 8-row block graph AND the injection graph separately (full-vocab logits only on the block path). We do not have graph reservation.
* `llama-context-atomic.cpp:117-130`: at draft load they verify the drafter's vocab text equals the target's token for token ("the draft indexes its Markov weights with the
  target's token ids") and that `target_layer_ids == {5,23,41,48}`. We check softcap/eps/layers/block (DRAFTER.md 5); adding the vocab-identity check is a cheap
  safety item if we ever accept third-party drafter GGUFs (effort: 20 lines).
* `docs/speculative.md:109-149`: generic DSpark text; "Currently only drafts with a Qwen3 backbone are supported" is stale relative to the Inkling graph.

---

## 7. Ranked recommendations (effect on inkcpu; all exact unless marked)
1. **Check C2: dump `mask_embd [2304 x 7]` of the AtomicChat GGUF; if the 7 columns differ, port per-slot mask embeddings into `ink::Drafter` and re-run `eval_acc`.**
   Only item with a plausible acceptance upside (positions >= 2). Effort 1 h to check, 1 day to adopt. Output-identical (drafter only).
2. **Check C1: compare `markov_w2`/`markov_w2.bias`/`markov_ctx_mix` with our fold `M`/`m0`/`softmax(fuse[3])`.** Settles "retrained vs re-export"; if W2 was folded
   through the bf16 lm_head and differs, fold ours the same way (cosine/ top-1 agreement on `eval_acc --fold-compare`). Effort 1-2 h. Expected effect <= 1.6 % of chain positions.
3. **Add the doubled-document copy fixture (`bench_spec lookup_copy`, 4000-token prompt = a text repeated twice) as a standing benchmark** for the lookup drafter and
   the T = 8 verify path (expected ~3.4x plain). Effort 30 min. Gives us a direct comparable to the published 105-113 tok/s numbers (ours would be ~55 tok/s at 16 threads).
4. **Keep the EWMA width policy; do not add reactive or heuristic-confidence gating.** The fork's own A/B data (section 1.4): reactive depth 3<->2 lost 8 %, selector-probability
   gating of the last row gained 0 %, depth 4 lost 11 % vs depth 3 at 53 % acceptance. Effort 0.
5. **Measure verify(T) for T = 2..5 against the bytes model per phase (gate_up, down) before touching kernels; if the GEMM path is tile-wasting at <= 5 rows per expert,
   add a count-specialised multi-row NVFP4 GEMV (weights decoded once, 1..5 activation rows) for the verify executor**, modelled on `inkling_group_dot<count>`.
   Also the kernel RESEARCH_WAVE2 item 12 (batch 2-4) needs. Effort 1 day measure, 3-5 days kernel. Upper bound = the gap between measured verify and its
   total-bytes floor: 1.45 -> 1.28 (T=2), 2.15 -> 1.70 (T=4), 3.43 -> 2.30 (T=8); realistic 5-15 % of a speculative step if the small-row GEMM is the cause
   (the gap may instead be the ALU-bound dequant chain of RESEARCH_WAVE2 item 4, which the int16 prefill chain addresses). Accumulation-order change: needs
   the parity gate (they excluded Q4_K gate/up for exactly this reason).
6. **Add the drafter/target vocabulary identity check at `ink_spec_load`** (token text per id, as `llama-context-atomic.cpp:126-129`). Safety only. Effort 1 h.
7. **Use their handover fixture as a second acceptance prompt** (`tools/expert-source/inkling-handover-4000.json`: 4000 tokens, their L(7) = 4.0): a realistic
   "summarise a repetitive log" task between our prose (1.5) and code (5-6) regimes. Effort 15 min (copy the token ids; read-only use of the file is allowed).
8. **Diff their Inkling PEG parser's continuation branch against our server template handling** (`common/parsers/inkling.cpp:81-96`) if/when we support prefilled
   reasoning continuation. Effort 30 min. Correctness, not speed.

## 8. What does not transfer, and why
* Everything SSD/Metal: expert store (LRU / class-LRU / `inkling-loop-age`), resident core banks, route index, early reads, preview/shadow router, prime wave,
  ready-8/ready-16 verify gating, whole-expert reads, budget ledger and pressure monitor, compact scheduler, Metal window/bridge backends. Our experts are in RAM;
  there is no read latency to hide and the next visit's routing is unknown until its router runs (their own shadow-router attempt failed its numerical gate).
* Fused Metal kernels for control / clip / conv-residual / SiTU / rel-mask / prefill attention / vocab add: we already have the CPU fusions; the remaining fixed
  cost is barriers and serial depth mixes, for which the fork offers nothing (its depth mix is unfused ggml).
* Prompt lookup (32-token exact suffix, fixed depth, greedy only, no feedback): a strict subset of our lookup drafter.
* Q4_0 KV: no quality evidence in the repo; it is a Metal kernel convenience there. Stays our RESEARCH_WAVE2 item 9 (precision trade-off, owner's call).
* The M4 Max decode numbers (Atomic 105-109 at IQ2_M n=7, Strata 113, Atomic Q4_K 77 at n=3): copy-mode fixture at 100 % acceptance plus draft depth and
  Metal kernel coverage; not a kernel or format insight for x86. Strata's SQ-10 pack is undocumented in the repo.
* DSpark graph itself: same math as ours (anchor + 7 masks, non-causal block, GRU chain, folded W2); the only differences are the two checks above and that
  they run the chain's 200k-wide matvecs in-graph without our joint_proj column split.
* Flash Next DFlash2 experiments: a different drafter (selector lattice) on a different model; only the policy conclusions (section 1.4) carry over.

## 9. File index (worktree /data/inkling-mini/research/atomic-engine-ews)
`src/models/inkling2-dspark.cpp` (332 lines: hparams 12-54, tensors 56-119, head 123-163, graph 192-332) | `src/models/dflash.cpp:36-70` (variant detection) |
`common/speculative.cpp:1375-1600` (DSpark impl; prompt lookup 1455-1476; injection 1478-1500; draft 1505-1600), `:3281-3287` (block-sized ubatch) |
`src/llama-context-atomic.cpp:91-138` (Inkling DSpark enable / draft load checks) | `src/llama-context.cpp:197-204, 580, 932-962, 1080-1089, 2499-2505, 3009-3016` |
`src/models/inkling2-loop.cpp:78-189` (looped target graph, env-gated kernels) | `src/models/inkling2.cpp:160-300, 300-470` (clip/mix/convolve/control/build_layer) |
`src/llama-graph.cpp:1899-1908, 2627-2637, 3884-3961` | `src/llama-batch.cpp:1214-1223` | `src/llama-expert-executor.cpp:155-260, 1278-1330, 1462-1476, 1894-1935, 2021-2074` |
`src/llama-expert-store.cpp:140-200, 380-420, 1301-1325` | `src/llama-inkling-core-banks.h` | `common/parsers/inkling.cpp` | `common/chat.cpp:1468-1471` |
`ggml/src/ggml-metal/ggml-metal-inkling-groups.cpp.inc` | `ggml/src/ggml-metal/kernels/mul_mm.metal:1022-1120`, `fa_aux.metal:5, 581-690`, `conv.metal:3-30`, `unary.metal:3-34` |
`ggml/src/ggml-metal/ggml-metal-device.cpp:2160-2162` | `tools/expert-source/bench-inkling-metal.py`, `fold-inkling-dspark.py`, `inkling-handover-4000.json`, `flash-next-handover-4000.txt` |
`README.md:22-157` | `docs/atomic-engine-release.md:36-56` | `docs/atomic-engine-inkling-low-bit-metal.json` | `docs/atomic-engine-flash-next-dflash-{depth2,depth3,confidence*,feedback}-*.json` |
`docs/atomic-engine-{early-reads,cache,route-index,preview}.md`.

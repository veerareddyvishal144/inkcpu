# Speculative decoding on the inkcpu engine (agent S2)

Owner: agent S2. Contract: docs/SPEC_CONTRACT.md. Drafter model and measured acceptance: docs/DSPARK.md. Status log: docs/status/S2.md.
Code: `include/ink/spec.h`, `src/spec/{sampler,policy,spec,dump_taps}.cpp`, verify executor in `src/engine/engine_prefill.cpp` (`engine_verify`, `engine_verify_commit`),
tap capture in `src/engine/engine_decode.cpp` and `engine_prefill.cpp`, C ABI in `include/ink/ink_abi.h` + `src/capi/ink_capi.cpp`, tests in `tests/spec/`, bench `bench/bench_spec.cpp`.

## 1. Goal and the hard property

Speculation must be invisible: with temperature 0 the token stream of the speculative loop equals plain greedy decode **token for token**, and the slot state after any step equals
the state plain decode would have left (so a later plain step, a stop in the middle of a window, or a switch back to plain never sees a difference). With temperature > 0 the
emitted tokens must be distributed exactly as the target sampler's tokens (exact speculative sampling) and reproducible per request seed. Speed is secondary to this property;
the adaptive policy exists so speculation is only used when it is predicted to pay.

## 2. Cycle (one `SpecDecoder::step`)

```
anchor a (sampled, not yet processed; position p = slot_pos)
  policy.choose(max_T) -> T in {1, 4, 8} (or a fixed width / plain)
  T == 1: Engine::decode(a) -> logits -> sample -> 1 token              (plain step; the tap hook commits the row to the drafter)
  T  > 1: drafter.draft(a, n = T-1) -> d_1..d_n [, dense q rows]
          engine_verify(slot, [a, d_1..d_n], T) -> logits of all T rows (+ taps of all rows); slot provisionally at p + T
          spec_accept(...) -> k accepted drafts + 1 token (correction or bonus) = k+1 emitted candidates
          apply max_emit and stop ids -> n_emit (1..k+1) emitted tokens; rows kept = n_emit (= anchor + the n_emit-1 emitted tokens that precede the last one)
          engine_verify_commit(slot, keep = n_emit) -> slot exactly "first n_emit rows processed" (p + n_emit)
          drafter.commit(taps of the kept rows)
  result: tokens[0..n_emit), stopped, counters, times; tokens[n_emit-1] is the next anchor (unless it is a stop id: it is never fed)
```

Position bookkeeping that the server relies on (V): after every step `slot_pos == p + n_tokens`; the rows processed in the step are `fed = [anchor, tokens[0..n_tokens-2]]`.
When the last emitted token is a stop id the slot holds exactly the tokens before it, as plain decode would (the stop token is never fed).

## 3. Verify executor and taps

`engine_verify` runs the T rows through the batched (prefill) executor with its own 8-row scratch (separate from the 256-token prefill scratch), computing the logits of every
row (the lm head on all rows) and the four DSpark taps of every row (stream entering visits 5 / 23 / 41 and the final normalised hidden state). The dense matmuls are the same
kernels as the decode path (bit-identical, agent G's GEMM == agent B1's GEMV). Attention has two modes:

- **exact** (`INK_SPEC_EXACT=1`, or `SpecOptions::exact = 1`): the decode attention kernels run per row, so every verify row is **bit-identical** to `Engine::decode` at any
  context length. This is the mode of the exactness tests.
- **fast** (default): the batched attention kernel reads the KV once for all rows; bit-identical to decode up to 767 visible keys (one key split), rounding-level different above
  (decode splits the keys in `nsplit >= 2` partial softmaxes). Above 768 keys a token whose plain top-1 / top-2 logit margin is at rounding level (~1e-3) can flip. This is the same
  caveat as agent P's prefill; greedy "exactness" there means agreement away from low-margin steps.

Taps: `Engine::tap_sink(slot, first_pos, rows, taps)` fires after every decode token and after every prefill chunk (not inside verify: the loop commits only the kept rows).
Null sink = zero cost (a pointer test). M1 verified taps decode vs batched bit-identical and logits with the sink on/off bit-identical.

## 4. Rollback: why `engine_verify_commit(keep)` restores exactly the plain state

The slot state is (pos, 48 x conv histories, 8 global KV layers, 40 SWA ring KV layers). After the verify of T rows at positions p..p+T-1:

1. **pos** is set to p + keep.
2. **Conv histories.** Each visit keeps the last 3 inputs of its four short convolutions (K, V, wo-out, moe-out). The verify records, per visit, the conv inputs of rows
   -3..T-1 (the 3 history rows before the verify plus every verify row). The history after exactly `keep` rows is rows keep-3..keep-1 of that record, which commit copies back.
   Rows of the record are computed from the same inputs the decode path would see for those rows (causal: row i depends only on rows < i and the history), so they are the values
   plain decode would have left (bit-identical in exact mode: the dense phases are identical kernels and attention is the decode kernel; in fast mode identical up to 767 keys).
3. **Global KV.** Cells of positions >= p + keep are never read: every attention of a later row at position q only reads positions <= q and all later rows are written before
   they are read (positions are overwritten in place by the next verify/decode). `n_blocks` may have grown; the blocks hold junk beyond pos, which is never addressed.
4. **SWA ring KV.** A rejected row at position p + j (j >= keep) wrote ring cell (p + j) mod 2048. The cell previously held position p + j - 2048, which is outside the window
   of every position >= p + keep (window = 2047 positions back; p + keep - 2047 > p + j - 2048 iff j >= keep - 1). So the overwritten cell was dead for every later read, and the
   junk is overwritten again before position p + j is ever attended (positions are written before they are read). Hence no later attention reads a rejected row's K/V.

The tests (`test_verify`) check this literally: after commit(keep) the slot equals a reference slot fed the first `keep` rows one by one (pos, memcmp of all 48 x 4 conv histories,
every global KV cell < pos, every SWA cell inside the window of pos) and the next 3 decode steps are bit-identical, for all keep in 1..T, T in {2, 4, 8}, at p = 40, p = 61
(rows crossing the 64-position KV block) and p = 2044 (rows crossing the 2048 global block and wrapping the SWA ring slot 2047 -> 0).

## 5. Acceptance and sampling

Sampler (`src/spec/sampler.cpp`) mirrors server/sampling.py: e = exp(max((x - max) / T, -80)); top_k keeps the k largest (ties: lower id), top_p renormalises over the top_k set and
keeps the smallest prefix reaching top_p, min_p keeps e >= min_p * best; greedy (T <= 0 or top_k == 1) is argmax (lowest id on ties). `test_sampler` checks the kept set and the probabilities
against a double-precision full-sort reference on 48 parameter sets. The random stream is a counter-based hash (`SpecRng`: draw k of seed s = splitmix64(s + k c)); the step is a pure
function of (slot state, request incl. seed); `next_seed` chains the steps. Distributions equal the Python sampler's; the draws are not bit-equal to numpy's PCG64.

Exact speculative sampling (`spec_accept`), per draft position i with target p_i (after temperature/top_k/top_p/min_p) and draft distribution q_i (dense rows from the drafter when it
samples; one-hot when it drafts by argmax): accept d_i iff u * q_i(d_i) < p_i(d_i); on rejection emit a token from norm(max(p_i - q_i, 0)) (if that residual is numerically zero,
p == q on the support, draw from p_i) and stop; after n acceptances the bonus token is drawn from p_{n}. Greedy params: accept iff d_i == argmax, the correction is the argmax
(no uniforms consumed). `test_sampler` checks the first emitted token follows p for q ~ p, q far from p, q with mass outside supp(p), uniform q and one-hot drafts (200k rounds,
chi-square); `test_spec sample` checks the same on the real engine from a restored state (chi-square of the first emitted token of 300 steps vs the analytic p of the plain logits).

Stop ids and max_emit are applied to the emitted sequence; the kept row count equals the emitted count (section 2), so the engine ends as plain decode of the same tokens would.

## 6. Drafter synchronisation

The drafter context must equal the engine slot before every draft. Every processed row reaches the drafter: plain decode and prefill chunks through the tap hook (committed lazily
by S1's drafter, flushed inside its single pool run), verify rows through the taps returned by `engine_verify` (only the kept rows are committed). `first_pos == 0` in the hook
means a new sequence: the drafter state is reset (and the policy's acceptance estimates). Rows the hook did not see (slot prefilled before the decoder existed, `slot_copy`)
mark the slot unsynced: it runs plain steps until the next reset (`SpecDecoder::synced`, `ink_spec_synced`, counter `sink_dropped`). The server therefore calls `ink_spec_load`
before prefilling.

## 7. Policy (`SpecPolicy`, `src/spec/policy.cpp`)

Per slot, clock-free (the loop feeds measured microseconds), deterministic (`test_policy` is a simulation with a fixed RNG):

- Conditional acceptance c_j = P(draft j accepted | 1..j-1 accepted) per position j = 1..7: running mean with 2 pseudo-observations at the prior (docs/DSPARK.md 9.2 pooled
  greedy priors) turning into an EWMA (alpha 0.15, ~16 steps memory). Positions never observed inherit the previous conditional with a 3 % decay. Expected tokens per step
  L(T) = 1 + sum_{m<T} prod_{j<=m} c_j.
- Times: EWMA (0.2) of the measured plain step, verify(T) per width, draft(n) per length; widths not measured yet use the bytes-model ratios of docs/ROUTING.md scaled by what was
  measured. Predicted speed-up(T) = L(T) t_plain / (draft(T-1) + verify(T)).
- Choice among {1, 4, 8} (modes: auto; off; fixed 2..8; env `INK_SPEC=auto|off|4|8`; per request override): the width with the best predicted speed-up if it exceeds `enter` = 1.04
  (hysteresis: `stay` = 1.00 once speculating), else plain. While speculating, every 16 spec steps one step uses the other width so its positions / verify time stay fresh
  (found necessary by the simulation: without it one unlucky T = 8 step crushed the untested positions' estimates and T = 4 stuck forever).
- Probing while plain: the first step of a sequence is plain (measures t_plain), the second is a T = 4 probe; afterwards a probe every 32 plain tokens. A promising probe (estimate
  within 15 % of the threshold or >= 2 accepted) re-probes 3 tokens later with observation weight 1/3 (a burst re-enters in a few steps after a regime change); an unpromising one
  doubles the interval up to 128. In the simulation: code-like regime (c = 0.9) 1.44x with T = 8 in 99 % of the spec steps, chat-like (c = 0.45, L(4) ~ 1.8) 0.99x with 98 % plain
  steps, re-entry after a chat -> code switch within the cap.

## 8. C ABI (include/ink/ink_abi.h)

`ink_spec_load(engine, drafter_gguf)` (loads S1's drafter, installs the hook), `ink_spec_free`, `ink_spec_enable(mode)`, `ink_spec_mode`, `ink_spec_synced(slot)`,
`ink_spec_step(engine, slot, const ink_spec_request*, ink_spec_result*)`, `ink_sample(engine, logits, params, seed, &next_seed)`. Structs (natural alignment, ctypes in
tests/spec/test_spec_capi.py): `ink_sampling_params {float temperature; int32 top_k; float top_p; float min_p}`; `ink_spec_request {int32 anchor; int32 mode; ink_sampling_params sp;
uint64 seed; const int32* stop_ids; int32 n_stop; int32 max_emit}`; `ink_spec_result {int32 tokens[8]; int32 n_tokens; int32 stopped; int32 fed[8]; int32 n_fed; int32 T; int32 drafted;
int32 accepted; int32 rc; double draft_us, verify_us, accept_us, total_us; uint64 next_seed}`. Sequence: ink_load -> ink_spec_load -> per request: ink_slot_reset, ink_forward(prompt)
-> ink_sample(logits) = first anchor -> loop ink_spec_step until stopped / budget. Without `ink_spec_load`, `ink_spec_step` is a sampled plain decode step (same sampler, same
stop / budget semantics), so the server has one code path.

## 9. Tests

| test | what | result |
|---|---|---|
| `test_verify quick/block/wrap` | verify rows == decode rows (bits), rollback == reference slot for every keep, next 3 steps bit-identical; p = 40, 61, 2044 | PASS (predecessor + wrap log) |
| `test_sampler` | sampler == reference semantics (48 sets), draws, exact speculative sampling histograms (6 q shapes) | PASS |
| `test_policy` | deterministic simulation of the policy (regimes, switch, priors, modes, clamps) | PASS |
| `test_spec quick/all` | greedy exactness with oracle / noisy / wrong doubles x T4 / T8 / mixed schedules (modes 0,2..8, budgets), final slot state == plain, stop suite, sampled suite | see section 10 |
| `test_spec_capi.py` | C ABI sequence with the real drafter: pos exactness, spec == plain, stop, budget, seeds, ink_sample vs numpy | see section 10 |

## 10. Numbers (shared host, 6 threads on CPUs 10,12,14,16,18,20, other agents running: indicative only)

Filled from the final runs; see docs/status/S2.md for the raw logs.

### 10.1 Exactness (test_spec all, exact attention, 8 prompts x 200 new tokens; log build-s2/data/test_spec_all.log)

56/56 greedy cells token-identical to plain decode; 48/48 full-length cells also end with the slot state identical to the plain slot (pos, conv histories, KV in window) and a
bit-identical next step. Cells: oracle drafter (full accept) x {T4, T8, mixed}, noisy oracle (30 % injected errors, accepted length 0..7) x {T4, T8, mixed}, wrong drafter (all
reject) x mixed (64 tokens; L = 1.00 as expected). Stop/budget suite 15/15 (stop at draft 5 of a T = 8 window -> 5 tokens, stopped, pos + 5; stop = bonus row; two stop ids; budgets
1 and 3; budget + stop; noisy drafter reaching a stop over 1-3 steps). Sampled suite (rerun on the flattest of the first 16 positions, p = 0.455 / 0.301 / 0.225 / 0.019 at temperature 0.8, top_k 8): plain draws 1500 chi-square 2.38 (df 4); 300
speculative steps from a restored state, first-token counts 150 / 82 / 59 / 9 vs expected 136 / 90 / 68 / 6, chi-square 5.23 (df 4, bound 30), same seed -> identical result. `test_spec_capi.py` with the real drafter: fixed T = 8 and auto streams == plain, pos exact after every step, stop mid-window ok.
bench_spec (fast attention, the default) also shows 0 mismatches vs plain in all 24 speculative cells, including long_context_qa at 715 + 80 positions.

### 10.2 Throughput and acceptance (bench_spec --n 80, real S1 drafter, fast attention, 6 threads, host load ~11; log build-s2/data/bench_spec.log)

| prompt | plain tok/s | fixed 4 tok/s (L) | fixed 8 tok/s (L) | auto tok/s (T hist) | R's L at T4 / T8 (DSPARK 9.2) |
|---|---|---|---|---|---|
| code_completion | 8.73 | 8.88 (3.48) | 9.03 (5.33) | 8.65 (71 plain / 3 x T4) | 3.67 / 5.92 |
| tool_call | 8.96 | 7.66 (2.79) | 8.02 (4.50) | 8.52 | - |
| chat_code | 9.03 | 8.72 (3.20) | 7.29 (4.21) | 8.21 (52 / 5 x T4 / 3 x T8) | 3.12 / 4.41 |
| reasoning | 8.56 | 7.72 (2.86) | 6.10 (3.40) | 8.18 | - |
| chat_answer | 8.79 | 4.99 (1.74) | 3.28 (1.78) | 8.37 | 1.77 / 1.79 |
| prose_continuation | 8.83 | 4.41 (1.56) | 2.83 (1.56) | 8.52 | - |
| chat_email | 9.03 | 5.63 (2.05) | 3.77 (2.16) | 8.09 | 1.73 / 1.85 |
| long_context_qa | 8.30 | 6.62 (2.44) | 5.07 (2.77) | 8.45 | - |
| **pooled** | **8.77** | **6.44 (2.52)** | **4.83 (3.21)** | **8.37** | 2.37 / 2.77 |

Measured step costs (means over the run): plain 111-120 ms; draft n = 3: 38-45 ms, n = 7: 51-55 ms; verify T = 4: 309-357 ms (2.8-3.0 x plain), T = 8: 486-534 ms (4.3-4.6 x plain);
accept + rollback + drafter commit 1-2 ms. The bytes model of docs/ROUTING.md predicted verify ratios of 2.47 (T = 4) and 3.72 (T = 8): on this host and thread count the batched
verify is 20-25 % more expensive than bandwidth-bound, so the break-even L is ~3.3 at T = 4 and ~4.9 at T = 8 and fixed speculation only pays on code_completion (+2-3 %).
Acceptance (L) matches R's llama.cpp table within the per-prompt noise (pooled 2.52 / 3.21 vs 2.37 / 2.77; code 3.48 / 5.33 vs 3.67 / 5.92; chat_answer 1.74 / 1.78 vs 1.77 / 1.79).

Policy behaviour: auto stayed plain on every prompt except chat_code (entered briefly: 5 x T4 + 3 x T8) and never lost more than 10 % (pooled 8.37 vs 8.77 = -4.5 %, the cost of
~3 probes per 80 tokens), where fixed 4 / 8 lose 27 % / 45 % pooled. On code_completion the predicted speed-up sat at the 1.04 entry threshold (L(4) 3.5 vs break-even 3.3) and
auto did not enter: correct given the measured costs, but the probes made it 1 % slower than plain. Caveats: 6 threads next to other agents; verify ratios (and so the verdict) may be
different on an idle 12-16 thread host, and the policy measures them at run time, so the decision adapts without re-tuning.

### 10.3 Regressions (build-s2, 6 threads)

test_prefill quick PASS (bit-identical), test_engine oracle: 5/6 PASS, m2_code_c cosine 0.991664 = C1's documented noise floor with identical digits (docs/ENGINE_DESIGN.md), i.e.
numerics unchanged by the tap hook and the loader change; tests/engine/test_capi.py PASS on the new libink.so; bench_decode A/B 6 threads ctx 16: build-c1b 9.53 tok/s vs build-s2
9.24 tok/s back to back at load 11-12 (-3 %); second pair in reverse order at load 5: build-s2 9.33 vs build-c1b 9.41 tok/s (-0.9 %). Indicative; the decode path itself is
unchanged apart from the null-tested tap hook.


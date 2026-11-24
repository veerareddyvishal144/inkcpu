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


## 11. Session-lookup drafter (agent L, wave 2; `src/spec/lookup.cpp`, `tests/spec/test_lookup.cpp`)

Evidence and break-even model: `docs/research/wave2_speculation.md` (recommendation #1) and `docs/RESEARCH_WAVE2.md` item 2. Coding-agent / editor turns re-emit text that is
already in the context (files, tool results, the previous turn): an exact n-gram match of the slot's own history predicts the next tokens for free, where the DSpark drafter
costs 19-33 ms per step. The drafts go through the same `engine_verify` + rejection sampler as DSpark drafts (one-hot q), so section 1's property is untouched: the emitted
tokens equal plain greedy decode token for token, and the slot state after every step equals plain decode's. The only cost of a wrong lookup draft is its verify rows
(0.25-0.5 plain steps per row), hence everything below is about firing on strong evidence only.

### 11.1 History and index (`LookupIndex`, one per slot inside the `SpecDecoder`)
* The history holds every token the slot processed, by position: the prompt (reported by `ink_forward` through `SpecDecoder::note_tokens`, or by a C++ caller), the rows a
  step feeds (anchor + the emitted tokens except the last one, recorded by `step()` itself), truncation on `ink_slot_rewind` (`note_truncate`) and clearing on `ink_slot_reset` /
  `ink_slot_new` / a first row at position 0 (`note_reset`, the tap hook). A step whose history length differs from the slot position repairs it first: longer -> truncate (a
  rewind it was not told about), shorter -> pad with unknown tokens (-1, never match; e.g. a slot filled by `slot_copy`). 4 bytes per position plus the index below.
* Index: open-addressing hash table from the 64-bit hash of the `n_min` (default 8) tokens ending at a position to the most recent such position, and `prev[pos]` = the
  previous position with the same key (a chain per key). `append` is O(1) without allocation (the table doubles when half full), `truncate(pos)` undoes the appends in
  reverse order (the chain heads are restored exactly, so the index after a rewind equals one built from the shorter history: `test_lookup` checks this chain for chain over
  800 random append / truncate / reset / pad operations), `reset` bumps a generation counter (O(1)). Grams containing an unknown token are not indexed.
* Query: the key of the last `n_min` tokens of history ++ anchor; the chain is walked for at most `max_candidates` = 16 entries (most recent first), each backward-extended to
  its true match length (cap `max_match` = 64); the longest match wins, ties go to the most recent. The draft is the continuation after the match, taken from the virtual
  sequence history ++ anchor ++ draft (so periodic text continues cyclically). Ids outside [0, n_vocab) and unknown positions end the draft. Cost: a few microseconds.

### 11.2 Width policy (`LookupPolicy`, one per slot)
Conditional acceptance c_{b,j} = P(draft j accepted | 1..j-1 accepted) per match-length bucket b in {[8,12), [12,16), [16,32), [32,inf)} and draft position j = 1..7, priors
0.70 / 0.80 / 0.90 / 0.95, running mean turning into an EWMA (0.1), unobserved positions inherit the previous one with a 3 % decay (as in section 7). The draft is free, so the
predicted speed-up of width T is L_b(T) / (verify(T) / plain), with the verify and plain times taken from the slot's DSpark policy (its measured EWMAs; before any measurement
the ratio priors of `SpecPolicyOptions::verify_ratio`; lookup steps feed their verify times back through `SpecPolicy::observe_verify_time`). The step uses the T in 2..8 with
the best predicted speed-up if it exceeds `enter` = 1.0 (break-even: 1.15 accepted tokens at T = 4, 2.43 at T = 8), else it declines the match. This is the research file's
"cap at lambda ~0.3 per marginal row" rule in closed form: the marginal row T+1 is added only while its expected gain prod_{j<=T} c_j exceeds its marginal verify cost. A bucket
whose estimate fell below break-even fires a T = 2 probe every `probe_every` = 8 declined matches (cost <= 0.45 plain steps per rejected probe, observation weight 1/3), so the
width grows back when acceptance recovers. Per-bucket counters (steps, drafted, accepted, declined) are exposed by `ink_spec_lookup_stats`.

### 11.3 Modes and precedence
* `auto` (1, the default): when the history ++ anchor has a match of >= n_min tokens, the lookup policy (11.2) picks its width and predicted speed-up; if the DSpark drafter
  is loaded and synced, the DSpark policy's best predicted speed-up over its widths (section 7, `SpecPolicy::speedup`, draft cost included) is compared with it and the
  source with the HIGHER prediction drafts the step (ties: lookup, the free draft). A match the lookup policy declines (below break-even) or yields to DSpark, and every
  step without a match, goes to the unchanged adaptive DSpark policy (it sees only the steps it drafts, so its probe schedule and acceptance estimates are unaffected by
  lookup steps). Without a drafter, auto = lookup + plain. Why the arbitration: in the first build a firing lookup always pre-empted DSpark, which cost 11 % on
  `lookup_toolcall` (DSpark accepts 250/265 rows there while the lookup matches are short, 2.5 accepted rows per step); the `yielded` counter of
  `ink_spec_lookup_stats` reports how often the lookup lost the comparison.
* `lookup` (9, `INK_SPEC=lookup`, server `--spec lookup` / request `"speculation": "lookup"`): lookup, else plain; the drafter (if loaded) idles.
* Fixed widths 2..8 and `off` are unchanged (DSpark only / plain).
* Mixing DSpark tokens into the remaining rows of a short lookup draft was NOT built (measured elsewhere to lower acceptance; research file, RustingStrata PR 74).
* `INK_SPEC_LOOKUP=0` disables the lookup drafter (auto behaves as before); `INK_SPEC_LOOKUP_NMIN`, `_ENTER`, `_MAXDRAFT`, `_PROBE` tune it.
* The DSpark context stays synchronised through lookup steps (the kept rows' taps are committed to the drafter exactly as after a DSpark step).

### 11.4 C ABI (additive; `ink_spec_result` layout unchanged)
`ink_forward`, `ink_slot_reset`, `ink_slot_new` and `ink_slot_rewind` feed the history (a decoder without a drafter is created lazily at the first `ink_forward`, so a server
needs no new call). `ink_spec_lookup(engine, on)` enables / disables the lookup drafter (and creates the drafter-less decoder: lookup-only speculation without `ink_spec_load`),
`ink_spec_lookup_enabled`, `ink_spec_last_source` (0 plain / 1 DSpark / 2 lookup for the last step), `ink_spec_lookup_stats(engine, slot, out, cap)` (4 buckets x steps,
drafted, accepted, declined, yielded). `ink_spec_enable` / `ink_spec_request.mode` accept 9. Server (`server/backends.py InkBackend._setup_spec`): with `--drafter`
(production, `tools/serve.sh`: `--drafter ... --spec auto`) nothing changes on the caller's side and auto = lookup first + DSpark; a drafter-less `--backend ink` server
now takes the speculative path through `ink_spec_lookup` by DEFAULT (no `--spec` = the engine default auto, or `--spec auto|lookup`), so its auto = lookup + plain;
`--spec off`, `INK_SPEC=off` or `--no-spec` keep the plain per-token path, and a libink.so without the symbol behaves as before. `/health` shows `spec.lookup: true`.
Knobs: `INK_SPEC_LOOKUP=0` (kill switch), `INK_SPEC_LOOKUP_NMIN` (8), `INK_SPEC_LOOKUP_ENTER` (1.0), `INK_SPEC_LOOKUP_MAXDRAFT` (7), `INK_SPEC_LOOKUP_PROBE` (8).

### 11.5 Tests and numbers
`test_lookup` (no engine): find() == brute force on random (V = 3 / 50, long chains) and copy-heavy streams, continuation incl. cyclic self-extension, rewind / reset / pad
consistency, unknown padding, policy (widening to 8 on acceptance, shut-off + probes + recovery on rejection, timing model). `test_spec quick` gained the `lookup` suite:
mode 9 with a 2- and 4-token key firing on every match (so most steps verify wrong lookup drafts of every length, some copy runs get accepted) and auto with the oracle double
(lookup first, DSpark fallback), with a mid-run rewind: tokens == plain, final slot state == plain, the decoder's history == the processed tokens. `bench_spec` modes
`dspark` (old auto) / `lookup` / `auto` and the copy-heavy prompts `lookup_docstring` (1743-token Python file, "add a docstring to every function and return the file"),
`lookup_toolcall` (a tool call whose JSON repeats a path list from the prompt), `lookup_typo` (fix one typo, print the function again). Numbers: `docs/status/L.md`.

Measured 2026-11-24 (agent L2, `build-l/sb2` = commit 5be85af + this section's code, `bench_spec` 12 threads on the default CPUs, fast verify attention, greedy, loadavg
8.7-14.1 because another session compiled throughout; the four cells of a prompt are consecutive, compare within a row). tok/s plain / dspark (= the old auto) / lookup / auto:

| prompt (p = prompt tokens) | plain | dspark | lookup | auto | auto vs max(plain, dspark) | lookup acceptance (auto) |
|---|---|---|---|---|---|---|
| lookup_docstring (1743), n = 300 | 14.00 | 21.30 | 24.29 | 24.39 | +14.5 % | [32+] 23 steps 160/161 rows, [16-31] 7 steps 41/41, 1 wasted row |
| lookup_toolcall (568), n = 300 | 14.00 | 24.36 | 15.69 | 24.45 | +0.4 % | 23 short matches yielded to DSpark (its prediction was higher), 0 lookup steps |
| lookup_typo (537), n = 300 | 14.05 | 25.85 | 29.24 | 29.43 | +13.8 % | [32+] 34 steps 238/238 rows, 0 wasted |
| pooled 3 lookup prompts | 14.02 | 23.69 | 21.62 | 25.89 | +9.3 % | before the arbitration fix: 14.39 / 23.80 / 21.51 / 24.81 |
| pooled 8 DSpark prompts, n = 120 | 15.04 | 15.04 | 15.35 | 15.50 | +3.1 % | fired 0-5 times per 120 tokens, 3 wasted rows in 960 tokens |

Per DSpark prompt auto vs max(plain, dspark): code_completion -0.8 %, tool_call +1.2, chat_code +4.0, reasoning +0.9, chat_answer +10.0 (a 16-31 match, 14/14 rows), prose
-1.6, chat_email -3.6 (dspark-only is -3.1 on it: DSpark probes 0/6 under load; the one lookup step wasted 2 rows ~ -0.5 %), long_context_qa -0.6; plain itself spans
14.74-15.43 in that window (+-2.3 %). 0 mismatches vs plain in all 44 cells (lookup and auto never flipped a token; the fast-attention caveat of section 3 for p > 768
remains for any verify step, DSpark or lookup: `INK_SPEC_EXACT=1` for bit-exactness). The same three prompts with `INK_SPEC_EXACT=1` (second measurement, loadavg
12-15): plain / dspark / lookup / auto = docstring 14.37 / 22.19 / 24.02 / 24.48, toolcall 14.81 / 25.28 / 15.75 / 25.30, typo 14.77 / 27.10 / 29.53 / 30.15, pooled
14.65 / 24.69 / 21.63 / 26.41, 0 mismatches; exact verify costs nothing measurable at these prompt lengths (T = 8 verify 248 ms exact vs 244-251 ms fast at p = 1743).
Gate logs: `build-l/data/{c1,c2,c3,c4}_*.log`, status: `docs/status/L.md`.

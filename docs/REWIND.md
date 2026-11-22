# REWIND: slot checkpoints, rewind and longest-common-prefix reuse (agent S3)

Files: `src/engine/engine_rewind.cpp` (checkpoint store, `Engine::slot_checkpoint / slot_rewind / slot_checkpoints`), hooks in
`engine_prefill.cpp` (chunk-end checkpoints, `hwm`), `engine_decode.cpp` (`forward` of >= 2 tokens, `hwm`), `engine_load.cpp` (reset / copy / free),
`include/ink/engine.h` (API, `SlotState::hwm`, `Tune::ckpt_*`), `include/ink/ink_abi.h` + `src/capi/ink_capi.cpp` (C ABI), `server/backends.py`
(`rewind / checkpoint / checkpoints`), `server/scheduler.py` (LCP policy), tests `tests/engine/test_rewind.cpp`, `tests/engine/test_rewind_capi.py`,
`server/tests/test_scheduler.py` (5 rewind tests), `server/tests/test_ink_real.py` (m, n). Build: `cmake -S . -B build-s3 -G Ninja && ninja -C build-s3 ink test_rewind`.

## 1. Problem

The editor client (Caret) sends a ~7,200-token prefix (system prompt + 32 tool schemas) on every request and re-renders the conversation
history differently from what the model generated (it drops `reasoning_content`), so the server's append-only slot cache ("the slot history must
be a full prefix of the new prompt") missed on every turn after the first: ~100-120 s of prefill per query (SERVING.md section 12), although
7,000+ tokens were identical to the previous request. Two conversations with the same system prompt shared nothing either.

## 2. What a slot's state is (why a checkpoint is small)

The recurrent state of a slot is exactly: `pos`, the 48 conv histories (`hist`: the last 3 inputs of the 4 short convolutions of every visit,
3.83 MB), the KV of the 8 global visits (grows with pos) and the KV of the 40 SWA visits (ring of `kSwaRingBlocks` = 32 blocks x 64 = 2048 cells,
cell = pos mod 2048). Controllers, message bank, attention-residual block states and all activations live in the per-token `Scratch`
(ENGINE_DESIGN.md section 4); S2's verify rollback (SPEC_DESIGN.md section 4) relies on the same fact and is tested bit-exactly. So a **shallow
checkpoint at position P = {P, copy of hist}** (3.83 MB) restores the plain state at P provided the KV that a token at position >= P can read is
still intact:

* global visits: a query at position q reads positions 0..q. After a rewind to P, rows >= P are junk but every row is written before it is read
  (the next token at P overwrites row P), so nothing needs saving. `n_blocks` stays grown (same rule as after a verify rollback).
* SWA visits: a query at q reads positions q-1023..q (`kSwaWindow` = 1024). After a rewind to P the rows >= P are rewritten before being read;
  the rows P-1023..P-1 must still hold the right data.

## 3. Restorability bound (shallow checkpoints) and its proof

Let k tokens have been appended after the checkpoint at P (positions P..P+k-1). They wrote ring cells (P+i) mod 2048, i < k, which previously
held positions P+i-2048. Such a cell belonged to the needed window P-1023..P-1 iff P-1023 <= P+i-2048 <= P-1, i.e. 1025 <= i <= 2047. So no needed
cell is damaged iff k-1 < 1025, i.e. **k <= 1025 = ring - (window - 1)** = `Engine::rewind_max_append(swa_blocks)`. The same number is the
prefill executor's `chunk_limit` (a chunk appends all its K/V before attending, so it may hold at most ring - 1023 tokens): the chunk interaction is
already covered by the bound. k = 1025 is accepted and bit-exact, k = 1026 is refused (test (4)).

`k` is NOT "live position - P": after a rewind to P2 > P the junk that the abandoned continuation wrote above P2 is still in the ring, so the slot
keeps `hwm` = the highest position ever written since the reset (decode, prefill, verify rows and rewinds all keep it monotone) and a shallow
checkpoint is restorable iff `hwm - P <= 1025`. Once dead it stays dead (hwm never decreases) and is evicted first.

**Deep checkpoints** additionally copy the ring blocks that hold the window of P (positions max(0, P-1023)..P-1 of the 40 SWA visits: <= 17 blocks
x 128 KiB x 2 tensors x 40 visits = 170 MiB) and restore them wholesale on rewind. Whole blocks are safe: a restored cell outside the window belongs
either to a position >= P (rewritten before it is read) or to a position < P-1023 (never read by a query >= P). Deep checkpoints are restorable until
evicted, no matter how many tokens followed. They are meant for prompt ends (a generation of > 1025 tokens, common with thinking on, would kill a
shallow one) and for points where two conversations were observed to diverge.

## 4. Engine API (`include/ink/engine.h`, `docs/REWIND.md` is the contract)

```cpp
int     Engine::slot_checkpoint(int slot, bool deep = false);      // 0 ok, -1 bad slot, -5 out of memory; no-op at pos 0 or when INK_CKPT_MAX=0
int64_t Engine::slot_rewind(int slot, int64_t pos);                // nearest restorable checkpoint <= pos: restores it, drops the ones above, returns it; -1 none
int     Engine::slot_checkpoints(int slot, int64_t* out, int max) const;   // restorable positions ascending; returns the count
static constexpr int64_t Engine::rewind_max_append(int swa_blocks);        // 1025 for the 32-block ring
```
Automatic shallow checkpoints: after every prefill chunk (`INK_PF_CHUNK`, 512 in production, 256 default) including the last one (= prompt end), and
at the end of `forward()` of >= 2 tokens on the decode-loop path. `decode()` (single tokens) never checkpoints (3.83 MB per token would be too much);
the server asks for one explicitly at the end of each generation. Store per slot: a ring of `INK_CKPT_MAX` = 8 entries (`Tune::ckpt_max`), of which at
most `INK_CKPT_DEEP` = 2 deep (`Tune::ckpt_deep`); an existing entry at the live position is reused (upgraded to deep on request); eviction order:
empty, dead shallow, oldest shallow, oldest deep (the oldest deep is demoted to shallow when a new deep one needs its buffer); a rewind "touches" the
entry it used. `slot_reset` drops all checkpoints, `slot_copy` leaves the destination without any, `slot_free` releases the buffers.

**Memory per slot**: buffers are allocated on first use and kept: 8 x 3.83 MB = 30.6 MB shallow + up to 2 x 170 MiB deep = **~370 MB worst case
per slot** (2 slots: ~740 MB on top of the 17 GB engine); a slot that never received a deep checkpoint holds <= 30.6 MB. `slot_ckpt_bytes()` reports it.
Checkpoint cost: 3.83 MB memcpy (~1 ms) per shallow, + 170 MiB (~30-50 ms single-threaded) per deep; a rewind costs the same copies in reverse.

Speculation (S2): the drafter's context is keyed by position; after a rewind to a lower position the next tap rows arrive at `first_pos` != the
drafter's position, `SpecDecoder::on_taps` marks the slot unsynced and `ink_spec_synced` returns 0: steps run plain until the next `slot_reset`
(the simplest correct behaviour; a drafter rewind would be the next step). A rewind to the live position changes nothing.

C ABI (`ink_abi.h`, additions only): `ink_slot_checkpoint(e, slot)`, `ink_slot_checkpoint_deep(e, slot)`, `ink_slot_rewind(e, slot, pos)`,
`ink_slot_checkpoints(e, slot, out, max)`; a `libink.so` without them keeps the server on the append-only policy (`InkBackend.supports_rewind`).

## 5. Server policy (`server/scheduler.py`, `SlotTable.acquire`)

1. Full-prefix path as before (a slot whose whole history is a prefix of the prompt; identical prompt = cached logits, nothing prefilled).
2. Otherwise the idle slot with the longest common prefix (token level, numpy) with the prompt, if LCP >= `--prefix-reuse-min` (default 1024;
   `ServerConfig.prefix_reuse_min`; 0 disables): `rewind(slot, min(LCP, len(prompt) - 1))` -> P; the history becomes `tokens[:P]`, `prompt[P:]` is
   prefilled, `cached_tokens` = P, the engine position is cross-checked (`BackendError` otherwise). The prefill chunk that reaches the LCP ends exactly
   there and gets a deep checkpoint (a third conversation sharing the same prefix then hits it exactly); if the LCP already is a checkpoint it is
   upgraded to deep. When no checkpoint <= LCP is restorable (`rewind` returns -1) the request falls back to 3.
3. Empty slot, else LRU slot reset.
Every prompt end gets a deep checkpoint (Caret: the re-rendered follow-up has LCP = previous prompt length; the generation in between may exceed
1025 tokens), every generation end a shallow one (a follow-up that differs only in the last tokens). The slot history invariant (history == tokens
the engine holds) is asserted after every rewind and speculative step. A rewind is preferred over an empty slot (saves the prefill now; the other
conversation, if it returns, rewinds to the shared prefix itself). `/metrics` is unchanged; `cached_tokens` / `timings.cache_n` report P.

## 6. Tests

| test | what it checks |
|---|---|
| `test_rewind quick` (1) | prefill A = 1500 wikitext tokens (chunk 512: checkpoints 512, 1024, 1500), decode 50 (no new checkpoints), `slot_rewind(1500)`, prefill a different 40-token tail, decode 10: every logits vector bit-identical to a fresh slot fed A + tail + 10; slot state equal (hist memcmp, every global KV cell < pos, every SWA cell in the window of pos) |
| (2) | `slot_rewind(1300)` lands on the 1024 chunk checkpoint (checkpoints above are dropped); tail + 10 decodes bit-identical to fresh A[:1024] + tail + 10 |
| (3) | refusal: the 512 checkpoint with hwm 1550 (1038 > 1025 appended) is not listed and `slot_rewind(512)` / `(1023)` return -1 leaving the slot unchanged |
| (4) | the exact bound: checkpoint 512, append until hwm - 512 == 1025: accepted and bit-exact (tail + 5 decodes vs fresh); hwm - 512 == 1026: refused, while 600 (938 appended) is accepted and bit-exact |
| (5) | deep checkpoint at 1500, then 1200 more tokens (wraps the 2048 ring): 1024 / 512 are dead, 1499 refused, `slot_rewind(1500)` + the tail of (1) reproduces (1)'s reference bit for bit; state equal to fresh; the entry can be reused |
| `test_rewind determinism` | scenario (1) at 3 threads (CPUs 26,28,30) and 6 threads (26-31): the written logits files are byte-identical |
| `tests/engine/test_rewind_capi.py` | the C ABI through ctypes: automatic checkpoints [512, 1024, 1300] after `ink_forward` of 1300 tokens, none after 20 decode steps, refusal (-1) and bad-slot returns, `ink_slot_rewind(1300)` + 40-token tail + 10 decodes bit-identical to a fresh slot, `ink_slot_checkpoint[_deep]`, rewind to the live position = no-op, `ink_slot_reset` drops everything |
| `server/tests/test_scheduler.py` (5 new, mock backend) | LCP reuse lands on the nearest checkpoint, the divergence point gets a deep checkpoint and a third conversation hits it exactly, the rewound output equals a cold run, history invariant after every request; threshold and refused rewind fall back to the LRU reset; prompt that is a strict prefix of the history processes >= 1 token; the slot with the longest LCP wins over an empty slot; a backend without rewind keeps the append-only policy |
| `server/tests/test_ink_real.py` m | real server (3 threads): three conversations with a 2.1k-token system prompt: request 2 rewinds (cached within one chunk below the LCP), request 3 hits the divergence checkpoint exactly, request 2's output equals its cold run, prefill time < 50 % of cold |
| n | the Caret pattern: turn 2 resends the history with the assistant turn re-rendered without `reasoning_content`: cached_tokens == turn 1's prompt length (deep prompt-end checkpoint), output equals a cold run |
| regressions | the 64 mock server tests, `test_prefill quick`, `test_capi.py`, `test_spec quick`, `test_engine oracle` on the build-s3 binaries |

Results: section 7.

## 7. Results and measurements (2026-11-22, build-s3, INK_CPUS=26,28,30 = 3 threads, host load 2.8-3.7: timings indicative, exactness exact)

Engine gates (`build-s3/logs/test_rewind_quick.log`, `test_rewind_capi.log`): every scenario of section 6 PASS. Rewind + tail + decodes are
BIT-IDENTICAL to a fresh slot in all five scenarios (11 + 11 + 6 + 4 + 11 logits vectors), the slot state after each scenario is equal (hist memcmp,
0 of 55.0 M / 51.1 M / 27.6 M KV cells read by the next token differ), the bound is exact (1025 appended: accepted and bit-exact; 1026: refused), the
deep checkpoint restored after 1200 appended tokens across the ring wrap reproduces the fresh reference bit for bit. Checkpoint memory held by the
test slot after scenario (5): 199.2 MiB (4 shallow + 1 deep). Prefill on 3 threads: 18-20 tok/s; a 40-token tail 2.9-3.1 s; the deep checkpoint
and rewind copies are not visible in these timings (< 0.1 s).

Determinism (`build-s3/logs/det3.log`, `det6.log`): scenario (1) at 3 threads (CPUs 26,28,30) and at 6 threads (26-31, SMT siblings): the 11 logits
vectors written by both runs are byte-identical (`cmp`), and each run passed its own fresh-slot comparison.

Regressions on the build-s3 binaries: the 64 mock server tests PASS (+ 5 new rewind tests = 69), `test_prefill quick` PASS (prefill == decode bit for
bit in both attention modes), `tests/engine/test_capi.py` PASS, `test_spec quick` PASS (0 failures), `test_engine oracle` reproduces C1's documented
table digit for digit (5/6, m2_code_c cosine 0.991664 = the documented noise floor; unchanged).

Server (`server/tests/test_ink_real.py` m, n through `python -m server --backend ink --threads 3`, prefill chunk 256, `build-s3/logs/ink_real_results_mn.json`):

| case | prompt tokens | cached_tokens | new tokens prefilled | prefill ms | cold prefill ms (same request, slots flushed) | output |
|---|---|---|---|---|---|---|
| (m) request 1: 2.1k-token system prompt + Q1 | 2134 | 0 | 2134 | 120855 | - | - |
| (m) request 2: same system prompt + Q2 (LCP 2122) | 2135 | **2048** (chunk checkpoint <= LCP) | 87 | **6878** | 120558 (17.7 tok/s) | == cold run (greedy) |
| (m) request 3: same system prompt + Q3 | 2135 | **2122** (= LCP: the divergence checkpoint of request 2) | 13 | **1501** | - | - |
| (n) Caret turn 1: system prompt + user, `reasoning_effort: low`, 40 tokens generated | 2134 | 0 | 2134 | ~121000 | - | - |
| (n) Caret turn 2: history re-rendered without reasoning_content + new user turn | 2148 | **2134** (= turn 1's prompt length, deep prompt-end checkpoint) | 14 | **1782** (TTFT 1829) | 121268 (TTFT 121272) | content + reasoning == cold run |

Rewind + speculation (`test_ink_real.py` o, spec server with the DSpark drafter, `build-s3/logs/ink_real_results_o.json`): request 1 on a fresh slot
speculates (`spec_synced` true); request 2 shares the system prompt, is rewound (cached 2048) and runs its speculative steps plain (`spec_synced`
false, every step T = 1, 0 drafted) with the same content as a `speculation: off` run; a request without the shared prefix resets the LRU slot and
is synced again (`spec_synced` true). This is the documented "simplest correct behaviour" of section 4.

The Caret follow-up turn went from 121.3 s to 1.8 s of prefill on 3 threads (68x; the cold number is what the append-only cache delivered on this
pattern). Scaled to the production deployment (16 threads, 60-70 tok/s, 7,215-token prefix: 100-120 s cold per turn, SERVING.md section 12) the
follow-up turn should prefill only the re-rendered assistant turn + the new user message (tens to a few hundred tokens: ESTIMATE 1-5 s + the
same-length decode), and a second conversation with the same system prompt + tool schemas starts from the nearest checkpoint below the shared
7,200 tokens (<= one prefill chunk of 512 to redo: ESTIMATE <= 8 s) instead of 100-120 s. Not measured on the production server (not touched).

## 8. Not done, risks

* The drafter is not rewound: a rewound slot speculates again only after its next reset (plain steps meanwhile; correct, slower on code-like
  text where speculation pays). Fix: a `Drafter::state_truncate(st, pos)` (the drafter keeps its own per-position state) plus a resync in
  `SpecDecoder` on rewind; the engine side would call it from `slot_rewind`.
* Checkpoint granularity = the prefill chunk (512 in production): a conversation that shares a prefix with a slot for the first time redoes up to
  one chunk (<= 8 s at 16 threads); the divergence-point checkpoint fixes it for the following conversations. Message-boundary checkpoints (end of
  the system prompt + tool schemas, known to the app) would make even the first one exact; not implemented.
* A rewind is preferred over an empty slot: the other conversation's tail cache is sacrificed (it rewinds to the shared prefix when it returns).
  With two slots and two editor threads this is the right trade; with many unrelated conversations the LRU reset stays the fallback.
* Memory: up to ~370 MB of checkpoint buffers per slot (allocated on first use, 2 deep x 170 MiB + 8 x 3.83 MB); `INK_CKPT_DEEP=0/1` and
  `INK_CKPT_MAX` tune it. The deep save/restore copies are single-threaded memcpy (~30-60 ms each at 170 MiB; not visible next to a prefill).
* The restorability bound is tied to `kSwaRingBlocks` x 64 and `kSwaWindow` through `Engine::rewind_max_append(tune.swa_blocks)`; a smaller
  ring (`INK_SWA_BLOCKS`, tests only) shrinks it accordingly.
* Timings above were taken on 3 threads on the shared host (load 2.8-6): indicative only; the production server (16 threads) was not touched,
  so the 7,215-token Caret numbers are estimates derived from its measured 60-70 tok/s prefill.
* The server records rewinds only as `cached_tokens` (and `/slots` -> `rewinds`); no per-request flag in the request log.

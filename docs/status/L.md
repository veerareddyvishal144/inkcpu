# Agent L (wave 2): session-lookup drafter (draft-model-free n-gram speculation) for the SpecDecoder

Start 2026-11-24 02:16 UTC, hard stop 06:16 UTC. Contract: docs/OPT_WAVE2.md. Dev CPUs 1,3,5,7,9,11,13,15; build dir build-l; all full-model runs under tools/with_bench_lock.sh.

- 02:16-02:22 read the contract, research (wave2_speculation.md), SPEC_DESIGN, spec.h / spec.cpp / policy.cpp, ink_abi.h / ink_capi.cpp, bench_spec, tests/spec, K2 status,
  scheduler._spec_step / backends.InkBackend. loadavg 5.6 (another session compiles Rust/C on this host).
  Design decisions (details in docs/SPEC_DESIGN.md section 11 once written):
  * LookupIndex per slot: token history (int32 per position) + open-addressing hash table from the hash of the last N_MIN = 8 tokens ending at a position to the most recent
    such position, with a `prev` chain per position (older occurrences). append = O(1); truncate(pos) = exact undo of the appends in reverse order (so a rewind leaves the
    index identical to one built from the shorter history); reset = generation bump (O(1)). Query: key of the last 8 tokens of history ++ anchor; walk <= 16 chain candidates,
    backward-extend each to its true match length (cap 64), pick the longest (tie: most recent). Draft = tokens following the match, cyclic self-extension when the
    continuation runs into the query itself (periodic text). Unknown positions (history gaps, e.g. a slot filled before the decoder existed) are padded with -1 and never match.
  * LookupPolicy per slot: conditional acceptance per (match-length bucket [8,12) [12,16) [16,32) [32,inf), draft position 1..7), priors 0.70 / 0.80 / 0.90 / 0.95, running
    mean -> EWMA 0.1; verify width T = argmax over 2..8 of L_b(T) * t_plain / verify(T) using the DSpark policy's measured step times (shared time model); fires only when the
    best predicted speed-up > 1.0 (draft cost 0 => break-even 1.15 accepted tokens at T=4, 2.43 at T=8). That is the lambda ~0.3 marginal-row rule in closed form.
  * Modes: auto (1) = lookup first, else the unchanged adaptive DSpark policy; lookup (9) = lookup, else plain; fixed 2..8 unchanged (DSpark only). INK_SPEC=lookup, INK_SPEC_LOOKUP=0 kill switch.
  * C ABI (additive): ink_forward / ink_slot_reset / ink_slot_rewind / ink_slot_new feed the decoder's history; new ink_spec_lookup(engine, on) creates the decoder without a
    drafter so lookup-only speculation works without ink_spec_load; ink_spec_enable accepts 9. ink_spec_result layout UNCHANGED (ctypes in the server mirrors it).
- 02:22-02:33 code written: include/ink/spec.h (LookupIndex, LookupOptions, LookupPolicy, kSpecModeLookup = 9, SpecResult::src / match_len, SpecDecoder::note_tokens /
  note_truncate / note_reset / lookup / lookup_policy), NEW src/spec/lookup.cpp, src/spec/spec.cpp (lookup first in auto / mode 9, history bookkeeping of anchor + fed rows,
  drafter commit kept on lookup steps), src/spec/policy.cpp (verify_ratio, observe_verify_time), src/capi/ink_capi.cpp + include/ink/ink_abi.h (hooks in ink_forward /
  ink_slot_reset / ink_slot_new / ink_slot_rewind; ink_spec_lookup, ink_spec_lookup_enabled, ink_spec_last_source, ink_spec_lookup_stats; ink_spec_enable accepts 9;
  ink_spec_result UNCHANGED), src/spec/CMakeLists.txt (+lookup.cpp, +test_lookup), NEW tests/spec/test_lookup.cpp, NEW prompts tests/spec/prompts/lookup_{docstring,toolcall,typo}.txt
  (+ .i32: 1743 / 568 / 537 tokens) + make_lookup_prompts.py, spec_test_util.h (3 names appended after the 8 DSpark prompts), bench/bench_spec.cpp (modes dspark / lookup / auto,
  prompt sets dspark / lookup / all, per-source and per-match-length acceptance, wasted rows, pooled over DSpark prompts / lookup prompts / all), server/backends.py (--spec lookup
  = mode 9; without --drafter and with --spec lookup|auto the backend calls ink_spec_lookup and uses the spec path; a plain server without --spec is unchanged).
- 02:31 first build of build-l: my code compiled; one linkage error of mine fixed (helpers moved out of extern "C"). Model-free gates on build-l: test_lookup PASS (random V=3/50,
  copies, periodic, 800 rewind/reset ops chain-identical to a rebuilt index, unknown padding, policy), test_policy PASS, test_sampler PASS. loadavg 0.8-2.6.
- 02:33 recovery added to LookupPolicy: a bucket below break-even fires a T=2 probe every 8 declined matches (otherwise it could never observe a recovery); test updated.
- 02:35 BLOCKED on the shared tree: agent P2's src/kernels_gemm/CMakeLists.txt (02:30) lists gemm_i16.cpp which does not exist yet, so cmake cannot regenerate build-l.
  Workaround (K2's lesson from yesterday): building from a private rsync snapshot build-l/snap -> build-l/sb (gitignored), re-synced before every build; the snapshot drops the
  missing file from P2's list only when it is absent. Binaries used below come from build-l/sb unless stated.
- 02:35 recovery probe weight 1/3 (dead bucket recovers after ~2 accepted probes); integer 9 rejected by parse_spec_mode (only the string "lookup" maps to it: the mock
  suite enumerates 9 as invalid) -> server mock suite 73/73 PASS; test_lookup PASS on the rebuilt snapshot. test_spec gained a `lookup` suite (mode 9 forced with 2-/4-token keys,
  auto lookup+oracle, mid-run rewind; tokens/state/history checks). Snapshot build clean.
- 02:38 locked gate chain started (first attempt 02:36 died in my lock helper: $LOG not exported; fixed) (odd CPUs, 8 threads for the correctness gates; default CPU list, 12 threads for the benches): test_spec quick -> test_verify quick ->
  test_spec_capi.py (with drafter) -> bench_spec lookup prompts n=300 (plain/dspark/lookup/auto) -> bench_spec DSpark prompts n=120. Logs build-l/data/{g1,g2,g3,b1,b2}_*.log.
  docs/SPEC_DESIGN.md section 11 written; docs/requests.md: __main__.py help text, P2 CMake hazard, new ABI symbols.
- 02:53 first locked chain done (logs build-l/data/g1..g3, b1, b2; loadavg 7-13 throughout: the other session compiles). GATES: test_verify quick PASS; test_spec_capi.py
  PASS with the real drafter (fixed T=8 and auto == plain token for token, stop / budget / seeds / ink_sample ok); test_spec quick: all 14 DSpark-double cells tokens == plain,
  state == plain; the new lookup suite: every cell tokens == plain, state == plain, history == processed tokens (incl. a mid-run rewind), 2 FAILs were my own too-strict
  assertion (lookup must fire on EVERY prompt; code_completion has no repeated 4-gram in 40 tokens) -> assertion made per config over the prompts, forced configs now start
  at T=8 (prior 0.99) so widths 2..8 are all exercised; rerun queued.
  BENCH 12 threads, lookup prompts n=300, fast attention (b1): tok/s plain / old auto (dspark) / lookup / new auto:
    lookup_docstring (p=1743) 14.01 / 21.76 / 23.41 / 24.25; lookup acceptance by match length [8-11] 2.2-2.25 per step, [12-15] 2.7-3.0, [16-31] 6.3-6.4, [32+] 6.4-6.8 of 7;
      wasted rows 10 (lookup) / 3 (auto) per 300 tokens; dspark cell: 35 mismatches vs plain = S2's fast-attention caveat above 768 keys (the DSpark path is unchanged by me;
      attribution run with build-final's binary queued), lookup / auto 0 mismatches.
    lookup_toolcall (p=568) 14.75 / 24.46 / 15.92 / 21.66: DSpark is excellent on the JSON (249/264), lookup matches are short ([8-11] 2.2/step, [12-15] 0.6/step) and in this binary
      a firing lookup always pre-empted DSpark -> auto lost 11 % vs dspark. Fix coded 02:40 (not in this binary): in auto the source with the higher PREDICTED speed-up drafts.
    lookup_typo (p=537) 14.44 / 25.51 / 29.38 / 29.80: [32+] 34 steps 238/238 rows, 0 wasted; pooled over the 3 prompts 14.39 / 23.80 / 21.51 / 24.81 (0 mismatches lookup/auto).
  BENCH DSpark prompts n=120 (b2, loadavg 10-13): pooled plain 15.15 / dspark 15.11 / lookup 15.41 / auto 15.32, 0 mismatches in all 24 cells. Per prompt auto vs plain:
    code_completion +5.3 %, tool_call -1.6 %, chat_code +4.5 %, reasoning -1.8 %, chat_answer +7.6 % (a 16-31 match, 18/18 rows), prose_continuation -3.2 %, chat_email -1.9 %,
    long_context_qa +0.9 %; lookup-only vs plain: never below -1.3 %. The prose losses are the DSpark probes (dspark mode: prose -6.2 %, chat_email -2.7 %) in a loaded window,
    and the DSpark policy's decisions differ run to run under this load (it is time-based). The lookup fired 1-5 times per 120 tokens on these prompts (8-11 matches, ~50 % rows).
- 02:55 shared tree now also fails to LINK (P2's engine_prefill.cpp references gemm_moe_nvfp4r_i16, not implemented yet), so a snapshot of the live tree is unusable.
  New isolation: build-l/snap2 = `git archive HEAD` (5be85af = the wave-2 baseline build-final) + my owned files from the working tree (spec.h, ink_abi.h, src/spec,
  ink_capi.cpp, bench_spec.cpp, tests/spec, server/backends.py, server/scheduler.py) -> build dir build-l/sb2. Everything reported from here on is sb2 (my change on top
  of the committed baseline, no P2 code). Fix coded 02:40 (predicted-speed-up arbitration lookup vs DSpark in auto) and the test fix go into sb2.

## Agent L2 (take-over after the orchestrator restart stopped L at ~02:58): finish, gate, report. Start 03:00 UTC, hard stop ~06:05 UTC.
- 03:00-03:08 read L.md, OPT_WAVE2/1, RESEARCH_WAVE2 item 2, wave2_speculation.md, all owned diffs. State verified: build-l/sb2 (git archive 5be85af + L's files) carries the
  02:40 arbitration fix (src/spec identical to the tree); only tests/spec/test_spec.cpp was edited after the snapshot (width-coverage assertion) -> test_spec rebuilt in sb2 (03:04).
  chain2.sh (pid 1617832) is alive: h1 `test_spec lookup` PASS (02:57, tokens/state/history == plain in all 6 cells); c1 = bench_spec lookup prompts x {plain,dspark,lookup,auto}
  with the arbitration fix got the lock at 03:04 (loadavg 10-11, the other session compiles); then c2 (8 DSpark prompts), r1/r2 (real server tests i..o), c3 (exact verify),
  c4 (build-final attribution of the dspark-cell mismatches), s1 (server smoke). run_real_tests.sh switched to models_fresh (= production's files) + INK_THREADS=8 as instructed.
  Server mock suite 73/73 PASS (03:03, no model). Production (pid 1489154) runs `--drafter ... --spec auto` from models_fresh: with the new libink its auto = lookup + DSpark.
- 03:08 server/backends.py: a drafter-less server now also uses the lookup path by DEFAULT (no --spec = engine default auto) and with --spec auto|lookup; --spec off / INK_SPEC=off /
  --no-spec keep the plain path; older libink.so without ink_spec_lookup: unchanged. tests/spec/test_spec_capi.py: model paths from INK_GGUF / INK_DRAFTER (old defaults kept).
  chain3.sh queued behind chain2: test_spec quick (incl. widths assertion), test_verify quick, test_spec_capi.py with and without the drafter (sb2 libink, models_fresh).
- 03:16 c1/c2 DONE (sb2 with the arbitration fix, 12 threads on the default CPUs, fast verify attention, loadavg 8.7-14.1 throughout: the other session compiles; all cells
  of a prompt are consecutive, so compare within a prompt). tok/s plain / dspark (old auto) / lookup / auto, auto vs max(plain, dspark), mismatches vs plain:
    lookup_docstring (p 1743) 14.00 / 21.30 / 24.29 / 24.39  +14.5 %  0/0/0   lookup [32+] 23 steps 160/161 rows, [16-31] 7 steps 41/41, wasted rows 1 (auto)
    lookup_toolcall  (p  568) 14.00 / 24.36 / 15.69 / 24.45   +0.4 %  0/0/0   auto: 23 short matches yielded to DSpark (predicted speed-up lower), 0 lookup steps -> = dspark
    lookup_typo      (p  537) 14.05 / 25.85 / 29.24 / 29.43  +13.8 %  0/0/0   [32+] 34 steps 238/238 rows, 0 wasted
    pooled 3 lookup prompts (n = 300): 14.02 / 23.69 / 21.62 / 25.89 (b1 before the fix: 14.39 / 23.80 / 21.51 / 24.81; the toolcall loss of 11 % is gone)
    8 DSpark prompts (n = 120): code_completion 14.79/15.38/15.46/15.26 (-0.8 %), tool_call 15.13/15.08/15.10/15.31 (+1.2), chat_code 14.74/16.62/15.39/17.29 (+4.0),
    reasoning 15.12/14.70/15.15/15.26 (+0.9), chat_answer 15.09/14.99/16.44/16.60 (+10.0: a 16-31 match, 14/14 rows), prose 15.24/14.66/15.26/15.00 (-1.6),
    chat_email 15.43/14.95/15.40/14.88 (-3.6: dspark-only is -3.1 on it = DSpark probes 0/6; the single lookup step wasted 2 rows ~ -0.5 %), long_context_qa 14.80/14.18/14.71/14.71 (-0.6)
    pooled 8 prompts: 15.04 / 15.04 / 15.35 / 15.50; lookup fired 0-5 times per 120 tokens, 3 wasted rows in 960 tokens; 0 mismatches in all 32 cells (44 cells total today).
  Noise in this window: plain itself spans 14.74-15.43 over the 8 prompts (+-2.3 %); the -3.6 % / -1.6 % cells are DSpark-policy probes under load (pre-existing behaviour, also
  in the dspark column), not lookup rows. chain2 continues: r1 (real tests i-l), r2 (m-o), c3 (exact verify), c4 (build-final attribution), s1 (server smoke).
- 03:20 r1 (real server tests with the drafter, sb2 libink, models_fresh, INK_CPUS odd, 8 threads, loadavg 9-15): test_i (greedy auto and fixed 4 == plain token for token on
  2 prompts x 64, counters present, synced) PASS, test_j (sampled fixed widths reproducible run to run, other seed differs) PASS, test_k (stop inside a verify window, slot
  history == fed tokens, next turn fully cached) PASS, test_l (tool-call turn with speculation valid) PASS. Numbers: build-l/data/ink_real_results.json (spec_* keys).
- 03:28 r2 (real server tests m, n, o; m/n on a DRAFTER-LESS server = the new default lookup path, o with the drafter; sb2 libink, models_fresh, odd CPUs, 8 threads,
  loadavg 8-16): test_m shared-prefix rewind PASS (req2 cached 2048 of LCP 2122, req3 cached 2122 exactly, rewound output == cold run), test_n Caret re-rendered turn PASS
  (turn 2 cached 2134 = turn 1's prompt, 14 new tokens, prefill 703 ms vs 45.4 s cold, output == cold), test_o rewound slot PASS (spec_synced false, T hist {1: 15},
  drafted 0, output == plain, fresh slot synced again). The rewind path keeps the lookup history consistent (truncate on ink_slot_rewind, append on the re-prefill).
  Request log of r2's drafter-less server (/tmp/inkreal_jb6vk73h/req.jsonl): all 11 requests took the speculative path (spec_mode -1 = engine default auto); the lookup drafted on
  4 of them (3/3, 3/3, 1/4, 6/9 rows accepted) while the rewound outputs still equalled the cold runs: the new server default is exercised on the real engine.
- 03:39 c3 = the 3 lookup prompts again with INK_SPEC_EXACT=1 (bit-exact decode attention kernels in verify; a second same-day measurement of c1, loadavg 12-15):
  plain / dspark / lookup / auto: docstring 14.37 / 22.19 / 24.02 / 24.48 (+10.3 % vs max), toolcall 14.81 / 25.28 / 15.75 / 25.30 (+0.1 %, 24 matches yielded to DSpark),
  typo 14.77 / 27.10 / 29.53 / 30.15 (+11.3 %); pooled 14.65 / 24.69 / 21.63 / 26.41; 0 mismatches in all 12 cells (56 cells today). Exact verify costs nothing measurable here
  (docstring verify 248 ms exact vs 244-251 ms fast at T = 8, p = 1743). c4 (build-final attribution) and s1 (server smoke) follow; three P2 runs are queued on the lock too.
- 03:42-04:08 remaining chains finished before the coordinator's stop request (nothing started afterwards): c4 attribution: build-final's OWN bench_spec (no L code) on
  lookup_docstring: plain 14.49 / auto 22.07 with 35/305 mismatches = the fast-verify-attention caveat above 768 keys pre-exists in production's DSpark auto (my runs: 0/56).
  s1 server smoke (drafter-less server, --spec lookup, odd CPUs 8 threads, docstring prompt 300 tokens): predicted tok/s off 12.56 / lookup 19.06 (97 steps, 202/205 rows) /
  auto 19.71 (213/219); auto == off token for token, lookup != off (one fast-verify flip at p = 1743, see c4; INK_SPEC_EXACT unset in the server), Caret-style re-rendered
  turn with rewind: lookup == plain (22 steps, 31/35 rows, 581 re-prefilled). chain3: test_spec quick (sb2, all suites incl. the lookup suite: tokens == plain, state == plain,
  history == processed tokens, widths T2/T4/T5/T6/T8 exercised) PASS, test_verify quick PASS, test_spec_capi.py with drafter PASS and --no-drafter PASS (auto == plain,
  fixed 8 == plain, synced 0). chain4: real tests a..h on the NEW drafter-less default (lookup path): 8/8 PASS (idle 0.25 % of a core, RSS peak 17.89 GiB, HTTP overhead
  0.036 ms/token); live tree (L + P2 i16 prefill) configured, linked 58/58 at 04:05, its test_spec quick PASS (0 failures) and test_lookup PASS: verify/decode unaffected by P2.

## FINAL (agent L2 for L, 04:15 UTC)
### 1. Result (bench_spec, 12 threads on the default CPUs, greedy n = 300 lookup prompts / 120 DSpark prompts, build-l/sb2 = 5be85af + this work, loadavg 8-15 all day:
###    another session compiled; cells of one prompt are consecutive). tok/s plain / dspark (= old auto) / lookup / auto; auto vs max(plain, dspark); mismatches vs plain
* lookup_docstring (p 1743): 14.00 / 21.30 / 24.29 / 24.39 (+14.5 %); exact verify 14.37 / 22.19 / 24.02 / 24.48 (+10.3 %). 0 mismatches.
* lookup_toolcall (p 568): 14.00 / 24.36 / 15.69 / 24.45 (+0.4 %; 23 short matches yielded to DSpark); exact 14.81 / 25.28 / 15.75 / 25.30 (+0.1 %). 0.
* lookup_typo (p 537): 14.05 / 25.85 / 29.24 / 29.43 (+13.8 %; [32+] 238/238 rows); exact 14.77 / 27.10 / 29.53 / 30.15 (+11.3 %). 0.
* pooled 3 lookup prompts: 14.02 / 23.69 / 21.62 / 25.89 (exact 14.65 / 24.69 / 21.63 / 26.41); before the arbitration fix 14.39 / 23.80 / 21.51 / 24.81 (toolcall -11 %).
* 8 DSpark prompts pooled: 15.04 / 15.04 / 15.35 / 15.50; per prompt auto vs max: -3.6 (chat_email; dspark-only -3.1 = DSpark probes) ... +10.0 (chat_answer); lookup fired
  0-5 times per 120 tokens, 3 wasted rows in 960 tokens. 0 mismatches in all 56 cells. Plain spans 14.74-15.43 in that window (+-2.3 % noise).
* Server (8 threads, odd CPUs, no drafter): predicted tok/s off 12.56 / lookup 19.06 / auto 19.71 on the docstring prompt.
### 2. Wired in
* DEFAULT: `auto` (mode 1, engine and server default) = session-lookup draft when its predicted speed-up beats the DSpark policy's, else the unchanged adaptive DSpark policy;
  without a drafter auto = lookup + plain. Server: with --drafter (production, serve.sh) nothing to change; a drafter-less server now takes the speculative path by default
  (no --spec, --spec auto|lookup); --spec off / INK_SPEC=off / --no-spec = plain path. Verify path, kVerifyMaxRows 8, ink_spec_result layout unchanged.
* FLAGS: INK_SPEC=lookup / --spec lookup / "speculation": "lookup" (mode 9 = lookup only, plain fallback); INK_SPEC_LOOKUP=0 kill switch (auto as before);
  INK_SPEC_LOOKUP_NMIN 8, _ENTER 1.0, _MAXDRAFT 7, _PROBE 8. New ABI (additive): ink_spec_lookup, ink_spec_lookup_enabled, ink_spec_last_source, ink_spec_lookup_stats.
* Files: include/ink/spec.h, src/spec/lookup.cpp (new), src/spec/spec.cpp, src/spec/policy.cpp, src/spec/CMakeLists.txt, src/capi/ink_capi.cpp, include/ink/ink_abi.h,
  bench/bench_spec.cpp, tests/spec/test_lookup.cpp (new), tests/spec/test_spec.cpp, tests/spec/spec_test_util.h, tests/spec/test_spec_capi.py (INK_GGUF/INK_DRAFTER env),
  tests/spec/prompts/lookup_*.{txt,i32} + make_lookup_prompts.py (new), server/backends.py, docs/SPEC_DESIGN.md 11, docs/requests.md (L 02:45, L2 03:25), docs/status/L.md.
  No git commits; scheduler.py untouched; nothing of P2's touched.
### 3. Gates (all PASS, logs build-l/data/)
test_lookup (index == brute force, rewind/reset chain-identical, policy) sb2 + live; test_spec quick sb2 (h2) + live tree L+P2 (h6): every cell tokens == plain, state ==
plain, lookup history == processed tokens incl. mid-run rewinds; test_verify quick (h3); test_spec_capi.py with drafter (h4) and --no-drafter (h5); server mock suite 73/73
(twice, after the backend change); real server tests i,j,k,l (r1, drafter), m,n,o (r2; m/n on the new drafter-less lookup default, rewinds consistent), a..h (r3, lookup
default); bench mismatches 0/56 cells; s1 smoke auto == off and rewound turn == plain.
### 4. Open
* Fast verify attention above 768 keys flips rare low-margin tokens (pre-existing: build-final's own DSpark auto 35/305 on docstring, c4; s1's lookup-only run differed once
  on the same prompt). Lookup adds verify steps on long copy-heavy prompts, so the exposure grows; INK_SPEC_EXACT=1 cost nothing measurable here (c3) -> consider it for the
  server, or re-measure the exact kernel at 7k contexts.
* Orchestrator: __main__.py --spec help (auto|off|lookup|2..8); README/ENGINE numbers; final idle-host run: bench_spec --prompts all --modes plain,dspark,lookup,auto @12 thr.
* Not built: filling the remaining rows of a short lookup draft with DSpark tokens (research #3), drafts > 7 rows (needs kVerifyMaxRows > 8 + verify slope), indexing files /
  tool results beyond the slot's own history (AgSpec corpus). LookupPolicy state resets with the slot and survives rewinds; DSpark probes under load still cost 1.6-3.6 % on
  prose prompts (unchanged behaviour).

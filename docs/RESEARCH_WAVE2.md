# RESEARCH_WAVE2: what else can make inkcpu faster (orchestrator synthesis, 2026-11-23)

Question from the owner after wave 1: "is this the only optimisation possible?" Four budgeted research agents (live web, 50 min / 20 searches / 15 fetches each)
covered draft-free speculation (`docs/research/wave2_speculation.md`), bytes per token (`wave2_bytes.md`), the system ceiling (`wave2_hardware.md`) and compute
kernels without VNNI (`wave2_compute.md`). This file ranks their findings against the measured profile of the wave-1 build (decode ~15 tok/s = 65 ms/token moving
3.87 GB; prefill ~85 tok/s; verify of 4 drafted tokens 2.15x a plain step; attention 8 ms/token at ctx 8000). Every number below is quoted from those files, which carry the URLs.

## The two floors on this machine
* Decode: 3.87 GB/token at the 80 GB/s streaming ceiling = 48 ms = **~20 tok/s** for any exact computation. We are at 15. The ~17 ms gap is expert phases at 52-68 GB/s
  instead of 80, ~7 ms of fixed per-token work, and attention streaming below the ceiling at long context.
* Prefill: without VNNI the exact int8 GEMM costs 5 vector uops per 64 MACs (25.6 MACs/cycle/core). The kernels run at 50-57 % of that peak. The compute agent's
  ISA floor is ~2.0-2.2 uops per 64 MACs; the best derived exact scheme is 2.75 uops (46.5 MACs/cycle).

## Ranked list

### Tier A: large, output-identical, or free
1. **Change the VM, not the code: Azure D32s_v6 (Emerald Rapids) at $1.613/h vs the current D32s_v3 at $1.536/h.** Measured read bandwidth 219 GB/s vs our 92.5; adds VNNI
   and AMX. The engine runs on it unchanged (ISA superset): bandwidth alone gives an ESTIMATED ~2x decode (~30+ tok/s at today's efficiency, ~45 at 80 % of stream); a VNNI
   prefill kernel (1 uop per 64 MACs instead of 5) would put prefill in the several-hundred tok/s range, AMX beyond that. D16s_v6 at $0.806/h (125 GB/s) already beats today's
   decode at half the price. $/Mtok drops from ~28 to ~10. Caveats: one bandwidth sample per size (cloud slices vary +-30 %: run a 10-minute STREAM/bench_stream spot
   check before committing); Azure's AMD slices are WORSE than today (D32as_v6 77 GB/s, D32as_v7 103 GB/s: CCD-bound), AWS c8a (Turin) is not (223 GB/s). Graviton4
   (c8g.8xlarge, 401 GB/s, $1.27/h) is the bandwidth-per-dollar winner but needs a NEON/SVE port of every kernel. Effort: 0 for the move, 1-2 weeks for VNNI/AMX kernels.
2. **Session-lookup drafter (suffix automaton / n-gram over prompt + generated text + tool results), DSpark as fallback.** Draft cost ~0, so the break-even is 1.15 accepted
   tokens per step at T=4 (2.43 at T=8). Evidence on the same model class: ISSTA'26 study on Qwen3-30B-A3B, Suffix Decoding 1.73x SWE-bench / 1.83x CanItEdit (EAGLE-3:
   0.41x / 0.87x); AgSpec session corpus 2.3-4.4x at batch 1 on SWE-bench; RustingStrata on Inkling 2 Mini: 112/112 suffix drafts accepted, 8.6 -> 16.3 tok/s. Expected
   here: copy-heavy edit steps 2.2-2.4x plain, whole edit turns ~1.9-2.0x (+10-15 % over today's DSpark auto), code generation neutral, prose 1.1-1.2x. Rules: fire only on
   long exact matches (n >= 8-12), draft length proportional to the match, cap rows at P(accept) >= ~0.3 because our marginal verify row costs 0.25-0.5 plain steps. Effort 2-4 days.
3. **Adaptive DSpark draft length per request** (online quantile/EMA of acceptance): prose 1.08x -> ~1.22x at T=2, pooled chat ~1.43x, code 1.76-1.80x at T=4-8 under our
   cost model; AgSpec reports +66 % throughput from the same idea. Effort 1-2 days. (Today's policy adapts T in {1,4,8} on probes; this replaces it with a per-request estimate.)

### Tier B: medium, exact integer arithmetic, float accumulation order changes (needs the parity gate, flagged like K3's x4)
4. **Prefill GEMM with int16 partial accumulation + weight-offset trick**: LUT emits w+12 as u8 (0..24), activations stay s8, 4 vpmaddubsw + 3 vpaddw + 1 vpmaddwd + 1 vpsubd
   + cvt + fma per 16 rows x 16 k = 2.75 uops per 64 MACs (1.82x the current chain; no saturation possible: pair sums <= 6144). Integer dot stays exact; the float
   accumulation moves from per-4 to per-16 weights, so prefill would no longer be bit-identical to decode (it would be MORE accurate). Prefill ESTIMATE 85 -> 130-160 tok/s
   (GEMMs 9.6 -> ~5.3 ms/token); after that the GEMM peak exceeds the DRAM ceiling at 9 tokens/expert. Also fixes the small-n tile waste (token-count-aware N tiles). Effort 1-2 weeks.
   Smaller first step: drop the sign transfer only (offset trick, 4.25 uops, +18 %) = K3's x4 idea, already prototyped (`gemm_moe_nvfp4r_x4`, +7-11 % on the MoE GEMMs).
5. **Decode GEMV 9 -> ~7.5 uops per 64 weights**: offset trick (-1) and integer UE4M3 sub-scales inside vpmaddwd (llama.cpp q4_K precedent) so cvt+fma run once per
   super-block (-1 to -1.5). Gate_up already streams at 68 of 80 GB/s, so the end-to-end gain is capped at ~5-10 % of decode. Effort 3-5 days.
6. **Exact lm_head screening**: Q8_0 copy of a rotated 25-40 % dimension prefix + per-row tail norms give strict Cauchy-Schwarz bounds; rescore only candidates on the full rows;
   Gumbel-max on bounds gives exact samples. Precedent on a 248k-row int8 head: 88.6 % of samples need nothing beyond the top-16 candidates, 0.46 % a full pass, 0 draws
   different. Saves 2-4 ms/token (3-6 %) if Inkling's candidate counts behave similarly: run a one-day offline experiment first. Effort high.
7. **Attention streaming at long context**: still 8 ms/token at ctx 8000 and 11 ms at 16000 after wave 1 (KV read 54-61 GB/s vs 80 possible); the remaining items are one barrier per
   visit, finish-imbalance of the split visits (thread 0 waits 34-46 us), and the fixed ~44 us per attention visit at any context (2 ms/token). Worth ~3-6 ms at 8k-16k. Effort 3-5 days.
8. **Phase H (down) call structure**: 144 tiles per visit, ~10 % barrier imbalance; fused 8-expert call with next-tile prefetch; ~1.5 ms/token. Effort 2-3 days.

### Tier C: precision trade-offs (owner's decision; all change outputs slightly)
9. **Q8_0 KV cache** (V first, then K): KV bytes halve; saves ~1.5-3 ms at ctx 1000, ~6 ms at 8000, ~17 ms at 32k. llama.cpp PR 7412: ppl 6.2322 -> 6.2344, KLD 0.0010.
   Caveat: sink-style attention is K-sensitive and Inkling has attn_sink_proj: test V-only first.
10. **Re-quantise the Q8_0 experts of layers 20-23 to NVFP4** (kernel exists): 271 -> 143 MB, ~1.5 ms/token (2.3 %). Unsloth's own UD-Q4_K_M differs from UD-Q4_K_XL only there
    (Q5_K) at KLD 0.0107 -> 0.0108; the Q8_0 is because those layers are MXFP8-native, not a sensitivity finding.
11. lm_head Q6_K/Q5_K: bytes -112/-173 MB but ~0-1 ms because no 6/5-bit kernel beats Q8_0's 80 GB/s here; only useful as the screening copy of item 6.

### Tier D: throughput, not latency
12. **Batch 2 (max 4) concurrent sequences per step**: the engine's own routing table gives expert bytes x1.59 at n=2 (x2.47 at n=4) with the 2.0 GB of shared tensors read once:
    0.64x bytes per emitted token at n=2 = ~24 tok/s aggregate (12 per stream), ~36 aggregate at n=4. Per-request latency never improves (every CPU MoE paper agrees). Needs the
    multix GEMV in the decode path. Useful only if several agents run at once.

### Ruled out by evidence (do not spend time here)
* Lossless weight compression: block-scaled 4-bit formats sit 1.1-1.3x above their entropy bound (<= 19 % of expert bytes) and the fastest CPU entropy decoders deliver
  ~1 GB/s/core vs the 3.1 GB/s/core we would need; DFloat11 has no CPU path.
* LUT GEMMs (T-MAC, bitnet.cpp TL2): exact 4-bit x int8 needs int16 tables (vpermw = 2 uops for 32 lookups = 16 MACs/uop vs 64 for vpmaddubsw); T-MAC's own x86 4-bit data is
  1.1-1.5x only in bandwidth-bound decode. SWAR packing: 32 MACs/uop at best. The fp32-FMA route: ~1.05x. All dominated by item 4.
* Tree verification, lookahead/Jacobi, EAGLE-3/HASS/Medusa heads (none exist for Inkling; EAGLE-3 lost to suffix decoding on edit/repair; lookahead needs 56-120x extra FLOPs),
  looped self-speculation as a sequential draft (1 pass costs ~0.5 plain steps per token, and the sibling varloop model collapses at 1 pass: 0.7 % vs 47.4 % DeepSWE).
* Expert caching / prefetch (hot-16 experts = 1.97 GB vs 35.8 MiB L3; the same token's three passes are separated by >= 1.07 GB of other weights), pass reordering (forbidden by the
  cross-pass attention over two residual blocks), SMT (wave 1: 20-32 threads lose), 1 GiB pages (no gain over 2 MiB for streaming), Azure AMD slices (CCD-bound bandwidth).
* Same-VM levers left: keep engine threads off CPU 0/1 (already) and move IRQ affinity off CPU 0 (CPU 0 takes 15x the Hyper-V VMBus interrupts of any other vCPU): 1-3 % jitter at most.

## Suggested order
1. Ten-minute spot test of D32s_v6 / D16s_v6 bandwidth (bench_stream); if it holds, the move is the biggest single win at ~5 % more per hour (or half the price at D16s_v6).
2. Session-lookup drafter + adaptive draft length (items 2-3): the largest software win for the editor workload, exact, ~1 week.
3. Prefill int16 chain (item 4) with the parity gate, then VNNI/AMX variants if the VM changes.
4. Items 5-8 as polish; items 9-10 only if the owner accepts measured KLD deltas.

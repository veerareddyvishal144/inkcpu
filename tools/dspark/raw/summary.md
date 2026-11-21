## Pooled acceptance (all prompts, full-length drafts only; L = mean tokens per verify step = accepted drafts + 1)

| mode | draft n (T=n+1) | prompts done | steps | **L +- SE** | accepted drafts/step | acc@1 | acc@2 | acc@3 | acc@4 | acc@5 | acc@6 | acc@7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 3 (T=4) | 8/8 | 260 | **2.37 +- 0.08** | 1.37 | 0.64 | 0.41 | 0.32 | - | - | - | - |
| greedy | 4 (T=5) | 8/8 | 238 | **2.57 +- 0.10** | 1.57 | 0.62 | 0.41 | 0.31 | 0.24 | - | - | - |
| greedy | 7 (T=8) | 8/8 | 214 | **2.77 +- 0.15** | 1.77 | 0.58 | 0.37 | 0.28 | 0.22 | 0.14 | 0.10 | 0.07 |
| t1prob | 3 (T=4) | 5/8 | 155 | **2.51 +- 0.10** | 1.51 | 0.69 | 0.47 | 0.35 | - | - | - | - |

acc@j = fraction of verify steps in which draft position j was accepted (cumulative: positions 1..j all accepted).

## Conditional acceptance P(position j accepted | positions < j accepted)

| mode | n | c@1 | c@2 | c@3 | c@4 | c@5 | c@6 | c@7 |
|---|---|---|---|---|---|---|---|---|
| greedy | 3 | 0.64 | 0.64 | 0.77 | - | - | - | - |
| greedy | 4 | 0.62 | 0.66 | 0.75 | 0.77 | - | - | - |
| greedy | 7 | 0.58 | 0.63 | 0.76 | 0.78 | 0.62 | 0.76 | 0.73 |
| t1prob | 3 | 0.69 | 0.68 | 0.74 | - | - | - | - |

## Per prompt: L (steps) 

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

## Bytes-model speed-up estimates (ROUTING.md, mean expert-union bytes; decode bandwidth-bound; speed-up = L * B1 / S(T))

S(T) = N + E(T) + Dr: N = 2014.6 MB, B1 = 3861.9 MB/token, E(T) from ROUTING.md (d). Dr variants: (R) ROUTING.md 635.8 MB drafter + 1 lm_head pass = 1125.5 MB; (M) refined = 523.8 layers + 9.1 head + 489.7 batched base lm_head pass + n x 102.4 MB (one pass over a precomputed bf16 [200058 x 256] matrix lm_head x hidden_out per chained position); (C) llama.cpp as implemented = 523.8 + 9.1 + 2 x n x 492 MB (two lm_head passes per position).

(Q) = (M) with the drafter layers and the folded matrix stored Q8_0 (278.3 MB layers + 54.4 MB per chained position; Unsloth reports the same acceptance for its Q8_0 drafter).

| mode | n | T | L measured | L* (R) | L* (M) | L* (Q) | speed-up (R) | speed-up (M) | speed-up (Q) | speed-up (C) | S(T) (M) MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 3 | 4 | 2.37 | 1.99 | 2.05 | 1.94 | 1.19x | 1.16x | 1.22x | 0.91x | 7898 |
| greedy | 4 | 5 | 2.57 | 2.16 | 2.24 | 2.13 | 1.19x | 1.15x | 1.21x | 0.85x | 8666 |
| greedy | 7 | 8 | 2.77 | 2.59 | 2.75 | 2.60 | 1.07x | 1.00x | 1.06x | 0.65x | 10631 |
| t1prob | 3 | 4 | 2.51 | 1.99 | 2.05 | 1.94 | 1.26x | 1.23x | 1.29x | 0.96x | 7898 |

## llama.cpp reference timing (this machine, 8 threads, shared host: indicative; tokens/s from the log's `decoded` line)

| mode | n | mean ms per step: drafter | target verify (T tokens) | feature injection | tokens/s (mean over prompts) | loadavg before (min..max 1-min) |
|---|---|---|---|---|---|---|
| greedy | 3 | 111 | 515 | 2.3 | 4.19 | 2.7..14.3 |
| greedy | 4 | 167 | 779 | 3.4 | 3.15 | 13.5..19.0 |
| greedy | 7 | 225 | 862 | 3.8 | 3.19 | 6.9..17.1 |
| t1prob | 3 | 159 | 572 | 2.9 | 3.29 | 13.9..15.7 |

## Histogram of accepted drafts per step (pooled; count of steps with k accepted)

- greedy n=3: 0:93, 1:60, 2:25, 3:82
- greedy n=4: 0:90, 1:51, 2:24, 3:17, 4:56
- greedy n=7: 0:89, 1:46, 2:19, 3:13, 4:18, 5:7, 6:6, 7:16
- t1prob n=3: 0:48, 1:34, 2:19, 3:54

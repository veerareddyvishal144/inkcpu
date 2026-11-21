## Traces

| data set | forwards used | tokens per forward | tokens | forwards skipped (warm-up etc.) |
|---|---|---|---|---|
| wiki | 4 | 512+512+512+512 | 2048 | 1 |
| code | 1 | 1255 | 1255 | 0 |
| chat | 1 | 895 | 895 | 0 |

## (a) distinct experts per visit for T consecutive tokens (mean over all 48 visits and all windows)

| T | independent 128(1-(1-8/128)^T) | wiki distinct | code distinct | chat distinct | mean distinct | wiki / indep | code / indep | chat / indep | mean / indep |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 15.50 | 12.59 | 12.89 | 12.55 | 12.68 | 0.812 | 0.832 | 0.810 | 0.818 |
| 3 | 22.53 | 16.39 | 16.71 | 16.31 | 16.47 | 0.727 | 0.742 | 0.724 | 0.731 |
| 4 | 29.12 | 19.65 | 20.00 | 19.52 | 19.72 | 0.675 | 0.687 | 0.670 | 0.677 |
| 5 | 35.30 | 22.54 | 22.93 | 22.35 | 22.61 | 0.639 | 0.650 | 0.633 | 0.640 |
| 6 | 41.10 | 25.14 | 25.63 | 24.89 | 25.22 | 0.612 | 0.624 | 0.606 | 0.614 |
| 7 | 46.53 | 27.50 | 28.12 | 27.20 | 27.61 | 0.591 | 0.604 | 0.585 | 0.593 |
| 8 | 51.62 | 29.68 | 30.44 | 29.31 | 29.81 | 0.575 | 0.590 | 0.568 | 0.577 |

By visit group (mean of the data sets), distinct experts and ratio to independent:

| visit group | T=2 | T=4 | T=8 |
|---|---|---|---|
| prelude L0-5 | 13.75 (0.887) | 23.37 (0.803) | 38.07 (0.738) |
| core pass 1 | 12.32 (0.795) | 18.66 (0.641) | 27.60 (0.535) |
| core pass 2 | 12.69 (0.819) | 19.50 (0.669) | 29.00 (0.562) |
| core pass 3 | 12.65 (0.816) | 19.65 (0.675) | 29.69 (0.575) |
| postlude L18-19 (NVFP4) | 11.65 (0.752) | 17.12 (0.588) | 24.85 (0.481) |
| postlude L20-23 (Q8_0) | 12.71 (0.820) | 19.68 (0.676) | 29.34 (0.568) |

## (b) overlap of the expert sets of the SAME token between the passes of the looped layers (layers 6-17; mean over layers and tokens)

| data set | |S1 and S2| of 8 | |S2 and S3| | |S1 and S3| | |S1 and S2 and S3| | |S1 or S2 or S3| (distinct over 3 passes, of 24) |
|---|---|---|---|---|---|
| wiki | 4.96 (62.0%) | 4.38 (54.8%) | 3.86 (48.3%) | 3.18 (39.7%) | 13.98 |
| code | 4.55 (56.9%) | 4.62 (57.7%) | 3.57 (44.6%) | 3.02 (37.7%) | 14.28 |
| chat | 4.69 (58.7%) | 4.28 (53.5%) | 3.41 (42.6%) | 2.84 (35.5%) | 14.47 |
| mean | 4.73 (59.2%) | 4.43 (55.3%) | 3.61 (45.1%) | 3.01 (37.7%) | 14.24 |
| independent | 0.50 (6.2%) | 0.50 | 0.50 | 0.031 | 22.53 |

Per core layer (mean of the data sets): |S1&S2|, |S2&S3|, |S1&S3|, union of 3

| layer | S1&S2 | S2&S3 | S1&S3 | union3 |
|---|---|---|---|---|
| 6 | 4.08 | 4.36 | 3.22 | 14.97 |
| 7 | 4.37 | 4.62 | 3.70 | 14.29 |
| 8 | 4.71 | 5.04 | 4.11 | 13.60 |
| 9 | 4.84 | 5.05 | 4.20 | 13.45 |
| 10 | 4.71 | 4.74 | 3.85 | 13.91 |
| 11 | 4.97 | 4.85 | 4.20 | 13.51 |
| 12 | 4.86 | 4.61 | 3.70 | 13.98 |
| 13 | 4.97 | 4.80 | 3.99 | 13.63 |
| 14 | 4.98 | 4.43 | 3.72 | 13.97 |
| 15 | 5.24 | 4.06 | 3.43 | 14.12 |
| 16 | 5.58 | 4.30 | 3.50 | 13.74 |
| 17 | 3.49 | 2.25 | 1.73 | 17.72 |

## (c) popularity skew: share of the selections taken by the 16 most used experts of a layer (uniform: 12.5% + sampling noise)

| layer | experts | wiki: per visit / pooled over passes | code: per visit / pooled over passes | chat: per visit / pooled over passes |
|---|---|---|---|---|
| 0 | NVFP4 | 40.8% / 39.7% | 41.6% / 41.6% | 45.1% / 45.1% |
| 1 | NVFP4 | 45.8% / 44.7% | 50.7% / 50.7% | 39.5% / 39.5% |
| 2 | NVFP4 | 34.5% / 33.1% | 48.4% / 48.4% | 35.9% / 35.9% |
| 3 | NVFP4 | 31.3% / 30.4% | 39.9% / 39.9% | 34.2% / 34.2% |
| 4 | NVFP4 | 31.3% / 29.8% | 43.9% / 43.9% | 34.8% / 34.8% |
| 5 | NVFP4 | 33.5% / 31.4% | 43.4% / 43.4% | 38.9% / 38.9% |
| 6 | NVFP4 | 46.1% / 40.7% | 38.3% / 36.4% | 44.4% / 40.4% |
| 7 | NVFP4 | 50.0% / 46.3% | 51.8% / 50.9% | 38.2% / 36.6% |
| 8 | NVFP4 | 56.1% / 53.8% | 55.4% / 54.6% | 44.6% / 44.1% |
| 9 | NVFP4 | 56.8% / 54.7% | 57.3% / 56.7% | 42.4% / 41.7% |
| 10 | NVFP4 | 53.5% / 50.7% | 49.5% / 48.8% | 41.0% / 39.1% |
| 11 | NVFP4 | 52.3% / 49.8% | 55.4% / 54.8% | 41.3% / 40.4% |
| 12 | NVFP4 | 46.5% / 43.2% | 46.4% / 45.0% | 40.8% / 37.9% |
| 13 | NVFP4 | 50.8% / 47.7% | 53.3% / 53.0% | 43.7% / 41.0% |
| 14 | NVFP4 | 54.6% / 50.4% | 51.1% / 50.0% | 51.6% / 47.4% |
| 15 | NVFP4 | 57.1% / 52.2% | 52.4% / 48.5% | 51.7% / 47.8% |
| 16 | NVFP4 | 52.4% / 46.2% | 53.7% / 49.7% | 55.7% / 49.6% |
| 17 | NVFP4 | 77.3% / 65.5% | 71.8% / 59.5% | 76.2% / 62.3% |
| 18 | NVFP4 | 63.0% / 61.4% | 83.6% / 83.6% | 51.0% / 51.0% |
| 19 | NVFP4 | 52.9% / 51.2% | 65.7% / 65.7% | 44.4% / 44.4% |
| 20 | Q8_0 | 56.3% / 56.0% | 61.8% / 61.8% | 44.6% / 44.6% |
| 21 | Q8_0 | 56.8% / 55.9% | 63.9% / 63.9% | 46.6% / 46.6% |
| 22 | Q8_0 | 55.1% / 54.1% | 58.0% / 58.0% | 50.1% / 50.1% |
| 23 | Q8_0 | 57.4% / 56.8% | 45.0% / 45.0% | 56.1% / 56.1% |
| null (random routing, same token count; 1 visit / 3 passes pooled) |  | 14.3% / 13.5% | 14.7% / 13.8% | 15.2% / 14.1% |

mean per-visit top-16 share over the 24 layers (mean of data sets): 49.8%, min 35.1% (L3), max 75.1% (L17)

## (d) expert bytes of a T-token verify step (MB = 10^6 B; 44 NVFP4 visits + 4 Q8_0 visits; per expert NVFP4 4478984 B, Q8_0 8460288 B)

| T | NVFP4 visits MB | Q8_0 visits MB | experts MB (mean of data sets) | x one-token step | x one-token step per token | independent MB | independent x | wiki x | code x | chat x |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1576.6 | 270.7 | 1847.3 | 1.00 | 1.000 | 1847.3 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 2498.0 | 430.1 | 2928.1 | 1.59 | 0.793 | 3579.2 | 1.94 | 1.57 | 1.61 | 1.57 |
| 3 | 3245.4 | 557.7 | 3803.1 | 2.06 | 0.686 | 5202.8 | 2.82 | 2.05 | 2.09 | 2.04 |
| 4 | 3888.0 | 665.9 | 4553.8 | 2.47 | 0.616 | 6725.0 | 3.64 | 2.46 | 2.50 | 2.44 |
| 5 | 4458.2 | 760.4 | 5218.7 | 2.82 | 0.565 | 8152.0 | 4.41 | 2.82 | 2.86 | 2.79 |
| 6 | 4974.6 | 845.1 | 5819.7 | 3.15 | 0.525 | 9489.8 | 5.14 | 3.14 | 3.20 | 3.11 |
| 7 | 5447.2 | 922.3 | 6369.6 | 3.45 | 0.493 | 10744.1 | 5.82 | 3.44 | 3.51 | 3.40 |
| 8 | 5883.2 | 993.1 | 6876.3 | 3.72 | 0.465 | 11919.9 | 6.45 | 3.71 | 3.80 | 3.66 |

## Break-even of speculative decoding on bandwidth grounds

Plain decoding reads B1 = N + E(1) per token. A speculative step reads S(T) = N + E(T) + Dr (N: non-expert bytes incl. lm_head, read once for the whole
T-token verify batch; E(T): measured expert bytes of T consecutive tokens; Dr: drafter bytes, 635.8 MB + optionally one lm_head pass) and yields L tokens
(L in [1, T]: accepted drafts + 1). Speculation wins on bandwidth iff S(T)/L < B1, i.e. L > L*(T) = S(T)/B1. `max x` = T*B1/S(T) is the speed-up if EVERY draft
is accepted (L = T): an upper bound, not an estimate.


### engine plan C (F32->bf16 at load, lm_head 200058 rows): N = 2014.6 MB, lm_head = 489.7 MB, E(1) = 1847.3 MB, B1 = 3861.9 MB

drafter bytes Dr = 1125.5 MB (drafter + one lm_head pass):

| T | verify step N+E(T) MB | step S(T) MB incl. drafter | L* if routing were independent | L* wiki | L* code | L* chat | L* mean | max x at L=T (mean) |
|---|---|---|---|---|---|---|---|---|
| 2 | 4943 | 6068 | 1.74 | 1.57 | 1.58 | 1.56 | 1.57 | 1.27 |
| 3 | 5818 | 6943 | 2.16 | 1.79 | 1.81 | 1.79 | 1.80 | 1.67 |
| 4 | 6568 | 7694 | 2.55 | 1.99 | 2.01 | 1.98 | 1.99 | 2.01 |
| 5 | 7233 | 8359 | 2.92 | 2.16 | 2.18 | 2.15 | 2.16 | 2.31 |
| 6 | 7834 | 8960 | 3.27 | 2.32 | 2.34 | 2.30 | 2.32 | 2.59 |
| 7 | 8384 | 9510 | 3.60 | 2.46 | 2.49 | 2.44 | 2.46 | 2.84 |
| 8 | 8891 | 10016 | 3.90 | 2.59 | 2.63 | 2.56 | 2.59 | 3.08 |

drafter bytes Dr = 635.8 MB (drafter only, no lm_head pass):

| T | verify step N+E(T) MB | step S(T) MB incl. drafter | L* if routing were independent | L* wiki | L* code | L* chat | L* mean | max x at L=T (mean) |
|---|---|---|---|---|---|---|---|---|
| 2 | 4943 | 5578 | 1.61 | 1.44 | 1.46 | 1.44 | 1.44 | 1.38 |
| 3 | 5818 | 6453 | 2.03 | 1.67 | 1.68 | 1.66 | 1.67 | 1.80 |
| 4 | 6568 | 7204 | 2.43 | 1.86 | 1.88 | 1.85 | 1.87 | 2.14 |
| 5 | 7233 | 7869 | 2.80 | 2.04 | 2.06 | 2.02 | 2.04 | 2.45 |
| 6 | 7834 | 8470 | 3.14 | 2.19 | 2.22 | 2.17 | 2.19 | 2.74 |
| 7 | 8384 | 9020 | 3.47 | 2.33 | 2.36 | 2.31 | 2.34 | 3.00 |
| 8 | 8891 | 9527 | 3.77 | 2.46 | 2.50 | 2.44 | 2.47 | 3.24 |


### GGUF as stored A (F32 router/controllers, lm_head 201024 rows): N = 2163.5 MB, lm_head = 492.1 MB, E(1) = 1847.3 MB, B1 = 4010.8 MB

drafter bytes Dr = 1127.9 MB (drafter + one lm_head pass):

| T | verify step N+E(T) MB | step S(T) MB incl. drafter | L* if routing were independent | L* wiki | L* code | L* chat | L* mean | max x at L=T (mean) |
|---|---|---|---|---|---|---|---|---|
| 2 | 5092 | 6219 | 1.71 | 1.55 | 1.56 | 1.54 | 1.55 | 1.29 |
| 3 | 5967 | 7094 | 2.12 | 1.77 | 1.78 | 1.76 | 1.77 | 1.70 |
| 4 | 6717 | 7845 | 2.50 | 1.95 | 1.97 | 1.94 | 1.96 | 2.04 |
| 5 | 7382 | 8510 | 2.85 | 2.12 | 2.14 | 2.11 | 2.12 | 2.36 |
| 6 | 7983 | 9111 | 3.19 | 2.27 | 2.29 | 2.25 | 2.27 | 2.64 |
| 7 | 8533 | 9661 | 3.50 | 2.40 | 2.44 | 2.39 | 2.41 | 2.91 |
| 8 | 9040 | 10168 | 3.79 | 2.53 | 2.57 | 2.51 | 2.54 | 3.16 |

drafter bytes Dr = 635.8 MB (drafter only, no lm_head pass):

| T | verify step N+E(T) MB | step S(T) MB incl. drafter | L* if routing were independent | L* wiki | L* code | L* chat | L* mean | max x at L=T (mean) |
|---|---|---|---|---|---|---|---|---|
| 2 | 5092 | 5727 | 1.59 | 1.42 | 1.44 | 1.42 | 1.43 | 1.40 |
| 3 | 5967 | 6602 | 2.00 | 1.64 | 1.66 | 1.64 | 1.65 | 1.82 |
| 4 | 6717 | 7353 | 2.37 | 1.83 | 1.85 | 1.82 | 1.83 | 2.18 |
| 5 | 7382 | 8018 | 2.73 | 2.00 | 2.02 | 1.98 | 2.00 | 2.50 |
| 6 | 7983 | 8619 | 3.06 | 2.15 | 2.17 | 2.13 | 2.15 | 2.79 |
| 7 | 8533 | 9169 | 3.38 | 2.28 | 2.31 | 2.26 | 2.29 | 3.06 |
| 8 | 9040 | 9676 | 3.67 | 2.41 | 2.45 | 2.38 | 2.41 | 3.32 |


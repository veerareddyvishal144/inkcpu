Greedy, 8 threads, taken under `tools/with_bench_lock.sh` after the acceptance matrix finished (my own matrix runs were not under the lock; other agents' load is in the last column). n=0 is plain decoding through the same binary (no draft). `ms per step: target verify` is the time of the target forward over T tokens, `drafter` the time of the drafting call (the ctx injection is ~2 ms).

| prompt | n (T=n+1) | tokens/s (80 new tokens) | speed vs n=0 | L | ms per step: drafter | ms per step: target verify | loadavg before / after |
|---|---|---|---|---|---|---|---|
| chat_answer | 0 (T=1) | nan | nanx | nan | nan | nan | 9.03 / 8.67 |
| chat_answer | 3 (T=4) | 2.97 | nanx | 1.77 | 106 | 495 | 8.67 / 11.35 |
| chat_answer | 7 (T=8) | 1.73 | nanx | 1.79 | 223 | 852 | 11.35 / 13.11 |
| code_completion | 0 (T=1) | nan | nanx | nan | nan | nan | 13.11 / 13.39 |
| code_completion | 3 (T=4) | 5.82 | nanx | 3.67 | 105 | 518 | 13.39 / 14.01 |
| code_completion | 7 (T=8) | 5.55 | nanx | 5.92 | 217 | 839 | 14.01 / 14.53 |

# Colibri (JustVugg/colibri) vs inkcpu on Inkling 2 Mini, same host, same GGUF (orchestrator, 2026-11-24)

## Colibri's published claims (README / docs/inkling.md, v2.2.0 and dev 1403453 of 2026-11-23)
* Inkling-2-Mini UD-Q4_K_XL: 40.9-41.1 tok/s decode on CUDA (RTX PRO 4000), 39.2-39.4 on Vulkan, **8.0-8.1 on the CPU (i5-13500, 6 threads)**; prompt 18.2-18.8 tok/s on that CPU.
  UD-IQ2_XXS: 46 tok/s GPU, 8.4 CPU. PR 2937 took the CPU path from 4.3 to 6.4 (then to 8.0 with a 16-row interleaved NVFP4 form, `GQ_NVFP4_R16`).
* Inkling-2 drafter (`INK2_MTP=1`, from the official checkpoint's mtp/ safetensors, not from the GGUF): acceptance 3.32 tokens per 4-row verify on code, 2.44 on prose
  (ours: 3.1-3.5 code, ~2.0 prose), but "a forward over n rows took about n decode steps" on their disk-bound setup, so it ships OFF by default. Exact: same bits with and without.
* The engine streams experts from disk into a RAM cache (26 GB budget here, 99.5 % hit rate once warm); GGUFs are read in place ("link" mode), NVFP4 experts byte for byte.

## Same-host measurement (this VM: Xeon 8171M 16 cores, 92.5 GB/s; GGUF /data/inkling-mini/models_fresh/Inkling-2-Mini-UD-Q4_K_XL.gguf; prompt 616 tokens, 100 generated, greedy)
Colibri dev 1403453 built with `make -C c inkling colibri` (gcc -O3 -march=native -fopenmp), model imported with `coli import` (link mode), run as shipped
(`SNAP=<dir> ./c/inkling --chat -p ... -n 100`; its OpenMP tuner re-execs with 16 physical-core threads; placement verified: 16 distinct CPUs). Three consecutive
runs, loadavg 5-14 (another session was compiling), `build-final2/colibri_ab/`:

| engine | prefill 616 tokens | decode | notes |
|---|---|---|---|
| Colibri warm1 / run2 / run3 | 68.5 / 67.1 / 66.7 s = 9.0-9.2 tok/s | 4.65 / 4.65 / 4.57 tok/s | phases per run: expert matmuls 58-59 s, attention 11.5 s, other 12 s; cache hit 99.5 %, RSS 15.8 GB |
| inkcpu build-final2 (same window) | 113.5-113.8 tok/s (chunk 512) | 15.3-15.5 tok/s (ctx 570) | plain decode, no speculation; with the lookup drafter copy-heavy outputs reach 24-29 |

**inkcpu is 3.3x faster at decode and ~12.5x at prompt processing than Colibri on identical hardware and model file.** Colibri's 8 tok/s claim comes from a
Raptor Lake desktop core (AVX-VNNI, higher clocks); on this Skylake-SP server it reaches 4.6. Its drafter, if enabled, would cost ~n plain steps per n-row
verify here too (it has no batched small-n expert kernels), so it cannot win back the gap with speculation.

Measurement pitfalls recorded: (1) never pin Colibri with `taskset` + `OMP_PLACES`: its tuner resets the affinity mask and re-execs, and the combination put all 16
threads on CPU 0 (0.7 tok/s); (2) its first run fills the expert cache from the page cache (fill 7-14 s) and saves a usage history, so compare warm runs only.

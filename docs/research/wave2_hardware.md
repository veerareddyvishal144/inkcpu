# Wave 2 hardware research (R2c): is the D32s_v3 the system ceiling for inkcpu?

Written 2026-11-23 by research agent R2c. Budget used: 20 web searches, 15 page fetches (+ Spare Cores JSON API calls via curl). FINAL.

Engine facts assumed (from the brief): decode moves 3.87 GB/token (so decode_est = 0.8 x read-bandwidth / 3.87 GB); prefill is int8-ALU-bound at
~0.86 TMAC/s peak on 16 threads of the 8171M (5 vector uops per 64 int8 MACs, no VNNI) and measures ~85 tok/s; decode measures ~15 tok/s.

## 0. Host facts (read on this VM, read-only)
- Azure IMDS: vmSize = **Standard_D32s_v3**, region **westus2**. lscpu: Xeon Platinum 8171M (Skylake-SP), 16 cores/32 threads, 1 socket, 1 NUMA node,
  L3 35.8 MiB, AVX-512 F/DQ/BW/VL/CD only (no vnni/bf16/amx). Hypervisor: Microsoft. 125 GiB RAM.
- Guest knobs as found: THP = `madvise`; `kernel.numa_balancing = 0`; no cpufreq driver (governor is not controllable from an Azure guest);
  1 GiB hugetlb pages = 0; cmdline has no isolcpus/nohz_full/rcu_nocbs; clocksource = hyperv_clocksource_tsc_page; kernel 6.8.0-1068-azure,
  CONFIG_HZ=1000, CONFIG_NO_HZ_FULL=y (built in, not enabled).
- /proc/interrupts: CPU0 has taken 992 M interrupts vs ~630-640 M on the next-busiest vCPUs; CPU0 carries **15x** the HYP (Hyper-V/VMBus
  callback) count of a typical vCPU (23.2 M vs ~1.6 M) and **4x** the CAL (function-call IPI) count (473 M vs ~120 M). No single device IRQ
  line is pinned >80 % to CPU0. See section 3.

## 1. Price of the current VM
- Standard_D32s_v3, Linux PAYG, West US 2: **$1.536/h** (cloudprice.net https://cloudprice.net/vm/Standard_D32s_v3 ; Spare Cores
  https://sparecores.com/server/azure/Standard_D32s_v3 lists 1.536 on-demand / 0.2839 spot in westus2). 3-yr RI 0.5896 $/h (Flexera).
- $/Mtok (single stream, at measured 15 tok/s): 1.536 / (15 x 3600 / 1e6) = **$28.4 per M output tokens**; at the 19-20 tok/s exact floor: ~$22.

## 2. Bandwidth and price evidence per candidate (measured, not vendor)
Source for bandwidth: Spare Cores `bw_mem` (lmbench, all vCPUs, 512 MB block = DRAM-resident, read/rdwr/write), pulled 2026-11-23 from
https://keeper.sparecores.net/server/<vendor>/<size>/benchmarks (web pages: https://sparecores.com/server/azure/Standard_D32s_v6 etc.).
Each is ONE observation on ONE host; cloud slices share the socket, so expect +/-30 % (the AMD Azure tuning guide literally tells you to run STREAM
to detect noisy neighbours: "a Standard_F8as_v6 placed on a single CCD should achieve around 50 GB/s", https://docs.amd.com/api/khub/documents/f7sv5mpDknw7erVRoZmtSA/content).
Prices: Spare Cores `/prices` endpoint, cheapest US-region Linux on-demand; cross-checked with cloudprice.net where noted.

| Instance | CPU (all-core GHz) | vCPU/RAM | read / write GB/s @512 MB | int8 ISA | $/h |
|---|---|---|---|---|---|
| Azure D32s_v3 (this VM class; SC host was 8272CL, ours 8171M) | Skylake/Cascade 2.6 | 32/128 | 108 / 49 (ours measured: 92.5 read) | AVX-512, no VNNI | 1.536 |
| Azure D16s_v3 | same | 16/64 | 81 / 47 | same | 0.768 |
| Azure D64s_v3 (2 NUMA nodes) | same | 64/256 | 217 / 96 | same | 3.072 |
| Azure D16s_v6 | EMR 8573C 3.0 | 16/64 | 125 / 71 | VNNI+AMX+BF16 | 0.806 |
| **Azure D32s_v6** | EMR 8573C 3.0 | 32/128 | **219 / 94** | VNNI+AMX+BF16 | 1.613 (cloudprice 1.613 East US) |
| Azure D64s_v6 | EMR 8573C | 64/256 | 264 / 113 | VNNI+AMX | 3.226 |
| Azure D96s_v6 (noisy sample) | EMR 8573C | 96/384 | 151 / 70 | VNNI+AMX | ~4.84 |
| Azure D32as_v6 | Genoa 9V74 3.7 | 32/128 | **77 / 57** | AVX-512 VNNI (Zen 4, 1 zmm/clk) | 1.453 |
| Azure D32s_v7 | Granite Rapids 6973P-C 3.6 | 32/128 | 169 / 66 | VNNI+AMX+FP16 | 2.032 (Central US) |
| Azure D64s_v7 | GNR 6973P-C | 64/256 | 337 / 132 | VNNI+AMX | 4.064 |
| Azure D32as_v7 | Turin 9V45 (4.5 boost) | 32/128 | 103 / 70 | AVX-512 VNNI (Zen 5, 2 zmm/clk) | 1.453 |
| Azure D64as_v7 | Turin 9V45 | 64/256 | 212 / 133 | Zen 5 VNNI | ~2.906 |
| Azure D96as_v7 | Turin 9V45 | 96/384 | 320 / 179 | Zen 5 VNNI | ~4.36 |
| AWS m7i.8xlarge | SPR 8488C 3.2 | 32/128 | 160 / 84 | VNNI+AMX | 1.6128 |
| AWS m7i.16xlarge | SPR 8488C | 64/256 | 231 / 96 | VNNI+AMX | 3.2256 |
| AWS m8i.8xlarge | GNR | 32/128 | 180 / 80 | VNNI+AMX | 1.6934 |
| AWS c8i.8xlarge | GNR | 32/64 | 181 / 85 | VNNI+AMX | ~1.50 (not pulled) |
| AWS m7a.8xlarge | Genoa 9R14 3.7 | 32/128 | 150 / 105 | Zen 4 VNNI | 1.8547 |
| AWS m7a.16xlarge | Genoa 9R14 | 64/256 | 274 / 156 | Zen 4 VNNI | ~3.709 |
| **AWS c8a.8xlarge** | Turin | 32/64 | **223 / 132** | Zen 5 VNNI | 1.7243 |
| **AWS c8g.8xlarge** (Arm, needs NEON port) | Graviton4 Neoverse V2 | 32/64 | **401 / 176** | NEON SDOT/I8MM | 1.2717 |
| AWS c8g.16xlarge (Arm) | Graviton4 | 64/128 | 469 / 201 | NEON SDOT/I8MM | 2.5434 |
| Hetzner AX162 bare metal | Genoa 9454P 48c/96t | 96/128-256 | not published; 12ch DDR5-4800 = 460 GB/s theoretical, real depends on DIMM population | Zen 4 VNNI | EUR 612-842/month + setup (Oct 2026 list; was EUR 199-319 in 2024-25) |
| OVH Advance (Turin 9255 24c / 9355 32c) | Turin | 48-64 thr | not published | Zen 5 VNNI | from $198/month base config |

Socket-level context (shivam2003.com 2026 roundup, vendor peaks): Graviton4 ~537 GB/s, Turin ~460, Genoa ~350, Emerald Rapids ~307,
Sapphire Rapids ~250 GB/s. Intel's own STREAM Triad study on EC2 (relative only): m7i vs m6a = 1.77x at 8 vCPU, 2.35x at 16, 3.19x at 32;
m7i vs m7a up to 1.48x at 8 vCPU (https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2024-02/stream-on-amazon-ec2-m7i-vs-m6a-and-m7a.pdf).
Graviton4 single core DRAM read 37.0 GB/s (Arm ASCT, https://dev.to/aws-builders/characterize-the-aws-graviton-memory-subsystem-using-asct-4e24).

Int8 peak per core (sources): AMX = 2048 int8 ops/cycle/core (1024 MAC) vs 256 ops (128 MAC) for 2-port AVX-512 VNNI (Intel Hot Chips 33 /
AMX whitepaper, https://www.hc33.hotchips.org/assets/program/conference/day1/HC2021.C1.4%20Intel%20Arijit.pdf , Intel's measured AMX/VNNI
speed-up 7.8x). uops.info VPDPBUSD zmm: Zen 5 and Emerald Rapids reciprocal throughput 0.50 (2/clk = 128 MAC/clk); Ice Lake 1.00
(https://uops.info/html-instr/VPDPBUSD_ZMM_K_ZMM_ZMM.html). Zen 4: 256-bit double-pumped, ~1 zmm/clk = 64 MAC/clk (inferred, not read off
uops.info). Our 8171M: 0.86 TMAC/s / 16 cores / ~2.1-2.6 GHz = ~21-26 MAC/clk/core effective.

## 3. Engine estimates and ranking (32-vCPU slices unless noted)
Method. decode_est = 0.8 x read GB/s / 3.87. prefill_peak_ratio = (cores x MAC/clk x GHz) / 0.86 TMAC/s; "VNNI prefill" = 85 tok/s x ratio x 0.6
(kernel efficiency haircut for a new VNNI path; our current path already runs near its uop limit); "AMX prefill" = VNNI x 3 (conservative;
theoretical 8x). All single-stream. $/Mtok = $/h / (decode x 0.0036). Current-VM row uses measured numbers.

| Rank | Instance | decode est (tok/s) | prefill est VNNI (AMX) tok/s | $/h | $/Mtok at est decode | Notes |
|---|---|---|---|---|---|---|
| 0 | Azure D32s_v3 (now) | 15 measured (19 floor) | 85 measured | 1.536 | 28.4 (22 at floor) | baseline |
| 1 | **Azure D32s_v6** (EMR, 16c) | **45** | ~365 (~1100 with AMX) | 1.613 | **~10** | Same price as today, 2.4x read BW, AMX. Best "no-port" jump on Azure. |
| 1b | Azure D16s_v6 | 26 | ~180 (~550 AMX) | 0.806 | ~8.7 | Cheapest $/Mtok on Azure x86; already beats current decode at half the price. |
| 2 | **AWS c8a.8xlarge** (Turin, Zen 5) | 46 | ~485 (no AMX) | 1.7243 | ~10.4 | Best x86 VNNI-only prefill; 64 GB RAM is enough for 17.6 GB model. |
| 3 | AWS m8i.8xlarge (GNR) | 37 | ~440 (~1300 AMX) | 1.6934 | ~12.7 | |
| 4 | AWS m7i.8xlarge (SPR) | 33 | ~390 (~1150 AMX) | 1.6128 | ~13.6 | |
| 5 | Azure D32s_v7 (GNR) | 35 | ~440 (~1300 AMX) | 2.032 | ~16 | Lower measured BW than D32s_v6 in this sample and 26 % dearer. |
| 6 | AWS m7a.8xlarge (Genoa) | 31 | ~225 | 1.8547 | ~16.6 | |
| 7 | Azure D64s_v6 | 55 | ~730 (~2200 AMX) | 3.226 | ~16.4 | 2 NUMA nodes likely; needs NUMA-aware expert placement. |
| 8 | Azure D64s_v7 | 70 | ~880 (~2600 AMX) | 4.064 | ~16 | Highest Azure x86 decode found. |
| 9 | Azure D32as_v7 (Turin) | 21 | ~485 | 1.453 | ~19 | Azure's Turin slice measured only 103 GB/s read. |
| 10 | Azure D32as_v6 (Genoa) | 16 | ~225 | 1.453 | ~25 | WORSE decode than today: 77 GB/s read (CCD-bound slice; AMD guide: ~10 GB/s/core). Avoid. |
| Arm | **AWS c8g.8xlarge** (Graviton4) | **83** | ~330-650 (SDOT/I8MM, unverified) | 1.2717 | **~4.3** | 401 GB/s read for 32 vCPU (no SMT = 32 real cores). Requires a NEON port of every kernel. |
| Arm | AWS c8g.16xlarge | 97 | 2x above | 2.5434 | ~7.3 | |
| Bare metal | Hetzner AX162 (9454P 48c) | 30-60 (depends on populated channels; 150-300 GB/s plausible) | ~640 (48 Zen 4 cores) | EUR 612/mo ~ $0.90/h-equiv | ~5-9 | Monthly commitment, setup fee EUR 304; no STREAM published; 2024 price was EUR 199. |

Prefill 16-vCPU x86 variants: halve the prefill column (cores halve), decode falls less than half (D16s_v6: 125 vs 219 GB/s read).

Takeaways for Q1
- On Azure the single biggest same-price jump is **D32s_v3 -> D32s_v6**: ~2.4x measured read bandwidth (219 vs 92.5-108 GB/s), AVX-512 VNNI
  (removes the 5-uop int8 penalty: ~7x int8 peak at 3.0 GHz) plus AMX (another 3-8x if a tile kernel is written), for +5 % price ($1.613 vs 1.536).
  Expected: decode ~45 tok/s, prefill several hundred tok/s with VNNI, 1000+ with AMX.
- AMD Azure slices are the trap: D32as_v6 measured 77 GB/s read (below today), D32as_v7 103 GB/s. AMD's own guide explains it: bandwidth is
  per-CCD (~80 GB/s/CCD, ~10 GB/s/core), so a 16-core slice sees 1-2 CCDs' worth, not the 12-channel socket. AWS's Turin c8a.8xlarge (223 GB/s)
  does not show this, so it is a placement/SKU property, not a Zen property.
- The true bandwidth ceiling per dollar is Arm: c8g.8xlarge gives 401 GB/s read for $1.27/h (~83 tok/s decode est, ~$4/Mtok) but every kernel
  (NVFP4 dequant, Q8_0 GEMV, attention) needs a NEON/SVE2 port and the int8 path changes to SDOT/SMMLA.
- The on-prem alternative (Hetzner AX162) is cheaper per hour only if run >~50 % of the month, and Hetzner's 2026 repricing (EUR 612-842/mo
  vs EUR 199-319 in 2024-25) erased most of its advantage over an Azure spot/RI D32s_v6.

## 4. Q2: is 92.5 GB/s typical for a 32-vCPU 8171M VM; any lever?
- Yes. Spare Cores measured 108 GB/s read / 49 GB/s write on a D32s_v3 (8272CL host, DDR4-2933 6ch); our 8171M host is DDR4-2666 6ch
  = 128 GB/s theoretical per socket, so 92.5 GB/s is 72 % of the socket's theoretical peak, i.e. the VM is already getting most of one socket.
  D16s_v3 measured 81 GB/s read, so halving the VM only loses 25 % of bandwidth; D64s_v3 (2 sockets) measured 217 GB/s read.
- Levers that exist: (a) **D64s_v3 = 2 sockets = 2x bandwidth for 2x price** ($3.072/h), but it is two NUMA nodes (MS CoreMark page notes
  E64-32s_v3 spans 2 NUMA nodes) so the engine must place experts/KV per node and avoid cross-socket traffic; (b) moving generation
  (section 3) is strictly better per dollar than going wider in Dv3.
- Levers that do NOT exist on this VM: SNC/cluster-on-die is a host BIOS setting, invisible to the guest; memory interleaving is fixed by the
  host; 1 GiB pages are available (`pdpe1gb` flag) but see Q5 for expected effect; there is no cpufreq driver, so no governor control.
- Published STREAM for Dv3/Ev3/Fsv2 from Microsoft: none. Azure's STREAM page only covers HBv2/3/4/5 (HBv4 ~650-780 GB/s DDR5, HBv3 ~330-350,
  HBv2 ~260) https://learn.microsoft.com/en-us/azure/high-performance-computing/performance-benchmarking/stream-benchmark .

## 5. Q3: the core-0 half-speed phenomenon
Evidence that vCPU 0 is the hypervisor's "landing" CPU, not an inherently slower core:
- On this VM, /proc/interrupts shows CPU0 with 15x the HYP (Hyper-V VMBus callback) interrupts and 4x the CAL IPIs of any other vCPU
  (23.2 M vs ~1.6 M; 473 M vs ~120 M), and 992 M total interrupts vs ~635 M for the next vCPU.
- Red Hat solution 6930111: on Azure VMs with up to 8 CPUs all Mellanox (SR-IOV) NIC interrupts land on CPU0 (https://access.redhat.com/solutions/6930111).
- Ubuntu linux-azure patch "PCI: hv: Only reuse existing IRTE allocation for Multi-MSI" fixed a regression where "all the interrupts now
  happen on 1 physical CPU" and made Multi-MSI vCPU assignment round-robin (https://lists.ubuntu.com/archives/kernel-team/2022-November/134537.html).
- Hyper-V stimer0 "Direct Mode" exists precisely to stop timer interrupts polluting the VMBus interrupt stream that is bound to CPU0
  (https://lkml.iu.edu/hypermail/linux/kernel/1710.3/05597.html).
- Also plausible but unproven: Hyper-V's root partition/host agents stealing the physical core that backs vCPU0 (noisy neighbour). We found
  no Microsoft statement that vCPU0 is scheduled differently.
Remedies (all guest-side, none measured in public sources for inference): keep the engine's threads off CPU0 (and its sibling, CPU16 under
2-way SMT numbering - check `lscpu -e` for the sibling id) and give the OS/network/timers that core; `irqbalance --banirq` or write
/proc/irq/*/smp_affinity to move device IRQs off the engine's cores; `isolcpus=/nohz_full=/rcu_nocbs=1-15,17-31` on the kernel cmdline to
quiet the compute vCPUs (CONFIG_NO_HZ_FULL=y is built into 6.8.0-azure, only the boot parameter is missing); a kernel with the Multi-MSI
round-robin fix (6.8 has it). If the engine already pins 15 or 31 threads and skips CPU0, there is nothing more to gain here; the remaining
loss is one core (1/16 = 6 % of ALU, ~0 % of bandwidth).

## 6. Q4: Apple silicon / Strix Halo as perspective
Measured Inkling 2 Mini numbers found (all single-stream decode unless stated):
- Strix Halo (Ryzen AI Max+ 395, 256 GB/s): llama.cpp Vulkan ~27-28 tok/s decode, ~481 tok/s prefill (HN https://news.ycombinator.com/item?id=50581260);
  llama.cpp ROCm 23.8 tok/s; AMD engineer's custom kernels 44.8 tok/s (https://runtimewire.com/article/gemini-4-neon-wrote-amd-kernels-that-nearly-doubled-inkling-2-mini-decode-speed);
  GUFO/HIP engine 29-30 tok/s, llama.cpp prompt 346-520 tok/s (https://llamaperf.com/gpu/strix-halo-128gb , https://github.com/testyfishy/inkling2-mini-strixhalo-gufo).
- Ryzen AI Max+ PRO 495 (192 GB): llama.cpp ROCm 30.7-31.4 tok/s decode, 264 tok/s prefill at 9k; Vulkan 23.8-23.9 tok/s, 361 tok/s prefill
  (https://petronellatech.com/ai/llm-benchmarks/hardware/ryzen-ai-max-pro-495/).
- Apple: M5 Pro Mac mini llama.cpp IQ4_NL 29.95 tok/s (only llama.cpp-on-Mac datapoint found); M4 Max mlx-serve 8-bit 84 tok/s decode,
  1010 tok/s prefill; M5 Ultra 118 tok/s, 4900 tok/s prefill (https://huggingface.co/ddalcu/Inkling-2-Mini-MLX-8bit); M3 Ultra oMLX 40.3 tok/s
  decode, 1169 tok/s prefill, MTP off (https://getrigbase.com/guides/run-inkling-2-mini-locally/); M5 Max MTPLX 85 tok/s plain, 128.8 with MTP
  (https://mtplx.com/models/inkling-2-mini/); EXO vendor claim 210 tok/s M5 Ultra / 125 tok/s M5 Max (unverified).
Perspective: an M4 Max (546 GB/s) decodes ~5-6x faster than this VM and ~2x faster than a D32s_v6 estimate; Strix Halo (256 GB/s) lands where
the D32s_v6 estimate lands (28-45 tok/s). None are rentable per hour in the same way; they are capital purchases (prices not researched here).

## 7. Q5: same-VM Linux levers (bandwidth-bound inference)
- NUMA balancing: already 0 here, and this VM is 1 NUMA node, so no effect. Evidence it matters on multi-node boxes: ubergarm (llama.cpp
  #12088) 5.9 -> 8.9 tok/s on R1-671B by collapsing SNC NUMA nodes and disabling balancing (https://github.com/ggml-org/llama.cpp/discussions/12088);
  V-Seek paper: balancing off + interleave best (https://arxiv.org/pdf/2503.17422); conflicting advice in llama.cpp #19102 for mmap loads.
- Huge pages: THP=madvise here (engine must madvise(MADV_HUGEPAGE) or use `always`). 2 MiB pages remove essentially all TLB cost for
  sequential weight streaming; 1 GiB pages add little for streaming: Paul Khuong's data shows 1 GiB beats 2 MiB only for random access over
  multi-GB (40 % faster page walks, 1 GiB TLB has only 4 L1 entries) and is equal when the 2 MiB TLB reach suffices
  (https://pvk.ca/Blog/2014/02/18/how-bad-can-1gb-pages-be/). AMD's Azure guide recommends explicit hugetlbfs for memory-bound work and THP=never
  only for latency-sensitive work. Expected gain here: low single digits for the top-8 expert gather (random 2-MiB-page touches across 17 GB),
  ~0 for dense streams. No published STREAM 1G-vs-2M delta found.
- CPU governor: not exposed in Azure guests (no cpufreq); AMD guide's `cpupower -g performance` advice applies to bare metal only.
- Timer tick / nohz_full / irq affinity: see section 5. CONFIG_NO_HZ_FULL=y is compiled in; add `nohz_full=1-15,17-31 rcu_nocbs=1-15,17-31`
  (or isolcpus) at boot and pin engine threads. No public measurement of the effect on LLM decode; HPC practice puts OS-jitter removal at
  1-3 % for tightly synchronised phases (our per-layer barriers qualify).
- SMT at the hypervisor: not controllable. Using 16 threads (one per physical core) vs 32 is the guest-side equivalent; for bandwidth-bound
  decode 16 threads usually reach the same bandwidth, and it frees the sibling for prefetch or the OS.
- Noisy-neighbour check: re-run a STREAM/bw_mem read at 512 MB-4 GB periodically; if it drops well below 92 GB/s, redeploy the VM (new host).

## 8. What I could not find
- Any Microsoft-published STREAM figure for D/E/F general-purpose sizes (only HB-series).
- Absolute STREAM Triad MB/s in Intel's m7i-vs-m7a PDF (ratios only).
- A published llama.cpp Inkling 2 Mini number on M4 Max specifically (only MLX numbers on M4 Max; llama.cpp only on M5 Pro).
- Hetzner's actual DIMM population / STREAM for AX162, and current per-SKU prices for OVH Turin configs with >=128 GB.
- uops.info Zen 4 VPDPBUSD row (inferred 1 zmm/clk from the 256-bit double-pumped design).
- Any measurement isolating 1 GiB hugetlb vs 2 MiB THP on CPU LLM decode, or nohz_full/isolcpus on decode tok/s.
- Confirmation from Microsoft of why vCPU0 runs slower beyond the interrupt evidence above.

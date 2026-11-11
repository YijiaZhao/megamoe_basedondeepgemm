# MegaMoE based on DeepGEMM (H20 / B200 / SM120)

Expert-parallel MoE layer kernels for decode-size batches, built on DeepGEMM: four explicit
Hopper/H20 APIs plus one shared Fable dynamic-M frontend, the Blackwell/B200 kernels under
`sm100_b200/`, the SM120 (RTX PRO 6000 / 6000D) EP decode kernel (`fp4_mega_moe_sm120`) and the SM120 TP kernel
(`mxfp4_moe_tp_sm120`). Split and Fused retain independent workspaces, layouts, schedulers, and ABI contracts.

Best measured MegaMoE kernel time (single kernel, forced-balanced, E384 / H3072 / I1280 / top-8 / EP8,
GPU 0, median of the last 3 of 30 streamed replays, µs at global M = 2 / 4 / 8 / 16; full tables in
[Performance results](#performance-results)):

| Platform | Kernel | M2 | M4 | M8 | M16 |
|---|---|---:|---:|---:|---:|
| 8x B200 (1965 MHz) | W4A4 MXFP4, `sm100_b200/` | 35.0 | 38.3 | 43.7 | 47.6 |
| 8x B200 (1965 MHz) | W4A4 NVFP4, `sm100_b200/` | 34.4 | 39.5 | 44.6 | 50.8 |
| 8x B200 (1965 MHz) | W-MXFP4 x A-FP8, `sm100_b200/` | 35.7 | 38.9 | 46.1 | 54.0 |
| 8x H20-3e (1830 MHz) | MXFP4 fused | 38.8 | 47.4 | 56.7 | 76.2 |
| 8x H20-3e (1830 MHz) | QoQ (W4A8-int) fused | 37.1 | 45.2 | 54.0 | 72.6 |

## Execution model: split (two kernels) versus fused (one kernel)

| | `*_mega_moe_split` | `*_mega_moe_fused` and every B200 kernel |
|---|---|---|
| CUDA kernels per MoE layer | 2: `sm90_mxfp4_mega_moe_l1_impl` (dispatch + L1) then `sm90_mxfp4_mega_moe_l2_impl` (L2 + combine) | 1 persistent kernel: dispatch + L1 + L2 + combine |
| L1 -> L2 hand-off | across the launch boundary through a ring workspace | inside the kernel: L1 epilogue TMA-stores the quantised SwiGLU output to a per-expert pool (resident in L2 cache) and releases an arrival counter; the L2 task acquires it |
| Where the time is read (nsys) | L1 kernel start -> L2 kernel end, launch gap included | one kernel start -> end |
| Frontend (router + quant + top-8) | separate kernel, both models | same |

Same node, same session, Mega-only, GPU 0, median of the last 3 replays
(`delivery/four_api_fable_timeline_last3_r4_20260909.md`; the fused kernel of that date, before its later optimisations):

| Precision | M | Split (2 kernels) | Fused (1 kernel) |
|---|---:|---:|---:|
| MXFP4 | 2 | 53.9 | 46.9 |
| MXFP4 | 8 | 97.3 | 59.1 |
| MXFP4 | 16 | 147.8 | 74.0 |
| QoQ | 2 | 49.5 | 43.8 |
| QoQ | 8 | 88.0 | 54.9 |

The split path was not optimised after 2026-09-03; the launch boundary alone is worth 6-7 µs at M = 2, and the fused
kernel additionally overlaps L1 tails with L2 starts and L2 tails with the combine. Both split APIs pass the same 8-rank
exact-quantised-reference gate as the fused ones (`delivery/four_api_correctness_h20_20260903.txt`: cos_min >= 0.99991,
norm ratio 0.997-1.0000, all four APIs PASS). Correctness of the B200 kernels: relative RMSE 0.23-0.25 % on every rank
against a pure-torch reference at every M (see the B200 document).

## Optimisation approach

The workload is decode-size: 1–2 tokens per expert per rank, so every GEMM tile is a 16-token (padded) slab against
128–256 weight rows. Almost no math; the kernel time is a chain of latencies (NVLink round trips, HBM weight
streaming, intra-kernel dependencies) plus the wait for the slowest rank. The optimisation work followed four
principles; the per-item tables with the measured effect of each change are in
[docs/B200_MEGAMOE_RESULTS.md](docs/B200_MEGAMOE_RESULTS.md) (B200, items 1–11) and
[docs/H20_MEGAMOE_RESULTS.md](docs/H20_MEGAMOE_RESULTS.md) (H20, env-knob table).

**1. Remove cross-rank round trips.** Each NVLink barrier costs a full round trip plus the wait for the slowest rank.
The upstream design had three (before the dispatch pull, before the combine, after the workspace cleanup). They are
replaced by counters embedded in the data flow: the source rank *pushes* routed rows straight into the destination's
expert pool (one remote ticket per row, first row prefetched before the ticket returns), then adds 1 to every rank's
DONE count; the CTA that finishes a rank's last L2 task adds 1 to every rank's DONE2 count (B200: through a
shared-memory mailbox so the sys-scope release fence does not stall the epilogue warp); a parity-rotated pool removes
the post-cleanup barrier. H20 additionally publishes per-token arrival counters so the combine warps start per token
(fine combine; on B200 the same idea was slower and is off).

**2. Start streaming weights before the routing is known.** Weight bytes dominate the L1 phase (31 MB of W1 plus 16 MB
of W2 per rank at M8, HBM-bound over 148 SMs). For tiny M every local expert owns a fixed (expert, N-block) -> SM slot
set (static slots), so the weight loader can stream an expert's rows as soon as its live count turns non-zero, during
the dispatch, and empty experts are skipped at run time by a 64-bit active mask. The slot map is closed-form, so each
warp role enumerates only its own slots (no scheduler iteration per expert). H20: stream-K unit ranges over all SMs
when there are fewer L1 tasks than SMs, wide 512-row L1 tasks at M = 16, K-block prefetch into L2 while waiting for
activations.

**3. Shorten every intra-kernel hand-off.** DONE is acquired once per CTA by a dispatch warp and broadcast with the 48
local counts through shared memory (the loader / MMA / epilogue warps spin on a shared flag instead of polling the
global word); the L1 -> L2 dependency is a per-expert arrival count released by the L1 epilogue after its TMA store;
TMEM is freed after the combine instead of on the critical path; per-task warp reductions are hoisted (L2 task total
computed once). The combine work is split into (token, 512 B chunk) items over all combine warps.

**4. Fit the tile to the tensor core, not the other way round.** swap-AB (weights on the MMA M side, tokens on N) so
that the 16-token slab is the N = 16 operand; 2-CTA `tcgen05` block-scaled MMA with UTCCP-loaded scale planes (B200);
K block 512 for the packed-FP4 rows (one 128 B swizzle atom per stage, 6 instead of 12 stages per L1 task); QoQ on
H20 folds the second scale into the int8 weight at decode time.

**Where the remaining time goes** (B200, MXFP4, M8, kernel-internal `globaltimer` stamps, the rank that enters last so
no cross-rank wait is included; ~39 µs total):

| Segment | ~µs | Bound by |
|---|---:|---|
| Dispatch chain: remote row write + completion wait + DONE | 7 | one NVLink round trip (latency, not bandwidth) |
| L1 phase: 160 CTA tasks on 148 SMs, 2 waves | 11 | HBM streaming of W1 during the first wave; the second wave overlaps with the W2 prefetch |
| L1 -> L2 hand-off + L2 phase | 5 | dependency wait + W2 streaming + remote BF16 stores |
| DONE2 wait + combine | 5 | the slowest rank, one more NVLink round trip |
| Kernel entry, barriers, cleanup | rest | |

What GPU 0 reads on top of that (5–9 µs) is the wait for the other ranks to enter the kernel, i.e. launch skew outside
the kernel.

**Tried and dropped** (each measured on 8 ranks, details and numbers in the B200 document): half-N tiles (BLOCK_N 64,
twice the tasks; a task is not shorter, the per-stage overhead dominates), split-K for the second L1 wave (correct;
the dense slot map loses the early weight streaming and the half tasks compete with the W2 prefetch for HBM), fusing
L1 and L2 into one task with the SwiGLU output kept on chip (bounded by the L2 phase minus the extra W2 streaming per
task, ~1.5–2 µs at best; blocked on B200 by a tensor-core stall, measured slower on H20), per-token fine combine on
B200, `cp.async` token tiles, one SF TMA per K block, deterministic dispatch slots, per-warpgroup L1 -> L2 signalling.
FlashInfer's CuTeDSL MegaMoE (SM100 NVFP4) uses the same single-kernel structure with the FC1 output through a global
buffer; its 128-token tiles leave it at DeepGEMM-upstream parity for this batch size.

## Validated APIs

```python
deep_gemm.mxfp4_mega_moe_split
deep_gemm.qoq_mega_moe_split
deep_gemm.mxfp4_mega_moe_fused
deep_gemm.qoq_mega_moe_fused
```

The E2E path for all four APIs uses the same frontend:

```python
deep_gemm.fable_router_quant_topk_frontend(
    hidden_states, router_weight, buffer, quant="mxfp4"  # or "qoq"
)
```

The Fable frontend performs, in one CUDA kernel:

```text
bf16 router logits (fp32 accumulate, bf16 rounding)
+ activation quantization (FP8 e4m3 per K128 group, or INT8 per row)
+ TopK8 (value desc, expert id asc on ties)
+ softmax over the 8 selected logits
```

The launch shape follows the problem shape (`csrc/fable_frontend.h`, `select_fe_path`):
`m <= 2` rows per rank (H 3072, E 384, top-8; the M = 2 .. 16 pipeline on 8 ranks) run the
CUDA-core K-split router `router_cc_lean_kernel` (77 CTAs x 5 experts x 4 K-part warps, weights
streamed straight into registers, one 32-bit key per (token, expert)); with
`DG_FE_SELECT_IN_MEGA=1` the frontend stops after the keys and the fused MegaMoE prologue selects
the top-8 + softmax. `m <= 16` rows run the mma.sync m16n8k16 router (experts on the MMA M
dimension, fragment-permuted weights) on the 96-CTA grid; every other shape runs the WMMA router.

## Validated hardware and software

| Item | Configuration |
|---|---|
| GPU | 8 x NVIDIA H20-3e |
| Architecture | SM90a |
| GPU memory | 143771 MiB per GPU |
| SM count | 78 per GPU |
| Locked SM clock | 1830 MHz |
| Python | 3.12 |
| PyTorch | 2.11.0+cu130 |
| CUDA toolkit | 13.0 |
| Distributed backend | NCCL, 8 ranks |
| Profiler | NVIDIA Nsight Systems, CUDA + NVTX trace, CUDA Graph node tracing |

Validated model shape:

```text
Experts:      384
Local experts: 48 per rank
Hidden:       3072
Intermediate: 1280
TopK:         8
EP:           8
Attention TP: 4 for E2E timelines
Attention DP: 2 for E2E timelines
Global M:     2, 4, 8, 16
```

Architecture diagram: [`docs/H20_FOUR_API_FUSED_ARCHITECTURE.html`](docs/H20_FOUR_API_FUSED_ARCHITECTURE.html).

### Clean-host Docker quick start

The validated image is pinned by digest:

```text
docker.io/lmsysorg/sglang@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7
```

Create the validated container on an 8-GPU H20 host:

```bash
docker pull \
  docker.io/lmsysorg/sglang@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7

docker run -d \
  --name four_api_build \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 64g \
  -v /raid:/raid \
  -w /raid/kimi \
  docker.io/lmsysorg/sglang@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7 \
  sleep infinity
```

The commands in the next sections run inside this container unless explicitly
marked as host-side. The container is not privileged, so GPU clocks are locked
on the host (`nvidia-smi -lgc 1830,1830`; `scripts/capture_four_api_h20_timelines_host.sh`
does it before launching the collector in the container).

## Clone and build

```bash
git clone --recursive \
  --branch main \
  https://github.com/YijiaZhao/megamoe_basedondeepgemm.git
cd megamoe_basedondeepgemm

export CUDA_HOME=/usr/local/cuda
export DG_CUTLASS_INCLUDE_PATH=$PWD/third-party/cutlass/include
bash develop.sh          # builds the extension; the fused MegaMoE JIT-compiles on first use
```

## Correctness verification

All gates run on one 8-GPU H20 node with the library defaults (`DG_FE_SELECT_IN_MEGA=1` where
the pipeline uses it) on the same kernels the performance table was captured with.
Details and per-cell tables: [`docs/fe5_correctness.md`](docs/fe5_correctness.md).

* **End-to-end vs a pure-torch real-MoE reference** (`tests/test_four_api_correctness.py
  --frontend fe --reference torch-moe`, 8 ranks): the reference computes the bf16 router GEMM
  in fp32 -> bf16-rounded logits -> top-8 (value desc, index asc) -> fp32 softmax over the 8
  logits -> torch per-token quantisation of x -> dequantised expert GEMMs (both layers,
  exact-quantised weights, SwiGLU, clamp) -> weighted combine, with the frontend's REAL routing
  of random hidden rows; none of the frontend / MegaMoE kernels is in the reference.
  Pass criteria: finite output, `cos_min >= 0.99`, `0.97 <= norm ratio <= 1.03`, per-(token,
  slot) `--slot-check` clean on 8/8 ranks. Result on this tree (T = 1 / 2 / 8 / 16 / 32 rows per
  rank, both fused APIs): frontend top-8 index sets equal the torch router's on every token,
  softmax weights within 1 fp32 ulp, MXFP4 x byte-identical to the torch cast (QoQ: <= 1 int8
  step on ~0.5 elements per 1000, an exact-.5 rounding-tie difference, x_sf identical);
  `cos_min` MXFP4 >= 0.99992, QoQ >= 0.99992, norm ratio 0.9998 .. 1.00004.
* **Forced-balanced routing** (`DG_FE_FORCE_BALANCED=1`): the frontend runs, then its routing
  is replaced by the balanced assignment the profiler's `DG_PROFILE_FORCE_BALANCED=1` memcpy
  writes (`expert = s * 48 + (g + 7 s) % 48`, weight 1/8, unrouted inactive rows) and the torch
  reference routes with it: all cells PASS (`cos_min` MXFP4 >= 0.99996, QoQ >= 0.99992).
* **50-seed sweep** (14 shapes x {SELECT_IN_MEGA 0, 1} x {normal, balanced}, 292 800 evaluated
  tokens): 0 failures, 0 top-8 disagreements, 0 Mega-prologue vs python-decode differences;
  worst `cos_min` MXFP4 0.99975 (M = 128 / 256), QoQ 0.99990.
* **Zero-row gate** (`tests/test_four_api_correctness.py --zero-rows K [--zero-ranks ...]`, 8 ranks, both
  fused APIs, `DG_FE_SELECT_IN_MEGA` 0 and 1): all-zero hidden rows (the E2E padding rows: T = 1 with
  zero rows on the ranks that own no token at M2 / M4, T = 2 with one zero and one real row per rank,
  T = 8 swapab path) -- the frontend leaves every zero row unrouted on the cc path, the kernel output of
  every zero row is exactly 0, and x / x_sf / the routing / y of the other rows are bit-identical between
  the pipeline with `DG_FE_ZERO_ROW_UNROUTED` 0 and 1 (5 seeds per cell; T = 2 with the split-K tail
  `DG_FP4_SPLITK_L1=1`: <= 2 bf16 ulps on 2 of 24 576 elements in one QoQ seed, bit-identical with the
  tail off). `tests/fe_dump_compare.py --zero-rows 1` knob 0 vs 1, 64 seeds: differences only in the
  routing outputs (`topk_idx`, `topk_weights`, keys) of the zero rows, `x` / `x_sf` identical.
* **Select-in-Mega gate** (`tests/test_select_in_mega.py`, 8 ranks, 50 seeds, M 2 / 16, both
  quants): 0 topk mismatches, y bit-identical between select-in-frontend and select-in-Mega.
* **Frontend output layout gate** (`tests/fe_dump_compare.py`): the frontend runs alone for
  64 seeds x rows {1, 2} x {mxfp4, qoq} and every buffer the fused Mega consumes (`x`, `x_sf`,
  `topk_idx`, `topk_weights`, the ticket area, the select-in-Mega `x` / `x_sf`, the compact
  key array) is byte-compared between two builds or env configurations, padding rows included;
  `--ref-torch` compares `x` / `x_sf` with the torch per-token casts.

Commands (inside the container, repo root; 8 GPUs unless noted):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1
# frontend standalone, local M = 1/2/4/8/16/32/64, both quants
torchrun --standalone --nproc_per_node=8 tests/test_fable_frontend_correctness.py
# four explicit APIs in one process; exact quantised references (synthetic balanced routing)
torchrun --standalone --nproc_per_node=8 tests/test_four_api_smoke.py
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py
# frontend + fused Mega vs the pure-torch MoE reference, T rows per rank (T = 1 2 8 16 32)
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --frontend fe --reference torch-moe --tokens 1
DG_FE_FORCE_BALANCED=1 torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --frontend fe --reference torch-moe --tokens 1 --seeds 50
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens 32 --hot-rows 12 --slot-check
# zero-row gate: M2-like owner layout (zero rows on ranks 1,2,3,5,6,7), mixed rows, swapab path
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py --apis mxfp4_mega_moe_fused qoq_mega_moe_fused \
    --frontend fe --reference torch-moe --tokens 1 --zero-rows 1 --zero-ranks 1,2,3,5,6,7 --seeds 5
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py --apis mxfp4_mega_moe_fused qoq_mega_moe_fused \
    --frontend fe --reference torch-moe --tokens 2 --zero-rows 1 --seeds 5 --zero-rows-y-ulp 2
python3 tests/fe_dump_compare.py --env-a DG_FE_ZERO_ROW_UNROUTED=0 --env-b DG_FE_ZERO_ROW_UNROUTED=1 --zero-rows 1 --seeds 64  # 1 GPU
bash tests/fe5_gates.sh                                     # the gate set above, T = 1 2 8 16 32 + slot checks
OUT=/raid/kimi/results/fe5c bash tests/fe5c_sweep.sh ; python3 scripts/summarize_fe5c_sweep.py /raid/kimi/results/fe5c/*.log
python3 tests/fe_dump_compare.py --root-a <reference checkout> --root-b . --seeds 64 --ref-torch   # 1 GPU
```

## Performance results

Per-platform result documents (one table + measurement method + reproduce each):

| Platform | Kernels | Mega-only, forced-balanced, GPU 0, µs at global M = 2 / 4 / 8 / 16 | Document |
|---|---|---|---|
| 8x H20-3e (SM90, 1830 MHz) | MXFP4 / QoQ fused (this repo) | 38.8 / 47.4 / 56.7 / 76.2 (MXFP4), 37.1 / 45.2 / 54.0 / 72.6 (QoQ) | [docs/H20_MEGAMOE_RESULTS.md](docs/H20_MEGAMOE_RESULTS.md) |
| 8x B200 (SM100, 1965 MHz) | W-MXFP4xA-FP8 (optimised upstream kernel) / W4A4 MXFP4 / W4A4 NVFP4 (new kernels) (`sm100_b200/`, on AichenF DeepGEMM `megamoe_nvfp4_dev`) | 35.7 / 38.9 / 46.1 / 54.0 (FP8 act), 35.0 / 38.3 / 43.7 / 47.6 (MXFP4), 34.4 / 39.5 / 44.6 / 50.8 (NVFP4) | [docs/B200_MEGAMOE_RESULTS.md](docs/B200_MEGAMOE_RESULTS.md) |
| 8x B200 (SM100, 1965 MHz) | Upstream baseline: [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) main `78b6900` W-MXFP4xA-FP8 `fp8xfp4` MegaMoE, kernel unmodified | 47.2 / 50.5 / 53.2 / 61.2 | [docs/B200_MEGAMOE_RESULTS.md](docs/B200_MEGAMOE_RESULTS.md) |
| 8x B200 (SM100, 1965 MHz) | **E2E** (frontend kernel + forced-balanced memcpy + MegaMoE, one CUDA graph): W-MXFP4xA-FP8 / W4A4 MXFP4 / W4A4 NVFP4 | 44.0 / 48.5 / 53.8 / 63.0 (FP8 act), 40.8 / 44.8 / 51.5 / 58.9 (MXFP4), 41.7 / 46.4 / 52.3 / 60.0 (NVFP4) | [docs/B200_MEGAMOE_RESULTS.md](docs/B200_MEGAMOE_RESULTS.md) |
| 8x H20-3e (SM90, 1830 MHz) | **E2E** (same graph, padding rows unrouted): MXFP4 / QoQ fused | 44.6 / 54.0 / 62.2 / 80.9 (MXFP4), 44.8 / 54.2 / 60.4 / 80.6 (QoQ) | [docs/H20_MEGAMOE_RESULTS.md](docs/H20_MEGAMOE_RESULTS.md) |

Same model scale for all rows (E384 / 48 local, H3072, I1280, top-8, EP8) and the same forced-balanced routing; every column is the
kernel span on GPU 0, median of the last 3 of 30 streamed replays (nsys on H20, torch-profiler kernel durations on B200). The E2E
rows span from the start of the frontend kernel (router + top-8 + softmax + activation quantisation) to the end of the MegaMoE
kernel, with the forced-balanced routing written by one memcpy node in between (frontend sources for B200: `sm100_b200/fe_sm100/`). The two
B200 rows were measured in the same Slurm job and container; upstream DeepGEMM has no Hopper MegaMoE and no FP4-activation (W4A4)
MegaMoE, so those kernels have no upstream counterpart.

SM120 (8x RTX 6000D, PCIe P2P, no NVLink; CUDA-event timing, so not in the nsys table above): `fp4_mega_moe_sm120`,
one EP kernel for W4A8 (MXFP4 weights x MXFP8 activations) and W4A4 MXFP4 / NVFP4. This README's shape, EP8, cold L2
(256 MB written before each replay), W4A8: 43.6 / 57.7 / 78.6 / 116.3 µs at M2 / 4 / 8 / 16; all formats, method and
correctness: [sm120_6000d/W4A4.md](sm120_6000d/W4A4.md).

## Relevant source files

```text
csrc/fable_frontend.cu
csrc/fable_frontend.h
csrc/apis/mega.hpp
csrc/apis/mega_fused.hpp
csrc/jit_kernels/impls/sm90_mxfp4_mega_moe.hpp
csrc/jit_kernels/impls/sm90_fp4_mega_moe_h20_fused.hpp
deep_gemm/mega/__init__.py
deep_gemm/mega/fused.py
deep_gemm/include/deep_gemm/impls/fable_frontend_device.cuh
deep_gemm/include/deep_gemm/impls/fable_cc_select.cuh
deep_gemm/include/deep_gemm/impls/sm90_mxfp4_mega_moe.cuh
deep_gemm/include/deep_gemm/impls/sm90_fp4_mega_moe_h20_fused.cuh
deep_gemm/include/deep_gemm/impls/sm90_fp4_mega_moe_h20_fused_body.inl
deep_gemm/include/deep_gemm/impls/sm120_mxfp4_mega_moe.cu
deep_gemm/mega/sm120.py
tests/test_mega_moe_sm120.py
deep_gemm/include/deep_gemm/impls/sm120_fp4_mega_moe.cu
deep_gemm/mega/sm120_fp4.py
tests/test_mega_moe_sm120_fp4.py
sm120_6000d/vllm/
tests/test_fable_frontend_correctness.py
tests/test_four_api_smoke.py
tests/test_four_api_correctness.py
tests/test_select_in_mega.py
tests/fe_dump_compare.py
tests/profile_four_api_h20.py
tests/fe5_campaign_streamed.sh
tests/fe5_summarize_campaign.py
scripts/capture_four_api_h20_timelines_host.sh
scripts/capture_four_api_h20_timelines.sh
scripts/summarize_four_api_h20_last3.py
```

## License

This repository is released under the [MIT License](LICENSE).

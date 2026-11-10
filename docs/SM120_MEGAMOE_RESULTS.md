# SM120 MegaMoE results (RTX 6000D, SM120)

Back to the [README](../README.md).

Decode MoE kernels for DeepSeek-V4-Flash (routed experts: E256, H4096, I2048, top-6, SwiGLU clamp 10) on 8x RTX 6000D
(SM120, 156 SMs, 112 MB L2, PCIe P2P between all 8 GPUs, GPUs 0-3 and 4-7 on two CPU sockets, no NVLink). SM clock:
locked at 2430 MHz for the TP kernel table and the manual vLLM table, default (boost up to 2430 MHz, as in CI)
everywhere else. Weights W-MXFP4
(E2M1, per-32 UE8M0), activations quantised in the kernel to MXFP8 (E4M3, per-32 UE8M0, scale 2^ceil(log2(amax/448))),
both block scales applied by the tensor core (`mma.sync.aligned.m16n8k32.kind::mxf8f6f4.block_scale`).

Source: `deep_gemm/include/deep_gemm/impls/sm120_mxfp4_mega_moe.cu` (torch extension, built for `sm_120a` on first use),
Python API in `deep_gemm/mega/sm120.py`.

| Kernel | API | Parallelism |
|---|---|---|
| EP MegaMoE | `get_symm_buffer_for_mega_moe_sm120`, `mxfp4_mega_moe_sm120` | expert parallel over PCIe P2P, one persistent kernel per rank: dispatch + FC1 + SwiGLU + FC2 + combine |
| TP MoE | `init_mxfp4_moe_tp_sm120`, `mxfp4_moe_tp_sm120` | MoE-TP (every expert local with I/TP), one kernel per call, output = this rank's partial sum (caller all-reduces) |

Both take checkpoint MXFP4 weights through `transform_mxfp4_weights_for_mega_moe_sm120`.

## Design

Same principles as the H20/B200 kernels: one persistent kernel, no cross-rank barriers, the token slab on the MMA N side
(swap-AB), counters and flags carried by the data flow.

- **Compute core.** Swap-AB `mma.sync` m16n8k32 (weights = A, 8 tokens = B). Weights go global -> registers in a
  lane-fragment layout (one `uint4` per lane per K64 step, cache-streaming `ld.global.cs`), with no shared-memory
  staging; the core reads at ~95 % of the measured 1347 GB/s DRAM peak. FC1 rows are interleaved per 16-row block
  (gate rows 8b..8b+7, then up rows 8b..8b+7), so each SwiGLU pair lands in one thread (accumulator rows g and g+8) and
  SwiGLU + MXFP8 quantisation of the FC2 input happen in registers.
- **EP dispatch.** Routing is exchanged first (each rank pushes its top-k ids to every peer), so every rank knows its
  per-expert token counts before any activation arrives and builds its work map with a warp scan. One CTA per token
  quantises the token once to MXFP8 and pushes 16 B stores over PCIe P2P into each destination expert pool. Per-source
  epoch flags are written by the producer into the consumer's memory (`st.release.sys`) and polled locally
  (`ld.acquire.sys`); there are no remote atomics.
- **Latency hiding.** At t = 0, before the routing exchange completes, every CTA prefetches into L2 the weights of the
  experts its own tokens select (64 MB budget); the first work item of each CTA is static and its weights are prefetched
  while the dispatch flags are pending.
- **Work items.** 128 weight rows x up to 8 tokens per item; FC2 output (bf16) is pushed straight back into the source
  rank's combine buffer, and the source combines its top-k locally.
- **TP kernel.** World 1: each work item quantises its token K slice into shared memory, FC1 is split-K (up to 16 ways)
  with fp32 reduction and the last split applies SwiGLU + MXFP8; FC2 folds the top-k weight into an fp32 reduction and
  the last CTA converts to bf16. Calls of more than 64 tokens are chunked by the wrapper.
- **Safety.** The last CTA resets the counters and advances the epoch (CUDA-graph replay safe). Every spin wait times
  out after 30 s and sets an error word (`Sm120MegaMoeBuffer.error()`, `mxfp4_moe_tp_sm120_error()`) instead of
  hanging the GPU. Expert id -1 (vLLM padding) is skipped.

## Kernel time per MoE layer

**EP4, DSV4-Flash** (ranks 0-3, 64 experts per rank, T tokens per rank = 4T global, random top-6 routing, identical
per-rank routing for both rows; both rows from one run). One MoE layer captured in a CUDA graph, 100 replays timed with
CUDA events, median of 5 runs, max over ranks, µs. Marlin = vLLM's `fused_marlin_moe` (MXFP4 W4A16) over the
all-gathered 4T tokens with this rank's experts, NCCL all-gather before and reduce-scatter after (vLLM's DP+EP
`allgather_reducescatter` path), captured the same way.

| Tokens per rank (global) | 1 (4) | 2 (8) | 8 (32) | 16 (64) | 20 (80) | 24 (96) | 32 (128) | 48 (192) | 64 (256) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mxfp4_mega_moe_sm120` | **120** | **166** | **362** | **531** | **577** | **587** | **627** | **633** | **641** |
| Marlin + AG/RS | 154 | 217 | 458 | 680 | 749 | 784 | 811 | 856 | 899 |
| Change | -22 % | -23 % | -21 % | -22 % | -23 % | -25 % | -23 % | -26 % | -29 % |

From 16 tokens per rank on the kernel is at the DRAM floor of the weights it reads, and from 24 on nearly all 64 local
experts are active, so its time flattens; the Marlin path keeps growing with the NCCL all-gather / reduce-scatter
volume. The CC80 CI point below runs at about 23 tokens per rank (median of the busiest DP rank).

**EP8, H20/B200 README shape** (E384 / H3072 / I1280 / top-8, 48 experts per rank, global M tokens over 8 ranks: one
row per rank up to M8, two at M16; the README's balanced routing `expert = s * 48 + (g + 7 s) % 48`, weight 1/8; GPU 0,
µs). Warm L2 = the README's streamed method (3 passes of 2 warm-up + 30 back-to-back graph replays, median of the last
3 replays, median over the passes; per-replay time from CUDA events, not nsys). Cold L2 = 256 MB written before each
replay, ranks released together by a barrier, median of 30 replays. Marlin + AG/RS as above with these routes. The
H20 / B200 rows for the same shape are in the README table.

| Global M | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| `mxfp4_mega_moe_sm120`, warm L2 | **45.1** | **53.2** | **69.6** | **106.5** |
| Marlin + AG/RS, warm L2 | 112.6 | 110.6 | 112.6 | 110.6 |
| `mxfp4_mega_moe_sm120`, cold L2 | 60.9 | 75.1 | 92.5 | 128.4 |
| new kernel, W4A8 (`fp4_mega_moe_sm120`, `'w4a8'`), cold L2 | **44.8** | **60.1** | **78.0** | **128.9** |
| new kernel, W4A4 MXFP4, cold L2 | 50.8 | 67.3 | 80.8 | 129.1 |
| new kernel, W4A4 NVFP4, cold L2 | 62.9 | 75.0 | 88.4 | 142.2 |
| Marlin + AG/RS, cold L2 | 134.3 | 141.0 | 162.8 | 203.6 |

Cold L2 is what a decode step sees (43 layers of weights per step, far beyond the L2): 2.2x / 1.9x / 1.8x / 1.6x
for `mxfp4_mega_moe_sm120`, 3.0x / 2.3x / 2.1x / 1.6x for the new kernel's W4A8 (same formats, 2026-11-10; all M,
the other flush method and what bounds it: [sm120_6000d/W4A4.md](../sm120_6000d/W4A4.md)). The README's H20 rows are
measured with the same 256 MiB write before each replay: 38.8 / 47.4 / 56.7 / 76.2 (MXFP4 fused).
Warm L2 favours Marlin at M16: each rank's 16 active experts (~94 MB) fit the 112 MB L2, so its GEMMs read from L2
(34 µs per call in the profiler, under the ~70 µs DRAM floor), while this kernel's cache-streaming weight loads do not
keep them there. At M2 the Marlin path is NCCL-bound (all-gather plus a 17-23 µs reduce-scatter across the two sockets).

**Where the time goes** (EP8, H20/B200 shape, in-kernel `globaltimer` stamps of eager calls, `STAMPS=1`; the per-GPU
timers are not synchronised, so each value is µs from that rank's own kernel start; rank 3, the rank that entered last):

| Phase ends at (µs) | M2 | M8 | M16 |
|---|---:|---:|---:|
| routing pushed to all ranks (stores + `__threadfence_system` + flags) | 4.6 | 5.4 | 5.9 |
| tokens quantised and pushed | 5.6 | 6.9 | 7.2 |
| arrival flags sent (second system fence) | 9.7 | 11.0 | 11.3 |
| per-expert work map built | 8.7 | 10.2 | 10.8 |
| every source's rows arrived (median CTA) | 10.8 | 15.6 | 17.9 |
| last FC1 item done | 26.1 | 40.4 | 80.6 |
| last FC2 item done, combine flags sent | 36.4 | 55.8 | 91.9 |
| every rank's combine flag received | 66.8 * | 62.2 | 100.6 |
| kernel end | 67.8 * | 63.7 | 102.1 |

\* The eager M2 call waits for ranks launched later; its graph replay is 45.1 µs. From arrival to the last FC2 item the
kernel takes 74 µs at M16 and 40 µs at M8 against ~72 / ~36 µs to read the active experts' weights from DRAM: the
GEMMs are at the bandwidth floor. The fixed ~30 µs is the dispatch chain before (two system fences, one PCIe round trip
for the rows) and the combine wait after; at M2 the 2 active experts give FC1 / FC2 only 40 / 48 work items, so the
GEMM phase is latency-bound (~25 µs against a ~9 µs read).

**TP** (one GPU, all 256 experts with I/TP rows, global T tokens). L2 flushed before every call (256 MB write), CUDA
graph, median of 30 calls, µs. Marlin = `fused_marlin_moe` on the same I/TP shard, same flush.

| | T = 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| TP8 `mxfp4_moe_tp_sm120` | **33.5** | **42** | **66.5** | **103** | **158** |
| TP8 Marlin | 57 | 61 | 86 | 112 | 177 |
| TP4 `mxfp4_moe_tp_sm120` | **43** | **60** | **99** | **154** | **245** |
| TP4 Marlin | 65 | 85 | 120 | 187 | 300 |

## vLLM serving (real DSV4-Flash-0731 weights)

vLLM main `0.32.1rc1.dev109+ge10ecb6bd` behind Dynamo 1.6 (image `toklin/vllm-sm120:e10ecb6bd-v41`), only the MoE
backend differs between the two sides: `kz_sm120_megamoe` / `kz_sm120_tp` (this kernel, `sm120_6000d/vllm/`) versus
`marlin`.

**Toklin CI, AgentX official** (Weka coding-agent traces replayed for 3600 s at CC concurrent sessions, seed 20260707,
10 warm-ups per lane; MMLU abstract-algebra gate; every point `valid`, gate pass rate 0.94 on both sides; harness
branch `benchmark/sm120-vllm-ci-kz` of toklin). PD = P on GPUs 0-3 (TP1 x PP4, Marlin on both sides) + D on GPUs 4-7
(DP4 attention, MoE EP4: this kernel versus Marlin), NIXL push, KV router. This kernel ran on host .20; the Marlin points
are the dashboard's (CC32 on .1, CC64 on .2, CC80 and TP8 on .20; Marlin CC32 rerun on .20, pipeline 75836358: 706.8
tok/s, the same).

| Layout | CC | Output tok/s | | Change | ITL p50 / p90, ms | | TTFT p50 / p90, s | | Pipelines (this / Marlin) |
|---|---:|---:|---:|---:|---|---|---|---|---|
| | | this | Marlin | | this | Marlin | this | Marlin | |
| PD (D: DP4/EP4) | 32 | **749.6** | 707.8 | +5.9 % | **23.9 / 26.9** | 26.3 / 30.3 | 1.61 / 4.65 | 1.57 / 4.40 | 75836354 / 75443560 |
| PD (D: DP4/EP4) | 64 | **1323.7** | 1230.4 | +7.6 % | **36.0 / 40.2** | 41.3 / 46.0 | 2.86 / 11.92 | 2.57 / 11.26 | 75836361 / 75299883 |
| PD (D: DP4/EP4) | 80 | **1594.3** | 1493.8 | +6.7 % | **43.7 / 47.9** | 49.1 / 52.9 | 3.91 / 15.60 | 3.56 / 14.42 | 75836367 / 75322268 |
| TP8 | 1 | **96.6** | 93.7 | +3.1 % | **6.82 / 8.10** | 7.16 / 8.72 | 0.96 / 2.75 | 0.95 / 2.55 | 75836275 / 75443558 |

The decode batch the D worker actually runs at CC80 (10 s samples of its four DP engines): 68 requests in total on
average (p50 74, p90 92), the busiest DP rank 23 at the median, i.e. the 16-24 tokens-per-rank columns of the EP4
table. TTFT is slightly higher with this kernel although P is identical: requests finish sooner and come back to P
sooner. The TP8 point ran a variant of `kz_sm120_tp` that hands calls above 64 tokens (prefill) to Marlin and keeps a
Marlin weight copy for it (KV pool 2.76 M tokens versus Marlin's 4.55 M); that variant is not in this repository.

**Manual `vllm bench serve`** (SM clock locked): random input 1024 / output 512 tokens, CC concurrent requests (8-request
warm-up first), every GPU idle before each point. Median inter-token latency, ms.

| CC | Layout | This kernel | Marlin | Change |
|---:|---|---:|---:|---:|
| 1 | TP8 | 6.38 | 6.69 | -4.6 % |
| 2 | TP8 | 6.67 | 6.94 | -3.9 % |
| 4 | TP8 | 7.23 | 7.63 | -5.2 % |
| 8 | TP8 | 8.77 | 8.90 | -1.5 % |
| 16 | TP8 | 11.10 | 11.30 | -1.8 % |
| 32 | DP4 / EP4 | 25.03 | 26.62 | -6.0 % |
| 64 | DP4 / EP4 | 31.06 | 34.52 | -10.0 % |
| 80 | DP4 / EP4 | 33.36 | 38.60 | -13.6 % |

Output check with the real weights: prompt NLL (teacher-forced) TP8 0.824 vs Marlin 0.822, DP4/EP4 0.892 vs 0.846
(W4A8 versus Marlin's W4A16; the two DP4 values were not taken on the same prompt set); greedy completions are coherent,
correct on arithmetic and code prompts, and identical across repeated requests.

## Correctness

`tests/test_mega_moe_sm120.py` builds random MXFP4 experts in checkpoint layout, runs them through
`transform_mxfp4_weights_for_mega_moe_sm120` and the kernel, and compares with an fp32 reference that applies the
kernel's MXFP8 rule to the activations and to the SwiGLU output (relative L2 error, pass < 2e-2; padding tokens with
expert id -1 included):

| Test | Shape | Tokens | rel L2 (worst) | Result |
|---|---|---|---:|---|
| EP, 4 ranks (`verify`) | H512, I256, 128 experts (32 per rank), top-6 | 1 / 3 / 8 / 13 / 64 per rank | 1.74e-3 | PASS |
| TP, 1 GPU (`tpverify`) | H512, I256, 32 experts, top-6 | 1 / 2 / 5 / 8 / 13 / 40 / 100 (> 64: chunked) | 1.71e-3 | PASS |

Error word 0 on every call.

## Reproduce

The SM120 kernels need only PyTorch and nvcc; `import deep_gemm` also needs the package's `_C`. In a CUDA 13 /
PyTorch >= 2.10 container without a GPU at build time (e.g. the vLLM image above):
`LIBRARY_PATH=/usr/local/cuda/lib64/stubs TORCH_CUDA_ARCH_LIST=9.0a CPATH=<site-packages>/nvidia/cu13/include ./develop.sh`.

```bash
# kernel (one GPU / 4 GPUs; the extension builds on first use into ~/.deep_gemm/sm120_mega_moe)
python tests/test_mega_moe_sm120.py tpverify
TPS=8 python tests/test_mega_moe_sm120.py tp 1 2 4 8 16
torchrun --nproc_per_node=4 tests/test_mega_moe_sm120.py verify
# EP kernel tables, straight on the kernel source (sm120_6000d/bench/; Marlin scripts need a vLLM environment)
cd sm120_6000d/bench
SAVE_ROUTING=/tmp/rt torchrun --nproc_per_node=4 ep_kernel_bench.py bench 1 2 8 16 20 24 32 48 64
ROUTING_DIR=/tmp/rt torchrun --nproc_per_node=4 marlin_ep4_bench.py 1 2 8 16 20 24 32 48 64
torchrun --nproc_per_node=8 ep_kernel_bench.py h20shape 2 4 8 16          # COLD=1 for the cold-L2 rows
torchrun --nproc_per_node=8 marlin_h20shape.py 2 4 8 16                   # COLD=1 likewise
STAMPS=1 STAMPS_DIR=/tmp torchrun --nproc_per_node=8 ep_kernel_bench.py h20shape 2 8 16 && (cd /tmp && python $OLDPWD/stamps_report.py 2 8 16)
# TP Marlin baseline
python marlin_moe_bench.py 8 1 2 4 8 16
```

vLLM: `sm120_6000d/vllm/patch_main_kz120.py` then `patch_main_kztp.py` patch a copy of the image's `vllm` package
(backends `kz_sm120_megamoe` and `kz_sm120_tp`); `kz_sm120_megamoe.py` goes to `vllm/model_executor/layers/fused_moe/`,
`kz_sm120_tp.py` to `.../fused_moe/experts/`, and `KZ_SM120_MM_SO` points at the built extension. `ci_serve.sh` starts
the server with the patched files bind-mounted (`PAR=tp8` for TP8, default DP4/EP4).

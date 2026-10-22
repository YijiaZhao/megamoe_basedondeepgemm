# SM120 MegaMoE results (RTX 6000D, SM120)

Back to the [README](../README.md).

Decode MoE kernels for DeepSeek-V4-Flash (routed experts: E256, H4096, I2048, top-6, SwiGLU clamp 10) on 8x RTX 6000D
(SM120, 156 SMs, 112 MB L2, PCIe P2P between all 8 GPUs, no NVLink), SM clock locked at 2430 MHz. Weights W-MXFP4
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
  lane-fragment layout (one `uint4` per lane per K64 step, `ld.global.nc.L1::no_allocate`), with no shared-memory
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

**EP4** (ranks 0-3, 64 experts per rank, T tokens per rank, identical per-rank routing for both columns). One MoE layer
captured in a CUDA graph, 100 replays timed with CUDA events, median of 5 runs, max over ranks, µs. Marlin = vLLM's
`fused_marlin_moe` (MXFP4 W4A16) over the all-gathered 4T tokens with this rank's experts, NCCL all-gather before and
reduce-scatter after (vLLM's DP+EP `allgather_reducescatter` path), captured the same way.

| Tokens per rank | 1 | 2 | 8 | 16 | 20 |
|---|---:|---:|---:|---:|---:|
| `mxfp4_mega_moe_sm120` | **123** | **165** | **363** | **536** | **581** |
| Marlin + AG/RS | 151 | 217 | 459 | 679 | 749 |

At 16 and 20 tokens per rank the kernel is at the DRAM floor of the weights it has to read; the remaining ~10-15 µs is
PCIe flag and routing latency.

**TP** (one GPU, all 256 experts with I/TP rows, global T tokens). L2 flushed before every call (256 MB write), CUDA
graph, median of 30 calls, µs. Marlin = `fused_marlin_moe` on the same I/TP shard, same flush.

| | T = 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| TP8 `mxfp4_moe_tp_sm120` | **33.5** | **42** | **66.5** | **103** | **158** |
| TP8 Marlin | 57 | 61 | 86 | 112 | 177 |
| TP4 `mxfp4_moe_tp_sm120` | **43** | **60** | **99** | **154** | **245** |
| TP4 Marlin | 65 | 85 | 120 | 187 | 300 |

## vLLM decode (real DSV4-Flash-0731 weights)

vLLM main `0.32.1rc1.dev109+ge10ecb6bd` (image `toklin/vllm-sm120:e10ecb6bd-v41`), only `--moe-backend` differs between
columns: `kz_sm120_megamoe` / `kz_sm120_tp` (this kernel, `sm120_6000d/vllm/`) versus `marlin`. `vllm bench serve`,
random input 1024 / output 512 tokens, CC concurrent requests (8-request warm-up first), every GPU idle before each
point. Median inter-token latency, ms.

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
(W4A8 versus Marlin's W4A16); greedy completions are coherent, correct on arithmetic and code prompts, and identical
across repeated requests.

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
SAVE_ROUTING=/tmp/rt torchrun --nproc_per_node=4 tests/test_mega_moe_sm120.py bench 1 2 8 16 20
# Marlin baselines (in a vLLM environment)
python sm120_6000d/bench/marlin_moe_bench.py 8 1 2 4 8 16
ROUTING_DIR=/tmp/rt torchrun --nproc_per_node=4 sm120_6000d/bench/marlin_ep4_bench.py 1 2 8 16 20
```

vLLM: `sm120_6000d/vllm/patch_main_kz120.py` then `patch_main_kztp.py` patch a copy of the image's `vllm` package
(backends `kz_sm120_megamoe` and `kz_sm120_tp`); `kz_sm120_megamoe.py` goes to `vllm/model_executor/layers/fused_moe/`,
`kz_sm120_tp.py` to `.../fused_moe/experts/`, and `KZ_SM120_MM_SO` points at the built extension. `ci_serve.sh` starts
the server with the patched files bind-mounted (`PAR=tp8` for TP8, default DP4/EP4).

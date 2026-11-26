# SM120 MegaMoE (RTX 6000D): W4A8 and W4A4 (MXFP4 / NVFP4)

W4A4 (MXFP4, NVFP4) and W4A8 (MXFP4 weights x MXFP8 activations) on one kernel:

| File | What |
|---|---|
| `deep_gemm/include/deep_gemm/impls/sm120_fp4_mega_moe.cu` | kernel + torch extension, one build per format (`SM120_FP4_NV`, `SM120_FP4_A8`) and transport (`SM120_FP4_GIN`) |
| `deep_gemm/include/deep_gemm/impls/sm120_fp4_mega_moe_gin.cu` | NIC transport session (host only): NCCL window, peer pointers, GIN device communicator |
| `deep_gemm/mega/sm120_fp4.py` | `Sm120Fp4MegaMoeBuffer`, `fp4_mega_moe_sm120`, `transform_fp4_weights_for_mega_moe_sm120`, `nvfp4_global_scales` |
| `tests/test_mega_moe_sm120_fp4.py` | correctness (`verify`), timing (`bench`) |
| [`ENV.md`](../sm120_6000d/ENV.md) | the environment variables it reads (none is needed; none changes the kernel) |

EP over PCIe P2P (or the NIC, see Transport), one persistent kernel per rank, no cross-rank barrier. The dispatch and the FC1 epilogue quantize
the activations; FC1 / FC2 run swap-AB `mma.sync`:

| Format | Weights and activations | MMA |
|---|---|---|
| `mxfp4` | E2M1, UE8M0 scale per 32 | `kind::mxf4nvf4` m16n8k64 `scale_vec::2X` |
| `w4a8` | weights E2M1, activations E4M3, UE8M0 scale per 32 | `kind::mxf8f6f4` m16n8k32 `scale_vec::1X`, two per K64 |
| `nvfp4` | E2M1, UE4M3 scale per 16, fp32 global scales: per expert for W13 / W2 (`weight_scale_2`), per tensor for the FC1 / FC2 inputs (`input_scale`, the same on every rank) | `kind::mxf4nvf4` m16n8k64 `scale_vec::4X` |

```python
from deep_gemm.mega import sm120_fp4
l1, l2 = sm120_fp4.transform_fp4_weights_for_mega_moe_sm120((w13, w13_sf), (w2, w2_sf), 'nvfp4')
gs = sm120_fp4.nvfp4_global_scales(fc1_input_scale, fc2_input_scale, w13_scale_2, w2_scale_2)  # nvfp4 only
buf = sm120_fp4.Sm120Fp4MegaMoeBuffer(group, 'nvfp4', num_experts, max_tokens_per_rank, top_k, hidden, inter)
sm120_fp4.fp4_mega_moe_sm120(y, x, topk_ids, topk_weights, l1, l2, buf, gs, activation_clamp=limit)
```

```bash
torchrun --nproc-per-node 8 tests/test_mega_moe_sm120_fp4.py verify mxfp4    # or nvfp4 / w4a8
torchrun --nproc-per-node 8 tests/test_mega_moe_sm120_fp4.py bench nvfp4 2 4 8 16 64 256 512 1024   # or mxfp4 / w4a8
```

## Where to use it

**This kernel is for decode-size batches: fewer than 256 tokens per rank and at most 128 tokens per local expert**
(tokens per expert = tokens per rank × top-k × ranks / experts). It is one persistent kernel per rank that streams every
active expert's weights once per call and moves tokens and partials itself; its tile shapes and work order are chosen
for that range. Above it, use a grouped-GEMM MoE: this kernel is not tuned there, and this document gives no numbers
there.

Against gc-moe on the same GPUs, with the same weights, routing and timing (see [vs gc-moe](#vs-gc-moe)), at up to 128
tokens per rank:

* MXFP4: faster at every point, both models.
* W4A8: faster at every point, except GLM-5.3-Flash at 128 tokens per rank (610.3 vs 535.8 µs).
* NVFP4: faster at every point on GLM-5.3-Flash; slower on DeepSeek-V4.1-Flash at 6-64 tokens per rank (by 0.7-5.2 %).

## Test hosts

Two kinds of 8x RTX 6000D host, the same GPUs, CPUs (2x AMD EPYC 9575F), driver (580.95.05) and NICs (4x ConnectX-8),
but a different PCIe fabric. The kernel moves tokens and partials over PCIe P2P stores (no NVLink), so the fabric
decides which pairs can use P2P and the times on one host do not carry over to the other.

| | R6KD-BRCM (10.6.142.20) | R6KD-CX8aaS (10.6.142.1, .3, .6) |
|---|---|---|
| GPU pairs 0-1, 2-3, 4-5, 6-7 behind | one Broadcom PEX890xx Gen5 switch each, with one ConnectX-8 | the PCIe switch inside one ConnectX-8 each |
| `nvidia-smi topo -m`: pair / same socket / other socket | PIX / NODE / SYS | PXB / NODE / SYS |
| NIC ports up | 4 x 400 Gb/s (one port per ConnectX-8) | 8 x 400 Gb/s (both ports) |
| Transport this kernel is measured with | `'p2p'`: every pair over P2P | `'lsa_where_reachable'`: the 3 same-socket peers over P2P, the 4 others over the NIC |
| Tables measured on it | [vs gc-moe](#vs-gc-moe) (first part); README's RTX 6000D rows | every other table here (EP8 on .1 / .6, EP16 on .1 + .6) |

P2P write bandwidth with SM stores (16 B per thread, the way the kernel pushes rows), 64 MB per destination, GB/s (per
source GPU), [`bench/p2pbw.cu`](../sm120_6000d/bench/p2pbw.cu) (`nvcc -O3 -gencode=arch=compute_120a,code=sm_120a
p2pbw.cu -o p2pbw && ./p2pbw`):

| Writes | R6KD-BRCM (.20) | R6KD-CX8aaS (.6) |
|---|---:|---:|
| GPU 0 -> 1 (same switch) | 51.1 | 51.0 |
| GPU 0 -> 2 (same socket, other switch) | 51.1 | 51.1 |
| GPU 0 -> 4 (other socket) | 47.1 | 46.5 |
| GPU 0 -> 1, 2, 3 at once (same socket) | 51.2 | 51.1 |
| GPU 0 -> 2, 3 at once (one other switch) | 51.1 | 51.1 |
| GPU 0 -> 2, 4 at once (same socket and other socket) | 51.1 | **2.0** |
| GPU 0 -> all 7 at once | 51.2 | **1.5** |
| every GPU -> its 7 peers at once | 34.1 | **0.8** |
| every GPU -> its 3 same-socket peers at once | 50.2 | 37.9 |

On R6KD-CX8aaS any single path runs at full speed, but a GPU that writes to a GPU on the other socket and to one behind
another root port at the same time drops to 1.5-2 GB/s; an all-to-all over every pair, which is what an EP layer
does, gets 0.8 GB/s per GPU. Hence the transport table there: P2P only within a socket, the other socket over the NIC.
On R6KD-BRCM every pair can go over P2P at once.

## Results (8x RTX 6000D, R6KD-CX8aaS hosts 10.6.142.1 / .6)

### Correctness

Against a reference that applies the kernel's quantization (activations and SwiGLU output), max
relative L2 over ranks, every tokens-per-item width (8 / 16 / 32 / 64 and the automatic choice), 1-128 tokens per rank
with padding tokens:

| Shape | `w4a8` | `mxfp4` | `nvfp4` |
|---|---:|---:|---:|
| small (H 512 / I 256, 128 experts, top-6) | 1.74e-3 | 1.73e-3 | 1.79e-3 |
| readme (H 3072 / I 1280, 384 experts, top-8) | 1.71e-3 | 1.69e-3 | 6.57e-3 |
| wide (H 5120 / I 256, 128 experts, top-6: the dispatch sends each token in several shared-memory chunks) | 1.72e-3 | 1.69e-3 | 1.72e-3 |

The same values at EP8 over `'p2p'` (10.6.142.1) and over `'lsa_where_reachable'` (10.6.142.6, 4 of the 7 peers over
the NIC). The NVFP4 readme value comes from the single-token case; likely the coarser UE4M3 scales of the FC2 input, not
checked further. EP16 (10.6.142.1 + .6, `'lsa_where_reachable'`, `NCCL_IB_TC=96`, small / readme): W4A8 1.78e-3 /
1.71e-3, MXFP4 1.77e-3 / 1.69e-3, NVFP4 1.79e-3 / 6.57e-3.
`stress` (`dsv41_eplb16` shape, 300 calls with random per-rank delays up to 2 ms; plain, `sparse` routing, and `hog`: a
GEMM competing on a side stream), every format, `'p2p'` (10.6.142.1): 0 mismatching outputs, 0 kernel error words.

### Timing

Every number below is from `tests/test_mega_moe_sm120_fp4.py bench <fmt> <M...> lsa_where_reachable` with
`BENCH_SHAPE=<shape>`:

* One call per point, captured in a CUDA graph. Before each replay 256 MB are written (cold L2: in serving other layers
  run between two calls of one layer), `torch.cuda.synchronize()`, then `dist.barrier()`; CUDA events around the replay
  on each rank. 32 replays, the median of the last 30; three such passes, the median of the three. The tables give GPU
  0's value, µs per MoE layer (dispatch, both GEMMs, combine; the routing top-k is the caller's and not included).
* Balanced routing, weight 1/top-k: copy s of global token g goes to rank s * world / top-k + g % (world / top-k)
  ((g * top-k + s) % world when top-k does not divide the world), each rank hands the copies it receives, in (g, s) order, to its local experts in turn. Random
  E2M1 weight bytes with fixed scales (UE8M0 124 / E4M3 0x38). Global M = tokens over all ranks; tokens / rank = M / ranks; tokens / expert = tokens / rank * top-k * ranks / experts:
  every expert gets exactly that many (± 1 where it does not divide).
* Hosts: GLM-5.3-Flash and DeepSeek-V4.1-Flash on 10.6.142.6, DeepSeek-V4-Flash and the README shape on 10.6.142.1;
  the same model (8x RTX 6000D, GPUs in pairs behind a ConnectX-8's PCIe switch, 4 ConnectX-8 / 8 ports), driver
  580, clocks not locked, NCCL 2.30.7, containers `toklin/vllm-sm120:e10ecb6bd-v41` (.6) / `-p3` (.1).
  `NCCL_IB_HCA=mlx5_0,...,mlx5_7`: NCCL maps the 3 peers of a rank's own CPU socket, so `'lsa_where_reachable'` sends
  to those over P2P and to the other 4 over the NIC (GIN). `NCCL_IB_TC=96` (the ports' lossless class, [`ENV.md`](../sm120_6000d/ENV.md)).

GLM-5.3-Flash (H 4096, I 2048, 256 experts, top-8), EP8:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.2 | 162.6 | 168.1 | 162.3 |
| 2 | 16 | 0.5 | 231.3 | 244.0 | 233.1 |
| 4 | 32 | 1 | 388.7 | 409.9 | 388.9 |
| 8 | 64 | 2 | 390.9 | 412.0 | 392.2 |
| 16 | 128 | 4 | 395.8 | 413.9 | 398.6 |
| 32 | 256 | 8 | 429.8 | 461.3 | 436.0 |
| 64 | 512 | 16 | 441.8 | 473.9 | 469.5 |
| 128 | 1024 | 32 | 499.9 | 536.6 | 606.7 |
| 192 | 1536 | 48 | 581.9 | 618.7 | 712.7 |

DeepSeek-V4.1-Flash (H 5120, I 2304, 384 experts, top-6), EP8:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.1 | 175.6 | 181.0 | 176.0 |
| 2 | 16 | 0.2 | 241.5 | 250.3 | 240.5 |
| 4 | 32 | 0.5 | 407.4 | 428.3 | 406.7 |
| 8 | 64 | 1 | 750.4 | 834.0 | 751.3 |
| 16 | 128 | 2 | 762.4 | 846.0 | 762.3 |
| 32 | 256 | 4 | 762.7 | 846.1 | 767.3 |
| 64 | 512 | 8 | 797.7 | 872.7 | 814.7 |
| 128 | 1024 | 16 | 827.1 | 898.2 | 893.7 |
| 192 | 1536 | 24 | 878.5 | 944.7 | 1004.3 |

DeepSeek-V4-Flash (H 4096, I 2048, 256 experts, top-6), EP8:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.2 | 152.0 | 155.6 | 151.8 |
| 2 | 16 | 0.4 | 193.8 | 201.1 | 193.8 |
| 4 | 32 | 0.8 | 310.7 | 326.0 | 311.0 |
| 8 | 64 | 1.5 | 390.7 | 412.1 | 390.9 |
| 16 | 128 | 3 | 392.7 | 417.4 | 393.1 |
| 32 | 256 | 6 | 404.7 | 425.3 | 406.1 |
| 64 | 512 | 12 | 440.8 | 463.7 | 456.0 |
| 128 | 1024 | 24 | 475.5 | 492.9 | 527.2 |
| 192 | 1536 | 36 | 537.9 | 569.9 | 647.8 |

README shape (H 3072, I 1280, 384 experts, top-8), EP8:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.2 | 122.8 | 127.3 | 123.5 |
| 2 | 16 | 0.3 | 161.2 | 167.1 | 163.1 |
| 4 | 32 | 0.7 | 221.2 | 232.4 | 222.8 |
| 8 | 64 | 1.3 | 299.7 | 312.9 | 300.4 |
| 16 | 128 | 2.7 | 303.4 | 316.9 | 307.0 |
| 32 | 256 | 5.3 | 312.0 | 323.4 | 316.9 |
| 64 | 512 | 10.7 | 348.6 | 375.2 | 366.7 |
| 128 | 1024 | 21.3 | 381.8 | 403.1 | 423.7 |
| 192 | 1536 | 32 | 420.3 | 447.0 | 492.2 |

At 2-64 tokens per rank the call is the read of the active experts' weights from DRAM plus the routing / row / combine
round trips between the ranks.

EP16, two hosts (10.6.142.1 node rank 0 + 10.6.142.6, 16 ranks): `tests/test_mega_moe_sm120_fp4.py bench` as above,
launched like `bench/run_ep16.sh` (same docker flags, `NCCL_IB_TC=96`, `NCCL_IB_HCA` unset: each rank reaches the 3 GPUs of its own socket
over P2P, the other 12 over the NIC; containers as above per host), every one of the 16 GPUs idle before each run. The
values are the slowest rank's (every rank's median as above, then the maximum over the 16 ranks); the layer waits for it.

GLM-5.3-Flash (H 4096, I 2048, 256 experts, top-8), EP16:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.5 | 177.5 | 182.7 | 177.4 |
| 2 | 32 | 1 | 256.9 | 267.2 | 257.4 |
| 4 | 64 | 2 | 258.5 | 269.6 | 260.8 |
| 8 | 128 | 4 | 274.1 | 284.3 | 274.3 |
| 16 | 256 | 8 | 291.7 | 303.6 | 292.5 |
| 32 | 512 | 16 | 302.5 | 318.5 | 316.6 |
| 64 | 1024 | 32 | 329.1 | 344.5 | 361.7 |
| 128 | 2048 | 64 | 405.0 | 419.2 | 480.0 |
| 192 | 3072 | 96 | 543.6 | 573.5 | 695.6 |

DeepSeek-V4.1-Flash (H 5120, I 2304, 384 experts, top-6), EP16:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.2 | 188.6 | 196.8 | 189.6 |
| 2 | 32 | 0.5 | 270.7 | 280.6 | 271.2 |
| 4 | 64 | 1 | 435.4 | 459.4 | 437.0 |
| 8 | 128 | 2 | 443.0 | 465.0 | 443.6 |
| 16 | 256 | 4 | 461.8 | 485.7 | 459.9 |
| 32 | 512 | 8 | 475.4 | 497.7 | 480.3 |
| 64 | 1024 | 16 | 494.4 | 529.0 | 513.1 |
| 128 | 2048 | 32 | 544.4 | 576.8 | 653.2 |
| 192 | 3072 | 48 | 609.2 | 644.0 | 732.2 |

DeepSeek-V4-Flash (H 4096, I 2048, 256 experts, top-6), EP16:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.4 | 163.1 | 167.8 | 163.2 |
| 2 | 32 | 0.8 | 222.0 | 228.6 | 220.5 |
| 4 | 64 | 1.5 | 260.0 | 270.0 | 260.3 |
| 8 | 128 | 3 | 267.2 | 274.9 | 264.8 |
| 16 | 256 | 6 | 280.3 | 289.9 | 282.1 |
| 32 | 512 | 12 | 296.7 | 311.3 | 299.4 |
| 64 | 1024 | 24 | 317.7 | 329.4 | 333.3 |
| 128 | 2048 | 48 | 368.4 | 379.9 | 435.0 |
| 192 | 3072 | 72 | 494.9 | 521.1 | 615.4 |

README shape (H 3072, I 1280, 384 experts, top-8), EP16:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.3 | 135.8 | 139.0 | 135.3 |
| 2 | 32 | 0.7 | 172.8 | 177.2 | 172.9 |
| 4 | 64 | 1.3 | 212.8 | 220.4 | 212.7 |
| 8 | 128 | 2.7 | 223.4 | 229.8 | 223.1 |
| 16 | 256 | 5.3 | 248.9 | 255.5 | 248.1 |
| 32 | 512 | 10.7 | 255.9 | 265.8 | 261.0 |
| 64 | 1024 | 21.3 | 275.4 | 280.7 | 291.4 |
| 128 | 2048 | 42.7 | 316.6 | 326.1 | 361.8 |
| 192 | 3072 | 64 | 376.7 | 384.1 | 456.6 |

At 4-32 tokens per rank the slowest rank is up to 12 % behind GPU 0.

`NCCL_IB_TC=96` matters here: unset, the NIC transport's RDMA leaves on priority 0, which is lossy on these ports
([`ENV.md`](../sm120_6000d/ENV.md)); the 32 / 64-token items' FC2 partials (up to 64 rows per put) overflow its receive buffer at
EP16, and the ranks waiting on the dropped data fall behind by milliseconds.

### vs gc-moe

[gc-moe](https://github.com/Bruce-Lee-LY/gc-moe) at
[b731747](https://github.com/Bruce-Lee-LY/gc-moe/commit/b7317477ace99fbe399e0b73eea9329fa6595261) (2026-11-24, GC MoE
0.7.0). Its SM120 operator `sm120_fp8_fp4_gc_moe` supports only W4A8 (MXFP4 weights × FP8 activations), and only
P2P through symmetric memory.

Setup: EP8 on 8x RTX 6000D at 10.6.142.20 (R6KD-BRCM, all 8 GPUs P2P), every GPU idle before each run. This kernel
runs with `transport='p2p'`. Latencies are µs per call, the slowest rank's median.

The columns:

* **gc README**: the RTX PRO 5000 numbers in gc-moe's
  [`docs/pro5000_gc_moe.md`](https://github.com/Bruce-Lee-LY/gc-moe/blob/b7317477ace99fbe399e0b73eea9329fa6595261/docs/pro5000_gc_moe.md)
  (GC MoE 0.6.0, FlashInfer 0.7.3, 8x RTX PRO 5000). Its models are GLM5.2 and DeepSeek-V4.1-Flash, so only the
  DeepSeek-V4.1-Flash table has this column, at its 9 points.
* **gc inner=100** and **gc inner=1**: gc-moe's own benchmark at b731747, run on these GPUs with flashinfer 0.7.3:

  ```
  scripts/run_sm120.sh 8 bench --arms gc --inner-iters N \
      --num-tokens-per-rank-list 1,2,3,4,6,8,12,16,24,32,48,64,96,128,160,192,256,320,384,448,512 \
      --num-max-tokens-per-rank 512 --cuda-graph-max-tokens-per-rank 512
  ```

  GLM-5.3-Flash adds `--models glm5.2 --hidden 4096 --intermediate-hidden 2048` (E256, top-8).
  * inner=100 is its default. L2 is flushed once per window, then 100 graph replays are averaged, so 99 of the 100
    calls find the previous call's weights in L2.
  * inner=1 flushes L2 before every replay ([gc-moe#8](https://github.com/Bruce-Lee-LY/gc-moe/issues/8)).
  * Its routing is its own, not the balanced routing below.
* **gc** and **W4A8 / MXFP4 / NVFP4**: both implementations with [`bench/gc_moe_compare.py`](../sm120_6000d/bench/gc_moe_compare.py),
  using the same weights and the same routing:
  * Weights: bf16 `randn * 0.05` (seed 7 + rank), quantized to MXFP4 by gc-moe's own helper. NVFP4 uses the same
    E2M1 bytes, with each UE8M0 scale as an E4M3 scale (weight global scale 1). The activation global scales are
    amax / 2688 over all token counts.
  * Routing: balanced, every expert gets the same number of tokens.
  * Timing: one call per point in a CUDA graph. Before each replay, 256 MiB are zeroed (cold L2) and one int is
    all-reduced on the stream. CUDA events are recorded around the replay. 10 warm-up and 30 timed replays per point.
  * This is the like-for-like comparison.

GLM-5.3-Flash, EP8:

| tokens / rank | global M | gc inner=100 | gc inner=1 | gc | W4A8 | MXFP4 | NVFP4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 138.6 | 175.6 | 164.9 | 120.8 | 118.8 | 124.9 |
| 2 | 16 | 213.7 | 250.4 | 243.3 | 204.8 | 202.8 | 213.0 |
| 3 | 24 | 260.7 | 305.5 | 322.1 | 280.6 | 280.6 | 295.9 |
| 4 | 32 | 270.0 | 315.6 | 401.8 | 360.5 | 360.4 | 385.0 |
| 6 | 48 | 309.2 | 358.7 | 402.1 | 362.5 | 362.5 | 385.0 |
| 8 | 64 | 330.5 | 380.7 | 406.1 | 364.6 | 364.5 | 387.1 |
| 12 | 96 | 352.3 | 403.1 | 410.3 | 366.6 | 364.5 | 387.1 |
| 16 | 128 | 367.9 | 414.0 | 414.3 | 364.5 | 366.6 | 387.1 |
| 24 | 192 | 398.6 | 446.6 | 435.8 | 367.6 | 366.6 | 387.0 |
| 32 | 256 | 370.4 | 422.6 | 418.4 | 372.7 | 366.6 | 389.1 |
| 48 | 384 | 398.6 | 462.7 | 435.8 | 385.0 | 372.7 | 393.2 |
| 64 | 512 | 405.8 | 458.7 | 441.5 | 415.7 | 378.9 | 401.4 |
| 96 | 768 | 475.7 | 534.8 | 498.8 | 487.4 | 403.4 | 421.9 |
| 128 | 1024 | 484.4 | 535.4 | 535.8 | 610.3 | 452.6 | 485.3 |
| 160 | 1280 | 545.4 | 597.9 | 626.9 | 720.9 | 534.5 | 571.4 |
| 192 | 1536 | 841.9 | 832.7 | 766.1 | 761.8 | 561.1 | 595.9 |

DeepSeek-V4.1-Flash, EP8:

| tokens / rank | global M | gc README | gc inner=100 | gc inner=1 | gc | W4A8 | MXFP4 | NVFP4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 190.9 | 183.0 | 220.6 | 169.5 | 127.0 | 127.0 | 133.2 |
| 2 | 16 | 281.1 | 270.6 | 314.1 | 259.5 | 208.9 | 208.9 | 219.1 |
| 3 | 24 | – | 329.7 | 379.2 | 340.6 | 292.9 | 292.9 | 309.2 |
| 4 | 32 | 401.8 | 369.0 | 421.8 | 428.7 | 376.8 | 376.8 | 403.4 |
| 6 | 48 | – | 447.2 | 507.6 | 597.6 | 550.9 | 548.9 | 614.4 |
| 8 | 64 | 589.1 | 536.4 | 605.9 | 776.8 | 735.2 | 731.1 | 817.1 |
| 12 | 96 | – | 614.9 | 693.5 | 778.8 | 729.1 | 727.0 | 813.1 |
| 16 | 128 | 706.8 | 642.2 | 724.3 | 780.8 | 729.1 | 727.0 | 817.1 |
| 24 | 192 | – | 679.5 | 761.8 | 784.9 | 729.1 | 725.0 | 817.1 |
| 32 | 256 | 746.2 | 693.4 | 784.1 | 789.9 | 728.0 | 725.0 | 815.1 |
| 48 | 384 | – | 700.9 | 792.1 | 800.3 | 735.2 | 729.1 | 806.9 |
| 64 | 512 | 773.8 | 719.9 | 817.6 | 805.5 | 741.3 | 733.2 | 811.0 |
| 96 | 768 | – | 784.3 | 912.6 | 826.8 | 771.0 | 743.4 | 821.2 |
| 128 | 1024 | 852.9 | 752.2 | 845.2 | 838.2 | 825.3 | 753.6 | 831.5 |
| 160 | 1280 | – | 783.9 | 870.9 | 872.1 | 886.8 | 772.1 | 849.9 |
| 192 | 1536 | – | 838.9 | 919.3 | 902.3 | 944.1 | 794.6 | 866.3 |

Against the "gc" column (same script, weights and routing), this kernel is faster at (points of the tables above):

| Shape | tokens / rank | W4A8 | MXFP4 | NVFP4 |
|---|---|---:|---:|---:|
| GLM-5.3-Flash | 1-128 | 13 / 14 (not 128) | 14 / 14 | 14 / 14 |
| GLM-5.3-Flash | 1-192 | 14 / 16 (not 128, 160) | 16 / 16 | 16 / 16 |
| DeepSeek-V4.1-Flash | 1-128 | 14 / 14 | 14 / 14 | 6 / 14 (not 6-64) |
| DeepSeek-V4.1-Flash | 1-192 | 14 / 16 (not 160, 192) | 16 / 16 | 8 / 16 (not 6-64) |

Accuracy, same script, 8 tokens per rank: cosine similarity of the output to an fp32 reference that uses bf16
activations and the dequantized weights (minimum over ranks). The activation quantization error is included. MXFP4 is
lower because its activations are E2M1 too.

| Shape | gc | W4A8 | MXFP4 | NVFP4 |
|---|---:|---:|---:|---:|
| GLM-5.3-Flash | 0.9989 | 0.9989 | 0.9741 | 0.9864 |
| DeepSeek-V4.1-Flash | 0.9989 | 0.9989 | 0.9749 | 0.9863 |

EP8 on 10.6.142.6 (R6KD-CX8aaS). On this host, P2P between GPUs behind different root ports collapses (see
[Transport](#transport)). gc-moe supports only P2P, so gc runs with full P2P. This kernel runs with
`--transport lsa_where_reachable`. The gc columns come from gc-moe's own benchmark (same command as above). W4A8 / MXFP4 / NVFP4
come from `bench/gc_moe_compare.py` (cold L2). All values are the slowest rank's median, µs.

GLM-5.3-Flash, EP8 on 10.6.142.6:

| tokens / rank | global M | gc inner=100 | gc inner=1 | W4A8 | MXFP4 | NVFP4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 230.0 | 262.8 | 161.8 | 161.7 | 165.9 |
| 2 | 16 | 482.6 | 484.1 | 243.7 | 241.7 | 253.9 |
| 3 | 24 | 678.7 | 637.7 | 325.6 | 325.6 | 342.0 |
| 4 | 32 | 1009.7 | 984.4 | 405.5 | 409.6 | 432.1 |
| 6 | 48 | 1659.1 | 1620.0 | 413.7 | 413.7 | 434.2 |
| 8 | 64 | 2322.2 | 2298.3 | 419.8 | 421.9 | 443.4 |
| 12 | 96 | 3554.9 | 3454.8 | 428.0 | 426.0 | 445.4 |
| 16 | 128 | 4691.4 | 4536.3 | 435.2 | 430.1 | 451.6 |
| 24 | 192 | 6828.6 | 6844.9 | 437.2 | 435.2 | 446.5 |
| 32 | 256 | 2585.6 | 2622.0 | 448.5 | 444.4 | 453.6 |
| 48 | 384 | 4105.5 | 3793.6 | 457.7 | 445.4 | 462.8 |
| 64 | 512 | 5552.8 | 4937.4 | 484.4 | 461.8 | 471.0 |
| 96 | 768 | 8541.6 | 8090.0 | 528.4 | 481.3 | 495.6 |
| 128 | 1024 | 11115.2 | 10863.3 | 611.3 | 514.0 | 543.7 |
| 160 | 1280 | 14692.6 | 14112.5 | 673.8 | 559.1 | 587.8 |
| 192 | 1536 | 18971.0 | 18934.5 | 720.9 | 583.7 | 620.5 |

DeepSeek-V4.1-Flash, EP8 on 10.6.142.6:

| tokens / rank | global M | gc inner=100 | gc inner=1 | W4A8 | MXFP4 | NVFP4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 229.3 | 230.5 | 174.1 | 174.1 | 180.2 |
| 2 | 16 | 419.6 | 408.8 | 255.0 | 252.9 | 264.2 |
| 3 | 24 | 546.9 | 517.0 | 337.9 | 335.8 | 352.2 |
| 4 | 32 | 777.1 | 769.1 | 423.9 | 421.9 | 446.4 |
| 6 | 48 | 1399.6 | 1311.5 | 596.0 | 593.9 | 647.2 |
| 8 | 64 | 2157.4 | 1964.2 | 771.1 | 771.1 | 851.9 |
| 12 | 96 | 3523.4 | 3364.0 | 771.8 | 770.0 | 853.0 |
| 16 | 128 | 4557.2 | 4420.0 | 785.4 | 783.4 | 866.3 |
| 24 | 192 | 6906.7 | 6755.8 | 788.5 | 782.3 | 867.3 |
| 32 | 256 | 2385.6 | 2406.9 | 796.6 | 792.6 | 868.3 |
| 48 | 384 | 3843.1 | 3428.0 | 798.7 | 796.7 | 866.3 |
| 64 | 512 | 5074.2 | 4917.7 | 813.1 | 804.9 | 868.4 |
| 96 | 768 | 7682.6 | 7526.3 | 838.7 | 813.0 | 881.6 |
| 128 | 1024 | 10063.6 | 9934.3 | 890.9 | 827.4 | 898.0 |
| 160 | 1280 | 12915.0 | 12777.5 | 945.1 | 853.0 | 919.6 |
| 192 | 1536 | 15457.4 | 15385.2 | 1004.5 | 875.5 | 942.1 |

### How the kernel spends the time

* Work order: each tile's FC2 items follow its FC1 items after `lag` tiles (FC1 of tile t + lag, then FC2 of tile t),
  so the partials go back over PCIe during the weight stream, not after it.
* Routing: each rank sends every peer its per-expert token counts first; the owners build the expert map from them
  before the token rows land.
* Cross-rank flags: one CTA polls the PCIe-written flags and republishes them on a local word, so the other CTAs do
  not read over PCIe. Flags are plain stores after `__threadfence_system`.
* Weight loads: streaming loads (`__ldcs`, evict-first): every call reads the weights from DRAM, and they need not
  stay in L2. With 8 tokens per item (one MMA per weight fragment) the GEMM loop is double-buffered: the next step's
  weights and scales are loaded while this step's MMAs run; with 16 / 32 / 64 the loop is unrolled twice instead.
* L2 prefetch: at t=0 the CTAs that do not dispatch pull the locally known experts' FC1 weights (up to 64 MB); after
  the expert map each CTA pulls its static first item's weights and scales, at any tile count (at 1 token per rank
  the weights then arrive between the expert map and the rows).
* Activation layout: the staged token rows and the FC1 output store each K256 block in the order the GEMM lane reads
  it. W4A4 (32 words): word kc * 8 + h * 4 + c of the MMA order at c * 8 + kc * 2 + h, so one K256 step of a lane's
  sub-tile is two 16 B loads plus one scale load. W4A8 (64 words, E4M3): word t * 8 + h * 4 + c (K32 step t) at
  c * 16 + t * 2 + h, one 16 B load per K64 chunk. The quantization writes the rows in that order; nothing is moved
  twice.
* Tokens per work item: 8 / 16 / 32 / 64 (each weight fragment feeds 1 / 2 / 4 / 8 sub-tiles of 8 tokens, so an
  expert's weights pass through L2 fewer times). The 64-token build runs one CTA per SM (255 registers: at 128 the
  accumulators and the step's activations spill); its `s_tile` (32 KB) is dynamic shared memory.
* Limits: up to 1024 tokens per rank at 16 ranks, top-8, with a NIC peer (`kMaxTiles`); H and I multiples of 256 (the
  GEMM loop's K256 steps).

NVFP4 is slower than MXFP4 because its weights carry twice the scale bytes (one E4M3 per 16 values).

#### Tokens per item (`tile_tokens=0`)

> **The automatic choice is an empirical default, not the optimum. Re-sweep it for the model, EP size, network and
> routing you deploy** (force each width: `tile_tokens=8 / 16 / 32 / 64`, or `TILE=8|16|32|64` for
> `tests/test_mega_moe_sm120_fp4.py bench` (with `bench/run_ep16.sh` at EP16); take the fastest per tokens per rank
> and change the rule in `sm120_fp4_mega_moe.cu`).

From e = T * top-k / local experts, the expert's I and the group size W:

| | I >= 2048 | I < 2048 |
|---|---|---|
| W <= 8 | 8 up to e 8, 16 below e 16, 32 below e 32, 64 from e 32 | 8 below e 16, 16 below e 48, 32 from e 48 |
| W > 8 | 8 up to e 8, 64 above | 8 below e 16, 16 from e 16 |
| W4A8, W <= 8 | as W4A4 | 8 below e 16, 16 below e 48, 64 from e 48 |
| W4A8, W > 8 | 8 up to e 12, 16 below e 48, 64 from e 48 | 8 below e 32, 16 below e 96, 64 from e 96 |

The first two rows are W4A4. At EP16 W4A8's 16-token items hold out to higher e than W4A4's. The widths come from
forced 8 / 16 / 32 / 64 sweeps of the four shapes above on these hosts (EP8 and EP16, MXFP4 and W4A8, 1-1024 tokens per
rank); where the shapes disagree on the fastest width the rule takes the one that loses least across them.

Why EP16 wants narrower items than EP8 at the same e: a NIC peer's FC2 partials leave in one put per source once all
of a tile's FC2 items (H / 128 of them) are done, so a wider item holds its rows back longer, and the rank waits for
the last tiles' puts at the end. P2P peers' partials are stored while the items run. At EP8 4 of 7 peers are NIC peers
on the same host, at EP16 12 of 15, 8 of them on the other host. Per-phase stamps, README shape MXFP4 at 320 tokens per
rank (µs from each rank's start, every rank: 8 at EP8, 16 at EP16; one replay each):

| | GPU work done | every partial in (wait) | end |
|---|---|---|---|
| EP8, 16-token items | 508-518 | 549-584 (32-69) | 563-598 |
| EP8, 32-token items | 467-481 | 531-596 (52-118) | 546-611 |
| EP16, 16-token items | 446-495 | 522-565 (35-102) | 536-580 |
| EP16, 32-token items | 398-439 | 522-596 (85-198) | 536-610 |

32-token items finish the GEMM work 37-56 µs earlier in both. At EP8 7 of 8 ranks then end at 546-550 µs (16-token
items: 563-598); at EP16 the longer wait eats the gain.

#### Where the time is at 128-224 tokens per rank

Per-phase stamps (µs from rank 0's first CTA, EP8 on 10.6.142.6), GLM-5.3-Flash at 128 tokens per rank: the rows of
every source are in at 132 (W4A8) / 106 (MXFP4), the GEMM work ends at 547 / 460, the combine at 596 / 491. Single-GPU
Nsight Compute of the same per-expert load (GLM-5.3-Flash, 48 tokens per expert): DRAM throughput 71 % of peak (W4A8)
and 84 % (MXFP4), SM throughput 30 % / 18 %, the warps mostly stalled on global loads (long scoreboard). The GEMM part
is bound by how much of the weight stream is in flight, not by the MMAs; W4A8's rows are twice the bytes of W4A4's, so
its token exchange is longer too.

## Transport

`Sm120Fp4MegaMoeBuffer(..., transport=)`, up to 16 ranks. `'p2p'` (default): one `cudaMalloc` shared over `cudaIpc`,
every peer over PCIe P2P stores. Every other value is one NCCL symmetric window plus a per-pair transport table: each
pair of ranks uses P2P stores (LSA) or NCCL GIN puts through the NIC. The tables and their rules are FlashInfer's Cake
SM120 MegaMoE's (`cake_transport.py`), ported to `deep_gemm/mega/sm120_fp4_transport.py`. Which pairs can use P2P at
all is NCCL's peer mapping of the window, not a rule of this code: on the CX8 hosts (.1/.3/.6) NCCL maps only the GPUs
of one CPU socket when `NCCL_IB_HCA` lists all 8 NICs, and every pair with one NIC.

| `transport` | Pairs over P2P |
|---|---|
| `'all_gin'` | none: every other rank over the NIC |
| `'all_lsa'` | all; an error if NCCL does not map a pair |
| `'lsa_where_reachable'` | the pairs NCCL maps (e.g. the same host), the others over the NIC |
| `'auto'` | the table a setup probe picks: it times `all_gin`, `all_lsa`, `lsa_where_reachable` and `lsa_where_faster` (P2P where mapped and its measured pair bandwidth is at least the NIC's) in an all-to-all of one full call's bytes per destination (a token row out, a bf16 partial row back per route, top-k / world routes per token); a mixed table must beat the best uniform one by 3 % and repeat within 1.5×; rank 0 decides |
| `'tuned'` | the table recorded for this deployment (`'auto'` with `DG_SM120_TRANSPORT_RECORD=1` records it, [`ENV.md`](../sm120_6000d/ENV.md)); `all_gin` when none is |
| `(c0, c1, ...)` | an explicit `world × world` table: 0 on the diagonal, 1 P2P, 2 NIC, symmetric |

NIC peers get NCCL GIN puts (NCCL Device API, from inside the kernel): the payload is staged in this rank's window,
then put to the peer's window. The token rows for one peer (values, scales and metadata, one pool row each) are
staged packed in (its local expert, slot) order and go in one put after the whole dispatch, to a receive region the
peer reads by that packed index; the partials of an FC2 tile go back once all of its FC2 items are done, one put per
source rank with its rows packed the same way. Each put adds its rows to a signal of the receiver (one per source and
kind; 1 for the routing counts), and the receiver waits for the number of rows it expects instead of a flag. The
window transports need a ProcessGroupNCCL group and NCCL 2.30.7 with the Device API (the `nvidia-nccl` wheel); call
`buffer.destroy()` before the process group is destroyed. On a host whose NICs cannot reach each other, pin one NIC:
`NCCL_IB_HCA==mlx5_0` ([`ENV.md`](../sm120_6000d/ENV.md)). On RoCE set `NCCL_IB_TC` to the lossless class ([`ENV.md`](../sm120_6000d/ENV.md)).

On the CX8 hosts every table above uses `'lsa_where_reachable'`: P2P writes from one GPU to GPUs behind several other
CPU root ports at once collapse on these hosts, so only same-socket pairs go over P2P.

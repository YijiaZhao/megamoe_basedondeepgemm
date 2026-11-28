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
* W4A8: faster at every point, except GLM-5.3-Flash at 128 tokens per rank (589.8 vs 535.8 µs).
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
| Tables measured on it | the EP8 tables "on 10.6.142.20"; [vs gc-moe](#vs-gc-moe) (first part); README's RTX 6000D rows | every other table here (EP8 on .6, EP16 on .1 + .6, correctness and `stress` on .1 / .6) |

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

## Results (8x RTX 6000D)

### Correctness

Against a reference that applies the kernel's quantization (activations and SwiGLU output), max
relative L2 over ranks, every tokens-per-item width (8 / 16 / 32 / 64 and the automatic choice), 1-128 tokens per rank
with padding tokens:

| Shape | `w4a8` | `mxfp4` | `nvfp4` |
|---|---:|---:|---:|
| small (H 512 / I 256, 128 experts, top-6) | 1.74e-3 | 1.73e-3 | 1.79e-3 |
| readme (H 3072 / I 1280, 384 experts, top-8) | 1.71e-3 | 1.69e-3 | 6.57e-3 |
| wide (H 5120 / I 256, 128 experts, top-6: the dispatch sends each token in several shared-memory chunks) | 1.72e-3 | 1.69e-3 | 1.72e-3 |

The same values at EP8 over `'p2p'` (10.6.142.1, 10.6.142.20) and over `'lsa_where_reachable'` (10.6.142.6, 4 of the 7 peers over
the NIC). The NVFP4 readme value comes from the single-token case; likely the coarser UE4M3 scales of the FC2 input, not
checked further. EP16 (10.6.142.1 + .6, `'lsa_where_reachable'`, `NCCL_IB_TC=96`, small / readme): W4A8 1.78e-3 /
1.71e-3, MXFP4 1.77e-3 / 1.69e-3, NVFP4 1.79e-3 / 6.57e-3.
`stress` (`dsv41_eplb16` shape, 300 calls with random per-rank delays up to 2 ms; plain, `sparse` routing, and `hog`: a
GEMM competing on a side stream), every format, `'p2p'` (10.6.142.1) and `'lsa_where_reachable'` (10.6.142.6): 0
mismatching outputs, 0 kernel error words.

### Timing

Every number below is from `tests/test_mega_moe_sm120_fp4.py bench <fmt> <M...> lsa_where_reachable` (`p2p` on 10.6.142.20) with
`BENCH_SHAPE=<shape>`:

* One call per point, captured in a CUDA graph. Before each replay 256 MB are written (cold L2: in serving other layers
  run between two calls of one layer), `torch.cuda.synchronize()`, then `dist.barrier()`; CUDA events around the replay
  on each rank. 32 replays, the median of the last 30; three such passes, the median of the three. The tables give GPU
  0's value, µs per MoE layer (dispatch, both GEMMs, combine; the routing top-k is the caller's and not included).
* Balanced routing, weight 1/top-k: copy s of global token g goes to rank s * world / top-k + g % (world / top-k)
  ((g * top-k + s) % world when top-k does not divide the world), each rank hands the copies it receives, in (g, s) order, to its local experts in turn. Random
  E2M1 weight bytes with fixed scales (UE8M0 124 / E4M3 0x38). Global M = tokens over all ranks; tokens / rank = M / ranks; tokens / expert = tokens / rank * top-k * ranks / experts:
  every expert gets exactly that many (± 1 where it does not divide).
* Host: every EP8 table on 10.6.142.6 (8x RTX 6000D, GPUs in pairs behind a ConnectX-8's PCIe switch, 4 ConnectX-8 /
  8 ports), driver 580, clocks not locked, NCCL 2.30.7, container `toklin/vllm-sm120:e10ecb6bd-v41`.
  `NCCL_IB_HCA=mlx5_0,...,mlx5_7`: NCCL maps the 3 peers of a rank's own CPU socket, so `'lsa_where_reachable'` sends
  to those over P2P and to the other 4 over the NIC (GIN). `NCCL_IB_TC=96` (the ports' lossless class, [`ENV.md`](../sm120_6000d/ENV.md)).

GLM-5.3-Flash (H 4096, I 2048, 256 experts, top-8), EP8:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.2 | 161.6 | 168.3 | 162.7 |
| 2 | 16 | 0.5 | 231.5 | 243.8 | 233.3 |
| 4 | 32 | 1 | 388.3 | 409.9 | 388.9 |
| 8 | 64 | 2 | 390.6 | 412.8 | 392.9 |
| 16 | 128 | 4 | 396.0 | 414.9 | 398.0 |
| 32 | 256 | 8 | 431.8 | 459.5 | 435.1 |
| 64 | 512 | 16 | 442.0 | 474.8 | 467.7 |
| 128 | 1024 | 32 | 499.9 | 527.2 | 594.1 |
| 192 | 1536 | 48 | 576.0 | 604.0 | 691.4 |
| 224 | 1792 | 56 | 612.5 | 635.3 | 728.3 |

DeepSeek-V4.1-Flash (H 5120, I 2304, 384 experts, top-6), EP8:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.1 | 175.1 | 180.9 | 177.0 |
| 2 | 16 | 0.2 | 240.2 | 250.2 | 241.3 |
| 4 | 32 | 0.5 | 405.8 | 428.5 | 406.8 |
| 8 | 64 | 1 | 751.1 | 834.4 | 751.1 |
| 16 | 128 | 2 | 760.1 | 845.1 | 761.5 |
| 32 | 256 | 4 | 762.6 | 846.2 | 767.0 |
| 64 | 512 | 8 | 798.2 | 870.9 | 812.7 |
| 128 | 1024 | 16 | 828.0 | 899.1 | 883.0 |
| 192 | 1536 | 24 | 874.7 | 943.0 | 939.2 |
| 224 | 1792 | 28 | 902.5 | 975.0 | 979.8 |

DeepSeek-V4-Flash (H 4096, I 2048, 256 experts, top-6), EP8:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.2 | 148.4 | 152.9 | 149.4 |
| 2 | 16 | 0.4 | 191.6 | 199.0 | 192.3 |
| 4 | 32 | 0.8 | 308.4 | 323.7 | 309.0 |
| 8 | 64 | 1.5 | 388.6 | 408.8 | 388.8 |
| 16 | 128 | 3 | 391.2 | 413.5 | 392.4 |
| 32 | 256 | 6 | 406.3 | 424.5 | 406.1 |
| 64 | 512 | 12 | 440.5 | 463.7 | 455.1 |
| 128 | 1024 | 24 | 470.2 | 494.4 | 501.6 |
| 192 | 1536 | 36 | 535.5 | 560.5 | 626.0 |
| 224 | 1792 | 42 | 558.0 | 585.0 | 653.1 |

README shape (H 3072, I 1280, 384 experts, top-8), EP8:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.2 | 121.4 | 124.8 | 121.8 |
| 2 | 16 | 0.3 | 159.9 | 164.2 | 159.7 |
| 4 | 32 | 0.7 | 219.0 | 231.8 | 221.1 |
| 8 | 64 | 1.3 | 295.7 | 309.7 | 297.8 |
| 16 | 128 | 2.7 | 297.8 | 311.8 | 302.6 |
| 32 | 256 | 5.3 | 306.8 | 321.2 | 311.6 |
| 64 | 512 | 10.7 | 347.1 | 373.1 | 363.7 |
| 128 | 1024 | 21.3 | 375.6 | 396.3 | 420.2 |
| 192 | 1536 | 32 | 413.5 | 441.7 | 486.6 |
| 224 | 1792 | 37.3 | 442.0 | 475.2 | 547.2 |

At 2-64 tokens per rank the call is the read of the active experts' weights from DRAM plus the routing / row / combine
round trips between the ranks.

EP8 on R6KD-BRCM (10.6.142.20), the other PCIe fabric ([Test hosts](#test-hosts)): `bench <fmt> <M...> p2p`, every pair
over P2P, otherwise as above (container `toklin/vllm-sm120:e10ecb6bd-p3`). This host has no second host to pair with,
so it has no EP16 tables.

GLM-5.3-Flash (H 4096, I 2048, 256 experts, top-8), EP8 on 10.6.142.20:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.2 | 119.3 | 124.1 | 120.2 |
| 2 | 16 | 0.5 | 203.3 | 212.3 | 204.0 |
| 4 | 32 | 1 | 361.0 | 380.9 | 360.9 |
| 8 | 64 | 2 | 363.5 | 386.3 | 364.0 |
| 16 | 128 | 4 | 364.4 | 384.4 | 366.3 |
| 32 | 256 | 8 | 366.9 | 387.3 | 372.1 |
| 64 | 512 | 16 | 378.8 | 400.1 | 409.8 |
| 128 | 1024 | 32 | 453.4 | 487.4 | 591.9 |
| 192 | 1536 | 48 | 558.7 | 593.6 | 732.7 |
| 224 | 1792 | 56 | 590.1 | 622.2 | 779.9 |

DeepSeek-V4.1-Flash (H 5120, I 2304, 384 experts, top-6), EP8 on 10.6.142.20:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.1 | 127.4 | 133.2 | 128.5 |
| 2 | 16 | 0.2 | 209.1 | 218.8 | 209.8 |
| 4 | 32 | 0.5 | 377.7 | 400.6 | 377.8 |
| 8 | 64 | 1 | 730.4 | 814.1 | 731.8 |
| 16 | 128 | 2 | 726.1 | 818.2 | 728.3 |
| 32 | 256 | 4 | 724.1 | 812.9 | 729.1 |
| 64 | 512 | 8 | 733.2 | 810.9 | 742.2 |
| 128 | 1024 | 16 | 753.2 | 829.8 | 805.0 |
| 192 | 1536 | 24 | 792.8 | 864.6 | 879.0 |
| 224 | 1792 | 28 | 830.3 | 886.9 | 918.1 |

DeepSeek-V4-Flash (H 4096, I 2048, 256 experts, top-6), EP8 on 10.6.142.20:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.2 | 102.3 | 105.2 | 103.0 |
| 2 | 16 | 0.4 | 160.1 | 167.8 | 160.4 |
| 4 | 32 | 0.8 | 280.9 | 294.9 | 280.6 |
| 8 | 64 | 1.5 | 361.8 | 381.8 | 362.1 |
| 16 | 128 | 3 | 364.6 | 385.6 | 365.6 |
| 32 | 256 | 6 | 365.0 | 385.3 | 366.9 |
| 64 | 512 | 12 | 371.5 | 392.9 | 385.9 |
| 128 | 1024 | 24 | 403.6 | 426.0 | 436.6 |
| 192 | 1536 | 36 | 494.7 | 525.5 | 589.6 |
| 224 | 1792 | 42 | 512.0 | 546.0 | 623.6 |

README shape (H 3072, I 1280, 384 experts, top-8), EP8 on 10.6.142.20:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.2 | 78.3 | 80.9 | 78.6 |
| 2 | 16 | 0.3 | 115.9 | 121.5 | 116.7 |
| 4 | 32 | 0.7 | 189.2 | 198.7 | 189.6 |
| 8 | 64 | 1.3 | 265.2 | 278.8 | 266.1 |
| 16 | 128 | 2.7 | 267.4 | 281.9 | 268.1 |
| 32 | 256 | 5.3 | 270.1 | 283.2 | 271.6 |
| 64 | 512 | 10.7 | 279.7 | 297.4 | 299.0 |
| 128 | 1024 | 21.3 | 310.2 | 331.3 | 388.3 |
| 192 | 1536 | 32 | 383.6 | 404.3 | 490.3 |
| 224 | 1792 | 37.3 | 433.5 | 449.5 | 553.6 |

EP16, two hosts (10.6.142.1 node rank 0 + 10.6.142.6, 16 ranks): `tests/test_mega_moe_sm120_fp4.py bench` as above,
launched like `bench/run_ep16.sh` (same docker flags, `NCCL_IB_TC=96`, `NCCL_IB_HCA` unset: each rank reaches the 3 GPUs of its own socket
over P2P, the other 12 over the NIC; containers `toklin/vllm-sm120:e10ecb6bd-p3` on .1, `-v41` on .6), every one of the 16 GPUs idle before each run. The
values are the slowest rank's (every rank's median as above, then the maximum over the 16 ranks); the layer waits for it.

GLM-5.3-Flash (H 4096, I 2048, 256 experts, top-8), EP16:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.5 | 176.7 | 182.7 | 177.4 |
| 2 | 32 | 1 | 257.1 | 266.8 | 257.6 |
| 4 | 64 | 2 | 260.1 | 269.3 | 260.4 |
| 8 | 128 | 4 | 274.1 | 284.3 | 274.9 |
| 16 | 256 | 8 | 292.8 | 303.7 | 295.1 |
| 32 | 512 | 16 | 303.1 | 315.6 | 318.0 |
| 64 | 1024 | 32 | 326.9 | 339.0 | 361.8 |
| 128 | 2048 | 64 | 401.3 | 413.2 | 474.8 |
| 192 | 3072 | 96 | 537.7 | 561.6 | 692.0 |
| 224 | 3584 | 112 | 582.4 | 605.9 | 736.0 |

DeepSeek-V4.1-Flash (H 5120, I 2304, 384 experts, top-6), EP16:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.2 | 189.1 | 196.0 | 189.0 |
| 2 | 32 | 0.5 | 271.2 | 281.1 | 270.7 |
| 4 | 64 | 1 | 435.3 | 458.5 | 434.2 |
| 8 | 128 | 2 | 443.1 | 464.5 | 443.1 |
| 16 | 256 | 4 | 461.3 | 487.0 | 462.0 |
| 32 | 512 | 8 | 476.9 | 498.7 | 480.8 |
| 64 | 1024 | 16 | 499.6 | 523.1 | 514.2 |
| 128 | 2048 | 32 | 547.0 | 569.7 | 654.3 |
| 192 | 3072 | 48 | 613.2 | 630.7 | 714.1 |
| 224 | 3584 | 56 | 640.5 | 665.9 | 759.5 |

DeepSeek-V4-Flash (H 4096, I 2048, 256 experts, top-6), EP16:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.4 | 162.6 | 167.3 | 162.5 |
| 2 | 32 | 0.8 | 220.2 | 228.2 | 220.8 |
| 4 | 64 | 1.5 | 260.0 | 269.9 | 260.2 |
| 8 | 128 | 3 | 264.0 | 273.5 | 265.7 |
| 16 | 256 | 6 | 279.6 | 289.7 | 281.0 |
| 32 | 512 | 12 | 298.0 | 309.9 | 297.0 |
| 64 | 1024 | 24 | 317.8 | 328.0 | 336.5 |
| 128 | 2048 | 48 | 369.2 | 379.1 | 425.3 |
| 192 | 3072 | 72 | 492.7 | 514.0 | 609.5 |
| 224 | 3584 | 84 | 515.4 | 537.4 | 650.9 |

README shape (H 3072, I 1280, 384 experts, top-8), EP16:

| tokens / rank | global M | tokens / expert | MXFP4 | NVFP4 | W4A8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.3 | 134.8 | 139.2 | 135.8 |
| 2 | 32 | 0.7 | 172.9 | 177.3 | 173.0 |
| 4 | 64 | 1.3 | 212.5 | 219.9 | 212.6 |
| 8 | 128 | 2.7 | 223.6 | 230.9 | 224.8 |
| 16 | 256 | 5.3 | 246.6 | 253.6 | 248.0 |
| 32 | 512 | 10.7 | 257.1 | 266.8 | 260.3 |
| 64 | 1024 | 21.3 | 275.2 | 282.7 | 291.0 |
| 128 | 2048 | 42.7 | 313.5 | 323.0 | 359.3 |
| 192 | 3072 | 64 | 379.9 | 384.5 | 456.0 |
| 224 | 3584 | 74.7 | 464.0 | 457.7 | 528.9 |

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
| 1 | 8 | 138.6 | 175.6 | 164.9 | 120.8 | 120.8 | 124.9 |
| 2 | 16 | 213.7 | 250.4 | 243.3 | 204.8 | 202.8 | 213.0 |
| 3 | 24 | 260.7 | 305.5 | 322.1 | 280.6 | 280.6 | 294.9 |
| 4 | 32 | 270.0 | 315.6 | 401.8 | 360.4 | 360.4 | 383.0 |
| 6 | 48 | 309.2 | 358.7 | 402.1 | 364.5 | 362.5 | 385.0 |
| 8 | 64 | 330.5 | 380.7 | 406.1 | 364.5 | 364.5 | 387.1 |
| 12 | 96 | 352.3 | 403.1 | 410.3 | 366.6 | 364.5 | 387.0 |
| 16 | 128 | 367.9 | 414.0 | 414.3 | 364.5 | 366.6 | 387.0 |
| 24 | 192 | 398.6 | 446.6 | 435.8 | 368.6 | 366.6 | 387.1 |
| 32 | 256 | 370.4 | 422.6 | 418.4 | 372.7 | 366.6 | 389.1 |
| 48 | 384 | 398.6 | 462.7 | 435.8 | 385.0 | 372.7 | 393.2 |
| 64 | 512 | 405.8 | 458.7 | 441.5 | 407.5 | 378.9 | 402.4 |
| 96 | 768 | 475.7 | 534.8 | 498.8 | 464.9 | 402.4 | 421.9 |
| 128 | 1024 | 484.4 | 535.4 | 535.8 | 589.8 | 454.6 | 489.4 |
| 160 | 1280 | 545.4 | 597.9 | 626.9 | 686.1 | 536.6 | 573.4 |
| 192 | 1536 | 841.9 | 832.7 | 766.1 | 731.1 | 559.1 | 593.9 |

DeepSeek-V4.1-Flash, EP8:

| tokens / rank | global M | gc README | gc inner=100 | gc inner=1 | gc | W4A8 | MXFP4 | NVFP4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 190.9 | 183.0 | 220.6 | 169.5 | 127.0 | 127.0 | 133.1 |
| 2 | 16 | 281.1 | 270.6 | 314.1 | 259.5 | 208.9 | 208.9 | 219.1 |
| 3 | 24 | – | 329.7 | 379.2 | 340.6 | 292.9 | 292.9 | 309.2 |
| 4 | 32 | 401.8 | 369.0 | 421.8 | 428.7 | 376.8 | 376.8 | 403.4 |
| 6 | 48 | – | 447.2 | 507.6 | 597.6 | 550.9 | 548.8 | 612.4 |
| 8 | 64 | 589.1 | 536.4 | 605.9 | 776.8 | 735.2 | 731.1 | 817.1 |
| 12 | 96 | – | 614.9 | 693.5 | 778.8 | 729.1 | 727.0 | 813.0 |
| 16 | 128 | 706.8 | 642.2 | 724.3 | 780.8 | 729.1 | 727.0 | 817.2 |
| 24 | 192 | – | 679.5 | 761.8 | 784.9 | 731.1 | 725.0 | 817.1 |
| 32 | 256 | 746.2 | 693.4 | 784.1 | 789.9 | 727.0 | 725.0 | 817.1 |
| 48 | 384 | – | 700.9 | 792.1 | 800.3 | 735.2 | 729.1 | 806.9 |
| 64 | 512 | 773.8 | 719.9 | 817.6 | 805.5 | 741.3 | 735.2 | 811.0 |
| 96 | 768 | – | 784.3 | 912.6 | 826.8 | 772.1 | 743.4 | 821.2 |
| 128 | 1024 | 852.9 | 752.2 | 845.2 | 838.2 | 802.8 | 751.6 | 831.5 |
| 160 | 1280 | – | 783.9 | 870.9 | 872.1 | 845.8 | 772.1 | 849.9 |
| 192 | 1536 | – | 838.9 | 919.3 | 902.3 | 882.7 | 794.6 | 868.3 |

Against the "gc" column (same script, weights and routing), this kernel is faster at (points of the tables above):

| Shape | tokens / rank | W4A8 | MXFP4 | NVFP4 |
|---|---|---:|---:|---:|
| GLM-5.3-Flash | 1-128 | 13 / 14 (not 128) | 14 / 14 | 14 / 14 |
| GLM-5.3-Flash | 1-192 | 14 / 16 (not 128, 160) | 16 / 16 | 16 / 16 |
| DeepSeek-V4.1-Flash | 1-128 | 14 / 14 | 14 / 14 | 6 / 14 (not 6-64) |
| DeepSeek-V4.1-Flash | 1-192 | 16 / 16 | 16 / 16 | 8 / 16 (not 6-64) |

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
| 1 | 8 | 230.0 | 262.8 | 161.8 | 161.8 | 165.9 |
| 2 | 16 | 482.6 | 484.1 | 243.7 | 241.7 | 251.9 |
| 3 | 24 | 678.7 | 637.7 | 325.6 | 324.6 | 340.0 |
| 4 | 32 | 1009.7 | 984.4 | 409.6 | 409.6 | 428.0 |
| 6 | 48 | 1659.1 | 1620.0 | 411.6 | 412.7 | 434.2 |
| 8 | 64 | 2322.2 | 2298.3 | 419.8 | 421.9 | 440.3 |
| 12 | 96 | 3554.9 | 3454.8 | 429.1 | 428.0 | 446.5 |
| 16 | 128 | 4691.4 | 4536.3 | 433.2 | 432.1 | 449.5 |
| 24 | 192 | 6828.6 | 6844.9 | 437.2 | 436.2 | 448.5 |
| 32 | 256 | 2585.6 | 2622.0 | 450.6 | 443.4 | 453.6 |
| 48 | 384 | 4105.5 | 3793.6 | 456.7 | 450.6 | 464.9 |
| 64 | 512 | 5552.8 | 4937.4 | 473.1 | 462.8 | 468.0 |
| 96 | 768 | 8541.6 | 8090.0 | 504.8 | 478.2 | 493.6 |
| 128 | 1024 | 11115.2 | 10863.3 | 596.0 | 513.0 | 530.4 |
| 160 | 1280 | 14692.6 | 14112.5 | 656.4 | 555.0 | 577.5 |
| 192 | 1536 | 18971.0 | 18934.5 | 702.5 | 580.6 | 606.2 |

DeepSeek-V4.1-Flash, EP8 on 10.6.142.6:

| tokens / rank | global M | gc inner=100 | gc inner=1 | W4A8 | MXFP4 | NVFP4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 229.3 | 230.5 | 174.1 | 174.1 | 178.2 |
| 2 | 16 | 419.6 | 408.8 | 256.0 | 255.0 | 264.2 |
| 3 | 24 | 546.9 | 517.0 | 337.9 | 335.8 | 352.2 |
| 4 | 32 | 777.1 | 769.1 | 421.9 | 421.9 | 446.4 |
| 6 | 48 | 1399.6 | 1311.5 | 592.9 | 593.9 | 649.2 |
| 8 | 64 | 2157.4 | 1964.2 | 770.0 | 772.1 | 852.0 |
| 12 | 96 | 3523.4 | 3364.0 | 770.0 | 771.1 | 854.0 |
| 16 | 128 | 4557.2 | 4420.0 | 786.4 | 786.4 | 864.3 |
| 24 | 192 | 6906.7 | 6755.8 | 790.5 | 784.4 | 866.3 |
| 32 | 256 | 2385.6 | 2406.9 | 796.6 | 792.5 | 866.3 |
| 48 | 384 | 3843.1 | 3428.0 | 798.7 | 798.7 | 867.3 |
| 64 | 512 | 5074.2 | 4917.7 | 812.0 | 802.8 | 868.4 |
| 96 | 768 | 7682.6 | 7526.3 | 838.6 | 813.0 | 880.7 |
| 128 | 1024 | 10063.6 | 9934.3 | 882.7 | 827.4 | 897.0 |
| 160 | 1280 | 12915.0 | 12777.5 | 909.3 | 851.9 | 921.6 |
| 192 | 1536 | 15457.4 | 15385.2 | 935.9 | 876.5 | 942.1 |

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
* Activation ring (W4A8 items of 32 / 64 tokens, W4A4 items of 64): the 8 warps of a CTA work on the same token rows,
  so the rows go once into a shared-memory ring of K256 steps (3 stages W4A8, 4 W4A4; 51 / 36 KB with 64-token items,
  25.5 KB with W4A8 32-token items, over `s_tile`) instead of every warp loading them from global memory. Each warp
  copies its 1/8 of a step with 16 B `cp.async` once it has finished the step before, a full and an empty mbarrier per
  stage, no CTA-wide sync; padding rows are not copied. W4A4 32-token items do not use it (2 % slower there).
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
  accumulators and the step's activations spill); `s_tile` (32 KB at 64 tokens) and the activation ring share one
  dynamic shared-memory block.
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
every source are in at 131 (W4A8) / 102 (MXFP4), the GEMM work ends at 534 / 454, the combine at 575 / 479. The GEMM
part is bound by the weight stream from DRAM, not by the MMAs; W4A8's rows are twice the bytes of W4A4's, so its token
exchange is longer too.

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

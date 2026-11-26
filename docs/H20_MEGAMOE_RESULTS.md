# H20 MegaMoE results (8x H20-3e)

Moved verbatim from the README of 2026-09-13; the measurement method, pass counts and scripts are as described here.
Back to the [README](../README.md); Blackwell results: [B200_MEGAMOE_RESULTS.md](B200_MEGAMOE_RESULTS.md).

## Performance results (2026-09-13)

Microseconds; M = global tokens over the 8 ranks (M2 / M4 / M8 = 1 row per rank, M16 = 2 rows);
fused backend (`mxfp4_mega_moe_fused` / `qoq_mega_moe_fused`); GPU 0, median of the last 3
replays, median over the clean passes (inter-rank Mega-start skew <= 20 us; pass counts in the
measurement method).

| | Precision | 0-token rows | M2 | M4 | M8 | M16 |
|---|---|---|---:|---:|---:|---:|
| E2E, normal routing | MXFP4 | 0-token rows ROUTED (no skip) | 64.0 | 64.4 | 68.0 | 80.9 |
| E2E, normal routing | MXFP4 | **0-token rows UNROUTED (skip, default)** | 53.4 | 59.4 | 67.8 | 80.8 |
| E2E, normal routing | QoQ | 0-token rows ROUTED (no skip) | 62.3 | 61.8 | 66.5 | 79.7 |
| E2E, normal routing | QoQ | **0-token rows UNROUTED (skip, default)** | 51.8 | 59.1 | 66.4 | 79.6 |
| E2E, forced-balanced | MXFP4 | 0-token rows ROUTED (no skip) | 45.0 | 53.9 | 63.0 | 80.7 |
| E2E, forced-balanced | MXFP4 | **0-token rows UNROUTED (skip, default)** | 44.6 | 54.0 | 62.2 | 80.9 |
| E2E, forced-balanced | QoQ | 0-token rows ROUTED (no skip) | 44.4 | 54.2 | 60.9 | 80.3 |
| E2E, forced-balanced | QoQ | **0-token rows UNROUTED (skip, default)** | 44.8 | 54.2 | 60.4 | 80.6 |
| Mega-only, balanced | MXFP4 | - | 38.8 | 47.4 | 56.7 | 76.2 |
| Mega-only, balanced | QoQ | - | 37.1 | 45.2 | 54.0 | 72.6 |
| FE kernel | both | 0-token rows ROUTED (no skip) | 2.5–2.7 | 2.5–2.7 | 2.5–2.7 | 2.8–3.0 |
| FE kernel | both | **0-token rows UNROUTED (skip, default)** | 2.8 | 2.8–2.9 | 2.7–2.8 | 3.0–3.1 |

0-token rows: at M2 / M4 only the ranks that own a token have a real input row; the other ranks' row is all-zero padding (a "0-token row"). **0-token rows ROUTED** = `DG_FE_ZERO_ROW_UNROUTED=0`, the frontend treats the 0-token row like a real token and routes it (to experts 0..7 on rank 0), so the MegaMoE does useless work for it. **0-token rows UNROUTED** = the default `DG_FE_ZERO_ROW_UNROUTED=1`, the frontend detects the all-zero row and marks it as not routed; the MegaMoE dispatches nothing for it and writes a zero output (exact: x = 0 gives y = 0). Only the normal-routing rows at M2 / M4 are affected.

Rows: **E2E** = frontend kernel + fused MegaMoE kernel in one CUDA graph, span from the frontend
start to the MegaMoE end; **normal routing** = the frontend's real top-8 of random hidden rows;
**forced-balanced** = the frontend runs, then one graph memcpy node overrides its routing with
the balanced assignment (`expert = s * 48 + (g + 7 s) % 48`, weight 1/8; `DG_PROFILE_FORCE_BALANCED=1`);
**Mega-only** = the fused MegaMoE kernel alone with the same balanced assignment; **FE kernel** =
the frontend kernel span (`router_cc_lean_kernel`, `DG_FE_SELECT_IN_MEGA=1`: the kernel ends after
the keys, the MegaMoE prologue selects). At M2 / M4 only the ranks that own a token carry a real row
(ranks 0, 4 / 0, 1, 4, 5); the other ranks' row is all-zero padding, which the frontend leaves
unrouted (`DG_FE_ZERO_ROW_UNROUTED=1`, exact: x = 0 gives y = 0). The 256 MiB L2 flush, the TP4
reduce-scatter and the TP4 all-gather sit between the replays, outside the graph.

**Measurement method:** host `10.6.131.8`, eight H20-3e with the SM clock locked at 1830 MHz
(`nvidia-smi -lgc 1830,1830`, verified by the capture script), container `fe5c_build` (the
pinned image above). All rows: `main` `9d2f7e9` (one build, `_C` extension sha256 `3acb4e2`;
the default code paths of this tree), one session 2026-09-13 05:27-07:44 UTC: 5 interleaved
passes (per pass: E2E normal routed, E2E normal unrouted, E2E forced-balanced routed, E2E
forced-balanced unrouted, Mega-only; `tests/fe5_campaign_zr_interleave.sh`) plus 3 extra E2E
passes for the cells with fewer than 3 clean passes; a cell is the median over its clean passes,
i.e. the passes whose last 3 replays all have an inter-rank Mega-start skew <= 20 us (E2E and FE:
5-8 clean of 8 passes per cell at M4 / M8 / M16, 2-7 of 8 at M2; Mega-only: 5 of 5).
`scripts/capture_four_api_h20_timelines.sh` runs `tests/profile_four_api_h20.py` under Nsight
Systems (`--trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none`) with
`DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30`: after 2 warm-up replays, the 30 measured replays of a
case are enqueued back-to-back on the stream (L2 flush, input copy, reduce-scatter, graph replay,
all-gather; no per-iteration `torch.cuda.synchronize()` / `dist.barrier()`), so the ranks align
through the on-stream collectives. `scripts/summarize_four_api_h20_last3.py` reads the report of
GPU 0, takes the last 3 replays, measures each span (E2E: frontend kernel start -> MegaMoE kernel
end; Mega-only: kernel start -> end; FE: frontend kernel start -> end) and reports their median;
`tests/fe5_summarize_campaign.py` takes the median over the passes (Mega-only: 5 passes). Env of
the runs: `DG_FE_SELECT_IN_MEGA=1 DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30`, plus
`DG_PROFILE_FORCE_BALANCED=1` for the forced-balanced rows and `DG_FE_ZERO_ROW_UNROUTED=0` for the
routed rows; every other knob at its library default.

Reproduce (inside the container, repo root; 8 idle GPUs, clock locked):

```bash
# E2E normal + forced-balanced, 3 passes each, then one Mega-only pass; prints the per-capture tables
bash tests/fe5_campaign_streamed.sh 3 /raid/kimi/results/fe5/cap_streamed            # ~16 min
python3 tests/fe5_summarize_campaign.py /raid/kimi/results/fe5/cap_streamed          # medians over the passes
# one capture set by hand (what the campaign script calls)
DG_FE_SELECT_IN_MEGA=1 DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30 DG_PROFILE_FORCE_BALANCED=0 \
  OUT=/raid/kimi/results/x SCOPES=e2e BACKENDS=fused QUANTS="mxfp4 qoq" TOKENS_LIST="2 4 8 16" FORCE=1 \
  bash scripts/capture_four_api_h20_timelines.sh
SCOPES=e2e BACKENDS=fused python3 scripts/summarize_four_api_h20_last3.py /raid/kimi/results/x
```

Env knobs of this tree (all read once per process; the table above uses the defaults except where noted):

| Knob | Default | Meaning |
|---|---|---|
| `DG_FE_SELECT_IN_MEGA` | 0 (the profiling driver and the campaign scripts set 1) | cc frontend ends after the 384 keys per token; the fused MegaMoE prologue selects top-8 + softmax |
| `DG_FE_ROUTER_L2_PERSIST` | 1 on the cc path, 0 otherwise | router weights in the persisting-L2 set-aside (access-policy-window launch attribute) |
| `DG_FE_ROUTER_WLAYOUT` | `fragment` | swapab path: router weights permuted once into m16n8k16 A-fragment order (`fable_router_weight_fragment_layout`); `row`, `pre` (caller permuted) |
| `DG_FE_STAMPS` | 0 | per-CTA `%globaltimer` phase stamps of the frontend kernel (`fable_frontend_stamps`) |
| `DG_FE_FORCE_BALANCED` | 0 | `tests/test_four_api_correctness.py`: the forced-balanced override, reference follows it |
| `DG_FE_ZERO_ROW_UNROUTED` | 1 | cc frontend: an all-zero hidden row (a padding row of a rank that owns no token) is left unrouted (all-zero keys / `topk_idx` -1, weight 0) instead of tie-broken onto experts 0..7 on rank 0; exact (x = 0 gives y = 0 either way), non-zero rows bit-identical |
| `DG_FP4_FINE_COMBINE` | 1 | per-token arrival counters replace the combine NVLink barrier (combine warps claim tokens from a per-launch ticket) |
| `DG_FP4_PUSH_DISPATCH`, `DG_FP4_PUSH_DISPATCH_MAX_M` | 1, 16 | source rank pushes routed rows + tickets into the destination pool during routing (<= MAX_M global tokens) |
| `DG_FP4_LEAN_ROUTING` | 1 | publish non-zero experts only; no cross-rank count broadcast under push |
| `DG_FP4_PUSH_DONE_FLAGS` | 1 | per-rank DONE count replaces NVLink barrier #1 (lean push) |
| `DG_FP4_QOQ_INLINE_S2` | 1 | QoQ: fold s2 into the int8 weight at decode, one int32 set per task, one promote |
| `DG_FP4_QIS2_PREFETCH_PACKED` | 1 | QoQ inline-s2 loop: packed-word LDS before the wgmma wait |
| `DG_FP4_STREAMK`, `DG_FP4_STREAMK_MAX_M` | 1, 8 | stream-K unit ranges over all 78 SMs when the launch has fewer L1 tasks than SMs (M <= 4 in practice) |
| `DG_FP4_L1_BN`, `DG_FP4_L2_BN`, `DG_FP4_BN512_MIN_M`, `DG_FP4_BN512_MAX_M` | 512, 256, 16, 16 | wide (512-row) L1 tasks for M = 16 |
| `DG_FP4_SPLITK_L1` | 1 | L1 tasks of the last partial wave run as two K halves |
| `DG_FP4_DIST_BCAST` | 1 | dispatch expert-count broadcast spread over all SMs |
| `DG_FP4_PREFETCH_KBLOCKS` | 0 under push dispatch or >= 16 rows per rank, else 8 | weight K-blocks prefetched into L2 per task while waiting for activations |
| `DG_FP4_BM8_STAGES` | 4 | pipeline depth of the BM8 tier (2 K128 blocks per stage; 1 with wide tasks) |
| `DG_FP4_HALF_TILE`, `DG_FP4_L2_HALFROW`, `DG_FP4_SPLITK_L2`, `DG_FP4_NVL_FAST_EPI`, `DG_FP4_SWAP_PIPE`, `DG_FP4_QIS2_FRAGS`, `DG_FP4_QIS2_ILV`, `DG_FP4_RF_PREFETCH_PACKED`, `DG_FP4_POOL_STRIDE_DEBUG`, `DG_FP4_SPIN_TIMEOUT` | 0 / 2 (frags) | alternative task shapes and debug switches, off in the table above (documented in `csrc/jit_kernels/impls/sm90_fp4_mega_moe_h20_fused.hpp`) |
| `DG_PROFILE_STREAMED`, `DG_PROFILE_ITERS` | 0, 8 | profiling: back-to-back replays (1) and replay count; the table uses 1 / 30 |
| `DG_PROFILE_FORCE_BALANCED` | 0 | profiling: forced-balanced routing in the E2E scope (memcpy override) |
| `DG_PROFILE_HOST_BARRIER` | 0 | profiling: `dist.barrier()` before each replay when not streamed |
| `DG_BENCH_FLUSH_L2_BYTES` | 256 MiB | profiling: L2 flush size between replays |


## H4096 / I1536 / E640 / top-8, EP8 and two-node EP16 (2026-11-25)

Hidden 4096, intermediate 1536, 640 experts (80 per rank in EP8, 40 per rank in EP16), top-8, MXFP4. EP8: the v2
kernel (`deep_gemm.mega.h20_v2.mxfp4_mega_moe_h20_v2`, `sm90_fp4_mega_moe_h20_v2.cu`: two independent CTAs per SM,
each two RS-wgmma warpgroups + a producer warp streaming weight blocks (bulk copy) and token blocks (TMA, 128B swizzle)
through a 4-stage ring, epoch flags instead of cross-rank barriers). EP16: the same v2 kernel per node over 2T rows (`deep_gemm.mega.h20_v2.H20V2EP16`): it first puts its T rows to the rail
peer (same local rank on the other node) with one NCCL GIN put and runs the node's 320 experts over its own and the
peer's rows; the combine sums the peer's rows per 256-column chunk and puts each row back (one GIN signal per row), the
own rows add the peer's partial once its signal arrives. Global M = tokens before top-k summed over all ranks; the test spreads them evenly
(`--global-tokens M`: rank r gets rows [M·r/W, M·(r+1)/W)).

| Global M (all ranks) | EP8 tokens/rank | EP8 µs | EP8 MBU | GC MoE EP8 µs | EP16 tokens/rank | EP16 µs | EP16 MBU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 77.4 | 30.5% | 91.5 | 0.5 | 99.0 | 11.9% |
| 16 | 2 | 106.2 | 44.4% | 110.3 | 1 | 114.6 | 20.6% |
| 24 | 3 | 136.0 | 52.0% | 150.7 | 1.5 | 123.6 | 28.6% |
| 32 | 4 | 161.1 | 58.6% | 189.8 | 2 | 142.5 | 33.1% |
| 40 | 5 | 190.8 | 61.8% | 208.3 | 2.5 | 159.6 | 37.0% |
| 48 | 6 | 223.0 | 63.5% | 238.1 | 3 | 172.3 | 41.1% |
| 56 | 7 | 251.0 | 65.8% | 268.4 | 3.5 | 181.3 | 45.5% |
| 64 | 8 | 281.5 | 67.1% | 294.3 | 4 | 195.3 | 48.3% |
| 72 | 9 | 307.7 | 69.0% | 324.9 | 4.5 | 223.4 | 47.5% |
| 80 | 10 | 337.2 | 70.0% | 353.2 | 5 | 232.6 | 50.7% |
| 88 | 11 | 337.3 | 69.9% | 353.5 | 5.5 | 232.4 | 50.8% |
| 96 | 12 | 337.3 | 69.9% | 352.1 | 6 | 232.4 | 50.8% |
| 104 | 13 | 337.4 | 69.9% | 374.7 | 6.5 | 234.0 | 50.4% |
| 112 | 14 | 338.5 | 69.7% | 354.3 | 7 | 233.2 | 50.6% |
| 120 | 15 | 337.9 | 69.8% | 354.6 | 7.5 | 235.0 | 50.2% |
| 128 | 16 | 338.1 | 69.8% | 353.3 | 8 | 235.7 | 50.0% |

Method (`tests/test_h20_ep16.py --impl v2`): forced-balanced routing, one CUDA graph per call, 256 MiB written before every
replay (cold L2), CUDA-event span of the replay, median of 30 replays per rank, max over the ranks. Inputs are
already quantised (FP8 rows + K128 scales, top-k ids / weights) in device tensors; no
frontend, no input quantisation in the span. EP8: the one v2 kernel launch, NVLink P2P (cudaIpc) only, clocks locked
at 1830 MHz. EP16: `--transport gin --impl v2`, the one v2 launch per rank with the cross-node exchange inside it
(NCCL GIN, GDAKI backend), two hosts, clocks locked at 1830 MHz. EP16 rows below 1 token per rank: half of the ranks have 1 row, the others 0. MBU = weight bytes of the experts hit (FP4 payload +
scales as stored) / (ranks × 4.0 TB/s × time).

GC MoE: [Bruce-Lee-LY/gc-moe](https://github.com/Bruce-Lee-LY/gc-moe) `sm90_fp8_fp4_gc_moe` at commit
[`b7317477ace99fbe399e0b73eea9329fa6595261`](https://github.com/Bruce-Lee-LY/gc-moe/commit/b7317477ace99fbe399e0b73eea9329fa6595261) (2026-11-24, "feat: add gc moe k3"), its own
`benchmarks/benchmark_sm90_gc_moe.py` with this shape and its routing replaced by the same forced-balanced routing, `--inner-iters 1` (L2 flushed
before every replay; the script's default 100 flushes once per 100 replays), clocks locked at 1830 MHz. It needs
peer-accessible GPUs (one host), so it has no EP16 column.

Correctness: every EP8 / EP16 point above, 2 seeds each, against the exact-quantised reference (FP8 activations, dequantised
weights, quantised intermediate): cos_min >= 0.99993; norm ratio 0.9997-1.0000.

```bash
# EP8 (one host)
torchrun --standalone --nproc-per-node 8 tests/test_h20_ep16.py --mode ep8 --quant mxfp4 --impl v2 \
  --hidden 4096 --intermediate 1536 --experts 640 --topk 8 --global-tokens 64
# EP16 (two hosts, node rank 0 / 1)
torchrun --nnodes 2 --node-rank <0|1> --nproc-per-node 8 --master-addr <node0> --master-port 29655 \
  tests/test_h20_ep16.py --mode ep16 --quant mxfp4 --transport gin --impl v2 \
  --hidden 4096 --intermediate 1536 --experts 640 --topk 8 --global-tokens 128
```

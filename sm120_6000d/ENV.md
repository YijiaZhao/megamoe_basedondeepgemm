# SM120 MegaMoE (`fp4_mega_moe_sm120`): environment variables

None is needed: with none set, the extension is built on first use into `~/.deep_gemm/sm120_mega_moe` and every kernel
choice comes from the call's arguments. None of them changes what the kernel computes or how fast it runs.

| Variable | Read by | Default | Effect |
|---|---|---|---|
| `DG_SM120_MEGA_MOE_BUILD_DIR` | `deep_gemm/mega/sm120_fp4.py` | `~/.deep_gemm/sm120_mega_moe` | where the torch extension is built and cached (one subdirectory per format) |
| `DG_SM120_FP4_MEGA_MOE_SO_MXFP4`, `..._NVFP4`, `..._W4A8` | `deep_gemm/mega/sm120_fp4.py` | unset | path of a prebuilt extension `.so` for that format (the build without the NIC code); when set, that build is not compiled |
| `TILE` | `tests/test_mega_moe_sm120_fp4.py bench` only | `0` | forces the tokens per work item (`8` or `16`) in the timing; `0` keeps the kernel's own choice |
| `DG_SM120_TRANSPORT_TABLES` | `deep_gemm/mega/sm120_fp4_transport.py` | `~/.deep_gemm/sm120_transport_tables.json` | the recorded transport tables `transport='tuned'` reads, one per deployment (format, world, shape, max tokens, P2P / GIN topology) |
| `DG_SM120_TRANSPORT_RECORD` | `deep_gemm/mega/sm120_fp4.py` | unset | `1`: `transport='auto'` records the table it picks there (rank 0) |
| `DG_SM120_TRANSPORT_REPORT` | `deep_gemm/mega/sm120_fp4.py` | unset | path: `transport='auto'` writes its probe report there (rank 0: pair bandwidths, every candidate's times, the choice) |

The tokens per work item of a normal call are the `tile_tokens` argument of `fp4_mega_moe_sm120` (`0`: chosen by the
kernel), and the transport is the buffer's `transport` argument; neither is an environment variable.

Window transports (every `transport` but `'p2p'`): NCCL reads its own variables. On a host whose NICs cannot reach each
other (on .20 they sit on different subnets, and the NCCL window registration fails), pin every rank to one NIC with
`NCCL_IB_HCA==mlx5_0`. `NCCL_IB_HCA` also decides which pairs NCCL maps for P2P, i.e. what `'lsa_where_reachable'` /
`'auto'` can use: on the CX8 hosts (.1/.3/.6) all 8 NICs map only the GPUs of one CPU socket, one NIC maps every pair.

## Two-host EP16 on the CX8 hosts (.3 + .1): the network setup the W4A4.md EP16 numbers were measured with

Launcher: [`bench/run_ep16.sh`](bench/run_ep16.sh), started on both hosts at once, node rank 0 on .3
(`MASTER=10.6.142.3`, port 29651). Checked again 2026-11-12 with `NCCL_DEBUG=INFO` (MXFP4 small EP16
`'lsa_where_reachable'`: PASS).

| Item | Value |
|---|---|
| Hosts | 10.6.142.3 (R6KD-CX8aaS-GPU-03) and 10.6.142.1 (R6KD-CX8aaS-GPU-01), same model, 8 x RTX 6000D each |
| Driver / kernel | 580.95.05; `amd_iommu=off` on the kernel command line; `nvidia_peermem` loaded (`gdrdrv` only on .3, not needed) |
| Container | `toklin/vllm-sm120:e10ecb6bd-p3`, NCCL 2.30.7+cuda13.3 (the `nvidia-nccl-cu13` wheel) |
| `docker run` flags the NIC path needs | `--network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --gpus all -u root` (root: the image's user cannot write the build dir) |
| Bootstrap network | `bond0` (`mlx5_bond_0`, 200 Gb/s, 10.6.142.x/24): `NCCL_SOCKET_IFNAME=bond0`, also torchrun's master address |
| Data NICs | 4 x ConnectX-8, both ports of each up: 8 x 400 Gb/s RoCE. Each port is its own /31 link to the fabric (172.16.{0,32,64,96,128,160,192,224}.x; .1 ends in .1, .3 in .5), with a 172.16.0.0/13 route through every port: every port of one host reaches every port of the other (ping-checked port by port) |
| `NCCL_IB_HCA` | unset (all NICs). NCCL lists the 8 ports plus `mlx5_bond_0`, merges the two ports of each card into one device (`NET/IB/9`-`12` in the log) and gives each GPU the card behind its own PCIe switch, with GPUDirect RDMA (`GDRDMA`, DMA-BUF) |
| NIC transport | `GIN_IB_GDAKI` (GIN type 3: the GPU posts the RDMA itself), 4 contexts per rank; the extension is built as `sm120_fp4_mega_moe_<fmt>_gin_gdaki` |
| P2P pairs | with all NICs NCCL maps only the 4 GPUs of a rank's own socket; `'lsa_where_reachable'` sends those over P2P and the other 12 ranks over the NIC |

| Card | Ports (netdev) | PCIe | Behind the switch of | Socket |
|---|---|---|---|---|
| 1 | `mlx5_2`, `mlx5_3` (`enp3s0f0np0`, `enp3s0f1np1`) | 03:00 | GPU 0-1 | 0 |
| 2 | `mlx5_0`, `mlx5_1` (`enp115s0f0np0`, `enp115s0f1np1`) | 73:00 | GPU 2-3 | 0 |
| 3 | `mlx5_6`, `mlx5_7` (`enp131s0f0np0`, `enp131s0f1np1`) | 83:00 | GPU 4-5 | 1 |
| 4 | `mlx5_4`, `mlx5_5` (`enp243s0f0np0`, `enp243s0f1np1`) | f3:00 | GPU 6-7 | 1 |

Other NIC settings change the numbers: with one NIC (`NCCL_IB_HCA==mlx5_0`) or two (`=mlx5_0,mlx5_4`) NCCL maps every
pair for P2P, but every rank then shares those NICs (single host: W4A4.md). Before a run: all 16 GPUs idle, every
port `ACTIVE` (`ibdev2netdev`, `cat /sys/class/infiniband/mlx5_*/ports/1/state`), and the other host's 172.16 address
of each port answering a ping from that port (`ping -I <netdev>`).

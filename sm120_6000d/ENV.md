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

# SM120 MegaMoE (`fp4_mega_moe_sm120`): environment variables

None is needed: with none set, the extension is built on first use into `~/.deep_gemm/sm120_mega_moe` and every kernel
choice comes from the call's arguments. None of them changes what the kernel computes or how fast it runs.

| Variable | Read by | Default | Effect |
|---|---|---|---|
| `DG_SM120_MEGA_MOE_BUILD_DIR` | `deep_gemm/mega/sm120_fp4.py` | `~/.deep_gemm/sm120_mega_moe` | where the torch extension is built and cached (one subdirectory per format) |
| `DG_SM120_FP4_MEGA_MOE_SO_MXFP4`, `..._NVFP4`, `..._W4A8` | `deep_gemm/mega/sm120_fp4.py` | unset | path of a prebuilt extension `.so` for that format (the build without the NIC code); when set, that build is not compiled |
| `TILE` | `tests/test_mega_moe_sm120_fp4.py bench` only | `0` | forces the tokens per work item (`8` or `16`) in the timing; `0` keeps the kernel's own choice |

The tokens per work item of a normal call are the `tile_tokens` argument of `fp4_mega_moe_sm120` (`0`: chosen by the
kernel), and the transport is the buffer's `transport` argument; neither is an environment variable.

NIC transport (`transport='auto'` / `'nic'`): NCCL reads its own variables. On a host whose NICs cannot reach each
other (on .20 they sit on different subnets, and the NCCL window registration fails), pin every rank to one NIC with
`NCCL_IB_HCA==mlx5_0`.

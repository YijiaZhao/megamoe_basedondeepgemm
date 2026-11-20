# SGLang 接入：DeepSeek-V4.1-Flash + SM120 MegaMoE

**SGLang 基础镜像：`lmsysorg/sglang:v0.5.22-cu130`**（`docker pull lmsysorg/sglang:v0.5.22-cu130`，不做任何修改）

一台 8×RTX 6000D，TP8 / DP8 attention / EP8，DSPARK 5 个 draft token，每个 rank BS16。
kernel 是本仓库的 `fp4_mega_moe_sm120`，用新格式 `w4a8_rm`（MXFP4 权重 × MXFP8 激活，直接读 checkpoint 的行主序权重，
与 SGLang 的 FlashInfer MoE 共用同一份 FP4 权重）。仓库里原来的文件不改：改过的 kernel 整份放在本目录 `kernel/` 下，SGLang 只从这里加载。

| `kernel/` 下的文件 | 相对仓库同名文件的改动 |
|---|---|
| `deep_gemm/include/deep_gemm/impls/sm120_fp4_mega_moe.cu` | 编译开关 `SM120_FP4_RM`（只给 `w4a8_rm`）：`item_gemm_rm` 读行主序权重（每 lane 读 16 字节，quad 内 4×4 转置成 `mma_w4a8` 的寄存器排布），L2 预取按行主序地址；`w13_up_first`：W13 是 up 行在前（FlashInfer 的顺序）。其他格式的代码不变；路由计数按 epoch 奇偶双缓冲（与仓库主 kernel 相同）。`w4a8_rm` 的 item：激活值按 K 分段用 `cp.async` 暂存到共享内存（item 的 8 个 warp 共用，只读一次，提前一段预取），每 rank ≥ 32 token 时一个 item 32 个 token、否则 16 个，item 里没有 token 的 8-token 子块跳过。可选的 kernel 内路由（`Route`）：传入 gate logits 时，CTA b 用一个 warp 算 token b 的路由（sqrtsoftplus、偏置 / 图像 token 的 `bias_vl`、top-k、归一化、缩放、EPLB 选副本 `t % n`，与 SGLang `moe_fused_gate` 的专家选择一致），写进 topk 缓冲区后其余 CTA 再开始分发 |
| `deep_gemm/mega/sm120_fp4.py` | `fp4_mega_moe_sm120` 的 `router_logits` 等关键字参数（kernel 内路由）；`FORMATS` 加 `w4a8_rm`；权重转换对 `w4a8_rm` 只重排 scale、权重原样返回；`w13_up_first` 参数；缓冲区参数 `slot`：扩展的状态是静态的，一个进程里每多一种 MoE 形状就占一个 slot，各加载一份扩展（`_s1` 后缀），主模型用 slot 0、DSPARK draft 模型用 slot 1 |
| `deep_gemm/include/deep_gemm/impls/sm120_fp4_mega_moe_gin.cu`、`deep_gemm/mega/sm120_fp4_transport.py` | 无改动（`sm120_fp4.py` 按相对路径找它们，一起拷过来）|
| `deep_gemm/mega/sm120_fp4_relayout.py`（新）、`tests/test_relayout.py` | `PACKED=1`（默认关，未做端到端测试）用：FlashInfer 行主序权重与 kernel 的 packed `w4a8` 布局互转 |
| `tests/bench_real_routing.py`（新）| 用 SGLang verify 步记录的真实路由回放 kernel，冷 L2，按最慢 rank 计时（文件头有命令）|
| `tests/bench_router_fused.py`（新）| kernel 内路由对 SGLang 路由（gate GEMM + `moe_fused_gate` + EPLB 映射）+ kernel：路由是否一致、输出差异、两者整层时间（文件头有命令）|
| `tests/test_mega_moe_sm120_fp4.py` | `w4a8_rm` 额外测 `up_first=True`：`torchrun --nproc_per_node=8 kernel/tests/test_mega_moe_sm120_fp4.py verify w4a8_rm`；形状 `dsv41` / `dsv41_draft`；`two_slots w4a8_rm`：主模型和 draft 两种形状在一个进程的 slot 0 / 1 上，直接跑和 CUDA graph 里各跑一遍 |

`kernel/` 来自本仓库 `39b07dc`（`sm120_fp4*` 四个文件和测试）加上表中改动。

## 前置条件

| 项 | 要求（测试时的值）|
|---|---|
| GPU | 一台 8×RTX 6000D 或 RTX PRO 6000（SM120），8 张卡空闲 |
| 驱动 | NVIDIA 驱动 580.95.05；Docker + NVIDIA Container Toolkit（`docker run --gpus all` 可用）|
| 主机内存 | engram 表放主机内存，测试机 2.2 TB |
| 模型 | `DeepSeek-V4.1-Flash-1017`（476 GB），放在 `$MODEL_DIR/DeepSeek-V4.1-Flash-1017` |
| 端口 | 30041 空闲（`PORT=` 可改）|
| 网卡（仅 `TRANSPORT=lsa_where_reachable`）| 8 个 mlx5 数据口 Up，主机加载 `nvidia_peermem`，见“网卡”一节 |

不需要编译 SGLang、不需要装 Python 包：镜像里已有 SGLang 0.5.22、FlashInfer、NCCL 2.30（带 Device API）和 nvcc。

## 一步步跑

```bash
git clone https://github.com/YijiaZhao/megamoe_basedondeepgemm.git
cd megamoe_basedondeepgemm/sm120_6000d/sglang
docker pull lmsysorg/sglang:v0.5.22-cu130
MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable EPLB=1 FUSED_ROUTER=1 bash launch.sh   # 最快的配置（CX8aaS 机型，EPLB，kernel 内路由）；容器名 dsv41_sglang
docker logs -f dsv41_sglang                      # 等到 "The server is fired up"（约 7 分钟，第一次含 kernel 编译）
curl -s localhost:30041/generate -H 'Content-Type: application/json' \
  -d '{"text":"1+1=","sampling_params":{"max_new_tokens":8,"temperature":0}}'
docker rm -f dsv41_sglang                        # 停掉
```

`EPLB=1` 用的放置文件是在本文的测试数据上统计的，只用来复现本文结果；实际部署要在自己的负载上重新统计，见“启动”一节的 EPLB 说明。

### 两台机器，EP16，主模型 MoE 全走 FlashInfer Cake（最快）

两台 8 卡机器（测试用 .6 + .1，CX8aaS），TP16 / DP16 attention / EP16。主模型的 MoE（prefill 和 decode，包括 CUDA graph）
全部走 FlashInfer 的 Cake SM120 MegaMoE，DSPARK draft 模型的 MoE 仍走本仓库 kernel。另外需要：

- FlashInfer main 的源码（含子模块），测试用 commit `82866b5`，两台机器同一路径（`FI_DIR`），启动时装进容器，JIT 缓存在 `FI_CACHE`；
- 两台机器之间 `bond0`（`IFNAME=` 可改）互通，node rank 0 的 29661 端口（`DIST_PORT=` 可改）对另一台开放；
- 模型在两台机器上同一路径。

```bash
FI=/path/flashinfer   # git clone --recursive https://github.com/flashinfer-ai/flashinfer.git && git -C flashinfer checkout 82866b5
E="MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable EPLB=1 CAKE=1 FI_DIR=$FI NNODES=2 MASTER=10.6.142.6"
# 两台同时起，node rank 0 先
env $E NODE_RANK=0 bash launch.sh --mem-fraction-static 0.66   # 在 10.6.142.6 上
env $E NODE_RANK=1 bash launch.sh --mem-fraction-static 0.66   # 在 10.6.142.1 上
# 等 node rank 0 的 curl localhost:30041/health 返回 200（约 10 分钟），然后在 node rank 0 上：
docker exec dsv41_sglang python3 -m sglang.test.few_shot_gsm8k --host 127.0.0.1 --port 30041 --num-questions 1319 --parallel 256
DP=16 OTHER_HOSTS=10.6.142.1 CLIENT_DIR=/path/nvidia_decode_client_20261111 DATA=/path/public_pg19_all.jsonl.gz bash bench_bs16.sh
```

- `EPLB=1` 在 `NNODES=2` 时用 `eplb_map_r16_ep16.pt`（`{'logical_count'}`，同一个 decode bench 上统计），384 个专家加 16 个副本，每卡 25 个；同样只是示例，实际使用要重新统计。
- `CAKE_MAX_TOKENS` 默认 2048，即 SGLang 每 rank 的 prefill 分块（`--chunked-prefill-size` = 2048 × 16）；设到 16384 时 EP8 上 Cake 的通信缓冲区分配（`ncclMemAlloc`）失败。
- `--mem-fraction-static 0.66`：EP16 时 Cake 和 NCCL 的缓冲区约占 14 GB 显存，0.8 / 0.76 会在 indexer 处 OOM；KV 池仍远大于测试所需。
- 多机时 engram 表每台机器一份（各机 local rank 0 建共享内存，本机 rank 分摊加载）。
- 测试时间：服务起来约 10 分钟，gsm8k 约 2 分钟，decode bench 约 25 分钟（预热占大部分）。

### 17–32 个 rank（例如四台机器 EP32）

`kernel/` 的默认构建最多 16 个 rank。group 超过 16 个 rank 时 `sm120_fp4.py` 自动换用 `_w32` 构建：同一份源码加
`-DSM120_FP4_MAX_WORLD=32 -DSM120_FP4_MAX_TILES=6400`，扩展名和缓存子目录带 `_w32` 后缀，16 个 rank 以内的构建不变。

| | 默认构建 | `_w32` 构建 |
|---|---|---|
| rank 数 | ≤ 16 | 17–32 |
| 每 rank 本地专家数（`kMaxEl`） | ≤ 256 | ≤ 128 |
| 工作项表（`kMaxTiles`，走网卡时要求 本地专家数 + rank 数 × 每 rank token 数 × top-k / 8 不超过它） | 2304 | 6400 |

- DeepSeek-V4.1 的 top-6 在 EP32、每卡 13 个专家（384 + 32 个副本）时，每 rank 的 token 上限是 266。SGLang 侧的上限检查
  （`mega_moe_sm120.py`）按所用构建的 `kMaxTiles` 计算（`sm120_fp4.max_tiles(world)`）。
- `launch.sh` 用 `NNODES=4` 起 EP32。本目录没有 EP32 的 EPLB 放置文件，`EPLB=1` 时要给 `EPLB_MAP=<文件> REDUNDANT=<副本数>`
  （放置与负载和卡数有关，要在同样的卡数上重新统计）。
- 仓库自带的测试直接可用：`torchrun --nnodes 4 --nproc-per-node 8 ... kernel/tests/test_mega_moe_sm120_fp4.py verify w4a8_rm all_gin`。

## SGLang 版本

公开镜像 `lmsysorg/sglang:v0.5.22-cu130`，SGLang commit `baa9befa381511cd4d56f83980951e5a45fffc6d`（0.5.22）。
镜像不改，启动容器时在 `/sgl-workspace/sglang` 里 `git apply sglang_v0.5.22_dsv41.patch`。

## 改了哪些代码（`sglang_v0.5.22_dsv41.patch`）

| 文件（`python/sglang/srt/` 下） | 改动 |
|---|---|
| `layers/attention/dsv4/v41_indexer/__init__.py` | 启动修复：`is_sm100_or_newer` 由 `>= 10` 改为 `== 10`，SM120 走 torch indexer（SM120 没有 SM100 的 DeepGEMM indexer kernel）。不用 MegaMoE 也需要 |
| `models/deepseek_v4.py` | `_run_moe_ffn_dp_sync` 开头：`use_sm120_mega_moe` 为真时直接 `forward_mega_moe`，跳过 DP gather / scatter（kernel 自己做 EP 分发和合并）|
| `layers/moe/mega_moe.py` | SM120 分支：`is_mega_moe_experts_ready` 认 SM120；`run_mega_routed_experts` 在 SM120 上调 `run_sm120_mega_routed`；`SGLANG_SM120_FUSED_ROUTER=1` 时算完 gate logits 不再调 `vision_topk` / `topk` 和 EPLB 映射，把 logits 交给 kernel 路由 |
| `layers/moe/mega_moe_sm120.py`（新文件）| 是否走 kernel（每个 rank 一次调用的 token 数 ≤ 128，所有 DP rank 统一判断；CUDA graph 按 capture 的批大小）、建缓冲区（每种 MoE 形状一个 slot：主模型 384 专家 top-6 是 slot 0，DSPARK draft 模型 128 专家 top-3 是 slot 1）、选传输方式、调 kernel |
| `layers/engram.py` | 多机：engram 主机表按机器各建一份（每台机器 local rank 0 建 memfd，本机其他 rank 打开它；行按本机 rank 分摊加载）。原来只有全局 rank 0 建，另一台机器打不开 |
| `layers/moe/mega_moe_sm120.py`（Cake 部分）| `SGLANG_SM120_CAKE_ALL=1`（`CAKE=1`）：主模型的 MoE 调用都走 FlashInfer `MoEEpLayer`；加载时 W13 按 Cake 要求原地交错 gate/up 行，scale 另存一份 Cake 布局；CUDA graph capture 时用 `stage_inputs` + `compute_staged`（各 rank 批大小一致，不做路由 all-gather、不同步主机）|
| `layers/quantization/mxfp4_flashinfer_cutlass_moe.py` | 加载权重时给 kernel 生成一份 UE8M0 scale；FP4 权重和 FlashInfer 共用一份 |
| `multimodal/dsv41/vl_routing.py` | EPLB：DSV4.1 的 router 走 `vision_topk`，上游只在 MegaMoE a2a 后端时把逻辑专家号换成物理专家号；`--moe-a2a-backend none` 时也换（不换则 gsm8k 掉到 0.02）|
| `models/deepseek_v2.py` | EPLB：冗余专家只加给主模型，DSPARK draft（128 专家）不加；`vision_topk` 传 `expert_location_dispatch_info` |
| `layers/moe/fused_moe_triton/layer.py` | EPLB：draft 模型的 MoE 加载权重时不用主模型的专家放置 |
| `../kernels/ops/attention/dsv4/fp4_indexer.py` | 新 Triton kernel `fp4_index_logits_grouped`：target verify 时同一请求的 6 行（5 个 draft token + 1）共用一次 K tile 的读取和 FP4 反量化，按实际长度跳过空 tile；结果与 `fp4_index_logits_decode` 比特级一致 |
| `layers/attention/dsv4/v41_indexer/scoring.py` | `SGLANG_SM120_INDEXER_GROUP` > 0 且是 verify 时，ratio-1/2 层的 decode 打分走上面的 kernel，不再生成 [行数, L] 的 slot 表，只对选中的 top-k 位置查 `req_to_token` |
| `layers/quantization/fp8_utils.py` | `SGLANG_SM120_MXFP8_B12X_MIN_N` > 0 时（SM120，`--fp8-gemm-backend flashinfer_cutlass`）：按 MXFP8 跑的 FP8 线性层里，输出行数 ≥ 该值且 K % 128 == 0 的走 FlashInfer 的 SM120 `b12x` kernel，其余仍走 CUTLASS；都用默认 tactic，权重和 scale 不变 |

router（gate + topk）用 SGLang 自己的，结果传给 kernel。主模型和 DSPARK draft 模型的 routed experts 都走 kernel，draft 也不再做 DP gather / scatter。
不走 kernel 的部分：prefill / chunked prefill（任一 rank 超过 128 token）、shared expert、attention。

## 启动

```bash
MODEL_DIR=/raid/models bash launch.sh                  # MegaMoE，卡间全走 PCIe P2P
MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable bash launch.sh   # MegaMoE，NCCL 映射不了 P2P 的卡间走网卡（CX8aaS）
MODEL_DIR=/raid/models MEGA=0 bash launch.sh           # 基线：原版 FlashInfer MoE（只带启动修复）
MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable EPLB=1 bash launch.sh   # 再开 EPLB（8 个冗余专家），最快；MEGA=0 也能加
MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable EPLB=1 DENSE_OPT=0 bash launch.sh   # FP8 GEMM 和 indexer 用 SGLang 自带的
```

**EPLB**（`EPLB=1`）：`--init-expert-location eplb_map_r8_ep8.pt --ep-num-redundant-experts 8`。主模型 384 个专家加 8 个副本，
每卡 49 个，热门专家复制、按负载摊到 8 张卡；draft 模型不动。`eplb_map_r8_ep8.pt` 是 `{'physical_to_logical_map': [40 层, 392]}`：
在同一个 decode bench（pg19，每 rank BS16、输入 100K）上统计主模型每层每个专家的 token 数，用 SGLang 的
`eplb_algorithms.rebalance_experts`（`auto`，单机，8 卡）算出来。换负载或卡数要另算一份，用 `EPLB_MAP=<文件> REDUNDANT=<副本数>` 传入。

本仓库关注的是 kernel 性能，不负责专家怎么统计、怎么放置；kernel 只按拿到的物理专家号计算，放置用什么都不影响代码。
`eplb_map_r8_ep8.pt` 和 `eplb_map_r16_ep16.pt` 只是复现本文结果用的示例：它们是用待测的这份数据统计出来的，结果表里开 EPLB 的数字
都是这份数据配它自己的放置测得的。真正使用时必须在自己的负载和卡数上重新统计（离线统计或 SGLang 在线统计都行），不能直接拿这两份文件用。

**FP8 GEMM 与 indexer**（`DENSE_OPT`，`MEGA=1` 时默认 `1`，`MEGA=0` 时默认 `0`）：
- `--fp8-gemm-backend flashinfer_cutlass`：模型的 FP8 线性层（attention 投影、共享专家等，32×32 块、ue8m0 scale）按 MXFP8 原样交给 FlashInfer CUTLASS，权重不变；默认会走未调优的 Triton。
- `SGLANG_SM120_MXFP8_B12X_MIN_N=4096`：上面这些层里输出行数 ≥ 4096 的（`wq_b`、`wo_b`、indexer `wq_b`、共享专家）改走 FlashInfer 的 SM120 `b12x` kernel，
  `wq_a + wkv`（1792 行）留在 CUTLASS。单测（每 rank 96 token，冷 L2）每层合计 302 → 245 µs。
- `SGLANG_SM120_INDEXER_GROUP=6`：verify 的 indexer 打分走分组 kernel（见上表）。正确性和单次耗时：`test_indexer_grouped.py`（文件头有命令）。

`DENSE_OPT=0` 回到 SGLang 自带的 Triton FP8 GEMM 和原 indexer。

`launch.sh` 把 `kernel/` 挂成 kernel 源码（`/opt/sm120_megamoe`），kernel 第一次调用时按 `sm_120a` 编译，缓存在 `BUILD_DIR`。
服务端参数和下面性能表用的一致，写在脚本里；多余参数追加到命令末尾。

环境变量（MegaMoE 时）：

| 变量 | 值 | 作用 |
|---|---|---|
| `SGLANG_SM120_MEGA_MOE` | `1` | 打开 kernel |
| `SGLANG_SHARED_EXPERT_TP1` | `1` | 必需：shared expert 在本 rank 完整计算（kernel 只算本 rank 的 token）|
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` | `128` | 每 rank token 上限，也是缓冲区大小 |
| `SGLANG_SM120_FUSED_ROUTER` | `0` / `1`（`FUSED_ROUTER=1`）| kernel 内路由（不用于哈希层、Cake、fused shared expert）|
| `SGLANG_SM120_MEGA_MOE_TRANSPORT` | `p2p` / `lsa_where_reachable` | 传输方式，见下节；其余可选值见 `kernel/deep_gemm/mega/sm120_fp4.py` 的 `TRANSPORTS` |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` | `1` | 上游开关（基线也开）：engram 表放主机内存，否则 `mem-fraction-static 0.8` 放不下 KV |
| `SGLANG_SM120_INDEXER_GROUP` | `6`（`DENSE_OPT=1`）| verify 每个请求的行数（draft token + 1），分组 indexer；`0` 为原 indexer |
| `SGLANG_SM120_MXFP8_B12X_MIN_N` | `4096`（`DENSE_OPT=1`）| 输出行数 ≥ 该值的 FP8 线性层走 `b12x`；`0` 全走 CUTLASS |

## 网卡（`TRANSPORT=lsa_where_reachable`）

两卡一对挂在 ConnectX-8 的 PCIe 交换芯片下的机型（CX8aaS，例如 .1/.3/.6），一张卡同时往多个 CPU root port 后的卡写 P2P 会塌到
1–2 GB/s，全 P2P 时 decode 比基线还慢约 4 倍。用 `lsa_where_reachable`：NCCL 能映射 P2P 的卡对（8 个网卡都给 NCCL 时即同一 CPU 插槽的卡）
走 P2P，其余走 NCCL GIN（网卡）。`auto` 的启动探测在这类机器上会选 `all_gin`，kernel 实测更慢（见 `../DESIGN.md`）。需要：

- `NCCL_IB_HCA` 列出 8 个数据口（默认 `mlx5_0`…`mlx5_7`；不要用管理网的 `mlx5_bond_0`），`ibdev2netdev` 确认都是 Up；
- 容器 `--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1`；
- `NCCL_CUMEM_ENABLE=1`（NCCL Device API 需要；SGLang 默认设成 0，否则报 `the NCCL communicator does not support the Device API`）；
- 主机加载 `nvidia_peermem`。

`launch.sh` 在 `TRANSPORT` 不是 `p2p` 时自动加上前三项。Broadcom 交换芯片的机型（例如 .20）用默认 `p2p`。

启动日志里每个 rank 有两种形状各一行 `SM120 MegaMoE buffer: EP8, 384 experts / top-6 (slot 0), transport ...`（draft 是
`128 experts / top-3 (slot 1)`），以及 `SM120 MegaMoE: first call on the mega kernel path (384 experts, 96 local tokens)`。

## 测速与精度

服务起来后（日志出现 `The server is fired up`，或 `curl localhost:30041/health` 返回 200），在同一台机器上：

```bash
# 性能：decode SSE bench，BS16 每 rank（客户端包和 pg19 数据不在本仓库）
CLIENT_DIR=/path/nvidia_decode_client_20261111 DATA=/path/public_pg19_all.jsonl.gz bash bench_bs16.sh
# 精度：gsm8k 全部 1319 题
docker exec dsv41_sglang python3 -m sglang.test.few_shot_gsm8k --host 127.0.0.1 --port 30041 --num-questions 1319 --parallel 128
```

`bench_bs16.sh` 先给 8 个 rank 各发一条短请求（rank 第一次 decode 后才导出客户端要读的
`sglang:realtime_tokens_total{mode="decode"}`），再用镜像里的 Python 在 CPU 容器里跑客户端 `run.sh`：mode 4、DP8、
每 rank 平均输入 100K（±1%）、输出 1100、BS16、稳态窗口 1 s。预热约 28 分钟，日志写到客户端目录的 `bench_bs16.log`。
测完换 `MEGA=0` 起基线再测一遍做对比（先 `docker rm -f dsv41_sglang`）。

## 结果

两台机器 EP16（.6 + .1，`lsa_where_reachable` + 8 网卡，EPLB 16 个冗余专家，`DENSE_OPT=1`）：每 rank BS16，全局 256 条，
每 rank 平均输入 100K，其余与下表客户端设置相同，缓存命中 99.87%。两列同一天、同两台机器、同一测法。

| | EP16，主模型 MoE 走本仓库 kernel | EP16，主模型 MoE 全走 Cake（最快）|
|---|---|---|
| TPOT P50 / 平均 / P95 (ms) | 31.7 / 31.6 / 42.3 | **30.9 / 31.0 / 40.4** |
| 全部输出 tok/s（16 卡）| 8461 | **8791** |
| 接受长度 / draft 接受率 | 2.42 / 28.46% | 2.39 / 27.77% |

Cake 这一列 gsm8k（1319 题，256 并发）0.901。

单机 EP8：

客户端：decode SSE bench（mode 4，pg19 真实文本，每 rank 平均输入 100K，输出 1100，BS16 每 rank，全局 128 条，稳态窗口 1 s，
严格检查全部通过，缓存命中 99.87%）。同一行两列在同一机型上测。

| 机器 | 传输 | 基线 TPOT P50 / 平均 / P95 (ms) | MegaMoE TPOT P50 / 平均 / P95 (ms) | TPOT 平均 | 全机输出 tok/s（基线 → MegaMoE）|
|---|---|---|---|---|---|
| .6（CX8aaS）| `lsa_where_reachable` + 8 网卡 | 58.5 / 57.6 / 75.5 | 48.2 / 48.1 / 65.1 | −16.5% | 2332 → 2745 |
| .6（CX8aaS），EPLB 8 个冗余专家，`DENSE_OPT=0` | `lsa_where_reachable` + 8 网卡 | 55.6 / 56.1 / 82.6 | 44.3 / 45.6 / 60.6 | −18.7% | 2589 → 3032 |
| .1（CX8aaS），EPLB 8 个冗余专家，`DENSE_OPT=1` 不含 `b12x`² | `lsa_where_reachable` + 8 网卡 | 55.6 / 56.1 / 82.6 | 36.2 / 35.5 / 48.2 | −36.7% | 2589 → 3826 |
| .6（CX8aaS），EPLB 8 个冗余专家，`DENSE_OPT=1`（默认），`FUSED_ROUTER=1`³ | `lsa_where_reachable` + 8 网卡 | 55.6 / 56.1 / 82.6 | **28.6 / 29.1 / 36.9** | −48.1% | 2589 → **4621** |
| .20（Broadcom）¹ | `p2p` | — / 57.3 / 80.2 | — / 48.8 / 64.7 | −14.8% | 2217 → 2818 |

¹ .20 是上一版测的：draft 模型的 MoE 还走 FlashInfer，kernel 是 `c9c8835`。
² 当时 FP8 线性层全走 CUTLASS（`SGLANG_SM120_MXFP8_B12X_MIN_N=0`）。基线列沿用 .6 的 EPLB 基线（同机型，未在 .1 上重测）。
每步间隔（SSE gap 平均）72.4 ms，`DENSE_OPT=0` 时 96.0 ms；按 10 步 profile，FP8 GEMM 每步 30.7 → 10.4 ms，indexer 打分加 slot 表约 13.4 → 7.3 ms。
³ 当前 kernel（激活值暂存到共享内存，每 rank ≥ 32 token 用 32-token item），kernel 内路由：每步间隔（SSE gap）P50 63.9 ms，
接受长度 2.39、draft 接受率 27.9%。不开 kernel 内路由（`FUSED_ROUTER=0`）31.5 / 30.3 / 40.9 ms、4430 tok/s、64.9 ms。上一版 kernel 同配置 33.1 / 32.7 / 43.7 ms、3989 tok/s、每步间隔 P50 69.5 ms；FlashInfer Cake（主模型
MoE 全走 Cake，`--mem-fraction-static 0.76`）同配置 34.9 / 33.6 / 41.4 ms、4085 tok/s、70.5 ms。
按 10 步 profile，FP8 GEMM 每步 10.8 → 7.5 ms（`b12x`），整步 72.9 → 69.6 ms，其余部分不变。

精度：gsm8k（1319 题，128 并发）基线 0.907，MegaMoE（主模型 + draft，.6）0.900；开 EPLB：基线 0.917，MegaMoE 0.918；
EPLB + `DENSE_OPT=1`：不含 `b12x` 0.909，含 `b12x` 0.903，当前 kernel 0.908，加 kernel 内路由 0.909。
逐层对比（同一输入，kernel 对 FlashInfer）：相对误差中位数 1.4%，余弦相似度 ≥ 0.9997。

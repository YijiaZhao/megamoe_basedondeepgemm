# SGLang 接入：DeepSeek-V4.1-Flash + SM120 MegaMoE

**SGLang 基础镜像：`lmsysorg/sglang:v0.5.22-cu130`**（`docker pull lmsysorg/sglang:v0.5.22-cu130`，不做任何修改）

一台 8×RTX 6000D，TP8 / DP8 attention / EP8，DSPARK 5 个 draft token，每个 rank BS16。
kernel 是本仓库的 `fp4_mega_moe_sm120`，用新格式 `w4a8_rm`（MXFP4 权重 × MXFP8 激活，直接读 checkpoint 的行主序权重，
与 SGLang 的 FlashInfer MoE 共用同一份 FP4 权重）。仓库里原来的文件不改：改过的 kernel 整份放在本目录 `kernel/` 下，SGLang 只从这里加载。

| `kernel/` 下的文件 | 相对仓库同名文件的改动 |
|---|---|
| `deep_gemm/include/deep_gemm/impls/sm120_fp4_mega_moe.cu` | 编译开关 `SM120_FP4_RM`（只给 `w4a8_rm`）：`item_gemm_rm` 读行主序权重（每 lane 读 16 字节，quad 内 4×4 转置成 `mma_w4a8` 的寄存器排布），L2 预取按行主序地址；`w13_up_first`：W13 是 up 行在前（FlashInfer 的顺序）。其他格式的代码不变；路由计数按 epoch 奇偶双缓冲（与仓库主 kernel 相同） |
| `deep_gemm/mega/sm120_fp4.py` | `FORMATS` 加 `w4a8_rm`；权重转换对 `w4a8_rm` 只重排 scale、权重原样返回；`w13_up_first` 参数；缓冲区参数 `slot`：扩展的状态是静态的，一个进程里每多一种 MoE 形状就占一个 slot，各加载一份扩展（`_s1` 后缀），主模型用 slot 0、DSPARK draft 模型用 slot 1 |
| `deep_gemm/include/deep_gemm/impls/sm120_fp4_mega_moe_gin.cu`、`deep_gemm/mega/sm120_fp4_transport.py` | 无改动（`sm120_fp4.py` 按相对路径找它们，一起拷过来）|
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
MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable EPLB=1 bash launch.sh   # 最快的配置（CX8aaS 机型，EPLB）；容器名 dsv41_sglang
docker logs -f dsv41_sglang                      # 等到 "The server is fired up"（约 7 分钟，第一次含 kernel 编译）
curl -s localhost:30041/generate -H 'Content-Type: application/json' \
  -d '{"text":"1+1=","sampling_params":{"max_new_tokens":8,"temperature":0}}'
docker rm -f dsv41_sglang                        # 停掉
```

## SGLang 版本

公开镜像 `lmsysorg/sglang:v0.5.22-cu130`，SGLang commit `baa9befa381511cd4d56f83980951e5a45fffc6d`（0.5.22）。
镜像不改，启动容器时在 `/sgl-workspace/sglang` 里 `git apply sglang_v0.5.22_dsv41.patch`。

## 改了哪些代码（`sglang_v0.5.22_dsv41.patch`）

| 文件（`python/sglang/srt/` 下） | 改动 |
|---|---|
| `layers/attention/dsv4/v41_indexer/__init__.py` | 启动修复：`is_sm100_or_newer` 由 `>= 10` 改为 `== 10`，SM120 走 torch indexer（SM120 没有 SM100 的 DeepGEMM indexer kernel）。不用 MegaMoE 也需要 |
| `models/deepseek_v4.py` | `_run_moe_ffn_dp_sync` 开头：`use_sm120_mega_moe` 为真时直接 `forward_mega_moe`，跳过 DP gather / scatter（kernel 自己做 EP 分发和合并）|
| `layers/moe/mega_moe.py` | SM120 分支：`is_mega_moe_experts_ready` 认 SM120；`run_mega_routed_experts` 在 SM120 上调 `run_sm120_mega_routed` |
| `layers/moe/mega_moe_sm120.py`（新文件）| 是否走 kernel（每个 rank 一次调用的 token 数 ≤ 128，所有 DP rank 统一判断；CUDA graph 按 capture 的批大小）、建缓冲区（每种 MoE 形状一个 slot：主模型 384 专家 top-6 是 slot 0，DSPARK draft 模型 128 专家 top-3 是 slot 1）、选传输方式、调 kernel |
| `layers/quantization/mxfp4_flashinfer_cutlass_moe.py` | 加载权重时给 kernel 生成一份 UE8M0 scale；FP4 权重和 FlashInfer 共用一份 |
| `multimodal/dsv41/vl_routing.py` | EPLB：DSV4.1 的 router 走 `vision_topk`，上游只在 MegaMoE a2a 后端时把逻辑专家号换成物理专家号；`--moe-a2a-backend none` 时也换（不换则 gsm8k 掉到 0.02）|
| `models/deepseek_v2.py` | EPLB：冗余专家只加给主模型，DSPARK draft（128 专家）不加；`vision_topk` 传 `expert_location_dispatch_info` |
| `layers/moe/fused_moe_triton/layer.py` | EPLB：draft 模型的 MoE 加载权重时不用主模型的专家放置 |

router（gate + topk）用 SGLang 自己的，结果传给 kernel。主模型和 DSPARK draft 模型的 routed experts 都走 kernel，draft 也不再做 DP gather / scatter。
不走 kernel 的部分：prefill / chunked prefill（任一 rank 超过 128 token）、shared expert、attention。

## 启动

```bash
MODEL_DIR=/raid/models bash launch.sh                  # MegaMoE，卡间全走 PCIe P2P
MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable bash launch.sh   # MegaMoE，NCCL 映射不了 P2P 的卡间走网卡（CX8aaS）
MODEL_DIR=/raid/models MEGA=0 bash launch.sh           # 基线：原版 FlashInfer MoE（只带启动修复）
MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable EPLB=1 bash launch.sh   # 再开 EPLB（8 个冗余专家），最快；MEGA=0 也能加
```

**EPLB**（`EPLB=1`）：`--init-expert-location eplb_map_r8_ep8.pt --ep-num-redundant-experts 8`。主模型 384 个专家加 8 个副本，
每卡 49 个，热门专家复制、按负载摊到 8 张卡；draft 模型不动。`eplb_map_r8_ep8.pt` 是 `{'physical_to_logical_map': [40 层, 392]}`：
在同一个 decode bench（pg19，每 rank BS16、输入 100K）上统计主模型每层每个专家的 token 数，用 SGLang 的
`eplb_algorithms.rebalance_experts`（`auto`，单机，8 卡）算出来。换负载或卡数要另算一份，用 `EPLB_MAP=<文件> REDUNDANT=<副本数>` 传入。

`launch.sh` 把 `kernel/` 挂成 kernel 源码（`/opt/sm120_megamoe`），kernel 第一次调用时按 `sm_120a` 编译，缓存在 `BUILD_DIR`。
服务端参数和下面性能表用的一致，写在脚本里；多余参数追加到命令末尾。

环境变量（MegaMoE 时）：

| 变量 | 值 | 作用 |
|---|---|---|
| `SGLANG_SM120_MEGA_MOE` | `1` | 打开 kernel |
| `SGLANG_SHARED_EXPERT_TP1` | `1` | 必需：shared expert 在本 rank 完整计算（kernel 只算本 rank 的 token）|
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` | `128` | 每 rank token 上限，也是缓冲区大小 |
| `SGLANG_SM120_MEGA_MOE_TRANSPORT` | `p2p` / `lsa_where_reachable` | 传输方式，见下节；其余可选值见 `kernel/deep_gemm/mega/sm120_fp4.py` 的 `TRANSPORTS` |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` | `1` | 上游开关（基线也开）：engram 表放主机内存，否则 `mem-fraction-static 0.8` 放不下 KV |

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

客户端：decode SSE bench（mode 4，pg19 真实文本，每 rank 平均输入 100K，输出 1100，BS16 每 rank，全局 128 条，稳态窗口 1 s，
严格检查全部通过，缓存命中 99.87%）。同一行两列在同一机型上测。

| 机器 | 传输 | 基线 TPOT P50 / 平均 / P95 (ms) | MegaMoE TPOT P50 / 平均 / P95 (ms) | TPOT 平均 | 全机输出 tok/s（基线 → MegaMoE）|
|---|---|---|---|---|---|
| .6（CX8aaS）| `lsa_where_reachable` + 8 网卡 | 58.5 / 57.6 / 75.5 | 48.2 / 48.1 / 65.1 | −16.5% | 2332 → 2745 |
| .6（CX8aaS），EPLB 8 个冗余专家 | `lsa_where_reachable` + 8 网卡 | 55.6 / 56.1 / 82.6 | **44.3 / 45.6 / 60.6** | −18.7% | 2589 → **3032** |
| .20（Broadcom）¹ | `p2p` | — / 57.3 / 80.2 | — / 48.8 / 64.7 | −14.8% | 2217 → 2818 |

¹ .20 是上一版测的：draft 模型的 MoE 还走 FlashInfer，kernel 是 `c9c8835`。

精度：gsm8k（1319 题，128 并发）基线 0.907，MegaMoE（主模型 + draft，.6）0.900；开 EPLB：基线 0.917，MegaMoE 0.918。
逐层对比（同一输入，kernel 对 FlashInfer）：相对误差中位数 1.4%，余弦相似度 ≥ 0.9997。

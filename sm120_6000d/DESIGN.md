# SM120 MegaMoE（`fp4_mega_moe_sm120`）设计说明

读者：要改这个 kernel 的人或 agent。先读第 1–3 节，改代码前读第 5 节（不变量）。
性能表、测法、正确性见 [W4A4.md](W4A4.md)；测试机拓扑见 README 的
[SM120 test host](../README.md#sm120-test-host-8x-rtx-6000d)；环境变量见 [ENV.md](ENV.md)。

## 1. 一句话

RTX 6000D / RTX PRO 6000（SM120）上的 EP MoE 层：每个 rank 启动一个常驻 kernel，在 kernel 内依次完成路由计数交换、token 分发、
FC1+SwiGLU、FC2、部分和发回、combine。rank 之间没有 barrier，只靠“数据写完再写 flag（或网卡信号）”同步。
权重 FP4；激活 FP4（MXFP4 / NVFP4）或 FP8（W4A8）；卡间走 PCIe P2P，跨机走网卡（NCCL GIN）。目标区间是 decode：每卡 1–128 个 token。

## 2. 硬件约束（所有设计的出发点）

| 约束 | 后果 |
|---|---|
| 卡间只有 PCIe（单对约 50 GB/s），没有 NVLink | 跨卡次数和跨卡轮询都很贵：先发小的计数，flag 只由一个 CTA 轮询，部分和要边算边发 |
| 张量核只有 warp 级 `mma.sync`，没有 wgmma / tcgen05 | 操作数在寄存器里：权重离线排成 fragment 顺序，直接 `ld.global` 进寄存器；N 固定为 8 |
| decode 计算量极小，时间大头是从显存读专家权重 | 一份权重尽量喂更多 token（双 N8）；读和算重叠（双缓冲）；提前把权重拉进 L2 |

## 3. 代码地图

| 文件 | 内容 |
|---|---|
| `deep_gemm/include/deep_gemm/impls/sm120_fp4_mega_moe.cu` | kernel 和 torch 扩展。一个格式编一份（`SM120_FP4_NV`、`SM120_FP4_A8`），网卡版另编一份（`SM120_FP4_GIN`） |
| `deep_gemm/include/deep_gemm/impls/sm120_fp4_mega_moe_gin.cu` | 网卡传输的 host 端会话：NCCL 对称窗口、对端指针、GIN 设备通信器 |
| `deep_gemm/mega/sm120_fp4.py` | Python API：`Sm120Fp4MegaMoeBuffer`、`fp4_mega_moe_sm120`、`transform_fp4_weights_for_mega_moe_sm120`、`nvfp4_global_scales` |
| `tests/test_mega_moe_sm120_fp4.py` | `verify`（对齐量化的参考实现）和 `bench`（README 形状、冷 L2） |

kernel 文件里的位置（`sm120_fp4_mega_moe.cu`）：

| 段落 | 入口 |
|---|---|
| 常量、计数器下标、信号编号 | 文件开头的 `constexpr`（`kCtr*`、`kSig*`、`kMaxWorld=16`、`kMaxTiles`） |
| 对端地址 / 网卡参数 / kernel 参数 | `struct Peers`、`struct Gin`、`struct Args` |
| 跨卡等待 | `wait_ge`、`gin_wait`、`wait_peers` |
| GEMM 内循环 | `item_gemm<NT>` |
| 激活量化 | `quant_fp4`、`store_q` |
| 专家表 / 工作队列 | `struct Map`、`token_src`、`item_decode`、`first_item` |
| kernel 主体 | `sm120_fp4_mega_moe_kernel<NT>`，按注释 `// ----------------` 分段 |
| host：显存布局、初始化、启动 | `layout`、`setup_local`、`init` / `open_peers`（P2P）、`window_bytes` / `init_window`（网卡）、`run` |

## 4. 一次调用的完整流程

grid = 每 SM 两个 CTA × SM 数（6000D：312 个 CTA）。每个 CTA 256 线程（8 个 warp）。

| 步骤 | 谁做 | 做什么 | 同步点 |
|---|---|---|---|
| 1 路由计数 | CTA `ndisp-1`（一个） | 统计本 rank 每个 (目标 rank, 专家) 的 token 数，写到各目标的 route 区 | 每个目标写一次 route flag |
| 2 t=0 预取 | 不做 dispatch 的 CTA | 把本 rank 已知会用到的本地专家的 W13 拉进 L2（最多 64 MB，本地专家 ≤64） | 无 |
| 3 dispatch | 前 T 个 CTA，一个 CTA 一个 token | 量化一次，把同一份结果推到 TOPK 个目标 pool 行 | 最后一个结束的 CTA 给每个目标写一次 recv flag |
| 4 建专家表 | 所有 CTA | 等 route flag，读计数，每个 CTA 各自算出一样的 `Map`（每专家 token 数、tile 前缀、lag） | 等 route flag |
| 5 首个 item 预取 | 所有 CTA | 把自己第一个 item 的权重拉进 L2（tile 数 >8 时才做） | 无 |
| 6 等数据行 | 所有 CTA | 等所有源 rank 的 recv flag | 等 recv flag |
| 7 工作队列 | 所有 CTA | FC1 item / FC2 item 交错执行，见第 6.5 节 | FC2 等该专家 FC1 计数（本卡） |
| 8 发回 | 做 FC2 的 CTA | bf16 部分和直接写到源 rank 的 `comb[t][k]` | 最后一个 FC2 item 给每个源 rank 写 comb flag |
| 9 combine | 所有 CTA | 等 comb flag，`out[t] = Σ_k w[t][k]·partial[t][k]`（fp32 累加） | 等 comb flag |
| 10 收尾 | 最后退出的 CTA | epoch 加一，清本次计数器 | 无 |

## 5. 不变量（改代码前必须知道）

1. **slot 编号约定**：专家 e 收到的 token 按“源 rank，然后源 token 顺序”编号。发送方的 slot = 本 rank 内排在这个 token 之前、
   也选了 e 的 token 数；接收方用 `token_src`，按计数前缀反推同一个编号。两边必须一致，否则 FC1 读错行。改发送顺序就要同时改 `token_src`。
2. **pool 行地址**：`(本地专家 × W + 源 rank) × TCAP + slot`，每行 `pool_rx(H) = H/kXB + H/kSfVec + 16` 字节：数值、scale，
   最后 16 字节里的 int = `源 token × 8 + k`。FC2 发回 P2P 源时靠这个 int 找 `comb[t][k]`。
3. **flag 的写法**：先写完数据，再 `__threadfence_system()`，再普通 store 写 flag（不用 release store，实测每次多约 1 µs）。
   接收方用 `ld.acquire.sys` 读。
4. **epoch**：flag 的值是调用序号，单调递增、从不清零，等待条件是 `flag >= epoch`。`ctr[kCtrEpoch]` 由最后退出的 CTA 更新。
5. **每次调用要复位的计数器**（最后退出的 CTA 负责）：`kCtrDispatch`、`kCtrQueue`、`kCtrL2Done`、每专家的 `kCtrL1`，
   网卡版另加每 tile 的 `kCtrTile`。新增计数器必须加进这里。
6. **网卡信号是累计值**：信号从不清零。接收方的期望值 = `gin_cum` 里的历史累计 + 本次应到数；本次应到数在退出时加进 `gin_cum`。
   发送方 put 时加多少，接收方就必须按同样口径算期望值。
7. **超时**：所有自旋等待最多 30 s（`kSpinNs`），超时写 `kCtrErr` 并退出，不会挂死 GPU；`buffer.error()` 读出是哪一步。
8. **每进程每格式一个 buffer**：`_KEY` 缓存了第一次的配置，形状不同会 assert。buffer 必须在 CUDA graph capture 之前创建。

## 6. 设计点（做什么、为什么、在哪）

### 6.1 先发计数，再发数据

- **做什么**：kernel 一开始由一个 CTA 发每个目标 rank 每个专家的 token 数（每个目标 EL 个 int），数据行由其它 CTA 同时发。
- **为什么**：计数只有几百字节，比数据行早到。接收方据此在数据到齐前就建好专家表、tile 划分和工作队列，并能提前预取首个 item 的权重
  （6.6）。slot 约定（不变量 1）让双方各自算出同一编号，不需要传 token 索引表。
- **在哪**：kernel 段 `routing exchange first`；接收侧段 `per-expert map`。

### 6.2 一个 token 只量化一次

- **做什么**：一个 CTA 负责一个 token。整块 CTA 读一次 bf16、量化一次（每组算 scale），结果放 shared memory，
  再用全部线程以 16 B store 推到 TOPK 个目标行。
- **为什么**：同一个 token 发给 8 个专家，数据完全一样。量化 8 次就是 8 倍的读和计算；从 shared memory 推送，8 份写也能同时发出。
- **hidden 很大时**：按 `kDispGroups` 个 scale 组分块，量化一块推一块，任何 H 都放得下。
- **注意**：这不是去重。发往同一 rank 的两个专家仍各写一行（见第 8 节）。
- **在哪**：段 `dispatch`。

### 6.3 swap-AB 与双 N8（每个 item 8 或 16 个 token）

- **MMA 形状固定**：FP4 是 `m16n8k64`，W4A8 是 `m16n8k32`，N 只能是 8，没有 n16。
- **swap-AB**：M=16 放权重行，N=8 放 token。decode 时每个专家常只有 1–2 个 token，放在 N=8 一侧浪费最小。
- **item**：一个专家的 128 行权重（8 warp × 16 行）乘 K 全长。
- **NT=1（单 N8）**：一个 item 8 个 token，每个权重 fragment 做 1 条 MMA。
- **NT=2（双 N8）**：一个 item 16 个 token，同一个权重 fragment 读进寄存器后连做 2 条 MMA（token 0–7、8–15）。
  权重读一次服务 16 个 token。时间大头是读权重，所以每个 token 摊到的权重读取减半。计算量不变。
- **选择规则**（`run()`）：`T × TOPK / EL > 8` 用 NT=2，否则 NT=1；`tile_tokens=8/16` 可强制指定。
  不超过 8 个 token 时 NT=2 的第二条 MMA 全是空 token，白做。

### 6.4 GEMM 内循环：双缓冲只给 NT=1

- **NT=1**：每个 fragment 只做 1 条 MMA，计算短，盖不住读权重的延迟。所以开两组寄存器，算这一步时下一步的权重和 scale 已经在读。
- **NT=2**：每个 fragment 做 2 条 MMA，`#pragma unroll 2` 展开两次就够盖住延迟。再开一组缓冲只会多占寄存器。
- **`__launch_bounds__(256, 2)`**：每 SM 两个 CTA，每线程 128 个寄存器。展开后有足够的读请求同时在飞。
- **在哪**：`item_gemm<NT>`，`kDb = NT == 1`。

### 6.5 工作队列：FC1 和 FC2 交错（lag）

- **item 种类**：FC1 item = 一个 tile（一个专家的 8 或 16 个 token）× W13 的 128 行；FC2 item = 一个 tile × W2 的 128 行。
- **顺序**（`item_decode`）：先做前 `lag` 个 tile 的 FC1；然后每做 tile t+lag 的 FC1，接着做 tile t 的 FC2；最后补完剩下的 FC2。
- **为什么**：FC2 的结果要经 PCIe 写回源 rank。如果全堆在最后，尾部会被 PCIe 卡住；交错后部分和在读权重的整个过程中陆续发出。
- **lag**：第一波 CTA 覆盖的 tile 数（grid ÷ 每 tile 的 FC1 item 数）+ 单个专家最多的 tile 数。领到 tile t 的 FC2 时，它的 FC1 基本已完成。
  FC2 仍会自旋等该专家的 `kCtrL1` 计数（本卡，便宜）。
- **第一个 item 静态分配**：CTA b 的第一个 item 是 `grid-1-b`，之后从原子队列 `kCtrQueue` 领。前面的 CTA 在做 dispatch，最晚入队，
  所以队头留给后面的 CTA。

### 6.6 冷 L2 下读权重的三招

测法是冷 L2（每次调用前写 256 MB 其它数据），权重每次都从显存读。

1. **流式读**：权重用 `__ldcs`（evict-first）读，不占 L2，给 pool 和中间结果留位置。
2. **t=0 预取**：不做 dispatch 的 CTA 一开始就把本 rank 已知会用到的本地专家的 W13 拉进 L2（`kPf0Bytes = 64 MB`，本地专家 ≤64 时）。
   这时数据行还在路上，显存带宽空着。
3. **首个 item 预取**：专家表建好后，每个 CTA 把自己静态第一个 item 的权重拉进 L2，再去等数据行。tile 数 ≤8 时不做，那时它只会推迟等待。

### 6.7 权重离线排成 MMA fragment 顺序（本 kernel 专用格式）

- **做什么**：加载时调用一次 `transform_fp4_weights_for_mega_moe_sm120`，把 checkpoint 排布重排成：每个 lane 一次 16 B 读，
  读到的正好是它在 `mma.sync` 里的 A 操作数。scale 也排成 MMA 要的字节位置。
- **W13 交错**：每 8 行 gate 接 8 行 up，SwiGLU 的一对值落在同一线程的累加器里，不用跨线程交换。
- **后果**：内循环里没有 shared memory 和 shuffle。但重排后的权重只有这个 kernel 能直接用，其它 kernel（Marlin、CUTLASS 等）
  有各自的排布，不能共用。

### 6.8 单 CTA 轮询跨卡 flag

- **flag 是什么**：接收方显存里的一个 int，由发送方在数据全部写完后写入，值为 epoch（不变量 3、4）。每组 flag 每个对端一个：
  route flag（计数到了）、recv flag（数据行到了）、comb flag（部分和到了）。
- **做什么**：只有最后一个 CTA 轮询，每个对端一个线程（`ld.acquire.sys`）。全部到齐后它写一个本地字（`ctr[kCtrFlag + i]`），
  其它 CTA 只用 `ld.acquire.gpu` 等这个本地字。
- **为什么**：PCIe 写进来的 flag 要在本卡 L2 里落地。几百个 CTA 同时 system 级轮询同一行，会和这笔写抢同一片 L2，
  flag 落地被推迟几微秒（实测现象；L2 内部机制是推断）。
- **在哪**：`wait_peers`。

### 6.9 缓冲区按最坏情况一次分配

- **大小**：`Sm120Fp4MegaMoeBuffer(num_max_tokens_per_rank=TCAP)` 初始化时一次分配。pool 是 `[本地专家][源 rank][TCAP]` 行，
  因为最坏情况下一个源 rank 的全部 token 都选同一个本地专家。
- **为什么不动态分配**：对端直接写我的显存，地址要在初始化时交换（cudaIpc 句柄或 NCCL 窗口）；实际用量要到 kernel 里路由交换后才知道；
  CUDA graph 要求地址固定。
- **量级**：README 形状、EP8、TCAP=64、MXFP4：pool 约 40 MB，中间缓冲约 16 MB，comb 约 3 MB，每卡约 60 MB。TCAP=128 时约翻倍。
- **在哪**：host `layout`。

## 7. 传输层

`'p2p'`（默认）：一块 `cudaMalloc`，`cudaIpc` 共享，所有对端都用 GPU 直接写（PCIe P2P）。其它取值都是一块 NCCL
对称窗口 + 一张“每对 rank 走 P2P 还是网卡”的表，表的种类和规则照搬 FlashInfer Cake 的 `cake_transport.py`
（`deep_gemm/mega/sm120_fp4_transport.py`）。哪些对能走 P2P 由 NCCL 对窗口的映射决定，代码不看拓扑：

| `transport` | 走 P2P 的对 |
|---|---|
| `'all_gin'` | 无，所有对端走网卡 |
| `'all_lsa'` | 全部；NCCL 映射不了某一对就报错 |
| `'lsa_where_reachable'` | NCCL 能映射的对，其余走网卡 |
| `'auto'` | 建 buffer 时实测：四张候选表（上面三张 + `lsa_where_faster`：能映射且单对实测 P2P 带宽不低于网卡）在一次满容量调用的字节量下做 all-to-all 打分，混合表要比最好的统一表快 3% 且 3 轮最慢/最快 ≤ 1.5 才选；rank 0 决定后广播 |
| `'tuned'` | 按部署记录的表（`'auto'` + `DG_SM120_TRANSPORT_RECORD=1` 记录）；没有记录就是 `all_gin` |
| 显式 `world × world` 元组 | 对角 0，P2P 1，网卡 2，必须对称 |

`'auto'` 的 probe kernel 照搬 Cake（`probe_send`：每个目标一个 CTA，P2P 用 cp.async.bulk 三级流水，网卡用分块 GIN put，
最后一个 CTA 先做设备侧会合再计时），在 GIN 版扩展里（`probe_round`），用窗口的 pool 区当收件箱、GIN 暂存区当源。

网卡路径（`SM120_FP4_GIN=1` 编译，`Gin::mask` 的位标记哪些对端走网卡）：

- **写法**：数据先写到本 rank 窗口里的暂存区，再由 kernel 里的线程发 NCCL GIN put，写到对方窗口里 P2P 会写的同一偏移。
- **去程打包**：发往同一网卡对端的副本行（数值 + scale + 元数据 int）在暂存区按“(对端本地专家, slot)”压缩排好，整个
  dispatch 结束后由最后一个离开 dispatch 的 CTA 每个对端一次 put，写到对端的接收区（`off_rrow`）；对端 FC1/FC2 读
  网卡来源的行时按同一压缩下标（`row_ptr`）。每行一次 put 时受网卡消息速率限制（.3 MXFP4 M1024：每行 put 1047 µs，
  打包 888 µs）。没有去重。
- **每个对端的副本数**：每个 CTA 开头用整个 CTA 并行数一次（`s_ncopy`），发 put、等回程信号、收尾累加信号总数都读它；
  原来是单线程在收尾时遍历 T×top-k，.3 上 M1024 收尾就要约 350 µs。
- **回程打包**：一个 tile 的 FC2 全部完成后（`kCtrTile` 计数），发往同一源 rank 的连续行一次 put。这些行在暂存区按
  “(本地专家, slot)”压缩顺序排列；源 rank 用 `cidx[t][k]` 找到自己的那一行。
- **信号代替 flag**：每个 put 给接收方的一个信号加上它带的行数（路由计数是 1）。信号按“种类 × 源 rank”编号（`kSigRoute/Rows/Comb + rank`），
  接收方等累计值到期望（不变量 6）。
- **实测**（单机 .20，所有 rank 共用 `mlx5_0`）：上面两项打包合起来，比每块数据单独 put 快 1.2×（M2）到 4.7×（M512）。两项各自的贡献没有分开测。
- **限制**：网卡版每 rank 最多 64 个 token（`kMaxTiles`）。

## 8. 已知限制和未解决问题

| 问题 | 现状 |
|---|---|
| 发往同一 rank 的多个专家副本没有去重（P2P 和网卡都是） | 随机路由下 README 形状约 34% 的发送是重复的。bench 的均衡路由每个 token 的 8 个专家正好分在 8 个 rank，测不出这个问题。改法：每 rank 发一份 + 接收方副本索引表，需要改 pool 布局和 FC1 读行 |
| RTX PRO 6000 主机（smc521ge-0049，双路 Xeon 8480+）M256 起比 .20 慢 6–26× | 单对 NCCL 带宽正常（44–47 GB/s），P2P 全通。推测是“一张卡的很多 SM 同时写多个远端”时带宽塌，未验证 |
| CX8 主机（.1/.3/.6）上跨根端口的 P2P all-to-all 塌到每卡约 0.7 GB/s | 用传输表绕开：8 个网卡都给 NCCL 时它只映射同插槽的对，`'lsa_where_reachable'` 即“同插槽 P2P、跨插槽网卡”。`'auto'` 的 probe 只搬数据，在 .3 上两张表差 1%，按 3% 规则选了 `all_gin`，而 kernel 实测 `lsa_where_reachable` 每个 M 都更快（W4A4.md） |
| 跨机只测过 EP16 两台 CX8 主机（.3 + .1） | 三种格式正确性通过，性能见 W4A4.md；Broadcom 主机（.20 类）之间、以及 auto 选表在跨机下的取舍没有测 |

试过、实测没有收益、已从代码删除的做法（不要重复尝试，除非条件变了）：

- 网卡去程“按 token 去重逐个 put”：在 .20 上 M2–512 都在 ±2% 以内（注意 bench 路由没有重复副本）。
- 每个 item 24 / 32 个 token；让权重留在 L2 的读法（只在热 L2 下有用，而测法只认冷 L2）。

## 9. 三种格式

一份源码，编译宏选格式：

| 格式 | 宏 | 激活 | scale | MMA |
|---|---|---|---|---|
| `mxfp4` | 默认 | E2M1 | 每 32 值一个 UE8M0 | `kind::mxf4nvf4` m16n8k64 `scale_vec::2X` |
| `nvfp4` | `SM120_FP4_NV=1` | E2M1 | 每 16 值一个 UE4M3，外加 fp32 全局 scale（权重每专家一个，FC1/FC2 输入每张量一个） | `kind::mxf4nvf4` m16n8k64 `scale_vec::4X` |
| `w4a8` | `SM120_FP4_A8=1` | E4M3 | 每 32 值一个 UE8M0 | `kind::mxf8f6f4` m16n8k32，每个 K64 两条 |

权重三种格式都是 E2M1。W4A8 的激活字节是 FP4 的两倍、每个 K64 要两条 MMA，token 多时（M1024）更早受计算限制。

## 10. 验证与测速

```bash
# 正确性：三种形状（small / readme / wide），每卡最多 128 个 token
torchrun --nproc-per-node 8 tests/test_mega_moe_sm120_fp4.py verify mxfp4 readme        # 可加 p2p / all_gin / all_lsa / lsa_where_reachable / auto / tuned
# 性能：README 形状，冷 L2，M = 全局 token 数
torchrun --nproc-per-node 8 tests/test_mega_moe_sm120_fp4.py bench mxfp4 2 4 8 16 64 256 512 1024
```

测速前 8 张卡必须全空。只报冷 L2。单机测网卡要设 `NCCL_IB_HCA==mlx5_0`（见 ENV.md）。

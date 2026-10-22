# One MoE layer of DSV4-Flash on vLLM's Marlin MXFP4 (W4A16) MoE, EP4 the way vLLM's DP+EP
# allgather_reducescatter path runs it (torchrun, 4 ranks): all-gather hidden states, top-k ids and
# weights of all ranks, Marlin over the 4T tokens with this rank's 64 experts (expert_map), reduce-scatter.
# Captured in a CUDA graph, timed over 100 replays. Every rank has T tokens.
import os
import statistics
import sys

import torch
import torch.distributed as dist
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_mxfp4_like
from vllm.scalar_type import scalar_types

H, I, E, K = 4096, 2048, 256, 6
rank, ws = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
L = E // ws
torch.manual_seed(20261022 + rank)


def experts(n, k):
    _, q, s = rand_marlin_weight_mxfp4_like(torch.empty(n, k, dtype=torch.bfloat16, device="cuda"), 32)[:3]
    return q.unsqueeze(0).expand(L, *q.shape).contiguous(), s.unsqueeze(0).expand(L, *s.shape).contiguous()


w1, s1 = experts(2 * I, H)
w2, s2 = experts(H, I)
emap = torch.full((E,), -1, dtype=torch.int32, device="cuda")
emap[rank * L : (rank + 1) * L] = torch.arange(L, dtype=torch.int32, device="cuda")
for T in [int(t) for t in sys.argv[1:]]:
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1
    w, ids = torch.topk(torch.softmax(torch.randn(T, E, device="cuda"), -1), K, -1)
    w, ids = w.float().contiguous(), ids.to(torch.int32).contiguous()
    if os.environ.get("ROUTING_DIR"):  # reuse another bench's exact per-rank routing
        r = torch.load(os.path.join(os.environ["ROUTING_DIR"], f"routing_T{T}_r{rank}.pt"))
        w, ids = r["w"].cuda().float().contiguous(), r["ids"].cuda().to(torch.int32).contiguous()
    xg = torch.empty(ws * T, H, dtype=torch.bfloat16, device="cuda")
    wg = torch.empty(ws * T, K, dtype=torch.float32, device="cuda")
    ig = torch.empty(ws * T, K, dtype=torch.int32, device="cuda")
    y = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")

    def step():
        dist.all_gather_into_tensor(xg, x)
        dist.all_gather_into_tensor(wg, w)
        dist.all_gather_into_tensor(ig, ids)
        out = fused_marlin_moe(
            xg, w1, w2, None, None, s1, s2, wg, ig,
            quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=E, expert_map=emap,
        )
        dist.reduce_scatter_tensor(y, out)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    dist.barrier()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    runs = []
    if os.environ.get("PROFILE") == "1":
        torch.cuda.profiler.start()
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        dist.barrier()
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(100):
            graph.replay()
        b.record()
        b.synchronize()
        runs.append(a.elapsed_time(b) * 1e3 / 100)
        dist.barrier()
    us = torch.tensor([statistics.median(runs)], device="cuda")
    allus = [torch.zeros_like(us) for _ in range(ws)]
    dist.all_gather(allus, us)
    if rank == 0:
        v = [round(a.item(), 1) for a in allus]
        print(f"MARLIN per_rank_tokens {T:3d} us_per_moe max {max(v):8.1f} ranks {v}", flush=True)
    del graph
    torch.cuda.synchronize()
    dist.barrier()
dist.destroy_process_group()

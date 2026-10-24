# marlin_h20shape.py M...: the H20/B200 README shape (E384 / H3072 / I1280 / top-8, EP8) on vLLM's Marlin MXFP4
# (W4A16) MoE the way vLLM's DP+EP allgather_reducescatter path runs it: all-gather hidden states / ids / weights,
# Marlin over every rank's rows with this rank's 48 experts (expert_map), reduce-scatter. Balanced routing and
# timing identical to mm_test.py h20shape: 3 passes of 2 warm-up + 30 streamed graph replays, median of the last 3.
import os, statistics, sys
import torch
import torch.distributed as dist
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_mxfp4_like
from vllm.scalar_type import scalar_types

H, I, E, K = 3072, 1280, 384, 8
rank, ws = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
L = E // ws
torch.manual_seed(20261024 + rank)


def experts(n, k):
    _, q, s = rand_marlin_weight_mxfp4_like(torch.empty(n, k, dtype=torch.bfloat16, device="cuda"), 32)[:3]
    return q.unsqueeze(0).expand(L, *q.shape).contiguous(), s.unsqueeze(0).expand(L, *s.shape).contiguous()


w1, s1 = experts(2 * I, H)
w2, s2 = experts(H, I)
emap = torch.full((E,), -1, dtype=torch.int32, device="cuda")
emap[rank * L:(rank + 1) * L] = torch.arange(L, dtype=torch.int32, device="cuda")
for M in [int(t) for t in sys.argv[1:]]:
    T = max(1, M // ws)
    ids = torch.full((T, K), -1, dtype=torch.int32)
    for i in range(T):
        gi = rank * T + i
        if gi < M:
            ids[i] = torch.tensor([s * L + (gi + 7 * s) % L for s in range(K)], dtype=torch.int32)
    ids = ids.cuda()
    w = torch.full((T, K), 1.0 / K, device="cuda")
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1
    xg = torch.empty(ws * T, H, dtype=torch.bfloat16, device="cuda")
    wg = torch.empty(ws * T, K, dtype=torch.float32, device="cuda")
    ig = torch.empty(ws * T, K, dtype=torch.int32, device="cuda")
    y = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")

    def step():
        dist.all_gather_into_tensor(xg, x)
        dist.all_gather_into_tensor(wg, w)
        dist.all_gather_into_tensor(ig, ids)
        out = fused_marlin_moe(xg, w1, w2, None, None, s1, s2, wg, ig, quant_type_id=scalar_types.float4_e2m1f.id,
                               global_num_experts=E, expert_map=emap)
        dist.reduce_scatter_tensor(y, out)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    dist.barrier()
    meds = []
    for p in range(3):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(33)]
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay(); graph.replay()
        for i in range(32):
            ev[i].record()
            if i < 30:
                graph.replay()
        torch.cuda.synchronize()
        ts = [ev[i].elapsed_time(ev[i + 1]) * 1e3 for i in range(30)]
        meds.append(statistics.median(ts[-3:]))
        dist.barrier()
    if os.environ.get("COLD") == "1":  # cold L2: flush 256 MB before every replay, time the replay alone
        flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        meds = []
        for p in range(3):
            ts = []
            for i in range(32):
                flush.fill_(i)
                torch.cuda.synchronize()
                dist.barrier()
                a_, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a_.record(); graph.replay(); b_.record(); b_.synchronize()
                ts.append(a_.elapsed_time(b_) * 1e3)
            meds.append(statistics.median(ts[2:]))
    if os.environ.get("KPROF") == "1":  # per-kernel GPU time of the replays on rank 0
        from torch.profiler import profile, ProfilerActivity
        torch.cuda.synchronize(); dist.barrier()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            agg = {}
            for ev in prof.events():
                if ev.device_type.name == "CUDA":
                    key = ev.name[:60]
                    agg[key] = agg.get(key, 0.0) + ev.device_time_total
            print(f"M{M} per-replay kernel us:", {k: round(v / 20, 1) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:8]}, flush=True)
    us = torch.tensor([statistics.median(meds)], device="cuda")
    allus = [torch.zeros_like(us) for _ in range(ws)]
    dist.all_gather(allus, us)
    if rank == 0:
        v = [round(a.item(), 1) for a in allus]
        print(f"MARLIN_H20SHAPE EP{ws} M{M:<3d} gpu0 {v[0]:6.1f} us  max-rank {max(v):6.1f}  ranks {v}", flush=True)
    del graph
    torch.cuda.synchronize()
    dist.barrier()
dist.destroy_process_group()

# EP MegaMoE kernel benchmarks on SM120 (the numbers of docs/SM120_MEGAMOE_RESULTS.md), straight on the kernel
# source (torch extension built into sm120_6000d/bench/build on first use):
#   torchrun --nproc_per_node=4 ep_kernel_bench.py bench 1 2 8 16 20 24 32 48 64
#       DSV4-Flash shape (H 4096, I 2048, 64 experts per rank, top-6), T tokens per rank, CUDA graph, 100 replays,
#       median of 5 runs, us per MoE layer; SAVE_ROUTING=<dir> saves each rank's routing for marlin_ep4_bench.py
#   torchrun --nproc_per_node=8 ep_kernel_bench.py h20shape 2 4 8 16
#       the H20/B200 README shape (E384 / H3072 / I1280 / top-8, EP8), global M tokens, balanced routing; warm L2
#       (3 passes of 2 warm-up + 30 streamed replays, median of the last 3) or COLD=1 (256 MB flush before every
#       replay); STAMPS=1 also saves the in-kernel phase stamps of 5 eager calls (stamps_report.py prints them)
import os
import statistics
import sys

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "deep_gemm", "include", "deep_gemm", "impls", "sm120_mxfp4_mega_moe.cu")
BUILD = os.path.join(HERE, "build")
os.makedirs(BUILD, exist_ok=True)


def build():
    return load(name="sm120mm_bench", sources=[SRC], build_directory=BUILD, verbose=False,
                extra_cuda_cflags=["-gencode=arch=compute_120a,code=sm_120a", "-O3", "-std=c++20", "--use_fast_math"])


if rank == 0:
    ext = build()
dist.barrier()
if rank != 0:
    ext = build()

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda")


def pack_w(codes):  # [E, N, K] uint8 nibble codes (kernel row order) -> [E, ...] uint8 lane-fragment layout
    E, N, K = codes.shape
    x = codes.view(E, N // 16, 2, 8, K // 64, 2, 2, 4, 4)  # [E, nb, half, g, kc, s, p, c, j]
    b = x[:, :, 0] | (x[:, :, 1] << 4)  # [E, nb, g, kc, s, p, c, j]
    return b.permute(0, 1, 3, 2, 6, 4, 5, 7).contiguous().view(E, -1)  # [E, nb, kc, g, c, s, p, j]


def pack_s(sf):  # [E, N, K/32] ue8m0 -> [E, nb, K/128, 16, 4]
    E, N, KS = sf.shape
    return sf.view(E, N // 16, 16, KS // 4, 4).permute(0, 1, 3, 2, 4).contiguous().view(E, -1)


def w13_rows(I):  # kernel row r -> logical row (gate 0..I-1, up I..2I-1); block b = gate 8b..8b+7, up 8b..8b+7
    r = torch.arange(2 * I, device="cuda")
    blk, w = r // 16, r % 16
    return torch.where(w < 8, blk * 8 + w, I + blk * 8 + (w - 8))


def mx_quant(x):  # [..., K] fp32 -> dequantized fp32 with the kernel's rule (per 32: 2^ceil(log2(amax/448)))
    s = x.unflatten(-1, (-1, 32))
    amax = s.abs().amax(-1, keepdim=True)
    e = torch.where(amax > 0, torch.ceil(torch.log2(amax / 448.0)), torch.full_like(amax, -127.0)).clamp(-127, 127)
    q = (s * torch.exp2(-e)).to(torch.float8_e4m3fn).float()
    return (q * torch.exp2(e)).flatten(-2)


def expert_codes(e, H, I):  # deterministic per global expert: logical rows
    gen = torch.Generator(device="cuda").manual_seed(1000 + e)
    c13 = torch.randint(0, 16, (2 * I, H), device="cuda", dtype=torch.uint8, generator=gen)
    s13 = torch.randint(122, 126, (2 * I, H // 32), device="cuda", dtype=torch.uint8, generator=gen)
    c2 = torch.randint(0, 16, (H, I), device="cuda", dtype=torch.uint8, generator=gen)
    s2 = torch.randint(122, 126, (H, I // 32), device="cuda", dtype=torch.uint8, generator=gen)
    return c13, s13, c2, s2


def dq(codes, sf):
    return E2M1[codes.long()] * torch.exp2(sf.float() - 127).repeat_interleave(32, -1)


def routing(T, E, K, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(T, E, device="cuda", generator=gen)
    w, ids = torch.topk(torch.softmax(logits, -1), K, -1)
    return ids.to(torch.int32).contiguous(), (w / w.sum(-1, keepdim=True)).float().contiguous()


LIMIT = 10.0
mode = sys.argv[1]
TOPK = 6
if mode == "h20shape":  # the H20/B200 README shape: E384 / H3072 / I1280 / top-8 / EP8, global M, balanced routing
    H, I, TK, TCAP = 3072, 1280, 8, 64
    EL = 384 // world
    E = EL * world
    g = torch.Generator(device="cuda").manual_seed(20261023 + rank)
    w13 = torch.randint(0, 256, (EL, (2 * I) * H // 2), device="cuda", dtype=torch.uint8, generator=g)
    ws13 = torch.full((EL, (2 * I) * H // 32), 124, device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (EL, H * I // 2), device="cuda", dtype=torch.uint8, generator=g)
    ws2 = torch.full((EL, H * I // 32), 124, device="cuda", dtype=torch.uint8)
    h = ext.init(rank, world, EL, TCAP, H, I, TK)
    hs = [torch.empty_like(h) for _ in range(world)]
    dist.all_gather_object(hs, h)
    ext.open_peers(hs)
    for M in [int(t) for t in sys.argv[2:]]:
        rows = max(1, M // world)  # M2/M4/M8: one row per rank, M16: two
        ids = torch.full((rows, TK), -1, dtype=torch.int32)
        for i in range(rows):  # README balanced assignment: expert = s * 48 + (g + 7 s) % 48, weight 1/8
            gi = rank * rows + i
            if gi < M:
                ids[i] = torch.tensor([s * EL + (gi + 7 * s) % EL for s in range(TK)], dtype=torch.int32)
        ids = ids.cuda()
        wts = torch.full((rows, TK), 1.0 / TK, device="cuda")
        x = (torch.randn(rows, H, device="cuda", generator=g) * 0.1).bfloat16()
        out = torch.empty(rows, H, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            ext.run(x, ids, wts, out, w13, ws13, w2, ws2, LIMIT, None)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ext.run(x, ids, wts, out, w13, ws13, w2, ws2, LIMIT, None)
        dist.barrier()
        meds = []
        for p in range(3):  # 3 passes of 2 warm-up + 30 streamed replays; per replay GPU time from events
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
        if os.environ.get("STAMPS") == "1":  # stamped eager calls; the last one, every rank, absolute globaltimer
            st = torch.zeros(ext.grid_size() * 16, dtype=torch.int64, device="cuda")
            for _ in range(5):
                torch.cuda.synchronize()
                dist.barrier()
                ext.run(x, ids, wts, out, w13, ws13, w2, ws2, LIMIT, st)
            torch.cuda.synchronize()
            alls = [torch.zeros_like(st) for _ in range(world)]
            dist.all_gather(alls, st)
            if rank == 0:
                torch.save([a.cpu() for a in alls], os.path.join(os.environ.get("STAMPS_DIR", "."), f"stamps_h20shape_M{M}.pt"))
        us = torch.tensor([statistics.median(meds), float(ext.counters()[3])], device="cuda")
        allus = [torch.zeros_like(us) for _ in range(world)]
        dist.all_gather(allus, us)
        if rank == 0:
            v = [round(a[0].item(), 1) for a in allus]
            print(f"H20SHAPE EP{world} M{M:<3d} gpu0 {v[0]:6.1f} us  max-rank {max(v):6.1f}  ranks {v} err {[int(a[1].item()) for a in allus]}", flush=True)
        del graph
        torch.cuda.synchronize()
        dist.barrier()
else:
    H, I, EL, TCAP = 4096, 2048, 64, 64
    E = EL * world
    g = torch.Generator(device="cuda").manual_seed(20261022 + rank)
    w13 = torch.randint(0, 256, (EL, (2 * I) * H // 2), device="cuda", dtype=torch.uint8, generator=g)
    ws13 = torch.full((EL, (2 * I) * H // 32), 124, device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (EL, H * I // 2), device="cuda", dtype=torch.uint8, generator=g)
    ws2 = torch.full((EL, H * I // 32), 124, device="cuda", dtype=torch.uint8)
    h = ext.init(rank, world, EL, TCAP, H, I, TOPK)
    hs = [torch.empty_like(h) for _ in range(world)]
    dist.all_gather_object(hs, h)
    ext.open_peers(hs)
    if rank == 0:
        print("grid", ext.grid_size(), flush=True)
    for T in [int(t) for t in sys.argv[2:]]:
        ids, wts = routing(T, E, TOPK, 20261022 + 100 * T + rank)
        if os.environ.get("SAVE_ROUTING"):
            os.makedirs(os.environ["SAVE_ROUTING"], exist_ok=True)
            torch.save({"ids": ids.cpu(), "w": wts.cpu()}, os.path.join(os.environ["SAVE_ROUTING"], f"routing_T{T}_r{rank}.pt"))
        x = (torch.randn(T, H, device="cuda", generator=g) * 0.1).bfloat16()
        out = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            ext.run(x, ids, wts, out, w13, ws13, w2, ws2, LIMIT, None)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ext.run(x, ids, wts, out, w13, ws13, w2, ws2, LIMIT, None)
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
        err = ext.counters()[3]
        if os.environ.get("STAMPS") == "1":  # one stamped eager call, per-rank phase breakdown
            st = torch.zeros(ext.grid_size() * 16, dtype=torch.int64, device="cuda")
            dist.barrier()
            ext.run(x, ids, wts, out, w13, ws13, w2, ws2, LIMIT, st)
            torch.cuda.synchronize()
            s = st.view(-1, 16).cpu().double()
            t0 = s[:, 0].min()
            us = lambda c: (s[:, c] - t0) / 1e3
            rows = [f"rank{rank} T={T} items/CTA {s[:,7].mean():.2f} (max {s[:,7].max():.0f})",
                    f"  pushed(med/max) {us(1).median():.1f}/{us(1).max():.1f}  flags {us(2).median():.1f}/{us(2).max():.1f}",
                    f"  work-end min/med/max {us(3).min():.1f}/{us(3).median():.1f}/{us(3).max():.1f}  L1-wait/CTA {s[:,6].mean()/1e3:.1f}us",
                    f"  comb-flags {us(4).median():.1f}/{us(4).max():.1f}  end {us(5).max():.1f}",
                    f"  start(max) {us(0).max():.1f} route-push {us(8).max():.1f} route-ok {us(9).median():.1f}/{us(9).max():.1f} map {us(10).max():.1f} pf {us(11).median():.1f}/{us(11).max():.1f} flag-sent {us(12).max():.1f}"]
            print("\n".join(rows), flush=True)
        active = len(set(int(e) for e in torch.cat([t for t in [ids.flatten()]]).tolist()))
        us = torch.tensor([statistics.median(runs), float(err)], device="cuda")
        allus = [torch.zeros_like(us) for _ in range(world)]
        dist.all_gather(allus, us)
        if rank == 0:
            v = [round(a[0].item(), 1) for a in allus]
            errs = [int(a[1].item()) for a in allus]
            print(f"SM120MM per_rank_tokens {T:3d} us_per_moe max {max(v):8.1f} ranks {v} err {errs}", flush=True)
        del graph
        torch.cuda.synchronize()
        dist.barrier()
dist.destroy_process_group()

# DSV4.1-Flash MegaMoE on real routing: replays the top-6 expert ids SGLang's verify steps produced (96 tokens per rank,
# EP8), mapped to physical experts through an EPLB placement, one CUDA graph per call, cold L2 (256 MB written before
# each replay), median of 20 replays; the per-call time is the slowest rank's. 'stamps' prints the per-phase
# globaltimer stamps (rank 0 and the slowest rank) of the first call instead.
#   torchrun --nproc_per_node=8 tests/bench_real_routing.py <fmt> <transport> <dump dir> <placement .pt> [calls] [stamps]
# dump dir: rank<r>.pt = list of (step, rows per rank, ids [rows, 6] logical); placement: {'physical_to_logical_map':
# [layers, experts]} (a permutation: no replicas). calls: first:last (default 160:200, one step of 40 layers).
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mega_moe_sm120_fp4 import LIMIT, SHAPES, STAMPS, fp4  # noqa: E402  (fp4 loaded by file path there)


def main():
    fmt, transport, dump, placement = sys.argv[1:5]
    rest = sys.argv[5:]
    first, last = (int(v) for v in next((a for a in rest if ':' in a), '160:200').split(':'))
    stamps = 'stamps' in rest
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        fp4._ext(fmt, world=world)
    dist.barrier()
    h, i, e, topk = SHAPES['dsv41']
    el = e // world
    calls = torch.load(f'{dump}/rank{rank}.pt')[first:last]
    p2l = torch.load(placement)['physical_to_logical_map'].long()  # [layers, e]
    l2p = torch.empty_like(p2l)
    l2p.scatter_(1, p2l, torch.arange(e).expand_as(p2l).contiguous())
    gen = torch.Generator(device='cuda').manual_seed(20261117 + rank)
    l1 = (torch.randint(0, 256, (el, 2 * i, h // 2), device='cuda', dtype=torch.uint8, generator=gen),
          torch.full((el, 2 * i, h // 32), 124, device='cuda', dtype=torch.uint8))
    l2 = (torch.randint(0, 256, (el, h, i // 2), device='cuda', dtype=torch.uint8, generator=gen),
          torch.full((el, h, i // 32), 124, device='cuda', dtype=torch.uint8))
    rows = max(c[2].shape[0] for c in calls)
    buf = fp4.Sm120Fp4MegaMoeBuffer(dist.group.WORLD, fmt, e, max(rows, 128), topk, h, i, transport)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')
    times = []
    for n, (step, _, ids_l) in enumerate(calls):
        layer = (first + n) % 40
        ids = ids_l.long()
        ids = torch.where(ids >= 0, l2p[layer][ids.clamp_min(0)], ids).int().cuda()
        t = ids.shape[0]
        x = (torch.randn(t, h, device='cuda', generator=gen) * 0.1).bfloat16()
        wts = torch.full((t, topk), 1.0 / topk, device='cuda')
        y = torch.empty(t, h, device='cuda', dtype=torch.bfloat16)
        st = torch.zeros(4096, 16, dtype=torch.int64, device='cuda') if stamps else None
        run = lambda: fp4.fp4_mega_moe_sm120(y, x, ids, wts, l1, l2, buf, None, activation_clamp=LIMIT, stamps=st)
        for _ in range(2):
            run()
        torch.cuda.synchronize()
        assert buf.error() == 0, (rank, n, buf.error())
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        dist.barrier()
        ts = []
        for k in range(22):
            flush.fill_(k)
            torch.cuda.synchronize()
            dist.barrier()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graph.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        us = torch.tensor([statistics.median(ts[2:])], device='cuda')
        allus = [torch.zeros_like(us) for _ in range(world)]
        dist.all_gather(allus, us)
        times.append(max(v.item() for v in allus))
        if stamps:
            s = st[st[:, 0] > 0].double()
            t0 = s[:, 0].min()
            line = torch.tensor([((s[:, k][s[:, k] > 0].max() - t0) / 1e3).item() if (s[:, k] > 0).any() else -1.0
                                 for k, _ in STAMPS], device='cuda')
            lines = [torch.zeros_like(line) for _ in range(world)]
            dist.all_gather(lines, line)
            if rank == 0:
                for r in range(world):
                    print(f'stamps call {first + n} rank{r}: ' + ' | '.join(
                        f'{name} {lines[r][j].item():.1f}' for j, (_, name) in enumerate(STAMPS) if lines[r][j] >= 0),
                        flush=True)
            del graph
            if n >= 2:
                break
            continue
        del graph
        torch.cuda.synchronize()
    if rank == 0:
        print(f'{fmt} {transport} real routing calls {first}:{first + len(times)}: mean {statistics.mean(times):.1f} '
              f'median {statistics.median(times):.1f} min {min(times):.1f} max {max(times):.1f} us (slowest rank)',
              flush=True)
    buf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

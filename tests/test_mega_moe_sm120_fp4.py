# SM120 W4A4 MegaMoE (deep_gemm/mega/sm120_fp4.py): MXFP4 and NVFP4.
#   torchrun --nproc_per_node=8 tests/test_mega_moe_sm120_fp4.py verify mxfp4 [small|readme|wide]
#       EP vs a quantization-exact reference, every tokens-per-item width (small: H512 / I256 / E128 / top-6;
#       readme: the README shape H3072 / I1280 / E384 / top-8; wide: H5120 / I256 / E128 / top-6, the token goes
#       out in several dispatch chunks)
#   torchrun --nproc_per_node=8 tests/test_mega_moe_sm120_fp4.py bench nvfp4 2 4 8 16 64 256 512
#       cold-L2 kernel time, README shape, global M tokens (TILE=8/16 forces the tokens per item); `w4a8` is this
#       kernel with MXFP8 activations; `stamps` instead of `bench` prints the per-phase kernel stamps
# The reference applies the kernel's quantization to the activations and to the SwiGLU output:
#   MXFP4: per 32, scale 2^ceil(log2(amax/6)), E2M1 round-to-nearest-even.
#   W4A8: per 32, scale 2^ceil(log2(amax/448)), E4M3.
#   NVFP4: per 16, scale E4M3(amax/6 / g) with the tensor's global scale g, E2M1 of v / (scale * g).
import importlib.util
import os
import statistics
import sys

import torch
import torch.distributed as dist


def _load(name):  # by file path: these modules need no other part of deep_gemm (whose package import wants _C)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deep_gemm', 'mega', f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fp4 = _load('sm120_fp4')
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
LIMIT = 10.0
A1, A2 = 4.0 / 2688, 64.0 / 2688  # NVFP4 FC1 / FC2 input global scales (amax / (6 * 448))


def e2m1_rn(v):  # round to the E2M1 grid, nearest even code, saturating at 6
    a = v.abs().clamp(max=6.0)
    q = torch.where(a < 2, torch.round(a * 2) / 2, torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2))
    return q * torch.sign(v)


def fp4_quant(x, fmt, gscale):  # quantize-dequantize along the last dim
    vec = fp4.FORMATS[fmt]
    s = x.unflatten(-1, (-1, vec))
    amax = s.abs().amax(-1, keepdim=True)
    if fmt == 'w4a8':  # MXFP8: E4M3 values, scale 2^ceil(log2(amax/448))
        e = torch.where(amax > 0, torch.ceil(torch.log2(amax / 448)), torch.full_like(amax, -127.0)).clamp(-127, 127)
        sc = torch.exp2(e)
        return ((s / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc).flatten(-2)
    if fmt == 'mxfp4':
        e = torch.where(amax > 0, torch.ceil(torch.log2(amax / 6)), torch.full_like(amax, -127.0)).clamp(-127, 127)
        sc = torch.exp2(e)
        q = e2m1_rn(s / sc)
        return (q * sc).flatten(-2)
    sc = (amax / 6 * (1 / gscale)).clamp(max=448).to(torch.float8_e4m3fn).float()
    q = torch.where(sc > 0, e2m1_rn(s * ((1 / gscale) / sc)), torch.zeros_like(s))
    return (q * sc * gscale).flatten(-2)


def expert_codes(e, h, i, fmt):  # deterministic per global expert, checkpoint row order (gate rows then up rows)
    vec = fp4.FORMATS[fmt]
    gen = torch.Generator(device='cuda').manual_seed(1000 + e)
    lo, hi = (122, 126) if fmt != 'nvfp4' else (0x30, 0x40)  # UE8M0 2^-5..2^-2 / E4M3 0.5..1.875
    c13 = torch.randint(0, 16, (2 * i, h), device='cuda', dtype=torch.uint8, generator=gen)
    s13 = torch.randint(lo, hi, (2 * i, h // vec), device='cuda', dtype=torch.uint8, generator=gen)
    c2 = torch.randint(0, 16, (h, i), device='cuda', dtype=torch.uint8, generator=gen)
    s2 = torch.randint(lo, hi, (h, i // vec), device='cuda', dtype=torch.uint8, generator=gen)
    g = torch.rand(2, device='cuda', generator=gen) * 0.05 + 0.05  # NVFP4 W13 / W2 global scales
    return c13, s13, c2, s2, g


def to_checkpoint(codes):  # [..., K] codes -> [..., K/2] bytes, low nibble first
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def dq(codes, sf, fmt, g):
    vec = fp4.FORMATS[fmt]
    s = torch.exp2(sf.float() - 127) if fmt != 'nvfp4' else sf.view(torch.float8_e4m3fn).float() * g
    return E2M1.to(codes.device)[codes.long()] * s.repeat_interleave(vec, -1)


def reference(x, ids, wts, expert_of, i, fmt):
    xdq = fp4_quant(x.float(), fmt, A1)
    ref = torch.zeros(x.shape, device='cuda')
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if e < 0:
                continue
            c13, s13, c2, s2, g = expert_of(e)
            hh = dq(c13, s13, fmt, g[0]) @ xdq[t]
            gate, up = hh[:i].clamp(max=LIMIT), hh[i:].clamp(-LIMIT, LIMIT)
            act = fp4_quant((gate * torch.sigmoid(gate) * up)[None], fmt, A2)[0]
            y = dq(c2, s2, fmt, g[1]) @ act
            ref[t] += wts[t, k] * y.bfloat16().float()
    return ref


def routing(t, e, k, seed):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    w, ids = torch.topk(torch.softmax(torch.randn(t, e, device='cuda', generator=gen), -1), k, -1)
    return ids.to(torch.int32).contiguous(), (w / w.sum(-1, keepdim=True)).float().contiguous()


SHAPES = {  # hidden, intermediate, experts, top-k
    'small': (512, 256, 128, 6),
    'readme': (3072, 1280, 384, 8),  # the README's validated shape (48 local experts at EP8)
    'wide': (5120, 256, 128, 6),
}


def kernel_inputs(fmt, experts):
    l1 = (torch.stack([to_checkpoint(c[0]) for c in experts]), torch.stack([c[1] for c in experts]))
    l2 = (torch.stack([to_checkpoint(c[2]) for c in experts]), torch.stack([c[3] for c in experts]))
    l1, l2 = fp4.transform_fp4_weights_for_mega_moe_sm120(l1, l2, fmt)
    gs = None
    if fmt == 'nvfp4':
        gs = fp4.nvfp4_global_scales(A1, A2, torch.stack([c[4][0] for c in experts]),
                                     torch.stack([c[4][1] for c in experts]))
    return l1, l2, gs


def verify(fmt, shape, rank, world, group):
    h, i, e, topk = SHAPES[shape]
    el = e // world
    l1, l2, gs = kernel_inputs(fmt, [expert_codes(rank * el + j, h, i, fmt) for j in range(el)])
    buf = fp4.Sm120Fp4MegaMoeBuffer(group, fmt, e, 64, topk, h, i)
    worst = 0.0
    for trial, t in enumerate([1, 3, 8, 13, 64] if shape == 'small' else [1, 3, 13, 40]):
        ids, wts = routing(t, e, topk, 77 + 10 * trial + rank)
        if trial % 2 == 1:  # padding token: every k is -1
            ids[-1] = -1
        x = (torch.randn(t, h, device='cuda', generator=torch.Generator(device='cuda').manual_seed(5 + rank + trial))
             * 0.5).bfloat16()
        ref = reference(x, ids, wts, lambda g: expert_codes(g, h, i, fmt), i, fmt)
        for tile in (0, 8, 16):  # 0: the automatic choice
            y = torch.zeros(t, h, device='cuda', dtype=torch.bfloat16)
            for _ in range(2):  # the second call checks the per-call state was reset
                fp4.fp4_mega_moe_sm120(y, x, ids, wts, l1, l2, buf, gs, activation_clamp=LIMIT, tile_tokens=tile)
            torch.cuda.synchronize()
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            worst = max(worst, rel)
            print(f'{fmt} rank{rank} T={t} tile={tile} err={buf.error()} rel_l2={rel:.2e}', flush=True)
    w = torch.tensor([worst], device='cuda')
    dist.all_reduce(w, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f'{fmt} {shape} EP{world} VERIFY max rel_l2 {w.item():.2e}', 'PASS' if w.item() < 2e-2 else 'FAIL',
              flush=True)


STAMPS = [(8, 'route-push'), (1, 'dispatched'), (12, 'flag-sent'), (9, 'route-ok'), (10, 'map-ok'), (2, 'rows-ok'),
          (14, 'first-FC2'), (13, 'last-FC1'), (15, 'comb-release'), (3, 'work-end'), (4, 'comb-ok'), (5, 'end')]


def bench(fmt, rank, world, group, ms, stamps=False):
    """README shape and method (docs/SM120_MEGAMOE_RESULTS.md, ep_kernel_bench.py h20shape): global M tokens over the
    ranks (one row per rank up to M = world), balanced routing expert = s * EL + (g + 7 s) % EL with weight 1/top-k.
    Cold L2: 256 MB written before each graph replay, ranks released by a barrier, median of 30 replays, median over
    3 passes."""
    h, i, e, topk = SHAPES['readme']
    el = e // world
    gen = torch.Generator(device='cuda').manual_seed(20261023 + rank)
    vec, sf = (32, 124) if fmt != 'nvfp4' else (16, 0x38)
    w13 = torch.randint(0, 256, (el, 2 * i, h // 2), device='cuda', dtype=torch.uint8, generator=gen)
    w2 = torch.randint(0, 256, (el, h, i // 2), device='cuda', dtype=torch.uint8, generator=gen)
    l1 = (w13, torch.full((el, 2 * i, h // vec), sf, device='cuda', dtype=torch.uint8))
    l2 = (w2, torch.full((el, h, i // vec), sf, device='cuda', dtype=torch.uint8))
    cap = max(max(1, m // world) for m in ms)
    tile = int(os.environ.get('TILE', '0'))
    gs = fp4.nvfp4_global_scales(A1, A2, torch.full((el,), 0.06, device='cuda'),
                                 torch.full((el,), 0.06, device='cuda')) if fmt == 'nvfp4' else None
    buf = fp4.Sm120Fp4MegaMoeBuffer(group, fmt, e, cap, topk, h, i)
    run_fn = lambda y, x, ids, w, st=None: fp4.fp4_mega_moe_sm120(y, x, ids, w, l1, l2, buf, gs,
                                                                  activation_clamp=LIMIT, tile_tokens=tile, stamps=st)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')
    for m in ms:
        rows = max(1, m // world)
        ids = torch.full((rows, topk), -1, dtype=torch.int32)
        for r in range(rows):
            gi = rank * rows + r
            if gi < m:
                ids[r] = torch.tensor([s * el + (gi + 7 * s) % el for s in range(topk)], dtype=torch.int32)
        ids = ids.cuda()
        wts = torch.full((rows, topk), 1.0 / topk, device='cuda')
        x = (torch.randn(rows, h, device='cuda', generator=gen) * 0.1).bfloat16()
        y = torch.empty(rows, h, device='cuda', dtype=torch.bfloat16)
        for _ in range(3):
            run_fn(y, x, ids, wts)
        torch.cuda.synchronize()
        assert buf.error() == 0, f'rank {rank}: kernel error word {buf.error()}'
        if stamps:  # per-phase globaltimer stamps (us since this rank's first CTA start, max over CTAs) of the last of
            # 8 graph replays, cold L2 as below
            st = torch.zeros(4096, 16, dtype=torch.int64, device='cuda')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_fn(y, x, ids, wts, st)
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(8):
                flush.fill_(1)
                torch.cuda.synchronize()
                dist.barrier()
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            del graph
            st = st[st[:, 0] > 0].double()
            t0 = st[:, 0].min()
            line = [f'{name} {((st[:, k][st[:, k] > 0].max() - t0) / 1e3).item():.1f}' for k, name in STAMPS
                    if (st[:, k] > 0).any()]
            if rank == 0:
                print(f'stamps {fmt} M{m} rank0: ' + ' | '.join(line), flush=True)
            continue
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_fn(y, x, ids, wts)
        dist.barrier()
        cold = []
        for _ in range(3):
            ts = []
            for k in range(32):
                flush.fill_(k)
                torch.cuda.synchronize()
                dist.barrier()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graph.replay()
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b) * 1e3)
            cold.append(statistics.median(ts[2:]))
        assert buf.error() == 0
        us = torch.tensor([statistics.median(cold)], device='cuda')
        allus = [torch.zeros_like(us) for _ in range(world)]
        dist.all_gather(allus, us)
        if rank == 0:
            print(f'{fmt} readme EP{world} tile {tile or "auto"} M{m} (T/rank {rows}): gpu0 cold '
                  f'{allus[0].item():.1f} us | max-rank {max(a.item() for a in allus):.1f}', flush=True)
        del graph
        torch.cuda.synchronize()
        dist.barrier()


def main():
    mode, fmt, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0:  # build once before the other ranks load it
        fp4._ext(fmt)
    dist.barrier()
    if mode == 'verify':
        verify(fmt, args[0] if args else 'small', rank, world, dist.group.WORLD)
    else:
        bench(fmt, rank, world, dist.group.WORLD, [int(a) for a in args] or [2, 4, 8, 16, 64, 256, 512],
              mode == 'stamps')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

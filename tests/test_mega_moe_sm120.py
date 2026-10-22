# SM120 MegaMoE through the deep_gemm wrapper: checkpoint-layout MXFP4 weights -> transform -> kernel.
#   torchrun --nproc_per_node=4 tests/test_mega_moe_sm120.py verify       EP vs fp32 reference (H 512, I 256, top-6)
#   torchrun --nproc_per_node=4 tests/test_mega_moe_sm120.py bench 1 8 16 EP, DSV4-Flash shape (H 4096, I 2048,
#                                                                        256 experts), CUDA graph, us per MoE layer
#                                                                        (SAVE_ROUTING=<dir> saves each rank's routing)
#   python tests/test_mega_moe_sm120.py tpverify                         TP kernel vs fp32 reference (32 experts)
#   TPS=8 python tests/test_mega_moe_sm120.py tp 1 2 4 8 16              TP kernel, DSV4-Flash shape with I/TPS, cold L2
# The reference applies the kernel's MXFP8 rule to activations (per 32: 2^ceil(log2(amax/448)), E4M3) and to the
# SwiGLU output; SwiGLU clamps gate at max and up at +-limit.
import os
import statistics
import sys

import torch
import torch.distributed as dist

import deep_gemm

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
LIMIT = 10.0
TOPK = 6


def mx_quant(x):
    s = x.unflatten(-1, (-1, 32))
    amax = s.abs().amax(-1, keepdim=True)
    e = torch.where(amax > 0, torch.ceil(torch.log2(amax / 448.0)), torch.full_like(amax, -127.0)).clamp(-127, 127)
    q = (s * torch.exp2(-e)).to(torch.float8_e4m3fn).float()
    return (q * torch.exp2(e)).flatten(-2)


def expert_codes(e, h, i):  # deterministic per global expert, checkpoint row order (gate rows then up rows)
    gen = torch.Generator(device='cuda').manual_seed(1000 + e)
    c13 = torch.randint(0, 16, (2 * i, h), device='cuda', dtype=torch.uint8, generator=gen)
    s13 = torch.randint(122, 126, (2 * i, h // 32), device='cuda', dtype=torch.uint8, generator=gen)
    c2 = torch.randint(0, 16, (h, i), device='cuda', dtype=torch.uint8, generator=gen)
    s2 = torch.randint(122, 126, (h, i // 32), device='cuda', dtype=torch.uint8, generator=gen)
    return c13, s13, c2, s2


def to_checkpoint(codes):  # [..., K] codes -> [..., K/2] bytes, low nibble first
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def kernel_weights(experts):
    l1 = (torch.stack([to_checkpoint(c[0]) for c in experts]), torch.stack([c[1] for c in experts]))
    l2 = (torch.stack([to_checkpoint(c[2]) for c in experts]), torch.stack([c[3] for c in experts]))
    return deep_gemm.transform_mxfp4_weights_for_mega_moe_sm120(l1, l2)


def dq(codes, sf):
    return E2M1.to(codes.device)[codes.long()] * torch.exp2(sf.float() - 127).repeat_interleave(32, -1)


def reference(x, ids, wts, expert_of, i, bf16_per_expert):  # EP pushes each expert's output as bf16
    xdq = mx_quant(x.float())
    ref = torch.zeros(x.shape, device='cuda')
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if e < 0:
                continue
            c13, s13, c2, s2 = expert_of(e)
            hh = dq(c13, s13) @ xdq[t]
            gate, up = hh[:i].clamp(max=LIMIT), hh[i:].clamp(-LIMIT, LIMIT)
            act = mx_quant((gate * torch.sigmoid(gate) * up)[None])[0]
            y = dq(c2, s2) @ act
            ref[t] += wts[t, k] * (y.bfloat16().float() if bf16_per_expert else y)
    return ref


def routing(t, e, k, seed):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    w, ids = torch.topk(torch.softmax(torch.randn(t, e, device='cuda', generator=gen), -1), k, -1)
    return ids.to(torch.int32).contiguous(), (w / w.sum(-1, keepdim=True)).float().contiguous()


def random_weights(e, h, i, seed):  # bench: random bytes in kernel layout, scales 2^-3
    g = torch.Generator(device='cuda').manual_seed(seed)
    return ((torch.randint(0, 256, (e, 2 * i, h // 2), device='cuda', dtype=torch.uint8, generator=g),
             torch.full((e, 2 * i, h // 32), 124, device='cuda', dtype=torch.uint8)),
            (torch.randint(0, 256, (e, h, i // 2), device='cuda', dtype=torch.uint8, generator=g),
             torch.full((e, h, i // 32), 124, device='cuda', dtype=torch.uint8)))


def ep(mode, args):
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    group = dist.group.WORLD
    if rank == 0:  # build once before the other ranks load it
        deep_gemm.mega.sm120._ext()
    dist.barrier()
    if mode == 'verify':
        h, i, el = 512, 256, 128 // world
        l1, l2 = kernel_weights([expert_codes(rank * el + j, h, i) for j in range(el)])
        buf = deep_gemm.get_symm_buffer_for_mega_moe_sm120(group, el * world, 64, TOPK, h, i)
        worst = 0.0
        for trial, t in enumerate([1, 3, 8, 13, 64]):
            ids, wts = routing(t, el * world, TOPK, 77 + 10 * trial + rank)
            if trial % 2 == 1:  # padding token: every k is -1
                ids[-1] = -1
            x = (torch.randn(t, h, device='cuda', generator=torch.Generator(device='cuda').manual_seed(5 + rank + trial)) * 0.5).bfloat16()
            y = torch.empty(t, h, device='cuda', dtype=torch.bfloat16)
            deep_gemm.mxfp4_mega_moe_sm120(y, x, ids, wts, l1, l2, buf, activation_clamp=LIMIT)
            torch.cuda.synchronize()
            ref = reference(x, ids, wts, lambda e: expert_codes(e, h, i), i, True)
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            worst = max(worst, rel)
            print(f'rank{rank} T={t} err={buf.error()} rel_l2={rel:.2e}', flush=True)
        dist.barrier()
        if rank == 0:
            print('EP VERIFY', 'PASS' if worst < 2e-2 else 'FAIL', flush=True)
    else:
        h, i, e = 4096, 2048, 256
        el = e // world
        l1, l2 = random_weights(el, h, i, 20261022 + rank)
        buf = deep_gemm.get_symm_buffer_for_mega_moe_sm120(group, e, 64, TOPK, h, i)
        for t in [int(a) for a in args]:
            ids, wts = routing(t, e, TOPK, 20261022 + 100 * t + rank)
            if os.environ.get('SAVE_ROUTING'):  # same routing for sm120_6000d/bench/marlin_ep4_bench.py (ROUTING_DIR)
                torch.save({'ids': ids.cpu(), 'w': wts.cpu()}, os.path.join(os.environ['SAVE_ROUTING'], f'routing_T{t}_r{rank}.pt'))
            x = (torch.randn(t, h, device='cuda') * 0.1).bfloat16()
            y = torch.empty(t, h, device='cuda', dtype=torch.bfloat16)
            run = lambda: deep_gemm.mxfp4_mega_moe_sm120(y, x, ids, wts, l1, l2, buf, activation_clamp=LIMIT)
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            dist.barrier()
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            runs = []
            for _ in range(5):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(100):
                    graph.replay()
                b.record()
                b.synchronize()
                runs.append(a.elapsed_time(b) * 1e3 / 100)
                dist.barrier()
            us = torch.tensor([statistics.median(runs), float(buf.error())], device='cuda')
            allus = [torch.zeros_like(us) for _ in range(world)]
            dist.all_gather(allus, us)
            if rank == 0:
                v = [round(a[0].item(), 1) for a in allus]
                print(f'EP{world} per_rank_tokens {t:3d} us_per_moe max {max(v):7.1f} ranks {v} '
                      f'err {[int(a[1].item()) for a in allus]}', flush=True)
            del graph
            torch.cuda.synchronize()
            dist.barrier()
    dist.destroy_process_group()


def tp(mode, args):
    if mode == 'tpverify':
        h, i, e = 512, 256, 32  # I >= 256: FC2 K must cover one unroll (4 x K64)
        experts = [expert_codes(j, h, i) for j in range(e)]
        l1, l2 = kernel_weights(experts)
        deep_gemm.init_mxfp4_moe_tp_sm120(e, TOPK, h, i)
        worst = 0.0
        for trial, t in enumerate([1, 2, 5, 8, 13, 40, 100]):
            ids, wts = routing(t, e, TOPK, 99 + trial)
            if trial % 2 == 1:
                ids[-1, :3] = -1
            x = (torch.randn(t, h, device='cuda', generator=torch.Generator(device='cuda').manual_seed(trial)) * 0.5).bfloat16()
            y = torch.empty(t, h, device='cuda', dtype=torch.bfloat16)
            for _ in range(2):  # the second call checks the scratch was re-zeroed
                deep_gemm.mxfp4_moe_tp_sm120(y, x, ids, wts, l1, l2, activation_clamp=LIMIT)
            torch.cuda.synchronize()
            ref = reference(x, ids, wts, lambda j: experts[j], i, False)
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            worst = max(worst, rel)
            print(f'tp T={t} err={deep_gemm.mxfp4_moe_tp_sm120_error()} rel_l2={rel:.2e}', flush=True)
        print('TP VERIFY', 'PASS' if worst < 2e-2 else 'FAIL', flush=True)
    else:
        tps = int(os.environ.get('TPS', '8'))
        h, i, e = 4096, 2048 // tps, 256
        l1, l2 = random_weights(e, h, i, 7)
        deep_gemm.init_mxfp4_moe_tp_sm120(e, TOPK, h, i)
        flush = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')
        for t in [int(a) for a in args]:
            ids, wts = routing(t, e, TOPK, 4242 + t)
            x = (torch.randn(t, h, device='cuda') * 0.1).bfloat16()
            y = torch.empty(t, h, device='cuda', dtype=torch.bfloat16)
            run = lambda: deep_gemm.mxfp4_moe_tp_sm120(y, x, ids, wts, l1, l2, activation_clamp=LIMIT)
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            ts = []
            for n in range(40):  # cold L2 every call
                flush.fill_(n)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graph.replay()
                b.record()
                b.synchronize()
                if n >= 10:
                    ts.append(a.elapsed_time(b) * 1e3)
            print(f'TP{tps} tokens {t:3d} us_per_moe {statistics.median(ts):7.1f} '
                  f'err {deep_gemm.mxfp4_moe_tp_sm120_error()}', flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    (tp if mode in ('tp', 'tpverify') else ep)(mode, sys.argv[2:])

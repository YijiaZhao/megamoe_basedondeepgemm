# In-kernel router of the SM120 MegaMoE (fp4_mega_moe_sm120 with router_logits) against SGLang's DeepSeek-V4.1
# routing in front of the same kernel, one MoE layer, EP over the ranks of this host:
#   A (SGLang): gate GEMM (tiny_gemm_bf16 up to 16 rows, cuBLAS bf16 x bf16 -> fp32 beyond) -> moe_fused_gate
#     (sqrtsoftplus via log1p, bias / bias_vl for image tokens, top-6, renormalize + 1e-20, x 1.5) -> EPLB 'dynamic'
#     (row t takes replica t % n) -> MegaMoE
#   B (fused): the same gate GEMM -> MegaMoE routing in its first phase
# The checkpoint's gate weight and biases of one layer, the EPLB placement eplb_map_r8_ep8.pt; random hidden states
# and random MXFP4 expert weights. Checks that both give the same routes and the same output bits, then times both (one
# CUDA graph each, 256 MB written before every replay, ranks released by a barrier, median of 30, slowest rank), and
# the MegaMoE alone with given routes.
#   torchrun --nproc_per_node=8 tests/bench_router_fused.py <model dir> <placement .pt> <transport> [tokens per rank ...]
import importlib.util
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mega_moe_sm120_fp4 import LIMIT, SHAPES, fp4  # noqa: E402

from safetensors import safe_open  # noqa: E402
from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16  # noqa: E402
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate  # noqa: E402

LAYER, EPS, SCALE = 12, 1e-20, 1.5


def gate_tensors(model):
    idx = json.load(open(os.path.join(model, 'model.safetensors.index.json')))['weight_map']
    out = []
    for name in ('weight', 'bias', 'bias_vl'):
        key = f'layers.{LAYER}.ffn.gate.{name}'
        with safe_open(os.path.join(model, idx[key]), 'pt', device=f'cuda:{torch.cuda.current_device()}') as f:
            out.append(f.get_tensor(key).contiguous())
    cfg = json.load(open(os.path.join(model, 'config.json')))
    return out[0], out[1].float(), out[2].float(), int(cfg.get('image_token_id', cfg['text_config'].get('image_token_id', -1)))


def main():
    model, placement, transport = sys.argv[1:4]
    ts = [int(v) for v in sys.argv[4:]] or [96]
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    rank, world = dist.get_rank(), dist.get_world_size()
    h, i, _, topk = SHAPES['dsv41']
    p2l = torch.load(placement)['physical_to_logical_map'][LAYER].long().cuda()  # [physical]
    e = p2l.numel()
    el = e // world
    nlog = 384
    reps = [[p for p in range(e) if p2l[p] == g] for g in range(nlog)]
    rmax = max(len(r) for r in reps)
    l2p = torch.full((nlog, rmax), -1, dtype=torch.int32, device='cuda')
    for g, r in enumerate(reps):
        l2p[g, :len(r)] = torch.tensor(r, dtype=torch.int32)
    nvalid = (l2p >= 0).sum(1)
    gw, bias, bias_vl, img = gate_tensors(model)
    gen = torch.Generator(device='cuda').manual_seed(20261119 + rank)
    l1 = (torch.randint(0, 256, (el, 2 * i, h // 2), device='cuda', dtype=torch.uint8, generator=gen),
          torch.full((el, 2 * i, h // 32), 124, device='cuda', dtype=torch.uint8))
    l2 = (torch.randint(0, 256, (el, h, i // 2), device='cuda', dtype=torch.uint8, generator=gen),
          torch.full((el, h, i // 32), 124, device='cuda', dtype=torch.uint8))
    buf = fp4.Sm120Fp4MegaMoeBuffer(dist.group.WORLD, 'w4a8_rm', e, max(max(ts), 128), topk, h, i, transport)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')

    for t in ts:
        x = (torch.randn(t, h, device='cuda', generator=gen) * 0.5).bfloat16()
        ids_in = torch.randint(0, 100000, (t,), device='cuda', generator=gen)  # text tokens (bias, not bias_vl)
        gate = (lambda: tiny_gemm_bf16(x, gw, out_dtype=torch.float32, max_m=16)) if t <= 16 else \
               (lambda: torch.mm(x, gw.t(), out_dtype=torch.float32))
        rows = torch.arange(t, device='cuda')[:, None]
        ya, yb, yc = (torch.empty(t, h, device='cuda', dtype=torch.bfloat16) for _ in range(3))
        ids_b = torch.empty(t, topk, dtype=torch.int32, device='cuda')
        w_b = torch.empty(t, topk, dtype=torch.float32, device='cuda')
        st = {}

        def path_a():
            wts, ids = moe_fused_gate(gate(), bias, topk=topk, scoring_func='sqrtsoftplus', renormalize=True,
                                      routed_scaling_factor=SCALE, apply_routed_scaling_factor_on_output=True,
                                      renormalize_epsilon=EPS, sqrtsoftplus_log1p=True, bias_alt=bias_vl,
                                      input_ids=ids_in, bias_alt_token_id=img)
            il = ids.long()
            phys = l2p[il, rows % nvalid[il]].int()
            st['a'] = (phys, wts)
            fp4.fp4_mega_moe_sm120(ya, x, phys, wts, l1, l2, buf, None, activation_clamp=LIMIT)

        def path_b():
            fp4.fp4_mega_moe_sm120(yb, x, ids_b, w_b, l1, l2, buf, None, activation_clamp=LIMIT,
                                   router_logits=gate(), router_bias=bias, router_bias_alt=bias_vl, input_ids=ids_in,
                                   alt_token_id=img, logical_to_all_physical=l2p, renormalize=True,
                                   renormalize_epsilon=EPS, apply_routed_scaling=True, routed_scaling_factor=SCALE)

        path_a()
        path_b()
        torch.cuda.synchronize()
        phys, wts = st['a']
        same_ids = torch.equal(phys, ids_b)
        same_w = torch.equal(wts, w_b)
        same_y = torch.equal(ya, yb)
        assert buf.error() == 0, buf.error()
        fixed_ids, fixed_w = phys.clone(), wts.clone()

        def path_c():  # the MegaMoE alone, routes given
            fp4.fp4_mega_moe_sm120(yc, x, fixed_ids, fixed_w, l1, l2, buf, None, activation_clamp=LIMIT)

        res = {}
        for name, fn in (('A SGLang router + MegaMoE', path_a), ('B fused router', path_b), ('C MegaMoE alone', path_c)):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            dist.barrier()
            tt = []
            for k in range(32):
                flush.fill_(k)
                torch.cuda.synchronize()
                dist.barrier()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                b.synchronize()
                tt.append(a.elapsed_time(b) * 1e3)
            us = torch.tensor([statistics.median(tt[2:])], device='cuda')
            allus = [torch.zeros_like(us) for _ in range(world)]
            dist.all_gather(allus, us)
            res[name] = max(v.item() for v in allus)
            del g
        flags = torch.tensor([same_ids, same_w, same_y], device='cuda', dtype=torch.int32)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        diffs = torch.tensor([(wts - w_b).abs().max().item() if t else 0.0,
                              ((ya.float() - yb.float()).norm() / ya.float().norm().clamp_min(1e-30)).item() if t else 0.0,
                              ((w_b.view(torch.int32) - wts.view(torch.int32)).abs().max().item()) if t else 0.0],
                             device='cuda')
        dist.all_reduce(diffs, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f'T/rank {t}: routes equal {bool(flags[0])}, weights equal {bool(flags[1])}, output bits equal '
                  f'{bool(flags[2])} (max weight diff {diffs[0].item():.1e}, {int(diffs[2].item())} ulp; output rel L2 '
                  f'{diffs[1].item():.1e}) | ' + ' | '.join(f'{k} {v:.1f} us' for k, v in res.items()), flush=True)
    dist.barrier()
    buf.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

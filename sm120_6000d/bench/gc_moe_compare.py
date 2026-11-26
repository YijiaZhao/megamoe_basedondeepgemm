"""gc-moe (github.com/Bruce-Lee-LY/gc-moe, SM120 `sm120_fp8_fp4_gc_moe`) vs this repo's SM120 MegaMoE: same weights,
same routing, same timing. Used for the "vs gc-moe" tables of sm120_6000d/W4A4.md.

GC_ROOT=<gc-moe checkout> MM_ROOT=<this repo> OURS_FMT=w4a8|mxfp4|nvfp4 \
torchrun --nproc_per_node=4 gc_moe_compare.py --impl gc|ours --shape glm53flash|dsv41|dsv4flash|readme \
    [--transport p2p] [--tokens 1,2,...] [--routings balanced] --out <json>
Per replay: zero 256 MiB (cold L2) -> on-stream all_reduce of one int -> event start -> graph replay -> event end.
"""
import argparse, importlib.util, json, os, shutil, statistics, subprocess, sys, tempfile
import torch
import torch.distributed as dist

GC_ROOT = os.environ.get("GC_ROOT", "/ws/gc-moe")
MM_ROOT = os.environ.get("MM_ROOT", "/ws/mm")
# gc-moe's test helpers import deep_gemm.utils.math only; give them this repo's copy without importing deep_gemm itself
_SHIM = tempfile.mkdtemp()
os.makedirs(os.path.join(_SHIM, "deep_gemm", "utils"))
open(os.path.join(_SHIM, "deep_gemm", "__init__.py"), "w").close()
open(os.path.join(_SHIM, "deep_gemm", "utils", "__init__.py"), "w").write("from .math import *\n")
shutil.copy(os.path.join(MM_ROOT, "deep_gemm", "utils", "math.py"), os.path.join(_SHIM, "deep_gemm", "utils", "math.py"))
sys.path.insert(0, _SHIM)
sys.path.insert(0, GC_ROOT)
from tests.test_sm120_gc_moe import (build_tokens, cast_grouped_weights_to_fp4, cast_tokens_to_fp8,  # noqa: E402
                                     pack_ue8m0_weight_scales, dequantize_mxfp4)

SHAPES = {"readme": (3072, 1280, 384, 8), "dsv41": (5120, 2304, 384, 6), "glm53flash": (4096, 2048, 256, 8), "dsv4flash": (4096, 2048, 256, 6)}  # H, I, E, top-k
CLAMP = 10.0


def load_ours():
    path = os.path.join(MM_ROOT, 'deep_gemm', 'mega', 'sm120_fp4.py')
    spec = importlib.util.spec_from_file_location('sm120_fp4', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def foreign_apps(world):
    # every compute process on the box; only this job's `world` ranks may be present
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return out if len(out) > world else []


def balanced_ids(T, rank, world, E, topk):
    # every expert gets the same number of copies (+-1): rank s*world/topk + g%(world/topk), then round-robin over local experts
    el = E // world
    stride = world // topk if world % topk == 0 else 0
    dest = (lambda g, s: s * stride + g % stride) if stride else (lambda g, s: (g * topk + s) % world)
    ids = torch.empty((T, topk), dtype=torch.int64)
    arrived = [0] * world
    for g in range(T * world):
        for s in range(topk):
            d = dest(g, s)
            if g // T == rank: ids[g % T, s] = d * el + arrived[d] % el
            arrived[d] += 1
    return ids.cuda(), torch.full((T, topk), 1.0 / topk, device="cuda")


def reference(x, ids, w, deq13, deq2, rank, world, el, inter):
    """bf16-weights fp32 reference over all ranks' tokens, this rank's experts, summed by all_reduce."""
    T = x.shape[0]
    xs = [torch.empty_like(x) for _ in range(world)]; dist.all_gather(xs, x)
    idss = [torch.empty_like(ids) for _ in range(world)]; dist.all_gather(idss, ids)
    ws = [torch.empty_like(w) for _ in range(world)]; dist.all_gather(ws, w)
    X, I, W = torch.cat(xs).float(), torch.cat(idss), torch.cat(ws)
    out = torch.zeros_like(X)
    for le in range(el):
        ge = rank * el + le
        tok, slot = (I == ge).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = X[tok] @ deq13[le].float().t()
        gate, up = h[:, :inter], h[:, inter:]
        act = torch.nn.functional.silu(gate.clamp(max=CLAMP)) * up.clamp(-CLAMP, CLAMP)
        out.index_add_(0, tok, (act @ deq2[le].float().t()) * W[tok, slot].unsqueeze(1))
    dist.all_reduce(out)
    return out[rank * T:(rank + 1) * T]


def act_amax(tokens, H, I, E, K, rank, world, el, p13, c13):
    """amax over all ranks and all token counts of the FC1 input and of the SwiGLU output under balanced routing."""
    a1 = a2 = 0.0
    deq13 = dequantize_mxfp4(p13, c13)
    for t in tokens:
        x, _, _ = build_tokens(t, H, E, K, rank, torch.device('cuda', torch.cuda.current_device()))
        ids, _ = balanced_ids(t, rank, world, E, K)
        a1 = max(a1, float(x.float().abs().max()))
        xs = [torch.empty_like(x) for _ in range(world)]; dist.all_gather(xs, x)
        idss = [torch.empty_like(ids) for _ in range(world)]; dist.all_gather(idss, ids)
        X, Ids = torch.cat(xs).float(), torch.cat(idss)
        for le in range(el):
            tok, _ = (Ids == rank * el + le).nonzero(as_tuple=True)
            if tok.numel():
                h = X[tok] @ deq13[le].float().t()
                act = torch.nn.functional.silu(h[:, :I].clamp(max=CLAMP)) * h[:, I:].clamp(-CLAMP, CLAMP)
                a2 = max(a2, float(act.abs().max()))
    del deq13
    m = torch.tensor([a1, a2], device='cuda'); dist.all_reduce(m, op=dist.ReduceOp.MAX)
    torch.cuda.empty_cache()
    return float(m[0]), float(m[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', required=True, choices=['gc', 'ours'])
    ap.add_argument('--shape', required=True, choices=list(SHAPES))
    ap.add_argument('--transport', default='p2p')
    ap.add_argument('--tokens', default='1,2,4,8,16,32,64,128,192,256,320,384,448,512,576,640,768,1024')
    ap.add_argument('--routings', default='balanced')
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    rank, world = dist.get_rank(), dist.get_world_size()
    H, I, E, K = SHAPES[a.shape]
    el = E // world
    tokens = [int(t) for t in a.tokens.split(',')]
    cap = max(tokens)

    # identical weights for both impls: bf16 randn*0.05 (seed 7+rank) -> MXFP4 (per-32 UE8M0)
    g = torch.Generator(device='cuda').manual_seed(7 + rank)
    w13 = torch.randn((el, 2 * I, H), generator=g, device='cuda', dtype=torch.bfloat16) * 0.05
    p13, c13 = cast_grouped_weights_to_fp4(w13); del w13
    w2 = torch.randn((el, H, I), generator=g, device='cuda', dtype=torch.bfloat16) * 0.05
    p2, c2 = cast_grouped_weights_to_fp4(w2); del w2
    torch.cuda.empty_cache()

    if a.impl == 'gc':
        import gc_moe
        gw = gc_moe.transform_weights_for_gc_moe_sm120((p13, pack_ue8m0_weight_scales(c13)),
                                                       (p2, pack_ue8m0_weight_scales(c2)))
        buf = gc_moe.get_symm_buffer_for_gc_moe_sm120(dist.group.WORLD, E, cap, K, H, I, activation='swiglu',
                                                      num_shared_experts=0)
        commit = subprocess.run(['git', '-C', GC_ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    else:
        fp4 = load_ours()
        s13, s2, GS = c13, c2, None
        if os.environ.get("OURS_FMT") == "nvfp4":
            # the same weights as NVFP4: each power-of-two UE8M0 scale as an E4M3 scale on both of its 16-value halves,
            # weight global scale 1, so they dequantize exactly as the MXFP4 ones; activation global scales from the
            # inputs' amax (FC1: the tokens, FC2: the SwiGLU output) over all ranks, amax / (6 * 448)
            def e4m3(c):
                v = torch.exp2(c.float() - 127.0)
                b = v.to(torch.float8_e4m3fn)
                assert torch.equal(b.float(), v), 'a weight scale is out of E4M3 range'
                return b.view(torch.uint8).repeat_interleave(2, dim=-1).contiguous()
            s13, s2 = e4m3(c13), e4m3(c2)
            a1, a2 = act_amax(tokens, H, I, E, K, rank, world, el, p13, c13)
            GS = fp4.nvfp4_global_scales(a1 / 2688, a2 / 2688, torch.ones(el, device='cuda'), torch.ones(el, device='cuda'))
        l1, l2 = fp4.transform_fp4_weights_for_mega_moe_sm120((p13.view(torch.uint8), s13), (p2.view(torch.uint8), s2),
                                                               os.environ.get("OURS_FMT", "w4a8"))
        buf = fp4.Sm120Fp4MegaMoeBuffer(dist.group.WORLD, os.environ.get("OURS_FMT", "w4a8"), E, cap, K, H, I, a.transport)
        commit = subprocess.run(['git', '-C', MM_ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    torch.cuda.synchronize()

    flush = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')
    one = torch.ones(1, dtype=torch.int32, device='cuda')
    rows = []
    for routing in a.routings.split(','):
        for T in tokens:
            x, rid, rw = build_tokens(T, H, E, K, rank, torch.device('cuda', local))
            if routing == 'balanced':
                rid, rw = balanced_ids(T, rank, world, E, K)
            y = torch.empty((T, H), dtype=torch.bfloat16, device='cuda')
            if a.impl == 'gc':
                xq, xsf = cast_tokens_to_fp8(x)
                buf.x[:T].copy_(xq); buf.x_sf[:T].copy_(xsf)
                buf.topk_idx[:T].copy_(rid); buf.topk_weights[:T].copy_(rw)
                fn = lambda: gc_moe.sm120_fp8_fp4_gc_moe(y, *gw, buf, recipe=(1, 1, 32), activation='swiglu',
                                                         activation_clamp=CLAMP, fast_math=True,
                                                         num_tokens_bound=max(1, T))
            else:
                ids32, rwf = rid.to(torch.int32), rw.float()
                fn = lambda: fp4.fp4_mega_moe_sm120(y, x, ids32, rwf, l1, l2, buf, GS, activation_clamp=CLAMP)
            for _ in range(3):
                fn()
            torch.cuda.synchronize(); dist.barrier()
            cos = None
            if T == 8:
                deq13 = dequantize_mxfp4(p13, c13); deq2 = dequantize_mxfp4(p2, c2)
                ref = reference(x, rid, rw.float(), deq13, deq2, rank, world, el, I)
                del deq13, deq2
                c = torch.nn.functional.cosine_similarity(ref.flatten(), y.float().flatten(), dim=0)
                dist.all_reduce(c, op=dist.ReduceOp.MIN)
                cos = float(c)
                torch.cuda.empty_cache()
            graph = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize(); dist.barrier()
            # before every point: no compute process on the box besides this job's ranks (util is checked before each run)
            if rank == 0:
                extra = foreign_apps(world)
                assert not extra, f'foreign GPU processes before point: {extra}'
            dist.barrier()
            ts = []
            for it in range(a.warmup + a.iters):
                flush.zero_()
                dist.all_reduce(one)
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record(); graph.replay(); en.record()
                torch.cuda.synchronize()
                if it >= a.warmup:
                    ts.append(st.elapsed_time(en) * 1e3)
            med = torch.tensor([statistics.median(ts)], device='cuda')
            meds = [torch.empty_like(med) for _ in range(world)]; dist.all_gather(meds, med)
            del graph
            row = dict(routing=routing, tokens_per_rank=T, global_m=T * world, gpu0_median_us=float(meds[0]),
                       max_rank_median_us=max(float(m) for m in meds), per_rank_median_us=[float(m) for m in meds],
                       cos_vs_ref=cos)
            if a.impl == 'ours':
                assert buf.error() == 0, f'kernel error word {buf.error()}'
            rows.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
    if rank == 0:
        with open(a.out, 'w') as f:
            json.dump(dict(impl=a.impl, shape=a.shape, world=world, transport=a.transport if a.impl == 'ours' else 'p2p (symmetric memory)',
                           commit=commit, torch=torch.__version__, rows=rows), f, indent=1)
    dist.barrier()
    if a.impl == 'gc':
        buf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

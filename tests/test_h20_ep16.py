"""H20 fused MegaMoE: EP8 (one node) vs two-node EP16 (deep_gemm/mega/ep16.py), same model shape.

E384 / H3072 / I1280 / top-8, T tokens per rank, forced-balanced routing:
  EP8 : token g (= rank * T + t) slot s -> rank s, expert s * 48 + (g + 7 s) % 48 (one route per rank);
  EP16: token g slot s -> rank (g + 2 s) % 16, expert rank * 24 + (g + 7 s) % 24 (8 distinct ranks per
        token, 4 on each node; every rank receives 8 T routes as in EP8).

verify: exact-quantised reference (FP8 / int8 activations, dequantised weights, quantised intermediate),
        cos / norm of y against it on every rank.
bench : one CUDA graph per call (EP8: the fused kernel; EP16: dispatch exchange + node kernel + combine
        exchange + add; EP16 also the node kernel alone; EP16 --transport gin: the one fused kernel, the
        cross-node exchange inside it, inputs already in the buffer as for EP8), cold L2 (256 MiB zero) and an on-stream
        all-reduce before every replay, CUDA-event span of the replay; median over the replays per rank,
        reported for rank 0 and as the max over the ranks.

torchrun --nnodes 2 --nproc-per-node 8 ... tests/test_h20_ep16.py --mode ep16
torchrun --nnodes 2 --nproc-per-node 8 ... tests/test_h20_ep16.py --mode ep16 --transport gin
torchrun --nproc-per-node 8 tests/test_h20_ep16.py --mode ep8
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if "--transport" in sys.argv and sys.argv[sys.argv.index("--transport") + 1] == "gin":
    import nvidia.nccl  # the fused kernel includes NCCL's device API (before the first JIT build)
    os.environ["DG_JIT_EXTRA_FLAGS"] = "-I" + os.path.join(list(nvidia.nccl.__path__)[0], "include")

import deep_gemm
from deep_gemm.mega.ep16 import H20EP16
from deep_gemm.quantization_mxfp4_fused import (
    dequantize_mxfp4_kernel_exact_fp32,
    mxfp4_row_reference_exponent,
    quantize_to_mxfp4 as quantize_to_mxfp4_fused,
)
from deep_gemm.quantization_qoq_fused import (
    dequantize_qoq_to_fp32,
    int8_bytes_to_float,
    per_token_cast_to_int8,
    quantize_to_qoq,
)
from deep_gemm.utils import per_token_cast_to_fp8


def _silu(x):
    return x * torch.sigmoid(x)


def _quantize_intermediate(x, qoq):
    groups = x.view(-1, 64)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    if qoq:
        scale = (amax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
        return ((groups / scale).round().clamp(-127, 127) * scale).view_as(x)
    scaled_amax = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.exp2(torch.ceil(torch.log2(scaled_amax)))
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).view_as(x)


def balanced_routing(mode, rank, t, topk, local_experts, world):
    g = rank * t + torch.arange(t, dtype=torch.int64, device="cuda")[:, None]
    s = torch.arange(topk, dtype=torch.int64, device="cuda")[None, :]
    if mode == "ep8":
        dst = s.expand(t, topk)
    else:
        dst = (g + 2 * s) % world
    idx = dst * local_experts + (g + 7 * s) % local_experts
    w = torch.full((t, topk), 1.0 / topk, dtype=torch.float32, device="cuda")
    return idx.contiguous(), w


# slot: (label, min-slot) -- see the slot list at the top of the fused kernel body
STAMP_SLOTS = {1: ("routing done (barrier #1)", False), 2: ("pull done", False),
               48: ("input put issued", False), 49: ("peer input rows seen", False), 50: ("peer rows copied in", False),
               3: ("first math task", True), 6: ("max token arrivals (fine combine)", False),
               53: ("first combined peer row put", True), 54: ("last combined peer row put", False),
               51: ("first own-row wait for peer partial", True), 52: ("last peer partial seen", False),
               7: ("combine end", False)}


def stamp_phases(ep, weights, args, y, flush, align, rank, world, replays=20):
    # One graph: reset the stamps, run the kernel. Per replay every slot minus the kernel entry (slot 0);
    # median over the replays, printed for every rank (globaltimer is per GPU: only within-rank deltas).
    stamps = torch.zeros(64, dtype=torch.int64, device="cuda")
    init = torch.zeros(64, dtype=torch.int64, device="cuda")
    for slot in [0] + [k for k, (_, is_min) in STAMP_SLOTS.items() if is_min]:
        init[slot] = 2 ** 63 - 1
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            stamps.copy_(init)
            ep.run(y, None, None, weights, activation_clamp=args.activation_clamp, phase_stamps=stamps)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        stamps.copy_(init)
        ep.run(y, None, None, weights, activation_clamp=args.activation_clamp, phase_stamps=stamps)
    torch.cuda.synchronize()
    dist.barrier()
    rows = []
    for _ in range(replays):
        flush.zero_()
        dist.all_reduce(align)
        g.replay()
        torch.cuda.synchronize()
        rows.append(stamps.clone())
    st = torch.stack(rows)
    slots = list(STAMP_SLOTS)
    rel = (st[:, slots] - st[:, :1]).double() / 1000.0
    med = rel.median(dim=0).values.float()
    allm = torch.empty(world, len(slots), device="cuda")
    dist.all_gather_into_tensor(allm, med.contiguous())
    if rank == 0:
        print("STAMPS us after kernel entry, median of", replays, "replays; columns = ranks", flush=True)
        for j, slot in enumerate(slots):
            vals = " ".join(f"{v:7.1f}" for v in allm[:, j].tolist())
            print(f"STAMP {slot:2d} {STAMP_SLOTS[slot][0]:<38s} {vals}", flush=True)
    del g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ep8", "ep16"], required=True)
    p.add_argument("--quant", choices=["mxfp4", "qoq"], default="mxfp4")
    p.add_argument("--tokens", type=int, default=8, help="tokens per rank")
    p.add_argument("--hidden", type=int, default=3072)
    p.add_argument("--intermediate", type=int, default=1280)
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--weight-scale", type=float, default=0.05)
    p.add_argument("--activation-clamp", type=float, default=10.0)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--overlap", action="store_true", help="EP16: own-row kernel runs during the dispatch exchange")
    p.add_argument("--stages", action="store_true", help="EP16: also time every stage in its own graph")
    p.add_argument("--transport", choices=["nccl", "gin"], default="nccl",
                   help="EP16 cross-node exchange: NCCL send/recv around the kernel, or NCCL GIN inside it")
    p.add_argument("--stamps", action="store_true",
                   help="EP16 gin: in-kernel phase timestamps (globaltimer, relative to kernel entry) per replay")
    args = p.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    expected = 8 if args.mode == "ep8" else 16
    assert world == expected, f"{args.mode} needs {expected} ranks, got {world}"
    world_group = dist.group.WORLD
    if args.mode == "ep16":
        node_groups = [dist.new_group(list(range(n * 8, n * 8 + 8))) for n in range(2)]
        node_group = node_groups[rank // 8]
    else:
        node_group = world_group
    qoq = args.quant == "qoq"
    t, H, I, topk = args.tokens, args.hidden, args.intermediate, args.topk
    local_experts = args.experts // world

    torch.manual_seed(20260805 + rank)
    w1_bf = torch.randn(local_experts, 2 * I, H, device="cuda", dtype=torch.bfloat16) * args.weight_scale
    w2_bf = torch.randn(local_experts, H, I, device="cuda", dtype=torch.bfloat16) * args.weight_scale
    if qoq:
        x_quant = lambda x: per_token_cast_to_int8(x, gran_k=128)  # noqa: E731
        q1, q2 = quantize_to_qoq(w1_bf), quantize_to_qoq(w2_bf)
        d1, d2 = dequantize_qoq_to_fp32(*q1), dequantize_qoq_to_fp32(*q2)
        weights = deep_gemm.transform_qoq_weights_for_mega_moe_fused(q1, q2)
    else:
        x_quant = lambda x: per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128)  # noqa: E731
        q1, q2 = quantize_to_mxfp4_fused(w1_bf), quantize_to_mxfp4_fused(w2_bf)
        d1 = dequantize_mxfp4_kernel_exact_fp32(q1[0], q1[1], mxfp4_row_reference_exponent(q1[1]))
        d2 = dequantize_mxfp4_kernel_exact_fp32(q2[0], q2[1], mxfp4_row_reference_exponent(q2[1]))
        weights = deep_gemm.transform_mxfp4_weights_for_mega_moe_fused(q1, q2)
    del w1_bf, w2_bf
    torch.cuda.empty_cache()
    kernel = deep_gemm.qoq_mega_moe_fused if qoq else deep_gemm.mxfp4_mega_moe_fused

    topk_idx, topk_w = balanced_routing(args.mode, rank, t, topk, local_experts, world)
    if args.mode == "ep16":
        ep = H20EP16(world_group, node_group, args.experts, t, topk, H, I, quant=args.quant, overlap=args.overlap,
                     transport=args.transport)
        gin = args.transport == "gin"

        def forward(y, xq, xs):
            ep(y, xq, xs, topk_idx, topk_w, weights, activation_clamp=args.activation_clamp)

        def ep16_run(y, xq, xs):
            # x / x_sf (gin: and the top-k) already in buffer rows [0, T) (as for the EP8 kernel-only graph)
            ep.run(y, None if gin else topk_idx, None if gin else topk_w, weights, activation_clamp=args.activation_clamp)

        def wait(works):
            for w in works:
                w.wait()

        stages = [
            ("stage pack (route split + pack)", lambda y, xq, xs: ep.pack(topk_idx, topk_w)),
            ("stage dispatch exchange (NCCL)", lambda y, xq, xs: wait(ep.dispatch())),
            ("stage unpack", lambda y, xq, xs: ep.unpack()),
            ("stage combine exchange (NCCL)", lambda y, xq, xs: wait(ep.combine())),
            ("stage add", lambda y, xq, xs: torch.add(ep.y_own, ep.y_back, out=y)),
        ]
        if gin:
            stages, node_kernel_only = [], None
        elif args.overlap:
            stages[3:3] = [
                ("stage node kernel, own rows (T)", lambda y, xq, xs: ep.node_kernel(
                    ep.y_own, weights, ep.buffer, args.activation_clamp)),
                ("stage node kernel, peer rows (T)", lambda y, xq, xs: ep.node_kernel(
                    ep.y_peer, weights, ep.peer_buffer, args.activation_clamp)),
            ]
            node_kernel_only = None
        else:
            def node_kernel_only(y, xq, xs):
                kernel(ep.y_node, *weights, ep.buffer, activation_clamp=args.activation_clamp)
    else:
        buffer = deep_gemm.get_fused_symm_buffer_for_mega_moe(node_group, args.experts, t, topk, H, I)

        def forward(y, xq, xs):
            buffer.x[:t].copy_(xq)
            buffer.x_sf[:t].copy_(xs)
            buffer.topk_idx[:t].copy_(topk_idx)
            buffer.topk_weights[:t].copy_(topk_w)
            kernel(y, *weights, buffer, activation_clamp=args.activation_clamp)

        def node_kernel_only(y, xq, xs):
            kernel(y, *weights, buffer, activation_clamp=args.activation_clamp)

    def x_reference(xq, xs):
        if qoq:
            return int8_bytes_to_float(xq) * xs[:, :1]
        return (xq.float().view(t, H // 128, 128) * xs.float().unsqueeze(-1)).view(t, H)

    y = torch.zeros(t, H, device="cuda", dtype=torch.bfloat16)
    ok = True
    if not args.skip_verify:
        for seed in range(args.seeds):
            torch.manual_seed(1000 + seed * 7919 + rank * 1000003)
            x_bf = torch.randn(t, H, device="cuda", dtype=torch.bfloat16)
            xq, xs = x_quant(x_bf)
            y.zero_()
            forward(y, xq, xs)
            torch.cuda.synchronize()
            G = t * world
            x_all = torch.empty(G, H, device="cuda")
            idx_all = torch.empty(G, topk, device="cuda", dtype=torch.int64)
            w_all = torch.empty(G, topk, device="cuda")
            dist.all_gather_into_tensor(x_all, x_reference(xq, xs).contiguous())
            dist.all_gather_into_tensor(idx_all, topk_idx)
            dist.all_gather_into_tensor(w_all, topk_w)
            routes = torch.zeros(G, topk, H, device="cuda")
            for g in range(G):
                for s in range(topk):
                    e = int(idx_all[g, s])
                    if e < 0 or e // local_experts != rank:
                        continue
                    le = e % local_experts
                    l1 = d1[le].float() @ x_all[g]
                    gate, up = l1[:I].clamp(max=args.activation_clamp), l1[I:].clamp(-args.activation_clamp, args.activation_clamp)
                    mid = _quantize_intermediate(_silu(gate) * up * float(w_all[g, s]), qoq)
                    routes[g, s] = (d2[le].float() @ mid).to(torch.bfloat16).float()
            dist.all_reduce(routes)
            ref = routes.sum(dim=1).to(torch.bfloat16).float()[rank * t:(rank + 1) * t]
            cos = torch.nn.functional.cosine_similarity(y.float(), ref, dim=-1).min()
            nr = torch.stack([(y.float().square().sum() / ref.square().sum().clamp_min(1e-30)).sqrt(),
                              (y.float() - ref).abs().max()])
            dist.all_reduce(cos, op=dist.ReduceOp.MIN)
            nr_max = nr.clone()
            dist.all_reduce(nr_max, op=dist.ReduceOp.MAX)
            nr_min = nr.clone()
            dist.all_reduce(nr_min, op=dist.ReduceOp.MIN)
            good = cos.item() >= 0.999 and 0.99 <= nr_min[0].item() and nr_max[0].item() <= 1.01
            ok &= good
            if rank == 0:
                print(f"VERIFY mode={args.mode} quant={args.quant} T={t} M={t * world} seed={seed} "
                      f"cos_min={cos.item():.6f} norm_ratio=[{nr_min[0].item():.4f},{nr_max[0].item():.4f}] "
                      f"max_abs={nr_max[1].item():.4g} -> {'PASS' if good else 'FAIL'}", flush=True)

    if not args.skip_bench:
        torch.manual_seed(42 + rank)
        xq, xs = x_quant(torch.randn(t, H, device="cuda", dtype=torch.bfloat16))
        flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
        align = torch.zeros(1, device="cuda")

        def bench(fn, label):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn(y, xq, xs)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(y, xq, xs)
            torch.cuda.synchronize()
            dist.barrier()
            evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iters + 3)]
            for a, b in evs:
                flush.zero_()
                dist.all_reduce(align)
                a.record()
                g.replay()
                b.record()
            torch.cuda.synchronize()
            us = torch.tensor(sorted(a.elapsed_time(b) * 1000 for a, b in evs[3:]), device="cuda")
            med = us[len(us) // 2].clone()
            med_all = torch.empty(world, device="cuda")
            dist.all_gather_into_tensor(med_all, med.view(1))
            if rank == 0:
                print(f"BENCH mode={args.mode} quant={args.quant} T={t} M={t * world} {label}: "
                      f"rank0_median={med.item():.1f}us max_rank_median={med_all.max().item():.1f}us "
                      f"min_rank_median={med_all.min().item():.1f}us (iters={args.iters}, cold L2)", flush=True)
            del g

        forward(y, xq, xs)          # fills the buffer inputs (EP8: the kernel-only graph reads them)
        if args.mode == "ep16":
            if gin:
                bench(ep16_run, "full gin (one fused kernel, cross-node exchange inside)")
            else:
                tag = "overlap" if args.overlap else "serial"
                bench(ep16_run, f"full {tag} (pack + dispatch exchange + unpack + node kernel + combine exchange + add)")
            if args.stages:
                for label, fn in stages:
                    bench(fn, label)
        if node_kernel_only is not None:
            bench(node_kernel_only, "kernel only")
        if args.mode == "ep16" and gin and args.stamps:
            stamp_phases(ep, weights, args, y, flush, align, rank, world)

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

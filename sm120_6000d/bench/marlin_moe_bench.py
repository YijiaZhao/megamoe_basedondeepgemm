# Marlin MXFP4 W4A16 MoE kernel latency for DSV4-Flash routed experts, TP-sharded (one GPU).
# usage: python marlin_moe_bench.py <tp> <global_tokens...>
import sys, statistics, torch
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_mxfp4_like
from vllm.scalar_type import scalar_types

E, H, I, TOPK = 256, 4096, 2048, 6
tp = int(sys.argv[1]); toks = [int(x) for x in sys.argv[2:]]
il = I // tp
dev = "cuda"
torch.manual_seed(20261009)

def experts(n, k):
    ref, q, s = rand_marlin_weight_mxfp4_like(torch.empty(n, k, dtype=torch.bfloat16, device=dev), 32)[:3]
    qs = q.unsqueeze(0).expand(E, *q.shape).contiguous()
    ss = s.unsqueeze(0).expand(E, *s.shape).contiguous()
    return qs, ss

w1, s1 = experts(2 * il, H)
w2, s2 = experts(H, il)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
print(f"# Marlin MXFP4 W4A16 MoE, TP{tp} (I_local={il}), routed only, {torch.cuda.get_device_name()}, cold L2")
for T in toks:
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    logits = torch.randn(T, E, device=dev)
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK, -1)
    run = lambda: fused_marlin_moe(x, w1, w2, None, None, s1, s2, w.float(), ids.to(torch.int32),
                                   quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=E)
    for _ in range(5): run()
    ts = []
    for _ in range(20):
        flush.zero_()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); run(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    print(f"tp{tp} global_tokens {T:6d} marlin_us {statistics.median(ts):9.1f}", flush=True)

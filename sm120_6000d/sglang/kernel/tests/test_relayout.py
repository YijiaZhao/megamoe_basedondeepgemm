# sm120_fp4_relayout: FlashInfer row-major (W13 up rows first) -> packed 'w4a8' in place equals
# transform_fp4_weights_for_mega_moe_sm120(gate-first rows, 'w4a8'), with the same scales as 'w4a8_rm' + w13_up_first;
# to_row_major rebuilds the row-major bytes exactly. Then the time of to_row_major for one DSV4.1 layer at EP8
# (49 local experts with EPLB's replica).
#   python3 tests/test_relayout.py
import importlib.util
import os

import torch


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deep_gemm', 'mega', f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fp4 = _load('sm120_fp4')
rl = _load('sm120_fp4_relayout')

torch.manual_seed(0)
e, h, i = 3, 5120, 2304
w13_up = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8, device='cuda')  # FlashInfer: up rows, gate rows
w2 = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8, device='cuda')
sf13_up = torch.randint(118, 130, (e, 2 * i, h // 32), dtype=torch.uint8, device='cuda')
sf2 = torch.randint(118, 130, (e, h, i // 32), dtype=torch.uint8, device='cuda')
gate_first = torch.cat(w13_up.chunk(2, dim=1)[::-1], dim=1)
sf13_gate_first = torch.cat(sf13_up.chunk(2, dim=1)[::-1], dim=1)
(ref13, ref_s13), (ref2, ref_s2) = fp4.transform_fp4_weights_for_mega_moe_sm120((gate_first, sf13_gate_first),
                                                                               (w2, sf2), 'w4a8')
(_, rm_s13), (_, rm_s2) = fp4.transform_fp4_weights_for_mega_moe_sm120((w13_up, sf13_up), (w2, sf2), 'w4a8_rm',
                                                                       w13_up_first=True)
a13, a2 = w13_up.clone(), w2.clone()
rl.to_packed_(a13, a2)
assert torch.equal(a13, ref13) and torch.equal(a2, ref2), 'to_packed_ differs from the w4a8 transform'
assert torch.equal(rm_s13, ref_s13) and torch.equal(rm_s2, ref_s2), 'w4a8 scales differ from w4a8_rm up_first'
o13, o2 = torch.empty_like(a13), torch.empty_like(a2)
rl.to_row_major(a13, a2, o13, o2)
assert torch.equal(o13, w13_up) and torch.equal(o2, w2), 'to_row_major does not rebuild the row-major bytes'
print('relayout round trip bit-exact; w4a8 scales == w4a8_rm(up_first) scales')

el = 49
big13 = torch.randint(0, 256, (el, 2 * i, h // 2), dtype=torch.uint8, device='cuda')
big2 = torch.randint(0, 256, (el, h, i // 2), dtype=torch.uint8, device='cuda')
out13, out2 = torch.empty_like(big13), torch.empty_like(big2)
for _ in range(3):
    rl.to_row_major(big13, big2, out13, out2)
torch.cuda.synchronize()
a, b = torch.cuda.Event(True), torch.cuda.Event(True)
a.record()
for _ in range(10):
    rl.to_row_major(big13, big2, out13, out2)
b.record()
b.synchronize()
ms = a.elapsed_time(b) / 10
gb = 2 * (big13.numel() + big2.numel()) / 1e9
print(f'to_row_major, one layer ({el} experts, {gb / 2:.2f} GB): {ms:.2f} ms ({gb / ms:.2f} TB/s read + write)')

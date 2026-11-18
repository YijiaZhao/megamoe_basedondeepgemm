# W4A8 weight layouts of the SM120 MegaMoE: FlashInfer CUTLASS MXFP4 MoE (row-major E2M1 bytes, low nibble first; W13
# with the up rows first) <-> the kernel's packed 'w4a8' layout (`_pack_w_a8` of the rows `_l1_rows(up_first=True)`
# picks). Serving keeps the packed layout, which the decode kernel streams at ~93% of DRAM bandwidth (row-major:
# ~80%), and rebuilds a layer's row-major weights into a scratch buffer only for FlashInfer's calls (prefill).
import importlib.util
import os
import sys

import torch
import triton
import triton.language as tl

fp4 = sys.modules.get('sm120_fp4')
if fp4 is None:  # by file path, as SGLang and the tests load sm120_fp4
    _spec = importlib.util.spec_from_file_location(
        'sm120_fp4', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sm120_fp4.py'))
    fp4 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(fp4)


def to_packed_(w13: torch.Tensor, w2: torch.Tensor, chunk: int = 4) -> None:
    """In place, `chunk` experts at a time: FlashInfer row-major W13 [E, 2I, H/2] (up rows first) and W2 [E, H, I/2]
    -> the packed 'w4a8' layout (the same bytes `transform_fp4_weights_for_mega_moe_sm120(..., 'w4a8')` makes from
    gate-first rows; its scales are 'w4a8_rm''s with `w13_up_first`)."""
    e, two_i, _ = w13.shape
    perm = fp4._l1_rows(two_i // 2, w13.device, up_first=True)
    for e0 in range(0, e, chunk):
        sl = slice(e0, min(e, e0 + chunk))
        w13[sl] = fp4._pack_w_a8(fp4._unpack_e2m1(w13[sl])[:, perm]).view(w13[sl].shape)
        w2[sl] = fp4._pack_w_a8(fp4._unpack_e2m1(w2[sl])).view(w2[sl].shape)


@triton.jit
def _to_row_major_kernel(src, dst, N, K2, INTER, L1: tl.constexpr):
    # one program per (expert, 16-row block, K64 chunk): src bytes [g 8][c 4][s 2][p 2][j 4] (low nibble row g, high row
    # g + 8, k = 64 kc + 32 s + 16 p + 4 c + j) -> dst rows, 32 bytes each (k pairs, low nibble even k)
    kc = tl.program_id(0)
    nb = tl.program_id(1)
    ex = tl.program_id(2).to(tl.int64)
    nkc = tl.num_programs(0)
    nnb = tl.num_programs(1)
    base = ((ex * nnb + nb) * nkc + kc) * 512
    i = tl.arange(0, 512)
    b = tl.load(src + base + i)
    v = tl.reshape(b, (8, 4, 2, 2, 4))           # [g, c, s, p, j]
    v = tl.permute(v, (0, 2, 3, 1, 4))           # [g, s, p, c, j]: k order within the K64 chunk
    v = tl.reshape(v, (8, 32, 2))                # [g, k pair, k odd]
    lo, hi = tl.split(v)                         # even k / odd k bytes, each [g, 32]
    o0 = (lo & 0xF) | ((hi & 0xF) << 4)          # kernel row g
    o1 = (lo >> 4) | ((hi >> 4) << 4)            # kernel row g + 8
    g = tl.arange(0, 8)
    q = tl.arange(0, 32)
    if L1:  # kernel rows 16 nb + {g, g + 8} = gate row 8 nb + g (at I + ..., up rows first) and up row 8 nb + g
        r0 = INTER + nb * 8 + g
        r1 = nb * 8 + g
    else:
        r0 = nb * 16 + g
        r1 = nb * 16 + 8 + g
    off = ex * N * K2 + kc * 32 + q[None, :]
    tl.store(dst + off + r0[:, None].to(tl.int64) * K2, o0)
    tl.store(dst + off + r1[:, None].to(tl.int64) * K2, o1)


def to_row_major(w13: torch.Tensor, w2: torch.Tensor, out13: torch.Tensor, out2: torch.Tensor) -> None:
    """Packed 'w4a8' W13 / W2 (from `to_packed_`) -> FlashInfer row-major into out13 / out2 (same shapes)."""
    for src, dst, l1 in ((w13, out13, True), (w2, out2, False)):
        e, n, k2 = src.shape
        assert dst.shape == src.shape and src.is_contiguous() and dst.is_contiguous()
        _to_row_major_kernel[(k2 * 2 // 64, n // 16, e)](src.view(torch.uint8), dst.view(torch.uint8), n, k2, n // 2, L1=l1)

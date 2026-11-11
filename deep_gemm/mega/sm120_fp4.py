# SM120 (RTX PRO 6000 / 6000D) decode MegaMoE, W4A4: FP4 weights x FP4 activations, SwiGLU, top-k combine, EP over
# PCIe P2P. Kernel: `include/deep_gemm/impls/sm120_fp4_mega_moe.cu`, one torch extension per format, built on first
# use for sm_120a:
#   * 'mxfp4': E2M1 with a UE8M0 scale per 32 values (weights and activations).
#   * 'nvfp4': E2M1 with a UE4M3 scale per 16 values, plus fp32 global dequant scales: one per expert for W13 and W2
#     (checkpoint `weight_scale_2`), one per tensor for the FC1 and FC2 inputs (checkpoint `input_scale`; the same
#     value on every rank, e.g. the max over experts).
#   * 'w4a8': MXFP4 weights x MXFP8 activations (E4M3, UE8M0 scale per 32): the formats of `sm120.py`, on this
#     kernel's dataflow.
# Dataflow: the dispatch quantizes each token once and pushes
# its top-k copies into the owners' expert pools, FC1 + SwiGLU + FC2 run as swap-AB `mma.sync kind::mxf4nvf4`
# m16n8k64, the bf16 partials go back to the source rank and are combined there. Epoch-tagged flags, no cross-rank
# barriers, CUDA-graph safe. One buffer per format per process.

import importlib.util
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'include', 'deep_gemm', 'impls', 'sm120_fp4_mega_moe.cu')
FORMATS = {'mxfp4': 32, 'nvfp4': 16, 'w4a8': 32}  # format -> values per block scale
_EXT = {}
_KEY = {}


def _ext(fmt: str):
    # environment variables: sm120_6000d/ENV.md
    if fmt not in _EXT:
        name = f'sm120_fp4_mega_moe_{fmt}'
        so = os.environ.get(f'DG_SM120_FP4_MEGA_MOE_SO_{fmt.upper()}')
        if so:
            spec = importlib.util.spec_from_file_location(name, so)
            _EXT[fmt] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_EXT[fmt])
        else:
            from torch.utils.cpp_extension import load
            build_dir = os.path.join(os.environ.get('DG_SM120_MEGA_MOE_BUILD_DIR',
                                                    os.path.join(os.path.expanduser('~'), '.deep_gemm', 'sm120_mega_moe')),
                                     name)
            os.makedirs(build_dir, exist_ok=True)
            _EXT[fmt] = load(name=name, sources=[_SRC], build_directory=build_dir,
                             extra_cuda_cflags=['-gencode=arch=compute_120a,code=sm_120a', '-O3', '-std=c++20',
                                                '--use_fast_math', f'-DSM120_FP4_NV={int(fmt == "nvfp4")}',
                                                f'-DSM120_FP4_A8={int(fmt == "w4a8")}'])
    return _EXT[fmt]


# ----------------------------------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------------------------------

def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:  # [E, N, K/2] uint8 -> [E, N, K] codes (low nibble first)
    p = packed.view(torch.uint8)
    return torch.stack((p & 0xF, p >> 4), dim=-1).flatten(-2)


def _pack_w(codes: torch.Tensor) -> torch.Tensor:  # [E, N, K] codes (kernel row order) -> m16n8k64 A-fragment layout
    e, n, k = codes.shape
    x = codes.view(e, n // 16, 2, 8, k // 64, 2, 4, 4, 2)  # [E, nb, half, g, kc, p, c, byte, nibble]
    b = x[..., 0] | (x[..., 1] << 4)                         # [E, nb, half, g, kc, p, c, byte]
    return b.permute(0, 1, 4, 3, 6, 5, 2, 7).contiguous().view(e, -1)  # [E, nb, kc, g, c, p, half, byte]


def _pack_w_a8(codes: torch.Tensor) -> torch.Tensor:  # [E, N, K] codes -> W4A8 layout (two m16n8k32 A fragments)
    e, n, k = codes.shape
    x = codes.view(e, n // 16, 2, 8, k // 64, 2, 2, 4, 4)  # [E, nb, half, g, kc, s, p, c, j]
    b = x[:, :, 0] | (x[:, :, 1] << 4)
    return b.permute(0, 1, 3, 2, 6, 4, 5, 7).contiguous().view(e, -1)


def _pack_sf(sf: torch.Tensor) -> torch.Tensor:  # [E, N, K/vec] -> [E, nb, K/(4 vec), 16, 4]
    e, n, ks = sf.shape
    return sf.view(torch.uint8).view(e, n // 16, 16, ks // 4, 4).permute(0, 1, 3, 2, 4).contiguous().view(e, -1)


def _l1_rows(intermediate_hidden: int, device) -> torch.Tensor:
    # kernel row -> checkpoint row (gate 0..I-1, up I..2I-1): each 16-row block holds gate rows 8b..8b+7 then up rows
    # 8b..8b+7, so a SwiGLU pair lands in one thread (accumulator rows g and g + 8)
    r = torch.arange(2 * intermediate_hidden, device=device)
    blk, w = r // 16, r % 16
    return torch.where(w < 8, blk * 8 + w, intermediate_hidden + blk * 8 + (w - 8))


def transform_fp4_weights_for_mega_moe_sm120(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    fmt: str,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Checkpoint FP4 -> kernel layout.

    `l1_weights`: (w13 [E, 2I, H/2] uint8, two E2M1 per byte, low nibble first, gate rows 0..I-1 then up rows;
                   sf [E, 2I, H/vec] (UE8M0 for 'mxfp4', vec 32; E4M3 for 'nvfp4', vec 16)).
    `l2_weights`: (w2 [E, H, I/2] uint8; sf [E, H, I/vec]).
    The outputs keep the input shapes (the bytes are permuted). NVFP4 global scales go to `nvfp4_global_scales`.
    """
    vec = FORMATS[fmt]
    (w13, sf13), (w2, sf2) = l1_weights, l2_weights
    e, two_i, half_h = w13.shape
    hidden, inter = half_h * 2, two_i // 2
    assert sf13.shape == (e, two_i, hidden // vec) and w2.shape == (e, hidden, inter // 2), (w13.shape, sf13.shape)
    assert sf2.shape == (e, hidden, inter // vec), sf2.shape
    perm = _l1_rows(inter, w13.device)
    pack_w = _pack_w_a8 if fmt == 'w4a8' else _pack_w
    return ((pack_w(_unpack_e2m1(w13)[:, perm]).view(e, two_i, half_h),
             _pack_sf(sf13[:, perm]).view(e, two_i, hidden // vec)),
            (pack_w(_unpack_e2m1(w2)).view(e, hidden, inter // 2),
             _pack_sf(sf2).view(e, hidden, inter // vec)))


def nvfp4_global_scales(fc1_input_scale: float, fc2_input_scale: float, w13_scale_2: torch.Tensor,
                        w2_scale_2: torch.Tensor) -> torch.Tensor:
    """Pack the NVFP4 dequant global scales of this rank's experts (once, at load): [2 + 2 * local experts] fp32."""
    head = torch.tensor([fc1_input_scale, fc2_input_scale], dtype=torch.float32, device=w13_scale_2.device)
    return torch.cat([head, w13_scale_2.float().flatten(), w2_scale_2.float().flatten()]).contiguous()


def _clamp(activation_clamp: Optional[float]) -> float:
    return float(activation_clamp) if activation_clamp is not None else 3.0e38


# ----------------------------------------------------------------------------------------------------------------------
# EP
# ----------------------------------------------------------------------------------------------------------------------

class Sm120Fp4MegaMoeBuffer:
    """Peer-visible pools (expert inputs, routing, combine) of this rank, opened by every peer over cudaIpc."""

    def __init__(self, group: dist.ProcessGroup, fmt: str, num_experts: int, num_max_tokens_per_rank: int,
                 num_topk: int, hidden: int, intermediate_hidden: int):
        assert fmt in FORMATS, fmt
        self.group, self.fmt = group, fmt
        self.rank, self.world = group.rank(), group.size()
        assert num_experts % self.world == 0
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.world
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        key = (self.rank, self.world, self.num_local_experts, num_max_tokens_per_rank, hidden, intermediate_hidden,
               num_topk)
        if fmt not in _KEY:
            handle = _ext(fmt).init(*key)
            handles = [None] * self.world
            dist.all_gather_object(handles, handle, group=group)
            _ext(fmt).open_peers(handles)
            _KEY[fmt] = key
        assert _KEY[fmt] == key, f'one SM120 {fmt} MegaMoE buffer per process: {_KEY[fmt]} vs {key}'

    def error(self) -> int:
        """Non-zero after a spin wait timed out (peer missing or stuck)."""
        return int(_ext(self.fmt).counters()[3])


def fp4_mega_moe_sm120(y: torch.Tensor,
                       x: torch.Tensor,
                       topk_idx: torch.Tensor,
                       topk_weights: torch.Tensor,
                       l1_weights: Tuple[torch.Tensor, torch.Tensor],
                       l2_weights: Tuple[torch.Tensor, torch.Tensor],
                       sym_buffer: Sm120Fp4MegaMoeBuffer,
                       global_scales: Optional[torch.Tensor] = None,
                       activation_clamp: Optional[float] = None,
                       tile_tokens: int = 0,
                       stamps: Optional[torch.Tensor] = None) -> torch.Tensor:
    """EP MegaMoE over the buffer's group. `x` [T, H] bf16 (T <= num_max_tokens_per_rank), `topk_idx` [T, K] global
    expert ids (-1 = skip), `topk_weights` [T, K]; `y` [T, H] bf16 receives the combined output. Weights are this
    rank's experts from `transform_fp4_weights_for_mega_moe_sm120`; 'nvfp4' also takes `nvfp4_global_scales`.
    `tile_tokens`: tokens per work item (8 / 16; each weight fragment feeds that many tokens), 0 = chosen
    from the expected tokens per local expert. Every rank of the group must call it."""
    assert x.shape[0] <= sym_buffer.num_max_tokens_per_rank
    assert (global_scales is not None) == (sym_buffer.fmt == 'nvfp4')
    _ext(sym_buffer.fmt).run(x.to(torch.bfloat16).contiguous(), topk_idx.to(torch.int32).contiguous(),
                             topk_weights.to(torch.float32).contiguous(), y, l1_weights[0], l1_weights[1],
                             l2_weights[0], l2_weights[1], global_scales, _clamp(activation_clamp), tile_tokens,
                             stamps)
    return y

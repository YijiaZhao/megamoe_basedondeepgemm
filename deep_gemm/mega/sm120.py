# SM120 (RTX PRO 6000 / 6000D) decode MegaMoE: W-MXFP4 x A-MXFP8, SwiGLU, top-k combine.
#
# Two kernels share one source (`include/deep_gemm/impls/sm120_mxfp4_mega_moe.cu`, a torch extension built on first
# use for sm_120a):
#   * EP: one persistent kernel per rank. Dispatch (MXFP8 quantization + 16 B pushes over PCIe P2P into the owner's
#     expert pool), FC1 + SwiGLU + FC2 (swap-AB `mma.sync kind::mxf8f6f4` m16n8k32, weights global -> registers),
#     bf16 push back to the source rank and the local top-k combine. Epoch-tagged flags are written by the producer
#     into the consumer's memory; there are no cross-rank barriers or remote atomics, and it is CUDA-graph safe.
#   * TP: world 1, every expert local with an I/TP slice (MoE-TP). Split-K FC1 with fp32 reduction, the last split
#     applies SwiGLU + MXFP8, FC2 folds the top-k weight into an fp32 reduction. The output is this rank's partial
#     sum; the caller all-reduces it.
# Every spin wait times out (30 s) and reports through an error word instead of hanging the GPU.
#
# The kernels keep process-global state (one EP buffer and one TP workspace per process, shared by every MoE layer,
# since layers run in stream order); a second init with a different shape is rejected.

import importlib.util
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'include', 'deep_gemm', 'impls', 'sm120_mxfp4_mega_moe.cu')
_EXT = None
_EP_KEY = None
_TP_KEY = None
_TP_MAX_TOKENS = 64  # TP workspace rows per call; longer inputs are chunked


def _ext():
    # `DG_SM120_MEGA_MOE_SO`: a prebuilt extension (e.g. built once outside a serving container)
    global _EXT
    if _EXT is None:
        so = os.environ.get('DG_SM120_MEGA_MOE_SO')
        if so:
            spec = importlib.util.spec_from_file_location('sm120_mxfp4_mega_moe', so)
            _EXT = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_EXT)
        else:
            from torch.utils.cpp_extension import load
            build_dir = os.environ.get('DG_SM120_MEGA_MOE_BUILD_DIR',
                                       os.path.join(os.path.expanduser('~'), '.deep_gemm', 'sm120_mega_moe'))
            os.makedirs(build_dir, exist_ok=True)
            _EXT = load(name='sm120_mxfp4_mega_moe', sources=[_SRC], build_directory=build_dir,
                        extra_cuda_cflags=['-gencode=arch=compute_120a,code=sm_120a', '-O3', '-std=c++20',
                                           '--use_fast_math'])
    return _EXT


# ----------------------------------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------------------------------

def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:  # [E, N, K/2] uint8 -> [E, N, K] codes (low nibble first)
    p = packed.view(torch.uint8)
    return torch.stack((p & 0xF, p >> 4), dim=-1).flatten(-2)


def _pack_w(codes: torch.Tensor) -> torch.Tensor:  # [E, N, K] codes (kernel row order) -> lane-fragment layout
    e, n, k = codes.shape
    x = codes.view(e, n // 16, 2, 8, k // 64, 2, 2, 4, 4)  # [E, nb, half, g, kc, s, p, c, j]
    b = x[:, :, 0] | (x[:, :, 1] << 4)
    return b.permute(0, 1, 3, 2, 6, 4, 5, 7).contiguous().view(e, -1)


def _pack_sf(sf: torch.Tensor) -> torch.Tensor:  # [E, N, K/32] ue8m0 -> [E, nb, K/128, 16, 4]
    e, n, ks = sf.shape
    return sf.view(torch.uint8).view(e, n // 16, 16, ks // 4, 4).permute(0, 1, 3, 2, 4).contiguous().view(e, -1)


def _l1_rows(intermediate_hidden: int, device) -> torch.Tensor:
    # kernel row -> checkpoint row (gate 0..I-1, up I..2I-1): each 16-row block holds gate rows 8b..8b+7 then up rows
    # 8b..8b+7, so a SwiGLU pair lands in one thread (accumulator rows g and g + 8)
    r = torch.arange(2 * intermediate_hidden, device=device)
    blk, w = r // 16, r % 16
    return torch.where(w < 8, blk * 8 + w, intermediate_hidden + blk * 8 + (w - 8))


def transform_mxfp4_weights_for_mega_moe_sm120(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Checkpoint MXFP4 -> kernel layout (EP and TP kernels use the same layout).

    `l1_weights`: (w13 [E, 2I, H/2] uint8, two E2M1 per byte, low nibble first, gate rows 0..I-1 then up rows;
                   sf [E, 2I, H/32] ue8m0).
    `l2_weights`: (w2 [E, H, I/2] uint8; sf [E, H, I/32] ue8m0).
    For MoE-TP pass this rank's I/TP slice. The outputs keep the input shapes (the bytes are permuted), so they can
    replace the checkpoint parameters in place.
    """
    (w13, sf13), (w2, sf2) = l1_weights, l2_weights
    e, two_i, half_h = w13.shape
    hidden, inter = half_h * 2, two_i // 2
    assert sf13.shape == (e, two_i, hidden // 32) and w2.shape == (e, hidden, inter // 2), (w13.shape, w2.shape)
    assert sf2.shape == (e, hidden, inter // 32), sf2.shape
    perm = _l1_rows(inter, w13.device)
    return ((_pack_w(_unpack_e2m1(w13)[:, perm]).view(e, two_i, half_h),
             _pack_sf(sf13[:, perm]).view(e, two_i, hidden // 32)),
            (_pack_w(_unpack_e2m1(w2)).view(e, hidden, inter // 2),
             _pack_sf(sf2).view(e, hidden, inter // 32)))


def _clamp(activation_clamp: Optional[float]) -> float:
    return float(activation_clamp) if activation_clamp is not None else 3.0e38


# ----------------------------------------------------------------------------------------------------------------------
# EP
# ----------------------------------------------------------------------------------------------------------------------

class Sm120MegaMoeBuffer:
    """Peer-visible pools (expert inputs, routing, combine) of this rank, opened by every peer over cudaIpc."""

    def __init__(self, group: dist.ProcessGroup, num_experts: int, num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int):
        global _EP_KEY
        self.group = group
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
        if _EP_KEY is None:
            handle = _ext().init(*key)
            handles = [None] * self.world
            dist.all_gather_object(handles, handle, group=group)
            _ext().open_peers(handles)
            _EP_KEY = key
        assert _EP_KEY == key, f'one SM120 MegaMoE buffer per process: {_EP_KEY} vs {key}'

    def error(self) -> int:
        """Non-zero after a spin wait timed out (peer missing or stuck)."""
        return int(_ext().counters()[3])


def get_symm_buffer_for_mega_moe_sm120(group: dist.ProcessGroup, num_experts: int, num_max_tokens_per_rank: int,
                                       num_topk: int, hidden: int, intermediate_hidden: int) -> Sm120MegaMoeBuffer:
    return Sm120MegaMoeBuffer(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)


def mxfp4_mega_moe_sm120(y: torch.Tensor,
                         x: torch.Tensor,
                         topk_idx: torch.Tensor,
                         topk_weights: torch.Tensor,
                         l1_weights: Tuple[torch.Tensor, torch.Tensor],
                         l2_weights: Tuple[torch.Tensor, torch.Tensor],
                         sym_buffer: Sm120MegaMoeBuffer,
                         activation_clamp: Optional[float] = None) -> torch.Tensor:
    """EP MegaMoE over the buffer's group. `x` [T, H] bf16 (T <= num_max_tokens_per_rank), `topk_idx` [T, K] global
    expert ids (-1 = skip), `topk_weights` [T, K]; `y` [T, H] bf16 receives the combined output. Weights are this
    rank's experts from `transform_mxfp4_weights_for_mega_moe_sm120`. Every rank of the group must call it."""
    assert x.shape[0] <= sym_buffer.num_max_tokens_per_rank
    _ext().run(x.to(torch.bfloat16).contiguous(), topk_idx.to(torch.int32).contiguous(),
               topk_weights.to(torch.float32).contiguous(), y,
               l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1], _clamp(activation_clamp), None)
    return y


# ----------------------------------------------------------------------------------------------------------------------
# TP
# ----------------------------------------------------------------------------------------------------------------------

def init_mxfp4_moe_tp_sm120(num_experts: int, num_topk: int, hidden: int, intermediate_hidden: int) -> None:
    """`intermediate_hidden` is this rank's slice (I/TP)."""
    global _TP_KEY
    key = (num_experts, hidden, intermediate_hidden, num_topk)
    if _TP_KEY is None:
        _ext().init_tp(num_experts, _TP_MAX_TOKENS, hidden, intermediate_hidden, num_topk)
        _TP_KEY = key
    assert _TP_KEY == key, f'one SM120 MoE-TP workspace per process: {_TP_KEY} vs {key}'


def mxfp4_moe_tp_sm120(y: torch.Tensor,
                       x: torch.Tensor,
                       topk_idx: torch.Tensor,
                       topk_weights: torch.Tensor,
                       l1_weights: Tuple[torch.Tensor, torch.Tensor],
                       l2_weights: Tuple[torch.Tensor, torch.Tensor],
                       activation_clamp: Optional[float] = None) -> torch.Tensor:
    """MoE-TP on one rank: `y` [T, H] bf16 receives this rank's partial sum (the caller all-reduces it). Inputs are
    processed in chunks of 64 tokens (one call per decode step)."""
    assert _TP_KEY is not None, 'call init_mxfp4_moe_tp_sm120 first'
    x = x.to(torch.bfloat16)
    ids = topk_idx.to(torch.int32)
    w = topk_weights.to(torch.float32)
    limit = _clamp(activation_clamp)
    for s in range(0, x.shape[0], _TP_MAX_TOKENS):
        e = min(s + _TP_MAX_TOKENS, x.shape[0])
        _ext().run_tp(x[s:e].contiguous(), ids[s:e].contiguous(), w[s:e].contiguous(), y[s:e],
                      l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1], limit, None)
    return y


def mxfp4_moe_tp_sm120_error() -> int:
    return int(_ext().tp_error())

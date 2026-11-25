# H20 decode MegaMoE on the SM120 kernel's dataflow (several independent CTAs per SM, epoch flags, no cross-rank
# barriers): kernel `include/deep_gemm/impls/sm90_fp4_mega_moe_h20_v2.cu`, a torch extension built on first use for
# sm_90a (DG_H20V2_CPS CTAs per SM, DG_H20V2_STAGES weight slots per CTA). Weights: the fused MXFP4 tiles of `transform_mxfp4_weights_for_mega_moe_fused` (dense tiles, row factors);
# inputs: E4M3 rows with one fp32 scale per 128 (per_token_cast_to_fp8(gran_k=128)), int64 top-k ids, fp32 weights.
# EP over NVLink P2P (cudaIpc), one buffer per process.

import os

import torch
import torch.distributed as dist

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, 'include', 'deep_gemm', 'impls', 'sm90_fp4_mega_moe_h20_v2.cu')
_EXT = {}
_KEY = {}


def _ext():
    cps, stages = int(os.environ.get('DG_H20V2_CPS', '2')), int(os.environ.get('DG_H20V2_STAGES', '4'))
    lag, pfl2 = int(os.environ.get('DG_H20V2_LAG', '2')), int(os.environ.get('DG_H20V2_PFL2', '8'))
    key = (cps, stages, lag, pfl2)
    if key not in _EXT:
        from torch.utils.cpp_extension import load
        name = f'h20_v2_mega_moe_c{cps}_s{stages}_l{lag}_p{pfl2}'
        build_dir = os.path.join(os.path.expanduser('~'), '.deep_gemm', 'h20_v2', name)
        os.makedirs(build_dir, exist_ok=True)
        _EXT[key] = load(name=name, sources=[_SRC], build_directory=build_dir,
                         extra_include_paths=[os.path.join(_ROOT, 'include')],
                         extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a', '-O3', '-std=c++17',
                                            '--use_fast_math', f'-DH20V2_CPS={cps}', f'-DH20V2_STAGES={stages}',
                                            f'-DH20V2_LAG={lag}', f'-DH20V2_PFL2={pfl2}'])
    return _EXT[key]


class H20V2Buffer:
    """Symmetric P2P buffer: pools for `num_max_tokens_per_rank` tokens from every rank, combine slots, flags."""

    def __init__(self, group, num_experts: int, num_max_tokens_per_rank: int, topk: int, hidden: int,
                 intermediate: int):
        world, rank = group.size(), group.rank()
        assert num_experts % world == 0
        self.ext = _ext()
        key = (world, num_experts, num_max_tokens_per_rank, topk, hidden, intermediate)
        assert not _KEY or _KEY['key'] == key, 'one H20V2Buffer configuration per process'
        if not _KEY:
            handle = self.ext.init(rank, world, num_experts // world, num_max_tokens_per_rank, hidden, intermediate, topk)
            handles = [None] * world
            dist.all_gather_object(handles, handle, group=group)
            self.ext.open_peers(handles)
            dist.barrier(group=group)
            _KEY['key'] = key
        self.group = group


_TILES = {}


def _v2_tiles(fused: torch.Tensor) -> torch.Tensor:
    """The fused tiles with each row's four per-K32 LUT indices (bytes 64..67, 0..15) stored x 8: the kernel adds the
    byte straight to the LUT address. Cached per weight tensor (an offline weight transform)."""
    key = (fused.data_ptr(), fused.numel())
    if key not in _TILES:
        t = fused.clone()
        rows = t.view(-1, 80)
        assert int(rows[:, 64:68].max()) < 32
        rows[:, 64:68] *= 8
        _TILES[key] = t
    return _TILES[key]


def mxfp4_mega_moe_h20_v2(y, x, x_sf, topk_ids, topk_w, l1, l2, buffer: H20V2Buffer, activation_clamp=10.0,
                          tile_tokens=0, stamps=None):
    """y[T, H] bf16 = sum_k w[t, k] * FFN_{e(t, k)}(x[t]); l1 / l2 = (fused tiles, tile-major scales, row factors)."""
    buffer.ext.run(x, x_sf, topk_ids, topk_w, y, _v2_tiles(l1[0]), l1[2], _v2_tiles(l2[0]), l2[2],
                   float(activation_clamp), int(tile_tokens), stamps)

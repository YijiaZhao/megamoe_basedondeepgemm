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


def _ext(gin_type: int = 0):
    # gin_type: the NCCL GIN type of the rail communicator (2 proxy, 3 GDAKI) for the EP16 build, 0 = EP8 (P2P only)
    cps, stages = int(os.environ.get('DG_H20V2_CPS', '2')), int(os.environ.get('DG_H20V2_STAGES', '4'))
    lag, pfl2 = int(os.environ.get('DG_H20V2_LAG', '2')), int(os.environ.get('DG_H20V2_PFL2', '8'))
    key = (cps, stages, lag, pfl2, gin_type)
    if key not in _EXT:
        from torch.utils.cpp_extension import load
        name = f'h20_v2_mega_moe_c{cps}_s{stages}_l{lag}_p{pfl2}' + (f'_gin{gin_type}' if gin_type else '')
        build_dir = os.path.join(os.path.expanduser('~'), '.deep_gemm', 'h20_v2', name)
        os.makedirs(build_dir, exist_ok=True)
        include = [os.path.join(_ROOT, 'include')]
        flags = ['-gencode=arch=compute_90a,code=sm_90a', '-O3', '-std=c++17', '--use_fast_math', f'-DH20V2_CPS={cps}',
                 f'-DH20V2_STAGES={stages}', f'-DH20V2_LAG={lag}', f'-DH20V2_PFL2={pfl2}']
        if gin_type:
            import nvidia.nccl
            include.append(os.path.join(list(nvidia.nccl.__path__)[0], 'include'))
            flags += ['-DH20V2_GIN=1', '-D_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE', '-DNCCL_GIN_GPI_ENABLE=0',
                      f'-DNCCL_GIN_PROXY_ENABLE={int(gin_type == 2)}', f'-DNCCL_GIN_GDAKI_ENABLE={int(gin_type == 3)}']
        _EXT[key] = load(name=name, sources=[_SRC], build_directory=build_dir, extra_include_paths=include,
                         extra_cuda_cflags=flags)
    return _EXT[key]


class H20V2Buffer:
    """Symmetric P2P buffer: pools for `num_max_tokens_per_rank` tokens from every rank, combine slots, flags.
    EP16 (`H20V2EP16`): `group` is the node's 8 ranks, `num_experts` the node's, the rows per rank 2T."""

    def __init__(self, group, num_experts: int, num_max_tokens_per_rank: int, topk: int, hidden: int,
                 intermediate: int, gin_type: int = 0):
        world, rank = group.size(), group.rank()
        assert num_experts % world == 0
        self.ext = _ext(gin_type)
        self.gin_args = []
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
                   float(activation_clamp), int(tile_tokens), stamps, buffer.gin_args)


class H20V2EP16:
    """Two-node EP16: global rank r = node * 8 + local, expert e on rank e // (E / 16). Each rank exchanges its T rows
    with its rail peer r ^ 8 inside the v2 launch (NCCL GIN, one 2-rank communicator per rail pair, the NIC buffers one
    symmetric window on it; deep_gemm/mega/ep16.py's session) and runs the node's experts over 2T rows. Call as
    `mxfp4_mega_moe_h20_v2(y, x, x_sf, topk_ids, topk_w, l1, l2, ep16.buffer)` with this rank's T rows and global
    top-k ids; the weights are this rank's E / 16 experts."""

    def __init__(self, world_group, node_group, num_experts: int, t: int, topk: int, hidden: int, intermediate: int):
        from .ep16 import _gin_session
        assert world_group.size() == 16 and node_group.size() == 8 and num_experts % 16 == 0 and t <= 256
        self.rank = world_group.rank()
        self.node = self.rank // 8
        local = self.rank % 8
        for i in range(8):  # every rank creates every rail communicator
            g = dist.new_group([i, i + 8])
            if i == local:
                rail = g
        dist.barrier(group=rail)  # ProcessGroupNCCL creates its communicator lazily
        comm = int(rail._get_backend(torch.device('cuda'))._comm_ptr())
        session = _gin_session()
        contexts = int(os.environ.get('DG_EP16_GIN_CONTEXTS', '4'))
        row = hidden + hidden // 128 * 4 + topk * 12
        nbytes = 2 * t * row + 2 * t * hidden * 2 + 16  # layout/h20_ep16_gin.cuh (+ epoch, send counter)
        self._gin = (session, session.create(comm, self.node, 2, nbytes, contexts))
        base, win, dev_comm, ctx, gin_type = (int(v) for v in session.info(self._gin[1]))
        assert ctx >= 1 and gin_type in (2, 3), f'NCCL GIN type {gin_type}: needs proxy (2) or GDAKI (3)'
        self.rail = rail
        dist.barrier(group=rail)
        self.buffer = H20V2Buffer(node_group, num_experts // 2, 2 * t, topk, hidden, intermediate, gin_type)
        self.buffer.gin_args = [dev_comm, win, base, self.node, t, ctx]
        # both nodes past the extension build before the first launch (its input wait times out after 30 s)
        dist.barrier(group=world_group)

    def destroy(self):
        torch.cuda.synchronize()
        self._gin[0].destroy(self._gin[1])

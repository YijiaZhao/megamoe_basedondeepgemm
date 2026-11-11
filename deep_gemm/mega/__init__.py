import os
import weakref
import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_ring_tokens: int,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu'):
        assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_ring_tokens = num_ring_tokens

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_ring_tokens
        )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    # To save buffer size, we enable ring buffer
    # TODO: move the wave concept into kernel and dynamically schedule
    # TODO: currently decoding may consume more memory than prefill
    # TODO: finer-grained wave
    num_min_ring_tokens, num_max_ring_tokens = \
        _C.get_ring_limit_for_mega_moe(num_max_tokens_per_rank, num_experts // group.size(), num_topk, group.size())
    if num_max_tokens_per_rank >= 6144:
        # We assume must be prefill (decode cannot have such size)
        # We try to give ~8 GB budget (within V4 Pro config)
        # And batch size is mostly stable, to save buffer size, we use 1 expert per wave
        num_ring_tokens = align(768 * 1024, _C.get_token_alignment_for_mega_moe())
    else:
        # Otherwise, we must ensure, like for EP64, 4K decoding batch size,
        # the wave heuristics can select the best number of experts per wave
        # In this case, the budget is roughly ~18 GB
        num_ring_tokens = _C.get_ring_limit_for_mega_moe(
            align(4096, _C.get_token_alignment_for_mega_moe()), 432 // 72, 6, 72)[1]
    num_ring_tokens = max(num_ring_tokens, num_min_ring_tokens)
    num_ring_tokens = min(num_ring_tokens, num_max_ring_tokens)

    # The SM90 MegaMoE family (`fp8xmxfp4`, including the W4A8-int variants)
    # currently runs the legacy arrival-count protocol expressed over the ring
    # counters; it requires the ring to cover the full pool (lap == 0 forever).
    # Use the legacy (pre-rewrite) full-pool size: one kMaxCandidateBlockM (192)
    # padding block per expert instead of the max ring limit's two, which
    # inflates the acts/SF working set by ~10-14% on typical decode shapes.
    # Every real block_m candidate is <= 192, so all pool offsets stay covered
    # and lap == 0 remains valid (also >= the one-expert-per-wave minimum).
    if mma_type == 'fp8xmxfp4':
        num_ring_tokens = _C.get_legacy_pool_tokens_for_mega_moe(
            num_max_tokens_per_rank, num_experts // group.size(), num_topk, group.size())
        num_ring_tokens = max(num_min_ring_tokens, min(num_ring_tokens, num_max_ring_tokens))

    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_ring_tokens,
        mma_type=mma_type, activation=activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
             Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
    if isinstance(l1_weights, tuple):
        # FP8: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens
    )


# =============================================================================
# SM90 (Hopper) MegaMoE family: MXFP4 / W4A8-int split kernels
# =============================================================================

def _interleave_l1_weights(l1_weights: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return _interleave_weights(l1_weights[0]), _interleave_weights(l1_weights[1])


def get_nvfp4_mega_moe_sm90_block_n(intermediate_hidden: int) -> int:
    """Choose the measured H20 deployment layout for an SM90 NVFP4 model."""
    assert intermediate_hidden > 0 and intermediate_hidden % 128 == 0
    return 128


def get_nvfp4_mega_moe_sm90_weight_layout(
        hidden: int, intermediate_hidden: int) -> str:
    """Choose the one lossless packed-value layout cached for an SM90 model."""
    assert hidden > 0 and hidden % 128 == 0
    assert intermediate_hidden > 0 and intermediate_hidden % 128 == 0
    return "prmt_groups" if hidden > 2 * intermediate_hidden else "marlin"


def get_mxfp4_mega_moe_sm90_block_n(
        hidden: int, intermediate_hidden: int,
        expected_m_per_rank: Optional[int] = None) -> int:
    """Pick the single deployment BLOCK_N from the workload's typical M.

    Measured H20 crossovers (MXFP4, 8 ranks): BN128 (swapAB+RF small-M path)
    wins below ~M190 on Flash-class shapes (ih<=2048) and ~M380 on Pro-class;
    BN256 wins above (-28% at large M). Defaults to BN128 when no hint given
    (decode-serving is the common case).
    """
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0
    if expected_m_per_rank is None:
        return 128
    crossover = 192 if intermediate_hidden <= 2048 else 384
    return 256 if expected_m_per_rank >= crossover else 128


def transform_mxfp4_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    block_n: Optional[int] = None,
    block_k: int = 128,
    group_size: int = 32,
    expected_m_per_rank: Optional[int] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Losslessly prepack OCP MXFP4 weights for the SM90 split kernel.

    Inputs contain packed E2M1 values ``(E, N, K/2)`` and row-major E8M0
    scales ``(E, N, K/32)``. Deployment policy: BN128 default, direct PRMT
    value groups (kernel policy is unconditional for MXFP4). The fused rows
    keep the 80-byte stride (64B values + 4B E8M0 + 12B padding).
    """
    from ..quantization_mxfp4 import (
        mxfp4_fuse_packed_with_scale_tile_major,
        mxfp4_scale_to_tile_major,
    )

    l1_packed, l1_scale = l1_weights
    l2_packed, l2_scale = l2_weights
    hidden = l1_packed.size(-1) * 2
    intermediate_hidden = l2_packed.size(-1) * 2
    use_prmt_groups = True
    if block_n is None:
        block_n = get_mxfp4_mega_moe_sm90_block_n(
            hidden, intermediate_hidden, expected_m_per_rank)

    assert block_n in (128, 256)
    assert block_k == 128
    assert group_size == 32
    assert l1_packed.dtype == torch.uint8 and l2_packed.dtype == torch.uint8
    assert l1_scale.dtype == torch.uint8 and l2_scale.dtype == torch.uint8
    assert l1_packed.dim() == 3 and l2_packed.dim() == 3
    assert l1_scale.dim() == 3 and l2_scale.dim() == 3

    l1_packed_il, l1_scale_il = _interleave_l1_weights((l1_packed, l1_scale))
    l1_scale_tm = mxfp4_scale_to_tile_major(
        l1_scale_il, block_n=block_n, block_k=block_k, group_size=group_size)
    l2_scale_tm = mxfp4_scale_to_tile_major(
        l2_scale, block_n=block_n, block_k=block_k, group_size=group_size)
    l1_packed_out = mxfp4_fuse_packed_with_scale_tile_major(
        l1_packed_il.contiguous(), l1_scale_tm, block_k=block_k,
        use_prmt_groups=use_prmt_groups, use_rf_fragments=True)
    l2_packed_out = mxfp4_fuse_packed_with_scale_tile_major(
        l2_packed.contiguous(), l2_scale_tm, block_k=block_k,
        use_prmt_groups=use_prmt_groups, use_rf_fragments=True)
    return (l1_packed_out, l1_scale_tm), (l2_packed_out, l2_scale_tm)


def transform_qoq_int4_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    block_n: int = 128,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Prepack the sole supported INT4 format: QoQ + asymmetric zero point.

    Inputs are ``(packed_uint4, coeff_plane_int32)`` from
    :func:`quantize_to_qoq_int4`. The coefficient bytes are
    ``[s2, z, s1_bf16]`` for every output row and K128 group.
    """
    from ..quantization_mxfp4 import (
        mxfp4_fuse_packed_with_scale_tile_major,
        qoq_plane_to_tile_major,
    )
    assert block_n in (128, 256)
    l1_packed, l1_plane = _interleave_l1_weights(l1_weights)
    l2_packed, l2_plane = l2_weights
    l1_tm = qoq_plane_to_tile_major(l1_plane, block_n)
    l2_tm = qoq_plane_to_tile_major(l2_plane, block_n)
    direct_nibble = os.environ.get("DG_W4A8_INT_DIRECT_NIBBLE", "0") != "0"
    use_prmt_groups = not direct_nibble

    def fuse(packed, plane_tm):
        fused = mxfp4_fuse_packed_with_scale_tile_major(
            packed.contiguous(), plane_tm, block_k=128,
            use_prmt_groups=use_prmt_groups, use_rf_fragments=True)
        return fused, plane_tm

    return fuse(l1_packed, l1_tm), fuse(l2_packed, l2_tm)


def mxfp4_mega_moe(y: torch.Tensor,
                   l1_weights: Tuple[torch.Tensor, torch.Tensor],
                   l2_weights: Tuple[torch.Tensor, torch.Tensor],
                   sym_buffer: SymmBuffer,
                   cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                   l1_global_scales: Optional[torch.Tensor] = None,
                   l2_global_scales: Optional[torch.Tensor] = None,
                   recipe: Optional[Tuple[int, int, int]] = None,
                   activation: str = 'swiglu',
                   activation_clamp: Optional[float] = None,
                   fast_math: bool = True,
                   attn_tp_size: int = 1,
                   attn_tp_group: Optional[dist.ProcessGroup] = None,
                   broadcast_output: bool = False,
                   router_input: Optional[torch.Tensor] = None,
                   router_weight: Optional[torch.Tensor] = None,
                   router_logits: Optional[torch.Tensor] = None,
                   router_renormalize: bool = True,
                   tp_combine_output: Optional[torch.Tensor] = None,
                   hidden_input: Optional[torch.Tensor] = None,
                   _cpp_op=None):
    """Run the SM90 split MegaMoE kernel with prepacked OCP MXFP4 weights.

    Weights come from ``transform_mxfp4_weights_for_mega_moe_sm90``. Values
    decode through an unscaled E2M1->E4M3 LUT and the per-32-K E8M0 dequant
    coefficient is applied in the WGMMA promotion (epilogue) stage.
    """
    l1_scale_metadata = l1_weights[1]
    l2_scale_metadata = l2_weights[1]
    assert l1_scale_metadata.dim() == 5 and l2_scale_metadata.dim() == 5
    cached_block_n = int(l1_scale_metadata.size(3))
    assert int(l2_scale_metadata.size(3)) == cached_block_n
    if recipe is None:
        recipe = (128, cached_block_n, 128)
    else:
        assert recipe[1] == cached_block_n

    assert attn_tp_size >= 1
    assert (router_input is None) == (router_weight is None)
    assert router_logits is None or router_input is None
    if router_input is not None:
        assert router_input.dtype == torch.bfloat16 and router_input.is_contiguous()
        assert router_weight.dtype == torch.bfloat16 and router_weight.is_contiguous()
        assert router_input.shape == (y.size(0), sym_buffer.hidden)
        assert router_weight.shape == (sym_buffer.num_experts, sym_buffer.hidden)
    if router_logits is not None:
        assert router_logits.dtype == torch.bfloat16 and router_logits.is_contiguous()
        assert router_logits.shape == (y.size(0), sym_buffer.num_experts)
    if tp_combine_output is not None:
        assert attn_tp_size > 1 and y.size(0) % attn_tp_size == 0
        assert tp_combine_output.dtype == torch.bfloat16
        assert tp_combine_output.is_contiguous()
        assert tp_combine_output.shape == (y.size(0) // attn_tp_size, y.size(1))
    if hidden_input is not None:
        assert hidden_input.dtype == torch.bfloat16 and hidden_input.is_contiguous()
        assert hidden_input.shape == y.shape
    assert sym_buffer.group.size() % attn_tp_size == 0
    if attn_tp_size > 1:
        assert attn_tp_group is not None, \
            "attn_tp_group is required when attn_tp_size > 1"
        assert attn_tp_group.size() == attn_tp_size
        # Rank layout is contiguous (DP outer, attention TP inner). This is
        # the layout used by SGLang DP-attention and by the SM90 kernel's
        # physical-source mapping.
        assert sym_buffer.group.rank() % attn_tp_size == attn_tp_group.rank()

    if _cpp_op is None:
        _cpp_op = _C.mxfp4_mega_moe
    _cpp_op(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        l1_global_scales,
        l2_global_scales,
        router_input,
        router_weight,
        router_logits,
        router_renormalize,
        tp_combine_output,
        hidden_input,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        attn_tp_size,
        recipe,
        activation,
        activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens,
    )

    if tp_combine_output is not None:
        dist.all_gather_into_tensor(y, tp_combine_output, group=attn_tp_group)
    elif attn_tp_size > 1 and broadcast_output:
        # Every expert rank combines a TP group's replicated token stream only
        # onto that group's rank 0. Replicate the completed MoE result back to
        # the remaining attention-TP ranks for the following attention block.
        root_global_rank = dist.get_global_rank(attn_tp_group, 0)
        dist.broadcast(y, src=root_global_rank, group=attn_tp_group)


# W4A8 INT4 has one supported format: canonical QoQ+ZP with full INT4 L1/L2
# and SHIFTXOR decode. DG_W4A8_INT=1 selects it; optional DIRECT_NIBBLE and
# ZSUB_XOR only change the equivalent decode implementation.
int4_mega_moe = mxfp4_mega_moe


def mxfp4_mega_moe_from_bf16(
    y: torch.Tensor,
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    l1_global_scales: Optional[torch.Tensor] = None,
    l2_global_scales: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    activation: str = 'swiglu',
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    attn_tp_size: int = 1,
    attn_tp_group: Optional[dist.ProcessGroup] = None,
    broadcast_output: bool = False,
    router_renormalize: bool = True,
    tp_combine_output: Optional[torch.Tensor] = None,
    router_logits_buffer: Optional[torch.Tensor] = None,
):
    """End-to-end BF16-hidden MegaMoE entry point.

    The timed operation starts from rank-local BF16 hidden states and performs
    router GEMM, activation quantization, fused TopK/dispatch, expert L1/L2,
    combine, and the final attention-TP collective. Output is BF16 ``y``.
    """
    assert hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
    assert router_weight.dtype == torch.bfloat16 and router_weight.is_contiguous()
    if tp_combine_output is None:
        assert hidden_states.shape == y.shape
        local_y = y
    else:
        assert attn_tp_size > 1 and attn_tp_group is not None
        assert tp_combine_output.dtype == torch.bfloat16 and tp_combine_output.is_contiguous()
        assert tp_combine_output.shape == hidden_states.shape
        assert y.shape == (hidden_states.size(0) * attn_tp_size, hidden_states.size(1))
        local_y = tp_combine_output
    assert router_weight.shape == (sym_buffer.num_experts, sym_buffer.hidden)
    assert attn_tp_group is not None or attn_tp_size == 1

    def finalize_tp_output():
        if tp_combine_output is None:
            return
        if broadcast_output:
            # Gather-to-root + broadcast, represented as disjoint token slots
            # followed by SUM reduction. Produces the same global token layout
            # as AllGather while exercising the root-combine communication mode.
            local_m = hidden_states.size(0)
            tp_rank = attn_tp_group.rank()
            y.zero_()
            y[tp_rank * local_m:(tp_rank + 1) * local_m].copy_(local_y)
            root_global_rank = dist.get_global_rank(attn_tp_group, 0)
            dist.reduce(y, dst=root_global_rank, group=attn_tp_group)
            dist.broadcast(y, src=root_global_rank, group=attn_tp_group)
        else:
            dist.all_gather_into_tensor(y, local_y, group=attn_tp_group)

    # Router weights are replicated over attention TP ranks; hidden tokens are
    # rank-local DP shards, so every rank runs the frontend.
    fuse_qoq_input_quant = os.environ.get("DG_W4A8_INT", "0") != "0"
    router_logits = None
    if fuse_qoq_input_quant:
        if router_logits_buffer is None:
            router_logits = torch.empty(
                (hidden_states.size(0), sym_buffer.num_experts),
                dtype=torch.bfloat16, device=hidden_states.device)
        else:
            assert router_logits_buffer.shape == (hidden_states.size(0), sym_buffer.num_experts)
            assert router_logits_buffer.dtype == torch.bfloat16 and router_logits_buffer.is_contiguous()
            router_logits = router_logits_buffer
        if recipe is None:
            recipe = (128, int(l1_weights[1].size(3)), 128)
        _C.qoq_bf16_mega_moe(
            local_y, hidden_states, router_weight, router_logits,
            sym_buffer.x[:hidden_states.size(0)],
            sym_buffer.x_sf[:hidden_states.size(0)],
            sym_buffer.topk_idx[:hidden_states.size(0)],
            sym_buffer.topk_weights[:hidden_states.size(0)],
            l1_weights, l2_weights,
            cumulative_local_expert_recv_stats,
            l1_global_scales, l2_global_scales, None,
            sym_buffer.buffer, sym_buffer.handle.buffer_ptrs,
            sym_buffer.group.rank(), sym_buffer.num_max_tokens_per_rank,
            sym_buffer.num_experts, sym_buffer.num_topk, attn_tp_size,
            recipe, activation, activation_clamp, fast_math,
            sym_buffer.num_ring_tokens)
        finalize_tp_output()
        return y
    else:
        # TP router weights are replicated while tokens are DP-sharded. Every
        # physical rank runs the local fused Router+FP8-quant+TopK frontend and
        # writes directly into the symmetric input views consumed by L1.
        num_tokens = hidden_states.size(0)
        for token_begin in range(0, num_tokens, 64):
            token_end = min(token_begin + 64, num_tokens)
            _C.mxfp4_router_quant_topk(
                hidden_states[token_begin:token_end], router_weight,
                sym_buffer.x[token_begin:token_end],
                sym_buffer.x_sf[token_begin:token_end],
                sym_buffer.topk_idx[token_begin:token_end],
                sym_buffer.topk_weights[token_begin:token_end],
            )

    mxfp4_mega_moe(
        local_y, l1_weights, l2_weights, sym_buffer,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        l1_global_scales=l1_global_scales,
        l2_global_scales=l2_global_scales,
        recipe=recipe,
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        attn_tp_size=attn_tp_size,
        attn_tp_group=attn_tp_group,
        broadcast_output=broadcast_output,
        router_logits=None,
        router_renormalize=router_renormalize,
        tp_combine_output=None,
        hidden_input=None,
    )
    finalize_tp_output()
    return y


# Explicit split API names.  Unlike the legacy ``mxfp4_mega_moe`` /
# ``int4_mega_moe`` aliases, these select their backend directly and therefore
# do not depend on mutating DG_W4A8_INT between calls.
def mxfp4_mega_moe_split(*args, **kwargs):
    kwargs["_cpp_op"] = _C.mxfp4_mega_moe_split
    return mxfp4_mega_moe(*args, **kwargs)


def qoq_mega_moe_split(*args, **kwargs):
    kwargs["_cpp_op"] = _C.qoq_mega_moe_split
    return mxfp4_mega_moe(*args, **kwargs)

from .fused import FusedSymmBuffer, get_fused_symm_buffer_for_mega_moe, transform_mxfp4_weights_for_mega_moe_fused, transform_qoq_weights_for_mega_moe_fused, mxfp4_mega_moe_fused, qoq_mega_moe_fused


_FRONTEND_STAMPS_BYTES = 256 * 8 * 8
_FRONTEND_CC_KEYS_OFF = 256 + 64 * 1024   # ticket area (256 B) + kCCKeysOff: compact [token][e] u32 keys of the cc router (fable_frontend.cu, ckeys = logits + kCCKeysOff, logits = workspace + 256)
FE_PATH_WMMA, FE_PATH_SWAPAB, FE_PATH_CC = 0, 1, 2     # csrc/fable_frontend.h kFEPath*


def fable_frontend_workspace_bytes(e: int) -> int:
    return 256 + 4 * 64 * e * 4 + _FRONTEND_STAMPS_BYTES


def fable_frontend_path(m: int, h: int = 3072, e: int = 384, topk: int = 8) -> int:
    """Launch shape the frontend uses for (m, h, e, topk) (csrc/fable_frontend.h `select_fe_path`): FE_PATH_CC
    (m <= 2, h 3072, e 384, top-8: the CUDA-core K-split router, 77 + 1 CTAs), FE_PATH_SWAPAB (m <= 16, h 3072,
    top-8: mma.sync m16n8k16 router on the 96 x 4 grid), else FE_PATH_WMMA."""
    return _C.fable_frontend_path(m, h, e, topk)


def _fe_wlayout_from_env(wlayout):
    """DG_FE_ROUTER_WLAYOUT: 'row' -> 0 = [e][h] router weights; 'fragment' (default) -> 1 =
    one-time host permutation into m16n8k16 A-fragment order (swapab path only; cached per weight
    tensor); 'pre' -> 2 = the caller already passes the permuted tensor."""
    if wlayout is None:
        wlayout = os.environ.get("DG_FE_ROUTER_WLAYOUT", "fragment")
    if isinstance(wlayout, str):
        wlayout = {"row": 0, "fragment": 1, "pre": 2, "0": 0, "1": 1, "2": 2}[wlayout.strip().lower()]
    return int(wlayout)


def fable_router_weight_fragment_layout(router_weight: torch.Tensor) -> torch.Tensor:
    """Permute [E, K] bf16 router weights (E % 16 == 0, K % 32 == 0) into the swapab
    A-fragment order the frontend reads with DG_FE_ROUTER_WLAYOUT=fragment: same bytes,
    laid out as [E/16 groups][K/32 blocks][half: rows g | g+8][lane = 4 g + t][8 bf16] where
    lane (g, t) holds W[16 G + 8 half + g][32 b + 8 t .. + 7]. One warp load instruction of the
    kernel (16 B per lane) then reads one contiguous 512 B run = 4 full 128 B lines. Returned
    tensor has the same shape [E, K] (contiguous) so it drops into the frontend API unchanged."""
    e, k = router_weight.shape
    assert router_weight.dtype == torch.bfloat16 and e % 16 == 0 and k % 32 == 0
    v = router_weight.contiguous().view(e // 16, 2, 8, k // 32, 4, 8)      # [G][half][g][b][t][8]
    return v.permute(0, 3, 1, 2, 4, 5).contiguous().view(e, k)          # [G][b][half][g][t][8]


_fe_fragment_weight_cache = {}        # id(weight tensor) -> (version, permuted copy); entry purged when the tensor dies


def _fe_router_weight_for_layout(router_weight, wlayout):
    """Cached one-time fragment permutation. Keyed by id() of the source tensor OBJECT with a
    weakref.finalize purge (a freed weight whose address / id is reused by a new tensor -- the
    per-seed weights of the tests -- cannot hit a stale entry; tensor keys themselves are unusable
    because dict key comparison would call Tensor.__eq__) plus the in-place version counter.
    Production callers should permute once at weight-transform time
    (fable_router_weight_fragment_layout) and pass wlayout='pre'."""
    if wlayout == 0:
        return router_weight
    key = id(router_weight)
    hit = _fe_fragment_weight_cache.get(key)
    if hit is None or hit[0] != router_weight._version:
        if hit is None:
            weakref.finalize(router_weight, _fe_fragment_weight_cache.pop, key, None)
        hit = _fe_fragment_weight_cache[key] = (router_weight._version, fable_router_weight_fragment_layout(router_weight))
    return hit[1]


def fable_frontend_router_ctas(m: int, e: int, h: int = 3072, topk: int = 8) -> int:
    """Router CTA count the frontend launch uses for (m, h, e, topk)."""
    return _C.fable_frontend_router_ctas(m, h, e, topk)


def fable_frontend_workspace(sym_buffer, e: int, device) -> torch.Tensor:
    """The Fable frontend workspace of ``sym_buffer`` (zero-initialised once, cached):
    [0, 256) tickets / hand-off counters, then the fp32 K-split partial logits (the cc router's
    compact key array lives inside them at +64 KB), then the optional phase stamps."""
    cache = getattr(sym_buffer, "_fable_frontend_cache", None)
    if cache is None:
        cache = sym_buffer._fable_frontend_cache = {}
    workspace = cache.get("workspace")
    required = fable_frontend_workspace_bytes(e)
    if workspace is None or workspace.numel() < required:
        workspace = cache["workspace"] = torch.zeros(required, dtype=torch.uint8, device=device)
    return workspace


def fable_router_quant_topk_frontend(hidden: torch.Tensor, router_weight: torch.Tensor,
                                     sym_buffer, quant: str = "mxfp4", stamps=None,
                                     l2_persist=None, wlayout=None, select_in_mega=None, zero_row_unrouted=None):
    """Fable dynamic-M fused Router + Quant + TopK8 + Softmax frontend (one launch).

    The launch shape follows the problem shape (``fable_frontend_path``): the CUDA-core K-split
    router for m <= 2 rows (h 3072, e 384, top-8), the swapab mma.sync router on the 96 x 4 grid
    for m <= 16 (h 3072, top-8), the WMMA router otherwise.
    ``stamps`` (env ``DG_FE_STAMPS``, default 0): record per-CTA %globaltimer phase stamps into
    the workspace; read them back with ``fable_frontend_stamps``.
    ``l2_persist`` (env ``DG_FE_ROUTER_L2_PERSIST``, default 1 on the cc path, else 0): 1 = pin the
    router weights in L2 with the persisting-L2 set-aside (access policy window launch attribute).
    ``wlayout`` (env ``DG_FE_ROUTER_WLAYOUT``, default ``fragment``; swapab path only -- the other
    paths read row-major weights and ignore it): ``fragment`` = the router weights are permuted ONCE
    on the host (``fable_router_weight_fragment_layout``, cached per weight tensor) into m16n8k16
    A-fragment order so every warp load instruction is one contiguous 512 B run; identical numerics;
    ``pre`` = the caller already passes the permuted tensor (no per-call work in this wrapper);
    ``row`` = row-major.
    ``select_in_mega`` (env ``DG_FE_SELECT_IN_MEGA``, default 0; cc path only): the frontend stops
    after the router CTAs stored the 384 keys per token (no ticket / last-arriver select) and the
    fused MegaMoE prologue selects the top-8 + softmax (``mxfp4|qoq_mega_moe_fused`` pick the key
    array up from this buffer's cache). topk_idx / topk_weights are NOT valid after such a call.
    ``zero_row_unrouted`` (env ``DG_FE_ZERO_ROW_UNROUTED``, default 1; cc path only): an all-zero hidden
    row (the E2E harness's padding rows on the ranks that own no token) is left unrouted -- topk_idx -1 /
    weight 0, or all-zero keys under ``select_in_mega`` -- instead of tie-broken onto experts 0..7 (all on
    rank 0, weight 1/8). Exact: x = 0 gives y = 0 either way; non-zero rows are bit-identical.
    """
    assert quant in ("mxfp4", "qoq")
    m, h = hidden.shape
    e = router_weight.size(0)
    topk = sym_buffer.topk_idx.size(1)
    path = fable_frontend_path(m, h, e, topk)
    if stamps is None:
        stamps = int(os.environ.get("DG_FE_STAMPS", "0"))
    if l2_persist is None:
        l2_persist = os.environ.get("DG_FE_ROUTER_L2_PERSIST")
        l2_persist = (1 if path == FE_PATH_CC else 0) if l2_persist is None else int(l2_persist)
    wlayout = _fe_wlayout_from_env(wlayout)
    if path == FE_PATH_SWAPAB:
        if wlayout == 1:
            router_weight = _fe_router_weight_for_layout(router_weight, 1)
        wlayout = 1 if wlayout in (1, 2) else 0
    elif wlayout == 2:
        raise ValueError(f"wlayout='pre' (fragment-layout router weight) needs the swapab path: m={m} h={h} e={e} topk={topk}")
    else:
        wlayout = 0
    if select_in_mega is None:
        select_in_mega = int(os.environ.get("DG_FE_SELECT_IN_MEGA", "0"))
    select_in_mega = int(bool(select_in_mega) and path == FE_PATH_CC)
    if zero_row_unrouted is None:
        zero_row_unrouted = int(os.environ.get("DG_FE_ZERO_ROW_UNROUTED", "1"))
    zero_row_unrouted = int(bool(zero_row_unrouted) and path == FE_PATH_CC)
    workspace = fable_frontend_workspace(sym_buffer, e, hidden.device)
    cache = sym_buffer._fable_frontend_cache
    if cache.get("keys") is None or cache["keys"].data_ptr() != workspace[_FRONTEND_CC_KEYS_OFF:].data_ptr():
        cache["keys"] = workspace[_FRONTEND_CC_KEYS_OFF:_FRONTEND_CC_KEYS_OFF + 2 * 512 * 4].view(torch.int32)   # flat, token t at [t * e, (t + 1) * e)
    cache["keys_active"] = cache["keys"] if select_in_mega else None
    views = cache.get(m)
    if views is None:
        views = cache[m] = (sym_buffer.x[:m], sym_buffer.x_sf[:m],
                            sym_buffer.topk_idx[:m], sym_buffer.topk_weights[:m])
    _C.fable_router_quant_topk_frontend(
        hidden, router_weight, views[0], views[1], views[2], views[3], workspace,
        0 if quant == "mxfp4" else 1, int(bool(stamps)), int(l2_persist), wlayout, select_in_mega, zero_row_unrouted)


def fable_frontend_keys(sym_buffer) -> torch.Tensor:
    """Flat [1024] int32 view of the cc router's compact key array (token t, expert x at [t * e + x], e = 384)
    in this buffer's frontend workspace; the array the fused Mega selects from under DG_FE_SELECT_IN_MEGA=1."""
    return sym_buffer._fable_frontend_cache["keys"]


def fable_frontend_stamps(sym_buffer, e: int) -> torch.Tensor:
    """[num_ctas, 8] int64 ns %globaltimer stamps of the last DG_FE_STAMPS=1 launch.

    cc path: router CTAs (``fable_frontend_router_ctas``): start / chunk0 landed / logits done /
    keys written (+ ticket) / - / loads issued / %smid / prologue done; spare CTA: start / - / - / - /
    - / quant done / %smid, and the last-arriving router CTA's select in its slots 1..4 (ticket won /
    keys read / select done / top-k written).
    swapab / WMMA grid: router CTAs (first 96 for E=384, m <= 16): start / chunk0 landed / mma done /
    ticket bumped; quant CTAs (next m): start / quant done / ticket seen / top-k done /
    [tiny: partials loaded / rounds done].
    """
    workspace = sym_buffer._fable_frontend_cache["workspace"]
    off = 256 + 4 * 64 * e * 4
    return workspace[off:off + _FRONTEND_STAMPS_BYTES].view(torch.int64).view(-1, 8).clone()

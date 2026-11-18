"""Two-node EP16 on the H20 fused MegaMoE (`mxfp4_mega_moe_fused` / `qoq_mega_moe_fused`).

Global rank r = node * 8 + local, 384 experts, 24 per rank: expert e lives on rank e // 24, i.e.
node e // 192. Every rank exchanges its rows once per call with its rail peer r ^ 8 (same local
GPU index on the other node, through the NIC behind that GPU's own PCIe switch):

1. split: one Triton kernel splits the top-k of the T own rows into the routes into this node
   (written to buffer rows [0, T), re-based to the node's 192 experts, the others -1) and the
   routes into the peer node (re-based to the peer's 192 experts, sent with the rows);
2. dispatch: one NCCL group sends x / x_sf of rows [0, T) and the peer routes straight from /
   into the buffers: the peer's rows land in buffer rows [T, 2T), no packing or unpacking;
3. one node-local fused MegaMoE launch over the 8 NVLink ranks, 192 experts, 2T rows per rank;
4. combine: one NCCL group sends the partial outputs of the peer's rows back and receives the
   partials of the own rows; y = own partial + peer partial (one add, fp32 then bf16).

A token crosses the NIC once even when several of its experts are on the other node.
Callers write x / x_sf of the own rows into `buffer.x[:T]` / `buffer.x_sf[:T]` (as for the EP8
API) and call `run`; `__call__` copies them first.
"""
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from . import fused

EP16_RANKS = 16
NODE_RANKS = 8


@triton.jit
def _split_routes(idx_ptr, w_ptr, here_idx_ptr, here_w_ptr, there_idx_ptr, there_w_ptr,
                  n, node_experts, node, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    e = tl.load(idx_ptr + offs, mask=mask, other=-1)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    valid = e >= 0
    owner = e // node_experts
    here = valid & (owner == node)
    there = valid & (owner != node)
    tl.store(here_idx_ptr + offs, tl.where(here, e - node * node_experts, -1), mask=mask)
    tl.store(here_w_ptr + offs, tl.where(here, w, 0.0), mask=mask)
    tl.store(there_idx_ptr + offs, tl.where(there, e - (1 - node) * node_experts, -1), mask=mask)
    tl.store(there_w_ptr + offs, tl.where(there, w, 0.0), mask=mask)


class H20EP16:
    def __init__(self, world_group, node_group, num_experts, max_tokens, topk, hidden, intermediate, quant="mxfp4"):
        assert dist.get_world_size(world_group) == EP16_RANKS and dist.get_world_size(node_group) == NODE_RANKS
        assert num_experts % EP16_RANKS == 0
        self.rank = dist.get_rank(world_group)
        self.node = self.rank // NODE_RANKS
        self.peer = self.rank ^ NODE_RANKS
        self.world_group = world_group
        self.node_experts = num_experts // 2
        self.t, self.topk, self.hidden = max_tokens, topk, hidden
        self.kernel = fused.qoq_mega_moe_fused if quant == "qoq" else fused.mxfp4_mega_moe_fused
        # node-local buffer: 192 experts over the 8 NVLink ranks, own + peer rows
        self.buffer = fused.get_fused_symm_buffer_for_mega_moe(
            node_group, self.node_experts, 2 * max_tokens, topk, hidden, intermediate)
        self.send_idx = torch.empty(max_tokens, topk, dtype=torch.int64, device="cuda")
        self.send_w = torch.empty(max_tokens, topk, dtype=torch.float32, device="cuda")
        self.y_node = torch.empty(2 * max_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        self.y_back = torch.empty(max_tokens, hidden, dtype=torch.bfloat16, device="cuda")

    def _sendrecv(self, pairs):
        ops = []
        for send, recv in pairs:
            ops.append(dist.P2POp(dist.isend, send, self.peer, self.world_group))
            ops.append(dist.P2POp(dist.irecv, recv, self.peer, self.world_group))
        for w in dist.batch_isend_irecv(ops):
            w.wait()

    def run(self, y, topk_idx, topk_weights, weights, activation_clamp=10.0):
        """x / x_sf of the own T rows already in buffer rows [0, T). y [T, H] bf16 out; topk_idx [T, k] int64
        global experts (-1 = unused slot); topk_weights [T, k] fp32; weights = this rank's 24 experts."""
        t, b = self.t, self.buffer
        n = t * self.topk
        _split_routes[(triton.cdiv(n, 128),)](
            topk_idx, topk_weights, b.topk_idx, b.topk_weights, self.send_idx, self.send_w,
            n, self.node_experts, self.node, BLOCK=128)
        u8 = lambda v: v.view(torch.uint8)  # noqa: E731  (NCCL has no FP8 type)
        self._sendrecv([(u8(b.x[:t]), u8(b.x[t:2 * t])), (b.x_sf[:t], b.x_sf[t:2 * t]),
                        (self.send_idx, b.topk_idx[t:2 * t]), (self.send_w, b.topk_weights[t:2 * t])])
        self.kernel(self.y_node, *weights, b, activation_clamp=activation_clamp)
        self._sendrecv([(self.y_node[t:2 * t], self.y_back)])
        torch.add(self.y_node[:t], self.y_back, out=y)

    def __call__(self, y, x, x_sf, topk_idx, topk_weights, weights, activation_clamp=10.0):
        self.buffer.x[:self.t].copy_(x)
        self.buffer.x_sf[:self.t].copy_(x_sf)
        self.run(y, topk_idx, topk_weights, weights, activation_clamp)

    def destroy(self):
        self.buffer.destroy()

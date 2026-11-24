"""Two-node EP16 on the H20 fused MegaMoE (`mxfp4_mega_moe_fused` / `qoq_mega_moe_fused`).

Global rank r = node * 8 + local, 384 experts, 24 per rank: expert e lives on rank e // 24, i.e.
node e // 192. Every rank exchanges its rows once per call with its rail peer r ^ 8 (same local
GPU index on the other node, through the NIC behind that GPU's own PCIe switch):

1. pack: one Triton kernel splits the top-k of the T own rows into the routes into this node
   (written to the own-row buffer, re-based to the node's 192 experts, the others -1) and the
   routes into the peer node (re-based to the peer's 192 experts), and packs x / x_sf of the own
   rows and the peer routes into one contiguous message;
2. dispatch: one NCCL send / recv of that message with the rail peer; one Triton kernel unpacks
   the received message into the peer-row buffer;
3. node-local fused MegaMoE over the 8 NVLink ranks, 192 experts;
4. combine: one NCCL send / recv returns the partial outputs of the peer's rows and receives the
   partials of the own rows; y = own partial + peer partial (one add, fp32 then bf16).

overlap=False: own and peer rows share one buffer (rows [0, T) own, [T, 2T) peer) and one kernel
launch over 2T rows runs after the dispatch.
overlap=True: own and peer rows live in two buffers of T rows; the kernel over the own rows runs
while the dispatch is in flight, the kernel over the peer rows after it. Both launches leave
`reserved_sms` SMs free: the MegaMoE grid needs every CTA resident (grid / NVLink barriers), so a
dispatch kernel holding an SM on one rank while a peer rank's MegaMoE launch holds all of them
deadlocks across the two nodes. The NCCL kernel must fit in the reserved SMs
(NCCL_MAX_NCHANNELS <= reserved_sms, set by the launcher).

transport='gin': everything above runs inside the one fused kernel launch (DG_H20_EP16_GIN): the
kernel puts the own rows to the rail peer with one NCCL GIN put (GPU-initiated RDMA) and copies the
peer's rows from the NIC receive region into rows [T, 2T) itself, maps the global top-k to this
node's experts on read, puts each combined peer row back and adds the peer's partial to each own row
before writing y. No NCCL kernel, no host wait, no pack / unpack / add kernels. The rail pairs get
their own 2-rank NCCL communicators; the NIC buffers are one NCCL symmetric window on them
(h20_ep16_gin_session.cu).

A token crosses the NIC once even when several of its experts are on the other node.
Callers write x / x_sf of the own rows into `buffer.x[:T]` / `buffer.x_sf[:T]` (as for the EP8
API) and call `run`; `__call__` copies them first.
"""
import contextlib
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .. import get_num_sms, set_num_sms
from . import fused

EP16_RANKS = 16
NODE_RANKS = 8
BLOCK = 256
_GIN_SESSION = []


def _gin_session():
    # host extension (NCCL symmetric window + device communicator), built once per process
    if not _GIN_SESSION:
        import nvidia.nccl
        from torch.utils.cpp_extension import load
        root = list(nvidia.nccl.__path__)[0]
        lib = os.path.join(root, 'lib', 'libnccl.so.2')
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'include', 'deep_gemm', 'impls', 'h20_ep16_gin_session.cu')
        build = os.path.join(os.path.expanduser('~'), '.deep_gemm', 'h20_ep16_gin_session')
        os.makedirs(build, exist_ok=True)
        _GIN_SESSION.append(load(name='h20_ep16_gin_session', sources=[src], build_directory=build,
                                 extra_include_paths=[os.path.join(root, 'include')],
                                 extra_ldflags=[lib, f'-Wl,-rpath,{os.path.dirname(lib)}']))
    return _GIN_SESSION[0]


def nccl_jit_flags():
    """The JIT flag that lets the fused kernel include NCCL's device API. Set DG_JIT_EXTRA_FLAGS to this
    before the first DeepGEMM kernel of the process is built (the JIT compiler reads it once)."""
    import nvidia.nccl
    return '-I' + os.path.join(list(nvidia.nccl.__path__)[0], 'include')


@triton.jit
def _pack(x_ptr, sf_ptr, idx_ptr, w_ptr, here_idx_ptr, here_w_ptr, pack_ptr, pack_f_ptr,
          nx, nsf, n, node_experts, node, BLOCK: tl.constexpr):
    # pack (int32 words): [x: nx][x_sf: nsf][peer routes: n][peer weights: n]
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mx = offs < nx
    tl.store(pack_ptr + offs, tl.load(x_ptr + offs, mask=mx), mask=mx)
    o = offs - nx
    ms = (o >= 0) & (o < nsf)
    tl.store(pack_ptr + nx + o, tl.load(sf_ptr + o, mask=ms), mask=ms)
    o = offs - nx - nsf
    mr = (o >= 0) & (o < n)
    e = tl.load(idx_ptr + o, mask=mr, other=-1)
    w = tl.load(w_ptr + o, mask=mr, other=0.0)
    valid = e >= 0
    owner = e // node_experts
    here = valid & (owner == node)
    there = valid & (owner != node)
    tl.store(here_idx_ptr + o, tl.where(here, e - node * node_experts, -1), mask=mr)
    tl.store(here_w_ptr + o, tl.where(here, w, 0.0), mask=mr)
    tl.store(pack_ptr + nx + nsf + o, tl.where(there, e - (1 - node) * node_experts, -1).to(tl.int32), mask=mr)
    tl.store(pack_f_ptr + nx + nsf + n + o, tl.where(there, w, 0.0), mask=mr)


@triton.jit
def _unpack(pack_ptr, pack_f_ptr, x_ptr, sf_ptr, idx_ptr, w_ptr, nx, nsf, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mx = offs < nx
    tl.store(x_ptr + offs, tl.load(pack_ptr + offs, mask=mx), mask=mx)
    o = offs - nx
    ms = (o >= 0) & (o < nsf)
    tl.store(sf_ptr + o, tl.load(pack_ptr + nx + o, mask=ms), mask=ms)
    o = offs - nx - nsf
    mr = (o >= 0) & (o < n)
    tl.store(idx_ptr + o, tl.load(pack_ptr + nx + nsf + o, mask=mr), mask=mr)
    tl.store(w_ptr + o, tl.load(pack_f_ptr + nx + nsf + n + o, mask=mr), mask=mr)


def _words(v):
    return v.contiguous().view(torch.uint8).reshape(-1).view(torch.int32)


class H20EP16:
    def __init__(self, world_group, node_group, num_experts, max_tokens, topk, hidden, intermediate,
                 quant="mxfp4", overlap=False, reserved_sms=2, transport="nccl"):
        assert dist.get_world_size(world_group) == EP16_RANKS and dist.get_world_size(node_group) == NODE_RANKS
        assert num_experts % EP16_RANKS == 0
        self.rank = dist.get_rank(world_group)
        self.node = self.rank // NODE_RANKS
        self.peer = self.rank ^ NODE_RANKS
        self.world_group = world_group
        self.node_experts = num_experts // 2
        self.t, self.topk, self.hidden, self.overlap = max_tokens, topk, hidden, overlap
        self.transport, self.gin_args = transport, None
        assert transport in ("nccl", "gin") and not (overlap and transport == "gin")
        self.kernel_sms = get_num_sms() - reserved_sms if overlap else get_num_sms()
        self.kernel = fused.qoq_mega_moe_fused if quant == "qoq" else fused.mxfp4_mega_moe_fused
        t = max_tokens
        if transport == "gin":
            assert os.environ.get("DG_JIT_EXTRA_FLAGS", "").find(nccl_jit_flags()) >= 0, \
                "set DG_JIT_EXTRA_FLAGS=deep_gemm.mega.ep16.nccl_jit_flags() before the first kernel build"
            self._init_gin(t, hidden, topk)
        make = lambda rows: fused.get_fused_symm_buffer_for_mega_moe(  # noqa: E731
            node_group, self.node_experts, rows, topk, hidden, intermediate)
        if overlap:
            # own rows and peer rows in separate buffers: two launches of T rows
            self.buffer, self.peer_buffer = make(t), make(t)
            own, peer = self.buffer, self.peer_buffer
            self.peer_rows = (peer.x[:t], peer.x_sf[:t], peer.topk_idx[:t], peer.topk_weights[:t])
            self.y_own = torch.empty(t, hidden, dtype=torch.bfloat16, device="cuda")
            self.y_peer = torch.empty(t, hidden, dtype=torch.bfloat16, device="cuda")
        else:
            # own rows [0, T), peer rows [T, 2T) of one buffer: one launch of 2T rows
            self.buffer = own = make(2 * t)
            self.peer_rows = (own.x[t:2 * t], own.x_sf[t:2 * t], own.topk_idx[t:2 * t], own.topk_weights[t:2 * t])
            self.y_node = torch.empty(2 * t, hidden, dtype=torch.bfloat16, device="cuda")
            self.y_own, self.y_peer = self.y_node[:t], self.y_node[t:2 * t]
        self.x_words, self.sf_words = _words(own.x[:t]), _words(own.x_sf[:t])
        self.nx, self.nsf, self.n = self.x_words.numel(), self.sf_words.numel(), t * topk
        words = self.nx + self.nsf + 2 * self.n
        self.send_pack = torch.empty(words, dtype=torch.int32, device="cuda")
        self.recv_pack = torch.empty(words, dtype=torch.int32, device="cuda")
        px, psf, _, _ = self.peer_rows
        self.peer_x_words, self.peer_sf_words = _words(px), _words(psf)
        assert self.peer_x_words.data_ptr() == px.data_ptr() and self.peer_sf_words.data_ptr() == psf.data_ptr()
        self.y_back = torch.empty(t, hidden, dtype=torch.bfloat16, device="cuda")

    def _init_gin(self, t, hidden, topk):
        # 2-rank rail communicators (every rank creates all of them), the NIC window on this rank's
        local = self.rank % NODE_RANKS
        for i in range(NODE_RANKS):
            g = dist.new_group([i, i + NODE_RANKS])
            if i == local:
                rail = g
        dist.barrier(group=rail)  # ProcessGroupNCCL creates its communicator lazily
        comm = int(rail._get_backend(torch.device("cuda"))._comm_ptr())
        session = _gin_session()
        # GIN contexts (one QP each): the kernel spreads its signals over them (signal s -> context s % n)
        gin_contexts = int(os.environ.get("DG_EP16_GIN_CONTEXTS", "4"))
        row = hidden + hidden // 128 * 4 + topk * 12
        nbytes = 2 * t * row + 2 * t * hidden * 2 + 16  # see H20Ep16Gin (+ epoch, send counter)
        self._gin = (session, session.create(comm, self.node, 2, nbytes, gin_contexts))
        base, win, dev_comm, contexts, gin_type = (int(v) for v in session.info(self._gin[1]))
        assert contexts >= 1 and gin_type in (2, 3), f"NCCL GIN type {gin_type}: needs proxy (2) or GDAKI (3)"
        self.gin_args = [dev_comm, win, base, self.node, t, gin_type, contexts]
        self.rail = rail
        dist.barrier(group=rail)

    def _sendrecv(self, send, recv):
        return dist.batch_isend_irecv([dist.P2POp(dist.isend, send, self.peer, self.world_group),
                                       dist.P2POp(dist.irecv, recv, self.peer, self.world_group)])

    # stages (also timed one by one by tests/test_h20_ep16.py)
    def pack(self, topk_idx, topk_weights):
        b = self.buffer
        _pack[(triton.cdiv(self.nx + self.nsf + self.n, BLOCK),)](
            self.x_words, self.sf_words, topk_idx, topk_weights, b.topk_idx, b.topk_weights,
            self.send_pack, self.send_pack.view(torch.float32),
            self.nx, self.nsf, self.n, self.node_experts, self.node, BLOCK=BLOCK)

    def dispatch(self):
        return self._sendrecv(self.send_pack, self.recv_pack)

    def unpack(self):
        _, _, idx, w = self.peer_rows
        _unpack[(triton.cdiv(self.nx + self.nsf + self.n, BLOCK),)](
            self.recv_pack, self.recv_pack.view(torch.float32), self.peer_x_words, self.peer_sf_words, idx, w,
            self.nx, self.nsf, self.n, BLOCK=BLOCK)

    def combine(self):
        return self._sendrecv(self.y_peer, self.y_back)

    @contextlib.contextmanager
    def _kernel_sms(self):
        device_sms = get_num_sms()
        set_num_sms(self.kernel_sms)
        try:
            yield
        finally:
            set_num_sms(device_sms)

    def node_kernel(self, y, weights, buffer, activation_clamp=10.0):
        with self._kernel_sms():
            self.kernel(y, *weights, buffer, activation_clamp=activation_clamp)

    def run(self, y, topk_idx, topk_weights, weights, activation_clamp=10.0, phase_stamps=None):
        """x / x_sf of the own T rows already in buffer rows [0, T). y [T, H] bf16 out; topk_idx [T, k] int64
        global experts (-1 = unused slot); topk_weights [T, k] fp32; weights = this rank's 24 experts.
        transport 'gin': topk_idx / topk_weights already in buffer rows [0, T) as well (pass None)."""
        if self.transport == "gin":
            if topk_idx is not None:
                self.buffer.topk_idx[:self.t].copy_(topk_idx)
                self.buffer.topk_weights[:self.t].copy_(topk_weights)
            self.kernel(y, *weights, self.buffer, activation_clamp=activation_clamp, phase_stamps=phase_stamps,
                        ep16_gin=self.gin_args)
            return
        self.pack(topk_idx, topk_weights)
        works = self.dispatch()
        if self.overlap:
            self.node_kernel(self.y_own, weights, self.buffer, activation_clamp)
        for w in works:
            w.wait()
        self.unpack()
        if self.overlap:
            self.node_kernel(self.y_peer, weights, self.peer_buffer, activation_clamp)
        else:
            self.node_kernel(self.y_node, weights, self.buffer, activation_clamp)
        for w in self.combine():
            w.wait()
        torch.add(self.y_own, self.y_back, out=y)

    def __call__(self, y, x, x_sf, topk_idx, topk_weights, weights, activation_clamp=10.0):
        self.buffer.x[:self.t].copy_(x)
        self.buffer.x_sf[:self.t].copy_(x_sf)
        self.run(y, topk_idx, topk_weights, weights, activation_clamp)

    def destroy(self):
        if self.gin_args is not None:
            torch.cuda.synchronize()
            self._gin[0].destroy(self._gin[1])
            self.gin_args = None
        self.buffer.destroy()
        if self.overlap:
            self.peer_buffer.destroy()

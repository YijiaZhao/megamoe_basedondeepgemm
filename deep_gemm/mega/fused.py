import types
import torch
import torch.distributed as dist
from .. import _C
from ..utils import align

_SM90_FUSED_LAYOUT_ATTR = "_deep_gemm_fp4_h20_fused_layout"

def _interleave(t, gran=8):
    g,n,*rest=t.shape; half=n//2
    gate=t[:,:half].reshape(g,half//gran,gran,*rest); up=t[:,half:].reshape(g,half//gran,gran,*rest)
    return torch.stack([gate,up],dim=2).reshape(g,n,*rest).contiguous()

def _mxfp4_rf_fragment_order(fused, forward=True):
    """Permute each 80B row's 64 FP4 bytes into WGMMA RF-fragment order (self-inverse).

    Marlin order: word j of K32 group g holds K = g*32 + 8j + [0..8). RF order: word c
    holds K = g*32 + 4c + [0..4) then g*32 + 16 + 4c + [0..4), which is exactly the
    m64nNk32 A fragment of thread ``lane % 4 == c``, so one uint4 per row covers all
    four K32 steps. Applied before ``_braid`` (per-word) and ``_fused_word_transpose``
    (moves whole words so chunk c holds word c of all four groups). The SMEM tile
    decoders un-transpose in registers and gather (q0.x, q1.x, q2.x, q3.x) / (q0.y, ...).
    """
    e,n,k=fused.shape; rows=fused.view(e,n,k//80,80).clone()
    codes=rows[...,:64].reshape(e,n,k//80,4,4,4)          # (g, word j, byte b): nibbles hi/lo = K 8j+b / 8j+4+b
    hi=(codes>>4)&15; lo=codes&15                         # hi[g,j,b] = K 8j+b ; lo[g,j,b] = K 8j+4+b
    kvals=torch.cat((hi,lo),dim=-1).reshape(e,n,k//80,4,32)  # (g, K=0..31) in Marlin order
    kk=torch.arange(32,device=fused.device)
    # RF word c, slot t (t<4: K 4c+t ; t>=4: K 16+4c+t-4)
    c=kk//8; t=kk%8; src=torch.where(t<4,4*c+t,16+4*c+(t-4))
    perm=kvals[...,src] if forward else kvals[...,torch.argsort(src)]
    perm=perm.reshape(e,n,k//80,4,4,8); hi=perm[...,:4]; lo=perm[...,4:]   # (group g, word c, 8 nibbles)
    rows[...,:64]=((hi<<4)|lo).reshape(e,n,k//80,64)
    return rows.view(e,n,k).contiguous()

def _braid(fused):
    e,n,k=fused.shape; rows=fused.view(e,n,k//80,80).clone(); packed=rows[...,:64].view(e,n,k//80,16,4)
    codes=torch.cat(((packed>>4)&15,packed&15),dim=-1); mag=codes&7; sign=codes>>3
    bs=torch.stack((sign[...,4],sign[...,0],sign[...,5],sign[...,1],sign[...,6],sign[...,2],sign[...,7],sign[...,3]),dim=-1)
    nib=mag|(bs<<3); rows[...,:64]=(nib[...,0::2]|(nib[...,1::2]<<4)).reshape(e,n,k//80,64)
    return rows.view(e,n,k).contiguous()

def _fused_word_transpose(fused):
    """Transpose the 4-byte words of each 80B row's 64 FP4 bytes: [K32 group g][word c] -> [c][g].

    After ``_mxfp4_rf_fragment_order`` word c of group g is the m64nNk32 A fragment of
    thread ``lane % 4 == c`` for K32 step g. The RS loader (``issue_swap_ab_half_rf``)
    reads one uint4 at byte ``c*16`` and uses word k for K32 step k, so chunk c must hold
    word c of all four groups. Scale bytes 64..79 are untouched. Self-inverse; commutes
    with ``_braid`` (per-word) and with the fragment reorder (moves whole words).
    """
    from ..quantization_mxfp4 import _mxfp4_word_transpose
    e,n,k=fused.shape; rows=fused.view(e,n,k//80,80).clone()
    rows[...,:64]=_mxfp4_word_transpose(rows[...,:64].contiguous())
    return rows.view(e,n,k).contiguous()

# Fused weights are dense tile-major bytes (E, N/256, K/128, 256, 80) under a nominal
# (E, N, K/128*80) shape; the helpers above only ever permute bytes *within* an 80 B
# row (viewing the tensor as (e, n, k//80, 80) is layout-agnostic), and `_interleave`
# runs on the pre-fuse packed/scale tensors, so the dense order is preserved.
def _check_fused_dense(f, block_n=256):
    e,n,k=f.shape; assert f.is_contiguous() and n%block_n==0 and k%80==0, f.shape; return f

def transform_mxfp4_weights_for_mega_moe_fused(l1,l2):
    from ..quantization_mxfp4_fused import mxfp4_row_reference_exponent,mxfp4_scale_to_relative_index,mxfp4_scale_to_tile_major,mxfp4_fuse_packed_with_scale_tile_major
    p1,s1=l1; p2,s2=l2; p1,s1=_interleave(p1),_interleave(s1)
    def prep(p,s):
        er=mxfp4_row_reference_exponent(s); idx=mxfp4_scale_to_relative_index(s,er); tm=mxfp4_scale_to_tile_major(idx,block_n=256,block_k=128); f=_check_fused_dense(_fused_word_transpose(_braid(_mxfp4_rf_fragment_order(mxfp4_fuse_packed_with_scale_tile_major(p,tm,block_k=128))))); return f,tm,torch.exp2(er.float()).contiguous()
    return prep(p1,s1),prep(p2,s2)

def transform_qoq_weights_for_mega_moe_fused(l1,l2):
    from ..quantization_qoq_fused import qoq_meta_to_tile_major,qoq_fuse_packed_with_meta_tile_major
    def prep(p,s2,z,s1):
        # RF fragment order + word transpose like MXFP4 (plain nibbles: no `_braid`, the
        # QoQ decoders shift/mask; meta bytes 64/65 are untouched by both helpers).
        tm=qoq_meta_to_tile_major(s2,z,block_n=256); return _check_fused_dense(_fused_word_transpose(_mxfp4_rf_fragment_order(qoq_fuse_packed_with_meta_tile_major(p.contiguous(),tm)))),tm,s1.float().contiguous()
    p,s2,z,s1=l1; return prep(_interleave(p),_interleave(s2),_interleave(z),_interleave(s1.unsqueeze(-1)).squeeze(-1)),prep(*l2)

class FusedSymmBuffer:
    def __init__(self, group, num_experts, max_tokens, topk, hidden, intermediate):
        self.group=group; self.num_experts=num_experts; self.num_max_tokens_per_rank=max_tokens; self.num_topk=topk; self.hidden=hidden; self.intermediate_hidden=intermediate
        n,slice_fn=_C.get_symm_buffer_size_for_fused_mega_moe(group.size(),num_experts,max_tokens,topk,hidden,intermediate)
        from torch.distributed._symmetric_memory import empty as symm_empty, rendezvous
        self.buffer=symm_empty(n,dtype=torch.int8,device='cuda'); self.handle=rendezvous(self.buffer,group=group); self.buffer.zero_(); group.barrier(); torch.cuda.synchronize()
        self.x,self.x_sf,self.topk_idx,self.topk_weights,self.l1_acts,self.l1_acts_sf,self.l2_acts,self.l2_acts_sf,self.combine_partials=slice_fn(self.buffer)
    def destroy(self): self.handle=None; self.buffer=None; self.group=None

def get_fused_symm_buffer_for_mega_moe(group,num_experts,max_tokens,topk,hidden,intermediate):
    return FusedSymmBuffer(group,num_experts,max_tokens,topk,hidden,intermediate)

def _fe_keys_for(b):
    """DG_FE_SELECT_IN_MEGA=1: the compact key array the Fable cc frontend left for this
    buffer (set by fable_router_quant_topk_frontend when it ran with select_in_mega), else None."""
    cache = getattr(b, "_fable_frontend_cache", None)
    return cache.get("keys_active") if cache else None

def mxfp4_mega_moe_fused(y,l1,l2,b,cumulative_local_expert_recv_stats=None,activation_clamp=10.0,fast_math=True,phase_stamps=None,fe_keys=None,ep16_gin=None):
    w1,sf1,rs1=l1; w2,sf2,rs2=l2
    if fe_keys is None: fe_keys=_fe_keys_for(b)
    _C.mxfp4_mega_moe_fused(y,(w1,sf1),(w2,sf2),cumulative_local_expert_recv_stats,rs1,rs2,b.buffer,b.handle.buffer_ptrs,b.group.rank(),b.num_max_tokens_per_rank,b.num_experts,b.num_topk,activation_clamp,fast_math,phase_stamps,fe_keys,ep16_gin or [])

def qoq_mega_moe_fused(y,l1,l2,b,cumulative_local_expert_recv_stats=None,activation_clamp=10.0,fast_math=True,phase_stamps=None,fe_keys=None,ep16_gin=None):
    w1,sf1,rs1=l1; w2,sf2,rs2=l2
    if fe_keys is None: fe_keys=_fe_keys_for(b)
    _C.qoq_mega_moe_fused(y,(w1,sf1),(w2,sf2),cumulative_local_expert_recv_stats,rs1,rs2,b.buffer,b.handle.buffer_ptrs,b.group.rank(),b.num_max_tokens_per_rank,b.num_experts,b.num_topk,activation_clamp,fast_math,phase_stamps,fe_keys,ep16_gin or [])

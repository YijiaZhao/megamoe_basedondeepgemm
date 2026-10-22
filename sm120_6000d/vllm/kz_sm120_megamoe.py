# SPDX-License-Identifier: Apache-2.0
"""SM120 decode MegaMoE (one persistent kernel: MXFP8 dispatch over PCIe P2P, W4A8 swap-AB FC1+SwiGLU+FC2,
P2P push back, top-k combine) behind vLLM's FlashInfer MoE-EP adapter as --moe-backend kz_sm120_megamoe.

The CUDA extension (sm120_mega_moe.cu) is prebuilt against this image's torch; KZ_SM120_MM_SO points at the .so.
Every MoE layer shares one set of peer-visible buffers (layers run in stream order); the per-layer state is only
the repacked weights.
"""

from __future__ import annotations

import importlib.util
import os

import torch

from vllm.distributed import get_ep_group

_EXT = None
_INIT_KEY = None


def _ext():
    global _EXT
    if _EXT is None:
        path = os.environ.get("KZ_SM120_MM_SO", "/raid/kimiz_mm/sm120mm/build/sm120mm.so")
        spec = importlib.util.spec_from_file_location("sm120mm", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
    return _EXT


def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:  # [E, N, K/2] uint8 -> [E, N, K] codes (low nibble first)
    p = packed.view(torch.uint8)
    return torch.stack((p & 0xF, p >> 4), dim=-1).flatten(-2)


def _pack_w(codes: torch.Tensor) -> torch.Tensor:  # [E, N, K] codes (kernel row order) -> lane-fragment layout
    E, N, K = codes.shape
    x = codes.view(E, N // 16, 2, 8, K // 64, 2, 2, 4, 4)  # [E, nb, half, g, kc, s, p, c, j]
    b = x[:, :, 0] | (x[:, :, 1] << 4)
    return b.permute(0, 1, 3, 2, 6, 4, 5, 7).contiguous().view(E, -1)


def _pack_s(sf: torch.Tensor) -> torch.Tensor:  # [E, N, K/32] ue8m0 -> [E, nb, K/128, 16, 4]
    E, N, KS = sf.shape
    return sf.view(torch.uint8).view(E, N // 16, 16, KS // 4, 4).permute(0, 1, 3, 2, 4).contiguous().view(E, -1)


def _w13_rows(I: int, device) -> torch.Tensor:  # kernel row -> checkpoint row (gate 0..I-1, up I..2I-1)
    r = torch.arange(2 * I, device=device)
    blk, w = r // 16, r % 16
    return torch.where(w < 8, blk * 8 + w, I + blk * 8 + (w - 8))


class KzSm120MegaMoe:
    def __init__(self, moe, w13, w2, w13_scale, w2_scale) -> None:
        ext = _ext()
        ep = get_ep_group()
        E_local, two_i, half_h = w13.shape
        H, I = half_h * 2, two_i // 2
        assert H == moe.hidden_dim and w2.shape == (E_local, H, I // 2), (w13.shape, w2.shape)
        top_k = moe.experts_per_token
        cap = moe.max_num_tokens
        key = (ep.rank_in_group, ep.world_size, E_local, cap, H, I, top_k)
        global _INIT_KEY
        if _INIT_KEY is None:
            handle = ext.init(ep.rank_in_group, ep.world_size, E_local, cap, H, I, top_k)
            handles = [None] * ep.world_size
            torch.distributed.all_gather_object(handles, handle, group=ep.cpu_group)
            ext.open_peers(handles)
            _INIT_KEY = key
        assert _INIT_KEY == key, (_INIT_KEY, key)
        perm = _w13_rows(I, w13.device)
        # repacked bytes keep each expert's byte count, so they go back into the checkpoint's 3-D shapes
        # (the modular-kernel path checks w1/w2 ranks); the kernel only uses the pointers
        self.w13 = _pack_w(_unpack_e2m1(w13)[:, perm]).view(E_local, 2 * I, H // 2)
        self.ws13 = _pack_s(w13_scale[:, perm]).view(E_local, 2 * I, H // 32)
        self.w2 = _pack_w(_unpack_e2m1(w2)).view(E_local, H, I // 2)
        self.ws2 = _pack_s(w2_scale).view(E_local, H, I // 32)
        self.limit = float(moe.swiglu_limit) if moe.swiglu_limit is not None else 3.0e38
        self.cap = cap

    def __call__(self, hidden_states, topk_ids, topk_weights):
        assert hidden_states.shape[0] <= self.cap
        out = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        _ext().run(
            hidden_states.to(torch.bfloat16).contiguous(),
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            out,
            self.w13,
            self.ws13,
            self.w2,
            self.ws2,
            self.limit,
            None,
        )
        return out

    def kernel_weights(self):
        return self.w13, self.ws13, self.w2, self.ws2


def error_code() -> int:
    return int(_ext().counters()[3])

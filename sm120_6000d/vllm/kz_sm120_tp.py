# SPDX-License-Identifier: Apache-2.0
"""SM120 decode MoE for MoE-TP (--moe-backend kz_sm120_tp): every expert local with an I/TP slice, one kernel per
call (per-item MXFP8 quantization of the token K slice, split-K W4A8 swap-AB FC1 + SwiGLU, FC2 with the top-k weight
folded into an fp32 reduction). The output is this rank's partial sum; vLLM's MoE-TP all-reduce follows unchanged.
Calls are chunked to the kernel's 64-token capacity (decode steps fit in one call)."""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import FlashInferMoeEpExperts
from vllm.model_executor.layers.fused_moe.kz_sm120_megamoe import _ext, _pack_s, _pack_w, _unpack_e2m1, _w13_rows
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

_TP_CHUNK = 64
_TP_INIT = None


class KzSm120Tp:
    def __init__(self, moe, w13, w2, w13_scale, w2_scale) -> None:
        ext = _ext()
        E, two_i, half_h = w13.shape
        H, I = half_h * 2, two_i // 2
        assert H == moe.hidden_dim and w2.shape == (E, H, I // 2), (w13.shape, w2.shape)
        key = (E, H, I, moe.experts_per_token)
        global _TP_INIT
        if _TP_INIT is None:
            ext.init_tp(E, _TP_CHUNK, H, I, moe.experts_per_token)
            _TP_INIT = key
        assert _TP_INIT == key, (_TP_INIT, key)
        perm = _w13_rows(I, w13.device)
        self.w13 = _pack_w(_unpack_e2m1(w13)[:, perm]).view(E, 2 * I, H // 2)
        self.ws13 = _pack_s(w13_scale[:, perm]).view(E, 2 * I, H // 32)
        self.w2 = _pack_w(_unpack_e2m1(w2)).view(E, H, I // 2)
        self.ws2 = _pack_s(w2_scale).view(E, H, I // 32)
        self.limit = float(moe.swiglu_limit) if moe.swiglu_limit is not None else 3.0e38

    def __call__(self, hidden_states, topk_ids, topk_weights, out):
        x = hidden_states.to(torch.bfloat16)
        ids = topk_ids.to(torch.int32)
        w = topk_weights.to(torch.float32)
        ext = _ext()
        for s in range(0, x.shape[0], _TP_CHUNK):
            e = min(s + _TP_CHUNK, x.shape[0])
            ext.run_tp(x[s:e].contiguous(), ids[s:e].contiguous(), w[s:e].contiguous(), out[s:e],
                       self.w13, self.ws13, self.w2, self.ws2, self.limit, None)
        return out

    def kernel_weights(self):
        return self.w13, self.ws13, self.w2, self.ws2


class KzSm120TpExperts(FlashInferMoeEpExperts):
    """Standard-format modular experts for MoE-TP; output is the rank-local partial (not reduced)."""

    @staticmethod
    def is_supported_config(cls, moe_config, weight_key, activation_key, activation_format):
        return mk.FusedMoEExperts.is_supported_config(cls, moe_config, weight_key, activation_key, activation_format)

    @staticmethod
    def _supports_current_device() -> bool:
        cap = current_platform.get_device_capability()
        return cap is not None and cap.major == 12

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_ep and not moe_parallel_config.use_batched_activation_format

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        adapter = KzSm120Tp(self.moe_config, layer.w13_weight, layer.w2_weight, layer.w13_weight_scale,
                            layer.w2_weight_scale)
        for name, tensor in zip(("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
                                adapter.kernel_weights()):
            replace_parameter(layer, name, tensor)
        self._kz_tp = adapter

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, expert_map,
              a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input) -> None:
        self._kz_tp(hidden_states, topk_ids, topk_weights, output)

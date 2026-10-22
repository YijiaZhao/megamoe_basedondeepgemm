# Register the SM120 decode MegaMoE kernel (kz_sm120_megamoe.py + sm120_mega_moe.cu) in vLLM main as
# --moe-backend kz_sm120_megamoe, through the FlashInfer MoE-EP adapter path (passthrough all2all, MXFP4 weights
# handed over raw). Usage: python3 patch_main_kz120.py <site-packages copy root>
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "vllm"
kernel = root / "config/kernel.py"
adapter = root / "model_executor/layers/fused_moe/flashinfer_moe_ep.py"
oracle = root / "model_executor/layers/fused_moe/oracle/mxfp4.py"


def sub(path: Path, old: str, new: str) -> None:
    s = path.read_text()
    assert s.count(old) == 1, (path.name, old[:70], s.count(old))
    path.write_text(s.replace(old, new))


# config/kernel.py
sub(kernel, '    "flashinfer_moe_ep_mega_deep_gemm",\n', '    "flashinfer_moe_ep_mega_deep_gemm",\n    "kz_sm120_megamoe",\n')
sub(kernel, 'FLASHINFER_MOE_EP_DEEP_GEMM = "flashinfer_moe_ep_mega_deep_gemm"\n',
    'FLASHINFER_MOE_EP_DEEP_GEMM = "flashinfer_moe_ep_mega_deep_gemm"\nKZ_SM120_MEGAMOE = "kz_sm120_megamoe"\n')
sub(kernel, "        FLASHINFER_MOE_EP_CUTEDSL,\n        FLASHINFER_MOE_EP_DEEP_GEMM,\n    }\n)",
    "        FLASHINFER_MOE_EP_CUTEDSL,\n        FLASHINFER_MOE_EP_DEEP_GEMM,\n        KZ_SM120_MEGAMOE,\n    }\n)")

# fused_moe/flashinfer_moe_ep.py
sub(adapter, "    FLASHINFER_MOE_EP_DEEP_GEMM,\n)\nfrom vllm.distributed import get_ep_group\n",
    "    FLASHINFER_MOE_EP_DEEP_GEMM,\n    KZ_SM120_MEGAMOE,\n)\nfrom vllm.distributed import get_ep_group\n")
sub(adapter, 'FlashInferMoeEpKernel = Literal["cutedsl", "deep_gemm"]', 'FlashInferMoeEpKernel = Literal["cutedsl", "deep_gemm", "kz120"]')
sub(adapter, "_SUPPORTED_CAPABILITIES = frozenset({(10, 0), (10, 3)})",
    "_SUPPORTED_CAPABILITIES = frozenset({(10, 0), (10, 3), (12, 0), (12, 1)})")
sub(adapter, """        weight_formats=frozenset({"mxfp4"}),
    ),
}
""", """        weight_formats=frozenset({"mxfp4"}),
    ),
    KZ_SM120_MEGAMOE: FlashInferMoeEpBackendSpec(
        kernel="kz120",
        weight_formats=frozenset({"mxfp4"}),
    ),
}
""")
sub(adapter, '    if spec.kernel == "deep_gemm" and vllm_config.parallel_config.enable_eplb:',
    '    if spec.kernel in ("deep_gemm", "kz120") and vllm_config.parallel_config.enable_eplb:')
sub(adapter, """    if spec.kernel == "deep_gemm":
        return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale)""", """    if spec.kernel in ("deep_gemm", "kz120"):
        return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale)""")
sub(adapter, """        spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)
        if spec.kernel == "deep_gemm":
            _expose_deep_gemm_to_flashinfer()""", """        spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)
        self._kz = None
        if spec.kernel == "kz120":
            from vllm.model_executor.layers.fused_moe.kz_sm120_megamoe import KzSm120MegaMoe

            self._kz = KzSm120MegaMoe(moe, weights.w13, weights.w2, weights.w13_scale, weights.w2_scale)
            self._mega_layer = None
            self._epilogue = FlashInferMoeEpEpilogue()
            return
        if spec.kernel == "deep_gemm":
            _expose_deep_gemm_to_flashinfer()""")
sub(adapter, """        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE-EP layer was destroyed")
        if hidden_states.shape[0] > self._max_num_tokens:""", """        if self._kz is not None:
            return self._kz(hidden_states, topk_ids, topk_weights)
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE-EP layer was destroyed")
        if hidden_states.shape[0] > self._max_num_tokens:""")
sub(adapter, """    def warmup(self) -> None:
        if self._mega_layer is None:
            return""", """    def warmup(self) -> None:
        if self._kz is not None or self._mega_layer is None:
            return""")
sub(adapter, """        reads, so a caller may permute experts in place (EPLB).
        \"\"\"
        if self._mega_layer is None:""", """        reads, so a caller may permute experts in place (EPLB).
        \"\"\"
        if self._kz is not None:
            return self._kz.kernel_weights()
        if self._mega_layer is None:""")
sub(adapter, """    def destroy(self) -> None:
        mega_layer = self._mega_layer""", """    def destroy(self) -> None:
        if getattr(self, "_kz", None) is not None:
            return
        mega_layer = self._mega_layer""")

# fused_moe/oracle/mxfp4.py
sub(oracle, "    FLASHINFER_MOE_EP_DEEP_GEMM,\n    MoEBackend,\n)",
    "    FLASHINFER_MOE_EP_DEEP_GEMM,\n    KZ_SM120_MEGAMOE,\n    MoEBackend,\n)")
sub(oracle, '    FLASHINFER_MOE_EP_DEEP_GEMM = "FLASHINFER_MOE_EP_DEEP_GEMM"\n',
    '    FLASHINFER_MOE_EP_DEEP_GEMM = "FLASHINFER_MOE_EP_DEEP_GEMM"\n    KZ_SM120_MEGAMOE = "KZ_SM120_MEGAMOE"\n')
sub(oracle, "    Mxfp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL,\n    Mxfp4MoeBackend.FLASHINFER_MOE_EP_DEEP_GEMM,\n)",
    "    Mxfp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL,\n    Mxfp4MoeBackend.FLASHINFER_MOE_EP_DEEP_GEMM,\n"
    "    Mxfp4MoeBackend.KZ_SM120_MEGAMOE,\n)")
sub(oracle, """        "flashinfer_moe_ep_mega_deep_gemm": [
            Mxfp4MoeBackend.FLASHINFER_MOE_EP_DEEP_GEMM
        ],
""", """        "flashinfer_moe_ep_mega_deep_gemm": [
            Mxfp4MoeBackend.FLASHINFER_MOE_EP_DEEP_GEMM
        ],
        "kz_sm120_megamoe": [Mxfp4MoeBackend.KZ_SM120_MEGAMOE],
""")
sub(oracle, """            FLASHINFER_MOE_EP_CUTEDSL
            if mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL
            else FLASHINFER_MOE_EP_DEEP_GEMM,""", """            {
                Mxfp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL: FLASHINFER_MOE_EP_CUTEDSL,
                Mxfp4MoeBackend.FLASHINFER_MOE_EP_DEEP_GEMM: FLASHINFER_MOE_EP_DEEP_GEMM,
                Mxfp4MoeBackend.KZ_SM120_MEGAMOE: KZ_SM120_MEGAMOE,
            }[mxfp4_backend],""")
print("patched")

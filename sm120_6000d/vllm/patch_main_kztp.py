# Register --moe-backend kz_sm120_tp (MoE-TP, all-reduce by vLLM) on top of patch_main_kz120.py.
# Usage: python3 patch_main_kztp.py <site-packages copy root>  (after patch_main_kz120.py)
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "vllm"
kernel = root / "config/kernel.py"
oracle = root / "model_executor/layers/fused_moe/oracle/mxfp4.py"


def sub(path: Path, old: str, new: str) -> None:
    s = path.read_text()
    assert s.count(old) == 1, (path.name, old[:70], s.count(old))
    path.write_text(s.replace(old, new))


sub(kernel, '    "kz_sm120_megamoe",\n', '    "kz_sm120_megamoe",\n    "kz_sm120_tp",\n')

sub(oracle, '    KZ_SM120_MEGAMOE = "KZ_SM120_MEGAMOE"\n',
    '    KZ_SM120_MEGAMOE = "KZ_SM120_MEGAMOE"\n    KZ_SM120_TP = "KZ_SM120_TP"\n')
sub(oracle, '        "kz_sm120_megamoe": [Mxfp4MoeBackend.KZ_SM120_MEGAMOE],\n',
    '        "kz_sm120_megamoe": [Mxfp4MoeBackend.KZ_SM120_MEGAMOE],\n'
    '        "kz_sm120_tp": [Mxfp4MoeBackend.KZ_SM120_TP],\n')
sub(oracle, """    elif backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS:
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import (  # noqa: E501
            FlashInferMoeEpExperts,
        )

        return [FlashInferMoeEpExperts]
""", """    elif backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS:
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import (  # noqa: E501
            FlashInferMoeEpExperts,
        )

        return [FlashInferMoeEpExperts]

    elif backend == Mxfp4MoeBackend.KZ_SM120_TP:
        from vllm.model_executor.layers.fused_moe.experts.kz_sm120_tp import KzSm120TpExperts

        return [KzSm120TpExperts]
""")
sub(oracle, """    if backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS:
        # The megakernel takes the model's dimensions and validates them itself.
        return hidden_size, intermediate_size""", """    if backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS or backend == Mxfp4MoeBackend.KZ_SM120_TP:
        # The megakernel takes the model's dimensions and validates them itself.
        return hidden_size, intermediate_size""")
sub(oracle, """    elif mxfp4_backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS:
        from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (""", """    elif mxfp4_backend == Mxfp4MoeBackend.KZ_SM120_TP:
        # raw checkpoint MXFP4; the experts class repacks it
        return (w13_weight, w2_weight, w13_weight_scale, w2_weight_scale, w13_bias, w2_bias)
    elif mxfp4_backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS:
        from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (""")
sub(oracle, """    if mxfp4_backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS:
        # The megakernel quantizes activations itself and owns its scales.""", """    if mxfp4_backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS or mxfp4_backend == Mxfp4MoeBackend.KZ_SM120_TP:
        # The megakernel quantizes activations itself and owns its scales.""")
print("patched")

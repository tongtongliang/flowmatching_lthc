"""Model registry for the public FlowMatching LTHC release."""

from .local_thc import LocalTHC_JiT_models, SharedReadFusedFinal12LocalTHCSharedAdaLNJiT
from .lthc_v2 import LocalTHCV2_JiT_models

MODEL_NAMES = (
    "lthc_b4_velocity",
    "local_thc_jit_shared_read_fused_final12_shared_adaln_b4",
    "local_thc_jit_shared_write_fused_final12_shared_adaln_b4",
    "local_thc_v2_fused_final12_shared_adaln_b4",
    "local_thc_v2_fused_final12_adaln_b4",
    "local_thc_v2_direct_fused_final12_shared_adaln_b4",
    "local_thc_v2_direct_fused_final12_adaln_b4",
)


def build_model(name: str = "lthc_b4_velocity", **kwargs):
    """Build the released LTHC-B/4 shared-read fused-final12 model.

    The long model name is kept for checkpoint compatibility with the research
    runs. ``lthc_b4_velocity`` is the public-facing alias.
    """
    if name in {
        "lthc_b4_velocity",
        "local_thc_jit_shared_read_fused_final12_shared_adaln_b4",
        "local_thc_jit_shared_write_fused_final12_shared_adaln_b4",
    }:
        return LocalTHC_JiT_models["SharedRead-FusedFinal12-LocalTHC-SharedAdaLN-JiT-B/4"](**kwargs)
    if name == "local_thc_v2_fused_final12_shared_adaln_b4":
        return LocalTHCV2_JiT_models["LocalTHCv2-FusedFinal12-SharedAdaLN-JiT-B/4"](**kwargs)
    if name == "local_thc_v2_fused_final12_adaln_b4":
        return LocalTHCV2_JiT_models["LocalTHCv2-FusedFinal12-AdaLN-JiT-B/4"](**kwargs)
    if name == "local_thc_v2_direct_fused_final12_shared_adaln_b4":
        return LocalTHCV2_JiT_models["LocalTHCv2-Direct-FusedFinal12-SharedAdaLN-JiT-B/4"](**kwargs)
    if name == "local_thc_v2_direct_fused_final12_adaln_b4":
        return LocalTHCV2_JiT_models["LocalTHCv2-Direct-FusedFinal12-AdaLN-JiT-B/4"](**kwargs)
    raise ValueError(f"unknown model: {name}; available={MODEL_NAMES}")


__all__ = ["MODEL_NAMES", "build_model", "SharedReadFusedFinal12LocalTHCSharedAdaLNJiT", "LocalTHCV2_JiT_models"]

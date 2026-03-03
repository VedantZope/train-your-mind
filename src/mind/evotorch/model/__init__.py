from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional

from mind.evotorch.model import basic, res_gn_gelu, cbam

ModelConfig = List[Dict[str, Any]]


def build_model(model_name: str, params: Optional[Dict[str, Any]] = None) -> Tuple[object, ModelConfig, str]:
    name = model_name.strip().lower()
    params = params or {}
    if name in {"basic", "two_layer", "baseline"}:
        model, cfgs = basic.build()
        return model, [asdict(c) for c in cfgs], "basic"
    if name in {"res_gn_gelu", "residual_gn_gelu"}:
        model, cfgs = res_gn_gelu.build(
            dilation=int(params.get("dilation", 1)),
            num_blocks=int(params.get("num_blocks", 1)),
        )
        return model, [asdict(c) for c in cfgs], "res_gn_gelu"
    if name in {"cbam"}:
        model, cfgs = cbam.build(dilation=int(params.get("dilation", 1)))
        return model, [asdict(c) for c in cfgs], "cbam"
    raise ValueError(f"Unknown model '{model_name}'. Use 'basic', 'res_gn_gelu', or 'cbam'.")

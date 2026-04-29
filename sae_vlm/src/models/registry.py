"""Backbone registry — load any model by name."""

from pathlib import Path
from typing import Union

import yaml

from .base import BackboneBase

_MODELS_CFG_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"

_FAMILY_TO_CLASS = {
    "clip":    "CLIPBackbone",
    "clipood": "CLIPoodBackbone",
    "dinov2":  "DINOv2Backbone",
    "align":   "ALIGNBackbone",
}


def load_model_cfg(name_or_path: str) -> dict:
    """Load model config YAML by short name (e.g. 'clip_vit_b16') or full path."""
    p = Path(name_or_path)
    if not p.exists():
        p = _MODELS_CFG_DIR / f"{name_or_path}.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def get_backbone(name_or_path: str, device: str = "cuda") -> BackboneBase:
    """Instantiate and load a backbone by config name or YAML path.

    Example:
        backbone = get_backbone("clip_vit_b16", device="cuda").load()
    """
    cfg = load_model_cfg(name_or_path)
    family = cfg.get("family", "").lower()
    cls_name = _FAMILY_TO_CLASS.get(family)
    if cls_name is None:
        raise ValueError(f"Unknown backbone family '{family}'. "
                         f"Supported: {list(_FAMILY_TO_CLASS.keys())}")

    # Lazy import to avoid circular deps
    if cls_name == "CLIPBackbone":
        from .clip_backbone import CLIPBackbone as Cls
    elif cls_name == "CLIPoodBackbone":
        from .clipood_backbone import CLIPoodBackbone as Cls
    elif cls_name == "DINOv2Backbone":
        from .dinov2_backbone import DINOv2Backbone as Cls
    elif cls_name == "ALIGNBackbone":
        from .align_backbone import ALIGNBackbone as Cls

    return Cls(model_cfg=cfg, device=device)

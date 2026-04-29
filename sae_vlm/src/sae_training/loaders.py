"""Checkpoint loaders for SAE and HookedVisionTransformer.

Consolidates load_sae / load_hooked_vit / load_lora_weights that previously
lived in the deleted tasks/utils.py.  All experiment scripts import from here.
"""

from pathlib import Path
from typing import List, Optional

import torch

from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.hooked_vit import HookedVisionTransformer


# ── SAE ───────────────────────────────────────────────────────────────────────

def load_sae(path: str, device: str = "cuda"):
    """Load a SparseAutoencoder from a .pt checkpoint.

    Returns (sae, cfg).
    """
    from types import SimpleNamespace
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("cfg", ckpt.get("config"))  # older checkpoints use 'config'
    if cfg is None:
        raise KeyError(f"SAE checkpoint at {path} has neither 'cfg' nor 'config' key")
    # Older checkpoints store cfg as a plain dict; wrap so attribute access works.
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)
    sae  = SparseAutoencoder(cfg, device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(device)
    return sae, cfg


# ── HookedViT ─────────────────────────────────────────────────────────────────

def load_hooked_vit(
    cfg,
    vit_type:    str,
    backbone:    str,
    device:      str = "cuda",
    model_path:  Optional[str] = None,
    config_path: Optional[str] = None,
    classnames:  Optional[List[str]] = None,
) -> HookedVisionTransformer:
    """Load a HookedVisionTransformer wrapping a CLIP (or MaPLe-CLIP) model.

    Parameters
    ----------
    cfg         : SparseAutoencoder config (used only for logging; may be None)
    vit_type    : "base" or "maple"
    backbone    : HuggingFace model ID, e.g. "openai/clip-vit-base-patch16"
    device      : "cuda" | "cpu"
    model_path  : path to MaPLe checkpoint (.pth.tar)
    config_path : path to MaPLe YAML config
    classnames  : class names required for MaPLe prompt learning
    """
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(backbone)

    if vit_type == "base":
        model = CLIPModel.from_pretrained(backbone)
        return HookedVisionTransformer(model, processor, device=device)

    if vit_type == "maple":
        from src.models.architecture.maple import CustomCLIP
        from src.models.config.maple import get_maple_config

        if config_path is None:
            raise ValueError("config_path is required for vit_type='maple'")
        if classnames is None:
            raise ValueError("classnames is required for vit_type='maple'")

        maple_cfg  = get_maple_config(config_path)
        clip_model = CLIPModel.from_pretrained(backbone)
        model      = CustomCLIP(maple_cfg, classnames, clip_model)

        if model_path:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            print(f"  [loader] Loaded MaPLe weights from {Path(model_path).name}")

        return HookedVisionTransformer(model, processor, device=device)

    raise ValueError(f"Unknown vit_type '{vit_type}'. Use 'base' or 'maple'.")


# Layer indices per position name, mirrored from CLIP_LoRA/loralib/utils.py
_TEXT_POSITIONS: dict = {
    "top1": [11], "top2": [10, 11], "top3": [9, 10, 11],
    "bottom": [0, 1, 2, 3], "mid": [4, 5, 6, 7], "up": [8, 9, 10, 11],
    "half-up": [6, 7, 8, 9, 10, 11], "half-bottom": [0, 1, 2, 3, 4, 5],
    "all": list(range(12)),
}
_VISION_POSITIONS: dict = {
    "ViT-B/16": {
        "top": [11], "top3": [9, 10, 11], "bottom": [0, 1, 2, 3],
        "mid": [4, 5, 6, 7], "up": [8, 9, 10, 11],
        "half-up": [6, 7, 8, 9, 10, 11], "half-bottom": [0, 1, 2, 3, 4, 5],
        "all": list(range(12)),
    },
    "ViT-B/32": {
        "bottom": [0, 1, 2, 3], "mid": [4, 5, 6, 7], "up": [8, 9, 10, 11],
        "half-up": [6, 7, 8, 9, 10, 11], "half-bottom": [0, 1, 2, 3, 4, 5],
        "all": list(range(12)),
    },
    "ViT-L/14": {
        "half-up": list(range(12, 24)), "half-bottom": list(range(12)),
        "all": list(range(24)),
    },
}


def _apply_clip_lora_deltas(
    vit: HookedVisionTransformer,
    ckpt: dict,
    device: str,
    backbone: str = "ViT-B/16",
) -> None:
    """Apply CLIP_LoRA delta weights (B@A) onto a HuggingFace CLIPModel in-place."""
    meta     = ckpt["metadata"]
    weights  = ckpt["weights"]
    scale    = meta["alpha"] / meta["r"]
    encoder  = meta["encoder"]   # "text" | "vision" | "both"
    position = meta["position"]  # e.g. "all", "top3"

    # Build ordered (encoder_type, layer_idx) list matching save_lora iteration order
    layer_map: List[tuple] = []
    if encoder in ("text", "both"):
        for idx in _TEXT_POSITIONS[position]:
            layer_map.append(("text", idx))
    if encoder in ("vision", "both"):
        for idx in _VISION_POSITIONS[backbone][position]:
            layer_map.append(("vision", idx))

    state_dict = {k: v.float() for k, v in vit.model.state_dict().items()}

    for lora_idx, (enc_type, layer_idx) in enumerate(layer_map):
        layer_key = f"layer_{lora_idx}"
        if layer_key not in weights:
            continue
        layer_w = weights[layer_key]

        # HuggingFace CLIPModel parameter prefix
        if enc_type == "text":
            prefix = f"text_model.encoder.layers.{layer_idx}.self_attn"
        else:
            prefix = f"vision_model.encoder.layers.{layer_idx}.self_attn"

        for proj in ("q_proj", "k_proj", "v_proj"):
            if proj not in layer_w:
                continue
            proj_w = layer_w[proj]
            A = proj_w.get("w_lora_A") if isinstance(proj_w, dict) else None
            B = proj_w.get("w_lora_B") if isinstance(proj_w, dict) else None
            if A is None or B is None:
                continue

            param_key = f"{prefix}.{proj}.weight"
            if param_key not in state_dict:
                continue

            delta = scale * (B.float() @ A.float())
            state_dict[param_key] = state_dict[param_key] + delta.to(state_dict[param_key].device)

    vit.model.load_state_dict(state_dict, strict=False)
    vit.model.to(device)


def load_lora_weights(
    vit: HookedVisionTransformer,
    checkpoint_path: str,
    device: str,
    backbone: str = "ViT-B/16",
) -> None:
    """Load LoRA weights into a HookedVisionTransformer.

    Accepts two formats:
    - Standard state dict (pre-merged model weights)
    - CLIP_LoRA format: {'weights': {layer_i: {proj: {w_lora_A, w_lora_B}}}, 'metadata': {...}}
      In this case the LoRA deltas (scale * B @ A) are applied directly to the HuggingFace
      CLIPModel's separate q/k/v projection weights.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "weights" in ckpt and "metadata" in ckpt:
        _apply_clip_lora_deltas(vit, ckpt, device, backbone=backbone)
    else:
        state_dict = ckpt.get("state_dict", ckpt)
        vit.model.load_state_dict(state_dict, strict=False)
        vit.model.to(device)

    print(f"  [loader] Loaded LoRA weights from {Path(checkpoint_path).name}")


# ── Combined helper ───────────────────────────────────────────────────────────

def get_sae_and_vit(
    sae_path:    str,
    vit_type:    str,
    device:      str,
    backbone:    str  = "openai/clip-vit-base-patch16",
    model_path:  Optional[str] = None,
    config_path: Optional[str] = None,
    classnames:  Optional[List[str]] = None,
):
    """Load SAE and HookedVisionTransformer together.

    Returns (sae, vit, cfg).
    """
    sae, cfg = load_sae(sae_path, device)
    vit      = load_hooked_vit(cfg, vit_type, backbone, device,
                               model_path=model_path, config_path=config_path,
                               classnames=classnames)
    return sae, vit, cfg

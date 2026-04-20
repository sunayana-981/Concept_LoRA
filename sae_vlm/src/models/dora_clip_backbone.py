"""DoRA-finetuned CLIP backbone.

Loads a vanilla HuggingFace CLIP model and merges DoRA delta weights
(saved by CLIP_LoRA/dora_finetune.py) directly into the model parameters.
The result behaves identically to CLIPBackbone but with the DoRA weights baked in.
"""

import math
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base import BackboneBase


# Position indices — mirrored from dora_finetune.py
_VISION_POSITIONS = {
    "ViT-B/16": {
        "top": [11], "top1": [11], "top2": [10, 11], "top3": [9, 10, 11],
        "bottom": [0, 1, 2, 3], "mid": [4, 5, 6, 7], "up": [8, 9, 10, 11],
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
_TEXT_POSITIONS = {
    "top1": [11], "top2": [10, 11], "top3": [9, 10, 11],
    "bottom": [0, 1, 2, 3], "mid": [4, 5, 6, 7], "up": [8, 9, 10, 11],
    "half-up": [6, 7, 8, 9, 10, 11], "half-bottom": [0, 1, 2, 3, 4, 5],
    "all": list(range(12)),
}


def _merge_dora_weights(model: CLIPModel, ckpt_path: str, backbone_name: str = "ViT-B/16") -> None:
    """Apply DoRA delta weights into model in-place.

    DoRA weight update:
        W_eff = (m / ‖W₀ + s·B@A‖_col) * (W₀ + s·B@A)

    Since m is initialized to ‖W₀‖_col and B is initialized to zero, at
    checkpoint time the weights are already the trained DoRA values stored as
    lora_A, lora_B, magnitude.  We reconstruct W_eff and write it directly
    into the HuggingFace state dict.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "weights" not in ckpt or "metadata" not in ckpt:
        raise ValueError(f"Unrecognized checkpoint format in {ckpt_path}. "
                         "Expected keys 'weights' and 'metadata'.")

    meta    = ckpt["metadata"]
    weights = ckpt["weights"]
    encoder = meta["encoder"]    # "text" | "vision" | "both"
    position = meta["position"]
    r       = meta["r"]
    alpha   = meta["alpha"]
    scaling = alpha / math.sqrt(r)

    # Build ordered list of (encoder_type, hf_layer_idx) matching save_dora iteration
    layer_map = []
    if encoder in ("text", "both"):
        for idx in _TEXT_POSITIONS[position]:
            layer_map.append(("text", idx))
    if encoder in ("vision", "both"):
        for idx in _VISION_POSITIONS[backbone_name][position]:
            layer_map.append(("vision", idx))

    state_dict = {k: v.float().clone() for k, v in model.state_dict().items()}

    proj_to_hf = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "proj": "out_proj"}

    for lora_idx, (enc_type, layer_idx) in enumerate(layer_map):
        layer_key = f"layer_{lora_idx}"
        if layer_key not in weights:
            continue
        layer_w = weights[layer_key]

        if enc_type == "text":
            prefix = f"text_model.encoder.layers.{layer_idx}.self_attn"
        else:
            prefix = f"vision_model.encoder.layers.{layer_idx}.self_attn"

        for dora_key, hf_key in proj_to_hf.items():
            if dora_key not in layer_w:
                continue
            pw = layer_w[dora_key]
            if not isinstance(pw, dict):
                continue

            lora_A    = pw.get("lora_A")
            lora_B    = pw.get("lora_B")
            magnitude = pw.get("magnitude")
            if lora_A is None or lora_B is None or magnitude is None:
                continue

            lora_A    = lora_A.float()
            lora_B    = lora_B.float()
            magnitude = magnitude.float()

            param_key = f"{prefix}.{hf_key}.weight"
            if param_key not in state_dict:
                continue

            W0 = state_dict[param_key]                     # (d_out, d_in)
            delta = scaling * (lora_B @ lora_A)            # (d_out, d_in)
            adapted = W0 + delta                           # (d_out, d_in)

            col_norms = adapted.norm(dim=1, keepdim=True).clamp(min=1e-8)
            W_eff = (magnitude.unsqueeze(1) / col_norms) * adapted

            state_dict[param_key] = W_eff

    model.load_state_dict(state_dict, strict=False)
    print(f"  [DoRACLIPBackbone] Merged DoRA weights from {Path(ckpt_path).name}")


class DoRACLIPBackbone(BackboneBase):
    """CLIP backbone with DoRA fine-tuning weights merged in.

    Parameters (in model_cfg):
        model_id:        HuggingFace CLIP model ID (e.g. openai/clip-vit-base-patch16)
        dora_ckpt:       path to dora_weights.pt from dora_finetune.py
        backbone_name:   e.g. "ViT-B/16" (used for position index lookup)
        d_model, n_layers, ...  (same as clip_vit_b16.yaml)
    """

    def load(self) -> "DoRACLIPBackbone":
        mid = self.model_cfg["model_id"]
        self.processor = CLIPProcessor.from_pretrained(mid)
        self.model = CLIPModel.from_pretrained(mid)

        dora_ckpt = self.model_cfg.get("dora_ckpt")
        if dora_ckpt:
            backbone_name = self.model_cfg.get("backbone_name", "ViT-B/16")
            _merge_dora_weights(self.model, dora_ckpt, backbone_name)

        self.model.eval().to(self.device)
        return self

    def get_activations(
        self,
        images: List[Image.Image],
        layer: int = -1,
        use_cls: bool = True,
    ) -> torch.Tensor:
        layer_idx = self.resolve_layer(layer)
        pv = self.process_images(images)

        captured = {}
        def _hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["hs"] = hs.detach().float()

        h = self.model.vision_model.encoder.layers[layer_idx].register_forward_hook(_hook)
        with torch.no_grad():
            self.model.vision_model(pixel_values=pv)
        h.remove()

        hs = captured["hs"]  # [B, T, d]
        if use_cls:
            return hs[:, 0, :]
        B, T, d = hs.shape
        return hs[:, 1:, :].reshape(B * (T - 1), d)

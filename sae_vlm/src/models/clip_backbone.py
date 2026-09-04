"""CLIP ViT-B/16 backbone wrapper."""

from typing import List, Optional
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, SiglipModel
try:
    from transformers import Siglip2Model
except ImportError:  # older transformers versions don't have Siglip2Model yet
    Siglip2Model = None

from .base import BackboneBase


class CLIPBackbone(BackboneBase):
    """
    Wraps openai/clip-vit-base-patch16 (or any HF CLIPModel).

    Hook target: model.vision_model.encoder.layers[i]
    CLS token:   sequence index 0
    Patch tokens: indices 1..196
    """

    def load(self) -> "CLIPBackbone":
        mid = self.model_cfg["model_id"]
        loader = self.model_cfg.get("loader", "clip").lower()
        trust_remote_code = self.model_cfg.get("trust_remote_code", False)

        if loader == "siglip2":
            self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=trust_remote_code)
            self.model = Siglip2Model.from_pretrained(mid, trust_remote_code=trust_remote_code).eval().to(self.device)
        elif loader == "siglip":
            self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=trust_remote_code)
            self.model = SiglipModel.from_pretrained(mid, trust_remote_code=trust_remote_code).eval().to(self.device)
        else:
            self.processor = CLIPProcessor.from_pretrained(mid, trust_remote_code=trust_remote_code)
            self.model = CLIPModel.from_pretrained(mid, trust_remote_code=trust_remote_code).eval().to(self.device)
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
            return hs[:, 0, :]                       # [B, d]
        B, T, d = hs.shape
        return hs[:, 1:, :].reshape(B * (T - 1), d)  # [B*(T-1), d]

    def get_all_token_activations(
        self, images: List[Image.Image], layer: int = -1
    ) -> torch.Tensor:
        """Returns full sequence [B, T, d] — useful for patch-level SAE."""
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
        return captured["hs"]  # [B, T, d]

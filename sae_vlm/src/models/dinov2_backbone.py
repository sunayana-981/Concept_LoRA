"""DINOv2 backbone wrapper."""

from typing import List
import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

from .base import BackboneBase


class DINOv2Backbone(BackboneBase):
    """
    Wraps facebook/dinov2-base (or any HF Dinov2Model).

    Hook target: model.encoder.layer[i]
    CLS token:   sequence index 0
    Patch tokens: indices 1..256
    """

    def load(self) -> "DINOv2Backbone":
        mid = self.model_cfg["model_id"]
        self.processor = AutoImageProcessor.from_pretrained(mid)
        self.model = Dinov2Model.from_pretrained(mid).eval().to(self.device)
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

        h = self.model.encoder.layer[layer_idx].register_forward_hook(_hook)
        with torch.no_grad():
            self.model(pixel_values=pv)
        h.remove()

        hs = captured["hs"]  # [B, T, d]
        if use_cls:
            return hs[:, 0, :]
        B, T, d = hs.shape
        return hs[:, 1:, :].reshape(B * (T - 1), d)

    def get_all_token_activations(
        self, images: List[Image.Image], layer: int = -1
    ) -> torch.Tensor:
        layer_idx = self.resolve_layer(layer)
        pv = self.process_images(images)

        captured = {}
        def _hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["hs"] = hs.detach().float()

        h = self.model.encoder.layer[layer_idx].register_forward_hook(_hook)
        with torch.no_grad():
            self.model(pixel_values=pv)
        h.remove()
        return captured["hs"]

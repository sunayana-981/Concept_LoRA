"""ALIGN (kakaobrain/align-base) backbone wrapper.

ALIGN uses an EfficientNet-B7-like vision encoder (not a ViT).
55 AlignVisionBlocks; last block [54] output is [B, 640, 10, 10].
GAP over spatial dims → [B, 640] serves as the 'CLS' equivalent.
"""

from typing import List
import torch
from PIL import Image
from transformers import AlignModel, AlignProcessor

from .base import BackboneBase


class ALIGNBackbone(BackboneBase):
    """
    Wraps kakaobrain/align-base (or any HF AlignModel).

    Hook target: model.vision_model.encoder.blocks[i]
    Block output: [B, C, H, W]  — EfficientNet spatial feature map
    GAP (use_cls=True):   [B, C]          — analogous to CLS token
    Spatial (use_cls=False): [B*H*W, C]   — all spatial positions
    """

    def load(self) -> "ALIGNBackbone":
        mid = self.model_cfg["model_id"]
        self.processor = AlignProcessor.from_pretrained(mid)
        self.model = AlignModel.from_pretrained(mid).eval().to(self.device)
        return self

    def process_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, text="", return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

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
            feat = out.detach().float()
            captured["feat"] = feat  # [B, C, H, W]

        h = self.model.vision_model.encoder.blocks[layer_idx].register_forward_hook(_hook)
        with torch.no_grad():
            self.model.vision_model(pixel_values=pv)
        h.remove()

        feat = captured["feat"]  # [B, C, H, W]
        if use_cls:
            return feat.mean(dim=[-2, -1])  # GAP → [B, C]
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B * H * W, C)

    def get_all_token_activations(
        self, images: List[Image.Image], layer: int = -1
    ) -> torch.Tensor:
        """Returns spatial tokens as [B, H*W, C] — analogous to patch tokens."""
        layer_idx = self.resolve_layer(layer)
        pv = self.process_images(images)

        captured = {}
        def _hook(_m, _i, out):
            captured["feat"] = out.detach().float()

        h = self.model.vision_model.encoder.blocks[layer_idx].register_forward_hook(_hook)
        with torch.no_grad():
            self.model.vision_model(pixel_values=pv)
        h.remove()

        feat = captured["feat"]  # [B, C, H, W]
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]

"""Base class for all VLM backbone wrappers."""

from abc import ABC, abstractmethod
from typing import List
import torch
from PIL import Image


class BackboneBase(ABC):
    """
    Unified interface for VLM vision encoders.

    All backbone wrappers expose:
      get_activations(images, layer)  →  Tensor[B, d_model]
      get_all_token_activations(...)  →  Tensor[B, T, d_model]   (ViT only)

    This lets train_sae.py and eval_sae.py be model-agnostic.
    """

    def __init__(self, model_cfg: dict, device: str = "cuda"):
        self.model_cfg = model_cfg
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> "BackboneBase":
        """Load model and processor onto self.device. Returns self."""

    @abstractmethod
    def get_activations(
        self,
        images: List[Image.Image],
        layer: int,
        use_cls: bool = True,
    ) -> torch.Tensor:
        """
        Run the vision encoder and return activations.

        Args:
            images:  list of PIL images
            layer:   which layer/block to hook (negative ok, e.g. -1 = last)
            use_cls: if True return CLS/GAP token [B, d_model];
                     if False return all spatial/patch tokens [B*T, d_model]

        Returns:
            Tensor on self.device, float32
        """

    def process_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Preprocess a list of PIL images to pixel_values tensor."""
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    @property
    def d_model(self) -> int:
        return self.model_cfg["d_model"]

    @property
    def n_layers(self) -> int:
        return self.model_cfg["n_layers"]

    def resolve_layer(self, layer: int) -> int:
        """Convert negative layer index to absolute (0-based)."""
        if layer >= 0:
            return layer
        return self.n_layers + layer

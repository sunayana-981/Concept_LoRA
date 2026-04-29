"""CLIPood backbone — base CLIP with an optional CLIPood fine-tuned state dict applied."""

from pathlib import Path
from typing import List

import torch
from PIL import Image

from .clip_backbone import CLIPBackbone


class CLIPoodBackbone(CLIPBackbone):
    """
    Identical to CLIPBackbone but loads a CLIPood fine-tuned CLIP state dict on top.

    The checkpoint (a full HF CLIPModel state_dict) is specified via
    model_cfg["clipood_checkpoint_path"], which train_sae.py injects from
    --clipood_checkpoint.  If the key is absent the backbone behaves like base CLIP.
    """

    def load(self) -> "CLIPoodBackbone":
        super().load()   # loads openai/clip-vit-base-patch16, processor, moves to device

        ckpt_path = self.model_cfg.get("clipood_checkpoint_path")
        if ckpt_path:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"CLIPood checkpoint not found: {ckpt_path}")
            state = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            state_dict = state.get("state_dict", state)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print(f"  [CLIPoodBackbone] Loaded weights from {ckpt_path.name}")
        else:
            print("  [CLIPoodBackbone] No checkpoint specified — using base CLIP weights")

        return self

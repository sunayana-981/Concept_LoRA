"""
# Portions of this file are based on code from the "HugoFry/mats_sae_training_for_ViTs" repository (MIT-licensed):
    https://github.com/HugoFry/mats_sae_training_for_ViTs/blob/main/sae_training/hooked_vit.py
"""

from contextlib import contextmanager
from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F

from src.sae_training.backbone_registry import get_backbone_spec, infer_backbone_name


# The Hook class does not currently only supports hooking on the following locations:
# 1 - residual stream post transformer block.
# 2 - mlp activations.
# More hooks can be added at a later date, but only post-module.
class Hook:
    def __init__(
        self,
        block_layer: int,
        module_name: str,
        hook_fn: Callable,
        is_custom: bool = None,
        backbone: Optional[str] = None,
        class_token: Optional[bool] = None,
        return_module_output=True,
    ):
        self.path_dict = {
            "resid": "",
        }
        assert module_name in self.path_dict.keys(), (
            f"Module name '{module_name}' not recognised."
        )
        # `is_custom` is the old CLIP-vs-MaPLe-only two-way switch, kept for
        # call sites that haven't been updated to pass `backbone=` directly.
        if backbone is None:
            backbone = "maple" if is_custom else "clip"
        self.backbone = backbone
        self.backbone_spec = get_backbone_spec(backbone)
        self.class_token = class_token
        self.return_module_output = return_module_output
        self.function = self.get_full_hook_fn(hook_fn)
        self.attr_path = self.get_attr_path(block_layer, module_name)

    def get_full_hook_fn(self, hook_fn: Callable):
        def full_hook_fn(module, module_input, module_output):
            # MaPLe ResidualAttentionBlock returns a list [x, prompts, counter];
            # standard CLIP/DINOv2/SigLIP2 layers return a tuple (tensor,);
            # ALIGN's AlignVisionBlock (EfficientNet, not a transformer) returns
            # a raw [B, C, H, W] feature map with no token-sequence structure.
            if isinstance(module_output, list):
                hook_fn_output = hook_fn(module_output[0])
                if self.return_module_output:
                    return module_output
                # Preserve the list structure so downstream layers
                # still receive [x, compound_prompts_deeper, counter]
                modified = list(module_output)
                if isinstance(hook_fn_output, tuple):
                    modified[0] = hook_fn_output[0]
                else:
                    modified[0] = hook_fn_output
                return modified
            elif not self.backbone_spec.is_transformer_block:
                # ALIGN conv block: [B, C, H, W] -> a pseudo token sequence so
                # downstream code (get_model_activations) can treat it exactly
                # like a transformer block's [B, seq, D] output. class_token
                # (GAP -> a single "CLS-like" pseudo-token) matches
                # ALIGNHook's calibrated default in
                # tasks/train_sae_align_imagenet.py; otherwise all spatial
                # positions become individual tokens.
                feat = module_output
                if self.class_token:
                    pseudo_seq = feat.mean(dim=[-2, -1]).unsqueeze(1)  # [B, 1, C]
                else:
                    b, c, h, w = feat.shape
                    pseudo_seq = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, H*W, C]
                hook_fn_output = hook_fn(pseudo_seq)
                if self.return_module_output:
                    return module_output
                if not self.return_module_output:
                    raise NotImplementedError(
                        "Modifying ALIGN conv-block output in place (e.g. SAE "
                        "reconstruction/ablation eval hooks) is not supported: "
                        "there is no lossless inverse from the pooled/flattened "
                        "pseudo-sequence back to [B, C, H, W]."
                    )
                return hook_fn_output
            else:
                hook_fn_output = hook_fn(module_output[0])
                if self.return_module_output:
                    return module_output
                else:
                    return hook_fn_output

        return full_hook_fn

    def get_attr_path(self, block_layer: int, module_name: str) -> str:
        attr_path = self.backbone_spec.block_attr_template.format(i=block_layer)
        attr_path += self.path_dict[module_name]
        return attr_path

    def get_module(self, model):
        return self.get_nested_attr(model, self.attr_path)

    def get_nested_attr(self, model, attr_path):
        """
        Gets a nested attribute from an object using a dot-separated path.
        """
        module = model
        attributes = attr_path.split(".")
        for attr in attributes:
            if "[" in attr:
                # Split at '[' and remove the trailing ']' from the index
                attr_name, index = attr[:-1].split("[")
                module = getattr(module, attr_name)[int(index)]
            else:
                module = getattr(module, attr)
        return module


class HookedVisionTransformer:
    def __init__(self, model, processor, device="cuda", backbone: Optional[str] = None, class_token: Optional[bool] = None):
        self.model = model.to(device)
        self.processor = processor
        # Back-compat: old call sites don't pass backbone= explicitly, so infer
        # it from the model class (works for CLIP/DINOv2/SigLIP2/ALIGN/MaPLe).
        self.backbone = backbone if backbone is not None else infer_backbone_name(model)
        self.backbone_spec = get_backbone_spec(self.backbone)
        self.class_token = class_token

    def _loss_or_zero(self, output):
        """Contrastive loss for VL backbones; a dummy zero loss for anything
        without a paired text tower (DINOv2) or without logits_per_image/text
        in its output (MaPLe's CustomCLIP returns a raw image_features tensor)."""
        if not self.backbone_spec.has_text_tower or isinstance(output, torch.Tensor):
            device = output.device if isinstance(output, torch.Tensor) else next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)
        return self.contrastive_loss(output.logits_per_image, output.logits_per_text)

    def run_with_cache(
        self,
        list_of_hook_locations: List[Tuple[int, str]],
        *args,
        return_type="output",
        **kwargs,
    ):
        cache_dict, list_of_hooks = self.get_caching_hooks(list_of_hook_locations)
        with self.hooks(list_of_hooks) as hooked_model:
            with torch.no_grad():
                output = hooked_model(*args, **kwargs)

        if return_type == "output":
            return output, cache_dict
        if return_type == "loss":
            return self._loss_or_zero(output), cache_dict
        else:
            raise Exception(
                f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'."
            )

    def get_caching_hooks(self, list_of_hook_locations: List[Tuple[int, str]]):
        """
        Note that the cache dictionary is index by the tuple (block_layer, module_name).
        """
        cache_dict = {}
        list_of_hooks = []

        def save_activations(name, activations):
            # .contiguous() guards against backbones whose block output isn't
            # contiguous (observed with SigLIP2's SiglipEncoderLayer) -- a
            # non-contiguous cached tensor later crashes geom_median's
            # .view(-1) call in initialize_b_dec_with_geometric_median. A
            # no-op for already-contiguous tensors (CLIP/DINOv2/ALIGN).
            cache_dict[name] = activations.detach().contiguous()

        for block_layer, module_name in list_of_hook_locations:
            hook_fn = partial(save_activations, (block_layer, module_name))
            hook = Hook(
                block_layer, module_name, hook_fn,
                backbone=self.backbone, class_token=self.class_token,
            )
            list_of_hooks.append(hook)
        return cache_dict, list_of_hooks

    @torch.no_grad()
    def run_with_hooks(
        self, list_of_hooks: List[Hook], *args, return_type="output", **kwargs
    ):
        with self.hooks(list_of_hooks) as hooked_model:
            with torch.no_grad():
                output = hooked_model(*args, **kwargs)
        if return_type == "output":
            return output
        if return_type == "loss":
            return self._loss_or_zero(output)
        else:
            raise Exception(
                f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'."
            )

    def train_with_hooks(
        self, list_of_hooks: List[Hook], *args, return_type="output", **kwargs
    ):
        with self.hooks(list_of_hooks) as hooked_model:
            output = hooked_model(*args, **kwargs)
        if return_type == "output":
            return output
        if return_type == "loss":
            return self._loss_or_zero(output)
        else:
            raise Exception(
                f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'."
            )

    def contrastive_loss(
        self,
        logits_per_image: Float[Tensor, "n_images n_prompts"],  # noqa: F722
        logits_per_text: Float[Tensor, "n_prompts n_images"],  # noqa: F722
    ):  # Assumes square matrices
        assert logits_per_image.size()[0] == logits_per_image.size()[1], (
            "The number of prompts does not match the number of images."
        )
        batch_size = logits_per_image.size()[0]
        labels = torch.arange(batch_size).long().to(logits_per_image.device)
        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        total_loss = (image_loss + text_loss) / 2
        return total_loss

    @contextmanager
    def hooks(self, hooks: List[Hook]):
        """

        This is a context manager for running a model with hooks. The funciton adds
        forward hooks to the model, and then returns the hooked model to be run with
        a foward pass. The funciton then cleans up by removing any hooks.

        Args:

          model VisionTransformer: The ViT that you want to run with the forward hook

          hooks List[Tuple[str, Callable]]: A list of forward hooks to add to the model.
            Each hook is a tuple of the module name, and the hook funciton.

        """
        hook_handles = []
        try:
            for hook in hooks:
                # Create a full hook funciton, with all the argumnets needed to run nn.module.register_forward_hook().
                # The hook functions are added to the output of the module.
                module = hook.get_module(self.model)
                handle = module.register_forward_hook(hook.function)
                hook_handles.append(handle)
            yield self.model
        finally:
            for handle in hook_handles:
                handle.remove()

    def to(self, device):
        self.model = self.model.to(device)

    def __call__(self, *args, return_type="output", **kwargs):
        return self.forward(*args, return_type=return_type, **kwargs)

    def forward(self, *args, return_type="output", **kwargs):
        if return_type == "output":
            return self.model(*args, **kwargs)
        elif return_type == "loss":
            output = self.model(*args, **kwargs)
            return self._loss_or_zero(output)
        else:
            raise Exception(
                f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'."
            )

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

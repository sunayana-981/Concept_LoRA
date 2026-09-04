"""
Visual Prompt Tuning (VPT-Deep, Jia et al. 2022, "Visual Prompt Tuning").

No prompt-tuning-on-images implementation existed anywhere in this repo
before this (MaPLe's prompts are a vision<->text *coupling* mechanism, not
plain VPT). This wraps any registered transformer-block vision backbone
(CLIP, DINOv2, SigLIP2 -- not ALIGN, whose EfficientNet vision tower has no
patch-token sequence to inject prompts into) and injects a fresh set of
learnable prompt tokens at every transformer block, following the paper's
"Deep" variant: prompts are re-injected (not carried through) at each layer,
so each layer's prompt tokens are dropped and replaced by the next layer's
own learned set rather than accumulating transformed versions of earlier
layers' prompts.

Mechanically this is done with forward pre/post hooks rather than modifying
each block's forward() directly, mirroring the non-invasive hooking style
already used by hooked_vit.py's Hook class for activation caching:
  - A forward-pre-hook on every block splices `[CLS, prompt_i, patches]`
    together right before that block runs (stripping whatever prompt slice
    the previous block's hook inserted, if any).
  - A forward hook on the LAST block strips the prompt slice back out of the
    final output, so the model's pooler / CLS readout is unaffected -- from
    the outside, a VPTVisionWrapper looks exactly like the wrapped backbone.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def _resolve_nested_attr(obj, path: str):
    for part in path.split("."):
        if "[" in part:
            name, idx = part[:-1].split("[")
            obj = getattr(obj, name)[int(idx)]
        else:
            obj = getattr(obj, part)
    return obj


class VPTVisionWrapper(nn.Module):
    """Wraps a HF ViT-style vision encoder (CLIPVisionModel/CLIPModel's
    vision_model, Dinov2Model, SiglipModel's vision_model) and injects
    learnable prompt tokens at every transformer block.

    `block_attr_template` and `num_layers` come from backbone_registry.py's
    BackboneSpec (block_attr_template, and the model config's
    num_hidden_layers) so the same wrapper works across backbones without
    per-architecture branching.
    """

    def __init__(
        self,
        vision_model: nn.Module,
        block_attr_template: str,
        num_layers: int,
        hidden_dim: int,
        n_prompt_tokens: int = 10,
        dropout: float = 0.0,
        patch_size: int = 16,
        num_channels: int = 3,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.block_attr_template = block_attr_template
        self.num_layers = num_layers
        self.n_prompt_tokens = n_prompt_tokens

        # Xavier-uniform bound from the official VPT implementation
        # (KMnP/vpt, src/models/vit_prompt/vit.py): fan_in is the patch
        # embedding's input dimensionality (num_channels * patch_size**2),
        # not the number of prompt tokens -- the token count has no bearing
        # on the natural activation scale prompts should start at, only the
        # embedding geometry does. (Using hidden_dim in place of fan_in
        # happens to be exact for any patch-16 ViT, since
        # 3*16*16 == 768 == hidden_size, but that's a coincidence specific
        # to patch-16 backbones -- it would be wrong for e.g. DINOv2's
        # patch_size=14, hence taking patch_size as an explicit argument.)
        fan_in = num_channels * patch_size * patch_size
        init_val = math.sqrt(6.0 / (fan_in + hidden_dim))
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.empty(1, n_prompt_tokens, hidden_dim).uniform_(-init_val, init_val))
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        self._hook_handles = []
        self._register_hooks()

    def _get_block(self, i: int) -> nn.Module:
        return _resolve_nested_attr(self.vision_model, self.block_attr_template.format(i=i))

    def _register_hooks(self) -> None:
        for i in range(self.num_layers):
            block = self._get_block(i)
            handle = block.register_forward_pre_hook(self._make_pre_hook(i), with_kwargs=True)
            self._hook_handles.append(handle)
        last_block = self._get_block(self.num_layers - 1)
        self._hook_handles.append(last_block.register_forward_hook(self._strip_prompt_hook))

    def _make_pre_hook(self, layer_idx: int):
        p = self.n_prompt_tokens

        def pre_hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            batch_size = hidden_states.shape[0]
            prompt = self.dropout(self.prompts[layer_idx].expand(batch_size, -1, -1).to(hidden_states.dtype))

            if layer_idx == 0:
                # Incoming sequence is [CLS, patches] -- no prompt slice yet.
                cls_tok, rest = hidden_states[:, :1], hidden_states[:, 1:]
            else:
                # Incoming sequence is [CLS, prompt_{i-1}, patches] (the
                # previous layer's own hook inserted p tokens right after
                # CLS, and the transformer block preserves sequence length,
                # so this slicing is exact). Drop them -- VPT-Deep learns a
                # fresh prompt per layer rather than propagating it.
                cls_tok, rest = hidden_states[:, :1], hidden_states[:, 1 + p:]

            new_hidden_states = torch.cat([cls_tok, prompt, rest], dim=1)

            if args:
                new_args = (new_hidden_states,) + args[1:]
                return new_args, kwargs
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = new_hidden_states
            return args, new_kwargs

        return pre_hook

    def _strip_prompt_hook(self, module, module_input, module_output):
        p = self.n_prompt_tokens
        if isinstance(module_output, tuple):
            hidden_states = module_output[0]
            stripped = torch.cat([hidden_states[:, :1], hidden_states[:, 1 + p:]], dim=1)
            return (stripped,) + module_output[1:]
        stripped = torch.cat([module_output[:, :1], module_output[:, 1 + p:]], dim=1)
        return stripped

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)

"""Registry describing how to load and hook each supported VLM backbone.

Originally this pipeline only ever ran on CLIP (and the MaPLe-adapted variant
of CLIP). This module centralizes the per-architecture differences needed to
generalize `hooked_vit.py` / `src/models/utils.py` to DINOv2, ALIGN, and
SigLIP2 as well, instead of hardcoding `isinstance(model, CLIPModel)` checks
or writing a fourth parallel hook implementation (DINOv2/ALIGN already had
one-off, non-shared hook classes in tasks/train_sae_dino_imagenet.py and
tasks/train_sae_align_imagenet.py -- the block-attribute paths here match
those exactly).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Type


class _CombinedProcessor:
    """Composes a separately-loaded image processor + tokenizer into a single
    Processor-like callable.

    Exists because google/siglip2-base-patch16-224's published config makes
    `AutoProcessor`/`SiglipProcessor.from_pretrained` crash: it hardcodes
    `SiglipTokenizer`, but the checkpoint's tokenizer files are actually for a
    `GemmaTokenizer`. `AutoImageProcessor` and `AutoTokenizer` loaded
    independently both resolve correctly (verified), so this composes them
    into the `images=, text=, return_tensors=, padding=` interface the rest
    of this codebase's processor call sites expect.
    """

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, return_tensors="pt", padding=True, **kwargs):
        from transformers.feature_extraction_utils import BatchFeature

        out = {}
        if images is not None:
            out.update(self.image_processor(images=images, return_tensors=return_tensors))
        if text is not None:
            # Callers (e.g. process_model_inputs's non-labeled path) pass
            # text="" deliberately, as a dummy placeholder, whenever they
            # only care about vision activations -- the model's forward still
            # requires input_ids to be present, so an empty string must still
            # be tokenized, not skipped (matches CLIPProcessor's behavior).
            out.update(self.tokenizer(text, return_tensors=return_tensors, padding=padding, truncation=True))
        return BatchFeature(out)

    @classmethod
    def from_pretrained(cls, model_id: str):
        from transformers import AutoImageProcessor, AutoTokenizer

        return cls(
            AutoImageProcessor.from_pretrained(model_id),
            AutoTokenizer.from_pretrained(model_id),
        )


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    model_cls: Optional[Type]
    # Anything exposing a `.from_pretrained(model_id)` classmethod that
    # returns a callable processor -- usually an HF *Processor class, but see
    # _CombinedProcessor above for siglip2's workaround.
    processor_cls: Optional[Type]
    default_model_id: str
    # False only for ALIGN: its vision tower is an EfficientNet (conv blocks
    # emitting [B, C, H, W] feature maps), not a transformer emitting a
    # [B, seq, D] token sequence.
    is_transformer_block: bool
    # `{i}` is replaced with the (possibly negative) block index.
    block_attr_template: str
    # False only for DINOv2: it's a vision-only self-supervised backbone with
    # no paired text encoder, so contrastive-loss evals are meaningless for it.
    has_text_tower: bool
    hidden_dim: int


BACKBONES: dict[str, BackboneSpec] = {}


def _register(spec: BackboneSpec) -> None:
    BACKBONES[spec.name] = spec


def _build_registry() -> None:
    from transformers import (
        AlignModel,
        AlignProcessor,
        AutoImageProcessor,
        CLIPModel,
        CLIPProcessor,
        Dinov2Model,
        SiglipModel,
    )

    _register(BackboneSpec(
        name="clip",
        model_cls=CLIPModel,
        processor_cls=CLIPProcessor,
        default_model_id="openai/clip-vit-base-patch16",
        is_transformer_block=True,
        block_attr_template="vision_model.encoder.layers[{i}]",
        has_text_tower=True,
        hidden_dim=768,
    ))
    _register(BackboneSpec(
        name="dino",
        model_cls=Dinov2Model,
        processor_cls=AutoImageProcessor,
        default_model_id="facebook/dinov2-base",
        is_transformer_block=True,
        # Dinov2Model itself IS the vision model (no `.vision_model` wrapper),
        # matching tasks/train_sae_dino_imagenet.py's DINOHook.
        block_attr_template="encoder.layer[{i}]",
        has_text_tower=False,
        hidden_dim=768,
    ))
    _register(BackboneSpec(
        name="siglip2",
        model_cls=SiglipModel,
        processor_cls=_CombinedProcessor,  # see class docstring: AutoProcessor is broken for this checkpoint
        default_model_id="google/siglip2-base-patch16-224",
        is_transformer_block=True,
        block_attr_template="vision_model.encoder.layers[{i}]",
        has_text_tower=True,
        hidden_dim=768,
    ))
    _register(BackboneSpec(
        name="align",
        model_cls=AlignModel,
        processor_cls=AlignProcessor,
        default_model_id="kakaobrain/align-base",
        is_transformer_block=False,
        # matches tasks/train_sae_align_imagenet.py's ALIGNHook.get_module
        block_attr_template="vision_model.encoder.blocks[{i}]",
        has_text_tower=True,
        hidden_dim=640,
    ))
    _register(BackboneSpec(
        name="maple",
        model_cls=None,  # loaded via get_adapted_clip, not a plain from_pretrained
        processor_cls=CLIPProcessor,
        default_model_id="openai/clip-vit-base-patch16",
        is_transformer_block=True,
        block_attr_template="image_encoder.transformer.resblocks[{i}]",
        has_text_tower=True,
        hidden_dim=768,
    ))


_build_registry()


def get_backbone_spec(name: str) -> BackboneSpec:
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Known backbones: {sorted(BACKBONES)}")
    return BACKBONES[name]


def infer_backbone_name(model) -> str:
    """Best-effort reverse lookup from a model instance to its registry name,
    so existing call sites that don't pass `backbone=` explicitly keep working."""
    for name, spec in BACKBONES.items():
        if spec.model_cls is not None and isinstance(model, spec.model_cls):
            return name
    # MaPLe's CustomCLIP has `image_encoder` but no `vision_model` -- matches
    # the isinstance/hasattr heuristic previously inlined in
    # masked_sae_trainer.py and hooked_vit.py's get_caching_hooks.
    if hasattr(model, "image_encoder") and not hasattr(model, "vision_model"):
        return "maple"
    raise ValueError(
        f"Cannot infer backbone for model of type {type(model)}; pass backbone= explicitly."
    )

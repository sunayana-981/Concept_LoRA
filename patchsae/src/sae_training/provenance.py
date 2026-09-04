"""Provenance helpers for controlled SAE initialization experiments.

The original trainers call their stopping counter ``total_training_tokens``,
but increment it by ``sae_acts.size(0)``.  For the default PatchSAE activation
shape that is the image batch dimension, not the number of patch activations.
New manifests therefore use ``training_examples`` for that counter and report
the derived activation-vector exposure separately.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional


PROVENANCE_SCHEMA_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for ``path`` without loading it all in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def infer_activation_vectors_per_example(
    *,
    image_width: int,
    image_height: int,
    patch_size: int,
    class_token_only: bool,
) -> int:
    """Infer how many ViT activation vectors one image contributes."""
    if class_token_only:
        return 1
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if image_width % patch_size or image_height % patch_size:
        raise ValueError(
            "image dimensions must be divisible by patch_size: "
            f"{image_width}x{image_height}, patch_size={patch_size}"
        )
    return (image_width // patch_size) * (image_height // patch_size) + 1


def build_experiment_metadata(
    *,
    condition: str,
    initialization: str,
    initialization_checkpoint: Optional[str],
    initialization_checkpoint_sha256: Optional[str],
    activation_dataset: str,
    target_dataset: str,
    activation_data_role: str,
    adapted_model_checkpoint: str,
    seed: int,
    block_layer: int,
    module_name: str,
    d_in: int,
    expansion_factor: int,
    gated_sae: bool,
    training_examples_requested: int,
    activation_vectors_per_example: int,
    protect_frac: float,
    adapted_model_checkpoint_sha256: Optional[str] = None,
    target_data_recipe: Optional[str] = None,
    target_data_inventory_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Build the JSON/pickle-safe factor record stored with every new arm."""
    if initialization not in {"scratch_random", "checkpoint"}:
        raise ValueError(f"unsupported SAE initialization: {initialization}")
    if initialization == "checkpoint" and not initialization_checkpoint:
        raise ValueError("checkpoint initialization requires a checkpoint path")
    if initialization == "scratch_random" and initialization_checkpoint:
        raise ValueError("scratch initialization cannot have an init checkpoint")
    if training_examples_requested <= 0:
        raise ValueError("training_examples_requested must be positive")
    if activation_vectors_per_example <= 0:
        raise ValueError("activation_vectors_per_example must be positive")

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "condition": condition,
        "sae_initialization": initialization,
        "initialization_checkpoint": initialization_checkpoint,
        "initialization_checkpoint_sha256": initialization_checkpoint_sha256,
        "activation_dataset": activation_dataset,
        "target_dataset": target_dataset,
        "activation_data_role": activation_data_role,
        "activation_model": "lora_adapted_clip",
        "adapted_model_checkpoint": adapted_model_checkpoint,
        "adapted_model_checkpoint_sha256": adapted_model_checkpoint_sha256,
        "target_data_recipe": target_data_recipe,
        "target_data_inventory_sha256": target_data_inventory_sha256,
        "data_order_seed": int(seed),
        "architecture": {
            "type": "gated" if gated_sae else "standard",
            "block_layer": int(block_layer),
            "module_name": module_name,
            "d_in": int(d_in),
            "expansion_factor": int(expansion_factor),
            "d_sae": int(d_in * expansion_factor),
        },
        "protect_frac": float(protect_frac),
        # This is deliberately called examples, not tokens: the trainer counter
        # advances by the leading image-batch dimension.
        "training_examples_requested": int(training_examples_requested),
        "activation_vectors_per_example": int(activation_vectors_per_example),
        "derived_activation_vector_exposure_requested": int(
            training_examples_requested * activation_vectors_per_example
        ),
        "legacy_trainer_counter_name": "total_training_tokens",
        "legacy_trainer_counter_unit": "images/examples",
    }


def build_training_metadata(
    *,
    examples_seen: int,
    training_steps: int,
    requested_examples: int,
    activation_vectors_per_example: int,
) -> dict[str, Any]:
    """Build actual exposure metadata for a saved checkpoint."""
    return {
        # Keep this legacy key so older analysis scripts remain compatible.
        "n_training_tokens": int(examples_seen),
        "n_training_examples": int(examples_seen),
        "training_examples_requested": int(requested_examples),
        "n_training_steps": int(training_steps),
        "counter_unit": "images/examples",
        "activation_vectors_per_example": int(activation_vectors_per_example),
        "derived_activation_vector_exposure": int(
            examples_seen * activation_vectors_per_example
        ),
    }

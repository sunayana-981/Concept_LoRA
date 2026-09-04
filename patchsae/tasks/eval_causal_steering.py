#!/usr/bin/env python3
"""Held-out, error-preserving causal interventions on CLIP SAE latents.

This is the rebuttal steering experiment for comparing a generic SAE (G-SAE)
with a target-domain fine-tuned SAE (FT-SAE) on the *same* LoRA-adapted CLIP.
The default scope is deliberately bounded: EuroSAT and PathMNIST, 20 selected
latents per SAE, and one activation-matched random control per selected latent.

Protocol
--------
1. Use only the training/selection split to score class-selective CLS latents.
2. Select ``--num_latents`` with deterministic class-balanced coverage.
3. Match each selected latent to a random non-selected latent using training
   activation prevalence/magnitude and decoder norm.  Evaluation labels and
   predictions are never used for selection or matching.
4. On a disjoint evaluation split, edit one latent at a time:

       x_recon = D(z)
       error   = x - x_recon
       x'      = D(z') + error = x + D(z' - z)

   Thus an unchanged latent is an exact no-op and SAE reconstruction error is
   preserved.  Ablation sets z_j'=0.  Amplification defaults to setting z_j to
   at least a target-class positive-activation quantile estimated on the
   selection split.  Only the CLS token is edited.
5. Separately evaluate full natural SAE reconstruction (D(z), without the
   error term) as a reconstruction-confound control.

No ``forward_clamp`` call or hard-coded SAE bias is used.

Run from ``patchsae/``:

    python tasks/eval_causal_steering.py --dry_run
    python tasks/eval_causal_steering.py \
        --sae_paths configs/rebuttal_sae_paths.json

See ``tasks/CAUSAL_STEERING.md`` for the preregistered analysis and schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from eval_medmnist_sae import _hf_collate_fn
from mdmnist_sae_eval import calculate_text_features
from src.sae_training.hooked_vit import Hook
from tasks.rebuttal_common import (
    add_common_args,
    build_dataset_splits,
    clear_model_caches,
    flush,
    get_sae,
    get_vit,
    load_common_registry_args,
    resolve_sae_path,
    validate_sae_condition_provenance,
)


DEFAULT_DATASETS = ["eurosat", "pathmnist"]
DEFAULT_CONDITIONS = ["gsae", "ftsae"]
PRIMARY_GROUP = {"ablate": "target", "amplify": "non_target"}

MANIFEST_COLUMNS = [
    "dataset",
    "sae_condition",
    "pair_id",
    "feature_role",
    "feature_id",
    "selection_rank",
    "target_class_index",
    "target_class",
    "selectivity_score",
    "target_mean_activation",
    "rest_mean_activation",
    "target_active_fraction",
    "global_mean_activation",
    "global_active_fraction",
    "decoder_norm",
    "matched_feature_id",
    "control_match_distance",
    "control_pool_rank",
    "amplify_mode",
    "amplify_value",
    "amplify_quantile",
    "n_positive_for_amplify",
    "sae_path",
    "block_layer",
    "module_name",
    "d_sae",
    "n_selection_images",
]

EFFECT_COLUMNS = [
    "dataset",
    "sae_condition",
    "pair_id",
    "feature_role",
    "feature_id",
    "selection_rank",
    "target_class_index",
    "target_class",
    "intervention",
    "eval_group",
    "causal_reference",
    "n_images",
    "baseline_accuracy",
    "intervened_accuracy",
    "delta_accuracy_pp",
    "baseline_target_rate",
    "intervened_target_rate",
    "delta_target_rate_pp",
    "mean_delta_target_logit",
    "mean_delta_target_probability",
    "mean_delta_target_margin",
    "se_delta_target_margin",
    "flip_to_target_rate",
    "flip_from_target_rate",
    "directional_margin_effect",
]

EXAMPLE_COLUMNS = [
    "dataset",
    "sae_condition",
    "pair_id",
    "feature_role",
    "feature_id",
    "selection_rank",
    "target_class_index",
    "target_class",
    "intervention",
    "causal_reference",
    "sample_order",
    "eval_source_index",
    "label_index",
    "label",
    "is_target_class",
    "baseline_prediction",
    "intervened_prediction",
    "baseline_correct",
    "intervened_correct",
    "baseline_target_logit",
    "intervened_target_logit",
    "delta_target_logit",
    "baseline_target_probability",
    "intervened_target_probability",
    "delta_target_probability",
    "baseline_target_margin",
    "intervened_target_margin",
    "delta_target_margin",
]


@dataclass
class ActivationStats:
    """Sufficient statistics computed only on the selection split."""

    class_count: np.ndarray
    class_sum: np.ndarray
    class_sum_sq: np.ndarray
    class_active_count: np.ndarray

    @property
    def n_images(self) -> int:
        return int(self.class_count.sum())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--sae_conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        choices=DEFAULT_CONDITIONS,
        help="The causal comparison is intentionally restricted to G-SAE/FT-SAE.",
    )
    parser.add_argument(
        "--num_latents",
        type=int,
        default=20,
        help="Number of preselected test latents per SAE (must be 10--20).",
    )
    parser.add_argument(
        "--selection_images_per_class",
        type=int,
        default=256,
        help="Stratified cap per class used for latent selection/matching.",
    )
    parser.add_argument(
        "--eval_images_per_class",
        type=int,
        default=64,
        help="Stratified held-out cap per class used for interventions.",
    )
    parser.add_argument(
        "--activation_threshold",
        type=float,
        default=1e-6,
        help="Feature activation threshold used only for prevalence/filtering.",
    )
    parser.add_argument(
        "--min_active_fraction",
        type=float,
        default=0.01,
        help="Minimum global selection-split prevalence for a candidate/control.",
    )
    parser.add_argument(
        "--min_target_active_fraction",
        type=float,
        default=0.05,
        help="Minimum target-class prevalence for a selected candidate.",
    )
    parser.add_argument(
        "--max_active_fraction",
        type=float,
        default=0.95,
        help="Discard near-universal features above this global prevalence.",
    )
    parser.add_argument(
        "--control_pool_size",
        type=int,
        default=50,
        help="Sample a control uniformly from this many nearest unmatched latents.",
    )
    parser.add_argument(
        "--amplify_mode",
        choices=["quantile", "multiply"],
        default="quantile",
        help="Quantile uses a train-calibrated floor; multiply scales natural z.",
    )
    parser.add_argument(
        "--amplify_quantile",
        type=float,
        default=0.9,
        help="Positive target-class selection activation quantile for steering.",
    )
    parser.add_argument(
        "--amplify_factor",
        type=float,
        default=2.0,
        help="z multiplier when --amplify_mode=multiply.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--debug_max_batches",
        type=int,
        default=None,
        help="Developer smoke-test cap for each pass; never use for paper results.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=5000,
        help="Latent-pair bootstrap replicates for 95%% confidence intervals.",
    )
    parser.add_argument(
        "--allow_missing_conditions",
        action="store_true",
        help="Skip missing checkpoints. Default is fail-fast to protect the comparison.",
    )
    parser.add_argument(
        "--no_write_examples",
        action="store_true",
        help="Do not write paired per-example effects (aggregate outputs remain).",
    )
    parser = add_common_args(
        parser, out_dir_default="out/rebuttal/causal_steering"
    )
    args = parser.parse_args(argv)

    if not 10 <= args.num_latents <= 20:
        parser.error("--num_latents must be between 10 and 20 inclusive")
    if args.selection_images_per_class < 1 or args.eval_images_per_class < 1:
        parser.error("selection/eval images per class must be positive")
    if not 0.0 < args.min_active_fraction < args.max_active_fraction <= 1.0:
        parser.error("require 0 < min_active_fraction < max_active_fraction <= 1")
    if not 0.0 < args.min_target_active_fraction <= 1.0:
        parser.error("--min_target_active_fraction must be in (0, 1]")
    if not 0.0 < args.amplify_quantile < 1.0:
        parser.error("--amplify_quantile must be in (0, 1)")
    if args.amplify_factor <= 1.0:
        parser.error("--amplify_factor must be > 1")
    if args.control_pool_size < 1:
        parser.error("--control_pool_size must be positive")
    if args.debug_max_batches is not None and args.debug_max_batches < 1:
        parser.error("--debug_max_batches must be positive")
    return args


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def resolve_local_path(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _PROJECT_ROOT / candidate
    return str(candidate.resolve())


def extract_targets(dataset: Dataset) -> np.ndarray:
    """Extract labels without decoding images, including nested Subsets."""

    if isinstance(dataset, Subset):
        parent = extract_targets(dataset.dataset)
        return parent[np.asarray(dataset.indices, dtype=np.int64)]

    if hasattr(dataset, "targets"):
        return np.asarray(getattr(dataset, "targets"), dtype=np.int64).reshape(-1)

    if hasattr(dataset, "labels"):
        labels = np.asarray(getattr(dataset, "labels"), dtype=np.int64).reshape(-1)
        mapping = getattr(dataset, "mapping", None)
        if mapping is not None:
            labels = np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)
        return labels

    raise TypeError(
        f"Cannot extract labels without image I/O from {type(dataset).__name__}"
    )


def make_stratified_subset(
    dataset: Dataset,
    num_classes: int,
    per_class: int,
    seed: int,
) -> tuple[Subset, np.ndarray]:
    targets = extract_targets(dataset)
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    missing: list[int] = []
    for class_index in range(num_classes):
        candidates = np.flatnonzero(targets == class_index)
        if candidates.size == 0:
            missing.append(class_index)
            continue
        candidates = rng.permutation(candidates)
        chosen.append(candidates[: min(per_class, candidates.size)])
    if missing:
        raise ValueError(f"dataset split has no images for class indices {missing}")
    indices = np.concatenate(chosen).astype(np.int64)
    indices = rng.permutation(indices)
    return Subset(dataset, indices.tolist()), indices


def capture_cls_activations(vit, cfg, images: Sequence[Any], device: str) -> torch.Tensor:
    inputs = vit.processor(
        images=list(images), text="", return_tensors="pt", padding=True
    ).to(device)
    _, cache = vit.run_with_cache([(cfg.block_layer, cfg.module_name)], **inputs)
    return cache[(cfg.block_layer, cfg.module_name)][:, 0, :].float()


def natural_sae_forward(sae, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output = sae(activations)
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError("SAE forward must return at least (reconstruction, latents)")
    reconstruction, latents = output[0], output[1]
    return reconstruction, latents


@torch.no_grad()
def compute_activation_stats(
    vit,
    sae,
    cfg,
    selection_ds: Dataset,
    num_classes: int,
    args: argparse.Namespace,
) -> ActivationStats:
    d_sae = int(sae.d_sae)
    counts = np.zeros(num_classes, dtype=np.int64)
    sums = np.zeros((num_classes, d_sae), dtype=np.float64)
    sums_sq = np.zeros((num_classes, d_sae), dtype=np.float64)
    active = np.zeros((num_classes, d_sae), dtype=np.int64)
    loader = DataLoader(
        selection_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_hf_collate_fn,
        num_workers=args.num_workers,
    )

    for batch_index, (images, labels) in enumerate(
        tqdm(loader, desc="selection activations", leave=False)
    ):
        if (
            args.debug_max_batches is not None
            and batch_index >= args.debug_max_batches
        ):
            break
        cls_acts = capture_cls_activations(vit, cfg, images, args.device)
        _, latents = natural_sae_forward(sae, cls_acts)
        z = latents.detach().float().cpu().numpy()
        y = labels.numpy()
        for class_index in np.unique(y):
            mask = y == class_index
            z_class = z[mask]
            counts[class_index] += int(mask.sum())
            sums[class_index] += z_class.sum(axis=0, dtype=np.float64)
            sums_sq[class_index] += np.square(z_class, dtype=np.float64).sum(
                axis=0, dtype=np.float64
            )
            active[class_index] += (
                z_class > args.activation_threshold
            ).sum(axis=0, dtype=np.int64)

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(
            "selection pass observed no samples for classes "
            f"{missing}; remove --debug_max_batches or increase it"
        )
    return ActivationStats(counts, sums, sums_sq, active)


def derive_selection_metrics(
    stats: ActivationStats,
    decoder_norm: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    counts = stats.class_count.astype(np.float64)
    total_count = float(counts.sum())
    means = stats.class_sum / counts[:, None]
    second = stats.class_sum_sq / counts[:, None]
    variances = np.maximum(second - np.square(means), 0.0)

    total_sum = stats.class_sum.sum(axis=0)
    total_sum_sq = stats.class_sum_sq.sum(axis=0)
    rest_count = total_count - counts
    rest_mean = (total_sum[None, :] - stats.class_sum) / rest_count[:, None]
    rest_second = (
        total_sum_sq[None, :] - stats.class_sum_sq
    ) / rest_count[:, None]
    rest_var = np.maximum(rest_second - np.square(rest_mean), 0.0)
    pooled_sd = np.sqrt(0.5 * (variances + rest_var) + 1e-12)
    score = (means - rest_mean) / pooled_sd

    class_active_fraction = stats.class_active_count / counts[:, None]
    global_active_fraction = stats.class_active_count.sum(axis=0) / total_count
    global_mean = total_sum / total_count

    eligible = (
        np.isfinite(score)
        & (score > 0)
        & (means > args.activation_threshold)
        & (class_active_fraction >= args.min_target_active_fraction)
        & (global_active_fraction[None, :] >= args.min_active_fraction)
        & (global_active_fraction[None, :] <= args.max_active_fraction)
        & np.isfinite(decoder_norm[None, :])
        & (decoder_norm[None, :] > 0)
    )
    return {
        "means": means,
        "rest_mean": rest_mean,
        "score": score,
        "eligible": eligible,
        "class_active_fraction": class_active_fraction,
        "global_active_fraction": global_active_fraction,
        "global_mean": global_mean,
        "decoder_norm": decoder_norm,
    }


def select_class_balanced_features(
    score: np.ndarray,
    eligible: np.ndarray,
    num_latents: int,
    seed: int,
) -> list[tuple[int, int, float]]:
    """Select unique top features under a fixed, seeded per-class quota.

    The quota depends only on ``num_latents``, class count, and ``seed``.  It
    therefore gives G-SAE and FT-SAE the same target-class composition even
    though the actual dictionary features are selected independently.
    """

    n_classes, d_sae = score.shape
    rng = np.random.default_rng(seed)
    class_order = rng.permutation(n_classes).tolist()
    selected: list[tuple[int, int, float]] = []
    used_features: set[int] = set()
    base_quota, remainder = divmod(num_latents, n_classes)
    quota = np.full(n_classes, base_quota, dtype=np.int64)
    for class_index in class_order[:remainder]:
        quota[class_index] += 1

    def best_for_class(class_index: int) -> tuple[int, float] | None:
        row = np.where(eligible[class_index], score[class_index], -np.inf).copy()
        if used_features:
            row[np.fromiter(used_features, dtype=np.int64)] = -np.inf
        feature_id = int(np.argmax(row))
        value = float(row[feature_id])
        if not np.isfinite(value):
            return None
        return feature_id, value

    for class_index in class_order:
        for _ in range(int(quota[class_index])):
            candidate = best_for_class(class_index)
            if candidate is None:
                raise ValueError(
                    f"class {class_index} lacks enough eligible unique features "
                    f"for its preregistered quota of {quota[class_index]}"
                )
            feature_id, value = candidate
            selected.append((class_index, feature_id, value))
            used_features.add(feature_id)
    if len(selected) != num_latents:
        raise AssertionError(
            f"selection produced {len(selected)} rather than {num_latents} features"
        )
    return selected


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def match_random_controls(
    selected: Sequence[tuple[int, int, float]],
    metrics: dict[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> list[tuple[int, float, int]]:
    """Return (control feature id, match distance, sampled pool rank) per pair."""

    rng = np.random.default_rng(seed)
    selected_ids = {feature_id for _, feature_id, _ in selected}
    used = set(selected_ids)
    global_active = metrics["global_active_fraction"]
    global_mean = metrics["global_mean"]
    decoder_norm = metrics["decoder_norm"]
    base_candidate = (
        np.isfinite(global_mean)
        & np.isfinite(global_active)
        & np.isfinite(decoder_norm)
        & (global_active >= args.min_active_fraction)
        & (global_active <= args.max_active_fraction)
        & (decoder_norm > 0)
    )
    matches: list[tuple[int, float, int]] = []

    for target_class, feature_id, _ in selected:
        candidate_ids = np.flatnonzero(base_candidate)
        if used:
            candidate_ids = candidate_ids[
                ~np.isin(candidate_ids, np.fromiter(used, dtype=np.int64))
            ]
        # A control must fire at least once for this target class, or later
        # amplification calibration (which needs a positive-activation
        # quantile) is undefined. This does not require the target-class
        # prevalence floor used for *selected* features (min_target_active_
        # fraction) -- only that calibration is well-defined.
        candidate_ids = candidate_ids[
            metrics["class_active_fraction"][target_class, candidate_ids] > 0.0
        ]
        if candidate_ids.size == 0:
            raise ValueError(
                "no unmatched active features with nonzero target-class "
                f"activity remain for controls (target_class={target_class})"
            )

        # Match intervention scale as well as raw prevalence.  All quantities
        # come from the selection split.
        matrix = np.column_stack(
            [
                np.log1p(global_mean[candidate_ids]),
                _logit(global_active[candidate_ids]),
                np.log1p(metrics["means"][target_class, candidate_ids]),
                _logit(
                    metrics["class_active_fraction"][target_class, candidate_ids]
                ),
                np.log(decoder_norm[candidate_ids] + 1e-12),
            ]
        )
        selected_vector = np.asarray(
            [
                np.log1p(global_mean[feature_id]),
                _logit(np.asarray([global_active[feature_id]]))[0],
                np.log1p(metrics["means"][target_class, feature_id]),
                _logit(
                    np.asarray(
                        [metrics["class_active_fraction"][target_class, feature_id]]
                    )
                )[0],
                np.log(decoder_norm[feature_id] + 1e-12),
            ]
        )
        scale = np.nanstd(matrix, axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        distance = np.sqrt(
            np.square((matrix - selected_vector[None, :]) / scale[None, :]).sum(
                axis=1
            )
        )
        order = np.argsort(distance, kind="stable")
        pool_n = min(args.control_pool_size, order.size)
        sampled_rank = int(rng.integers(pool_n))
        candidate_position = int(order[sampled_rank])
        control_id = int(candidate_ids[candidate_position])
        matches.append(
            (control_id, float(distance[candidate_position]), sampled_rank + 1)
        )
        used.add(control_id)
    return matches


def build_manifest(
    dataset: str,
    sae_condition: str,
    sae_path: str,
    sae,
    cfg,
    classnames: Sequence[str],
    stats: ActivationStats,
    metrics: dict[str, np.ndarray],
    selected: Sequence[tuple[int, int, float]],
    controls: Sequence[tuple[int, float, int]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, ((target, feature_id, score), control) in enumerate(
        zip(selected, controls), start=1
    ):
        control_id, match_distance, pool_rank = control
        pair_id = f"{dataset}:{sae_condition}:{rank:02d}"
        for role, current_id, matched_id in [
            ("selected", feature_id, control_id),
            ("control", control_id, feature_id),
        ]:
            rows.append(
                {
                    "dataset": dataset,
                    "sae_condition": sae_condition,
                    "pair_id": pair_id,
                    "feature_role": role,
                    "feature_id": int(current_id),
                    "selection_rank": rank,
                    "target_class_index": int(target),
                    "target_class": classnames[target],
                    "selectivity_score": float(metrics["score"][target, current_id]),
                    "target_mean_activation": float(
                        metrics["means"][target, current_id]
                    ),
                    "rest_mean_activation": float(
                        metrics["rest_mean"][target, current_id]
                    ),
                    "target_active_fraction": float(
                        metrics["class_active_fraction"][target, current_id]
                    ),
                    "global_mean_activation": float(
                        metrics["global_mean"][current_id]
                    ),
                    "global_active_fraction": float(
                        metrics["global_active_fraction"][current_id]
                    ),
                    "decoder_norm": float(metrics["decoder_norm"][current_id]),
                    "matched_feature_id": int(matched_id),
                    "control_match_distance": float(match_distance),
                    "control_pool_rank": int(pool_rank),
                    "amplify_mode": args.amplify_mode,
                    "amplify_value": float("nan"),
                    "amplify_quantile": float(args.amplify_quantile),
                    "n_positive_for_amplify": 0,
                    "sae_path": sae_path,
                    "block_layer": int(cfg.block_layer),
                    "module_name": str(cfg.module_name),
                    "d_sae": int(sae.d_sae),
                    "n_selection_images": stats.n_images,
                }
            )
    return rows


@torch.no_grad()
def calibrate_amplification(
    vit,
    sae,
    cfg,
    selection_ds: Dataset,
    manifest: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Fill manifest amplification values from target-class positive train z."""

    if args.amplify_mode == "multiply":
        for row in manifest:
            row["amplify_value"] = float(args.amplify_factor)
            row["n_positive_for_amplify"] = -1
        return

    feature_ids = np.asarray([row["feature_id"] for row in manifest], dtype=np.int64)
    values: list[list[np.ndarray]] = [[] for _ in manifest]
    loader = DataLoader(
        selection_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_hf_collate_fn,
        num_workers=args.num_workers,
    )
    feature_tensor = torch.as_tensor(feature_ids, device=args.device)
    for batch_index, (images, labels) in enumerate(
        tqdm(loader, desc="amplification calibration", leave=False)
    ):
        if (
            args.debug_max_batches is not None
            and batch_index >= args.debug_max_batches
        ):
            break
        cls_acts = capture_cls_activations(vit, cfg, images, args.device)
        _, latents = natural_sae_forward(sae, cls_acts)
        selected_z = latents.index_select(-1, feature_tensor).detach().float().cpu().numpy()
        y = labels.numpy()
        for column, row in enumerate(manifest):
            class_mask = y == row["target_class_index"]
            if class_mask.any():
                positive = selected_z[class_mask, column]
                positive = positive[positive > args.activation_threshold]
                if positive.size:
                    values[column].append(positive)

    for row, chunks in zip(manifest, values):
        positive = np.concatenate(chunks) if chunks else np.asarray([], dtype=float)
        if positive.size == 0:
            raise ValueError(
                "no positive target-class activations available to calibrate "
                f"{row['feature_role']} feature {row['feature_id']}"
            )
        row["amplify_value"] = float(
            np.quantile(positive, args.amplify_quantile)
        )
        row["n_positive_for_amplify"] = int(positive.size)


def error_preserving_latent_edit(
    sae,
    activations: torch.Tensor,
    feature_id: int,
    intervention: str,
    amplify_mode: str = "quantile",
    amplify_value: float = 0.0,
) -> torch.Tensor:
    """Edit exactly one natural latent while retaining SAE reconstruction error."""

    reconstruction, latents = natural_sae_forward(sae, activations)
    edited_latents = latents.clone()
    if intervention == "ablate":
        edited_latents[..., feature_id] = 0.0
    elif intervention == "amplify":
        if amplify_mode == "quantile":
            floor = torch.as_tensor(
                amplify_value,
                dtype=edited_latents.dtype,
                device=edited_latents.device,
            )
            edited_latents[..., feature_id] = torch.maximum(
                edited_latents[..., feature_id], floor
            )
        elif amplify_mode == "multiply":
            edited_latents[..., feature_id] *= amplify_value
        else:
            raise ValueError(f"unknown amplify_mode={amplify_mode!r}")
    else:
        raise ValueError(f"unknown intervention={intervention!r}")

    delta_z = (
        edited_latents[..., feature_id] - latents[..., feature_id]
    ).unsqueeze(-1)
    decoder_direction = sae.W_dec[feature_id].to(
        dtype=reconstruction.dtype, device=reconstruction.device
    )
    edited_reconstruction = reconstruction + delta_z * decoder_direction
    reconstruction_error = activations.to(reconstruction.dtype) - reconstruction
    return edited_reconstruction + reconstruction_error


def make_intervention_hook(
    sae,
    cfg,
    feature_id: int | None,
    intervention: str,
    amplify_mode: str = "quantile",
    amplify_value: float = 0.0,
) -> Hook:
    """Create a CLS-only HF CLIP residual-stream intervention hook."""

    def hook_fn(activations: torch.Tensor):
        output = activations.clone()
        cls_activations = output[:, 0, :]
        if intervention == "reconstruct":
            reconstruction, _ = natural_sae_forward(sae, cls_activations)
            edited = reconstruction
        else:
            if feature_id is None:
                raise ValueError("feature_id is required for a latent intervention")
            edited = error_preserving_latent_edit(
                sae=sae,
                activations=cls_activations,
                feature_id=feature_id,
                intervention=intervention,
                amplify_mode=amplify_mode,
                amplify_value=amplify_value,
            )
        output[:, 0, :] = edited.to(output.dtype)
        return (output,)

    return Hook(
        cfg.block_layer,
        cfg.module_name,
        hook_fn,
        return_module_output=False,
        is_custom=False,
    )


@torch.no_grad()
def run_prediction_pass(
    vit,
    dataset: Dataset,
    text_features: torch.Tensor,
    args: argparse.Namespace,
    hook: Hook | None = None,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_hf_collate_fn,
        num_workers=args.num_workers,
    )
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    normalized_text = F.normalize(text_features.float(), dim=-1)
    logit_scale = vit.model.logit_scale.exp().detach()

    for batch_index, (images, labels) in enumerate(
        tqdm(loader, desc="prediction pass", leave=False)
    ):
        if (
            args.debug_max_batches is not None
            and batch_index >= args.debug_max_batches
        ):
            break
        inputs = vit.processor(
            images=images, text="", return_tensors="pt", padding=True
        ).to(args.device)
        if hook is None:
            output = vit(return_type="output", **inputs)
        else:
            output = vit.run_with_hooks([hook], return_type="output", **inputs)
        image_features = F.normalize(output.image_embeds.float(), dim=-1)
        logits = logit_scale.float() * image_features @ normalized_text.t()
        logits_chunks.append(logits.cpu().numpy())
        labels_chunks.append(labels.numpy())

    if not logits_chunks:
        raise ValueError("prediction pass produced no batches")
    logits_np = np.concatenate(logits_chunks)
    labels_np = np.concatenate(labels_chunks).astype(np.int64)
    probabilities = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy()
    return {
        "logits": logits_np,
        "probabilities": probabilities,
        "predictions": logits_np.argmax(axis=-1).astype(np.int64),
        "labels": labels_np,
    }


def target_metrics(predictions: dict[str, np.ndarray], target: int) -> dict[str, np.ndarray]:
    logits = predictions["logits"]
    probabilities = predictions["probabilities"]
    other_logits = logits.copy()
    other_logits[:, target] = -np.inf
    return {
        "logit": logits[:, target],
        "probability": probabilities[:, target],
        "margin": logits[:, target] - other_logits.max(axis=1),
    }


def _safe_rate(numerator: np.ndarray, denominator_mask: np.ndarray) -> float:
    denominator = int(denominator_mask.sum())
    if denominator == 0:
        return float("nan")
    return float(numerator[denominator_mask].mean() * 100.0)


def aggregate_feature_effects(
    manifest_row: dict[str, Any],
    intervention: str,
    baseline: dict[str, np.ndarray],
    intervened: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    if not np.array_equal(baseline["labels"], intervened["labels"]):
        raise ValueError("baseline/intervention label order mismatch")
    target = int(manifest_row["target_class_index"])
    labels = baseline["labels"]
    base_pred = baseline["predictions"]
    int_pred = intervened["predictions"]
    base_target = target_metrics(baseline, target)
    int_target = target_metrics(intervened, target)
    delta_logit = int_target["logit"] - base_target["logit"]
    delta_probability = int_target["probability"] - base_target["probability"]
    delta_margin = int_target["margin"] - base_target["margin"]

    masks = {
        "all": np.ones(labels.shape[0], dtype=bool),
        "target": labels == target,
        "non_target": labels != target,
    }
    rows: list[dict[str, Any]] = []
    for group, mask in masks.items():
        n_images = int(mask.sum())
        if n_images == 0:
            continue
        base_correct = base_pred[mask] == labels[mask]
        int_correct = int_pred[mask] == labels[mask]
        base_is_target = base_pred[mask] == target
        int_is_target = int_pred[mask] == target
        group_delta_margin = delta_margin[mask]
        directional = (
            -float(group_delta_margin.mean())
            if intervention == "ablate"
            else float(group_delta_margin.mean())
        )
        flip_to = (int_pred[mask] == target) & (base_pred[mask] != target)
        flip_from = (int_pred[mask] != target) & (base_pred[mask] == target)
        rows.append(
            {
                "dataset": manifest_row["dataset"],
                "sae_condition": manifest_row["sae_condition"],
                "pair_id": manifest_row["pair_id"],
                "feature_role": manifest_row["feature_role"],
                "feature_id": manifest_row["feature_id"],
                "selection_rank": manifest_row["selection_rank"],
                "target_class_index": target,
                "target_class": manifest_row["target_class"],
                "intervention": intervention,
                "causal_reference": "raw_model_no_hook",
                "eval_group": group,
                "causal_reference": "raw_model_no_hook",
                "n_images": n_images,
                "baseline_accuracy": float(base_correct.mean() * 100.0),
                "intervened_accuracy": float(int_correct.mean() * 100.0),
                "delta_accuracy_pp": float(
                    (int_correct.mean() - base_correct.mean()) * 100.0
                ),
                "baseline_target_rate": float(base_is_target.mean() * 100.0),
                "intervened_target_rate": float(int_is_target.mean() * 100.0),
                "delta_target_rate_pp": float(
                    (int_is_target.mean() - base_is_target.mean()) * 100.0
                ),
                "mean_delta_target_logit": float(delta_logit[mask].mean()),
                "mean_delta_target_probability": float(
                    delta_probability[mask].mean()
                ),
                "mean_delta_target_margin": float(group_delta_margin.mean()),
                "se_delta_target_margin": float(
                    group_delta_margin.std(ddof=1) / math.sqrt(n_images)
                )
                if n_images > 1
                else float("nan"),
                "flip_to_target_rate": _safe_rate(
                    flip_to, base_pred[mask] != target
                ),
                "flip_from_target_rate": _safe_rate(
                    flip_from, base_pred[mask] == target
                ),
                "directional_margin_effect": directional,
            }
        )
    return rows


def build_example_rows(
    manifest_row: dict[str, Any],
    intervention: str,
    classnames: Sequence[str],
    eval_source_indices: np.ndarray,
    baseline: dict[str, np.ndarray],
    intervened: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    n = baseline["labels"].shape[0]
    source_indices = eval_source_indices[:n]
    target = int(manifest_row["target_class_index"])
    base_target = target_metrics(baseline, target)
    int_target = target_metrics(intervened, target)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        label = int(baseline["labels"][i])
        base_pred = int(baseline["predictions"][i])
        int_pred = int(intervened["predictions"][i])
        rows.append(
            {
                "dataset": manifest_row["dataset"],
                "sae_condition": manifest_row["sae_condition"],
                "pair_id": manifest_row["pair_id"],
                "feature_role": manifest_row["feature_role"],
                "feature_id": manifest_row["feature_id"],
                "selection_rank": manifest_row["selection_rank"],
                "target_class_index": target,
                "target_class": manifest_row["target_class"],
                "intervention": intervention,
                "sample_order": i,
                "eval_source_index": int(source_indices[i]),
                "label_index": label,
                "label": classnames[label],
                "is_target_class": bool(label == target),
                "baseline_prediction": classnames[base_pred],
                "intervened_prediction": classnames[int_pred],
                "baseline_correct": bool(base_pred == label),
                "intervened_correct": bool(int_pred == label),
                "baseline_target_logit": float(base_target["logit"][i]),
                "intervened_target_logit": float(int_target["logit"][i]),
                "delta_target_logit": float(
                    int_target["logit"][i] - base_target["logit"][i]
                ),
                "baseline_target_probability": float(
                    base_target["probability"][i]
                ),
                "intervened_target_probability": float(
                    int_target["probability"][i]
                ),
                "delta_target_probability": float(
                    int_target["probability"][i] - base_target["probability"][i]
                ),
                "baseline_target_margin": float(base_target["margin"][i]),
                "intervened_target_margin": float(int_target["margin"][i]),
                "delta_target_margin": float(
                    int_target["margin"][i] - base_target["margin"][i]
                ),
            }
        )
    return rows


def reconstruction_control_row(
    dataset: str,
    condition: str,
    baseline: dict[str, np.ndarray],
    reconstruction: dict[str, np.ndarray],
) -> dict[str, Any]:
    if not np.array_equal(baseline["labels"], reconstruction["labels"]):
        raise ValueError("baseline/reconstruction label order mismatch")
    labels = baseline["labels"]
    base_correct = baseline["predictions"] == labels
    recon_correct = reconstruction["predictions"] == labels
    agreement = baseline["predictions"] == reconstruction["predictions"]
    p = np.clip(baseline["probabilities"], 1e-12, 1.0)
    q = np.clip(reconstruction["probabilities"], 1e-12, 1.0)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    return {
        "dataset": dataset,
        "sae_condition": condition,
        "reference": "raw_model_no_hook",
        "control_intervention": "natural_sae_reconstruction_without_error",
        "n_images": int(labels.size),
        "baseline_accuracy": float(base_correct.mean() * 100.0),
        "reconstruction_accuracy": float(recon_correct.mean() * 100.0),
        "delta_accuracy_pp": float(
            (recon_correct.mean() - base_correct.mean()) * 100.0
        ),
        "prediction_agreement": float(agreement.mean() * 100.0),
        "mean_probability_kl": float(kl.mean()),
        "mean_absolute_logit_delta": float(
            np.abs(reconstruction["logits"] - baseline["logits"]).mean()
        ),
    }


def build_paired_effects(effects_df: pd.DataFrame) -> pd.DataFrame:
    if effects_df.empty:
        return pd.DataFrame()
    keys = [
        "dataset",
        "sae_condition",
        "pair_id",
        "selection_rank",
        "target_class_index",
        "target_class",
        "intervention",
        "eval_group",
    ]
    metrics = [
        "delta_accuracy_pp",
        "delta_target_rate_pp",
        "mean_delta_target_logit",
        "mean_delta_target_probability",
        "mean_delta_target_margin",
        "directional_margin_effect",
    ]
    selected = effects_df[effects_df.feature_role == "selected"][
        keys + ["feature_id", *metrics]
    ].rename(
        columns={
            "feature_id": "selected_feature_id",
            **{metric: f"selected_{metric}" for metric in metrics},
        }
    )
    control = effects_df[effects_df.feature_role == "control"][
        keys + ["feature_id", *metrics]
    ].rename(
        columns={
            "feature_id": "control_feature_id",
            **{metric: f"control_{metric}" for metric in metrics},
        }
    )
    paired = selected.merge(control, on=keys, how="inner", validate="one_to_one")
    for metric in metrics:
        paired[f"selected_minus_control_{metric}"] = (
            paired[f"selected_{metric}"] - paired[f"control_{metric}"]
        )
    return paired.sort_values(keys).reset_index(drop=True)


def bootstrap_mean_ci(
    values: Iterable[float], n_bootstrap: int, seed: int
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(array.mean())
    if array.size == 1 or n_bootstrap < 1:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(n_bootstrap, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def build_aggregate_results(
    paired_df: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    if paired_df.empty:
        return pd.DataFrame()
    primary = paired_df[
        paired_df.apply(
            lambda row: row["eval_group"] == PRIMARY_GROUP[row["intervention"]],
            axis=1,
        )
    ]
    rows: list[dict[str, Any]] = []
    for (dataset, condition, intervention), group in primary.groupby(
        ["dataset", "sae_condition", "intervention"], sort=True
    ):
        adjusted = group[
            "selected_minus_control_directional_margin_effect"
        ].to_numpy()
        selected = group["selected_directional_margin_effect"].to_numpy()
        control = group["control_directional_margin_effect"].to_numpy()
        mean, low, high = bootstrap_mean_ci(
            adjusted,
            args.bootstrap_samples,
            stable_seed(args.seed, dataset, condition, intervention, "bootstrap"),
        )
        rows.append(
            {
                "dataset": dataset,
                "sae_condition": condition,
                "intervention": intervention,
                "primary_eval_group": PRIMARY_GROUP[intervention],
                "n_latent_pairs": int(len(group)),
                "mean_selected_directional_margin_effect": float(
                    np.nanmean(selected)
                ),
                "mean_control_directional_margin_effect": float(
                    np.nanmean(control)
                ),
                "mean_control_adjusted_directional_margin_effect": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def build_condition_contrasts(
    paired_df: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    if paired_df.empty:
        return pd.DataFrame()
    primary = paired_df[
        paired_df.apply(
            lambda row: row["eval_group"] == PRIMARY_GROUP[row["intervention"]],
            axis=1,
        )
    ]
    rows: list[dict[str, Any]] = []
    metric = "selected_minus_control_directional_margin_effect"
    for (dataset, intervention), group in primary.groupby(
        ["dataset", "intervention"], sort=True
    ):
        generic = group[group.sae_condition == "gsae"][metric].dropna().to_numpy()
        finetuned = group[group.sae_condition == "ftsae"][metric].dropna().to_numpy()
        if generic.size == 0 or finetuned.size == 0:
            continue
        observed = float(finetuned.mean() - generic.mean())
        if args.bootstrap_samples > 0:
            rng = np.random.default_rng(
                stable_seed(args.seed, dataset, intervention, "condition_contrast")
            )
            generic_draw = rng.choice(
                generic,
                size=(args.bootstrap_samples, generic.size),
                replace=True,
            ).mean(axis=1)
            finetuned_draw = rng.choice(
                finetuned,
                size=(args.bootstrap_samples, finetuned.size),
                replace=True,
            ).mean(axis=1)
            low, high = np.quantile(finetuned_draw - generic_draw, [0.025, 0.975])
        else:
            low = high = float("nan")
        rows.append(
            {
                "dataset": dataset,
                "intervention": intervention,
                "contrast_type": "arm_level_independent_latent_sets",
                "primary_eval_group": PRIMARY_GROUP[intervention],
                "n_gsae_latent_pairs": int(generic.size),
                "n_ftsae_latent_pairs": int(finetuned.size),
                "ftsae_minus_gsae_control_adjusted_effect": observed,
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def checkpoint_preflight(
    args: argparse.Namespace,
    registry: dict[str, Any],
    lora_checkpoints: dict[str, str],
    sae_paths: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset in args.datasets:
        lora_path = resolve_local_path(lora_checkpoints.get(dataset))
        lora_ok = bool(lora_path and os.path.isfile(lora_path))
        if not lora_ok:
            missing.append(f"{dataset}: LoRA checkpoint ({lora_path})")
        for condition in args.sae_conditions:
            sae_path = resolve_local_path(
                resolve_sae_path(condition, dataset, sae_paths, args)
            )
            sae_ok = bool(sae_path and os.path.isfile(sae_path))
            if not sae_ok:
                missing.append(f"{dataset}: {condition} checkpoint ({sae_path})")
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_registered": dataset in registry,
                    "lora_path": lora_path,
                    "lora_exists": lora_ok,
                    "sae_condition": condition,
                    "sae_path": sae_path,
                    "sae_exists": sae_ok,
                }
            )
    return rows, missing


def write_outputs(
    out_dir: str,
    manifest_rows: list[dict[str, Any]],
    effect_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    reconstruction_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    run_metadata: dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    manifest_df = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    effects_df = pd.DataFrame(effect_rows, columns=EFFECT_COLUMNS)
    paired_df = build_paired_effects(effects_df)
    aggregate_df = build_aggregate_results(paired_df, args)
    contrasts_df = build_condition_contrasts(paired_df, args)

    manifest_df.to_csv(os.path.join(out_dir, "latent_manifest.csv"), index=False)
    effects_df.to_csv(os.path.join(out_dir, "effects.csv"), index=False)
    paired_df.to_csv(os.path.join(out_dir, "paired_effects.csv"), index=False)
    aggregate_df.to_csv(
        os.path.join(out_dir, "aggregate_results.csv"), index=False
    )
    contrasts_df.to_csv(
        os.path.join(out_dir, "condition_contrasts.csv"), index=False
    )
    pd.DataFrame(reconstruction_rows).to_csv(
        os.path.join(out_dir, "reconstruction_controls.csv"), index=False
    )
    if not args.no_write_examples:
        pd.DataFrame(example_rows, columns=EXAMPLE_COLUMNS).to_csv(
            os.path.join(out_dir, "per_example_effects.csv.gz"),
            index=False,
            compression="gzip",
        )
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as handle:
        json.dump(run_metadata, handle, indent=2, default=str)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    default_paths = _PROJECT_ROOT / "configs" / "rebuttal_sae_paths.json"
    if args.sae_paths is None and default_paths.is_file():
        args.sae_paths = str(default_paths)

    registry, lora_checkpoints, sae_paths = load_common_registry_args(args)
    # Common registries historically mix absolute paths and paths relative to
    # patchsae/.  Normalize the model map before get_vit performs its own
    # existence check so invocation from the repository root is also safe.
    lora_checkpoints = {
        name: resolve_local_path(path) for name, path in lora_checkpoints.items()
    }
    preflight_rows, missing = checkpoint_preflight(
        args, registry, lora_checkpoints, sae_paths
    )
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "preflight.json"), "w") as handle:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "args": vars(args),
                "checks": preflight_rows,
                "missing": missing,
            },
            handle,
            indent=2,
            default=str,
        )
    print(pd.DataFrame(preflight_rows).to_string(index=False))
    if args.dry_run:
        print(f"\nWrote {args.out_dir}/preflight.json; no model was loaded.")
        if missing:
            print("Missing requirements:")
            for item in missing:
                print(f"  - {item}")
        return 0
    if missing and not args.allow_missing_conditions:
        raise SystemExit(
            "Missing required comparison inputs:\n  - "
            + "\n  - ".join(missing)
            + "\nUse --dry_run to inspect or --allow_missing_conditions to skip."
        )

    manifest_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    run_metadata: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_version": 1,
        "causal_reference": "raw_model_no_hook",
        "latent_edit": "x + D(z_intervened) - D(z)",
        "reconstruction_control": "D(z) without retained reconstruction error",
        "condition_contrast": "arm-level; latent sets are independently selected",
        "args": vars(args),
        "preflight": preflight_rows,
        "completed_cells": [],
        "skipped_cells": [],
    }

    for dataset_name in args.datasets:
        try:
            classnames, train_ds, eval_ds = build_dataset_splits(
                dataset_name, registry[dataset_name], args
            )
            selection_ds, _ = make_stratified_subset(
                train_ds,
                len(classnames),
                args.selection_images_per_class,
                stable_seed(args.seed, dataset_name, "selection_subset"),
            )
            eval_subset, eval_indices = make_stratified_subset(
                eval_ds,
                len(classnames),
                args.eval_images_per_class,
                stable_seed(args.seed, dataset_name, "evaluation_subset"),
            )
            vit = get_vit("lora", dataset_name, lora_checkpoints, args)
            if vit is None:
                raise FileNotFoundError("LoRA checkpoint unavailable")
            text_features = calculate_text_features(vit, args.device, classnames)
            baseline = run_prediction_pass(
                vit, eval_subset, text_features, args, hook=None
            )
        except Exception as exc:
            if not args.allow_missing_conditions:
                raise
            run_metadata["skipped_cells"].append(
                {"dataset": dataset_name, "sae_condition": "*", "reason": str(exc)}
            )
            continue

        for condition in args.sae_conditions:
            condition_path = resolve_local_path(
                resolve_sae_path(condition, dataset_name, sae_paths, args)
            )
            if not condition_path or not os.path.isfile(condition_path):
                run_metadata["skipped_cells"].append(
                    {
                        "dataset": dataset_name,
                        "sae_condition": condition,
                        "reason": f"missing checkpoint: {condition_path}",
                    }
                )
                continue
            cell_start = time.time()
            try:
                sae, cfg = get_sae(condition_path, args.device)
                validate_sae_condition_provenance(condition, cfg)
                if int(cfg.d_in) != 768:
                    raise ValueError(f"expected d_in=768, got {cfg.d_in}")
                if int(cfg.block_layer) != -2 or str(cfg.module_name) != "resid":
                    raise ValueError(
                        "comparison requires matched -2/resid SAEs; got "
                        f"{cfg.block_layer}/{cfg.module_name}"
                    )
                decoder_norm = (
                    sae.W_dec.detach().float().norm(dim=-1).cpu().numpy()
                )
                stats = compute_activation_stats(
                    vit, sae, cfg, selection_ds, len(classnames), args
                )
                metrics = derive_selection_metrics(stats, decoder_norm, args)
                selected = select_class_balanced_features(
                    metrics["score"],
                    metrics["eligible"],
                    args.num_latents,
                    stable_seed(args.seed, dataset_name, "target_class_quota"),
                )
                controls = match_random_controls(
                    selected,
                    metrics,
                    args,
                    stable_seed(args.seed, dataset_name, condition, "controls"),
                )
                cell_manifest = build_manifest(
                    dataset_name,
                    condition,
                    condition_path,
                    sae,
                    cfg,
                    classnames,
                    stats,
                    metrics,
                    selected,
                    controls,
                    args,
                )
                calibrate_amplification(
                    vit, sae, cfg, selection_ds, cell_manifest, args
                )

                reconstruction_hook = make_intervention_hook(
                    sae, cfg, None, "reconstruct"
                )
                reconstructed = run_prediction_pass(
                    vit,
                    eval_subset,
                    text_features,
                    args,
                    hook=reconstruction_hook,
                )
                reconstruction_rows.append(
                    reconstruction_control_row(
                        dataset_name, condition, baseline, reconstructed
                    )
                )

                for manifest_row in tqdm(
                    cell_manifest,
                    desc=f"{dataset_name}/{condition} features",
                    leave=False,
                ):
                    for intervention in ("ablate", "amplify"):
                        hook = make_intervention_hook(
                            sae=sae,
                            cfg=cfg,
                            feature_id=int(manifest_row["feature_id"]),
                            intervention=intervention,
                            amplify_mode=args.amplify_mode,
                            amplify_value=float(manifest_row["amplify_value"]),
                        )
                        intervened = run_prediction_pass(
                            vit, eval_subset, text_features, args, hook=hook
                        )
                        effect_rows.extend(
                            aggregate_feature_effects(
                                manifest_row,
                                intervention,
                                baseline,
                                intervened,
                            )
                        )
                        if not args.no_write_examples:
                            example_rows.extend(
                                build_example_rows(
                                    manifest_row,
                                    intervention,
                                    classnames,
                                    eval_indices,
                                    baseline,
                                    intervened,
                                )
                            )
                    flush()

                manifest_rows.extend(cell_manifest)
                run_metadata["completed_cells"].append(
                    {
                        "dataset": dataset_name,
                        "sae_condition": condition,
                        "sae_path": condition_path,
                        "seconds": time.time() - cell_start,
                        "n_selected_latents": len(selected),
                        "n_controls": len(controls),
                        "n_selection_images": stats.n_images,
                        "n_eval_images": int(baseline["labels"].size),
                    }
                )
                # Checkpoint intermediate results after each expensive cell.
                write_outputs(
                    args.out_dir,
                    manifest_rows,
                    effect_rows,
                    example_rows,
                    reconstruction_rows,
                    args,
                    run_metadata,
                )
            except Exception as exc:
                if not args.allow_missing_conditions:
                    raise
                run_metadata["skipped_cells"].append(
                    {
                        "dataset": dataset_name,
                        "sae_condition": condition,
                        "reason": repr(exc),
                    }
                )
                print(f"[SKIP] {dataset_name}/{condition}: {exc}")
        clear_model_caches()

    run_metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_outputs(
        args.out_dir,
        manifest_rows,
        effect_rows,
        example_rows,
        reconstruction_rows,
        args,
        run_metadata,
    )
    print(f"Wrote causal steering outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

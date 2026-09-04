#!/usr/bin/env python3
"""
Paired target-model SAE compatibility evaluation.

This script answers a different question from the broad DAMS sweep:

    Given a fine-tuned/adapted target model, which SAE is the better coordinate
    system for that *same* model's hidden states: the generic base SAE or the
    domain-adapted SAE?

The comparison is paired: both SAEs see the same dataset, the same examples,
the same target model, and, by default, the same transformer layer. This avoids
the weak comparison where the base SAE is evaluated on base-CLIP activations
while the adapted SAE is evaluated on adapted-CLIP activations.

For each target-model activation matrix H and SAE s, we report legacy holistic
diagnostics and the task-subspace/feature-level DAMS final components:

  R2   = reconstruction explained variance of H by SAE_s.
  SUS  = chance-normalised held-out balanced accuracy of a ridge readout on
         frozen SAE activations.
  DAS  = class-balanced kernel target alignment between SAE activations and
         labels.
  TSF  = reconstruction fidelity restricted to the between-class task subspace.
  FP   = activation-frequency-weighted entropy purity of individual features.
  TFD  = chance-normalised top-k one-vs-rest feature AUC.
  Hoyer = activation sparsity sanity check on the pooled SAE code.

The legacy Model-Conditioned SAE score is a weighted geometric mean:

  MCS = (R2_+^w_r2 * SUS^w_sus * DAS^w_das)^(1 / (w_r2+w_sus+w_das))

The final DAMS score is:

  DAMS_final = (TSF * FP * TFD)^(1/3)

DAMS_final is the main selection metric because TSF measures preservation of
the task-discriminative geometry while FP/TFD measure whether that preserved
geometry is represented by clean, class-specific concept features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.dams import (  # noqa: E402
    compute_activation_hoyer_sparsity,
    compute_domain_alignment_score,
    compute_cv_topk_feature_discriminability,
    compute_effective_coverage,
    compute_feature_purity_score,
    compute_sae_utility_score,
    compute_task_subspace_fidelity,
)
from src.sae_training.loaders import load_sae  # noqa: E402
from sweep_dams_hyperparams import (  # noqa: E402
    ActivationCapture,
    DEFAULT_BASE_SAE,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_LORA_ROOT,
    RunSpec,
    build_loader_for_run,
    discover_runs,
)


def _layer_index(block_layer: int, n_layers: int) -> int:
    return block_layer if block_layer >= 0 else n_layers + block_layer


def _safe01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def model_conditioned_score(
    r2: float,
    sus: float,
    das: float,
    w_r2: float = 1.0,
    w_sus: float = 2.0,
    w_das: float = 1.0,
    eps: float = 1e-6,
) -> float:
    """Weighted geometric mean of target-model compatibility components."""
    weights = [w_r2, w_sus, w_das]
    values = [_safe01(r2), _safe01(sus), _safe01(das)]
    total_w = sum(w for w in weights if w > 0)
    if total_w <= 0:
        raise ValueError("At least one MCS weight must be positive.")

    log_sum = 0.0
    for value, weight in zip(values, weights):
        if weight <= 0:
            continue
        log_sum += weight * math.log(max(eps, value))
    return float(math.exp(log_sum / total_w))


def dams_final_score(
    tsf: float,
    fp: float,
    tfd: float,
    eps: float = 1e-6,
) -> float:
    """DAMS final: geometric mean of task-subspace fidelity and feature quality."""
    values = [_safe01(tsf), _safe01(fp), _safe01(tfd)]
    return float(math.exp(sum(math.log(max(eps, v)) for v in values) / len(values)))


@torch.no_grad()
def compute_pooled_activations_fast(
    sae,
    features: torch.Tensor,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Max-pool SAE activations over patch tokens with batched encoder calls."""
    if features.ndim != 3:
        pooled = []
        for i in tqdm(range(0, features.shape[0], batch_size), desc="SAE encode", leave=False):
            chunk = features[i : i + batch_size].to(device)
            z = sae.encode(chunk) if hasattr(sae, "encode") else sae(chunk)
            if isinstance(z, (tuple, list)):
                z = z[1]
            pooled.append(z.detach().cpu())
        return torch.cat(pooled, dim=0)

    n_images, seq_len, d_model = features.shape
    patches_per_image = seq_len - 1
    flat = features[:, 1:, :].contiguous().view(n_images * patches_per_image, d_model)
    image_ids = torch.arange(n_images, device=device).repeat_interleave(patches_per_image)
    pooled = None

    for start in tqdm(range(0, flat.shape[0], batch_size), desc="SAE encode", leave=False):
        end = min(start + batch_size, flat.shape[0])
        chunk = flat[start:end].to(device)
        z = sae.encode(chunk) if hasattr(sae, "encode") else sae(chunk)
        if isinstance(z, (tuple, list)):
            z = z[1]

        if pooled is None:
            pooled = torch.full(
                (n_images, z.shape[-1]),
                -torch.inf,
                dtype=z.dtype,
                device=device,
            )

        ids = image_ids[start:end]
        if hasattr(pooled, "index_reduce_"):
            pooled.index_reduce_(0, ids, z, reduce="amax", include_self=True)
        else:
            # Conservative fallback for older PyTorch builds.
            for img_id in ids.unique():
                mask = ids == img_id
                pooled[img_id] = torch.maximum(pooled[img_id], z[mask].max(dim=0).values)

        del chunk, z

    if pooled is None:
        raise RuntimeError("No SAE activations were produced.")
    pooled = torch.nan_to_num(pooled, neginf=0.0, posinf=0.0)
    return pooled.cpu()


@torch.no_grad()
def extract_features_for_target(
    run: RunSpec,
    layer: int,
    args: argparse.Namespace,
    clip_cache: Dict[Optional[str], Tuple[torch.nn.Module, object]],
    subset_index_cache: Dict[str, List[int]],
) -> Tuple[torch.Tensor, List[int], int]:
    """Extract target-model hidden states for one dataset/model/layer."""
    model, dataset, loader = build_loader_for_run(
        run=run,
        device=args.device,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_samples=args.max_samples,
        subset_seed=args.subset_seed,
        subset_index_cache=subset_index_cache,
        clip_cache=clip_cache,
    )

    n_layers = len(model.visual.transformer.resblocks)
    block = model.visual.transformer.resblocks[_layer_index(layer, n_layers)]

    cap = ActivationCapture()
    cap.register(block)

    feat_batches = []
    labels: List[int] = []
    for images, labs in tqdm(loader, desc=f"extract {run.dataset} layer={layer}", leave=False):
        model.encode_image(images.to(args.device))
        feat_batches.append(cap.act.cpu())
        labels.extend(labs.tolist())

    cap.remove()
    return torch.cat(feat_batches, dim=0), labels, dataset.num_classes


def evaluate_sae_on_features(
    sae_path: Path,
    features: torch.Tensor,
    labels: List[int],
    num_classes: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    sae, cfg = load_sae(str(sae_path), args.device)
    sae.eval().to(args.device)

    r2, mse_per_dim, var_per_dim = compute_effective_coverage(
        sae,
        features,
        device=args.device,
        batch_size=args.sae_batch_size,
    )
    acts = compute_pooled_activations_fast(
        sae,
        features,
        device=args.device,
        batch_size=args.sae_batch_size,
    )
    sus, balanced_acc, chance = compute_sae_utility_score(
        acts,
        labels,
        num_classes,
        n_splits=args.utility_splits,
        ridge=args.utility_ridge,
        top_features=args.utility_top_features,
    )
    das = compute_domain_alignment_score(
        acts,
        labels,
        num_classes,
        subsample=args.das_subsample,
    )
    tsf, tsf_stats = compute_task_subspace_fidelity(
        sae,
        features,
        labels,
        num_classes,
        device=args.device,
        batch_size=args.sae_batch_size,
        token_mode=args.tsf_token_mode,
        max_components=args.tsf_max_components,
    )
    fp, fp_stats = compute_feature_purity_score(
        acts,
        labels,
        num_classes,
        active_threshold=args.active_threshold,
        min_fire_count=args.min_fire_count,
        min_fire_frac=args.min_fire_frac,
        chunk_size=args.fp_chunk_size,
    )
    tfd, tfd_auc, tfd_stats = compute_cv_topk_feature_discriminability(
        acts,
        labels,
        num_classes,
        top_k=args.tfd_top_k,
        n_splits=args.tfd_splits,
        active_threshold=args.active_threshold,
        min_fire_count=args.min_fire_count,
        min_fire_frac=args.min_fire_frac,
        min_pos=args.tfd_min_pos,
        chunk_size=args.tfd_chunk_size,
    )
    hoyer, hoyer_stats = compute_activation_hoyer_sparsity(acts)
    mcs = model_conditioned_score(
        r2=r2,
        sus=sus,
        das=das,
        w_r2=args.w_r2,
        w_sus=args.w_sus,
        w_das=args.w_das,
    )
    dams_final = dams_final_score(tsf=tsf, fp=fp, tfd=tfd)

    out = {
        "layer": int(cfg.block_layer),
        "r2": float(r2),
        "r2_clipped": _safe01(r2),
        "mse_per_dim": float(mse_per_dim),
        "var_per_dim": float(var_per_dim),
        "sus": float(sus),
        "balanced_acc": float(balanced_acc),
        "chance": float(chance),
        "das": float(das),
        "mcs": float(mcs),
        "tsf": float(tsf),
        "fp": float(fp),
        "tfd": float(tfd),
        "tfd_auc": float(tfd_auc),
        "hoyer_sparsity": float(hoyer),
        "dams_final": float(dams_final),
    }
    out.update(tsf_stats)
    out.update(fp_stats)
    out.update(tfd_stats)
    out.update(hoyer_stats)

    del sae, acts
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired target-model SAE compatibility evaluation")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--base-sae", type=Path, default=DEFAULT_BASE_SAE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--lora-root", type=Path, default=DEFAULT_LORA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "out" / "paired_sae_compatibility")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--allow-layer-mismatch", action="store_true",
                        help="Evaluate non-layer-matched adapted SAEs on their own layer. Default keeps only base-layer matches.")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--sae-batch-size", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--subset-seed", type=int, default=0)

    parser.add_argument("--das-subsample", type=int, default=2000)
    parser.add_argument("--utility-top-features", type=int, default=4096)
    parser.add_argument("--utility-splits", type=int, default=3)
    parser.add_argument("--utility-ridge", type=float, default=1.0)
    parser.add_argument("--tsf-token-mode", choices=["cls", "mean_patch", "mean_all"], default="cls",
                        help="Image-level feature used for TSF. CLS matches CLIP image-readout geometry.")
    parser.add_argument("--tsf-max-components", type=int, default=0,
                        help="Cap TSF discriminant rank; 0 keeps the empirical between-class rank.")
    parser.add_argument("--active-threshold", type=float, default=0.0)
    parser.add_argument("--min-fire-count", type=int, default=5)
    parser.add_argument("--min-fire-frac", type=float, default=0.005)
    parser.add_argument("--fp-chunk-size", type=int, default=4096)
    parser.add_argument("--tfd-top-k", type=int, default=200)
    parser.add_argument("--tfd-splits", type=int, default=3)
    parser.add_argument("--tfd-min-pos", type=int, default=2)
    parser.add_argument("--tfd-chunk-size", type=int, default=2048)
    parser.add_argument("--w-r2", type=float, default=1.0)
    parser.add_argument("--w-sus", type=float, default=2.0)
    parser.add_argument("--w-das", type=float, default=1.0)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs, skipped = discover_runs(
        checkpoint_root=args.checkpoint_root,
        base_sae_path=args.base_sae,
        data_root=args.data_root,
        lora_root=args.lora_root,
        dataset_filter=args.datasets,
    )
    base_sae, base_cfg = load_sae(str(args.base_sae), args.device)
    base_layer = int(base_cfg.block_layer)
    del base_sae
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    adapted = [r for r in runs if r.kind != "Base SAE"]
    if not args.allow_layer_mismatch:
        filtered = []
        for r in adapted:
            _, cfg = load_sae(str(r.sae_path), args.device)
            layer = int(cfg.block_layer)
            if layer == base_layer:
                filtered.append(r)
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        adapted = filtered

    if not adapted:
        raise RuntimeError("No adapted SAEs left after filtering.")

    print("=" * 90)
    print("Paired target-model SAE compatibility")
    print("=" * 90)
    print(f"Base SAE layer: {base_layer}")
    print(f"Adapted candidates: {len(adapted)}")
    print(f"Layer mismatch allowed: {args.allow_layer_mismatch}")

    clip_cache: Dict[Optional[str], Tuple[torch.nn.Module, object]] = {}
    subset_index_cache: Dict[str, List[int]] = {}
    feature_cache: Dict[Tuple[str, Optional[str], int], Tuple[torch.Tensor, List[int], int]] = {}
    base_metric_cache: Dict[Tuple[str, Optional[str], int], Dict[str, float]] = {}

    rows = []
    for idx, run in enumerate(adapted, start=1):
        _, tmp_cfg = load_sae(str(run.sae_path), args.device)
        target_layer = int(tmp_cfg.block_layer)
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        if not args.allow_layer_mismatch and target_layer != base_layer:
            continue

        cache_key = (run.dataset, str(run.lora_path) if run.lora_path else None, target_layer)
        if cache_key not in feature_cache:
            print("\n" + "-" * 90)
            print(f"Extracting target activations | dataset={run.dataset} layer={target_layer} lora={run.lora_path}")
            feature_cache[cache_key] = extract_features_for_target(
                run=run,
                layer=target_layer,
                args=args,
                clip_cache=clip_cache,
                subset_index_cache=subset_index_cache,
            )

        features, labels, num_classes = feature_cache[cache_key]

        if cache_key not in base_metric_cache:
            print(f"Evaluating base SAE on adapted target model | dataset={run.dataset} layer={target_layer}")
            base_metric_cache[cache_key] = evaluate_sae_on_features(
                sae_path=args.base_sae,
                features=features,
                labels=labels,
                num_classes=num_classes,
                args=args,
            )

        print(f"Evaluating adapted SAE [{idx}/{len(adapted)}] | {run.name}")
        adapted_metrics = evaluate_sae_on_features(
            sae_path=run.sae_path,
            features=features,
            labels=labels,
            num_classes=num_classes,
            args=args,
        )
        base_metrics = base_metric_cache[cache_key]

        row = {
            "dataset": run.dataset,
            "adapted_name": run.name,
            "kind": run.kind,
            "source_group": run.source_group,
            "run_id": run.run_id,
            "target_layer": target_layer,
            "n_samples": len(labels),
            "num_classes": num_classes,
            "base_dams_final": base_metrics["dams_final"],
            "adapted_dams_final": adapted_metrics["dams_final"],
            "dams_final_gain": adapted_metrics["dams_final"] - base_metrics["dams_final"],
            "base_tsf": base_metrics["tsf"],
            "adapted_tsf": adapted_metrics["tsf"],
            "tsf_gain": adapted_metrics["tsf"] - base_metrics["tsf"],
            "base_tsf_rank": base_metrics["tsf_rank"],
            "adapted_tsf_rank": adapted_metrics["tsf_rank"],
            "base_fp": base_metrics["fp"],
            "adapted_fp": adapted_metrics["fp"],
            "fp_gain": adapted_metrics["fp"] - base_metrics["fp"],
            "base_tfd": base_metrics["tfd"],
            "adapted_tfd": adapted_metrics["tfd"],
            "tfd_gain": adapted_metrics["tfd"] - base_metrics["tfd"],
            "base_tfd_auc": base_metrics["tfd_auc"],
            "adapted_tfd_auc": adapted_metrics["tfd_auc"],
            "tfd_auc_gain": adapted_metrics["tfd_auc"] - base_metrics["tfd_auc"],
            "base_tfd_auc_ge_0_8": base_metrics["tfd_features_auc_gt_0_8"],
            "adapted_tfd_auc_ge_0_8": adapted_metrics["tfd_features_auc_gt_0_8"],
            "tfd_auc_ge_0_8_gain": adapted_metrics["tfd_features_auc_gt_0_8"] - base_metrics["tfd_features_auc_gt_0_8"],
            "base_tfd_auc_ge_0_9": base_metrics["tfd_features_auc_gt_0_9"],
            "adapted_tfd_auc_ge_0_9": adapted_metrics["tfd_features_auc_gt_0_9"],
            "tfd_auc_ge_0_9_gain": adapted_metrics["tfd_features_auc_gt_0_9"] - base_metrics["tfd_features_auc_gt_0_9"],
            "base_fp_supported_features": base_metrics["fp_supported_features"],
            "adapted_fp_supported_features": adapted_metrics["fp_supported_features"],
            "base_hoyer_sparsity": base_metrics["hoyer_sparsity"],
            "adapted_hoyer_sparsity": adapted_metrics["hoyer_sparsity"],
            "hoyer_sparsity_gain": adapted_metrics["hoyer_sparsity"] - base_metrics["hoyer_sparsity"],
            "base_active_features_per_sample": base_metrics["active_features_per_sample"],
            "adapted_active_features_per_sample": adapted_metrics["active_features_per_sample"],
            "active_features_per_sample_delta": (
                adapted_metrics["active_features_per_sample"] - base_metrics["active_features_per_sample"]
            ),
            "base_mcs": base_metrics["mcs"],
            "adapted_mcs": adapted_metrics["mcs"],
            "mcs_gain": adapted_metrics["mcs"] - base_metrics["mcs"],
            "base_r2": base_metrics["r2"],
            "adapted_r2": adapted_metrics["r2"],
            "r2_gain": adapted_metrics["r2"] - base_metrics["r2"],
            "base_sus": base_metrics["sus"],
            "adapted_sus": adapted_metrics["sus"],
            "sus_gain": adapted_metrics["sus"] - base_metrics["sus"],
            "base_das": base_metrics["das"],
            "adapted_das": adapted_metrics["das"],
            "das_gain": adapted_metrics["das"] - base_metrics["das"],
            "base_balanced_acc": base_metrics["balanced_acc"],
            "adapted_balanced_acc": adapted_metrics["balanced_acc"],
            "base_chance": base_metrics["chance"],
            "adapted_chance": adapted_metrics["chance"],
            "sae_path": str(run.sae_path),
            "lora_path": str(run.lora_path) if run.lora_path else None,
        }
        rows.append(row)
        print(
            f"DAMS_final base={row['base_dams_final']:.4f} adapted={row['adapted_dams_final']:.4f} "
            f"gain={row['dams_final_gain']:.4f} | TSF gain={row['tsf_gain']:.4f} "
            f"FP gain={row['fp_gain']:.4f} TFD gain={row['tfd_gain']:.4f} "
            f"Hoyer gain={row['hoyer_sparsity_gain']:.4f} "
            f"(AUC>=.8 gain={row['tfd_auc_ge_0_8_gain']})"
        )

    if not rows:
        raise RuntimeError("No rows evaluated.")

    full_csv = args.output_dir / "paired_sae_compatibility_full.csv"
    with full_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_rows = []
    for dataset in sorted({r["dataset"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        best = max(ds_rows, key=lambda r: r["dams_final_gain"])
        best_rows.append(best)

    summary_csv = args.output_dir / "paired_sae_compatibility_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_rows)

    manifest = {
        "base_sae": str(args.base_sae),
        "base_layer": base_layer,
        "layer_mismatch_allowed": args.allow_layer_mismatch,
        "max_samples": args.max_samples,
        "subset_seed": args.subset_seed,
        "mcs_weights": {"r2": args.w_r2, "sus": args.w_sus, "das": args.w_das},
        "dams_final": {
            "formula": "(TSF * FP * TFD_norm)^(1/3)",
            "tsf_token_mode": args.tsf_token_mode,
            "tsf_max_components": args.tsf_max_components,
            "tfd_top_k": args.tfd_top_k,
            "active_threshold": args.active_threshold,
            "min_fire_count": args.min_fire_count,
            "min_fire_frac": args.min_fire_frac,
            "tfd_min_pos": args.tfd_min_pos,
        },
        "skipped": skipped,
    }
    manifest_json = args.output_dir / "paired_sae_compatibility_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2))

    print("\nBest paired adapted SAE per dataset by DAMS_final")
    print(
        f"{'Dataset':<12} {'Base':>10} {'Adapted':>10} {'Gain':>10} "
        f"{'TSF gain':>10} {'FP gain':>10} {'TFD gain':>10} {'Hoyer':>10} {'AUC>=.8':>9}  Winner"
    )
    print("-" * 136)
    for row in best_rows:
        print(
            f"{row['dataset']:<12} {row['base_dams_final']:>10.4f} {row['adapted_dams_final']:>10.4f} "
            f"{row['dams_final_gain']:>10.4f} {row['tsf_gain']:>10.4f} {row['fp_gain']:>10.4f} "
            f"{row['tfd_gain']:>10.4f} {row['hoyer_sparsity_gain']:>10.4f} "
            f"{row['tfd_auc_ge_0_8_gain']:>9}  {row['adapted_name']}"
        )

    print("\nSaved:")
    print(f"- {full_csv}")
    print(f"- {summary_csv}")
    print(f"- {manifest_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified {model, sae, dataset} eval matrix driver.

Generalizes eval_medmnist_sae.py's HF-CLIP zero-shot + SAE-reconstruction
pipeline to an arbitrary set of datasets, sweeping
{dataset} x {vit_type in base,lora} x
{sae_condition in none,gsae,ftsae,scratchsae,fullftsae}
and writing one tidy row per valid cell to results.csv.

Usage:
    python tasks/eval_matrix.py --dry_run --datasets eurosat
    python tasks/eval_matrix.py --datasets pets eurosat pathmnist \
        --lora_checkpoints configs/rebuttal_lora_checkpoints.json \
        --sae_paths configs/rebuttal_sae_paths.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sae_classification1 import SAEHookHandler
from mdmnist_sae_eval import calculate_text_features
from eval_medmnist_sae import _hf_collate_fn
from tasks.rebuttal_common import (
    VIT_TYPES, SAE_CONDITIONS, EXPECTED_BLOCK_LAYER, EXPECTED_D_IN,
    DEFAULT_DATASET_REGISTRY, DEFAULT_GSAE_PATH,
    flush, clear_model_caches, load_registry, build_dataset_splits,
    get_vit, resolve_sae_path, get_sae, validate_sae_condition_provenance,
    is_valid_cell, cell_cache_path,
)

RESULT_COLUMNS = [
    "dataset", "vit_type", "sae_condition", "sae_path",
    "training_seed", "evaluation_seed",
    "zeroshot_acc", "l0", "dead_frac", "recon_cosine", "fve",
    "label_entropy_mean", "n_images", "skipped",
]


def result_training_seed(sae_condition, args):
    """Only learned target-domain SAE arms carry an SAE-training seed."""
    if sae_condition in {"none", "gsae", "gsae_gated"}:
        return None
    return args.training_seed


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+",
                    default=["pets", "eurosat", "pathmnist", "imagenet_subset"])
    p.add_argument("--vit_types", nargs="+", default=VIT_TYPES, choices=VIT_TYPES)
    p.add_argument("--sae_conditions", nargs="+", default=SAE_CONDITIONS,
                    choices=SAE_CONDITIONS)
    p.add_argument("--lora_checkpoints", type=str, default=None,
                    help="JSON mapping {dataset: lora_checkpoint_path}. "
                         "Overrides the registry's per-dataset default.")
    p.add_argument("--sae_paths", type=str, default=None,
                    help="JSON mapping {dataset: {condition: sae_checkpoint_path}}.")
    p.add_argument("--dataset_registry", type=str, default=DEFAULT_DATASET_REGISTRY)
    p.add_argument("--imagenet_subset_dir", type=str, default=None,
                    help="ImageFolder root for the imagenet_subset dataset.")
    p.add_argument("--gsae_path", type=str, default=DEFAULT_GSAE_PATH)
    p.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch16")
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--training_seed",
        type=int,
        default=None,
        help="SAE-training seed represented by --sae_paths. This is recorded "
             "separately from --seed, which fixes evaluation examples.",
    )
    p.add_argument("--batch_size", type=int, default=8,
                    help="Image batch size. Keep small for 49,152-wide SAEs.")
    p.add_argument("--max_images", type=int, default=10000,
                    help="Cap on total images per dataset before the train/eval split.")
    p.add_argument("--val_split", type=float, default=0.15,
                    help="Fraction of images held out as the eval split "
                         "(the rest is the train split used for label-entropy profiles).")
    p.add_argument("--l0_threshold", type=float, default=1e-5)
    p.add_argument("--label_entropy_topn", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="out/rebuttal/matrix")
    p.add_argument("--cache_dir", type=str, default="out/rebuttal/cache")
    p.add_argument("--reuse_cache", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════

def capture_activations(vit, cfg, images, args):
    inputs = vit.processor(images=images, text="", return_tensors="pt", padding=True).to(args.device)
    _, cache = vit.run_with_cache([(cfg.block_layer, cfg.module_name)], **inputs)
    return cache[(cfg.block_layer, cfg.module_name)]


@torch.no_grad()
def compute_sae_quality(vit, sae, cfg, eval_ds, args, max_batches=None):
    """One pass over eval_ds with NO reconstruction hook registered: captures the
    original activations and runs them through the SAE directly to measure
    faithfulness, without any live forward-hook interference."""
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=_hf_collate_fn, num_workers=2)

    d_sae = sae.d_sae
    ever_active = torch.zeros(d_sae, dtype=torch.bool, device=args.device)
    b_dec = sae.b_dec.detach().float() if hasattr(sae, "b_dec") else None

    l0_sum, l0_tokens = 0.0, 0
    cos_sum, cos_n = 0.0, 0
    err_sum, var_sum = 0.0, 0.0

    for bidx, (images, _labels) in enumerate(tqdm(loader, desc="sae quality", leave=False)):
        if max_batches is not None and bidx >= max_batches:
            break
        acts = capture_activations(vit, cfg, images, args).float()
        sae_out, feat_acts, _ = sae(acts)

        active = feat_acts > args.l0_threshold
        l0_sum += active.float().sum().item()
        l0_tokens += active.shape[0] * active.shape[1]
        ever_active |= active.any(dim=0).any(dim=0)

        cos = F.cosine_similarity(
            acts.reshape(-1, acts.shape[-1]), sae_out.reshape(-1, sae_out.shape[-1]), dim=-1
        )
        cos_sum += cos.sum().item()
        cos_n += cos.numel()

        err_sum += (acts - sae_out).pow(2).sum(dim=-1).sum().item()
        center = b_dec if b_dec is not None else acts.mean(dim=(0, 1), keepdim=True)
        var_sum += (acts - center).pow(2).sum(dim=-1).sum().item()

    if l0_tokens == 0:
        return dict(l0=float("nan"), dead_frac=float("nan"),
                    recon_cosine=float("nan"), fve=float("nan"))

    return dict(
        l0=l0_sum / l0_tokens,
        dead_frac=1.0 - ever_active.float().mean().item(),
        recon_cosine=cos_sum / max(cos_n, 1),
        fve=1.0 - err_sum / max(var_sum, 1e-12),
    )


@torch.no_grad()
def compute_label_entropy(vit, sae, cfg, train_ds, classnames, args, max_batches=None):
    """cls_sae_cnt-style per-class CLS-token activation profile on the train split,
    normalized per feature into a class distribution, then averaged entropy over
    the top-N most-active features."""
    num_classes = len(classnames)
    d_sae = sae.d_sae
    cls_sum = np.zeros((num_classes, d_sae), dtype=np.float64)
    cls_count = np.zeros(num_classes, dtype=np.int64)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=_hf_collate_fn, num_workers=2)

    for bidx, (images, labels) in enumerate(tqdm(loader, desc="label entropy", leave=False)):
        if max_batches is not None and bidx >= max_batches:
            break
        acts = capture_activations(vit, cfg, images, args).float()
        cls_acts = acts[:, 0, :]
        _, feat_acts, _ = sae(cls_acts)
        feat_np = feat_acts.cpu().numpy()
        for i, label in enumerate(labels.tolist()):
            cls_sum[label] += feat_np[i]
            cls_count[label] += 1

    if cls_count.sum() == 0:
        return float("nan")

    cls_mean = cls_sum / np.clip(cls_count, 1, None)[:, None]
    totals = cls_mean.sum(axis=0) + 1e-12
    p = cls_mean / totals
    entropy = -(p * np.log(p + 1e-12)).sum(axis=0)
    activity = cls_mean.mean(axis=0)

    topn = min(args.label_entropy_topn, d_sae)
    top_idx = np.argsort(-activity)[:topn]
    return float(entropy[top_idx].mean())


@torch.no_grad()
def compute_zeroshot_acc(vit, sae, cfg, classnames, eval_ds, args, max_batches=None):
    text_features = calculate_text_features(vit, args.device, classnames)
    tf = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = vit.model.logit_scale.exp()

    hook = None
    if sae is not None:
        hook = SAEHookHandler(vit.model, sae, sae_layer=cfg.block_layer, sae_mode="original")
        hook.register()

    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=_hf_collate_fn, num_workers=2)
    correct, total = 0, 0
    try:
        for bidx, (images, labels) in enumerate(tqdm(loader, desc="zeroshot acc", leave=False)):
            if max_batches is not None and bidx >= max_batches:
                break
            inputs = vit.processor(images=images, text="", return_tensors="pt", padding=True).to(args.device)
            out = vit(return_type="output", **inputs)
            img_feat = out.image_embeds
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = logit_scale * (img_feat @ tf.t())
            preds = logits.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    finally:
        if hook is not None:
            hook.remove()

    if total == 0:
        return float("nan"), 0
    return correct / total * 100.0, total


# ═════════════════════════════════════════════════════════════════════════
# Per-cell evaluation with caching
# ═════════════════════════════════════════════════════════════════════════

def evaluate_cell(dataset, vit_type, sae_condition, splits, lora_checkpoints, sae_paths,
                   args, max_batches=None):
    row = dict(dataset=dataset, vit_type=vit_type, sae_condition=sae_condition,
               sae_path=None, training_seed=result_training_seed(sae_condition, args),
               evaluation_seed=args.seed,
               zeroshot_acc=float("nan"), l0=float("nan"),
               dead_frac=float("nan"), recon_cosine=float("nan"), fve=float("nan"),
               label_entropy_mean=float("nan"), n_images=0, skipped=True)

    if not is_valid_cell(vit_type, sae_condition):
        return row, "invalid combo (sae requires lora vit_type)"

    cache_condition = (
        f"{sae_condition}__trainseed{result_training_seed(sae_condition, args)}"
        f"__evalseed{args.seed}"
    )
    cpath = cell_cache_path(args.cache_dir, dataset, vit_type, cache_condition)
    if args.reuse_cache and max_batches is None and os.path.exists(cpath):
        with open(cpath) as f:
            return json.load(f), None

    vit = get_vit(vit_type, dataset, lora_checkpoints, args)
    if vit is None:
        return row, "missing lora checkpoint"

    sae_path = resolve_sae_path(sae_condition, dataset, sae_paths, args)
    if sae_condition != "none" and (not sae_path or not os.path.exists(sae_path)):
        return row, f"missing sae checkpoint for condition={sae_condition}"

    classnames, train_ds, eval_ds = splits
    sae, cfg = (None, None)
    if sae_path is not None:
        sae, cfg = get_sae(sae_path, args.device)
        try:
            validate_sae_condition_provenance(sae_condition, cfg)
        except ValueError as exc:
            return row, str(exc)

    acc, n_images = compute_zeroshot_acc(vit, sae, cfg, classnames, eval_ds, args, max_batches)
    row.update(zeroshot_acc=acc, n_images=n_images, sae_path=sae_path, skipped=False)

    if sae is not None:
        quality = compute_sae_quality(vit, sae, cfg, eval_ds, args, max_batches)
        row.update(quality)
        row["label_entropy_mean"] = compute_label_entropy(
            vit, sae, cfg, train_ds, classnames, args, max_batches)

    if args.reuse_cache and max_batches is None:
        os.makedirs(args.cache_dir, exist_ok=True)
        with open(cpath, "w") as f:
            json.dump(row, f, indent=2)

    return row, None


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.reuse_cache:
        os.makedirs(args.cache_dir, exist_ok=True)

    registry = load_registry(args.dataset_registry)
    for name in args.datasets:
        if name not in registry:
            print(f"[FATAL] Dataset '{name}' not found in registry {args.dataset_registry}")
            sys.exit(1)

    lora_checkpoints = {name: registry[name].get("lora_checkpoint") for name in args.datasets}
    if args.lora_checkpoints:
        with open(args.lora_checkpoints) as f:
            lora_checkpoints.update(json.load(f))

    sae_paths = {}
    if args.sae_paths:
        with open(args.sae_paths) as f:
            sae_paths = json.load(f)

    print("=" * 78)
    print(f"EVAL MATRIX{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"datasets={args.datasets}  vit_types={args.vit_types}  "
          f"sae_conditions={args.sae_conditions}")
    print("=" * 78)

    splits_by_dataset = {}
    load_errors = {}
    for name in args.datasets:
        try:
            splits_by_dataset[name] = build_dataset_splits(name, registry[name], args)
            print(f"[OK] dataset '{name}': "
                  f"{len(splits_by_dataset[name][1])} train / "
                  f"{len(splits_by_dataset[name][2])} eval images, "
                  f"{len(splits_by_dataset[name][0])} classes")
        except Exception as e:
            load_errors[name] = str(e)
            print(f"[SKIP] dataset '{name}': {e}")

    rows = []
    skip_reasons = {}
    max_batches = 1 if args.dry_run else None

    for vit_type in args.vit_types:
        for dataset in args.datasets:
            if dataset not in splits_by_dataset:
                for sae_condition in args.sae_conditions:
                    row = dict(dataset=dataset, vit_type=vit_type, sae_condition=sae_condition,
                               sae_path=None, training_seed=result_training_seed(sae_condition, args),
                               evaluation_seed=args.seed,
                               zeroshot_acc=float("nan"), l0=float("nan"),
                               dead_frac=float("nan"), recon_cosine=float("nan"), fve=float("nan"),
                               label_entropy_mean=float("nan"), n_images=0, skipped=True)
                    rows.append(row)
                    skip_reasons[(dataset, vit_type, sae_condition)] = load_errors[dataset]
                continue

            for sae_condition in args.sae_conditions:
                t0 = time.time()
                try:
                    row, reason = evaluate_cell(
                        dataset, vit_type, sae_condition, splits_by_dataset[dataset],
                        lora_checkpoints, sae_paths, args, max_batches=max_batches)
                except Exception as e:
                    row = dict(dataset=dataset, vit_type=vit_type, sae_condition=sae_condition,
                               sae_path=None, training_seed=result_training_seed(sae_condition, args),
                               evaluation_seed=args.seed,
                               zeroshot_acc=float("nan"), l0=float("nan"),
                               dead_frac=float("nan"), recon_cosine=float("nan"), fve=float("nan"),
                               label_entropy_mean=float("nan"), n_images=0, skipped=True)
                    reason = f"exception: {e}"
                finally:
                    # get_vit/get_sae cache CUDA objects for reuse, but a full
                    # matrix has one LoRA model and SAE per dataset. Retaining
                    # them across cells accumulated ~20 GiB by UCF101.
                    clear_model_caches()
                rows.append(row)
                if reason:
                    skip_reasons[(dataset, vit_type, sae_condition)] = reason
                status = "SKIP" if row["skipped"] else "OK"
                print(f"[{status}] {dataset:16s} {vit_type:5s} {sae_condition:10s}  "
                      f"acc={row['zeroshot_acc']:.2f}  ({time.time() - t0:.1f}s)"
                      + (f"  reason={reason}" if reason else ""))
                flush()

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    prefix = "dry_run_" if args.dry_run else ""
    csv_path = os.path.join(args.out_dir, f"{prefix}results.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": args.dry_run,
        "args": {k: v for k, v in vars(args).items()},
        "n_cells": len(rows),
        "n_ok": int((~df["skipped"]).sum()),
        "n_skipped": int(df["skipped"].sum()),
        "skip_reasons": {"|".join(k): v for k, v in skip_reasons.items()},
        "dataset_load_errors": load_errors,
    }
    summary_path = os.path.join(args.out_dir, f"{prefix}summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 78)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"{summary['n_ok']} OK / {summary['n_skipped']} skipped / {summary['n_cells']} total")
    if args.dry_run:
        print("\nPlanned matrix:")
        print(df[["dataset", "vit_type", "sae_condition", "skipped"]].to_string(index=False))


if __name__ == "__main__":
    main()

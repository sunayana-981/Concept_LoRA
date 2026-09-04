#!/usr/bin/env python3
"""
Dataset-agnostic top-k SAE feature steering (clamp ON/OFF) eval.

Generalizes mdmnist_sae_eval.py's per-class top-k masking protocol
(compute_cls_sae_activations / create_sae_hooks / evaluate_topk_masking) to
an arbitrary dataset from the tasks/rebuttal_common.py registry, sweeping
{dataset} x {vit_type=lora} x {sae_condition in gsae,ftsae} x {k}.

For each cell:
  1. Compute per-class SAE activation profiles (cls_sae_cnt) on the train split.
  2. For each class and each k in --k_values: clamp the class's top-k SAE
     features ON (rest untouched... clamp_feat_dim covers all features, so ON
     means "only these k are forced active"), then OFF (these k forced to 0),
     and measure per-class zero-shot accuracy on the eval split, vs. a
     no-SAE baseline.

If FT-SAE features are more causally class-aligned than G-SAE features,
FT-SAE's OFF-drop and ON-gain should exceed G-SAE's.

Usage:
    python tasks/eval_steering.py --dry_run --datasets eurosat
    python tasks/eval_steering.py --datasets pathmnist eurosat \
        --sae_paths configs/rebuttal_sae_paths.json \
        --lora_checkpoints configs/rebuttal_lora_checkpoints.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.sae_training.hooked_vit import Hook
from mdmnist_sae_eval import calculate_text_features, SAE_BIAS
from eval_medmnist_sae import _hf_collate_fn
from tasks.rebuttal_common import (
    add_common_args, flush, load_common_registry_args, build_dataset_splits,
    get_vit, resolve_sae_path, get_sae, cell_cache_path,
)

STEERING_SAE_CONDITIONS = ["gsae", "ftsae"]
DEFAULT_K_VALUES = [1, 5, 10, 50]

RESULT_COLUMNS = ["dataset", "vit_type", "sae_condition", "class", "k", "mode", "acc", "n_images"]


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["pets", "eurosat", "pathmnist"])
    p.add_argument("--vit_types", nargs="+", default=["lora"], choices=["lora"],
                    help="Steering is only meaningful on the fine-tuned model "
                         "(a no-SAE baseline is computed on this same vit).")
    p.add_argument("--sae_conditions", nargs="+", default=STEERING_SAE_CONDITIONS,
                    choices=STEERING_SAE_CONDITIONS)
    p.add_argument("--k_values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    p = add_common_args(p, out_dir_default="out/rebuttal/steering")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════
# Per-class SAE activation profile (cls_sae_cnt), cached compatibly with
# Task 1's out/rebuttal/cache/ directory.
# ═════════════════════════════════════════════════════════════════════════

def profile_cache_path(cache_dir, dataset, vit_type, sae_condition):
    return cell_cache_path(cache_dir, dataset, vit_type, f"{sae_condition}_profile", suffix="npy")


@torch.no_grad()
def compute_class_profile(vit, sae, cfg, train_ds, num_classes, args, max_batches=None):
    """Mean per-class CLS-token SAE feature activation. Returns np.ndarray
    [num_classes, d_sae]. Same computation as
    mdmnist_sae_eval.compute_cls_sae_activations, adapted to datasets that
    already yield PIL images (no to_pil round-trip needed)."""
    d_sae = sae.d_sae
    layer, module = cfg.block_layer, cfg.module_name

    cls_sum = np.zeros((num_classes, d_sae), dtype=np.float64)
    cls_count = np.zeros(num_classes, dtype=np.int64)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=_hf_collate_fn, num_workers=2)

    for bidx, (images, labels) in enumerate(tqdm(loader, desc="cls_sae_cnt", leave=False)):
        if max_batches is not None and bidx >= max_batches:
            break
        inputs = vit.processor(images=images, text="", return_tensors="pt", padding=True).to(args.device)
        _, cache = vit.run_with_cache([(layer, module)], **inputs)
        cls_acts = cache[(layer, module)][:, 0, :].float()

        _, sae_cache = sae.run_with_cache(cls_acts)
        feat_np = sae_cache["hook_hidden_post"].cpu().float().numpy()

        for i, label in enumerate(labels.tolist()):
            cls_sum[label] += feat_np[i]
            cls_count[label] += 1

    for c in range(num_classes):
        if cls_count[c] > 0:
            cls_sum[c] /= cls_count[c]
    return cls_sum


def get_class_profile(vit, sae, cfg, train_ds, num_classes, dataset, vit_type,
                       sae_condition, args, max_batches=None):
    cpath = profile_cache_path(args.cache_dir, dataset, vit_type, sae_condition)
    if args.reuse_cache and max_batches is None and os.path.exists(cpath):
        return np.load(cpath)
    profile = compute_class_profile(vit, sae, cfg, train_ds, num_classes, args, max_batches)
    if args.reuse_cache and max_batches is None:
        os.makedirs(args.cache_dir, exist_ok=True)
        np.save(cpath, profile)
    return profile


# ═════════════════════════════════════════════════════════════════════════
# Top-k clamp hooks (from mdmnist_sae_eval.create_sae_hooks)
# ═════════════════════════════════════════════════════════════════════════

def create_sae_hooks(cfg, cls_features, sae, device, hook_type="on"):
    """hook_type='on': clamp the given features to 1, all others to 0.
    hook_type='off': clamp the given features to 0, all others to 1."""
    d_sae = cfg.d_sae
    clamp_feat_dim = torch.ones(d_sae).bool()

    if hook_type == "on":
        clamp_value = torch.zeros(d_sae, device=device)
        clamp_value[cls_features] = 1.0
    else:
        clamp_value = torch.ones(d_sae, device=device)
        clamp_value[cls_features] = 0.0

    def hook_fn(activations):
        activations[:, :, :] = (
            sae.forward_clamp(activations[:, :, :], clamp_feat_dim=clamp_feat_dim,
                               clamp_value=clamp_value)[0]
            - SAE_BIAS
        )
        return (activations,)

    return [Hook(cfg.block_layer, cfg.module_name, hook_fn,
                 return_module_output=False, is_custom=False)]


@torch.no_grad()
def get_predictions(vit, inputs, text_features, hooks=None):
    if hooks:
        vit_out = vit.run_with_hooks(hooks, return_type="output", **inputs)
    else:
        vit_out = vit(return_type="output", **inputs)
    image_features = vit_out.image_embeds
    logit_scale = vit.model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.t()
    return logits.argmax(dim=-1).cpu().numpy().tolist()


# ═════════════════════════════════════════════════════════════════════════
# Per-cell steering evaluation
# ═════════════════════════════════════════════════════════════════════════

def evaluate_steering_cell(dataset, vit_type, sae_condition, splits, lora_checkpoints,
                            sae_paths, args, max_batches=None):
    """Returns (rows, skip_reason). rows is a list of dicts matching RESULT_COLUMNS."""
    if vit_type != "lora":
        return [], "invalid combo (steering requires lora vit_type)"

    vit = get_vit(vit_type, dataset, lora_checkpoints, args)
    if vit is None:
        return [], "missing lora checkpoint"

    sae_path = resolve_sae_path(sae_condition, dataset, sae_paths, args)
    if not sae_path or not os.path.exists(sae_path):
        return [], f"missing sae checkpoint for condition={sae_condition}"
    sae, cfg = get_sae(sae_path, args.device)

    classnames, train_ds, eval_ds = splits
    num_classes = len(classnames)

    profile = get_class_profile(vit, sae, cfg, train_ds, num_classes, dataset,
                                 vit_type, sae_condition, args, max_batches)

    text_features = calculate_text_features(vit, args.device, classnames)

    # Group eval-split images by class (mirrors evaluate_topk_masking).
    class_images = defaultdict(list)
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=_hf_collate_fn, num_workers=2)
    for bidx, (images, labels) in enumerate(loader):
        if max_batches is not None and bidx >= max_batches:
            break
        for img, label in zip(images, labels.tolist()):
            class_images[label].append(img)

    d_sae = cfg.d_sae
    k_values = [k for k in args.k_values if k < d_sae]
    rows = []

    for cls_idx in range(num_classes):
        imgs = class_images.get(cls_idx, [])
        if not imgs:
            continue

        sorted_feats = profile[cls_idx].argsort()[::-1]
        preds_by_key = defaultdict(list)

        for start in range(0, len(imgs), args.batch_size):
            batch = imgs[start:start + args.batch_size]
            inputs = vit.processor(images=batch, text="", return_tensors="pt", padding=True).to(args.device)

            preds_by_key["baseline"].extend(get_predictions(vit, inputs, text_features))
            flush()

            for k in k_values:
                cls_features = sorted_feats[:k].tolist()
                hooks_on = create_sae_hooks(cfg, cls_features, sae, args.device, "on")
                preds_by_key[f"on_{k}"].extend(get_predictions(vit, inputs, text_features, hooks_on))
                flush()
                hooks_off = create_sae_hooks(cfg, cls_features, sae, args.device, "off")
                preds_by_key[f"off_{k}"].extend(get_predictions(vit, inputs, text_features, hooks_off))
                flush()

        n = len(imgs)
        baseline_acc = sum(1 for p in preds_by_key["baseline"] if p == cls_idx) / n * 100.0
        rows.append(dict(dataset=dataset, vit_type=vit_type, sae_condition=sae_condition,
                          **{"class": classnames[cls_idx]}, k=0, mode="baseline",
                          acc=baseline_acc, n_images=n))
        for k in k_values:
            on_acc = sum(1 for p in preds_by_key[f"on_{k}"] if p == cls_idx) / n * 100.0
            off_acc = sum(1 for p in preds_by_key[f"off_{k}"] if p == cls_idx) / n * 100.0
            rows.append(dict(dataset=dataset, vit_type=vit_type, sae_condition=sae_condition,
                              **{"class": classnames[cls_idx]}, k=k, mode="on",
                              acc=on_acc, n_images=n))
            rows.append(dict(dataset=dataset, vit_type=vit_type, sae_condition=sae_condition,
                              **{"class": classnames[cls_idx]}, k=k, mode="off",
                              acc=off_acc, n_images=n))

    return rows, None


# ═════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════

def build_summary(df):
    """Per (dataset, sae_condition, k): mean Δacc(ON) and mean Δacc(OFF)
    across classes, relative to that class's own baseline."""
    if df.empty:
        return []
    baseline = df[df["mode"] == "baseline"].set_index(
        ["dataset", "vit_type", "sae_condition", "class"])["acc"]

    out = []
    for (dataset, vit_type, sae_condition, k), g in df[df["mode"] != "baseline"].groupby(
            ["dataset", "vit_type", "sae_condition", "k"]):
        deltas_on, deltas_off = [], []
        for _, row in g.iterrows():
            key = (dataset, vit_type, sae_condition, row["class"])
            if key not in baseline.index:
                continue
            b = baseline.loc[key]
            if row["mode"] == "on":
                deltas_on.append(row["acc"] - b)
            elif row["mode"] == "off":
                deltas_off.append(row["acc"] - b)
        out.append(dict(
            dataset=dataset, vit_type=vit_type, sae_condition=sae_condition, k=int(k),
            mean_on_delta=float(np.mean(deltas_on)) if deltas_on else float("nan"),
            mean_off_delta=float(np.mean(deltas_off)) if deltas_off else float("nan"),
            n_classes=len(deltas_on),
        ))
    return sorted(out, key=lambda r: (r["dataset"], r["sae_condition"], r["k"]))


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

    registry, lora_checkpoints, sae_paths = load_common_registry_args(args)

    print("=" * 78)
    print(f"EVAL STEERING{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"datasets={args.datasets}  vit_types={args.vit_types}  "
          f"sae_conditions={args.sae_conditions}  k_values={args.k_values}")
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

    all_rows = []
    skip_reasons = {}
    max_batches = 1 if args.dry_run else None

    for vit_type in args.vit_types:
        for dataset in args.datasets:
            if dataset not in splits_by_dataset:
                for sae_condition in args.sae_conditions:
                    skip_reasons[(dataset, vit_type, sae_condition)] = load_errors[dataset]
                continue
            for sae_condition in args.sae_conditions:
                t0 = time.time()
                try:
                    rows, reason = evaluate_steering_cell(
                        dataset, vit_type, sae_condition, splits_by_dataset[dataset],
                        lora_checkpoints, sae_paths, args, max_batches=max_batches)
                except Exception as e:
                    rows, reason = [], f"exception: {e}"
                all_rows.extend(rows)
                if reason:
                    skip_reasons[(dataset, vit_type, sae_condition)] = reason
                status = "SKIP" if not rows else "OK"
                print(f"[{status}] {dataset:16s} {vit_type:5s} {sae_condition:10s}  "
                      f"rows={len(rows)}  ({time.time() - t0:.1f}s)"
                      + (f"  reason={reason}" if reason else ""))
                flush()

    df = pd.DataFrame(all_rows, columns=RESULT_COLUMNS)
    prefix = "dry_run_" if args.dry_run else ""
    csv_path = os.path.join(args.out_dir, f"{prefix}steering_results.csv")
    df.to_csv(csv_path, index=False)

    summary_rows = build_summary(df)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": args.dry_run,
        "args": {k: v for k, v in vars(args).items()},
        "n_rows": len(all_rows),
        "skip_reasons": {"|".join(k): v for k, v in skip_reasons.items()},
        "dataset_load_errors": load_errors,
        "per_dataset_sae_k": summary_rows,
    }
    summary_path = os.path.join(args.out_dir, f"{prefix}summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 78)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"{len(all_rows)} rows written, {len(skip_reasons)} cell(s) skipped")
    if args.dry_run:
        print("\nPlanned summary (dry-run, 1 batch/class only — deltas are not meaningful):")
        for r in summary_rows:
            print(f"  {r['dataset']:14s} {r['sae_condition']:6s} k={r['k']:<4d} "
                  f"ON Δ={r['mean_on_delta']:+.2f}  OFF Δ={r['mean_off_delta']:+.2f}")


if __name__ == "__main__":
    main()

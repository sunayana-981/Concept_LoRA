#!/usr/bin/env python3
"""
Qualitative figure: top-9 activating images per class-selective SAE feature,
G-SAE vs FT-SAE, for a given dataset + the LoRA model.

Built on top_activating.py's SAE-activation extraction (get_sae_acts) and
tasks/eval_steering.py's per-class activation profile + cache (get_class_profile),
rather than re-deriving either from scratch.

NOTE on layer source: top_activating.py's own get_top_neurons() reads
`vision_model(pixel_values=...).last_hidden_state`, i.e. the *final* encoder
layer's output, regardless of the SAE's cfg.block_layer. Our SAEs are all
trained at block_layer=-2 (the repo-wide convention — see tasks/rebuttal_common.py),
so feeding them final-layer activations here would silently score every
feature on the wrong tensor. This script instead captures patch activations
at cfg.block_layer via HookedVisionTransformer.run_with_cache (the same
mechanism tasks/eval_matrix.py and tasks/eval_steering.py use), and reuses
get_sae_acts only for its SAE-forward-tuple unpacking.

Pipeline per (dataset, sae_condition in {gsae, ftsae}):
  1. Per-class CLS-token activation profile (via eval_steering.get_class_profile,
     cached compatibly with Tasks 1/2 under out/rebuttal/cache/).
  2. Among the --n_candidates most active features, pick the --n_select with
     lowest label entropy (highest class selectivity).
  3. Stream the dataset once, keeping a top---top_n min-heap of (activation,
     image, class) per selected feature — memory bounded regardless of
     dataset size.
  4. Save one 3x3 grid PNG per feature + index.csv, plus one G-SAE-vs-FT-SAE
     side-by-side comparison PNG.

Usage:
    python tasks/top_activating_grid.py --dry_run --dataset pathmnist
    python tasks/top_activating_grid.py --dataset pathmnist \
        --lora_checkpoints configs/rebuttal_lora_checkpoints.json \
        --sae_paths configs/rebuttal_sae_paths.json
"""

import argparse
import itertools
import json
import os
import sys
import time
import heapq
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from top_activating import get_sae_acts
from eval_medmnist_sae import _hf_collate_fn
from tasks.rebuttal_common import (
    add_common_args, flush, load_common_registry_args, build_dataset_splits,
    get_vit, resolve_sae_path, get_sae,
)
from tasks.eval_steering import get_class_profile

QUALITATIVE_SAE_CONDITIONS = [
    "gsae", "ftsae", "scratchsae", "masked", "masked_gated"
]


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="pathmnist")
    p.add_argument("--vit_type", type=str, default="lora", choices=["lora"])
    p.add_argument("--sae_conditions", nargs="+", default=QUALITATIVE_SAE_CONDITIONS,
                    choices=QUALITATIVE_SAE_CONDITIONS)
    p.add_argument("--n_candidates", type=int, default=500,
                    help="Pool of most-active features to search for class selectivity.")
    p.add_argument("--n_select", type=int, default=8,
                    help="Number of lowest-entropy (most class-selective) features to visualize.")
    p.add_argument("--top_n", type=int, default=9,
                    help="Top-activating images to keep per feature.")
    p = add_common_args(p, out_dir_default="out/rebuttal/qualitative")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════
# Feature selection: most-active candidates -> lowest label entropy
# ═════════════════════════════════════════════════════════════════════════

def select_class_selective_features(cls_mean, classnames, n_candidates, n_select):
    """cls_mean: [num_classes, d_sae] mean CLS activation per class per feature."""
    d_sae = cls_mean.shape[1]
    n_candidates = min(n_candidates, d_sae)
    n_select = min(n_select, n_candidates)

    activity = cls_mean.mean(axis=0)
    candidate_idx = np.argsort(-activity)[:n_candidates]

    cand = cls_mean[:, candidate_idx]
    totals = cand.sum(axis=0) + 1e-12
    p = cand / totals
    entropy = -(p * np.log(p + 1e-12)).sum(axis=0)

    order = np.argsort(entropy)[:n_select]
    selected = candidate_idx[order]
    dominant = cand[:, order].argmax(axis=0)

    return [
        dict(feature_id=int(selected[i]),
             label_entropy=float(entropy[order[i]]),
             dominant_class=classnames[int(dominant[i])],
             mean_act=float(activity[selected[i]]))
        for i in range(len(selected))
    ]


# ═════════════════════════════════════════════════════════════════════════
# Streaming top-N images per feature (bounded-memory heap)
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def batched_patch_acts(vit, sae, cfg, images, args):
    """[B, num_patches, d_sae] SAE feature activations over patch tokens
    (CLS excluded), captured at cfg.block_layer. Reuses get_sae_acts for the
    SAE forward-pass tuple handling."""
    inputs = vit.processor(images=images, text="", return_tensors="pt", padding=True).to(args.device)
    _, cache = vit.run_with_cache([(cfg.block_layer, cfg.module_name)], **inputs)
    acts_in = cache[(cfg.block_layer, cfg.module_name)]  # [B, seq_len, d_in], incl. CLS
    patches = acts_in[:, 1:, :].float()
    return get_sae_acts(sae, patches)  # [B, num_patches, d_sae]


@torch.no_grad()
def find_top_activating_images(vit, sae, cfg, dataset, feature_ids, classnames, args,
                                top_n=9, max_batches=None):
    """Returns {feature_id: [(act, PIL_image, classname), ...]} sorted descending,
    keeping only the top_n per feature at any time (bounded memory)."""
    heaps = {fid: [] for fid in feature_ids}
    counter = itertools.count()
    fid_tensor = torch.tensor(feature_ids, device=args.device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         collate_fn=_hf_collate_fn, num_workers=2)

    for bidx, (images, labels) in enumerate(tqdm(loader, desc="streaming top images", leave=False)):
        if max_batches is not None and bidx >= max_batches:
            break
        acts = batched_patch_acts(vit, sae, cfg, images, args)  # [B, P, d_sae]
        sel = acts.index_select(-1, fid_tensor)                  # [B, P, n_feat]
        max_per_feat, _ = sel.max(dim=1)                          # [B, n_feat]
        max_per_feat = max_per_feat.cpu().numpy()

        for bi in range(len(images)):
            classname = classnames[labels[bi].item()]
            for fi, fid in enumerate(feature_ids):
                val = float(max_per_feat[bi, fi])
                item = (val, next(counter), images[bi].copy(), classname)
                if len(heaps[fid]) < top_n:
                    heapq.heappush(heaps[fid], item)
                elif val > heaps[fid][0][0]:
                    heapq.heapreplace(heaps[fid], item)

    return {fid: sorted(heaps[fid], key=lambda x: -x[0]) for fid in feature_ids}


# ═════════════════════════════════════════════════════════════════════════
# Figure rendering
# ═════════════════════════════════════════════════════════════════════════

def save_feature_grid(top_items, feature_id, sae_condition, out_path):
    n = len(top_items)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            act, _, img, classname = top_items[i]
            ax.imshow(img)
            ax.set_title(f"#{i+1}  act={act:.2f}\n{classname}", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{sae_condition}  feature {feature_id}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_comparison_figure(selections_by_sae, top_items_by_sae, out_path,
                            row_order):
    row_order = [s for s in row_order if s in selections_by_sae]
    ncols = max(len(selections_by_sae[s]) for s in row_order)
    fig, axes = plt.subplots(len(row_order), ncols,
                              figsize=(2.4 * ncols, 2.6 * len(row_order)))
    axes = np.atleast_2d(axes)

    for r, sae_condition in enumerate(row_order):
        selection = selections_by_sae[sae_condition]
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(selection):
                continue
            feat = selection[c]
            items = top_items_by_sae[sae_condition].get(feat["feature_id"], [])
            if not items:
                continue
            act, _, img, classname = items[0]
            ax.imshow(img)
            ax.set_title(f"f{feat['feature_id']}  {classname}\nact={act:.2f}", fontsize=8)
        # axis("off") above suppresses set_ylabel, so use a figure-level label
        # positioned at this row's vertical center instead.
        row_top = 1 - r / len(row_order)
        row_bottom = 1 - (r + 1) / len(row_order)
        fig.text(0.01, (row_top + row_bottom) / 2, sae_condition.upper(),
                  fontsize=12, fontweight="bold", rotation=90,
                  va="center", ha="left")

    fig.suptitle("Top-activating feature (rank 1), " + " vs ".join(s.upper() for s in row_order), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(dataset_out_dir, exist_ok=True)
    if args.reuse_cache:
        os.makedirs(args.cache_dir, exist_ok=True)

    setattr(args, "datasets", [args.dataset])
    registry, lora_checkpoints, sae_paths = load_common_registry_args(args)

    print("=" * 78)
    print(f"TOP-ACTIVATING GRID{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"dataset={args.dataset}  vit_type={args.vit_type}  sae_conditions={args.sae_conditions}")
    print("=" * 78)

    classnames, train_ds, eval_ds = build_dataset_splits(args.dataset, registry[args.dataset], args)
    num_classes = len(classnames)
    print(f"[OK] dataset '{args.dataset}': {len(train_ds)} train / {len(eval_ds)} eval images, "
          f"{num_classes} classes")

    vit = get_vit(args.vit_type, args.dataset, lora_checkpoints, args)
    if vit is None:
        print(f"[FATAL] missing lora checkpoint for dataset '{args.dataset}'")
        sys.exit(1)

    max_batches = 1 if args.dry_run else None
    n_select = 2 if args.dry_run else args.n_select
    top_n = 3 if args.dry_run else args.top_n

    index_rows = []
    selections_by_sae, top_items_by_sae = {}, {}

    for sae_condition in args.sae_conditions:
        sae_path = resolve_sae_path(sae_condition, args.dataset, sae_paths, args)
        if not sae_path or not os.path.exists(sae_path):
            print(f"[SKIP] {sae_condition}: missing sae checkpoint")
            continue
        sae, cfg = get_sae(sae_path, args.device)

        print(f"\n[{sae_condition}] computing per-class activation profile...")
        profile = get_class_profile(vit, sae, cfg, train_ds, num_classes, args.dataset,
                                     args.vit_type, sae_condition, args, max_batches)

        selection = select_class_selective_features(profile, classnames, args.n_candidates, n_select)
        print(f"[{sae_condition}] selected {len(selection)} class-selective features: "
              + ", ".join(f"f{s['feature_id']}(H={s['label_entropy']:.2f},{s['dominant_class']})"
                           for s in selection))

        feature_ids = [s["feature_id"] for s in selection]
        top_items = find_top_activating_images(vit, sae, cfg, eval_ds, feature_ids, classnames,
                                                 args, top_n=top_n, max_batches=max_batches)

        sae_out_dir = os.path.join(dataset_out_dir, sae_condition)
        os.makedirs(sae_out_dir, exist_ok=True)
        for feat in selection:
            fid = feat["feature_id"]
            items = top_items.get(fid, [])
            fig_path = os.path.join(sae_out_dir, f"feature_{fid}.png")
            if items:
                save_feature_grid(items, fid, sae_condition, fig_path)
            index_rows.append(dict(sae_condition=sae_condition, feature_id=fid,
                                    label_entropy=feat["label_entropy"],
                                    dominant_class=feat["dominant_class"],
                                    mean_act=feat["mean_act"], n_images_found=len(items)))

        selections_by_sae[sae_condition] = selection
        top_items_by_sae[sae_condition] = top_items
        flush()

    index_path = os.path.join(dataset_out_dir, "index.csv")
    pd.DataFrame(index_rows, columns=["sae_condition", "feature_id", "label_entropy",
                                       "dominant_class", "mean_act", "n_images_found"]
                 ).to_csv(index_path, index=False)
    print(f"\nWrote {index_path}")

    if len(selections_by_sae) >= 2:
        suffix = "_vs_".join(args.sae_conditions)
        comp_path = os.path.join(dataset_out_dir, f"comparison_{suffix}.png")
        save_comparison_figure(selections_by_sae, top_items_by_sae, comp_path,
                                row_order=args.sae_conditions)
        print(f"Wrote {comp_path}")
    else:
        print("[SKIP] comparison figure: need >=2 sae_conditions with results")

    summary_path = os.path.join(dataset_out_dir, f"{'dry_run_' if args.dry_run else ''}summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dry_run": args.dry_run,
            "args": {k: v for k, v in vars(args).items()},
            "sae_conditions_found": list(selections_by_sae.keys()),
            "n_features_per_sae": {k: len(v) for k, v in selections_by_sae.items()},
        }, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

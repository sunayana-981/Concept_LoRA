#!/usr/bin/env python3
"""
t-SNE comparison of SAE feature space across conditions (e.g. gsae vs masked),
for a given dataset + the LoRA model.

Unlike kmeans.py (which depends on tasks/compute_sae_feature_data.py's
precomputed feature_data/ and only supports vit_type in {base, maple}), this
reuses tasks/rebuttal_common.py's loaders -- the same LoRA-aware machinery
tasks/eval_matrix.py and tasks/top_activating_grid.py already use -- so it
works directly against our masked-fine-tune checkpoints without a separate
precompute step.

For each sae_condition, we sample --n_samples eval images, compute the CLS-
token SAE feature vector (sparse, d_sae-dim) for each, reduce with PCA then
t-SNE, and plot colored by true class label. A tighter, better-separated
embedding under one condition is qualitative evidence that its dictionary
carries more class-discriminative structure -- the geometric counterpart of
the zero-shot accuracy numbers in eval_matrix.py.

Usage:
    python tasks/tsne_sae_features.py --dataset cub2002011 --sae_conditions gsae masked \
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
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_medmnist_sae import _hf_collate_fn
from tasks.rebuttal_common import (
    add_common_args, flush, load_common_registry_args, build_dataset_splits,
    get_vit, resolve_sae_path, get_sae,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="cub2002011")
    p.add_argument("--vit_type", type=str, default="lora", choices=["lora"])
    p.add_argument("--sae_conditions", nargs="+", default=["gsae", "masked"])
    p.add_argument("--n_samples", type=int, default=600,
                    help="Number of eval images to embed per condition.")
    p.add_argument("--n_clusters", type=int, default=10,
                    help="KMeans clusters for the quantitative purity metric.")
    p.add_argument("--pca_dim", type=int, default=50)
    p = add_common_args(p, out_dir_default="out/rebuttal/tsne")
    return p.parse_args()


@torch.no_grad()
def compute_cls_sae_features(vit, sae, cfg, dataset, args, n_samples):
    """Returns (features [n, d_sae], labels [n])."""
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         collate_fn=_hf_collate_fn, num_workers=2)
    feats, labels = [], []
    n_seen = 0
    for images, batch_labels in tqdm(loader, desc="SAE CLS features", leave=False):
        inputs = vit.processor(images=images, text="", return_tensors="pt", padding=True).to(args.device)
        _, cache = vit.run_with_cache([(cfg.block_layer, cfg.module_name)], **inputs)
        acts = cache[(cfg.block_layer, cfg.module_name)][:, 0, :].float()  # CLS token
        _, feature_acts, _ = sae.forward(acts)
        feats.append(feature_acts.cpu().numpy())
        labels.append(batch_labels.numpy())
        n_seen += len(images)
        if n_seen >= n_samples:
            break
    return np.concatenate(feats, axis=0)[:n_samples], np.concatenate(labels, axis=0)[:n_samples]


def embed_and_plot(feats, labels, classnames, title, out_path, pca_dim, n_clusters, seed):
    pca_dim = min(pca_dim, feats.shape[0] - 1, feats.shape[1])
    pca = PCA(n_components=pca_dim, random_state=seed)
    reduced = pca.fit_transform(feats)

    tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30)
    coords = tsne.fit_transform(reduced)

    km = KMeans(n_clusters=min(n_clusters, len(np.unique(labels))), random_state=seed, n_init=10)
    cluster_ids = km.fit_predict(reduced)
    ari = adjusted_rand_score(labels, cluster_ids)
    try:
        sil = silhouette_score(reduced, labels)
    except ValueError:
        sil = float("nan")

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=10, alpha=0.75)
    ax.set_title(f"{title}\nARI(cluster,label)={ari:.3f}  silhouette={sil:.3f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return dict(ari=float(ari), silhouette=float(sil), n_points=int(feats.shape[0]))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(dataset_out_dir, exist_ok=True)

    setattr(args, "datasets", [args.dataset])
    registry, lora_checkpoints, sae_paths = load_common_registry_args(args)

    print("=" * 78)
    print(f"t-SNE SAE FEATURE COMPARISON  dataset={args.dataset}  conditions={args.sae_conditions}")
    print("=" * 78)

    classnames, train_ds, eval_ds = build_dataset_splits(args.dataset, registry[args.dataset], args)
    print(f"[OK] dataset '{args.dataset}': {len(eval_ds)} eval images, {len(classnames)} classes")

    vit = get_vit(args.vit_type, args.dataset, lora_checkpoints, args)
    if vit is None:
        print(f"[FATAL] missing lora checkpoint for dataset '{args.dataset}'")
        sys.exit(1)

    metrics = {}
    for sae_condition in args.sae_conditions:
        sae_path = resolve_sae_path(sae_condition, args.dataset, sae_paths, args)
        if not sae_path or not os.path.exists(sae_path):
            print(f"[SKIP] {sae_condition}: missing sae checkpoint")
            continue
        sae, cfg = get_sae(sae_path, args.device)

        print(f"\n[{sae_condition}] computing CLS-token SAE features over {args.n_samples} images...")
        feats, labels = compute_cls_sae_features(vit, sae, cfg, eval_ds, args, args.n_samples)
        print(f"[{sae_condition}] features: {feats.shape}, L0={np.mean(feats > 0) * feats.shape[1]:.1f}")

        out_path = os.path.join(dataset_out_dir, f"tsne_{sae_condition}.png")
        m = embed_and_plot(feats, labels, classnames,
                            title=f"{args.dataset}  {sae_condition.upper()}  (LoRA)",
                            out_path=out_path, pca_dim=args.pca_dim,
                            n_clusters=args.n_clusters, seed=args.seed)
        metrics[sae_condition] = m
        print(f"[{sae_condition}] wrote {out_path}  ARI={m['ari']:.3f}  silhouette={m['silhouette']:.3f}")
        flush()

    summary_path = os.path.join(dataset_out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset,
            "sae_conditions": args.sae_conditions,
            "metrics": metrics,
        }, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()

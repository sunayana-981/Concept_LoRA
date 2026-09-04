#!/usr/bin/env python3
"""
t-SNE of SAE latent activations — 3-way SAE comparison.

For each dataset we compare:
  1. Base backbone + Base SAE
  2. LoRA backbone + Base SAE   (cross condition)
  3. LoRA backbone + Adapted SAE

Each condition:
  1. Run images → get SAE latent vector h per image
  2. t-SNE on h, color by class

This shows whether improvements come from the LoRA backbone alone,
the adapted SAE alone, or both together.

Usage:
    python visualize_latent_tsne.py \
        --datasets eurosat medmnist caltech101 \
        --max_images 2000 \
        --out_dir out/latent_tsne
"""

import argparse
import os
import sys
import random
import inspect
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

# Global plotting style tuned for very large, presentation-friendly text.
plt.rcParams.update({
    "font.size": 33,
    "axes.titlesize": 39,
    "axes.labelsize": 35,
    "legend.fontsize": 27,
    "legend.title_fontsize": 29,
    "figure.titlesize": 41,
    "xtick.labelsize": 29,
    "ytick.labelsize": 29,
})

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.registry import get_backbone
from src.data.dataset_registry import get_dataset, get_classnames, get_label_key, load_registry
from src.sae_training.loaders import load_sae, load_lora_weights


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_SAE_PATH  = os.path.join(_ROOT, "out", "sae_weight", "base", "out.pt")
BASE_SAE_LAYER = -2
BACKBONE_ID    = "clip_vit_b16"
LORA_ROOT      = os.path.join(_ROOT, "..", "lora_weights", "vitb16")

ADSAE_CONFIGS: Dict[str, dict] = {
    "caltech101": {
        "sae_path": os.path.join(
            _ROOT, "out", "checkpoints", "caltech101", "ted4zuln",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-3_resid_49152.pt"),
        "layer": -3,
        "lora_path": os.path.join(LORA_ROOT, "caltech101", "16shots", "seed1", "lora_weights.pt"),
    },
    "eurosat": {
        "sae_path": os.path.join(
            _ROOT, "out", "checkpoints", "eurosat", "m8tn3p5s",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-1_resid_49152.pt"),
        "layer": -1,
        "lora_path": os.path.join(LORA_ROOT, "eurosat", "16shots", "seed1", "lora_weights.pt"),
    },
    "medmnist": {
        "sae_path": os.path.join(
            _ROOT, "out", "checkpoints", "medmnist", "d2ygd3bb",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "medmnist", "16shots", "seed1", "lora_weights.pt"),
    },
    "dtd": {
        "sae_path": os.path.join(
            _ROOT, "out", "checkpoints", "dtd", "sd5h6hxv",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "dtd", "16shots", "seed42", "lora_weights.pt"),
    },
    "ucf101": {
        "sae_path": os.path.join(
            _ROOT, "out", "checkpoints", "ucf101", "j04tcnkc",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "ucf101", "16shots", "seed1", "lora_weights.pt"),
    },
}

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#e6beff",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_base_backbone(device):
    bb = get_backbone(BACKBONE_ID, device=device).load()
    bb.model.eval()
    return bb


def load_lora_backbone(lora_path, device):
    bb = get_backbone(BACKBONE_ID, device=device).load()
    class _W:
        def __init__(self, m): self.model = m
    load_lora_weights(_W(bb.model), lora_path, device)
    bb.model.eval()
    return bb


def _collate(processor, label_key):
    def fn(batch):
        imgs   = [item["image"].convert("RGB") for item in batch]
        labels = [item[label_key] for item in batch]
        pv     = processor(images=imgs, return_tensors="pt")["pixel_values"]
        return pv, torch.tensor(labels, dtype=torch.long)
    return fn


def get_colors(num_classes):
    if num_classes <= len(PALETTE):
        return PALETTE[:num_classes]
    cmap = plt.cm.turbo
    return [matplotlib.colors.rgb2hex(cmap(i / max(num_classes - 1, 1)))
            for i in range(num_classes)]


def stratified_subset(dataset, label_key, max_samples, seed=42):
    """Class-balanced subset."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    labels = dataset[label_key]
    by_class = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[int(lab)].append(idx)
    rng = random.Random(seed)
    for idxs in by_class.values():
        rng.shuffle(idxs)
    class_ids = sorted(by_class.keys())
    per_class = max(1, max_samples // len(class_ids))
    selected = []
    for cid in class_ids:
        selected.extend(by_class[cid][:per_class])
    rng.shuffle(selected)
    return dataset.select(selected[:max_samples])


# ── Extract SAE latents ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(backbone, sae, dataset, label_key, layer,
                    batch_size=64, device="cuda", num_workers=4):
    """
    Run all images through backbone → hook layer → SAE encoder.
    Returns:
        latents: (N, d_sae) sparse activation vectors
        labels:  (N,) class labels
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        collate_fn=_collate(backbone.processor, label_key),
    )
    layer_idx = backbone.resolve_layer(layer)
    cap = {}

    def hook(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        cap["cls"] = hs.detach().float()[:, 0, :]

    h = backbone.model.vision_model.encoder.layers[layer_idx].register_forward_hook(hook)

    all_latents, all_labels = [], []
    for pv, labels in tqdm(loader, desc="    extracting latents", leave=False):
        backbone.model.vision_model(pixel_values=pv.to(device))
        _, latents, _ = sae(cap["cls"])  # (B, d_sae)
        all_latents.append(latents.cpu().float())
        all_labels.append(labels)

    h.remove()
    return torch.cat(all_latents), torch.cat(all_labels)


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def run_tsne(latents_np, perplexity=30, seed=42):
    """SVD(50) → t-SNE(2)."""
    n, d = latents_np.shape
    n_svd = min(50, n - 1, d)
    if d > n_svd:
        reduced = TruncatedSVD(n_components=n_svd, random_state=seed).fit_transform(latents_np)
    else:
        reduced = latents_np

    perp = min(perplexity, max(5, n // 4))
    tsne_kwargs = dict(
        n_components=2, perplexity=perp, random_state=seed,
        learning_rate="auto", init="pca",
    )
    sig = inspect.signature(TSNE.__init__).parameters
    if "max_iter" in sig:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000

    return TSNE(**tsne_kwargs).fit_transform(reduced.astype(np.float32))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_tsne_comparison(
    emb_base, labels_base,
    emb_cross, labels_cross,
    emb_adapted, labels_adapted,
    classnames, ds_name,
    silhouette_base, silhouette_cross, silhouette_adapted,
    sparsity_base, sparsity_cross, sparsity_adapted,
    out_path,
):
    """
    3-panel t-SNE:
      left:  Base backbone + Base SAE
      mid:   LoRA backbone + Base SAE
      right: LoRA backbone + Adapted SAE
    """
    num_classes = len(classnames)
    colors = get_colors(num_classes)

    fig, axes = plt.subplots(1, 3, figsize=(48, 14))

    panels = [
        ("Base backbone + Base SAE", emb_base, labels_base.numpy(),
         silhouette_base, sparsity_base),
        ("LoRA backbone + Base SAE", emb_cross, labels_cross.numpy(),
         silhouette_cross, sparsity_cross),
        (f"LoRA backbone + Adapted SAE ({ds_name})", emb_adapted, labels_adapted.numpy(),
         silhouette_adapted, sparsity_adapted),
    ]

    for ax, (title, emb, labs, sil, sparsity) in zip(axes, panels):
        # Scatter
        point_colors = [colors[int(l) % len(colors)] for l in labs]
        ax.scatter(
            emb[:, 0], emb[:, 1],
            c=point_colors, s=46, alpha=0.75,
            linewidths=0.0, zorder=2,
        )

        # Large class labels at class centroids (no legend).
        for cls_idx, cls_name in enumerate(classnames):
            mask = (labs == cls_idx)
            if not np.any(mask):
                continue
            cx = float(np.median(emb[mask, 0]))
            cy = float(np.median(emb[mask, 1]))
            txt = ax.text(
                cx, cy, cls_name,
                color=colors[cls_idx % len(colors)],
                fontsize=34, fontweight="bold",
                ha="center", va="center", zorder=5,
            )
            txt.set_path_effects([
                pe.Stroke(linewidth=5, foreground="white"),
                pe.Normal(),
            ])

        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle="--", alpha=0.1, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # High-res PNG for quick viewing, plus vector PDF for Overleaf.
    fig.savefig(out_path, dpi=450, bbox_inches="tight", facecolor="white")
    out_path_pdf = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(out_path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    print(f"  Saved → {out_path_pdf}")


def plot_similarity_heatmaps(
    latents_base, labels_base,
    latents_cross, labels_cross,
    latents_adapted, labels_adapted,
    classnames, ds_name, out_path,
):
    """
    Class-centroid cosine similarity matrices for 3 conditions.
    """
    num_classes = len(classnames)

    def compute_centroid_sim(latents, labels):
        centroids = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() == 0:
                centroids.append(torch.zeros(latents.shape[1]))
            else:
                cent = latents[mask].mean(dim=0)
                centroids.append(cent)
        C = torch.stack(centroids)  # (num_classes, d_sae)
        C_norm = C / (C.norm(dim=1, keepdim=True) + 1e-8)
        return (C_norm @ C_norm.t()).numpy()

    sim_base = compute_centroid_sim(latents_base, labels_base)
    sim_cross = compute_centroid_sim(latents_cross, labels_cross)
    sim_adapted = compute_centroid_sim(latents_adapted, labels_adapted)

    all_sims = np.stack([sim_base, sim_cross, sim_adapted], axis=0)
    data_min = float(np.min(all_sims))
    # Keep zero visually meaningful while still showing negative similarities if present.
    vmin = min(-0.1, data_min)
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=1.0)

    fig = plt.figure(figsize=(30, 10), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    for ax, sim, title in [
        (axes[0], sim_base, "Base backbone + Base SAE"),
        (axes[1], sim_cross, "LoRA backbone + Base SAE"),
        (axes[2], sim_adapted, f"LoRA backbone + Adapted SAE ({ds_name})"),
    ]:
        im = ax.imshow(sim, cmap="RdBu_r", norm=norm, interpolation="nearest", aspect="equal")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        short_names = [n if len(n) <= 14 else (n[:13] + "…") for n in classnames]
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=18)
        ax.set_yticklabels(short_names, fontsize=18)
        ax.set_title(title, fontsize=24, fontweight="bold")

        # Annotate diagonal mean and off-diagonal mean
        diag = np.diag(sim).mean()
        mask_off = ~np.eye(num_classes, dtype=bool)
        off_diag = sim[mask_off].mean()
        ax.text(
            0.02, 0.98,
            f"Diag mean: {diag:.3f}\nOff-diag mean: {off_diag:.3f}\n"
            f"Ratio: {diag / (off_diag + 1e-8):.1f}×",
            transform=ax.transAxes, fontsize=15,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999", alpha=0.86),
        )
        ax.tick_params(length=0)

    fig.savefig(out_path, dpi=450, bbox_inches="tight", facecolor="white")
    out_path_pdf = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(out_path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    print(f"  Saved → {out_path_pdf}")


def plot_sparsity_comparison(
    latents_base, labels_base,
    latents_cross, labels_cross,
    latents_adapted, labels_adapted,
    classnames, ds_name, out_path,
):
    """
    Per-class L0 sparsity (how many neurons fire) for all 3 conditions.
    """
    num_classes = len(classnames)

    def per_class_l0(latents, labels):
        l0s = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() == 0:
                l0s.append(0)
            else:
                active = (latents[mask] > 0).float().sum(dim=1)  # per image
                l0s.append(float(active.mean()))
        return l0s

    l0_base = per_class_l0(latents_base, labels_base)
    l0_cross = per_class_l0(latents_cross, labels_cross)
    l0_adapted = per_class_l0(latents_adapted, labels_adapted)

    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.6), 5))
    x = np.arange(num_classes)
    w = 0.26

    ax.bar(x - w, l0_base, w, color="#7faed4", edgecolor="white",
           label="Base backbone + Base SAE", zorder=3)
    ax.bar(x, l0_cross, w, color="#f2b880", edgecolor="white",
           label="LoRA backbone + Base SAE", zorder=3)
    ax.bar(x + w, l0_adapted, w, color="#d47f7f", edgecolor="white",
           label="LoRA backbone + Adapted SAE", zorder=3)

    short_names = [n[:12] for n in classnames]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean active neurons (L₀)", fontsize=10)
    ax.set_title(
        f"Per-Class Sparsity — {ds_name}\n"
        "Fewer active neurons = more selective representation.",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    # Overall stats
    mean_base = np.mean(l0_base)
    mean_cross = np.mean(l0_cross)
    mean_adapted = np.mean(l0_adapted)
    ax.text(
        0.98, 0.95,
        f"Mean L₀: Base={mean_base:.0f}, Cross={mean_cross:.0f}, Adapted={mean_adapted:.0f}",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#aaa", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Per-dataset pipeline ──────────────────────────────────────────────────────

def process_dataset(ds_name, args, registry, device):
    if ds_name not in ADSAE_CONFIGS:
        print(f"[WARN] No config for {ds_name}, skipping.")
        return

    cfg = ADSAE_CONFIGS[ds_name]
    out_dir_ds = os.path.join(args.out_dir, ds_name)
    os.makedirs(out_dir_ds, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  Dataset: {ds_name.upper()}")
    print(f"{'═'*65}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_full = get_dataset(ds_name, registry=registry, max_samples=None)
    classnames = get_classnames(ds_name, dataset=dataset_full, registry=registry)
    label_key  = get_label_key(ds_name, dataset_full, registry=registry)
    dataset = stratified_subset(dataset_full, label_key, args.max_images, seed=42)
    num_classes = len(classnames)
    print(f"  {len(dataset):,} images (from {len(dataset_full):,}), {num_classes} classes")

    # ── Load SAEs + backbones ─────────────────────────────────────────────────
    for path, name in [(cfg["sae_path"], "AdSAE"), (BASE_SAE_PATH, "BaseSAE"),
                       (cfg["lora_path"], "LoRA")]:
        if not os.path.isfile(path):
            print(f"  [SKIP] {name} not found: {path}"); return

    base_sae, _ = load_sae(BASE_SAE_PATH, device); base_sae.eval()
    ad_sae,   _ = load_sae(cfg["sae_path"], device); ad_sae.eval()
    bb_base = load_base_backbone(device)
    bb_lora = load_lora_backbone(cfg["lora_path"], device)

    # ── Extract latents ───────────────────────────────────────────────────────
    # Condition 1: Base backbone + Base SAE
    print("\n  Extracting: Base backbone + Base SAE")
    lat_base, lab_base = extract_latents(
        bb_base, base_sae, dataset, label_key, BASE_SAE_LAYER,
        batch_size=args.batch_size, device=device, num_workers=args.num_workers,
    )

    # Condition 2: LoRA backbone + Base SAE (cross condition)
    print("  Extracting: LoRA backbone + Base SAE")
    lat_cross, lab_cross = extract_latents(
        bb_lora, base_sae, dataset, label_key, BASE_SAE_LAYER,
        batch_size=args.batch_size, device=device, num_workers=args.num_workers,
    )

    # Condition 3: LoRA backbone + Adapted SAE
    print("  Extracting: LoRA backbone + Adapted SAE")
    lat_adapted, lab_adapted = extract_latents(
        bb_lora, ad_sae, dataset, label_key, cfg["layer"],
        batch_size=args.batch_size, device=device, num_workers=args.num_workers,
    )

    print(
        f"  Latent shapes: base={lat_base.shape}, cross={lat_cross.shape}, adapted={lat_adapted.shape}"
    )

    # ── Compute metrics ───────────────────────────────────────────────────────
    # L0 sparsity
    l0_base = float((lat_base > 0).float().sum(dim=1).mean())
    l0_cross = float((lat_cross > 0).float().sum(dim=1).mean())
    l0_adapted = float((lat_adapted > 0).float().sum(dim=1).mean())
    print(
        f"  Mean L₀ sparsity: base={l0_base:.0f}, cross={l0_cross:.0f}, adapted={l0_adapted:.0f}"
    )

    # Silhouette score (on SVD-reduced latents for speed)
    print("  Computing silhouette scores...")
    n_svd = min(50, lat_base.shape[0] - 1, lat_base.shape[1])

    svd_base = TruncatedSVD(n_components=n_svd, random_state=42).fit_transform(
        lat_base.numpy())
    sil_base = silhouette_score(svd_base, lab_base.numpy(),
                                sample_size=min(2000, len(lab_base)))

    n_svd_cross = min(50, lat_cross.shape[0] - 1, lat_cross.shape[1])
    svd_cross = TruncatedSVD(n_components=n_svd_cross, random_state=42).fit_transform(
        lat_cross.numpy())
    sil_cross = silhouette_score(svd_cross, lab_cross.numpy(),
                                 sample_size=min(2000, len(lab_cross)))

    n_svd_ad = min(50, lat_adapted.shape[0] - 1, lat_adapted.shape[1])
    svd_adapted = TruncatedSVD(n_components=n_svd_ad, random_state=42).fit_transform(
        lat_adapted.numpy())
    sil_adapted = silhouette_score(svd_adapted, lab_adapted.numpy(),
                                   sample_size=min(2000, len(lab_adapted)))
    print(
        f"  Silhouette: base={sil_base:.3f}, cross={sil_cross:.3f}, adapted={sil_adapted:.3f}"
    )

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print("  Running t-SNE (base)...")
    emb_base = run_tsne(lat_base.numpy(), perplexity=args.perplexity, seed=42)
    print("  Running t-SNE (cross)...")
    emb_cross = run_tsne(lat_cross.numpy(), perplexity=args.perplexity, seed=42)
    print("  Running t-SNE (adapted)...")
    emb_adapted = run_tsne(lat_adapted.numpy(), perplexity=args.perplexity, seed=42)

    # ── Plot 1: t-SNE comparison ──────────────────────────────────────────────
    print("  Plotting t-SNE comparison...")
    plot_tsne_comparison(
        emb_base, lab_base,
        emb_cross, lab_cross,
        emb_adapted, lab_adapted,
        classnames, ds_name,
        sil_base, sil_cross, sil_adapted,
        l0_base, l0_cross, l0_adapted,
        out_path=os.path.join(out_dir_ds, "tsne_comparison.png"),
    )

    # ── Plot 2: Similarity heatmaps ───────────────────────────────────────────
    if num_classes <= 30:
        print("  Plotting similarity heatmaps...")
        plot_similarity_heatmaps(
            lat_base, lab_base,
            lat_cross, lab_cross,
            lat_adapted, lab_adapted,
            classnames, ds_name,
            out_path=os.path.join(out_dir_ds, "class_similarity.png"),
        )

    # ── Plot 3: Sparsity comparison ───────────────────────────────────────────
    print("  Plotting sparsity comparison...")
    plot_sparsity_comparison(
        lat_base, lab_base,
        lat_cross, lab_cross,
        lat_adapted, lab_adapted,
        classnames, ds_name,
        out_path=os.path.join(out_dir_ds, "sparsity_comparison.png"),
    )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del bb_base, bb_lora, base_sae, ad_sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  Summary for {ds_name}:")
    print(
        f"    Silhouette:  base={sil_base:.3f}  cross={sil_cross:.3f}  adapted={sil_adapted:.3f}"
    )
    print(
        f"      Δcross-base={sil_cross - sil_base:+.3f}  Δadapted-base={sil_adapted - sil_base:+.3f}"
    )
    print(f"    Mean L₀:     base={l0_base:.0f}  cross={l0_cross:.0f}  adapted={l0_adapted:.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="t-SNE of SAE latent activations — base, cross, adapted"
    )
    parser.add_argument("--datasets", nargs="+",
                        default=list(ADSAE_CONFIGS.keys()),
                        choices=list(ADSAE_CONFIGS.keys()))
    parser.add_argument("--max_images", type=int, default=2000)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="out/latent_tsne")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    registry = load_registry()
    for ds in args.datasets:
        process_dataset(ds, args, registry, device)

    print(f"\n Done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()

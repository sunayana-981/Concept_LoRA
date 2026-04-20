#!/usr/bin/env python3
"""
Batch demo: run DINOv2-base SAE on 100 ImageNet images and produce:

  1. summary_grid.png       — 100 images, each overlaid with top-1 SAE feature seg mask
  2. neuron_gallery.png     — top recurring SAE neurons across the 100 images,
                              showing their top reference ImageNet images
  3. results.json           — per-image top neurons + activation values

Usage:
  cd /home/sunayana/Documents/Concept_LoRA/patchsae
  python batch_demo_dino_sae.py [--n_images 100] [--top_k 5] [--seed 42]
"""

import argparse, json, math, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# ── Config ────────────────────────────────────────────────────────────────────

SAE_PATH        = "out/checkpoints/4s0816zv/final_sparse_autoencoder_facebook/dinov2-base_-1_resid_49152.pt"
FEAT_DATA_DIR   = "out/feature_data/dinov2_base/imagenet"
CLASSNAMES_PATH = "configs/classnames/imagenet_classnames.txt"
OUT_DIR         = "out/batch_demo_dino"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE   = 14
IMAGE_SIZE   = 224
PATCHES_SIDE = IMAGE_SIZE // PATCH_SIZE  # 16
N_PATCHES    = PATCHES_SIDE ** 2         # 256

os.makedirs(OUT_DIR, exist_ok=True)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_classnames(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_sae(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    sae  = SparseAutoencoder(cfg, device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(device)
    return sae


def load_dino(device):
    proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").eval().to(device)
    return model, proc


def load_artifacts(feat_dir):
    max_idx  = torch.load(os.path.join(feat_dir, "max_activating_image_indices.pt"), map_location="cpu")
    max_val  = torch.load(os.path.join(feat_dir, "max_activating_image_values.pt"),  map_location="cpu")
    mean_act = torch.load(os.path.join(feat_dir, "sae_mean_acts.pt"), map_location="cpu").numpy()
    return max_idx, max_val, mean_act


# ── Core inference ────────────────────────────────────────────────────────────

def get_sae_acts(image_pil, dino, proc, sae, device):
    """Run DINOv2 + SAE on a PIL image. Returns [257, N_FEAT] numpy array."""
    pv = proc(images=image_pil.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB"),
               return_tensors="pt")["pixel_values"].to(device)
    layer = dino.encoder.layer[11]
    captured = {}
    def _hook(_m, _i, out):
        captured["hs"] = (out[0] if isinstance(out, tuple) else out).detach().float()
    h = layer.register_forward_hook(_hook)
    with torch.no_grad():
        dino(pixel_values=pv)
    h.remove()
    hs = captured["hs"][0]  # [257, 768]
    with torch.no_grad():
        _, cache = sae.run_with_cache(hs)
    return cache["hook_hidden_post"].cpu().numpy()  # [257, N_FEAT]


def get_seg_mask(acts_2d, feat_idx):
    """
    Upsample patch activations for feat_idx to IMAGE_SIZE×IMAGE_SIZE.
    acts_2d: [257, N_FEAT] numpy
    Returns: [IMAGE_SIZE, IMAGE_SIZE] numpy float in [0,1]
    """
    patch_acts = acts_2d[1:, feat_idx].reshape(PATCHES_SIDE, PATCHES_SIDE)
    t = torch.tensor(patch_acts).unsqueeze(0).unsqueeze(0).float()
    mask = F.interpolate(t, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                         align_corners=False)[0, 0].numpy()
    mn, mx = mask.min(), mask.max()
    return (mask - mn) / (mx - mn + 1e-10)


def overlay_mask(img_pil, mask):
    """Darken non-activating regions; return RGBA PIL image."""
    img = np.array(img_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    darkened = (img * 0.12).astype(np.uint8)
    result = img.copy()
    result[mask < 0.15] = darkened[mask < 0.15]
    return Image.fromarray(result)


def top_neurons(acts_2d, noisy_thr, k=5, use_cls=True):
    """
    Return top-k non-noisy SAE feature indices for this image.
    use_cls=True  → CLS token (index 0)
    use_cls=False → mean over patch tokens (1..256)
    """
    vec = acts_2d[0] if use_cls else acts_2d[1:].mean(0)
    noisy = acts_2d[0]  # not used for filter; we filter on global mean
    vec = vec.copy()
    # zero noisy features using global mean_acts threshold (passed as scalar)
    return np.argsort(vec)[::-1][:k].tolist()


# ── Plot 1: summary grid ──────────────────────────────────────────────────────

def plot_summary_grid(records, classnames, out_path, ncols=10):
    """
    Grid of n_images cells. Each cell: original image overlaid with top-1 feature seg mask.
    Caption: class name | top feature id | act value.
    """
    n = len(records)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 2.0))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, rec in enumerate(records):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.imshow(rec["overlay"])
        ax.axis("off")
        cls = classnames[rec["label"]] if rec["label"] >= 0 else "?"
        cls_short = cls[:14] + "…" if len(cls) > 15 else cls
        ax.set_title(f"{cls_short}\nF{rec['top_feat']} act={rec['top_act']:.1f}",
                     fontsize=5, pad=2)

    # hide unused cells
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle("DINOv2 SAE — top-1 feature segmentation mask per image\n"
                 "(brighter = higher activation for top SAE feature)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 2: neuron gallery ────────────────────────────────────────────────────

def plot_neuron_gallery(neuron_counter, max_idx, max_val, ref_dataset,
                        classnames, out_path, top_n_neurons=12, n_ref=8):
    """
    Top-N most-fired SAE neurons across the 100 images.
    Each row: [feature info | top reference images from ImageNet].
    """
    top_neurons_list = [feat for feat, _ in neuron_counter.most_common(top_n_neurons)]

    n_imgs = min(n_ref, max_idx.shape[1])
    fig = plt.figure(figsize=(n_imgs * 1.9 + 2.5, top_n_neurons * 2.2 + 0.8))
    gs  = gridspec.GridSpec(
        top_n_neurons, n_imgs + 1,
        figure=fig,
        width_ratios=[2.0] + [1.0] * n_imgs,
        hspace=0.55, wspace=0.06,
        left=0.01, right=0.99, top=0.93, bottom=0.03,
    )
    palette = plt.cm.tab10.colors

    for row, feat_id in enumerate(top_neurons_list):
        color = palette[row % len(palette)]
        count = neuron_counter[feat_id]

        # Info panel
        ax_info = fig.add_subplot(gs[row, 0])
        ax_info.axis("off")
        ax_info.add_patch(plt.Rectangle(
            (0, 0.78), 1, 0.22, transform=ax_info.transAxes,
            color=color, alpha=0.85, clip_on=False,
        ))
        ax_info.text(0.5, 0.89, f"Feature {feat_id}",
                     transform=ax_info.transAxes,
                     ha="center", va="center", fontsize=8,
                     fontweight="bold", color="white")
        ax_info.text(0.05, 0.73, f"top-1 in {count} images",
                     transform=ax_info.transAxes,
                     ha="left", va="top", fontsize=7, color="#333")

        # Top reference images
        for col in range(n_imgs):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.axis("off")
            gi  = max_idx[feat_id, col].item()
            val = max_val[feat_id, col].item()
            if gi == 0 and val == 0:
                continue
            try:
                item = ref_dataset[int(gi)]
                img  = item["image"].convert("RGB").resize((160, 160))
                lbl  = item.get("label", -1)
                cname = classnames[lbl][:14] if 0 <= lbl < len(classnames) else ""
                ax.imshow(img, aspect="auto")
                for sp in ax.spines.values():
                    sp.set_edgecolor(color); sp.set_linewidth(2)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(f"{cname}\n{val:.1f}", fontsize=5.5, labelpad=2, color="#333")
            except Exception:
                ax.text(0.5, 0.5, "err", ha="center", va="center", fontsize=7)

    fig.suptitle("Most-activated SAE features across 100 ImageNet images\n"
                 "— top reference images (from pre-computed max-activating index) —",
                 fontsize=9, fontweight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 3: per-image top neurons strip ───────────────────────────────────────

def plot_per_image_strip(records, classnames, out_path, top_k=3):
    """
    For each image: show original + top-K seg masks side by side.
    n_images rows × (top_k+1) columns.
    """
    n = len(records)
    ncols = top_k + 1
    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 2.0, n * 1.8))
    if n == 1:
        axes = axes[np.newaxis, :]

    for r, rec in enumerate(records):
        # Original
        ax0 = axes[r, 0]
        ax0.imshow(rec["image"])
        ax0.axis("off")
        cls = classnames[rec["label"]] if rec["label"] >= 0 else "?"
        ax0.set_title(cls[:18], fontsize=6, pad=2)

        for k in range(top_k):
            ax = axes[r, k + 1]
            if k < len(rec["top_feats"]):
                feat_id = rec["top_feats"][k]
                mask = get_seg_mask(rec["acts"], feat_id)
                ov = overlay_mask(rec["image_pil"], mask)
                ax.imshow(ov)
                ax.axis("off")
                ax.set_title(f"F{feat_id}\nact={rec['top_acts'][k]:.1f}", fontsize=5.5, pad=2)
            else:
                ax.axis("off")

    col_labels = ["Original"] + [f"Top-{k+1} feature" for k in range(top_k)]
    for c, lbl in enumerate(col_labels):
        axes[0, c].set_title(lbl, fontsize=7, fontweight="bold", pad=10)

    fig.suptitle("Per-image top SAE features (DINOv2-base, CLS token)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=100)
    parser.add_argument("--top_k",    type=int, default=5,
                        help="Top SAE features per image to record")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--split",    default="train",
                        help="ImageNet split to sample from (train or validation)")
    parser.add_argument("--strip_rows", type=int, default=20,
                        help="How many images to include in the per-image strip plot")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading models and artifacts...")
    classnames = load_classnames(CLASSNAMES_PATH)
    sae        = load_sae(SAE_PATH, DEVICE)
    dino, proc = load_dino(DEVICE)
    max_idx, max_val, mean_act_np = load_artifacts(FEAT_DATA_DIR)

    # Adaptive noisy threshold (95th pct of alive features)
    alive_mean = mean_act_np[mean_act_np > 0]
    noisy_thr  = float(np.percentile(alive_mean, 95)) if len(alive_mean) > 0 else 0.1
    print(f"  Noisy threshold (adaptive 95th-pct): {noisy_thr:.4f}")

    print(f"Loading ImageNet {args.split}...")
    dataset = load_dataset("evanarlian/imagenet_1k_resized_256",
                           split=args.split, trust_remote_code=True)
    indices = rng.choice(len(dataset), size=min(args.n_images, len(dataset)), replace=False)
    indices = sorted(indices.tolist())
    subset  = dataset.select(indices)

    print(f"Processing {len(subset)} images...")
    records        = []
    neuron_counter = Counter()

    for i, item in enumerate(tqdm(subset, desc="Inference")):
        img_pil = item["image"].convert("RGB")
        label   = item.get("label", -1)

        acts = get_sae_acts(img_pil, dino, proc, sae, DEVICE)  # [257, N_FEAT]

        # CLS token activations; zero noisy features
        cls_vec = acts[0].copy()
        cls_vec[mean_act_np > noisy_thr] = 0.0

        sorted_feats = np.argsort(cls_vec)[::-1]
        top_feats = sorted_feats[:args.top_k].tolist()
        top_acts  = [float(cls_vec[f]) for f in top_feats]

        top1_feat = top_feats[0] if top_feats else -1
        top1_act  = top_acts[0]  if top_acts  else 0.0

        # Seg mask for top-1 feature (for summary grid)
        if top1_feat >= 0 and top1_act > 0:
            mask = get_seg_mask(acts, top1_feat)
            overlay = overlay_mask(img_pil, mask)
        else:
            overlay = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))

        neuron_counter.update(top_feats)

        records.append({
            "idx":       int(indices[i]),
            "label":     int(label),
            "top_feat":  top1_feat,
            "top_act":   top1_act,
            "top_feats": top_feats,
            "top_acts":  top_acts,
            "image":     img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)),
            "image_pil": img_pil,
            "overlay":   overlay,
            "acts":      acts,
        })

    # Save JSON summary (serialisable fields only)
    json_records = [
        {"idx": r["idx"], "label": r["label"],
         "classname": classnames[r["label"]] if 0 <= r["label"] < len(classnames) else "",
         "top_feats": r["top_feats"], "top_acts": r["top_acts"]}
        for r in records
    ]
    json_path = os.path.join(OUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(json_records, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\nGenerating plots...")

    # 1. Summary grid (all images, top-1 seg mask overlay)
    plot_summary_grid(records, classnames,
                      os.path.join(OUT_DIR, "summary_grid.png"))

    # 2. Neuron gallery (most-fired neurons across the 100 images + ref images)
    print("Loading full ImageNet for reference image retrieval...")
    ref_dataset = load_dataset("evanarlian/imagenet_1k_resized_256",
                               split=args.split, trust_remote_code=True)
    plot_neuron_gallery(neuron_counter, max_idx, max_val, ref_dataset,
                        classnames, os.path.join(OUT_DIR, "neuron_gallery.png"))

    # 3. Per-image strip (first strip_rows images, top-3 seg masks each)
    strip_records = records[:args.strip_rows]
    plot_per_image_strip(strip_records, classnames,
                         os.path.join(OUT_DIR, "per_image_strip.png"), top_k=3)

    print(f"\nAll outputs → {OUT_DIR}/")
    print(f"  summary_grid.png   — {len(records)} images × top-1 seg mask")
    print(f"  neuron_gallery.png — top-12 neurons + reference images")
    print(f"  per_image_strip.png— first {args.strip_rows} images × top-3 masks")
    print(f"  results.json       — per-image top features")


if __name__ == "__main__":
    main()

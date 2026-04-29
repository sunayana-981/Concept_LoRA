#!/usr/bin/env python3
"""
Visualization: Top-M images × Top-N neurons across three SAE conditions.

Viz 1 — tSNE of SAE latent activations, colored by condition:
           • ZS + BSAE        (zero-shot CLIP  + base ImageNet SAE)
           • Adapted + BAE    (LoRA CLIP        + base SAE = Cross-SAE)
           • Adapted + AdSAE  (LoRA CLIP        + domain-adapted SAE)
         Shows that AdSAE clusters the same images more tightly by concept.

Viz 2 — Image grid of the actual top-activating images per neuron × condition.
         Shows which images each SAE "sees" as maximally relevant per concept.

Usage:
    python visualize_neuron_tsne.py \
        --datasets caltech101 eurosat medmnist dtd ucf101 \
        --top_n_neurons 8 --top_m_images 6 \
        --out_dir out/neuron_viz
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.registry import get_backbone
from src.data.dataset_registry import get_dataset, get_classnames, get_label_key, load_registry
from src.sae_training.loaders import load_sae, load_lora_weights
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# ── Dataset / SAE configuration ───────────────────────────────────────────────

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
        "zs_acc": 92.25, "cross_acc": 83.03, "adsae_acc": 88.05,
    },
    "eurosat": {
        "sae_path": os.path.join(
            _ROOT, "out", "checkpoints", "eurosat", "m8tn3p5s",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-1_resid_49152.pt"),
        "layer": -1,
        "lora_path": os.path.join(LORA_ROOT, "eurosat", "16shots", "seed1", "lora_weights.pt"),
        "zs_acc": 42.04, "cross_acc": 41.55, "adsae_acc": 83.71,
    },
    "medmnist": {
        "sae_path": os.path.join(
            _ROOT, "out", "checkpoints", "medmnist", "d2ygd3bb",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "medmnist", "16shots", "seed1", "lora_weights.pt"),
        "zs_acc": 19.8, "cross_acc": 46.7, "adsae_acc": 90.75,
    },
    "dtd": {
        "sae_path": os.path.join(
            _ROOT, "out", "checkpoints", "dtd", "sd5h6hxv",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "dtd", "16shots", "seed42", "lora_weights.pt"),
        "zs_acc": 43.6, "cross_acc": 69.07, "adsae_acc": 53.2,
    },
    "ucf101": {
        "sae_path": os.path.join(
            _ROOT, "out", "checkpoints", "ucf101", "j04tcnkc",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "ucf101", "16shots", "seed1", "lora_weights.pt"),
        "zs_acc": 64.23, "cross_acc": 42.55, "adsae_acc": 78.2,
    },
    "cub2002011": {
        "sae_path": os.path.join(
            _ROOT, "out", "checkpoints", "cub2002011", "578p9z8f",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "cub2002011", "16shots", "seed1", "lora_weights.pt"),
        "zs_acc": None, "cross_acc": None, "adsae_acc": None,
    },
}

CONDITION_NAMES   = ["ZS + BSAE", "Adapted + BAE\n(Cross-SAE)", "Adapted + AdSAE"]
CONDITION_SHORT   = ["ZS+BSAE", "Cross-SAE", "AdSAE"]
CONDITION_COLORS  = ["#4878D0", "#EE854A", "#6ACC65"]
CONDITION_MARKERS = ["o", "s", "^"]
NEURON_CMAP       = plt.cm.tab20


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_base_backbone(device: str):
    bb = get_backbone(BACKBONE_ID, device=device).load()
    bb.model.eval()
    return bb


def load_lora_backbone(lora_path: str, device: str):
    bb = get_backbone(BACKBONE_ID, device=device).load()

    class _W:
        def __init__(self, m): self.model = m

    load_lora_weights(_W(bb.model), lora_path, device)
    bb.model.eval()
    return bb


# ── Feature extraction ────────────────────────────────────────────────────────

def _collate(processor, label_key: str):
    def fn(batch):
        imgs   = [item["image"].convert("RGB") for item in batch]
        labels = [item[label_key] for item in batch]
        pv     = processor(images=imgs, return_tensors="pt")["pixel_values"]
        return pv, torch.tensor(labels, dtype=torch.long)
    return fn


@torch.no_grad()
def extract_sae_latents(backbone, sae, dataset, label_key, layer,
                         batch_size=64, device="cuda",
                         num_workers=4, max_images=None):
    """Run full dataset; return (latents [N,d_sae], labels [N], indices [N])."""
    if max_images:
        dataset = dataset.select(range(min(max_images, len(dataset))))

    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=(device == "cuda"),
                           collate_fn=_collate(backbone.processor, label_key))
    layer_idx = backbone.resolve_layer(layer)
    cap       = {}

    def hook(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        cap["cls"] = hs.detach().float()[:, 0, :]

    h       = backbone.model.vision_model.encoder.layers[layer_idx].register_forward_hook(hook)
    lats, lbls, idxs = [], [], []
    offset  = 0
    for pv, labels in tqdm(loader, desc="  extract", leave=False):
        backbone.model.vision_model(pixel_values=pv.to(device))
        _, lat, _ = sae(cap["cls"])
        B = lat.shape[0]
        lats.append(lat.cpu()); lbls.append(labels)
        idxs.append(torch.arange(offset, offset + B))
        offset += B
    h.remove()
    return torch.cat(lats), torch.cat(lbls), torch.cat(idxs)


@torch.no_grad()
def extract_for_indices(backbone, sae, dataset, label_key, layer,
                         indices, batch_size=64, device="cuda"):
    """Extract SAE latents for specific image indices."""
    sub     = dataset.select(indices)
    loader  = DataLoader(sub, batch_size=batch_size, shuffle=False,
                         num_workers=0,
                         collate_fn=_collate(backbone.processor, label_key))
    layer_idx = backbone.resolve_layer(layer)
    cap       = {}

    def hook(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        cap["cls"] = hs.detach().float()[:, 0, :]

    h    = backbone.model.vision_model.encoder.layers[layer_idx].register_forward_hook(hook)
    lats = []
    for pv, _ in loader:
        backbone.model.vision_model(pixel_values=pv.to(device))
        _, lat, _ = sae(cap["cls"])
        lats.append(lat.cpu())
    h.remove()
    return torch.cat(lats)


# ── tSNE ──────────────────────────────────────────────────────────────────────

def run_tsne(latents: np.ndarray, perplexity=30, seed=42) -> np.ndarray:
    """TruncatedSVD(50) → tSNE(2). Returns [N, 2]."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE

    n, d        = latents.shape
    n_comp      = min(50, n - 1, d)
    reduced     = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(latents) \
                  if d > n_comp else latents
    perp        = min(perplexity, max(5, n // 4))
    return TSNE(n_components=2, perplexity=perp, random_state=seed,
                max_iter=1000, learning_rate="auto", init="pca").fit_transform(
                    reduced.astype(np.float32))


# ── Viz 1: tSNE ───────────────────────────────────────────────────────────────

def plot_tsne(embedding, condition_labels, neuron_labels,
              neuron_ids, ds_name, accs, classnames_per_neuron, out_path):
    """
    Left panel : tSNE scatter (condition = color, neuron = marker shape).
    Each neuron cluster centroid is annotated with its neuron ID + top class.
    Right panel: accuracy bar chart showing ZS / Cross-SAE / AdSAE.
    """
    n_neurons   = len(neuron_ids)
    n_imgs      = embedding.shape[0] // 3
    nc_colors   = [NEURON_CMAP(i / max(n_neurons - 1, 1)) for i in range(n_neurons)]

    fig = plt.figure(figsize=(17, 6.5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3.2, 1],
                            wspace=0.28, left=0.05, right=0.97,
                            top=0.88, bottom=0.10)

    # ── tSNE panel ────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])

    # Draw convex hull ellipses per (condition, neuron) cluster
    for ci in range(3):
        for ni in range(n_neurons):
            mask = (condition_labels == ci) & (neuron_labels == ni)
            pts  = embedding[mask]
            if len(pts) < 3:
                continue
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            rx = pts[:, 0].std() + 1e-3
            ry = pts[:, 1].std() + 1e-3
            ell = mpatches.Ellipse(
                (cx, cy), 2.8 * rx, 2.8 * ry,
                linewidth=1.2, edgecolor=CONDITION_COLORS[ci],
                facecolor=CONDITION_COLORS[ci], alpha=0.07, zorder=1,
            )
            ax.add_patch(ell)

    # Scatter points
    for ci, (cname, ccolor, cmark) in enumerate(
            zip(CONDITION_SHORT, CONDITION_COLORS, CONDITION_MARKERS)):
        mask = condition_labels == ci
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[nc_colors[neuron_labels[j]] for j in np.where(mask)[0]],
            marker=cmark, s=70, alpha=0.92, linewidths=0.6,
            edgecolors=ccolor, zorder=3,
        )

    ax.set_title(f"tSNE of SAE Latents — {ds_name}\n"
                 "Each point = one image; marker shape/edge = condition; fill color = neuron ID.",
                 fontsize=9, pad=6)
    ax.set_xlabel("tSNE dim 1", fontsize=9)
    ax.set_ylabel("tSNE dim 2", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    # ── Accuracy bar panel ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    labels = ["ZS\n(BSAE)", "Cross-\nSAE\n(BAE)", "Adapted\n(AdSAE)"]
    vals   = [accs.get("zs_acc"), accs.get("cross_acc"), accs.get("adsae_acc")]
    valid  = [v for v in vals if v is not None]

    bars = ax2.bar(
        range(len(labels)),
        [v if v is not None else 0 for v in vals],
        color=CONDITION_COLORS, edgecolor="white", width=0.6, zorder=3,
    )
    for bar, v in zip(bars, vals):
        if v is not None:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(valid) * 0.02,
                     f"{v:.1f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    # Annotate AdSAE gain over Cross-SAE
    if accs.get("cross_acc") and accs.get("adsae_acc"):
        gain = accs["adsae_acc"] - accs["cross_acc"]
        sign = "+" if gain >= 0 else ""
        ax2.annotate(
            f"AdSAE {sign}{gain:.1f}pp\nvs Cross-SAE",
            xy=(2, accs["adsae_acc"]), xytext=(1.5, max(valid) * 0.92),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
            fontsize=7.5, color="#225522", fontweight="bold",
            ha="center",
        )

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0, max(valid) * 1.22 if valid else 100)
    ax2.set_ylabel("Zero-shot Accuracy (%)", fontsize=9)
    ax2.set_title(f"Accuracy Ablation\n({ds_name})", fontsize=9, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)

    fig.suptitle(
        f"Neuron Representation Analysis  |  CLIP + LoRA + SAE  |  {ds_name}  "
        f"({n_imgs} images × 3 conditions, top-{n_neurons} AdSAE neurons)",
        fontsize=10, fontweight="bold", y=0.97,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Viz 1 → {out_path}")


# ── Viz 2: Image grid ─────────────────────────────────────────────────────────

def _pil(dataset, idx):
    item = dataset[int(idx)]
    img  = item["image"] if isinstance(item, dict) else item[0]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img.convert("RGB").resize((110, 110))


def plot_image_grid(dataset, top_per_neuron, latents_per_cond,
                    all_img_indices, neuron_ids, classnames,
                    label_key, ds_name, out_path, n_imgs=6):
    """
    Layout: n_neurons rows × (3 groups of n_imgs columns).
    Each row = one neuron.  Each group = one condition.
    Border color = condition.  Caption = class + activation value.
    Header row shows condition name + accuracy stats.
    """
    n_neurons = len(neuron_ids)
    n_conds   = 3
    grp       = min(n_imgs, 6)

    # column layout: [grp images | sep | grp images | sep | grp images]
    col_w_img  = 1.15
    col_w_sep  = 0.18
    col_w_lbl  = 1.3
    total_cols = n_conds * grp + (n_conds - 1)   # image cols + separators
    total_rows = n_neurons + 1                    # +1 for header row

    fig_w = col_w_lbl + total_cols * col_w_img + (n_conds - 1) * col_w_sep
    fig_h = total_rows * 1.45

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Build width_ratios: [label_col, cond0_imgs..., sep, cond1_imgs..., sep, cond2_imgs...]
    width_ratios = [col_w_lbl]
    for ci in range(n_conds):
        width_ratios += [col_w_img] * grp
        if ci < n_conds - 1:
            width_ratios.append(col_w_sep)

    gs = gridspec.GridSpec(
        total_rows, len(width_ratios),
        figure=fig, hspace=0.55, wspace=0.04,
        left=0.01, right=0.99, top=0.94, bottom=0.02,
        width_ratios=width_ratios,
    )

    # Column offsets per condition (accounting for label col + sep cols)
    # label=0, cond0=[1..grp], sep, cond1=[grp+2..2*grp+1], sep, cond2=[2*grp+3..3*grp+2]
    cond_starts = [1, grp + 2, 2 * grp + 3]

    local_map = {gidx: li for li, gidx in enumerate(all_img_indices)}

    # ── Header row ────────────────────────────────────────────────────────────
    for ci, (cshort, ccolor) in enumerate(zip(CONDITION_SHORT, CONDITION_COLORS)):
        x0   = cond_starts[ci]
        # Span the full group width
        ax_h = fig.add_subplot(gs[0, x0:x0 + grp])
        ax_h.set_facecolor(ccolor)
        ax_h.axis("off")
        acc_txt = ""
        key = ["zs_acc", "cross_acc", "adsae_acc"][ci]
        # (accs not directly available here; just show condition name)
        ax_h.text(0.5, 0.5, f"{cshort}", transform=ax_h.transAxes,
                  ha="center", va="center", fontsize=9, fontweight="bold",
                  color="white")

    ax_lbl0 = fig.add_subplot(gs[0, 0])
    ax_lbl0.axis("off")
    ax_lbl0.text(0.5, 0.5, "Neuron\n(top class)", transform=ax_lbl0.transAxes,
                 ha="center", va="center", fontsize=7.5, fontweight="bold")

    # ── Image rows ────────────────────────────────────────────────────────────
    for row, (ni_idx, nid) in enumerate(zip(range(n_neurons), neuron_ids), start=1):

        # Neuron label column
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.axis("off")
        top_cls = classnames[
            int(np.argmax(
                np.array([
                    dataset[int(gidx)][label_key]
                    for gidx in top_per_neuron[ni_idx][2][:4]   # AdSAE top images
                    if gidx < len(dataset)
                ] or [0])
            ))
        ] if len(classnames) > 0 else f"N{nid}"

        ax_lbl.text(0.5, 0.7, f"N{ni_idx+1}  (#{nid})",
                    transform=ax_lbl.transAxes,
                    ha="center", va="center", fontsize=7, fontweight="bold")
        ax_lbl.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.02), 0.90, 0.60,
            boxstyle="round,pad=0.02",
            facecolor=NEURON_CMAP(ni_idx / max(n_neurons - 1, 1)),
            alpha=0.25, transform=ax_lbl.transAxes,
        ))

        for ci, ccolor in enumerate(CONDITION_COLORS):
            x0         = cond_starts[ci]
            img_list   = top_per_neuron[ni_idx][ci]
            lat        = latents_per_cond[ci]   # [N_imgs, d_sae]

            for col_off in range(grp):
                ax = fig.add_subplot(gs[row, x0 + col_off])
                ax.axis("off")
                if col_off >= len(img_list):
                    ax.set_facecolor("#f5f5f5")
                    continue
                gidx = img_list[col_off]
                try:
                    pil  = _pil(dataset, gidx)
                    item = dataset[int(gidx)]
                    lbl  = item.get(label_key, -1) if isinstance(item, dict) else -1
                    cls_name = (classnames[lbl][:12] if 0 <= lbl < len(classnames) else "")

                    li   = local_map.get(gidx)
                    actv = float(lat[li, nid]) if li is not None else 0.0

                    ax.imshow(pil, aspect="auto")
                    for sp in ax.spines.values():
                        sp.set_visible(True)
                        sp.set_edgecolor(ccolor)
                        sp.set_linewidth(2.2)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_xlabel(f"{cls_name}\nact={actv:.1f}",
                                  fontsize=5.5, labelpad=1.5, color="#222")
                    if row == 1:
                        ax.set_title(f"#{col_off+1}", fontsize=5.5, pad=1.5,
                                     color=ccolor, fontweight="bold")
                except Exception:
                    ax.text(0.5, 0.5, "err", ha="center", va="center",
                            fontsize=7, color="gray")

    fig.suptitle(
        f"Top-{grp} Activating Images per Neuron  |  {ds_name}\n"
        "Rows = top neurons (selected by AdSAE).  "
        "Columns grouped by condition.  "
        "Border color = condition.  "
        "AdSAE neurons capture domain-specific concepts missed by Cross-SAE.",
        fontsize=8.5, fontweight="bold", y=0.98,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Viz 2 → {out_path}")


# ── Per-dataset pipeline ──────────────────────────────────────────────────────

def process_dataset(ds_name, args, registry, device):
    if ds_name not in ADSAE_CONFIGS:
        print(f"[WARN] No config for {ds_name}, skipping.")
        return

    cfg       = ADSAE_CONFIGS[ds_name]
    out_dir_ds = os.path.join(args.out_dir, ds_name)
    os.makedirs(out_dir_ds, exist_ok=True)

    print(f"\n{'═'*65}\n  Dataset: {ds_name.upper()}\n{'═'*65}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset    = get_dataset(ds_name, registry=registry, max_samples=args.max_dataset_images)
    classnames = get_classnames(ds_name, dataset=dataset, registry=registry)
    label_key  = get_label_key(ds_name, dataset, registry=registry)
    print(f"  {len(dataset):,} images, {len(classnames)} classes")

    # ── SAEs ──────────────────────────────────────────────────────────────────
    print("  Loading SAEs...")
    if not os.path.isfile(cfg["sae_path"]):
        print(f"  [SKIP] AdSAE not found: {cfg['sae_path']}"); return
    base_sae, _ = load_sae(BASE_SAE_PATH, device); base_sae.eval()
    ad_sae,   _ = load_sae(cfg["sae_path"],   device); ad_sae.eval()

    # ── Backbones ─────────────────────────────────────────────────────────────
    print("  Loading ZS backbone..."); bb_zs   = load_base_backbone(device)
    if not os.path.isfile(cfg["lora_path"]):
        print(f"  [SKIP] LoRA weights not found: {cfg['lora_path']}"); return
    print("  Loading LoRA backbone..."); bb_lora = load_lora_backbone(cfg["lora_path"], device)

    conditions = [
        (bb_zs,   base_sae, BASE_SAE_LAYER, "ZS + BSAE"),
        (bb_lora, base_sae, BASE_SAE_LAYER, "Adapted + BAE"),
        (bb_lora, ad_sae,   cfg["layer"],   "Adapted + AdSAE"),
    ]

    # ── 1. AdSAE pass over full dataset ───────────────────────────────────────
    print(f"\n[1] AdSAE pass ({len(dataset)} images)...")
    bb_ad, sae_ad, layer_ad, _ = conditions[2]
    latents_ad, labels_ad, idx_ad = extract_sae_latents(
        bb_ad, sae_ad, dataset, label_key, layer_ad,
        batch_size=args.batch_size, device=device, num_workers=args.num_workers,
    )

    # ── 2. Top-N neurons ──────────────────────────────────────────────────────
    print(f"\n[2] Selecting top-{args.top_n_neurons} neurons...")
    firing_rate = (latents_ad > 0).float().mean(dim=0)
    mean_act    = latents_ad.mean(dim=0)
    alive       = (firing_rate > 5e-4) & (firing_rate < 0.5)
    scores      = mean_act.clone(); scores[~alive] = -1.0
    top_nids    = torch.argsort(scores, descending=True)[:args.top_n_neurons].tolist()
    print(f"    neuron IDs: {top_nids}")

    # ── 3. Top-M images per neuron (AdSAE) ────────────────────────────────────
    print(f"\n[3] Top-{args.top_m_images} images per neuron...")
    top_adsae: List[List[int]] = []
    for nid in top_nids:
        vals = latents_ad[:, nid]
        k    = min(args.top_m_images, int((vals > 0).sum().item()))
        if k == 0:
            top_adsae.append([]); continue
        top_adsae.append(idx_ad[torch.argsort(vals, descending=True)[:k]].tolist())

    all_img_idx = sorted(set(i for imgs in top_adsae for i in imgs))
    print(f"    unique images: {len(all_img_idx)}")
    if not all_img_idx:
        print("  [SKIP] no images selected"); return

    # ── 4. Latents for selected images under all 3 conditions ─────────────────
    print("\n[4] Extracting latents for selected images (3 conditions)...")
    lats_per_cond: List[torch.Tensor] = []
    for bb, sae, layer, cname in conditions:
        print(f"    {cname}...")
        lats_per_cond.append(extract_for_indices(
            bb, sae, dataset, label_key, layer,
            indices=all_img_idx, batch_size=args.batch_size, device=device,
        ))

    # ── 5. tSNE ───────────────────────────────────────────────────────────────
    print("\n[5] Running tSNE...")
    n_imgs    = len(all_img_idx)
    lat_np    = np.stack([l.numpy() for l in lats_per_cond], axis=1)  # [N,3,d]
    lat_flat  = lat_np.reshape(n_imgs * 3, -1)    # interleave: [img0c0,img0c1,img0c2,img1c0,...]
    cond_lbl  = np.tile(np.arange(3), n_imgs)

    img2neuron  = {gidx: ni for ni, imgs in enumerate(top_adsae) for gidx in imgs}
    neuron_lbl  = np.tile(
        np.array([img2neuron.get(gidx, 0) for gidx in all_img_idx]), 3
    )
    embedding   = run_tsne(lat_flat, perplexity=args.tsne_perplexity)

    # Per-neuron top class (from AdSAE top images)
    classnames_per_neuron = []
    for ni, imgs in enumerate(top_adsae):
        lbls = [dataset[int(g)].get(label_key, 0) for g in imgs[:4] if g < len(dataset)]
        if lbls:
            top_cls = classnames[int(np.bincount(lbls).argmax())] if classnames else f"N{top_nids[ni]}"
        else:
            top_cls = f"N{top_nids[ni]}"
        classnames_per_neuron.append(top_cls)

    # ── 6. Viz 1: tSNE ────────────────────────────────────────────────────────
    print("\n[6] Plotting Viz 1 (tSNE)...")
    plot_tsne(
        embedding, cond_lbl, neuron_lbl,
        neuron_ids=top_nids, ds_name=ds_name, accs=cfg,
        classnames_per_neuron=classnames_per_neuron,
        out_path=os.path.join(out_dir_ds, "tsne_neurons.png"),
    )

    # ── 7. Viz 2: image grid ──────────────────────────────────────────────────
    print("\n[7] Plotting Viz 2 (image grid)...")
    # For each condition, re-rank the same image pool by activation on each neuron
    top_per_neuron_all = []
    local_map = {gidx: li for li, gidx in enumerate(all_img_idx)}
    for ni_idx, nid in enumerate(top_nids):
        row = []
        for ci, lat in enumerate(lats_per_cond):
            vals     = lat[:, nid]
            k        = min(args.top_m_images, int((vals > 0).sum().item()))
            top_loc  = torch.argsort(vals, descending=True)[:max(k, 1)].tolist()
            row.append([all_img_idx[li] for li in top_loc])
        top_per_neuron_all.append(row)

    plot_image_grid(
        dataset, top_per_neuron_all, lats_per_cond,
        all_img_idx, top_nids, classnames, label_key,
        ds_name, os.path.join(out_dir_ds, "top_images_grid.png"),
        n_imgs=min(args.top_m_images, 6),
    )

    del bb_zs, bb_lora, base_sae, ad_sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neuron tSNE + image grid visualization")
    parser.add_argument("--datasets",            nargs="+",
                        default=list(ADSAE_CONFIGS.keys()),
                        choices=list(ADSAE_CONFIGS.keys()))
    parser.add_argument("--top_n_neurons",       type=int, default=8)
    parser.add_argument("--top_m_images",        type=int, default=6)
    parser.add_argument("--max_dataset_images",  type=int, default=2000)
    parser.add_argument("--batch_size",          type=int, default=64)
    parser.add_argument("--num_workers",         type=int, default=4)
    parser.add_argument("--tsne_perplexity",     type=int, default=30)
    parser.add_argument("--device",              default="cuda")
    parser.add_argument("--out_dir",             default="out/neuron_viz")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU."); device = "cpu"

    registry = load_registry()
    for ds in args.datasets:
        process_dataset(ds, args, registry, device)

    print("\n[INFO] Done. Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()

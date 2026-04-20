#!/usr/bin/env python3
"""
ALIGN (kakaobrain/align-base) SAE evaluation on ImageNet-1k.

Follows the same pipeline as eval_dino_sae_imagenet.py:
  - max_activating_image_indices, sae_mean_acts, sae_sparsity
  - cls_sae_cnt  → monosemanticity (selectivity, top1_fraction)
  - Top-K feature visualisation plots

Key difference from DINOv2 eval:
  ALIGN uses EfficientNet-B7, NOT a ViT.
  The "CLS-equivalent" activation is global-average-pool of the last block:
      hook → [B, 640, 10, 10]  →  GAP  →  [B, 640]

Usage:
    cd /home/sunayana/Documents/Concept_LoRA/patchsae
    python -u eval_align_sae_imagenet.py 2>&1 | tee out/eval_align_imagenet.log
    python -u eval_align_sae_imagenet.py --replot   # regenerate plots only
"""

import json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AlignModel, AlignProcessor

sys.path.insert(0, os.path.dirname(__file__))
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# ── Config ────────────────────────────────────────────────────────────────────

SAE_PATH         = "out/checkpoints/wy7pa30o/final_sparse_autoencoder_kakaobrain/align-base_-1_resid_20480.pt"
CLASSNAMES_PATH  = "configs/classnames/imagenet_classnames.txt"
FEATURE_DATA_DIR = "out/feature_data/align_base/imagenet"
OUT_DIR          = "out/eval_align_imagenet"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH            = 32        # 346×346 images — keep lower than DINOv2's 64
N_TOP_IMAGES     = 10
NOISY_THRESHOLD  = 0.1      # CLIP-calibrated default; overridden adaptively
ACT_THRESHOLD    = 0.2      # binarisation threshold for cls_sae_cnt
MAX_IMAGES       = 100_000  # None = full 1.28M
N_VIS_FEATURES   = 16
N_VIS_IMAGES     = 8
N_BLOCKS         = 55       # total AlignVisionBlocks in kakaobrain/align-base

os.makedirs(FEATURE_DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_classnames(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_sae(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    sae  = SparseAutoencoder(cfg, device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(device)
    return sae, cfg


def load_align(device):
    proc  = AlignProcessor.from_pretrained("kakaobrain/align-base")
    model = AlignModel.from_pretrained("kakaobrain/align-base").eval().to(device)
    return model, proc


# ── Hook ─────────────────────────────────────────────────────────────────────

class BlockCapture:
    """Captures AlignVisionBlock output and applies GAP → [B, C]."""
    def __init__(self):
        self.output = None   # [B, C] after GAP
        self._h = None

    def register(self, block):
        def _hook(_m, _i, out):
            # EfficientNet block → [B, C, H, W]; GAP → [B, C]
            feat = out.detach().float()
            self.output = feat.mean(dim=[-2, -1])
        self._h = block.register_forward_hook(_hook)

    def remove(self):
        if self._h:
            self._h.remove()
            self._h = None


# ── Collate ───────────────────────────────────────────────────────────────────

def make_collate(proc):
    def collate(batch):
        imgs, labels = [], []
        for item in batch:
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(img)
            labels.append(item["label"])
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"]
        return pv, torch.tensor(labels, dtype=torch.long)
    return collate


# ── Rolling top-K ─────────────────────────────────────────────────────────────

def get_new_top_k(first_values, first_indices, second_values, second_indices, k):
    total_values  = torch.cat([first_values,  second_values],  dim=1)
    total_indices = torch.cat([first_indices, second_indices], dim=1)
    new_values, idx_of_idx = torch.topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, idx_of_idx)
    return new_values, new_indices


# ── Main eval pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(align, sae, proc, dataset, block_idx, classnames):
    N_FEAT    = sae.W_enc.shape[1]
    N_CLASSES = len(classnames)
    collate   = make_collate(proc)
    loader    = DataLoader(
        dataset, batch_size=BATCH, shuffle=False,
        num_workers=4, pin_memory=(DEVICE == "cuda"),
        collate_fn=collate,
    )

    cap = BlockCapture()
    cap.register(align.vision_model.encoder.blocks[block_idx])

    # Storage
    max_values  = torch.zeros(N_FEAT, N_TOP_IMAGES)
    max_indices = torch.zeros(N_FEAT, N_TOP_IMAGES, dtype=torch.long)
    total_acts  = torch.zeros(N_FEAT)
    sparsity    = torch.zeros(N_FEAT)
    cls_cnt     = torch.zeros(N_CLASSES, N_FEAT)

    total_l0  = 0.0
    total_mse = 0.0
    total_ev  = 0.0
    total_n   = 0
    images_processed = 0

    for pixel_values, labels in tqdm(loader, desc="  eval", leave=True):
        pixel_values = pixel_values.to(DEVICE)
        labels_cpu   = labels
        B = pixel_values.shape[0]

        # ALIGN vision forward — hook captures GAP'd features
        align.vision_model(pixel_values=pixel_values)
        gap_act = cap.output  # [B, 640]

        # SAE forward
        sae_out, feat_acts, _ = sae(gap_act)
        feat_acts_cpu = feat_acts.detach().cpu()

        # Global metrics
        total_l0  += (feat_acts_cpu > 0).float().sum(-1).mean().item() * B
        mse_s      = (sae_out.cpu() - gap_act.cpu()).pow(2).sum(-1)
        var_s      = gap_act.cpu().pow(2).sum(-1).clamp(min=1e-8)
        total_mse += mse_s.mean().item() * B
        total_ev  += (1 - mse_s / var_s).mean().item() * B
        total_n   += B

        # sae_mean_acts & sparsity
        sae_acts_T = feat_acts_cpu.T                          # [N_FEAT, B]
        total_acts += sae_acts_T.sum(dim=1)
        sparsity   += (sae_acts_T > 0).sum(dim=1).float()

        # Rolling top-K images per feature
        k_batch = min(N_TOP_IMAGES, B)
        batch_vals, batch_local_idx = torch.topk(sae_acts_T, k=k_batch, dim=1)
        batch_global_idx = batch_local_idx + images_processed
        max_values, max_indices = get_new_top_k(
            max_values, max_indices, batch_vals, batch_global_idx, k=N_TOP_IMAGES,
        )

        # Class-wise activation counts (binary threshold)
        binary  = (feat_acts_cpu > ACT_THRESHOLD).float()
        one_hot = F.one_hot(labels_cpu, N_CLASSES).float()
        cls_cnt += one_hot.T @ binary

        images_processed += B

    cap.remove()

    # Finalise
    fired_mask    = sparsity > 0
    sae_mean_acts = torch.zeros(N_FEAT)
    sae_mean_acts[fired_mask] = total_acts[fired_mask] / sparsity[fired_mask]
    sae_sparsity  = sparsity / max(images_processed, 1)

    # Monosemanticity
    cls_cnt_np  = cls_cnt.numpy()
    feat_total  = cls_cnt_np.sum(0)
    dead_mask   = feat_total == 0
    alive_feats = np.where(~dead_mask)[0]

    top1_fraction = np.zeros(N_FEAT)
    selectivity   = np.zeros(N_FEAT)

    for f in alive_feats:
        cnt   = cls_cnt_np[:, f]
        total = cnt.sum()
        p     = cnt / total
        top1_fraction[f] = p.max()
        H_norm           = -(p[p > 0] * np.log(p[p > 0])).sum() / math.log(N_CLASSES)
        selectivity[f]   = 1.0 - H_norm

    # Adaptive noisy threshold: 95th percentile of alive mean_acts
    alive_mean = sae_mean_acts.numpy()[~dead_mask]
    adaptive_noisy_thr = float(np.percentile(alive_mean, 95)) if len(alive_mean) > 0 else NOISY_THRESHOLD
    noisy_mask  = (sae_mean_acts.numpy() > adaptive_noisy_thr) & ~dead_mask
    alive_mask  = ~dead_mask
    usable_mask = alive_mask & ~noisy_mask

    n_dead, n_alive = dead_mask.sum(), alive_mask.sum()
    n_noisy, n_usable = noisy_mask.sum(), usable_mask.sum()

    alive_sel   = selectivity[alive_mask]
    alive_top1  = top1_fraction[alive_mask]
    usable_sel  = selectivity[usable_mask]

    metrics = {
        "n_images":               int(images_processed),
        "l0":                     total_l0 / total_n,
        "recon_mse":              total_mse / total_n,
        "explained_variance":     total_ev  / total_n,
        "dead_pct":               float(n_dead   / N_FEAT * 100),
        "alive_pct":              float(n_alive  / N_FEAT * 100),
        "noisy_pct":              float(n_noisy  / N_FEAT * 100),
        "usable_pct":             float(n_usable / N_FEAT * 100),
        "n_dead":                 int(n_dead),
        "n_alive":                int(n_alive),
        "n_noisy":                int(n_noisy),
        "n_usable":               int(n_usable),
        "adaptive_noisy_thr":     adaptive_noisy_thr,
        "top1_fraction_mean":     float(alive_top1.mean())         if n_alive > 0 else 0.0,
        "selectivity_mean":       float(alive_sel.mean())          if n_alive > 0 else 0.0,
        "frac_sel_gt_0.5":        float((alive_sel > 0.50).mean()) if n_alive > 0 else 0.0,
        "frac_sel_gt_0.75":       float((alive_sel > 0.75).mean()) if n_alive > 0 else 0.0,
        "frac_sel_gt_0.9":        float((alive_sel > 0.90).mean()) if n_alive > 0 else 0.0,
        "usable_selectivity_mean":float(usable_sel.mean())         if n_usable > 0 else 0.0,
        "usable_frac_sel_gt_0.5": float((usable_sel > 0.5).mean()) if n_usable > 0 else 0.0,
    }
    artifacts = {
        "max_activating_image_indices": max_indices,
        "max_activating_image_values":  max_values,
        "sae_mean_acts":                sae_mean_acts,
        "sae_sparsity":                 sae_sparsity,
        "cls_sae_cnt":                  cls_cnt_np,
        "top1_fraction":                top1_fraction,
        "selectivity":                  selectivity,
        "dead_mask":                    dead_mask,
        "alive_mask":                   alive_mask,
        "noisy_mask":                   noisy_mask,
        "usable_mask":                  usable_mask,
        "adaptive_noisy_thr":           adaptive_noisy_thr,
    }
    return metrics, artifacts


# ── Plots (identical structure to eval_dino_sae_imagenet.py) ──────────────────

def _shorten(name, maxlen=18):
    return name if len(name) <= maxlen else name[:maxlen - 1] + "…"


def plot_top_activating_grid(dataset, artifacts, classnames,
                              n_feats=N_VIS_FEATURES, n_imgs=N_VIS_IMAGES):
    import matplotlib.gridspec as gridspec

    alive   = np.where(artifacts["alive_mask"])[0]
    cls_cnt = artifacts["cls_sae_cnt"]

    has_cls   = cls_cnt[:, alive].sum(0) > 0
    sparsity  = artifacts["sae_sparsity"].numpy()[alive]
    not_noisy = sparsity < 0.10
    valid     = alive[has_cls & not_noisy]
    if len(valid) == 0:
        valid = alive[has_cls]
    if len(valid) == 0:
        print("  [skip] no features with class data for visualisation")
        return

    sel_valid = artifacts["selectivity"][valid]
    n_feats   = min(n_feats, len(valid))
    top_feats = valid[np.argsort(sel_valid)[::-1]][:n_feats]
    print(f"  Visualising {n_feats} features from {len(valid)} valid "
          f"(has_cls & sparsity<10%) out of {len(alive)} alive")

    max_indices = artifacts["max_activating_image_indices"]
    max_values  = artifacts["max_activating_image_values"]
    n_stored    = min(n_imgs, max_indices.shape[1])

    col_widths = [1.8] + [1.0] * n_stored + [1.4]
    n_cols     = len(col_widths)
    fig = plt.figure(figsize=(sum(col_widths) * 1.05, n_feats * 2.6 + 0.8))
    gs  = gridspec.GridSpec(
        n_feats, n_cols, figure=fig,
        width_ratios=col_widths,
        hspace=0.55, wspace=0.06,
        left=0.02, right=0.98, top=0.93, bottom=0.03,
    )
    palette = plt.cm.tab10.colors

    for row, feat_id in enumerate(top_feats.tolist()):
        sel        = float(artifacts["selectivity"][feat_id])
        t1         = float(artifacts["top1_fraction"][feat_id])
        fire_count = int(artifacts["sae_sparsity"].numpy()[feat_id] * MAX_IMAGES)
        color      = palette[row % len(palette)]

        feat_cls   = cls_cnt[:, feat_id]
        total_fire = feat_cls.sum()
        top5_idx   = np.argsort(feat_cls)[::-1][:5]
        top5_names = [_shorten(classnames[i]) for i in top5_idx if i < len(classnames)]
        top5_frac  = feat_cls[top5_idx] / max(total_fire, 1)

        ax_info = fig.add_subplot(gs[row, 0])
        ax_info.axis("off")
        ax_info.add_patch(plt.Rectangle(
            (0, 0.82), 1, 0.18, transform=ax_info.transAxes,
            color=color, alpha=0.85, clip_on=False,
        ))
        ax_info.text(0.5, 0.91, f"Feature {feat_id}",
                     transform=ax_info.transAxes,
                     ha="center", va="center", fontsize=8,
                     fontweight="bold", color="white")
        ax_info.text(0.05, 0.78,
                     f"Selectivity: {sel:.3f}\n"
                     f"Top-1 frac:  {t1:.3f}\n"
                     f"Fire count:  ~{fire_count:,}\n"
                     f"Top class:\n  {_shorten(classnames[top5_idx[0]], 20)}",
                     transform=ax_info.transAxes,
                     ha="left", va="top", fontsize=7, linespacing=1.55,
                     family="monospace")
        ax_info.add_patch(plt.Rectangle(
            (0, 0), 0.04, 1, transform=ax_info.transAxes,
            color=color, alpha=0.9, clip_on=False,
        ))

        for col in range(n_stored):
            ax_img = fig.add_subplot(gs[row, col + 1])
            ax_img.axis("off")
            gi  = max_indices[feat_id, col].item()
            val = max_values[feat_id, col].item()
            if gi == 0 and val == 0:
                ax_img.set_facecolor("#f0f0f0")
                continue
            try:
                item    = dataset[int(gi)]
                pil_img = item["image"].convert("RGB").resize((160, 160))
                img_lbl = item.get("label", -1)
                cname   = _shorten(classnames[img_lbl]) if 0 <= img_lbl < len(classnames) else ""
                ax_img.imshow(pil_img, aspect="auto")
                for sp in ax_img.spines.values():
                    sp.set_edgecolor(color); sp.set_linewidth(2)
                ax_img.set_xticks([]); ax_img.set_yticks([])
                ax_img.set_xlabel(f"{cname}\nact={val:.1f}",
                                  fontsize=6, labelpad=2, color="#333333")
            except Exception:
                ax_img.text(0.5, 0.5, "err", ha="center", va="center",
                            fontsize=8, color="gray")

        ax_bar = fig.add_subplot(gs[row, n_cols - 1])
        ax_bar.barh(range(len(top5_names)), top5_frac[:len(top5_names)][::-1],
                    color=color, alpha=0.8)
        ax_bar.set_yticks(range(len(top5_names)))
        ax_bar.set_yticklabels(top5_names[::-1], fontsize=6)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Fraction of fires", fontsize=6)
        ax_bar.xaxis.set_tick_params(labelsize=6)
        ax_bar.set_title("Top classes", fontsize=6, pad=2)
        ax_bar.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Top most-selective SAE features — ALIGN (kakaobrain/align-base) on ImageNet\n"
        "Sorted by selectivity (1 − H_norm).  Each row = one SAE feature.",
        fontsize=10, fontweight="bold",
    )
    path = os.path.join(OUT_DIR, "top_monosemantic_features.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_monosemanticity_histograms(artifacts):
    mask     = artifacts["alive_mask"]
    sel      = artifacts["selectivity"][mask]
    top1     = artifacts["top1_fraction"][mask]
    sparsity = artifacts["sae_sparsity"].numpy()[mask]
    if len(sel) == 0:
        print("  [skip] no alive features for histogram")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.hist(top1, bins=min(50, len(top1)), color="#4C72B0", edgecolor="none")
    ax.axvline(0.5,  color="orange", lw=1.2, linestyle="--", label="0.5")
    ax.axvline(0.75, color="red",    lw=1.2, linestyle="--", label="0.75")
    ax.set_xlabel("Top-1 class fraction")
    ax.set_ylabel("# features")
    ax.set_title(f"Top-1 Fraction\nmean={top1.mean():.3f}  |  {(top1>0.5).mean()*100:.1f}% > 0.5")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(sel, bins=min(50, len(sel)), color="#55A868", edgecolor="none")
    ax.axvline(0.5,  color="orange", lw=1.2, linestyle="--", label="sel=0.5")
    ax.axvline(0.75, color="red",    lw=1.2, linestyle="--", label="sel=0.75")
    ax.set_xlabel("Selectivity = 1 − H_norm")
    ax.set_ylabel("# features")
    ax.set_title(f"Selectivity (Monosemanticity)\nmean={sel.mean():.3f}  |  {(sel>0.5).mean()*100:.1f}% > 0.5")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(np.log10(sparsity + 1e-6), bins=min(60, len(sparsity)),
            color="#DD8452", edgecolor="none")
    ax.set_xlabel("log10(sparsity)  = fraction of images that fire")
    ax.set_ylabel("# features")
    ax.set_title(f"Feature Sparsity\n{len(sel):,} alive features")

    thr = artifacts["adaptive_noisy_thr"]
    plt.suptitle(
        "ALIGN SAE — Monosemanticity on ImageNet (repo definition)\n"
        f"Adaptive noisy threshold={thr:.4f} (95th pct alive mean_acts), "
        f"Act threshold={ACT_THRESHOLD}",
        fontsize=9,
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "monosemanticity_histograms.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_class_feature_heatmap(artifacts, classnames, top_n_classes=30, top_n_feats=50):
    cls_cnt = artifacts["cls_sae_cnt"]
    alive   = np.where(artifacts["alive_mask"])[0]
    if len(alive) == 0:
        print("  [skip] no alive features for heatmap")
        return

    top_n_feats   = min(top_n_feats, len(alive))
    top_n_classes = min(top_n_classes, cls_cnt.shape[0])

    feat_totals  = cls_cnt[:, alive].sum(0)
    top_f_local  = np.argsort(feat_totals)[::-1][:top_n_feats]
    top_f_idx    = alive[top_f_local]
    class_totals = cls_cnt[:, top_f_idx].sum(1)
    top_c_idx    = np.argsort(class_totals)[::-1][:top_n_classes]

    sub      = cls_cnt[np.ix_(top_c_idx, top_f_idx)]
    sub_norm = sub / (sub.max(0, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(
        figsize=(max(10, top_n_feats * 0.35), max(6, top_n_classes * 0.28))
    )
    im = ax.imshow(sub_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(top_n_classes))
    ax.set_yticklabels(
        [classnames[i] if i < len(classnames) else str(i) for i in top_c_idx], fontsize=7
    )
    ax.set_xticks(range(top_n_feats))
    ax.set_xticklabels([f"F{i}" for i in top_f_idx], fontsize=5, rotation=90)
    ax.set_xlabel("SAE Feature")
    ax.set_ylabel("ImageNet Class")
    ax.set_title(
        f"Class × Feature Activation Heatmap  "
        f"(top {top_n_feats} alive × top {top_n_classes} classes)\n"
        "Normalised per feature — bright = class dominates this feature"
    )
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "class_feature_heatmap.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_mean_acts_and_sparsity(artifacts):
    mean_acts = artifacts["sae_mean_acts"].numpy()
    sparsity  = artifacts["sae_sparsity"].numpy()
    dead      = artifacts["dead_mask"]
    thr       = artifacts["adaptive_noisy_thr"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    alive_ma = mean_acts[~dead]
    ax.hist(alive_ma, bins=min(80, len(alive_ma)), color="steelblue", edgecolor="none")
    ax.axvline(thr, color="red", lw=1.5, linestyle="--",
               label=f"adaptive noisy thr={thr:.4f}")
    ax.set_xlabel("Conditioned mean activation")
    ax.set_ylabel("# features")
    ax.set_title(f"SAE Mean Acts (alive features)\n"
                 f"{(alive_ma > thr).sum():,} noisy  |  {(alive_ma <= thr).sum():,} clean")
    ax.legend(fontsize=8)

    ax = axes[1]
    alive_sp = sparsity[~dead]
    ax.hist(np.log10(alive_sp + 1e-8), bins=60, color="salmon", edgecolor="none")
    ax.set_xlabel("log10(sparsity)  fraction of images that activate feature")
    ax.set_ylabel("# features")
    ax.set_title(f"SAE Feature Sparsity\n{len(alive_sp):,} alive features")

    plt.suptitle("ALIGN SAE — Mean Acts & Sparsity Diagnostics", fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "mean_acts_sparsity.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Artifact I/O ──────────────────────────────────────────────────────────────

def save_artifacts(artifacts, save_dir):
    torch.save(artifacts["max_activating_image_indices"],
               os.path.join(save_dir, "max_activating_image_indices.pt"))
    torch.save(artifacts["max_activating_image_values"],
               os.path.join(save_dir, "max_activating_image_values.pt"))
    torch.save(artifacts["sae_mean_acts"],
               os.path.join(save_dir, "sae_mean_acts.pt"))
    torch.save(artifacts["sae_sparsity"],
               os.path.join(save_dir, "sae_sparsity.pt"))
    np.save(os.path.join(save_dir, "cls_sae_cnt.npy"),   artifacts["cls_sae_cnt"])
    np.save(os.path.join(save_dir, "selectivity.npy"),   artifacts["selectivity"])
    np.save(os.path.join(save_dir, "top1_fraction.npy"), artifacts["top1_fraction"])
    print(f"  Artifacts saved → {save_dir}/")


def load_artifacts_from_disk(save_dir):
    max_idx  = torch.load(os.path.join(save_dir, "max_activating_image_indices.pt"), map_location="cpu")
    max_val  = torch.load(os.path.join(save_dir, "max_activating_image_values.pt"),  map_location="cpu")
    mean_act = torch.load(os.path.join(save_dir, "sae_mean_acts.pt"), map_location="cpu")
    sparsity = torch.load(os.path.join(save_dir, "sae_sparsity.pt"),  map_location="cpu")
    cls_cnt  = np.load(os.path.join(save_dir, "cls_sae_cnt.npy"))
    sel      = np.load(os.path.join(save_dir, "selectivity.npy"))
    top1     = np.load(os.path.join(save_dir, "top1_fraction.npy"))

    dead_mask  = sparsity.numpy() == 0
    alive_mask = ~dead_mask
    alive_mean = mean_act.numpy()[alive_mask]
    thr        = float(np.percentile(alive_mean, 95)) if len(alive_mean) > 0 else NOISY_THRESHOLD
    noisy_mask  = (mean_act.numpy() > thr) & alive_mask
    usable_mask = alive_mask & ~noisy_mask

    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    return metrics, {
        "max_activating_image_indices": max_idx,
        "max_activating_image_values":  max_val,
        "sae_mean_acts":                mean_act,
        "sae_sparsity":                 sparsity,
        "cls_sae_cnt":                  cls_cnt,
        "selectivity":                  sel,
        "top1_fraction":                top1,
        "dead_mask":                    dead_mask,
        "alive_mask":                   alive_mask,
        "noisy_mask":                   noisy_mask,
        "usable_mask":                  usable_mask,
        "adaptive_noisy_thr":           thr,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip eval; reload saved artifacts and regenerate plots only")
    args = parser.parse_args()

    print(f"\nDevice:  {DEVICE}")
    print(f"SAE:     {SAE_PATH}")
    print(f"Out:     {OUT_DIR}/")
    print("=" * 70)

    classnames = load_classnames(CLASSNAMES_PATH)
    print(f"Classes: {len(classnames)}")

    print("\nLoading ImageNet-1k from HuggingFace...")
    dataset = load_dataset("evanarlian/imagenet_1k_resized_256",
                           split="train", trust_remote_code=True)
    if MAX_IMAGES is not None:
        dataset = dataset.select(range(min(MAX_IMAGES, len(dataset))))
    print(f"Dataset: {len(dataset):,} images")

    if args.replot:
        print(f"\n[--replot] Loading saved artifacts from {FEATURE_DATA_DIR}/")
        metrics, artifacts = load_artifacts_from_disk(FEATURE_DATA_DIR)
    else:
        sae, cfg = load_sae(SAE_PATH, DEVICE)
        block_idx = cfg.block_layer if cfg.block_layer >= 0 else N_BLOCKS + cfg.block_layer
        print(f"SAE loaded — d_in={cfg.d_in}, d_sae={cfg.d_in * cfg.expansion_factor}, "
              f"block_layer={cfg.block_layer} (blocks[{block_idx}]), "
              f"l1={cfg.l1_coefficient}")

        print("\nLoading ALIGN (kakaobrain/align-base)...")
        align, proc = load_align(DEVICE)

        print(f"\nRunning eval  (batch={BATCH}, "
              f"act_thr={ACT_THRESHOLD}, images={len(dataset):,})...")
        metrics, artifacts = run_eval(align, sae, proc, dataset, block_idx, classnames)

    # Print results
    N_SAE = artifacts["max_activating_image_indices"].shape[0]
    print(f"\n{'='*70}")
    print("RESULTS — ALIGN SAE @ block_layer=-1 on ImageNet-1k")
    print(f"{'='*70}")
    print(f"  Images evaluated       : {metrics['n_images']:,}")
    print(f"  L0                     : {metrics['l0']:.2f}  active features / image")
    print(f"  Reconstruction MSE     : {metrics['recon_mse']:.5f}")
    print(f"  Explained variance     : {metrics['explained_variance']:.4f}")
    print(f"  Dead features          : {metrics['dead_pct']:.2f}%  ({metrics['n_dead']:,} / {N_SAE:,})")
    print(f"  Alive features         : {metrics['alive_pct']:.2f}%  ({metrics['n_alive']:,})")
    print(f"  Noisy features         : {metrics['noisy_pct']:.2f}%  ({metrics['n_noisy']:,})")
    print(f"  Usable features        : {metrics['usable_pct']:.2f}%  ({metrics['n_usable']:,})")
    print()
    print("  ── Monosemanticity (from cls_sae_cnt, all alive features) ──────")
    print(f"  Top-1 class fraction   : {metrics['top1_fraction_mean']:.4f}")
    print(f"  Selectivity (1−H_norm) : {metrics['selectivity_mean']:.4f}")
    print(f"  Sel > 0.50             : {metrics['frac_sel_gt_0.5']:.3f}  "
          f"({int(metrics['frac_sel_gt_0.5'] * metrics['n_alive']):,} features)")
    print(f"  Sel > 0.75             : {metrics['frac_sel_gt_0.75']:.3f}  "
          f"({int(metrics['frac_sel_gt_0.75'] * metrics['n_alive']):,} features)")
    print(f"  Sel > 0.90             : {metrics['frac_sel_gt_0.9']:.3f}  "
          f"({int(metrics['frac_sel_gt_0.9'] * metrics['n_alive']):,} features)")
    print(f"  Usable sel mean        : {metrics['usable_selectivity_mean']:.4f}")
    print(f"{'='*70}")

    if not args.replot:
        print("\nSaving artifacts...")
        save_artifacts(artifacts, FEATURE_DATA_DIR)
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print("\nGenerating plots...")
    plot_top_activating_grid(dataset, artifacts, classnames)
    plot_monosemanticity_histograms(artifacts)
    plot_class_feature_heatmap(artifacts, classnames)
    plot_mean_acts_and_sparsity(artifacts)

    print(f"\nAll outputs → {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()

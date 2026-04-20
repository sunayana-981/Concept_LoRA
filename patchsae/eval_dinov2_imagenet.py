#!/usr/bin/env python3
"""
DINOv2-base SAE evaluation on ImageNet-1k.

Metrics:
  - L0, dead%, MSE, explained variance
  - Per-feature label entropy (normalized) — monosemanticity proxy
  - Monosemanticity score = 1 - H_norm  (1=perfectly selective, 0=uniform)
  - Fraction of alive features with selectivity > 0.5
  - Feature activation frequency histogram
  - Selectivity histogram

Plots:
  - Top-activating image grids for the 16 most *selective* alive features
  - Feature activation frequency histogram
  - Feature selectivity histogram
  - Summary bar chart

Usage:
    cd /home/sunayana/Documents/Concept_LoRA/patchsae
    python -u eval_dinov2_imagenet.py
"""

import json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoImageProcessor, Dinov2Model

sys.path.insert(0, os.path.dirname(__file__))
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# ── Config ────────────────────────────────────────────────────────────────────

SAE_PATH = "out/checkpoints/4s0816zv/final_sparse_autoencoder_facebook/dinov2-base_-1_resid_49152.pt"
OUT_DIR  = "out/eval_imagenet"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
BATCH    = 64
TOP_FEATS_VIS = 16   # features to visualise in top-activating grid
TOP_IMGS_VIS  = 8    # images per feature row
MAX_IMAGES    = 100_000  # None = full 1.28M train split; 100k gives ~65 min runtime

# ImageNet-1k from HuggingFace (same dataset used in training)
IMAGENET_HF = dict(
    path="evanarlian/imagenet_1k_resized_256",
    split="train",
    trust_remote_code=True,
)
N_CLASSES = 1000
LOG_C = math.log(N_CLASSES)   # normalisation constant for entropy

os.makedirs(OUT_DIR, exist_ok=True)

# ── Model loading ─────────────────────────────────────────────────────────────

def load_sae(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    sae  = SparseAutoencoder(cfg, device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(device)
    return sae, cfg


def load_dino(device):
    proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    return model, proc


# ── Layer hook ────────────────────────────────────────────────────────────────

class LayerCapture:
    def __init__(self):
        self.output = None
        self._h = None

    def register(self, layer):
        def _hook(m, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            self.output = hs.detach().float()
        self._h = layer.register_forward_hook(_hook)

    def remove(self):
        if self._h:
            self._h.remove()
            self._h = None


# ── HF ImageNet collate ───────────────────────────────────────────────────────

def make_collate(proc):
    def collate(batch):
        imgs   = []
        labels = []
        for item in batch:
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(img)
            labels.append(item["label"])
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"]
        return pv, torch.tensor(labels, dtype=torch.long)
    return collate


# ── Main eval ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_imagenet(dino, sae, proc, dataset, layer_idx):
    N_FEAT = sae.W_enc.shape[1]

    collate = make_collate(proc)
    loader  = DataLoader(
        dataset, batch_size=BATCH, shuffle=False,
        num_workers=4, pin_memory=(DEVICE == "cuda"),
        collate_fn=collate,
    )

    cap = LayerCapture()
    cap.register(dino.encoder.layer[layer_idx])

    total_l0   = 0.0
    total_mse  = 0.0
    total_ev   = 0.0
    total_imgs = 0

    feat_fire  = torch.zeros(N_FEAT)                        # fire count per feature
    label_cnt  = torch.zeros(N_FEAT, N_CLASSES)             # [N_FEAT, 1000]
    # top-TOP_IMGS_VIS per feature: list of (activation_value, global_dataset_idx)
    topk_buf   = [[] for _ in range(N_FEAT)]

    global_idx = 0

    for pixel_values, labels in tqdm(loader, desc="  ImageNet", leave=True):
        pixel_values = pixel_values.to(DEVICE)
        labels = labels.cpu()
        B = pixel_values.shape[0]

        dino(pixel_values=pixel_values)
        cls_act = cap.output[:, 0, :]          # [B, 768] — CLS token

        sae_out, feat_acts, _ = sae(cls_act)
        feat_acts = feat_acts.cpu()            # [B, N_FEAT]

        # --- scalar metrics ---
        total_l0 += (feat_acts > 0).float().sum(dim=-1).mean().item() * B
        mse_s  = (sae_out.cpu() - cls_act.cpu()).pow(2).sum(-1)
        var_s  = cls_act.cpu().pow(2).sum(-1).clamp(min=1e-8)
        ev_s   = 1 - mse_s / var_s
        total_mse  += mse_s.mean().item() * B
        total_ev   += ev_s.mean().item()  * B
        total_imgs += B

        # --- sparsity / label accumulators ---
        active = (feat_acts > 0).float()                   # [B, N_FEAT]
        feat_fire += active.sum(0)
        oh = F.one_hot(labels, N_CLASSES).float()          # [B, 1000]
        label_cnt += active.T @ oh                         # [N_FEAT, 1000]

        # --- top-K images per feature ---
        for b in range(B):
            gi    = global_idx + b
            fired = (feat_acts[b] > 0).nonzero(as_tuple=True)[0]
            for f in fired.tolist():
                v = feat_acts[b, f].item()
                buf = topk_buf[f]
                buf.append((v, gi))
                # keep only top TOP_IMGS_VIS by activation value to bound memory
                if len(buf) > TOP_IMGS_VIS * 4:
                    buf.sort(reverse=True)
                    topk_buf[f] = buf[:TOP_IMGS_VIS * 4]

        global_idx += B

    cap.remove()

    # Final trim to TOP_IMGS_VIS per feature
    for f in range(N_FEAT):
        topk_buf[f] = sorted(topk_buf[f], reverse=True)[:TOP_IMGS_VIS]

    # ── Derived metrics ───────────────────────────────────────────────────────
    dead_mask = feat_fire == 0
    alive_mask = ~dead_mask
    dead_pct  = dead_mask.float().mean().item() * 100.0
    n_active  = alive_mask.sum().item()

    # Normalised label entropy per alive feature
    counts_a = label_cnt[alive_mask]                        # [n_alive, 1000]
    probs    = counts_a / counts_a.sum(1, keepdim=True).clamp(min=1e-9)
    ent      = -(probs * (probs + 1e-12).log()).sum(1)      # [n_alive]
    ent_norm = ent / LOG_C                                  # 0=monosemantic, 1=uniform

    # Selectivity = 1 - H_norm  (1 = fully monosemantic)
    selectivity = 1.0 - ent_norm                            # [n_alive]

    # fire-weighted mean entropy (gives more weight to high-fire features)
    w = feat_fire[alive_mask]
    h_wt_norm  = (ent_norm * w / w.sum()).sum().item()
    h_uw_norm  = ent_norm.mean().item()
    sel_mean   = selectivity.mean().item()
    sel_wt     = (selectivity * w / w.sum()).sum().item()
    frac_mono  = (selectivity > 0.5).float().mean().item()  # fraction with sel>0.5

    # Per-feature selectivity array (full size, 0 for dead features)
    selectivity_full = torch.zeros(N_FEAT)
    selectivity_full[alive_mask] = selectivity

    metrics = {
        "dataset":        "imagenet",
        "n_images":       total_imgs,
        "num_classes":    N_CLASSES,
        "l0":             total_l0  / total_imgs,
        "recon_mse":      total_mse / total_imgs,
        "explained_var":  total_ev  / total_imgs,
        "dead_pct":       dead_pct,
        "n_active":       int(n_active),
        "h_wt_norm":      h_wt_norm,
        "h_uw_norm":      h_uw_norm,
        "monosemanticity_score_mean":      sel_mean,
        "monosemanticity_score_weighted":  sel_wt,
        "frac_monosemantic_above_0.5":     frac_mono,
        "frac_monosemantic_above_0.75":    (selectivity > 0.75).float().mean().item(),
        "frac_monosemantic_above_0.9":     (selectivity > 0.90).float().mean().item(),
    }

    return metrics, feat_fire, selectivity_full, topk_buf


# ── Visualisation ─────────────────────────────────────────────────────────────

def get_pil(dataset, global_idx):
    """Retrieve a PIL image from the HF dataset by index."""
    img = dataset[int(global_idx)]["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def plot_top_activating_grid(dataset, topk_buf, selectivity_full, n_feats=TOP_FEATS_VIS, n_imgs=TOP_IMGS_VIS):
    """
    For the n_feats most *selective* alive features, plot a grid of their
    top-n_imgs activating images.
    """
    alive_feats = (selectivity_full > 0).nonzero(as_tuple=True)[0]
    if len(alive_feats) == 0:
        print("  [skip] no alive features")
        return

    # Sort alive features by selectivity (descending)
    sel_alive = selectivity_full[alive_feats]
    order     = sel_alive.argsort(descending=True)
    top_feat_ids = alive_feats[order][:n_feats].tolist()

    fig, axes = plt.subplots(n_feats, n_imgs, figsize=(n_imgs * 2.2, n_feats * 2.4))
    fig.suptitle(
        f"Top {n_feats} most selective features — ImageNet\n"
        "(each row = one SAE feature, sorted by selectivity = 1 − H_norm)",
        fontsize=10, y=1.01,
    )

    for row, feat_id in enumerate(top_feat_ids):
        entries   = topk_buf[feat_id]
        sel_score = selectivity_full[feat_id].item()
        for col in range(n_imgs):
            ax = axes[row, col]
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    f"F{feat_id}\nsel={sel_score:.2f}",
                    fontsize=7, rotation=0, labelpad=55, va="center",
                )
            if col >= len(entries):
                continue
            val, gi = entries[col]
            try:
                img = get_pil(dataset, gi).resize((112, 112))
                ax.imshow(img)
                ax.set_title(f"act={val:.2f}", fontsize=6, pad=2)
            except Exception:
                ax.text(0.5, 0.5, "err", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "top_selective_features_imagenet.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_selectivity_histogram(selectivity_full, feat_fire):
    """Histogram of per-feature selectivity scores (alive features only)."""
    alive_sel = selectivity_full[feat_fire > 0].numpy()
    if len(alive_sel) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: selectivity histogram
    ax = axes[0]
    ax.hist(alive_sel, bins=50, color="steelblue", edgecolor="none")
    ax.axvline(0.5,  color="orange", linestyle="--", linewidth=1.2, label="sel=0.5")
    ax.axvline(0.75, color="red",    linestyle="--", linewidth=1.2, label="sel=0.75")
    ax.set_xlabel("Selectivity = 1 − H_norm")
    ax.set_ylabel("# features")
    ax.set_title(f"Feature Selectivity (Monosemanticity)\n"
                 f"{len(alive_sel):,} alive features | mean={alive_sel.mean():.3f}")
    ax.legend(fontsize=8)

    # Right: entropy histogram
    ax = axes[1]
    ent_norm = 1.0 - alive_sel
    ax.hist(ent_norm, bins=50, color="salmon", edgecolor="none")
    ax.axvline(0.5,  color="orange", linestyle="--", linewidth=1.2, label="H=0.5")
    ax.axvline(0.25, color="green",  linestyle="--", linewidth=1.2, label="H=0.25")
    ax.set_xlabel("Normalised label entropy H_norm")
    ax.set_ylabel("# features")
    ax.set_title("Feature Label Entropy\n[0=monosemantic, 1=uniform]")
    ax.legend(fontsize=8)

    plt.suptitle("DINOv2-base SAE — Monosemanticity Analysis on ImageNet", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "monosemanticity_histogram.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_fire_freq_histogram(feat_fire):
    """Log-scale histogram of per-feature fire counts."""
    counts = feat_fire[feat_fire > 0].numpy()
    if len(counts) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(np.log10(counts + 1), bins=60, color="steelblue", edgecolor="none")
    ax.set_xlabel("log10(fire count + 1)")
    ax.set_ylabel("# features")
    ax.set_title(f"Feature Activation Frequency — ImageNet\n"
                 f"({len(counts):,} alive / {len(feat_fire):,} total)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "feat_freq_hist_imagenet.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def plot_selectivity_vs_fire(selectivity_full, feat_fire):
    """Scatter: fire count vs selectivity (helps see if monosemantic = rare)."""
    alive = feat_fire > 0
    x = feat_fire[alive].numpy()
    y = selectivity_full[alive].numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(np.log10(x + 1), y, s=2, alpha=0.4, color="steelblue")
    ax.set_xlabel("log10(fire count + 1)")
    ax.set_ylabel("Selectivity (1 − H_norm)")
    ax.set_title("Selectivity vs. Activation Frequency\nDINOv2-base SAE — ImageNet")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "selectivity_vs_fire.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDevice: {DEVICE}")
    print(f"SAE:    {SAE_PATH}")
    print(f"Plots → {OUT_DIR}/")
    print("=" * 70)

    sae, cfg = load_sae(SAE_PATH, DEVICE)
    print(f"SAE loaded — d_in={cfg.d_in}, d_sae={cfg.d_sae}, "
          f"class_token={cfg.class_token}, layer={cfg.block_layer}")

    print("\nLoading DINOv2-base...")
    dino, proc = load_dino(DEVICE)
    layer_idx  = 12 + cfg.block_layer if cfg.block_layer < 0 else cfg.block_layer
    print(f"Hooked layer index: {layer_idx}")

    print("\nLoading ImageNet-1k from HuggingFace...")
    dataset = load_dataset(**IMAGENET_HF)
    if MAX_IMAGES is not None:
        dataset = dataset.select(range(min(MAX_IMAGES, len(dataset))))
    print(f"Dataset size: {len(dataset):,} images, {N_CLASSES} classes")

    print(f"\n{'─'*60}")
    print("Running eval on ImageNet ...")
    metrics, feat_fire, selectivity_full, topk_buf = eval_imagenet(
        dino, sae, proc, dataset, layer_idx
    )

    # ── Print results ──────────────────────────────────────────────────────────
    N_SAE = cfg.d_sae
    print(f"\n{'='*70}")
    print("RESULTS — DINOv2-base SAE on ImageNet-1k")
    print(f"{'='*70}")
    print(f"  Images evaluated       : {metrics['n_images']:,}")
    print(f"  L0 (active feats/token): {metrics['l0']:.2f}")
    print(f"  Reconstruction MSE     : {metrics['recon_mse']:.5f}")
    print(f"  Explained variance     : {metrics['explained_var']:.4f}")
    print(f"  Dead features          : {metrics['dead_pct']:.2f}%  "
          f"({N_SAE - metrics['n_active']:,} / {N_SAE:,})")
    print(f"  Alive features         : {metrics['n_active']:,}")
    print()
    print("  ── Monosemanticity ────────────────────────────────────────────")
    print(f"  Label entropy (H_norm, wt)   : {metrics['h_wt_norm']:.4f}  [0=mono, 1=uniform]")
    print(f"  Label entropy (H_norm, uw)   : {metrics['h_uw_norm']:.4f}")
    print(f"  Selectivity score (mean)     : {metrics['monosemanticity_score_mean']:.4f}  [0=uniform, 1=mono]")
    print(f"  Selectivity score (wt mean)  : {metrics['monosemanticity_score_weighted']:.4f}")
    print(f"  Fraction sel > 0.50          : {metrics['frac_monosemantic_above_0.5']:.4f}  "
          f"({int(metrics['frac_monosemantic_above_0.5'] * metrics['n_active']):,} features)")
    print(f"  Fraction sel > 0.75          : {metrics['frac_monosemantic_above_0.75']:.4f}  "
          f"({int(metrics['frac_monosemantic_above_0.75'] * metrics['n_active']):,} features)")
    print(f"  Fraction sel > 0.90          : {metrics['frac_monosemantic_above_0.9']:.4f}  "
          f"({int(metrics['frac_monosemantic_above_0.9'] * metrics['n_active']):,} features)")
    print(f"{'='*70}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_top_activating_grid(dataset, topk_buf, selectivity_full)
    plot_selectivity_histogram(selectivity_full, feat_fire)
    plot_fire_freq_histogram(feat_fire)
    plot_selectivity_vs_fire(selectivity_full, feat_fire)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "dinov2_sae_imagenet_eval.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {out_json}")
    print(f"Plots saved   → {OUT_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Full eval + top-activating-feature visualisation for the DINOv2-base SAE.

Metrics  : L0, dead%, MSE, explained variance, label entropy (per dataset)
Plots    : top-activating-image grids for the 16 most active features
           (mirrors demo.ipynb style)

Usage:
    cd /home/sunayana/Documents/Concept_LoRA/patchsae
    python -u eval_dinov2_sae_full.py
"""

import json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Dinov2Model

sys.path.insert(0, os.path.dirname(__file__))
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# ── Config ────────────────────────────────────────────────────────────────────

SAE_PATH  = "out/checkpoints/4s0816zv/final_sparse_autoencoder_facebook/dinov2-base_-1_resid_49152.pt"
DATA_ROOT = "/home/sunayana/Documents/Concept_LoRA/data"
OUT_DIR   = "out/eval_plots"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH     = 64
TOP_FEATS = 16   # features to visualise
TOP_IMGS  = 8    # images per feature grid

DATASETS = {
    "caltech101": {
        "root":    f"{DATA_ROOT}/caltech-101/Caltech101/101_ObjectCategories",
        "exclude": {"BACKGROUND_Google"},
    },
    "medmnist": {
        "root":    f"{DATA_ROOT}/pathmnist_imagefolder",
        "exclude": set(),
    },
    "eurosat": {
        "root":    f"{DATA_ROOT}/eurosat/2750",
        "exclude": set(),
    },
}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

class FilteredImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform, exclude_classes=None):
        full = tv_datasets.ImageFolder(root=root, transform=transform)
        excl = exclude_classes or set()
        keep = [i for i, (_, l) in enumerate(full.samples)
                if full.classes[l] not in excl]
        kept_cls = sorted({full.classes[full.targets[i]] for i in keep})
        old2new  = {full.class_to_idx[c]: ni for ni, c in enumerate(kept_cls)}
        self._ds, self._idx, self._map = full, keep, old2new
        self.classes     = kept_cls
        self.num_classes = len(kept_cls)
        # keep original PIL images for visualisation
        self._raw = tv_datasets.ImageFolder(root=root)

    def __len__(self):  return len(self._idx)

    def __getitem__(self, i):
        img, lbl = self._ds[self._idx[i]]
        return img, self._map[lbl]

    def get_pil(self, i):
        path, _ = self._raw.samples[self._idx[i]]
        return Image.open(path).convert("RGB")


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
        if self._h: self._h.remove(); self._h = None


# ── Load models ───────────────────────────────────────────────────────────────

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


def make_transform(proc):
    def t(img):
        if img.mode != "RGB": img = img.convert("RGB")
        return proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
    return t


# ── Per-dataset eval ──────────────────────────────────────────────────────────

@torch.no_grad()
def eval_dataset(dino, sae, proc, ds_name, ds_info, layer_idx):
    transform = make_transform(proc)
    ds  = FilteredImageFolder(ds_info["root"], transform, ds_info["exclude"])
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=(DEVICE == "cuda"))
    N_FEAT = sae.W_enc.shape[1]
    C      = ds.num_classes

    cap = LayerCapture()
    cap.register(dino.encoder.layer[layer_idx])

    total_l0    = 0.0
    total_mse   = 0.0
    total_ev    = 0.0
    total_imgs  = 0
    feat_fire   = torch.zeros(N_FEAT)
    label_cnt   = torch.zeros(N_FEAT, C)
    # store (max_activation, global_img_idx) per feature
    feat_max_val = torch.full((N_FEAT,), -1e9)
    feat_max_idx = torch.zeros(N_FEAT, dtype=torch.long)
    # top-8 per feature: list of (act_value, global_idx)
    topk_buf = [[] for _ in range(N_FEAT)]   # will trim later

    global_idx = 0

    for imgs, labels in tqdm(loader, desc=f"  {ds_name}", leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.cpu()
        B = imgs.shape[0]

        dino(pixel_values=imgs)
        cls_act = cap.output[:, 0, :]      # [B, 768] — CLS token only

        sae_out, feat_acts, loss_dict = sae(cls_act)
        feat_acts = feat_acts.cpu()        # [B, N_FEAT]

        # L0
        total_l0 += (feat_acts > 0).float().sum(dim=-1).mean().item() * B

        # MSE & EV per sample
        mse_s  = (sae_out.cpu() - cls_act.cpu()).pow(2).sum(-1)   # [B]
        var_s  = cls_act.cpu().pow(2).sum(-1).clamp(min=1e-8)
        ev_s   = 1 - mse_s / var_s
        total_mse  += mse_s.mean().item() * B
        total_ev   += ev_s.mean().item()  * B
        total_imgs += B

        # Dead & label entropy accumulators
        active = (feat_acts > 0).float()              # [B, N_FEAT]
        feat_fire += active.sum(0)
        oh = F.one_hot(labels, C).float()             # [B, C]
        label_cnt += active.T @ oh                    # [N_FEAT, C]

        # Track top-K images per feature (for visualisation)
        for b in range(B):
            gi = global_idx + b
            fired = (feat_acts[b] > 0).nonzero(as_tuple=True)[0]
            for f in fired.tolist():
                v = feat_acts[b, f].item()
                topk_buf[f].append((v, gi))
        global_idx += B

    cap.remove()

    # Trim topk_buf to top-TOP_IMGS per feature
    for f in range(N_FEAT):
        topk_buf[f] = sorted(topk_buf[f], reverse=True)[:TOP_IMGS]

    # Metrics
    dead_mask = feat_fire == 0
    dead_pct  = dead_mask.float().mean().item() * 100.0
    n_active  = (~dead_mask).sum().item()

    act_mask  = ~dead_mask
    counts_a  = label_cnt[act_mask]
    probs     = counts_a / counts_a.sum(1, keepdim=True).clamp(min=1e-9)
    ent       = -(probs * (probs + 1e-12).log()).sum(1)
    w         = feat_fire[act_mask]
    h_wt_norm = (ent * w / w.sum()).sum().item() / math.log(C) if C > 1 else 0.0
    h_uw_norm = ent.mean().item() / math.log(C) if C > 1 else 0.0

    metrics = {
        "dataset":          ds_name,
        "n_images":         total_imgs,
        "num_classes":      C,
        "l0":               total_l0  / total_imgs,
        "recon_mse":        total_mse / total_imgs,
        "explained_var":    total_ev  / total_imgs,
        "dead_pct":         dead_pct,
        "n_active":         int(n_active),
        "h_wt_norm":        h_wt_norm,
        "h_uw_norm":        h_uw_norm,
    }

    return metrics, feat_fire, topk_buf, ds


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_top_activating_grid(ds, topk_buf, feat_fire, ds_name, n_feats=TOP_FEATS, n_imgs=TOP_IMGS):
    """
    For the n_feats most-active features, create a grid of their
    top-n_imgs activating images — mirrors demo.ipynb style.
    """
    # Select top-n_feats alive features by total fire count
    alive_feats = feat_fire.nonzero(as_tuple=True)[0]
    if len(alive_feats) == 0:
        print(f"  [skip] no alive features for {ds_name}")
        return

    top_feat_ids = alive_feats[feat_fire[alive_feats].argsort(descending=True)][:n_feats]

    fig, axes = plt.subplots(n_feats, n_imgs, figsize=(n_imgs * 2.2, n_feats * 2.2))
    fig.suptitle(f"Top {n_feats} features — {ds_name}\n(each row = one SAE feature, sorted by activation frequency)",
                 fontsize=10, y=1.01)

    for row, feat_id in enumerate(top_feat_ids.tolist()):
        entries = topk_buf[feat_id]           # list of (val, global_idx)
        fire_count = int(feat_fire[feat_id].item())
        for col in range(n_imgs):
            ax = axes[row, col]
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"F{feat_id}\n({fire_count} fires)",
                              fontsize=7, rotation=0, labelpad=50, va="center")
            if col >= len(entries):
                continue
            val, gi = entries[col]
            try:
                pil_img = ds.get_pil(gi)
                pil_img = pil_img.resize((112, 112))
                ax.imshow(pil_img)
                ax.set_title(f"act={val:.2f}", fontsize=6, pad=2)
            except Exception as e:
                ax.text(0.5, 0.5, "err", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"top_activating_{ds_name}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_activation_histogram(feat_fire, ds_name):
    """Log-scale histogram of per-feature fire counts."""
    counts = feat_fire[feat_fire > 0].numpy()
    if len(counts) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(np.log10(counts + 1), bins=60, color="steelblue", edgecolor="none")
    ax.set_xlabel("log10(fire count + 1)")
    ax.set_ylabel("# features")
    ax.set_title(f"Feature activation frequency — {ds_name}\n"
                 f"({len(counts):,} alive / {len(feat_fire):,} total)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"feat_freq_hist_{ds_name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_bar(results):
    """Side-by-side bar chart of key metrics across datasets."""
    ds_names = [r["dataset"] for r in results]
    metrics  = ["l0", "explained_var", "dead_pct", "h_wt_norm"]
    labels   = ["L0 (active feats/token)", "Explained Variance", "Dead Features %", "Label Entropy (norm, wt)"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m, lbl in zip(axes, metrics, labels):
        vals = [r[m] for r in results]
        bars = ax.bar(ds_names, vals, color=["#4C72B0","#DD8452","#55A868"])
        ax.set_title(lbl, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(vals),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("DINOv2-base SAE Eval Summary\n"
                 "out/checkpoints/4s0816zv  |  layer -1  |  d_sae=49152",
                 fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "summary_metrics.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
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
    layer_idx  = 12 + cfg.block_layer if cfg.block_layer < 0 else cfg.block_layer  # 11
    print(f"Hooked layer index: {layer_idx}")

    all_results = []

    for ds_name, ds_info in DATASETS.items():
        print(f"\n{'─'*60}")
        print(f"Dataset: {ds_name}")
        metrics, feat_fire, topk_buf, ds = eval_dataset(
            dino, sae, proc, ds_name, ds_info, layer_idx
        )
        all_results.append(metrics)

        # ── Print metrics ─────────────────────────────────────────────────
        print(f"  L0               : {metrics['l0']:.2f}")
        print(f"  Recon MSE        : {metrics['recon_mse']:.5f}")
        print(f"  Explained var    : {metrics['explained_var']:.4f}")
        print(f"  Dead features    : {metrics['dead_pct']:.2f}%  "
              f"({cfg.d_sae - metrics['n_active']:,} / {cfg.d_sae:,})")
        print(f"  Active features  : {metrics['n_active']:,}")
        print(f"  Label entropy wt : {metrics['h_wt_norm']:.4f}  "
              f"[0=monosemantic, 1=uniform]")
        print(f"  Label entropy uw : {metrics['h_uw_norm']:.4f}")

        # ── Plots ─────────────────────────────────────────────────────────
        print(f"  Generating plots...")
        plot_top_activating_grid(ds, topk_buf, feat_fire, ds_name)
        plot_feature_activation_histogram(feat_fire, ds_name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n")
    print("=" * 90)
    print(f"{'Dataset':<14} {'L0':>8} {'EV':>8} {'MSE':>10} "
          f"{'Dead%':>8} {'Active':>8} {'H_wt':>8} {'H_uw':>8}")
    print("=" * 90)
    for r in all_results:
        print(f"{r['dataset']:<14} {r['l0']:>8.2f} {r['explained_var']:>8.4f} "
              f"{r['recon_mse']:>10.5f} {r['dead_pct']:>8.2f} "
              f"{r['n_active']:>8,} {r['h_wt_norm']:>8.4f} {r['h_uw_norm']:>8.4f}")
    print("=" * 90)

    plot_summary_bar(all_results)

    # Save JSON
    out_json = "out/dinov2_sae_eval_full.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved to {out_json}")
    print(f"Plots saved to   {OUT_DIR}/")


if __name__ == "__main__":
    main()

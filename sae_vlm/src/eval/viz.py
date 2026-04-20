"""Visualisation helpers for SAE feature analysis."""

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def _shorten(name: str, maxlen: int = 18) -> str:
    return name if len(name) <= maxlen else name[:maxlen - 1] + "…"


def plot_top_activating_grid(
    dataset,
    artifacts: dict,
    classnames: List[str],
    out_dir: str,
    model_tag: str = "",
    dataset_tag: str = "",
    n_feats: int = 16,
    n_imgs: int = 8,
):
    """Feature gallery: info panel | top images | class bar chart."""
    alive     = np.where(artifacts["alive_mask"])[0]
    cls_cnt   = artifacts["cls_sae_cnt"]           # [N_CLASS, N_FEAT]
    sparsity  = artifacts["sae_sparsity"].numpy() if hasattr(artifacts["sae_sparsity"], "numpy") else artifacts["sae_sparsity"]

    has_cls   = cls_cnt[:, alive].sum(0) > 0
    not_noisy = sparsity[alive] < 0.10
    valid     = alive[has_cls & not_noisy]
    if len(valid) == 0:
        valid = alive[has_cls]
    if len(valid) == 0:
        print("  [skip] no features with class data for visualisation")
        return

    sel_valid = artifacts["selectivity"][valid]
    n_feats   = min(n_feats, len(valid))
    top_feats = valid[np.argsort(sel_valid)[::-1]][:n_feats]

    max_indices = artifacts["max_activating_image_indices"]
    max_values  = artifacts["max_activating_image_values"]
    n_stored    = min(n_imgs, max_indices.shape[1])

    col_widths = [1.8] + [1.0] * n_stored + [1.4]
    n_cols = len(col_widths)
    row_h  = 2.6
    fig = plt.figure(figsize=(sum(col_widths) * 1.05, n_feats * row_h + 0.8))
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
        fire_count = int(sparsity[feat_id] * artifacts.get("n_images", 100_000))
        color      = palette[row % len(palette)]

        feat_cls = cls_cnt[:, feat_id]
        total    = feat_cls.sum()
        top5_idx   = np.argsort(feat_cls)[::-1][:5]
        top5_names = [_shorten(classnames[i]) for i in top5_idx if i < len(classnames)]
        top5_frac  = feat_cls[top5_idx] / max(total, 1)

        ax_info = fig.add_subplot(gs[row, 0])
        ax_info.axis("off")
        ax_info.add_patch(plt.Rectangle((0, 0.82), 1, 0.18,
                                         transform=ax_info.transAxes,
                                         color=color, alpha=0.85, clip_on=False))
        ax_info.text(0.5, 0.91, f"Feature {feat_id}",
                     transform=ax_info.transAxes,
                     ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        info_txt = (f"Selectivity: {sel:.3f}\n"
                    f"Top-1 frac:  {t1:.3f}\n"
                    f"Fire count:  ~{fire_count:,}\n"
                    f"Top class:\n  {_shorten(classnames[top5_idx[0]], 20)}")
        ax_info.text(0.05, 0.78, info_txt, transform=ax_info.transAxes,
                     ha="left", va="top", fontsize=7, linespacing=1.55, family="monospace")
        ax_info.add_patch(plt.Rectangle((0, 0), 0.04, 1, transform=ax_info.transAxes,
                                         color=color, alpha=0.9, clip_on=False))

        for col in range(n_stored):
            ax_img = fig.add_subplot(gs[row, col + 1])
            ax_img.axis("off")
            gi  = max_indices[feat_id, col].item()
            val = max_values[feat_id, col].item()
            if gi == 0 and val == 0:
                ax_img.set_facecolor("#f0f0f0")
                continue
            try:
                item     = dataset[int(gi)]
                pil_img  = item["image"].convert("RGB").resize((160, 160))
                img_lbl  = item.get("label", -1)
                cls_name = _shorten(classnames[img_lbl]) if 0 <= img_lbl < len(classnames) else ""
                ax_img.imshow(pil_img, aspect="auto")
                for spine in ax_img.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                ax_img.set_xlabel(f"{cls_name}\nact={val:.1f}",
                                   fontsize=6, labelpad=2, color="#333333")
            except Exception:
                ax_img.text(0.5, 0.5, "err", ha="center", va="center", fontsize=8, color="gray")

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

    title = f"Top most-selective SAE features"
    if model_tag:
        title += f" — {model_tag}"
    if dataset_tag:
        title += f" on {dataset_tag}"
    fig.suptitle(title + "\nSorted by selectivity (1 − H_norm).  Each row = one SAE feature.",
                 fontsize=10, fontweight="bold")

    path = os.path.join(out_dir, "top_monosemantic_features.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_monosemanticity_histograms(artifacts: dict, out_dir: str,
                                     model_tag: str = "", dataset_tag: str = ""):
    mask      = artifacts["alive_mask"]
    sel       = artifacts["selectivity"][mask]
    top1      = artifacts["top1_fraction"][mask]
    sparsity_ = artifacts["sae_sparsity"]
    sparsity  = (sparsity_.numpy() if hasattr(sparsity_, "numpy") else sparsity_)[mask]

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
    ax.set_title(f"Selectivity\nmean={sel.mean():.3f}  |  {(sel>0.5).mean()*100:.1f}% > 0.5")
    ax.legend(fontsize=8)

    ax = axes[2]
    import numpy as np_
    ax.hist(np_.log10(sparsity + 1e-6), bins=min(60, len(sparsity)), color="#DD8452", edgecolor="none")
    ax.set_xlabel("log10(sparsity)")
    ax.set_ylabel("# features")
    ax.set_title(f"Feature Sparsity\n{len(sel):,} alive features")

    thr = artifacts.get("adaptive_noisy_thr", "?")
    tag = f"{model_tag} on {dataset_tag}" if model_tag or dataset_tag else ""
    plt.suptitle(f"SAE Monosemanticity {tag}\nAdaptive noisy thr={thr:.3f}", fontsize=9)
    plt.tight_layout()

    path = os.path.join(out_dir, "monosemanticity_histograms.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_class_feature_heatmap(artifacts: dict, classnames: List[str], out_dir: str,
                                 top_n_classes: int = 30, top_n_feats: int = 50):
    cls_cnt = artifacts["cls_sae_cnt"]
    alive   = np.where(artifacts["alive_mask"])[0]
    if len(alive) == 0:
        print("  [skip] no alive features for heatmap")
        return

    top_n_feats   = min(top_n_feats, len(alive))
    top_n_classes = min(top_n_classes, cls_cnt.shape[0])

    feat_totals = cls_cnt[:, alive].sum(0)
    top_f_local = np.argsort(feat_totals)[::-1][:top_n_feats]
    top_f_idx   = alive[top_f_local]

    class_totals = cls_cnt[:, top_f_idx].sum(1)
    top_c_idx    = np.argsort(class_totals)[::-1][:top_n_classes]

    sub      = cls_cnt[np.ix_(top_c_idx, top_f_idx)]
    sub_norm = sub / (sub.max(0, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(max(10, top_n_feats * 0.35), max(6, top_n_classes * 0.28)))
    im = ax.imshow(sub_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(top_n_classes))
    ax.set_yticklabels([classnames[i] if i < len(classnames) else str(i) for i in top_c_idx], fontsize=7)
    ax.set_xticks(range(top_n_feats))
    ax.set_xticklabels([f"F{i}" for i in top_f_idx], fontsize=5, rotation=90)
    ax.set_xlabel("SAE Feature")
    ax.set_ylabel("Class")
    ax.set_title(f"Class × Feature Activation Heatmap "
                 f"(top {top_n_feats} alive features × top {top_n_classes} classes)\n"
                 "Normalised per feature — bright = class dominates")
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()

    path = os.path.join(out_dir, "class_feature_heatmap.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_mean_acts_and_sparsity(artifacts: dict, out_dir: str, tag: str = ""):
    mean_acts = artifacts["sae_mean_acts"]
    mean_acts = mean_acts.numpy() if hasattr(mean_acts, "numpy") else mean_acts
    sparsity  = artifacts["sae_sparsity"]
    sparsity  = sparsity.numpy() if hasattr(sparsity, "numpy") else sparsity
    dead      = artifacts["dead_mask"]
    thr       = artifacts.get("adaptive_noisy_thr", 0.1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    alive_ma = mean_acts[~dead]
    ax.hist(alive_ma, bins=min(80, len(alive_ma)), color="steelblue", edgecolor="none")
    ax.axvline(thr, color="red", lw=1.5, linestyle="--",
               label=f"noisy thr={thr:.2f}")
    ax.set_xlabel("Conditioned mean activation")
    ax.set_ylabel("# features")
    ax.set_title(f"SAE Mean Acts (alive)\n"
                 f"{(alive_ma > thr).sum():,} noisy  |  {(alive_ma <= thr).sum():,} clean")
    ax.legend(fontsize=8)

    ax = axes[1]
    alive_sp = sparsity[~dead]
    ax.hist(np.log10(alive_sp + 1e-8), bins=60, color="salmon", edgecolor="none")
    ax.set_xlabel("log10(sparsity)")
    ax.set_ylabel("# features")
    ax.set_title(f"SAE Feature Sparsity\n{len(alive_sp):,} alive features")

    plt.suptitle(f"Mean Acts & Sparsity Diagnostics {tag}", fontsize=10)
    plt.tight_layout()

    path = os.path.join(out_dir, "mean_acts_sparsity.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

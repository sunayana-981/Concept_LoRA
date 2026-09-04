#!/usr/bin/env python3
"""Method figure for masked SAE fine-tuning, built from the actual pipeline in
tasks/train_sae_masked_finetune.py (Steps 1-5) and src/sae_training/masked_sae_trainer.py
(gradient masking + decoder renorm)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ---- palette (dataviz skill reference palette) ----
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BLUE = "#2a78d6"       # target-domain / trainable / free units
GREY = "#d8d6cd"       # frozen / protected units fill
GREY_DARK = "#7a7870"  # protected border + text
RED = "#e34948"        # blocked gradient

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = INK

fig, ax = plt.subplots(figsize=(13.4, 5.6))
ax.set_xlim(0, 13.4)
ax.set_ylim(0, 5.6)
ax.axis("off")


def box(x, y, w, h, label, sub=None, fc="white", ec=INK, lw=1.4, fs=10.5,
        subfs=8.5, subcolor=INK_SECONDARY, zorder=3, text_color=INK, bold=True):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(b)
    cy = y + h / 2 + (0.12 if sub else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center", fontsize=fs,
             color=text_color, fontweight="bold" if bold else "normal", zorder=zorder + 1)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.17, sub, ha="center", va="center",
                 fontsize=subfs, color=subcolor, zorder=zorder + 1)
    return (x, y, w, h)


def arrow(p0, p1, color=INK, lw=1.6, style="-|>", zorder=2,
          connectionstyle="arc3,rad=0.0", label=None, label_pos=0.5, label_dy=0.14,
          fs=8.2, label_color=None, ls="solid"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                         linewidth=lw, color=color, linestyle=ls, zorder=zorder,
                         connectionstyle=connectionstyle, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        mx = p0[0] + (p1[0] - p0[0]) * label_pos
        my = p0[1] + (p1[1] - p0[1]) * label_pos
        ax.text(mx, my + label_dy, label, ha="center", va="bottom", fontsize=fs,
                 color=label_color or color, style="italic")


def panel_title(x, y, text, tagcolor):
    ax.add_patch(Rectangle((x, y), 0.16, 0.34, facecolor=tagcolor, edgecolor="none", zorder=4))
    ax.text(x + 0.28, y + 0.17, text, ha="left", va="center", fontsize=12.5,
             fontweight="bold", color=INK)


def label_chip(x, y, text, color, fs=8.6, bold=True):
    """Small white-backed label so text stays legible over hatched/tinted fills."""
    t = ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
                 fontweight="bold" if bold else "normal", zorder=7)
    t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.6))


# =====================================================================
# PANEL A (top) -- estimate protected units on the reference distribution
# =====================================================================
PA_Y = 4.05
panel_title(0.15, 5.05, "1  Estimate reference-active features (adapter-specific)", GREY_DARK)

a1 = box(0.3, PA_Y, 1.5, 0.85, "ImageNet", sub="reference images")
a2 = box(2.25, PA_Y, 1.75, 0.85, r"LoRA-CLIP  $\hat{L}$", sub="target adapter, block ℓ")
a3 = box(4.45, PA_Y, 1.75, 0.85, "Generic SAE  S", sub=r"$d_{\mathrm{SAE}}{=}49{,}152$")

arrow((a1[0] + a1[2], a1[1] + a1[3] / 2), (a2[0], a2[1] + a2[3] / 2))
arrow((a2[0] + a2[2], a2[1] + a2[3] / 2), (a3[0], a3[1] + a3[3] / 2))

# activity ranking mini-chart
bar_x0, bar_w, bar_h = 6.65, 2.35, 0.85
ax.add_patch(FancyBboxPatch((bar_x0, PA_Y), bar_w, bar_h, boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.4, edgecolor=INK, facecolor="white", zorder=3))
rng = np.random.default_rng(7)
heights = np.sort(rng.exponential(0.22, 16))[::-1]
heights = np.clip(heights, 0.05, 0.5)
bw = 0.115
x0s = bar_x0 + 0.2 + np.arange(16) * bw
for i, hgt in enumerate(heights):
    col = GREY_DARK if i < 3 else "#e4e2da"
    ax.add_patch(Rectangle((x0s[i], PA_Y + 0.14), bw * 0.72, hgt, facecolor=col,
                            edgecolor="none", zorder=4))
ax.plot([bar_x0 + 0.16, bar_x0 + 0.16 + 3 * bw], [PA_Y + 0.14 + 0.56, PA_Y + 0.14 + 0.56],
        color=RED, lw=1.3, ls=(0, (4, 2)), zorder=5)
ax.text(bar_x0 + bar_w / 2, PA_Y + bar_h + 0.09, "unit activity (ranked)", ha="center", va="bottom",
        fontsize=7.6, color=INK_SECONDARY)
ax.text(bar_x0 + bar_w / 2, PA_Y - 0.16, "top protect_frac = 20%", ha="center", va="top",
        fontsize=8.0, color=RED, style="italic")

arrow((a3[0] + a3[2], a3[1] + a3[3] / 2), (bar_x0, PA_Y + bar_h / 2))

mask_w = 2.4
mask_x = 9.75
maskbox = box(mask_x, PA_Y, mask_w, 0.85, "Protected mask ℙ", sub="9,830 / 49,152 (20%)",
              fc="#f3f2ee", ec=GREY_DARK, text_color=GREY_DARK, fs=9.8, subcolor=GREY_DARK)
arrow((bar_x0 + bar_w, PA_Y + bar_h / 2), (maskbox[0], maskbox[1] + maskbox[3] / 2), color=GREY_DARK)

# =====================================================================
# PANEL B (main) -- masked fine-tuning on the target domain
# =====================================================================
PB_Y = 1.15
panel_title(0.15, 3.05, "2  Masked fine-tune on the target domain", BLUE)

b1 = box(0.3, PB_Y, 1.5, 0.85, "{Target}", sub="e.g. EuroSAT")
b2 = box(2.25, PB_Y, 1.85, 0.85, r"LoRA-CLIP  $\hat{L}$", sub=r"$L + \Delta W$", text_color=BLUE)
arrow((b1[0] + b1[2], b1[1] + b1[3] / 2), (b2[0], b2[1] + b2[3] / 2))

# SAE split box: protected (left, grey/hatched) + free (right, blue)
sae_x, sae_y, sae_w, sae_h = 4.85, PB_Y - 0.35, 3.2, 1.75
ax.add_patch(FancyBboxPatch((sae_x, sae_y), sae_w, sae_h, boxstyle="round,pad=0.03,rounding_size=0.1",
                             linewidth=1.8, edgecolor=INK, facecolor="white", zorder=3))
ax.text(sae_x + sae_w / 2, sae_y + sae_h + 0.1, "Sparse Autoencoder  S   (= ℙ from Step 1)",
        ha="center", va="bottom", fontsize=9.6, fontweight="bold", color=INK)

split_x = sae_x + sae_w * 0.42
prot = Rectangle((sae_x + 0.08, sae_y + 0.08), split_x - sae_x - 0.16, sae_h - 0.16,
                  facecolor=GREY, edgecolor=GREY_DARK, linewidth=1.1, hatch="///", zorder=4, alpha=0.9)
ax.add_patch(prot)
free = Rectangle((split_x + 0.08, sae_y + 0.08), sae_x + sae_w - split_x - 0.16, sae_h - 0.16,
                  facecolor="#eaf2fc", edgecolor=BLUE, linewidth=1.1, zorder=4)
ax.add_patch(free)

prot_cx = sae_x + (split_x - sae_x) / 2
free_cx = split_x + (sae_x + sae_w - split_x) / 2

label_chip(prot_cx, sae_y + sae_h - 0.34, "PROTECTED  (20%)", GREY_DARK, fs=8.6)
label_chip(prot_cx, sae_y + 0.32, "unit gradients\nmasked", RED, fs=8.2)
label_chip(free_cx, sae_y + sae_h - 0.34, "FREE  (80%)", BLUE, fs=8.6)
label_chip(free_cx, sae_y + 0.32, "trainable", BLUE, fs=8.6)

arrow((b2[0] + b2[2], b2[1] + b2[3] / 2), (sae_x, sae_y + sae_h / 2), lw=1.8)

recon = box(8.55, PB_Y, 1.55, 0.85, r"$\hat{z}$", sub="reconstruction")
arrow((sae_x + sae_w, sae_y + sae_h / 2), (recon[0], recon[1] + recon[3] / 2), lw=1.8)

loss = box(10.6, PB_Y, 1.65, 0.85, "Loss", sub=r"MSE $+\ \lambda\Vert h \Vert_1$", ec=INK)
arrow((recon[0] + recon[2], recon[1] + recon[3] / 2), (loss[0], loss[1] + loss[3] / 2), lw=1.8)

# --- gradient feedback: routed BELOW the main row, re-entering the Free block
# from below, so it never crosses Panel A or any text ---
fb_y = PB_Y - 0.75
arrow((loss[0] + loss[2] / 2, loss[1]), (loss[0] + loss[2] / 2, fb_y), color=BLUE, lw=1.8,
      connectionstyle="arc3,rad=0.0")
arrow((loss[0] + loss[2] / 2, fb_y), (free_cx, fb_y), color=BLUE, lw=1.8,
      connectionstyle="arc3,rad=0.0")
arrow((free_cx, fb_y), (free_cx, sae_y), color=BLUE, lw=1.8,
      connectionstyle="arc3,rad=0.0")
ax.text((loss[0] + loss[2] / 2 + free_cx) / 2, fb_y + 0.09, r"masked $\nabla$ updates free feature parameters",
        ha="center", va="bottom", fontsize=8.2, color=BLUE, style="italic")

# decoder renorm footnote
ax.text(sae_x + sae_w / 2, sae_y - 0.18,
        r"all decoder rows renormalized; shared $b_{\rm dec}$ remains trainable",
        ha="center", va="top", fontsize=7.8, color=INK_MUTED, style="italic")

ax.text(sae_x + sae_w / 2, fb_y - 0.28,
        r"$\mathcal{L} = \Vert S(\hat{H}(x)) - \hat{H}(x)\Vert_2^2 + \lambda_{\ell_1}\Vert h(\hat{H}(x))\Vert_1,"
        r"\qquad \hat{H}(x) = \hat{L}(x)$",
        ha="center", va="center", fontsize=9.5, color=INK)

fig.tight_layout()
out_pdf = "/home/sunayana/Documents/Concept_LoRA/eccv_workshop/figures/method_overview.pdf"
out_png = "/home/sunayana/Documents/Concept_LoRA/eccv_workshop/figures/method_overview.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15)
fig.savefig(out_png, bbox_inches="tight", pad_inches=0.15, dpi=200)
print("saved", out_pdf, out_png)

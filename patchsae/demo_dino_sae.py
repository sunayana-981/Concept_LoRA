#!/usr/bin/env python3
"""
DINOv2-base SAE Demo — mirrors demo.ipynb / SAETester logic.

Given an input image:
  1. show_patches()               → 16×16 patch grid with a highlighted patch
  2. get_top_neurons(patch_idx)   → top SAE features for that patch token
  3. show_top_images(neuron_idx)  → top activating ImageNet images per neuron
                                    (from pre-computed max_activating_image_indices.pt)
  4. get_segmentation_mask(feat)  → spatial SAE activation heatmap overlay
  5. run(patch_idx)               → full pipeline in one call, saves all plots

DINOv2 patch geometry (patch_size=14, image=224×224):
  224 / 14 = 16 patches per side  →  256 patch tokens + 1 CLS = 257 total tokens
  token 0          = CLS
  token 1 .. 256   = patches, row-major: token(i,j) = 1 + i*16 + j

Note: SAE was trained on CLS activations. Applying it to patch tokens is an
extrapolation from the same layer / same d_model — useful for spatial
visualisation but not calibrated to the training distribution.

Requirements: run eval_dino_sae_imagenet.py first to produce
  out/feature_data/dinov2_base/imagenet/

Usage:
  cd /home/sunayana/Documents/Concept_LoRA/patchsae
  python demo_dino_sae.py --image /path/to/image.jpg --patch_row 8 --patch_col 8
  python demo_dino_sae.py --image /path/to/image.jpg --patch_idx 88 --top_k 5 --num_images 8
"""

import argparse, math, os, sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# ── Config ────────────────────────────────────────────────────────────────────

SAE_PATH         = "out/checkpoints/4s0816zv/final_sparse_autoencoder_facebook/dinov2-base_-1_resid_49152.pt"
FEAT_DATA_DIR    = "out/feature_data/dinov2_base/imagenet"
CLASSNAMES_PATH  = "configs/classnames/imagenet_classnames.txt"
OUT_DIR          = "out/demo_dino"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE   = 14   # DINOv2 patch size in pixels
IMAGE_SIZE   = 224  # input resolution
PATCHES_SIDE = IMAGE_SIZE // PATCH_SIZE   # 16
N_PATCHES    = PATCHES_SIDE ** 2          # 256
N_TOKENS     = N_PATCHES + 1              # 257 (incl. CLS at idx 0)

NOISY_THRESHOLD = 0.1   # SAETester default (CLIP-calibrated); overridden adaptively for DINOv2

os.makedirs(OUT_DIR, exist_ok=True)

# ── Utilities ─────────────────────────────────────────────────────────────────

def _load_classnames(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def _shorten(s, n=20):
    return s if len(s) <= n else s[:n-1] + "…"


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_sae(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    sae  = SparseAutoencoder(cfg, device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(device)
    return sae, cfg


def _load_dino(device):
    proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").eval().to(device)
    return model, proc


# ── DinoDemoSAE class (mirrors SAETester in src/demo/core.py) ─────────────────

class DinoDemoSAE:
    """
    Interactive SAE demo for DINOv2-base.  Mirrors SAETester from demo.ipynb.

    Patch indexing (matches demo.ipynb convention):
      patch_idx is 1-based in the demo (0 = CLS).  Here we keep the same:
        token_idx = patch_idx   (0 = CLS, 1..256 = patches)
    """

    def __init__(
        self,
        dino,
        proc,
        sae,
        sae_mean_acts,           # [N_FEAT] — from sae_mean_acts.pt
        max_act_indices,         # [N_FEAT, K] — from max_activating_image_indices.pt
        max_act_values,          # [N_FEAT, K]
        dataset,                 # HF ImageNet dataset (for top-image retrieval)
        classnames,
        noisy_threshold=NOISY_THRESHOLD,
        device=DEVICE,
    ):
        self.dino             = dino
        self.proc             = proc
        self.sae              = sae
        mean_np = sae_mean_acts.numpy()  # [N_FEAT]
        self.sae_mean_acts    = mean_np
        self.max_act_indices  = max_act_indices        # [N_FEAT, K]
        self.max_act_values   = max_act_values         # [N_FEAT, K]
        self.dataset          = dataset
        self.classnames       = classnames
        # Adapt noisy threshold: 95th percentile of alive-feature mean_acts
        # (0.1 is CLIP-calibrated; DINOv2 acts are ~10-30x larger)
        from src.sae_training.config import Config as _Cfg  # avoid circular at module level
        alive_mean = mean_np[mean_np > 0]
        if len(alive_mean) > 0:
            adaptive = float(np.percentile(alive_mean, 95))
            self.noisy_threshold = adaptive
            print(f"  Adaptive noisy threshold: {adaptive:.4f}  (95th-pct of {len(alive_mean)} alive features)")
        else:
            self.noisy_threshold = noisy_threshold
        self.device           = device
        self._image           = None
        self._pixel_values    = None
        self._sae_acts        = None  # [1, N_TOKENS, N_FEAT] after register_image

    # ── Image registration ────────────────────────────────────────────────────

    def register_image(self, path):
        """Load an image and run it through DINOv2 + SAE (all 257 tokens)."""
        img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        self._image = img

        pv = self.proc(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        self._pixel_values = pv

        # DINOv2 forward — capture layer output via hook
        layer = self.dino.encoder.layer[11]
        captured = {}
        def _hook(_m, _i, out):
            captured["hs"] = (out[0] if isinstance(out, tuple) else out).detach().float()
        h = layer.register_forward_hook(_hook)
        with torch.no_grad():
            self.dino(pixel_values=pv)
        h.remove()

        hs = captured["hs"]  # [1, 257, 768]

        # Apply SAE to ALL tokens (same encoder, same d_model)
        # Shape: [257, 768] → SAE → [257, N_FEAT]
        all_tokens = hs[0]  # [257, 768]
        with torch.no_grad():
            _, cache = self.sae.run_with_cache(all_tokens)
        feat_acts = cache["hook_hidden_post"]  # [257, N_FEAT]
        self._sae_acts = feat_acts.cpu()       # keep on CPU

        print(f"Registered image: {path}")
        print(f"  SAE acts shape:  {self._sae_acts.shape}  (tokens × features)")
        print(f"  CLS L0={( self._sae_acts[0] > 0).sum().item()}  "
              f"patch mean L0={( self._sae_acts[1:] > 0).float().sum(-1).mean().item():.1f}")

    # ── Noisy filter (mirrors SAETester._filter_out_nosiy_activation) ─────────

    def _filter_noisy(self, acts_1d_or_2d):
        """Zero out features whose global mean_act > noisy_threshold."""
        noisy_idx = (self.sae_mean_acts > self.noisy_threshold).nonzero()[0].tolist()
        out = deepcopy(acts_1d_or_2d)
        if out.ndim == 1:
            out[noisy_idx] = 0
        else:
            out[:, noisy_idx] = 0
        return out

    # ── show_patches (mirrors SAETester.show_patches / _plot_patches) ─────────

    def show_patches(self, highlight_patch_idx=None, save_path=None):
        """
        Show the image as a 16×16 grid of patches.
        highlight_patch_idx: 0-based patch index (0 = top-left).
        """
        img_arr = np.array(self._image)  # [224, 224, 3]
        fig, axes = plt.subplots(PATCHES_SIDE, PATCHES_SIDE, figsize=(8, 8))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        for i in range(PATCHES_SIDE):
            for j in range(PATCHES_SIDE):
                patch_idx_0based = i * PATCHES_SIDE + j
                y0, y1 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
                x0, x1 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE
                patch = img_arr[y0:y1, x0:x1]

                ax = axes[i, j]
                ax.imshow(patch)
                ax.axis("off")

                if highlight_patch_idx is not None and patch_idx_0based == highlight_patch_idx:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(3)
                    ax.set_xticks([])
                    ax.set_yticks([])

        fig.suptitle(
            f"Image patches (16×16, patch_size=14px)"
            + (f"  —  highlighted patch {highlight_patch_idx}" if highlight_patch_idx is not None else ""),
            fontsize=10,
        )
        path = save_path or os.path.join(OUT_DIR, "patches.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Patches saved: {path}")
        return path

    # ── get_top_neurons (mirrors SAETester.get_top_neurons) ───────────────────

    def get_top_neurons(self, patch_idx=None, top_k=5):
        """
        Get the top-K SAE features for a specific patch token.
        patch_idx: token index (0 = CLS, 1..256 = patches).
                   If None, uses the mean across all patch tokens.
        Returns list of feature indices (numpy ints).
        """
        assert self._sae_acts is not None, "call register_image() first"
        acts = self._sae_acts.numpy()  # [257, N_FEAT]

        if patch_idx is None:
            # Mean across all patch tokens (skip CLS), then filter noisy
            token_acts = self._filter_noisy(acts[1:].mean(0))   # [N_FEAT]
        else:
            token_acts = self._filter_noisy(acts[patch_idx])    # [N_FEAT]

        top_neurons = np.argsort(token_acts)[::-1][:top_k]

        print(f"\nTop {top_k} neurons for {'CLS' if patch_idx==0 else f'patch {patch_idx}' if patch_idx else 'mean patches'}:")
        for rank, f in enumerate(top_neurons):
            act_val = token_acts[f]
            class_hint = ""
            print(f"  #{rank+1}  F{f:5d}  act={act_val:.3f}{class_hint}")

        return top_neurons.tolist()

    # ── get_segmentation_mask (mirrors SAETester.get_segmentation_mask) ───────

    def get_segmentation_mask(self, feat_idx, save_path=None):
        """
        Show how feature feat_idx activates spatially across the 16×16 patch grid.
        Returns a PIL image of the overlay.
        """
        assert self._sae_acts is not None, "call register_image() first"
        acts = self._sae_acts.numpy()  # [257, N_FEAT]

        # Patch token activations for this feature (skip CLS token 0)
        patch_acts = acts[1:, feat_idx]   # [256]
        patch_acts = patch_acts.reshape(PATCHES_SIDE, PATCHES_SIDE)  # [16, 16]

        # Upsample to image size
        mask_t = torch.tensor(patch_acts).unsqueeze(0).unsqueeze(0).float()  # [1,1,16,16]
        mask   = F.interpolate(mask_t, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                               align_corners=False)[0, 0].numpy()
        mask   = (mask - mask.min()) / (mask.max() - mask.min() + 1e-10)

        # Overlay on original image (darken non-activating regions — same as demo)
        img_arr = np.array(self._image)[..., :3]
        base_opacity = 30
        rgba = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
        rgba[..., :3] = img_arr
        darkened = (img_arr * (base_opacity / 255)).astype(np.uint8)
        rgba[mask == 0, :3] = darkened[mask == 0]
        rgba[..., 3] = 255

        overlay = Image.fromarray(rgba)

        # Save side-by-side: original | heatmap | overlay
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(self._image)
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(patch_acts, cmap="hot", aspect="auto")
        axes[1].set_title(f"Feature {feat_idx} activation\n(16×16 patch grid)", fontsize=9)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\n(bright = high activation)", fontsize=9)
        axes[2].axis("off")

        plt.suptitle(f"Segmentation mask — Feature {feat_idx}", fontsize=10)
        plt.tight_layout()
        path = save_path or os.path.join(OUT_DIR, f"seg_mask_F{feat_idx}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Seg mask saved: {path}")
        return overlay

    # ── show_ref_images (mirrors SAETester.show_ref_images_of_neuron_indices) ──

    def show_ref_images_of_neuron_indices(self, neuron_indices, num_images=8,
                                          activation_context=None, save_path=None):
        """
        For each neuron in neuron_indices, show its top-num_images activating
        ImageNet images. Mirrors SAETester._plot_images style.

        activation_context: optional dict {feat_id: act_value} for the query image.
        """
        n_neurons = len(neuron_indices)
        n_imgs    = min(num_images, self.max_act_indices.shape[1])

        fig = plt.figure(figsize=(n_imgs * 1.9 + 2.5, n_neurons * 2.5 + 0.8))
        gs  = gridspec.GridSpec(
            n_neurons, n_imgs + 1,
            figure=fig,
            width_ratios=[1.8] + [1.0] * n_imgs,
            hspace=0.5, wspace=0.06,
            left=0.02, right=0.98, top=0.92, bottom=0.04,
        )

        palette = plt.cm.tab10.colors

        for row, feat_id in enumerate(neuron_indices):
            color = palette[row % len(palette)]
            query_act = activation_context.get(feat_id, None) if activation_context else None

            # ── Info panel ────────────────────────────────────────────────
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
            info = f"query act: {query_act:.2f}" if query_act else "global top images"
            ax_info.text(0.05, 0.75, info,
                         transform=ax_info.transAxes,
                         ha="left", va="top", fontsize=7, color="#444444")

            # ── Top images ────────────────────────────────────────────────
            for col in range(n_imgs):
                ax = fig.add_subplot(gs[row, col + 1])
                ax.axis("off")
                gi  = self.max_act_indices[feat_id, col].item()
                val = self.max_act_values[feat_id, col].item()
                if gi == 0 and val == 0:
                    continue
                try:
                    item    = self.dataset[int(gi)]
                    img     = item["image"].convert("RGB").resize((160, 160))
                    lbl     = item.get("label", -1)
                    cls_name = _shorten(self.classnames[lbl]) if 0 <= lbl < len(self.classnames) else ""
                    ax.imshow(img, aspect="auto")
                    for sp in ax.spines.values():
                        sp.set_edgecolor(color); sp.set_linewidth(2)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_xlabel(f"{cls_name}\nact={val:.1f}",
                                  fontsize=6, labelpad=2, color="#333")
                except Exception:
                    ax.text(0.5, 0.5, "err", ha="center", va="center", fontsize=8)

        fig.suptitle("Top activating ImageNet images per SAE feature",
                     fontsize=10, fontweight="bold")
        path = save_path or os.path.join(OUT_DIR, "top_images_neurons.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  Top images saved: {path}")
        return path

    # ── run (mirrors SAETester.run — the main pipeline) ───────────────────────

    def run(self, highlight_patch_idx=None, top_k=5, num_images=8, show_seg_masks=True):
        """
        Full pipeline:
          1. Show patches
          2. Get top neurons for the highlighted patch
          3. Show top activating images for those neurons
          4. (optionally) Show segmentation masks for each neuron
        """
        assert self._sae_acts is not None, "call register_image() first"

        # token_idx: patch_idx is 0-based (0=top-left patch).
        # Token 0 is CLS, so patch 0 = token 1.
        token_idx = (highlight_patch_idx + 1) if highlight_patch_idx is not None else None

        # 1. Patch grid
        patch_path = self.show_patches(highlight_patch_idx=highlight_patch_idx)

        # 2. Top neurons
        top_neurons = self.get_top_neurons(patch_idx=token_idx, top_k=top_k)

        # Activation values at the query patch for each top neuron
        if token_idx is not None:
            acts_np = self._sae_acts.numpy()
            query_acts = {f: float(acts_np[token_idx, f]) for f in top_neurons}
        else:
            query_acts = None

        # 3. Top activating images
        ref_path = self.show_ref_images_of_neuron_indices(
            top_neurons, num_images=num_images, activation_context=query_acts
        )

        # 4. Segmentation masks
        if show_seg_masks:
            for feat_id in top_neurons:
                self.get_segmentation_mask(feat_id)

        print(f"\nAll outputs → {OUT_DIR}/")
        return top_neurons


# ── Load pre-computed artifacts ───────────────────────────────────────────────

def load_feature_artifacts(feat_dir):
    max_idx  = torch.load(os.path.join(feat_dir, "max_activating_image_indices.pt"), map_location="cpu")
    max_val  = torch.load(os.path.join(feat_dir, "max_activating_image_values.pt"),  map_location="cpu")
    mean_act = torch.load(os.path.join(feat_dir, "sae_mean_acts.pt"), map_location="cpu")
    return max_idx, max_val, mean_act


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to query image")
    parser.add_argument("--patch_idx", type=int, default=None,
                        help="0-based patch index to analyse (0=top-left, 255=bottom-right). "
                             "Default: mean over all patches.")
    parser.add_argument("--patch_row", type=int, default=None,
                        help="Row (0-15) of patch to highlight (alternative to --patch_idx)")
    parser.add_argument("--patch_col", type=int, default=None,
                        help="Col (0-15) of patch to highlight (alternative to --patch_idx)")
    parser.add_argument("--top_k",      type=int, default=5,  help="Top neurons to find")
    parser.add_argument("--num_images", type=int, default=8,  help="Top images per neuron")
    parser.add_argument("--no_seg",     action="store_true",   help="Skip segmentation masks")
    parser.add_argument("--max_images", type=int, default=100_000)
    args = parser.parse_args()

    # Resolve patch index
    if args.patch_row is not None and args.patch_col is not None:
        patch_idx = args.patch_row * PATCHES_SIDE + args.patch_col
    else:
        patch_idx = args.patch_idx  # may be None

    print(f"\nDevice:     {DEVICE}")
    print(f"Image:      {args.image}")
    print(f"Patch idx:  {patch_idx}  (row={patch_idx//PATCHES_SIDE}, col={patch_idx%PATCHES_SIDE})"
          if patch_idx is not None else "Patch idx:  None (mean over all patches)")
    print("=" * 60)

    # Load everything
    classnames = _load_classnames(CLASSNAMES_PATH)
    print("Loading SAE...")
    sae, cfg = _load_sae(SAE_PATH, DEVICE)
    print("Loading DINOv2...")
    dino, proc = _load_dino(DEVICE)
    print("Loading artifacts...")
    max_idx, max_val, mean_act = load_feature_artifacts(FEAT_DATA_DIR)
    print(f"Loading ImageNet (first {args.max_images:,} images for retrieval)...")
    dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="train",
                           trust_remote_code=True).select(range(args.max_images))

    # Build demo object
    demo = DinoDemoSAE(
        dino=dino, proc=proc, sae=sae,
        sae_mean_acts=mean_act,
        max_act_indices=max_idx,
        max_act_values=max_val,
        dataset=dataset,
        classnames=classnames,
    )

    # Register image and run
    demo.register_image(args.image)
    demo.run(
        highlight_patch_idx=patch_idx,
        top_k=args.top_k,
        num_images=args.num_images,
        show_seg_masks=not args.no_seg,
    )


if __name__ == "__main__":
    main()

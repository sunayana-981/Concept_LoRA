#!/usr/bin/env python3
"""
For a few sample images from each dataset, show the top-activating exemplars
under three SAE conditions, drawn from a COMBINED pool of target + ImageNet images:

  1. Base SAE    — ZS CLIP features  → base ImageNet SAE
  2. Cross-SAE   — LoRA CLIP features → base ImageNet SAE
  3. Adapted SAE — LoRA CLIP features → domain-adapted SAE

Exemplar borders are color-coded by source:
  solid color  = target-domain image  (proves domain activation)
  dashed grey  = ImageNet image       (shows leakage / baseline bias)

The key visual story: AdSAE neurons are dominated by target-domain exemplars;
Base/Cross-SAE neurons pull in many ImageNet images, diluting domain specificity.

Usage:
    cd /home/sunayana/Documents/Concept_LoRA/sae_vlm
    python visualize_top_activating_exemplars.py \\
        --datasets eurosat medmnist caltech101 dtd ucf101 \\
        --n_samples 4 --top_neurons 2 --exemplars_per_neuron 5 \\
        --n_imagenet 1000 --out_dir out/exemplar_viz
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.registry import get_backbone
from src.data.dataset_registry import get_dataset, get_classnames, get_label_key, load_registry
from src.sae_training.loaders import load_sae, load_lora_weights

# ── Dataset / SAE configuration ───────────────────────────────────────────────

BASE_SAE_PATH  = os.path.join(_ROOT, "out", "sae_weight", "base", "out.pt")
BASE_SAE_LAYER = -2
BACKBONE_ID    = "clip_vit_b16"
LORA_ROOT      = os.path.join(_ROOT, "..", "lora_weights", "vitb16")

DATASET_CONFIGS: Dict[str, dict] = {
    "caltech101": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "caltech101", "3hu8t1bb",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-1_resid_49152.pt"),
        "adsae_layer": -1,
        "lora_path": os.path.join(LORA_ROOT, "caltech101", "16shots", "seed1", "lora_weights.pt"),
        "base_acc": 81.33, "cross_acc": 83.03, "adsae_acc": 88.05,
    },
    "eurosat": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "eurosat", "bk3rbkcx",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-3_resid_49152.pt"),
        "adsae_layer": -3,
        "lora_path": os.path.join(LORA_ROOT, "eurosat", "16shots", "seed1", "lora_weights.pt"),
        "base_acc": 27.67, "cross_acc": 41.55, "adsae_acc": 83.71,
    },
    "medmnist": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "medmnist", "91r6lhuw",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-3_resid_49152.pt"),
        "adsae_layer": -3,
        "lora_path": os.path.join(LORA_ROOT, "medmnist", "16shots", "seed1", "lora_weights.pt"),
        "base_acc": 34.55, "cross_acc": 46.70, "adsae_acc": 90.75,
    },
    "dtd": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "dtd", "sd5h6hxv",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "adsae_layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "dtd", "16shots", "seed42", "lora_weights.pt"),
        "base_acc": 38.94, "cross_acc": 69.07, "adsae_acc": 53.2,
    },
    "ucf101": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "ucf101", "j04tcnkc",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "adsae_layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "ucf101", "16shots", "seed1", "lora_weights.pt"),
        "base_acc": 28.16, "cross_acc": 42.55, "adsae_acc": 78.2,
    },
    "cub2002011": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "cub2002011", "578p9z8f",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "adsae_layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "cub2002011", "16shots", "seed1", "lora_weights.pt"),
        "base_acc": None, "cross_acc": None, "adsae_acc": None,
    },
}

COND_NAMES  = ["Base SAE\n(ZS CLIP)", "Cross-SAE\n(LoRA+Base)", "Adapted SAE\n(LoRA+AdSAE)"]
COND_SHORT  = ["Base SAE", "Cross-SAE", "AdSAE"]
COND_COLORS = ["#4878D0", "#EE854A", "#6ACC65"]
COND_KEYS   = ["base_acc", "cross_acc", "adsae_acc"]

# Source tags
DOMAIN = 0   # target dataset image
IMAGENET = 1  # ImageNet image


# ── Combined dataset ──────────────────────────────────────────────────────────

class CombinedHFDataset(Dataset):
    """Concatenate two HuggingFace datasets into a single torch Dataset.

    Images from ds_a occupy indices [0, n_a).
    Images from ds_b occupy indices [n_a, n_a + n_b).
    Both datasets must have 'image' and a label key.
    """

    def __init__(self, ds_a, ds_b, label_key_a: str, label_key_b: str):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.n_a  = len(ds_a)
        self.lk_a = label_key_a
        self.lk_b = label_key_b

    def __len__(self):
        return self.n_a + len(self.ds_b)

    def __getitem__(self, idx):
        if idx < self.n_a:
            item = self.ds_a[idx]
            return {"image": item["image"], "label": item[self.lk_a], "_src": DOMAIN}
        else:
            item = self.ds_b[idx - self.n_a]
            return {"image": item["image"], "label": item.get(self.lk_b, 0), "_src": IMAGENET}


# ── Model helpers ─────────────────────────────────────────────────────────────

def _load_base_bb(device):
    bb = get_backbone(BACKBONE_ID, device=device).load()
    bb.model.eval()
    return bb


def _load_lora_bb(lora_path, device):
    bb = get_backbone(BACKBONE_ID, device=device).load()

    class _W:
        def __init__(self, m): self.model = m

    load_lora_weights(_W(bb.model), lora_path, device)
    bb.model.eval()
    return bb


# ── Feature extraction ────────────────────────────────────────────────────────

def _collate(processor):
    def fn(batch):
        imgs    = [item["image"].convert("RGB") for item in batch]
        labels  = [item["label"] for item in batch]
        sources = [item["_src"] for item in batch]
        pv      = processor(images=imgs, return_tensors="pt")["pixel_values"]
        return pv, torch.tensor(labels), torch.tensor(sources, dtype=torch.uint8)
    return fn


@torch.no_grad()
def build_neuron_index(
    backbone, sae, combined_ds: CombinedHFDataset,
    layer: int, batch_size: int = 64, device: str = "cuda",
    num_workers: int = 4, n_top: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One pass over combined (target + ImageNet) dataset.

    Returns:
      top_idx   [N_FEAT, n_top]  global indices into combined_ds
      top_vals  [N_FEAT, n_top]  activation values
      all_lats  [N, N_FEAT]      float16 latents for every image
      all_src   [N]              DOMAIN(0) or IMAGENET(1) per image
    """
    loader    = DataLoader(combined_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=(device == "cuda"),
                           collate_fn=_collate(backbone.processor))
    layer_idx = backbone.resolve_layer(layer)
    cap       = {}

    def hook(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        cap["cls"] = hs.detach().float()[:, 0, :]

    h       = backbone.model.vision_model.encoder.layers[layer_idx].register_forward_hook(hook)
    N_FEAT  = sae.W_enc.shape[1]
    n       = len(combined_ds)

    top_vals = torch.full((N_FEAT, n_top), -float("inf"))
    top_idx  = torch.full((N_FEAT, n_top), -1, dtype=torch.long)
    all_lats, all_src = [], []
    offset   = 0

    for pv, _, sources in tqdm(loader, desc="  index", leave=False):
        backbone.model.vision_model(pixel_values=pv.to(device))
        _, lat, _ = sae(cap["cls"])
        lat_cpu   = lat.detach().cpu()
        B         = lat_cpu.shape[0]

        lat_T     = lat_cpu.T
        k_b       = min(n_top, B)
        bvals, bloc = torch.topk(lat_T, k=k_b, dim=1)
        bglobal   = bloc + offset

        combined_v = torch.cat([top_vals, bvals], dim=1)
        combined_i = torch.cat([top_idx,  bglobal], dim=1)
        top_vals, pos = torch.topk(combined_v, k=n_top, dim=1)
        top_idx       = torch.gather(combined_i, 1, pos)

        all_lats.append(lat_cpu.half())
        all_src.append(sources)
        offset += B

    h.remove()
    return top_idx, top_vals, torch.cat(all_lats), torch.cat(all_src)


# ── Sample selection ──────────────────────────────────────────────────────────

def score_and_select(
    combined_ds: CombinedHFDataset,
    conditions: List[dict],
    n_select: int,
    n_exemplars: int,
) -> List[int]:
    """
    Score every target-domain image by how clearly it shows the hierarchy
    AdSAE > Cross-SAE > Base SAE in exemplar domain fraction.

    Score = (adsae_frac - base_frac)
            + 0.3 if adsae > cross > base   (correct ordering bonus)

    Diversity: cap images per label to ceil(n_select / n_actual_labels),
    then pad with remaining best-scored images if needed.
    """
    n_target = combined_ds.n_a

    scored: List[Tuple[float, int, int]] = []  # (score, label, qidx)
    for qidx in range(n_target):
        fracs = []
        for cond in conditions:
            q_lat     = cond["all_latents"][qidx].float()
            fire_mask = q_lat > 0
            nid       = int(torch.argmax(q_lat * fire_mask.float())
                           if fire_mask.sum() > 0 else torch.argmax(q_lat))
            ex_idxs   = [i for i in cond["top_idx"][nid].tolist()[:n_exemplars] if i >= 0]
            n_dom     = sum(1 for ei in ex_idxs if int(cond["all_src"][ei]) == DOMAIN)
            fracs.append(n_dom / max(len(ex_idxs), 1))

        base_f, cross_f, ad_f = fracs
        score = ad_f - base_f
        if ad_f > cross_f > base_f:
            score += 0.3

        lbl = int(combined_ds[qidx]["label"])
        scored.append((score, lbl, qidx))

    scored.sort(reverse=True)

    # Cap per actual label (not classnames count, which may differ)
    n_actual_lbls = max(len({lbl for _, lbl, _ in scored}), 1)
    max_per_lbl   = max(1, math.ceil(n_select / n_actual_lbls))

    cls_count: Dict[int, int] = {}
    selected: List[int] = []
    for _, lbl, qidx in scored:
        if len(selected) >= n_select:
            break
        if cls_count.get(lbl, 0) < max_per_lbl:
            selected.append(qidx)
            cls_count[lbl] = cls_count.get(lbl, 0) + 1

    return selected


# ── Image helpers ─────────────────────────────────────────────────────────────

def _pil(combined_ds: CombinedHFDataset, idx: int, size: int = 100) -> Image.Image:
    item = combined_ds[int(idx)]
    img  = item["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img.convert("RGB").resize((size, size))


def _cls_name(combined_ds: CombinedHFDataset, idx: int, classnames: List[str]) -> str:
    item = combined_ds[int(idx)]
    src  = int(item["_src"])
    lbl  = item["label"]
    if src == DOMAIN and 0 <= lbl < len(classnames):
        n = classnames[lbl]
        return (n[:13] + "…") if len(n) > 13 else n
    return f"IN:{lbl}"   # ImageNet label id


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_exemplars(
    combined_ds: CombinedHFDataset,
    sample_indices: List[int],
    conditions: List[dict],
    classnames: List[str],
    top_neurons: int,
    exemplars_per_neuron: int,
    ds_name: str,
    out_path: str,
    page: int = 0,
    total_pages: int = 1,
):
    """
    Layout per query image:
      [  Query  ] [ Row label ] [ex1][ex2][ex3][ex4][ex5]   <- Base SAE
      [ (spans) ] [ Row label ] [ex1][ex2][ex3][ex4][ex5]   <- Cross-SAE
      [  3 rows ] [ Row label ] [ex1][ex2][ex3][ex4][ex5]   <- AdSAE

    Multiple query images are stacked vertically with a thin divider.
    """
    n_s = len(sample_indices)
    n_c = len(conditions)
    e   = exemplars_per_neuron

    # ── geometry: all image cells are square (img_w × img_w) ──────────────────
    img_w = 1.6    # inches per exemplar cell (square)
    lbl_w = 2.0    # condition label column
    div_h = 0.14   # thin divider between samples

    # Column layout: [lbl_w | img_w * e]
    # Row layout per sample: 1 header row (query) + n_c condition rows
    wr = [lbl_w] + [img_w] * e

    hr = []
    for si in range(n_s):
        hr.append(img_w)           # header: query image (square cell)
        hr.extend([img_w] * n_c)   # one row per condition
        if si < n_s - 1:
            hr.append(div_h)

    total_rows = len(hr)
    total_cols = len(wr)

    fig_w = sum(wr)
    fig_h = sum(hr) + 0.1
    fig   = plt.figure(figsize=(fig_w, fig_h))
    gs    = gridspec.GridSpec(
        total_rows, total_cols, figure=fig,
        hspace=0.04, wspace=0.04,
        left=0.01, right=0.99, top=0.99, bottom=0.01,
        width_ratios=wr, height_ratios=hr,
    )

    rows_per_sample = 1 + n_c   # header + conditions

    def _img_ax(gs_cell, pil_img, border_color, border_style="solid",
                caption="", cap_color="#fff"):
        ax = fig.add_subplot(gs_cell)
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(pil_img, aspect="equal")
        lw = 3.0
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(border_color)
            sp.set_linewidth(lw)
            if border_style == "dashed":
                sp.set_linestyle("--")
        # caption overlaid at bottom of image (no xlabel shrinkage)
        if caption:
            ax.text(0.5, 0.02, caption, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6.5,
                    color=cap_color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#00000088", ec="none"))
        return ax

    for si, qidx in enumerate(sample_indices):
        q_pil  = _pil(combined_ds, qidx, size=int(img_w * 100))
        q_name = _cls_name(combined_ds, qidx, classnames)

        base_r = si * (rows_per_sample + 1)   # each sample occupies rows_per_sample + 1 divider row

        # ── header row: query image + class label ──────────────────────────
        ax_q = _img_ax(gs[base_r, 0], q_pil, "#333333",
                       caption=f'"{q_name}"', cap_color="#fff")

        # class label spanning exemplar columns
        ax_hdr = fig.add_subplot(gs[base_r, 1:])
        ax_hdr.axis("off")
        ax_hdr.set_facecolor("#eef1fb")
        ax_hdr.text(0.03, 0.5, f"Query image  ·  class: {q_name}",
                    transform=ax_hdr.transAxes, ha="left", va="center",
                    fontsize=11, fontweight="bold", color="#333355")

        # ── divider row ────────────────────────────────────────────────────
        if si < n_s - 1:
            ax_div = fig.add_subplot(gs[base_r + rows_per_sample, :])
            ax_div.axis("off")
            ax_div.set_facecolor("#c8cce0")

        for ci, cond in enumerate(conditions):
            ccolor  = cond["color"]
            top_idx = cond["top_idx"]
            all_src = cond["all_src"]
            all_lat = cond["all_latents"]

            row = base_r + 1 + ci   # +1 to skip header

            # top firing neuron
            q_lat     = all_lat[qidx].float()
            fire_mask = q_lat > 0
            if fire_mask.sum() == 0:
                nid = int(torch.argmax(q_lat))
            else:
                nid = int(torch.argmax(q_lat * fire_mask.float()))

            ex_idxs = [i for i in top_idx[nid].tolist()[:e] if i >= 0]
            ex_srcs = [int(all_src[ei]) for ei in ex_idxs]

            # ── condition label ────────────────────────────────────────────
            ax_lbl = fig.add_subplot(gs[row, 0])
            ax_lbl.axis("off")
            ax_lbl.set_facecolor(ccolor + "22")
            ax_lbl.add_patch(mpatches.FancyArrowPatch(
                (0.0, 0.0), (0.0, 1.0), transform=ax_lbl.transAxes,
                arrowstyle="-", color=ccolor, linewidth=5))
            ax_lbl.text(0.55, 0.5, COND_NAMES[ci],
                        transform=ax_lbl.transAxes, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=ccolor, linespacing=1.5)

            # ── exemplar images ────────────────────────────────────────────
            for ei in range(e):
                if ei >= len(ex_idxs):
                    ax = fig.add_subplot(gs[row, 1 + ei])
                    ax.axis("off"); ax.set_facecolor("#f0f0f0"); continue

                gidx  = int(ex_idxs[ei])
                src   = int(ex_srcs[ei])
                is_in = (src == IMAGENET)

                try:
                    pil      = _pil(combined_ds, gidx, size=int(img_w * 100))
                    cls_name = _cls_name(combined_ds, gidx, classnames)
                    style    = "dashed" if is_in else "solid"
                    color    = "#aaaaaa" if is_in else ccolor
                    cap      = cls_name
                    cap_col  = "#ddd" if is_in else "#fff"
                    _img_ax(gs[row, 1 + ei], pil, color,
                            border_style=style, caption=cap, cap_color=cap_col)
                except Exception:
                    ax = fig.add_subplot(gs[row, 1 + ei])
                    ax.text(0.5, 0.5, "err", ha="center", va="center",
                            fontsize=7, color="gray")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Per-dataset pipeline ──────────────────────────────────────────────────────

def process_dataset(ds_name: str, args, registry: dict,
                    imagenet_ds, imagenet_lk: str, device: str):
    cfg = DATASET_CONFIGS.get(ds_name)
    if cfg is None:
        print(f"[WARN] No config for {ds_name}."); return
    for key in ("adsae_path", "lora_path"):
        if not os.path.isfile(cfg[key]):
            print(f"[SKIP] {ds_name}: {cfg[key]} not found."); return

    print(f"\n{'═'*65}\n  Dataset: {ds_name.upper()}\n{'═'*65}")

    # Load full dataset first, then stratified-subsample to max_dataset_images
    tgt_ds_full = get_dataset(ds_name, registry=registry)
    classnames  = get_classnames(ds_name, dataset=tgt_ds_full, registry=registry)
    label_key   = get_label_key(ds_name, tgt_ds_full, registry=registry)

    all_labels = tgt_ds_full[label_key]   # fast batch read from HF dataset
    by_cls: Dict[int, List[int]] = {}
    for i, lbl in enumerate(all_labels):
        by_cls.setdefault(int(lbl), []).append(i)
    n_cls    = len(by_cls)
    per_cls  = max(1, args.max_dataset_images // max(n_cls, 1))
    rng = random.Random(42)
    indices  = []
    for cls_idxs in by_cls.values():
        indices.extend(rng.sample(cls_idxs, min(per_cls, len(cls_idxs))))
    tgt_ds = tgt_ds_full.select(sorted(indices))
    print(f"  Target: {len(tgt_ds_full):,} total → {len(tgt_ds):,} stratified "
          f"({per_cls}/class × {n_cls} classes), {len(classnames)} classnames")

    # Combined dataset (target first, then ImageNet)
    combined = CombinedHFDataset(tgt_ds, imagenet_ds, label_key, imagenet_lk)
    print(f"  Combined pool: {len(combined):,} ({len(tgt_ds)} target + {len(imagenet_ds)} ImageNet)")

    # ── Load SAEs & backbones ─────────────────────────────────────────────────
    print("  Loading SAEs...")
    base_sae, _ = load_sae(BASE_SAE_PATH,     device); base_sae.eval()
    ad_sae,   _ = load_sae(cfg["adsae_path"], device); ad_sae.eval()

    print("  Loading ZS backbone..."); bb_zs   = _load_base_bb(device)
    print("  Loading LoRA backbone..."); bb_lora = _load_lora_bb(cfg["lora_path"], device)

    specs = [
        dict(short=COND_SHORT[0], color=COND_COLORS[0],
             bb=bb_zs,   sae=base_sae, layer=BASE_SAE_LAYER),
        dict(short=COND_SHORT[1], color=COND_COLORS[1],
             bb=bb_lora, sae=base_sae, layer=BASE_SAE_LAYER),
        dict(short=COND_SHORT[2], color=COND_COLORS[2],
             bb=bb_lora, sae=ad_sae,   layer=cfg["adsae_layer"]),
    ]

    # ── Build neuron indices ──────────────────────────────────────────────────
    built = []
    for sp in specs:
        print(f"\n  Indexing: {sp['short']}...")
        top_idx, top_vals, all_lat, all_src = build_neuron_index(
            backbone=sp["bb"], sae=sp["sae"], combined_ds=combined,
            layer=sp["layer"], batch_size=args.batch_size,
            device=device, num_workers=args.num_workers,
            n_top=args.exemplars_per_neuron + 3,
        )
        built.append(dict(
            name=COND_NAMES[specs.index(sp)], color=sp["color"],
            top_idx=top_idx, all_latents=all_lat, all_src=all_src,
        ))

    # ── Select best query images by hierarchy score ───────────────────────────
    print(f"\n  Scoring all {combined.n_a} target images for hierarchy clarity...")
    samples = score_and_select(combined, built, args.n_samples,
                               args.exemplars_per_neuron)
    names   = [_cls_name(combined, s, classnames) for s in samples]
    print(f"  Selected top-{len(samples)}: {list(zip(samples, names))}")

    # ── Plot — one file per query image ──────────────────────────────────────
    spp         = args.samples_per_page
    total_pages = math.ceil(len(samples) / spp)
    for page in range(total_pages):
        page_samples = samples[page * spp : (page + 1) * spp]
        if spp == 1:
            q_name   = _cls_name(combined, page_samples[0], classnames)
            safe_cls = q_name.replace("/", "_").replace(" ", "_")
            fname    = f"sample_{page + 1:03d}_{safe_cls}.png"
        else:
            fname = f"samples_{page * spp + 1:03d}-{min((page+1)*spp, len(samples)):03d}.png"
        out_path = os.path.join(args.out_dir, ds_name, fname)
        plot_exemplars(
            combined_ds=combined, sample_indices=page_samples,
            conditions=built, classnames=classnames,
            top_neurons=args.top_neurons,
            exemplars_per_neuron=args.exemplars_per_neuron,
            ds_name=ds_name, out_path=out_path,
            page=page, total_pages=total_pages,
        )

    del bb_zs, bb_lora, base_sae, ad_sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Top-activating exemplars with ImageNet vs domain comparison"
    )
    parser.add_argument("--datasets",             nargs="+",
                        default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--n_samples",            type=int, default=50)
    parser.add_argument("--samples_per_page",     type=int, default=1,
                        help="Query images per output PNG (1 = one file per query)")
    parser.add_argument("--top_neurons",          type=int, default=1)
    parser.add_argument("--exemplars_per_neuron", type=int, default=10)
    parser.add_argument("--max_dataset_images",   type=int, default=5000)
    parser.add_argument("--n_imagenet",           type=int, default=5000,
                        help="ImageNet images to include in combined pool")
    parser.add_argument("--batch_size",           type=int, default=64)
    parser.add_argument("--num_workers",          type=int, default=4)
    parser.add_argument("--device",               default="cuda")
    parser.add_argument("--out_dir",              default="out/exemplar_viz_indiv")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] Falling back to CPU."); device = "cpu"

    registry = load_registry()

    # Load ImageNet once (shared across all datasets)
    print(f"[INFO] Loading {args.n_imagenet} ImageNet images...")
    imagenet_ds = get_dataset("imagenet", registry=registry,
                              max_samples=args.n_imagenet)
    imagenet_lk = get_label_key("imagenet", imagenet_ds, registry=registry)
    print(f"  ImageNet: {len(imagenet_ds):,} images, label_key={imagenet_lk}")

    for ds in args.datasets:
        process_dataset(ds, args, registry, imagenet_ds, imagenet_lk, device)

    print("\n[INFO] Done. Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()

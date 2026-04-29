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
import os
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

    top_vals = torch.zeros(N_FEAT, n_top)
    top_idx  = torch.zeros(N_FEAT, n_top, dtype=torch.long)
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

def pick_samples(combined_ds: CombinedHFDataset, n_samples: int,
                 all_lats: torch.Tensor, classnames: List[str]) -> List[int]:
    """Pick n_samples diverse query images from the TARGET domain only."""
    n_target = combined_ds.n_a
    # Group target indices by class
    by_cls: Dict[int, List[int]] = {}
    for i in range(n_target):
        item = combined_ds.ds_a[i]
        lbl  = item.get(combined_ds.lk_a, 0)
        by_cls.setdefault(int(lbl), []).append(i)

    n_cls  = len(by_cls)
    step   = max(1, n_cls // n_samples)
    chosen = sorted(by_cls.keys())[::step][:n_samples]

    samples = []
    for cls in chosen:
        cands = [c for c in by_cls[cls] if c < all_lats.shape[0]]
        if not cands:
            continue
        acts  = all_lats[cands].float().sum(dim=1)
        best  = cands[int(acts.argmax())]
        samples.append(best)

    # pad if needed
    if len(samples) < n_samples:
        samples += list(range(n_samples - len(samples)))
    return samples[:n_samples]


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
    cfg: dict,
    out_path: str,
):
    n_s   = len(sample_indices)
    n_c   = len(conditions)
    e     = exemplars_per_neuron
    top_k = top_neurons
    n_tgt = combined_ds.n_a

    # ── geometry ──────────────────────────────────────────────────────────────
    img_w   = 1.08
    lbl_w   = 1.35
    qry_w   = 1.45
    sep_w   = 0.15

    # cols: [query | cond_label | (e imgs + sep) * top_k]
    wr = [qry_w, lbl_w]
    for ni in range(top_k):
        wr += [img_w] * e
        if ni < top_k - 1:
            wr.append(sep_w)

    total_cols = len(wr)
    row_h      = img_w + 0.58
    total_rows = n_s * (n_c + 1)   # +1 sample header per sample
    fig_h      = total_rows * row_h + 1.5

    fig = plt.figure(figsize=(sum(wr), fig_h))
    gs  = gridspec.GridSpec(
        total_rows, total_cols, figure=fig,
        hspace=0.52, wspace=0.04,
        left=0.01, right=0.99, top=0.94, bottom=0.01,
        width_ratios=wr,
    )

    # neuron column starts (accounting for label col + sep cols)
    neu_starts = []
    c = 2
    for ni in range(top_k):
        neu_starts.append(c)
        c += e + (1 if ni < top_k - 1 else 0)

    row = 0

    for si, qidx in enumerate(sample_indices):
        q_pil   = _pil(combined_ds, qidx, size=int(qry_w * 88))
        q_name  = _cls_name(combined_ds, qidx, classnames)

        # ── sample divider ────────────────────────────────────────────────────
        ax_hdr = fig.add_subplot(gs[row, :])
        ax_hdr.axis("off")
        ax_hdr.set_facecolor("#eef1fb")
        ax_hdr.text(0.01, 0.5, f"Sample {si+1}  ·  class: \"{q_name}\"",
                    transform=ax_hdr.transAxes, ha="left", va="center",
                    fontsize=9, fontweight="bold", color="#222244")
        row += 1

        for ci, cond in enumerate(conditions):
            ccolor  = cond["color"]
            top_idx = cond["top_idx"]      # [N_FEAT, n_top]
            all_src = cond["all_src"]      # [N] uint8
            all_lat = cond["all_latents"]  # [N, N_FEAT] float16
            acc_key = COND_KEYS[ci]

            # top firing neurons for the query
            q_lat       = all_lat[qidx].float()
            fire_mask   = q_lat > 0
            if fire_mask.sum() == 0:
                top_neu = torch.argsort(q_lat, descending=True)[:top_k].tolist()
            else:
                top_neu = torch.argsort(q_lat * fire_mask.float(),
                                        descending=True)[:top_k].tolist()

            # ── query image (once, spanning all condition sub-rows) ───────────
            ax_q = fig.add_subplot(gs[row, 0])
            ax_q.axis("off")
            if ci == 0:
                ax_q.imshow(q_pil, aspect="auto")
                for sp in ax_q.spines.values():
                    sp.set_visible(True); sp.set_edgecolor("#333"); sp.set_linewidth(2)
                ax_q.set_xticks([]); ax_q.set_yticks([])
                ax_q.set_xlabel(f"Query\n{q_name}", fontsize=7, labelpad=2, color="#333")
            else:
                ax_q.set_facecolor("#f8f8f8")

            # ── condition label ────────────────────────────────────────────────
            ax_lbl = fig.add_subplot(gs[row, 1])
            ax_lbl.axis("off")
            ax_lbl.set_facecolor(ccolor + "18")
            acc = cfg.get(acc_key)
            acc_str = f"\nAcc: {acc:.1f}%" if acc else ""
            ax_lbl.text(0.5, 0.55, COND_NAMES[ci] + acc_str,
                        transform=ax_lbl.transAxes, ha="center", va="center",
                        fontsize=7, fontweight="bold", color=ccolor, linespacing=1.4)
            ax_lbl.add_patch(mpatches.FancyArrowPatch(
                (0.0, 0.0), (0.0, 1.0), transform=ax_lbl.transAxes,
                arrowstyle="-", color=ccolor, linewidth=4))

            # ── exemplar groups ────────────────────────────────────────────────
            for ni in range(top_k):
                nid      = top_neu[ni] if ni < len(top_neu) else 0
                act_val  = float(q_lat[nid])
                ex_idxs  = top_idx[nid].tolist()   # global combined indices
                ex_srcs  = [int(all_src[ei]) for ei in ex_idxs]

                # Domain-fraction annotation on first exemplar
                n_dom = sum(1 for s in ex_srcs if s == DOMAIN)
                n_in  = sum(1 for s in ex_srcs if s == IMAGENET)
                frac  = n_dom / max(len(ex_srcs), 1)

                col0 = neu_starts[ni]
                for ei in range(e):
                    ax = fig.add_subplot(gs[row, col0 + ei])
                    ax.axis("off")
                    if ei >= len(ex_idxs):
                        ax.set_facecolor("#f0f0f0"); continue

                    gidx  = int(ex_idxs[ei])
                    src   = int(ex_srcs[ei])
                    is_in = (src == IMAGENET)

                    try:
                        pil      = _pil(combined_ds, gidx, size=92)
                        cls_name = _cls_name(combined_ds, gidx, classnames)

                        ax.imshow(pil, aspect="auto")

                        # border: solid condition color = target, dashed grey = ImageNet
                        if is_in:
                            ax.set_facecolor("#dddddd")
                            for sp in ax.spines.values():
                                sp.set_visible(True)
                                sp.set_edgecolor("#999999")
                                sp.set_linewidth(1.5)
                                sp.set_linestyle("--")
                        else:
                            for sp in ax.spines.values():
                                sp.set_visible(True)
                                sp.set_edgecolor(ccolor)
                                sp.set_linewidth(2.0)

                        ax.set_xticks([]); ax.set_yticks([])

                        # Caption: first exemplar gets neuron id + act + domain fraction
                        if ei == 0:
                            cap = (f"N#{nid}  act={act_val:.1f}\n"
                                   f"dom={n_dom}/{len(ex_srcs)} "
                                   f"({'▲' if frac >= 0.5 else '▼'}domain)")
                            font_color = "#116611" if frac >= 0.5 else "#882222"
                        else:
                            cap = f"{'[IN]' if is_in else ''}{cls_name}"
                            font_color = "#666" if is_in else "#222"

                        ax.set_xlabel(cap, fontsize=5.2, labelpad=1.5,
                                      color=font_color)
                    except Exception:
                        ax.text(0.5, 0.5, "err", ha="center", va="center",
                                fontsize=6, color="gray")

            row += 1

    # ── legend & suptitle ─────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor="white", edgecolor=COND_COLORS[0], linewidth=2,
                       label="Base SAE"),
        mpatches.Patch(facecolor="white", edgecolor=COND_COLORS[1], linewidth=2,
                       label="Cross-SAE"),
        mpatches.Patch(facecolor="white", edgecolor=COND_COLORS[2], linewidth=2,
                       label="AdSAE"),
        mpatches.Patch(facecolor="white", edgecolor="#999", linewidth=1.5,
                       linestyle="--", label="ImageNet exemplar"),
        mpatches.Patch(facecolor="white", edgecolor="#555", linewidth=2,
                       label="Target-domain exemplar"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    accs = [cfg.get(k) for k in COND_KEYS]
    acc_str = "  |  ".join(
        f"{s}: {a:.1f}%" for s, a in zip(COND_SHORT, accs) if a is not None
    )
    fig.suptitle(
        f"Top Activating Exemplars  (target + ImageNet pool)  ·  {ds_name}\n"
        f"Acc:  {acc_str}\n"
        "Solid borders = target-domain image.  Dashed grey = ImageNet image.  "
        "dom=X/Y = how many of Y top exemplars are from the target domain.\n"
        "AdSAE neurons are dominated by target-domain exemplars; "
        "Base/Cross-SAE pull in ImageNet images, diluting domain specificity.",
        fontsize=8.5, fontweight="bold", y=0.998, va="top",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
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

    tgt_ds     = get_dataset(ds_name, registry=registry,
                             max_samples=args.max_dataset_images)
    classnames = get_classnames(ds_name, dataset=tgt_ds, registry=registry)
    label_key  = get_label_key(ds_name, tgt_ds, registry=registry)
    print(f"  Target: {len(tgt_ds):,} images, {len(classnames)} classes")

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

    # ── Select sample query images ────────────────────────────────────────────
    print(f"\n  Selecting {args.n_samples} sample images...")
    samples = pick_samples(combined, args.n_samples, built[2]["all_latents"], classnames)
    names   = [_cls_name(combined, s, classnames) for s in samples]
    print(f"  Samples: {list(zip(samples, names))}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.out_dir, ds_name, "top_activating_exemplars.png")
    plot_exemplars(
        combined_ds=combined, sample_indices=samples,
        conditions=built, classnames=classnames,
        top_neurons=args.top_neurons,
        exemplars_per_neuron=args.exemplars_per_neuron,
        ds_name=ds_name, cfg=cfg, out_path=out_path,
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
    parser.add_argument("--n_samples",            type=int, default=4)
    parser.add_argument("--top_neurons",          type=int, default=2)
    parser.add_argument("--exemplars_per_neuron", type=int, default=5)
    parser.add_argument("--max_dataset_images",   type=int, default=2000)
    parser.add_argument("--n_imagenet",           type=int, default=1000,
                        help="ImageNet images to include in combined pool")
    parser.add_argument("--batch_size",           type=int, default=64)
    parser.add_argument("--num_workers",          type=int, default=4)
    parser.add_argument("--device",               default="cuda")
    parser.add_argument("--out_dir",              default="out/exemplar_viz")
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

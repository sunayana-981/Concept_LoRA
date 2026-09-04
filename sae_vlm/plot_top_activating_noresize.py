#!/usr/bin/env python3
"""
Top-activating exemplar plots with native-resolution query images.

Identical logic to visualize_top_activating_exemplars.py except:
  - Query (input) images are shown at their original resolution and aspect
    ratio -- _pil_orig() is used; no PIL .resize() call is made.
  - The GridSpec query column width is derived from the native aspect ratio so
    the image fills its cell without distortion.
  - Exemplar images are still thumbnail-resized for display efficiency.

Usage:
    cd /home/sunayana/Documents/Concept_LoRA/sae_vlm
    python plot_top_activating_noresize.py \
        --datasets eurosat medmnist \
        --n_samples 3 --exemplars_per_neuron 5 \
        --out_dir out/exemplar_viz_noresize
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
    },
    "eurosat": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "eurosat", "bk3rbkcx",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-3_resid_49152.pt"),
        "adsae_layer": -3,
        "lora_path": os.path.join(LORA_ROOT, "eurosat", "16shots", "seed1", "lora_weights.pt"),
    },
    "medmnist": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "medmnist", "91r6lhuw",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-3_resid_49152.pt"),
        "adsae_layer": -3,
        "lora_path": os.path.join(LORA_ROOT, "medmnist", "16shots", "seed1", "lora_weights.pt"),
    },
    "dtd": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "dtd", "sd5h6hxv",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "adsae_layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "dtd", "16shots", "seed42", "lora_weights.pt"),
    },
    "ucf101": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "ucf101", "j04tcnkc",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "adsae_layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "ucf101", "16shots", "seed1", "lora_weights.pt"),
    },
    "cub2002011": {
        "adsae_path": os.path.join(_ROOT, "out", "checkpoints", "cub2002011", "578p9z8f",
            "final_sparse_autoencoder_openai",
            "clip-vit-base-patch16_-2_resid_49152.pt"),
        "adsae_layer": -2,
        "lora_path": os.path.join(LORA_ROOT, "cub2002011", "16shots", "seed1", "lora_weights.pt"),
    },
}

COND_NAMES  = ["Base SAE\n(ZS CLIP)", "Cross-SAE\n(LoRA+Base)", "Adapted SAE\n(LoRA+AdSAE)"]
COND_SHORT  = ["Base SAE", "Cross-SAE", "AdSAE"]
COND_COLORS = ["#4878D0", "#EE854A", "#6ACC65"]

DOMAIN   = 0
IMAGENET = 1


# ── Combined dataset ──────────────────────────────────────────────────────────

class CombinedHFDataset(Dataset):
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

    top_vals = torch.full((N_FEAT, n_top), -float("inf"))
    top_idx  = torch.full((N_FEAT, n_top), -1, dtype=torch.long)
    all_lats, all_src = [], []
    offset   = 0

    for pv, _, sources in tqdm(loader, desc="  index", leave=False):
        backbone.model.vision_model(pixel_values=pv.to(device))
        _, lat, _ = sae(cap["cls"])
        lat_cpu   = lat.detach().cpu()
        B         = lat_cpu.shape[0]

        lat_T       = lat_cpu.T
        k_b         = min(n_top, B)
        bvals, bloc = torch.topk(lat_T, k=k_b, dim=1)
        bglobal     = bloc + offset

        combined_v  = torch.cat([top_vals, bvals], dim=1)
        combined_i  = torch.cat([top_idx,  bglobal], dim=1)
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
    n_target = combined_ds.n_a
    scored: List[Tuple[float, int, int]] = []
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

def _pil_orig(combined_ds: CombinedHFDataset, idx: int) -> Image.Image:
    """Return the image at its ORIGINAL resolution — no PIL resize."""
    item = combined_ds[int(idx)]
    img  = item["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img.convert("RGB")


def _pil_thumb(combined_ds: CombinedHFDataset, idx: int, size: int = 160) -> Image.Image:
    """Return the image resized to a square thumbnail (for exemplar cells only)."""
    item = combined_ds[int(idx)]
    img  = item["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img.convert("RGB").resize((size, size), Image.LANCZOS)


def _cls_name(combined_ds: CombinedHFDataset, idx: int, classnames: List[str]) -> str:
    item = combined_ds[int(idx)]
    src  = int(item["_src"])
    lbl  = item["label"]
    if src == DOMAIN and 0 <= lbl < len(classnames):
        n = classnames[lbl]
        return (n[:13] + "…") if len(n) > 13 else n
    return f"IN:{lbl}"


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_exemplars(
    combined_ds: CombinedHFDataset,
    sample_indices: List[int],
    conditions: List[dict],
    classnames: List[str],
    exemplars_per_neuron: int,
    ds_name: str,
    out_path: str,
):
    """
    Layout (columns): [ Query (native res) | Cond label | ex_1 … ex_e ]

    The query column width is computed from the native aspect ratio of the
    first query image so that imshow(aspect="auto") fills the cell exactly
    without any distortion.  Exemplar images (column 2+) are thumbnail-
    resized purely for display efficiency.

    Row layout: n_c rows per query sample, thin divider between samples.
    """
    n_s = len(sample_indices)
    n_c = len(conditions)
    e   = exemplars_per_neuron

    img_w = 1.6    # inches per square exemplar cell and per condition row height
    div_h = 0.14   # divider between samples
    lbl_w = 1.8    # condition-label column width (inches)

    # Derive query column width from the first image's native aspect ratio
    q0      = _pil_orig(combined_ds, sample_indices[0])
    nw, nh  = q0.size          # PIL: (width, height)
    # The query cell spans n_c rows each of height img_w; total cell height = n_c * img_w
    q_col_w = (n_c * img_w) * (nw / nh)

    # Column widths: [query | label | e exemplar cells]
    col_widths = [q_col_w, lbl_w] + [img_w] * e

    # Row heights: n_c per sample, divider between samples
    row_heights = []
    for si in range(n_s):
        row_heights.extend([img_w] * n_c)
        if si < n_s - 1:
            row_heights.append(div_h)

    fig = plt.figure(figsize=(sum(col_widths), sum(row_heights) + 0.1))
    gs  = gridspec.GridSpec(
        len(row_heights), len(col_widths), figure=fig,
        hspace=0.04, wspace=0.04,
        left=0.01, right=0.99, top=0.99, bottom=0.01,
        width_ratios=col_widths, height_ratios=row_heights,
    )

    def _draw_thumb(gs_cell, pil_img, border_color,
                    border_style="solid", caption="", cap_color="#fff"):
        ax = fig.add_subplot(gs_cell)
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(pil_img, aspect="equal")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(border_color)
            sp.set_linewidth(3.0)
            if border_style == "dashed":
                sp.set_linestyle("--")
        if caption:
            ax.text(0.5, 0.02, caption, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6.5, color=cap_color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#00000088", ec="none"))
        return ax

    for si, qidx in enumerate(sample_indices):
        # base row for this sample: si samples × (n_c data rows + 1 divider)
        base_r = si * (n_c + 1)

        # ── Query image: native resolution, spans all n_c condition rows ──────
        q_pil  = _pil_orig(combined_ds, qidx)   # NO PIL resize
        q_name = _cls_name(combined_ds, qidx, classnames)

        ax_q = fig.add_subplot(gs[base_r:base_r + n_c, 0])
        ax_q.set_xticks([]); ax_q.set_yticks([])
        # aspect="auto" fills the cell exactly; q_col_w was set to match native
        # aspect so no distortion occurs even though matplotlib stretches to fill.
        ax_q.imshow(q_pil, aspect="auto")
        for sp in ax_q.spines.values():
            sp.set_visible(True); sp.set_edgecolor("#333333"); sp.set_linewidth(3.0)
        ax_q.text(0.5, 0.02, f'"{q_name}"', transform=ax_q.transAxes,
                  ha="center", va="bottom", fontsize=7.5, color="#fff",
                  bbox=dict(boxstyle="round,pad=0.15", fc="#00000088", ec="none"))
        ax_q.set_title(f"Query · {ds_name}  [{nw}×{nh}px]",
                       fontsize=8, pad=2, color="#333")

        # ── Thin divider between samples ──────────────────────────────────────
        if si < n_s - 1:
            ax_div = fig.add_subplot(gs[base_r + n_c, :])
            ax_div.axis("off")
            ax_div.set_facecolor("#c8cce0")

        # ── One row per condition ─────────────────────────────────────────────
        for ci, cond in enumerate(conditions):
            ccolor  = cond["color"]
            row     = base_r + ci

            # Top-firing neuron for this query under this condition
            q_lat     = cond["all_latents"][qidx].float()
            fire_mask = q_lat > 0
            nid = int(torch.argmax(q_lat * fire_mask.float())
                      if fire_mask.sum() > 0 else torch.argmax(q_lat))

            ex_idxs = [i for i in cond["top_idx"][nid].tolist()[:e] if i >= 0]
            ex_srcs = [int(cond["all_src"][ei]) for ei in ex_idxs]

            # Condition label (column 1)
            ax_lbl = fig.add_subplot(gs[row, 1])
            ax_lbl.axis("off")
            ax_lbl.set_facecolor(ccolor + "22")
            ax_lbl.add_patch(mpatches.FancyArrowPatch(
                (0.0, 0.0), (0.0, 1.0), transform=ax_lbl.transAxes,
                arrowstyle="-", color=ccolor, linewidth=5))
            ax_lbl.text(0.55, 0.5, COND_NAMES[ci],
                        transform=ax_lbl.transAxes, ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=ccolor, linespacing=1.5)

            # Exemplar images (columns 2 … 2+e-1)
            for ei in range(e):
                if ei >= len(ex_idxs):
                    ax_e = fig.add_subplot(gs[row, 2 + ei])
                    ax_e.axis("off"); ax_e.set_facecolor("#f0f0f0"); continue

                gidx  = int(ex_idxs[ei])
                src   = int(ex_srcs[ei])
                is_in = (src == IMAGENET)
                try:
                    thumb    = _pil_thumb(combined_ds, gidx, size=int(img_w * 100))
                    cls_name = _cls_name(combined_ds, gidx, classnames)
                    style    = "dashed" if is_in else "solid"
                    color    = "#aaaaaa" if is_in else ccolor
                    cap_col  = "#ddd" if is_in else "#fff"
                    _draw_thumb(gs[row, 2 + ei], thumb, color,
                                border_style=style, caption=cls_name, cap_color=cap_col)
                except Exception:
                    ax_e = fig.add_subplot(gs[row, 2 + ei])
                    ax_e.text(0.5, 0.5, "err", ha="center", va="center",
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

    print(f"\n{'='*65}\n  Dataset: {ds_name.upper()}\n{'='*65}")

    tgt_ds_full = get_dataset(ds_name, registry=registry)
    classnames  = get_classnames(ds_name, dataset=tgt_ds_full, registry=registry)
    label_key   = get_label_key(ds_name, tgt_ds_full, registry=registry)

    # Stratified subsample to max_dataset_images
    all_labels = tgt_ds_full[label_key]
    by_cls: Dict[int, List[int]] = {}
    for i, lbl in enumerate(all_labels):
        by_cls.setdefault(int(lbl), []).append(i)
    n_cls   = len(by_cls)
    per_cls = max(1, args.max_dataset_images // max(n_cls, 1))
    rng     = random.Random(42)
    indices = []
    for cls_idxs in by_cls.values():
        indices.extend(rng.sample(cls_idxs, min(per_cls, len(cls_idxs))))
    tgt_ds = tgt_ds_full.select(sorted(indices))
    print(f"  Target: {len(tgt_ds_full):,} -> {len(tgt_ds):,} stratified "
          f"({per_cls}/class x {n_cls} classes), {len(classnames)} classnames")

    combined = CombinedHFDataset(tgt_ds, imagenet_ds, label_key, imagenet_lk)
    print(f"  Combined pool: {len(combined):,} ({len(tgt_ds)} target + {len(imagenet_ds)} ImageNet)")

    print("  Loading SAEs...")
    base_sae, _ = load_sae(BASE_SAE_PATH,     device); base_sae.eval()
    ad_sae,   _ = load_sae(cfg["adsae_path"], device); ad_sae.eval()

    print("  Loading ZS backbone...")
    bb_zs   = _load_base_bb(device)
    print("  Loading LoRA backbone...")
    bb_lora = _load_lora_bb(cfg["lora_path"], device)

    specs = [
        dict(short=COND_SHORT[0], color=COND_COLORS[0], bb=bb_zs,   sae=base_sae, layer=BASE_SAE_LAYER),
        dict(short=COND_SHORT[1], color=COND_COLORS[1], bb=bb_lora, sae=base_sae, layer=BASE_SAE_LAYER),
        dict(short=COND_SHORT[2], color=COND_COLORS[2], bb=bb_lora, sae=ad_sae,   layer=cfg["adsae_layer"]),
    ]

    built = []
    for sp in specs:
        print(f"\n  Indexing: {sp['short']}...")
        top_idx, _, all_lat, all_src = build_neuron_index(
            backbone=sp["bb"], sae=sp["sae"], combined_ds=combined,
            layer=sp["layer"], batch_size=args.batch_size,
            device=device, num_workers=args.num_workers,
            n_top=args.exemplars_per_neuron + 3,
        )
        built.append(dict(
            name=COND_NAMES[specs.index(sp)], color=sp["color"],
            top_idx=top_idx, all_latents=all_lat, all_src=all_src,
        ))

    print(f"\n  Scoring {combined.n_a} target images for hierarchy clarity...")
    samples = score_and_select(combined, built, args.n_samples, args.exemplars_per_neuron)
    names   = [_cls_name(combined, s, classnames) for s in samples]
    print(f"  Selected top-{len(samples)}: {list(zip(samples, names))}")

    # One output file per query image
    for pi, qidx in enumerate(samples):
        q_name   = _cls_name(combined, qidx, classnames)
        safe_cls = q_name.replace("/", "_").replace(" ", "_")
        fname    = f"sample_{pi + 1:03d}_{safe_cls}.png"
        out_path = os.path.join(args.out_dir, ds_name, fname)
        plot_exemplars(
            combined_ds=combined,
            sample_indices=[qidx],
            conditions=built,
            classnames=classnames,
            exemplars_per_neuron=args.exemplars_per_neuron,
            ds_name=ds_name,
            out_path=out_path,
        )

    del bb_zs, bb_lora, base_sae, ad_sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Top-activating exemplars -- query images at native resolution"
    )
    parser.add_argument("--datasets",             nargs="+",
                        default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--n_samples",            type=int, default=3,
                        help="Number of query images per dataset")
    parser.add_argument("--exemplars_per_neuron", type=int, default=5)
    parser.add_argument("--max_dataset_images",   type=int, default=5000)
    parser.add_argument("--n_imagenet",           type=int, default=1000,
                        help="ImageNet images to include in the combined pool")
    parser.add_argument("--batch_size",           type=int, default=64)
    parser.add_argument("--num_workers",          type=int, default=4)
    parser.add_argument("--device",               default="cuda")
    parser.add_argument("--out_dir",              default="out/exemplar_viz_noresize")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] Falling back to CPU."); device = "cpu"

    registry = load_registry()

    print(f"[INFO] Loading {args.n_imagenet} ImageNet images...")
    imagenet_ds = get_dataset("imagenet", registry=registry, max_samples=args.n_imagenet)
    imagenet_lk = get_label_key("imagenet", imagenet_ds, registry=registry)
    print(f"  ImageNet: {len(imagenet_ds):,} images, label_key={imagenet_lk}")

    for ds in args.datasets:
        process_dataset(ds, args, registry, imagenet_ds, imagenet_lk, device)

    print("\n[INFO] Done. Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()

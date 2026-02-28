#!/usr/bin/env python3
"""
Monosemanticity evaluation for MedMNIST SAEs.

Runs monosemanticity scoring on:
  1. MedMNIST SAEs (from out/checkpoints/medmnist) on LoRA-adapted CLIP
  2. MaPLe SAE on MaPLe-adapted CLIP
  3. Optionally: baseline SAE on base CLIP

Usage (from patchsae/):
    # All medmnist SAEs + maple SAE
    python full_pipeline_medmnist.py

    # Only a specific medmnist SAE layer
    python full_pipeline_medmnist.py --layer -2

    # Include base SAE on base CLIP
    python full_pipeline_medmnist.py --include_base

    # Skip maple
    python full_pipeline_medmnist.py --skip_maple
"""

import argparse, gc, glob, json, math, os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
from tqdm import tqdm

# ── patchsae imports ──────────────────────────────────────────────────────
try:
    from tasks.utils import load_sae, load_hooked_vit, get_sae_and_vit
    from src.sae_training.hooked_vit import Hook
except ImportError as e:
    print(f"[FATAL] {e}\nRun from the patchsae/ directory."); sys.exit(1)

from integrate_mono import (
    per_class_monosemanticity,
    dataset_monosemanticity,
    topk_neurons_overall,
)

try:
    import clip as openai_clip
    HAS_OPENAI_CLIP = True
except ImportError:
    HAS_OPENAI_CLIP = False

# ══════════════════════════════════════════════════════════════════════════
# Config — edit these if your paths differ
# ══════════════════════════════════════════════════════════════════════════
MEDMNIST_SAE_DIR = "/home/sunayana/Documents/Concept_LoRA/patchsae/out/checkpoints/masked_finetune/"
MAPLE_SAE_PATH   = "out/checkpoints/medmnist_maple/qcx3o72n/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt"
BASE_SAE_PATH    = "data/sae_weight/base/out.pt"

LORA_CHECKPOINT  = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
LORA_MERGED_PATH = "/home/sunayana/Documents/Concept_LoRA/lora_merged/clip_vitb16_medmnist_lora_merged.pt"

MAPLE_MODEL_PATH  = "/home/sunayana/Documents/model.pth.tar-5"
MAPLE_CONFIG_PATH = "configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml"

DATA_DIR          = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
NPZ_PATH          = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
CLASSNAMES_PATH   = "configs/classnames/medmnist_classnames.json"

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

BACKBONE = "openai/clip-vit-base-patch16"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
BATCH    = 32


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_classnames():
    with open(CLASSNAMES_PATH) as f:
        return json.load(f)


def get_imagefolder_classnames():
    ds = datasets.ImageFolder(root=DATA_DIR)
    idx_to_name = {i: n.replace("_", " ") for n, i in ds.class_to_idx.items()}
    return [idx_to_name[i] for i in range(len(idx_to_name))]


def build_npz_to_if_mapping():
    """Map NPZ label indices to ImageFolder label indices."""
    ds = datasets.ImageFolder(root=DATA_DIR)
    if_map = {n.replace('_', ' ').lower(): i for n, i in ds.class_to_idx.items()}
    mapping = {}
    for npz_idx, classname in enumerate(MEDMNIST_CLASSES):
        key = classname.lower()
        mapping[npz_idx] = if_map.get(key, npz_idx)
    return mapping


class NpzTestDatasetPIL(Dataset):
    """Test split from .npz, returns (PIL image, label)."""
    def __init__(self, npz_path, label_mapping):
        from torchvision import transforms
        data = np.load(npz_path)
        self.imgs = data['test_images']
        self.labels = data['test_labels'].flatten()
        self.mapping = label_mapping
        self.resize = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i])
        img = self.resize(img)  # returns PIL
        return img, self.mapping[int(self.labels[i])]


class NpzTestDatasetTensor(Dataset):
    """Test split from .npz, returns (tensor, label) for OpenAI CLIP."""
    def __init__(self, npz_path, label_mapping, transform):
        data = np.load(npz_path)
        self.imgs = data['test_images']
        self.labels = data['test_labels'].flatten()
        self.mapping = label_mapping
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i])
        img = self.transform(img)  # returns tensor
        return img, self.mapping[int(self.labels[i])]


def pil_collate(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)


def discover_medmnist_saes(base_dir, layer_filter=None):
    """Find final SAE checkpoints, keeping one per layer (last by sort)."""
    paths = sorted(glob.glob(os.path.join(base_dir, "*/final*/*.pt")))
    best = {}
    for p in paths:
        ckpt = torch.load(p, map_location="cpu")
        cfg_d = ckpt.get("cfg", ckpt.get("config"))
        layer = getattr(cfg_d, "block_layer",
                        cfg_d.get("block_layer", "?") if isinstance(cfg_d, dict) else "?")
        best[layer] = p
        del ckpt
    gc.collect()

    if layer_filter is not None:
        best = {k: v for k, v in best.items() if k == layer_filter}

    print(f"  Discovered {len(best)} medmnist SAE(s):")
    for layer, p in sorted(best.items()):
        print(f"    layer {layer}: {p}")
    return list(best.values())


# ══════════════════════════════════════════════════════════════════════════
# LoRA: build OpenAI CLIP with merged LoRA weights
# ══════════════════════════════════════════════════════════════════════════

def _extract_lora_AB(ld, proj):
    if proj in ld and isinstance(ld[proj], dict):
        try:
            return ld[proj]["w_lora_A"], ld[proj]["w_lora_B"]
        except Exception:
            pass
    try:
        return ld[f"{proj}.w_lora_A"], ld[f"{proj}.w_lora_B"]
    except Exception:
        return None, None


def build_lora_openai_clip(lora_path, device):
    """Load OpenAI CLIP ViT-B/16 and merge LoRA weights."""
    if not HAS_OPENAI_CLIP:
        print("[FATAL] pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    model, preprocess = openai_clip.load("ViT-B/16", device=device)

    lora_state = torch.load(lora_path, map_location=device)
    if "weights" not in lora_state:
        model.load_state_dict(lora_state)
        return model, preprocess

    layers, meta = lora_state["weights"], lora_state["metadata"]
    scale = meta["alpha"] / math.sqrt(meta["r"])
    print(f"    LoRA: rank={meta['r']}, alpha={meta['alpha']}, scale={scale:.4f}")

    with torch.no_grad():
        for i in range(12):
            key = f"layer_{i}"
            if key not in layers:
                continue
            ld = layers[key]
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _extract_lora_AB(ld, proj)
                if A is None:
                    continue
                w[off:off + d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)

        for i in range(12, 24):
            key = f"layer_{i}"
            if key not in layers:
                continue
            ld = layers[key]
            w = model.visual.transformer.resblocks[i - 12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _extract_lora_AB(ld, proj)
                if A is None:
                    continue
                w[off:off + d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)

    print("    LoRA merged OK.")
    return model, preprocess


# ══════════════════════════════════════════════════════════════════════════
# Extraction: HF CLIP (base / maple)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_hf(vit, sae, cfg, loader, vit_type, device):
    """
    Extract CLIP image embeddings + SAE activations using HF path.
    Returns (image_embeddings, sae_activations, labels) all on CPU.
    """
    all_emb, all_act, all_lab = [], [], []

    hook_locations = [(cfg.block_layer, cfg.module_name)]

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="  extract")):
        inputs = vit.processor(images=images, text="",
                               return_tensors="pt", padding=True).to(device)

        # Image embeddings + intermediate activations at the SAE layer
        if vit_type == "maple":
            # MaPLe: forward() returns image features; use run_with_cache
            out, vit_cache = vit.run_with_cache(hook_locations, **inputs)
            img_feat = out  # CustomCLIP.forward returns image features
            vit_features = vit_cache[(cfg.block_layer, cfg.module_name)]
        else:
            # HF CLIP: separate call for embeddings, run_with_cache for activations
            img_feat = vit.model.get_image_features(pixel_values=inputs["pixel_values"])
            _, vit_cache = vit.run_with_cache(hook_locations, **inputs)
            vit_features = vit_cache[(cfg.block_layer, cfg.module_name)]

        # SAE activations
        # For maple, vit_features is [seq, batch, d_model] — transpose to [batch, seq, d]
        if vit_type == "maple" and vit_features.dim() == 3 and vit_features.shape[0] != vit_features.shape[1]:
            # Heuristic: if dim0 > dim1, it's [seq, batch, d] — transpose
            if vit_features.shape[0] > vit_features.shape[1]:
                vit_features = vit_features.transpose(0, 1)

        _, sae_cache = sae.run_with_cache(vit_features)
        sae_acts = sae_cache["hook_hidden_post"]

        # Pool to CLS token if 3D  → [batch, hidden]
        if sae_acts.dim() == 3:
            sae_acts = sae_acts[:, 0, :]

        all_emb.append(img_feat.cpu())
        all_act.append(sae_acts.cpu())
        all_lab.append(labels)

        if batch_idx % 20 == 0:
            print(f"    batch {batch_idx}/{len(loader)}")

    return torch.cat(all_emb), torch.cat(all_act), torch.cat(all_lab)


# ══════════════════════════════════════════════════════════════════════════
# Extraction: OpenAI CLIP (LoRA)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_openai(model, sae, cfg, loader, device):
    """
    Extract CLIP image embeddings + SAE activations using OpenAI CLIP model.
    Registers a capture hook at the SAE's target layer.
    Returns (image_embeddings, sae_activations, labels) all on CPU.
    """
    all_emb, all_act, all_lab = [], [], []
    captured = {}

    num_blocks = len(model.visual.transformer.resblocks)
    layer = cfg.block_layer
    if layer < 0:
        layer = num_blocks + layer
    block = model.visual.transformer.resblocks[layer]

    def hook_fn(module, inp, output):
        # OpenAI: output is [seq_len, batch, d_model]
        captured["act"] = output.transpose(0, 1).float()

    handle = block.register_forward_hook(hook_fn)

    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="  extract")):
        captured.clear()
        imgs = imgs.to(device)

        # Image embeddings
        img_feat = model.encode_image(imgs)

        # Get captured activations
        vit_features = captured["act"]  # [batch, seq, d_model]

        # SAE activations
        _, sae_cache = sae.run_with_cache(vit_features)
        sae_acts = sae_cache["hook_hidden_post"]

        if sae_acts.dim() == 3:
            sae_acts = sae_acts[:, 0, :]

        all_emb.append(img_feat.float().cpu())
        all_act.append(sae_acts.cpu())
        all_lab.append(labels)

        if batch_idx % 20 == 0:
            print(f"    batch {batch_idx}/{len(loader)}")

    handle.remove()
    return torch.cat(all_emb), torch.cat(all_act), torch.cat(all_lab)


# ══════════════════════════════════════════════════════════════════════════
# Monosemanticity analysis  (from integrate_mono)
# ══════════════════════════════════════════════════════════════════════════

def analyze_monosemanticity(image_embeddings, neuron_activations, labels,
                            num_classes, tag=""):
    """Compute and print monosemanticity scores; return summary dict."""
    print(f"\n  Computing monosemanticity for: {tag}")
    print(f"    embeddings={image_embeddings.shape}, "
          f"activations={neuron_activations.shape}, classes={num_classes}")

    # Dataset-wide
    ms_scores, avg_ms = dataset_monosemanticity(image_embeddings, neuron_activations)
    print(f"    Overall avg monosemanticity: {avg_ms:.4f}")

    # Top-k neurons
    topk_idx, topk_vals = topk_neurons_overall(ms_scores, k=10)
    top10_avg = topk_vals[:10].mean().item()
    print(f"    Top-10 neuron avg:           {top10_avg:.4f}")

    top10_data = []
    for rank, (nid, sc) in enumerate(zip(topk_idx[:10], topk_vals[:10]), 1):
        print(f"      {rank:2d}. Neuron {nid:5d}: {sc:.4f}")
        top10_data.append({"rank": rank, "neuron_id": nid.item(),
                           "monosemantic_score": sc.item(), "tag": tag})
    top10_data.append({"rank": "AVG", "neuron_id": "TOP_10",
                       "monosemantic_score": top10_avg, "tag": tag})

    # Per-class
    per_class = per_class_monosemanticity(
        image_embeddings, neuron_activations, labels, num_classes, k=10
    )
    class_avgs = []
    per_class_data = []
    for cid, neuron_scores in per_class.items():
        cavg = sum(s for _, s in neuron_scores[:10]) / max(len(neuron_scores[:10]), 1)
        class_avgs.append(cavg)
        for rank, (nid, sc) in enumerate(neuron_scores[:10], 1):
            per_class_data.append({"class_id": cid, "rank": rank,
                                    "neuron_id": nid, "score": sc,
                                    "class_avg": cavg, "tag": tag})

    overall_class_avg = sum(class_avgs) / max(len(class_avgs), 1)
    print(f"    Per-class avg monosemanticity: {overall_class_avg:.4f}")

    return {
        "overall_avg": avg_ms,
        "top10_avg": top10_avg,
        "class_avg": overall_class_avg,
        "top10_data": top10_data,
        "per_class_data": per_class_data,
        "ms_scores": ms_scores,
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Monosemanticity analysis for MedMNIST SAEs")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--batch_size", type=int, default=BATCH)
    ap.add_argument("--layer", type=int, default=None,
                    help="Only evaluate this layer (e.g. -2). Default: all layers.")
    ap.add_argument("--include_base", action="store_true",
                    help="Also evaluate base SAE on base CLIP")
    ap.add_argument("--skip_maple", action="store_true",
                    help="Skip MaPLe SAE evaluation")
    ap.add_argument("--skip_lora", action="store_true",
                    help="Skip LoRA/MedMNIST SAE evaluation")
    ap.add_argument("--save_dir", default="out/monosemanticity_medmnist")
    args = ap.parse_args()

    device = args.device
    batch_size = args.batch_size
    os.makedirs(args.save_dir, exist_ok=True)

    classnames = get_imagefolder_classnames()
    num_classes = len(classnames)
    label_mapping = build_npz_to_if_mapping()
    print(f"MedMNIST: {num_classes} classes: {classnames}")
    print(f"Label mapping (npz→IF): {label_mapping}")

    # PIL test dataset (for HF path: base / maple)
    pil_ds = NpzTestDatasetPIL(NPZ_PATH, label_mapping)
    pil_loader = DataLoader(pil_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=pil_collate)

    # Tensor test dataset (for OpenAI CLIP path: LoRA)
    from torchvision import transforms as T
    oai_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                     (0.26862954, 0.26130258, 0.27577711)),
    ])
    oai_ds = NpzTestDatasetTensor(NPZ_PATH, label_mapping, oai_transform)
    oai_loader = DataLoader(oai_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Test dataset: {len(pil_ds)} images, "
          f"PIL loader={len(pil_loader)} batches, "
          f"OAI loader={len(oai_loader)} batches\n")

    all_summaries = []

    # ── 1) MedMNIST SAEs on LoRA-adapted CLIP ────────────────────
    if not args.skip_lora:
        print("=" * 70)
        print("  MEDMNIST SAEs  ×  LoRA CLIP")
        print("=" * 70)

        sae_paths = discover_medmnist_saes(MEDMNIST_SAE_DIR,
                                           layer_filter=args.layer)
        if not sae_paths:
            print("  ⚠ No medmnist SAEs found! Skipping LoRA eval.")
        else:
            # Build LoRA model once
            print("\n  Loading LoRA-adapted OpenAI CLIP...")
            lora_model, lora_preprocess = build_lora_openai_clip(
                LORA_CHECKPOINT, device)
            lora_model.eval()

            for sae_path in sae_paths:
                sae, cfg = load_sae(sae_path, device)
                tag = f"medmnist_SAE_layer{cfg.block_layer}_x_LoRA"
                print(f"\n  ── SAE: layer={cfg.block_layer}, path={sae_path}")

                emb, act, lab = extract_openai(lora_model, sae, cfg,
                                               oai_loader, device)
                res = analyze_monosemanticity(emb, act, lab, num_classes, tag=tag)
                res["sae_source"] = "medmnist"
                res["model"] = "lora"
                res["layer"] = cfg.block_layer
                all_summaries.append(res)

                # Save CSVs
                pd.DataFrame(res["top10_data"]).to_csv(
                    os.path.join(args.save_dir, f"{tag}_top10.csv"), index=False)
                pd.DataFrame(res["per_class_data"]).to_csv(
                    os.path.join(args.save_dir, f"{tag}_perclass.csv"), index=False)

                del sae; flush()

            del lora_model; flush()

    # ── 2) MaPLe SAE on MaPLe CLIP ───────────────────────────────
    if not args.skip_maple and os.path.exists(MAPLE_SAE_PATH):
        print("\n" + "=" * 70)
        print("  MAPLE SAE  ×  MaPLe CLIP")
        print("=" * 70)

        maple_sae, maple_cfg = load_sae(MAPLE_SAE_PATH, device)
        tag = f"maple_SAE_layer{maple_cfg.block_layer}_x_MaPLe"
        print(f"  SAE: layer={maple_cfg.block_layer}")

        print("  Loading MaPLe-adapted HF CLIP...")
        vit_maple = load_hooked_vit(
            maple_cfg, "maple", BACKBONE, device,
            model_path=MAPLE_MODEL_PATH,
            config_path=MAPLE_CONFIG_PATH,
            classnames=classnames,
        )
        vit_maple.model.eval()

        emb, act, lab = extract_hf(vit_maple, maple_sae, maple_cfg,
                                   pil_loader, "maple", device)
        res = analyze_monosemanticity(emb, act, lab, num_classes, tag=tag)
        res["sae_source"] = "maple"
        res["model"] = "maple"
        res["layer"] = maple_cfg.block_layer
        all_summaries.append(res)

        pd.DataFrame(res["top10_data"]).to_csv(
            os.path.join(args.save_dir, f"{tag}_top10.csv"), index=False)
        pd.DataFrame(res["per_class_data"]).to_csv(
            os.path.join(args.save_dir, f"{tag}_perclass.csv"), index=False)

        del vit_maple, maple_sae; flush()

    # ── 3) Optional: base SAE on base CLIP ────────────────────────
    if args.include_base and os.path.exists(BASE_SAE_PATH):
        print("\n" + "=" * 70)
        print("  BASE SAE  ×  Base CLIP")
        print("=" * 70)

        base_sae, base_cfg = load_sae(BASE_SAE_PATH, device)
        tag = f"base_SAE_layer{base_cfg.block_layer}_x_Base"

        vit_base = load_hooked_vit(base_cfg, "base", BACKBONE, device)
        vit_base.model.eval()

        emb, act, lab = extract_hf(vit_base, base_sae, base_cfg,
                                   pil_loader, "base", device)
        res = analyze_monosemanticity(emb, act, lab, num_classes, tag=tag)
        res["sae_source"] = "base"
        res["model"] = "base"
        res["layer"] = base_cfg.block_layer
        all_summaries.append(res)

        pd.DataFrame(res["top10_data"]).to_csv(
            os.path.join(args.save_dir, f"{tag}_top10.csv"), index=False)
        pd.DataFrame(res["per_class_data"]).to_csv(
            os.path.join(args.save_dir, f"{tag}_perclass.csv"), index=False)

        del vit_base, base_sae; flush()

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  MONOSEMANTICITY SUMMARY  ({len(all_summaries)} evaluations)")
    print("=" * 70)
    print(f"  {'Tag':<45s}  {'Overall':>8s}  {'Top10':>8s}  {'Class':>8s}")
    print(f"  {'─'*45}  {'─'*8}  {'─'*8}  {'─'*8}")

    summary_rows = []
    for s in all_summaries:
        tag = f"{s['sae_source']}_layer{s['layer']}_{s['model']}"
        print(f"  {tag:<45s}  {s['overall_avg']:>8.4f}  "
              f"{s['top10_avg']:>8.4f}  {s['class_avg']:>8.4f}")
        summary_rows.append({
            "sae_source": s["sae_source"],
            "model": s["model"],
            "layer": s["layer"],
            "overall_avg": s["overall_avg"],
            "top10_avg": s["top10_avg"],
            "class_avg": s["class_avg"],
        })
    print("=" * 70)

    # Save summary
    summary_path = os.path.join(args.save_dir, "monosemanticity_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

    # Save full results
    full_path = os.path.join(args.save_dir, "monosemanticity_full.pt")
    torch.save(all_summaries, full_path)
    print(f"Saved full results to {full_path}")


if __name__ == "__main__":
    main()

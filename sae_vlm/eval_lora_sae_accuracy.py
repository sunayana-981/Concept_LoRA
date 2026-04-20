#!/usr/bin/env python3
"""
LoRA CLIP Accuracy Evaluation: plain LoRA vs LoRA + ImageNet SAE vs LoRA + dataset SAE

For each dataset (eurosat, caltech101, medmnist):
  1. LoRA model alone          — zero-shot top-1 accuracy, no SAE
  2. LoRA model + ImageNet SAE — base SAE (data/sae_weight/base/out.pt) via hook
  3. LoRA model + dataset SAE  — each final checkpoint in out/checkpoints/{dataset}/ (non-maple)

Usage (run from patchsae/ directory):
    python eval_lora_sae_accuracy.py
    python eval_lora_sae_accuracy.py --datasets eurosat caltech101
    python eval_lora_sae_accuracy.py --batch_size 32 --save_json out/lora_sae_acc.json
"""

import argparse
import gc
import glob
import json
import math
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

# ── Project imports (must run from patchsae/) ─────────────────────────────────
try:
    from src.sae_training.loaders import load_sae
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] {e}\nRun from the patchsae/ directory.")
    sys.exit(1)

try:
    import clip as openai_clip
except ImportError:
    print("[FATAL] OpenAI CLIP not found.\n"
          "  Install: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

# ── Paths / constants ─────────────────────────────────────────────────────────
BACKBONE      = "ViT-B/16"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SAE_PATH = "data/sae_weight/base/out.pt"          # absolute or relative to patchsae/
CKPT_ROOT     = "out/checkpoints"
LORA_ROOT     = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
DATA_ROOT     = "/home/sunayana/Documents/Concept_LoRA/data"

# MedMNIST / PathMNIST tissue types
MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

DATASET_CFG = {
    "eurosat": {
        "data_dir":   f"{DATA_ROOT}/eurosat/2750",
        "lora_path":  f"{LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
        "class_names": None,        # auto-discovered from ImageFolder structure
        "use_npz":    False,
        "split":      "all",        # use whole ImageFolder (no separate test split)
    },
    "caltech101": {
        "data_dir":   f"{DATA_ROOT}/caltech-101",
        "lora_path":  f"{LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz":    False,
        "split":      "all",
        "exclude_classes": {"BACKGROUND_Google"},
    },
    "medmnist": {
        "data_dir":   f"{DATA_ROOT}/pathmnist_imagefolder",  # for label mapping
        "npz_path":   f"{DATA_ROOT}/pathmnist.npz",
        "lora_path":  f"{LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
        "class_names": MEDMNIST_CLASSES,
        "use_npz":    True,         # use .npz test split
        "split":      "test",
    },
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ── LoRA weight merging ───────────────────────────────────────────────────────

def _lora_AB(layer_dict, proj_name):
    """Extract LoRA A and B matrices for one projection."""
    d = layer_dict
    if proj_name in d and isinstance(d[proj_name], dict):
        try:
            return d[proj_name]["w_lora_A"], d[proj_name]["w_lora_B"]
        except (KeyError, TypeError):
            pass
    try:
        return d[f"{proj_name}.w_lora_A"], d[f"{proj_name}.w_lora_B"]
    except KeyError:
        return None, None


def build_lora_clip(lora_path: str, device: str):
    """
    Load OpenAI CLIP ViT-B/16 and merge LoRA delta weights in-place.
    Returns (model, preprocess).
    """
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    model.eval()

    lora_state = torch.load(lora_path, map_location=device)
    if "weights" not in lora_state:
        print("  [WARN] No 'weights' key in LoRA checkpoint — using base CLIP.")
        return model, preprocess

    layers_dict = lora_state["weights"]
    meta        = lora_state["metadata"]
    scale       = meta["alpha"] / math.sqrt(meta["r"])
    print(f"  LoRA: rank={meta['r']}, alpha={meta['alpha']}, scale={scale:.6f}")

    with torch.no_grad():
        # Text encoder layers 0–11
        for i in range(12):
            ld = layers_dict.get(f"layer_{i}")
            if ld is None:
                continue
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _lora_AB(ld, proj)
                if A is None:
                    continue
                delta = scale * (B.float().to(device) @ A.float().to(device))
                w[off:off + d] += delta.to(w.dtype)

        # Vision encoder layers 12–23
        for i in range(12, 24):
            ld = layers_dict.get(f"layer_{i}")
            if ld is None:
                continue
            w = model.visual.transformer.resblocks[i - 12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _lora_AB(ld, proj)
                if A is None:
                    continue
                delta = scale * (B.float().to(device) @ A.float().to(device))
                w[off:off + d] += delta.to(w.dtype)

    print("  LoRA weights merged.")
    return model, preprocess


# ── Dataset loading ───────────────────────────────────────────────────────────

def _find_imagefolder_root(root: str) -> str:
    """
    Walk down directory tree to find the level where class sub-directories
    each contain image files (handles nested layouts like 101_ObjectCategories/).
    """
    subdirs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No sub-directories in {root}")

    # Check whether immediate subdirs contain images → root is the class level
    for sd in subdirs[:3]:
        sd_path = os.path.join(root, sd)
        if any(f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
               for f in os.listdir(sd_path)):
            return root

    # Otherwise recurse
    for sd in sorted(subdirs):
        sd_path = os.path.join(root, sd)
        inner = [d for d in os.listdir(sd_path)
                 if os.path.isdir(os.path.join(sd_path, d))]
        if inner:
            try:
                return _find_imagefolder_root(sd_path)
            except (FileNotFoundError, PermissionError):
                continue

    raise FileNotFoundError(f"No image class directories found under {root}")


class FilteredImageFolder(Dataset):
    """ImageFolder with class exclusion and contiguous label remapping."""

    def __init__(self, root: str, transform, exclude_classes=None):
        img_root    = _find_imagefolder_root(root)
        full_ds     = datasets.ImageFolder(root=img_root, transform=transform)
        exclude     = exclude_classes or set()

        keep_indices    = [i for i, (_, lbl) in enumerate(full_ds.samples)
                           if full_ds.classes[lbl] not in exclude]
        kept_cls_names  = sorted({full_ds.classes[full_ds.targets[i]]
                                   for i in keep_indices})
        old_to_new      = {full_ds.class_to_idx[c]: new_i
                           for new_i, c in enumerate(kept_cls_names)}

        self._dataset    = full_ds
        self._indices    = keep_indices
        self._label_map  = old_to_new
        self.class_names = kept_cls_names

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        img, old_lbl = self._dataset[self._indices[idx]]
        return img, self._label_map[old_lbl]


class MedMNISTNpzDataset(Dataset):
    """
    PathMNIST test split from .npz with label remapping to match the
    ImageFolder ordering used during SAE training.
    """

    def __init__(self, npz_path: str, imagefolder_root: str,
                 preprocess, split: str = "test"):
        data = np.load(npz_path)
        imgs_key  = f"{split}_images"
        lbls_key  = f"{split}_labels"
        if imgs_key not in data:
            raise KeyError(f"'{imgs_key}' not found in {npz_path}. "
                           f"Available keys: {list(data.keys())}")
        self.imgs      = data[imgs_key]
        self.labels    = data[lbls_key].flatten().astype(int)
        self.preprocess = preprocess

        # Build NPZ-class-index → ImageFolder-class-index mapping
        if_root = _find_imagefolder_root(imagefolder_root)
        if_ds   = datasets.ImageFolder(root=if_root)
        if_map  = {n.replace("_", " ").lower(): i
                   for n, i in if_ds.class_to_idx.items()}

        self.label_map = {}
        for npz_idx, cls_name in enumerate(MEDMNIST_CLASSES):
            self.label_map[npz_idx] = if_map.get(cls_name.lower(), npz_idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx]).convert("RGB")
        img = self.preprocess(img)
        return img, self.label_map[int(self.labels[idx])]


def load_test_dataset(cfg_dict: dict, preprocess) -> tuple:
    """
    Returns (dataset, class_names).
    Uses NpzDataset for medmnist, FilteredImageFolder for others.
    """
    if cfg_dict.get("use_npz", False):
        ds = MedMNISTNpzDataset(
            npz_path=cfg_dict["npz_path"],
            imagefolder_root=cfg_dict["data_dir"],
            preprocess=preprocess,
            split=cfg_dict.get("split", "test"),
        )
        # Class names must match the ImageFolder label ordering (alphabetical),
        # because MedMNISTNpzDataset remaps npz labels to ImageFolder indices.
        # Using MEDMNIST_CLASSES directly (npz semantic order) would cause a
        # label↔text mismatch and garbage accuracy.
        if_root = _find_imagefolder_root(cfg_dict["data_dir"])
        if_ds   = datasets.ImageFolder(root=if_root)
        class_names = [n.replace("_", " ") for n in if_ds.classes]
    else:
        ds = FilteredImageFolder(
            root=cfg_dict["data_dir"],
            transform=preprocess,
            exclude_classes=cfg_dict.get("exclude_classes"),
        )
        class_names = cfg_dict.get("class_names") or ds.class_names

    return ds, class_names


# ── Text features (zero-shot) ─────────────────────────────────────────────────

@torch.no_grad()
def get_text_features(model, class_names: list, device: str) -> torch.Tensor:
    """
    Average 80 OpenAI prompt templates → L2-normalized text feature matrix.
    Shape: (n_classes, 512).
    """
    n = len(class_names)
    mean_f = torch.zeros(n, 512, device=device)

    for tmpl in openai_imagenet_template:
        prompts = [tmpl(c) for c in class_names]
        tokens  = openai_clip.tokenize(prompts, truncate=True).to(device)
        feats   = model.encode_text(tokens).float()
        mean_f  += l2_normalize(feats)

    mean_f /= len(openai_imagenet_template)
    return l2_normalize(mean_f)


# ── SAE hook ──────────────────────────────────────────────────────────────────

class SAEForwardHook:
    """
    Registers a PyTorch forward hook on an OpenAI CLIP visual resblock that
    replaces the block's output with the SAE reconstruction.

    OpenAI CLIP resblock output shape: [seq_len, batch, d_model].
    SparseAutoencoder expects             [batch,   seq,  d_model].
    """

    def __init__(self, model, sae, sae_cfg, device: str):
        num_blocks = len(model.visual.transformer.resblocks)
        layer = sae_cfg.block_layer
        if layer < 0:
            layer = num_blocks + layer
        if not (0 <= layer < num_blocks):
            raise ValueError(
                f"block_layer={sae_cfg.block_layer} → resblock[{layer}] is out of "
                f"range for a model with {num_blocks} blocks."
            )

        self.sae     = sae
        self.device  = device
        self.block   = model.visual.transformer.resblocks[layer]
        self.handle  = None
        self._logged = False
        print(f"  [SAEHook] resblock[{layer}] "
              f"(block_layer={sae_cfg.block_layer}, d_sae={sae.d_sae})")

    def _hook_fn(self, module, input, output):
        orig_dtype = output.dtype

        # [seq, batch, d_model] → [batch, seq, d_model]
        act_bsd = output.transpose(0, 1).float()

        # SparseAutoencoder.forward returns (sae_out, feature_acts, loss_dict)
        sae_out, _, _ = self.sae(act_bsd)

        if not self._logged:
            self._logged = True
            mse = (act_bsd - sae_out).pow(2).mean().item()
            cos = F.cosine_similarity(
                act_bsd.reshape(-1, act_bsd.shape[-1]),
                sae_out.reshape(-1, sae_out.shape[-1]),
                dim=-1,
            ).mean().item()
            print(f"  [SAEHook debug] MSE={mse:.6f}  cos={cos:.6f}")

        # [batch, seq, d_model] → [seq, batch, d_model]
        return sae_out.transpose(0, 1).to(orig_dtype)

    def register(self):
        self.handle = self.block.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ── Accuracy computation ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_accuracy(
    model,
    text_features: torch.Tensor,
    loader: DataLoader,
    device: str,
) -> float:
    """Zero-shot top-1 accuracy using OpenAI CLIP image encoder."""
    logit_scale = model.logit_scale.exp()
    tf = l2_normalize(text_features.float())

    correct = total = 0
    for imgs, labels in tqdm(loader, desc="    eval", leave=False):
        img_feat = l2_normalize(model.encode_image(imgs.to(device)).float())
        logits   = logit_scale * (img_feat @ tf.t())
        preds    = logits.argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total * 100.0


# ── SAE checkpoint discovery ──────────────────────────────────────────────────

def discover_final_saes(sae_dir: str) -> list:
    """
    Find final-checkpoint .pt files under sae_dir/*/final*/*.pt.
    Returns list of (run_id, block_layer, path), skipping NaN checkpoints.
    """
    pattern = os.path.join(sae_dir, "*/final*/*.pt")
    paths   = sorted(glob.glob(pattern))

    results = []
    for p in paths:
        try:
            ckpt  = torch.load(p, map_location="cpu")
            cfg   = ckpt.get("cfg", ckpt.get("config"))
            layer = (getattr(cfg, "block_layer", None)
                     if not isinstance(cfg, dict)
                     else cfg.get("block_layer", None))

            # Skip checkpoints with NaN encoder weights
            sd = ckpt.get("state_dict", ckpt)
            w  = sd.get("W_enc") if isinstance(sd, dict) else None
            if w is not None and torch.isnan(w).any():
                print(f"  [SKIP] {p} — NaN in W_enc")
                del ckpt
                gc.collect()
                continue

            run_id = p.split(os.sep)[-3]   # …/run_id/final_xxx/file.pt
            del ckpt
            gc.collect()
            results.append((run_id, layer, p))
        except Exception as e:
            print(f"  [SKIP] {p} — {e}")

    return results


# ── Per-dataset evaluation ────────────────────────────────────────────────────

def evaluate_dataset(
    dataset_name: str,
    cfg_dict: dict,
    base_sae_path: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict:
    """Run all three accuracy conditions for one dataset."""
    print(f"\n{'═'*70}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'═'*70}")

    results = {"dataset": dataset_name, "conditions": []}

    # ── 1. Load LoRA CLIP ─────────────────────────────────────────────────────
    lora_path = cfg_dict["lora_path"]
    if not os.path.isfile(lora_path):
        print(f"  [SKIP] LoRA weights not found: {lora_path}")
        return results

    print(f"\n[1] Loading LoRA CLIP\n    {lora_path}")
    model, preprocess = build_lora_clip(lora_path, device)
    model.eval()

    # ── 2. Load test dataset ──────────────────────────────────────────────────
    print(f"\n[2] Loading {dataset_name} dataset...")
    try:
        ds, class_names = load_test_dataset(cfg_dict, preprocess)
    except Exception as e:
        print(f"  [ERROR] Could not load dataset: {e}")
        del model
        flush()
        return results

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    print(f"  {len(ds)} images | {len(class_names)} classes")
    print(f"  Classes (first 5): {class_names[:5]}")

    # ── 3. Text features ──────────────────────────────────────────────────────
    print("\n[3] Building text features (80 templates)...")
    text_feat = get_text_features(model, class_names, device)
    print(f"  text_features: {text_feat.shape}")

    # ── Condition A: LoRA only ────────────────────────────────────────────────
    print("\n[A] LoRA model (no SAE)")
    acc_lora = compute_accuracy(model, text_feat, loader, device)
    print(f"    → {acc_lora:.2f}%")
    results["conditions"].append({
        "name": "LoRA (no SAE)",
        "accuracy": round(acc_lora, 4),
        "sae_path": None,
        "sae_layer": None,
        "run_id": None,
    })

    # ── Condition B: LoRA + ImageNet SAE ─────────────────────────────────────
    print(f"\n[B] LoRA + ImageNet (base) SAE\n    {base_sae_path}")
    if not os.path.isfile(base_sae_path):
        print(f"  [SKIP] Base SAE not found: {base_sae_path}")
    else:
        try:
            base_sae, base_cfg = load_sae(base_sae_path, device)
            base_sae.eval()
            hook = SAEForwardHook(model, base_sae, base_cfg, device).register()
            acc_base = compute_accuracy(model, text_feat, loader, device)
            hook.remove()
            del base_sae
            flush()
            print(f"    → {acc_base:.2f}%")
            results["conditions"].append({
                "name": "LoRA + ImageNet SAE",
                "accuracy": round(acc_base, 4),
                "sae_path": base_sae_path,
                "sae_layer": base_cfg.block_layer,
                "run_id": "imagenet_base",
            })
        except Exception as e:
            print(f"  [ERROR] {e}")

    # ── Conditions C+: LoRA + per-dataset SAEs ────────────────────────────────
    sae_dir = os.path.join(CKPT_ROOT, dataset_name)
    print(f"\n[C] Discovering dataset-specific SAEs in {sae_dir}/")
    dataset_saes = discover_final_saes(sae_dir)
    if not dataset_saes:
        print("  None found.")
    else:
        print(f"  Found {len(dataset_saes)} SAE(s):")
        for run_id, layer, path in dataset_saes:
            print(f"    run={run_id}  layer={layer}  {path}")

    for run_id, layer, sae_path in dataset_saes:
        label = f"LoRA + {dataset_name} SAE (run={run_id}, layer={layer})"
        print(f"\n    {label}")
        try:
            ds_sae, ds_cfg = load_sae(sae_path, device)
            ds_sae.eval()
            hook = SAEForwardHook(model, ds_sae, ds_cfg, device).register()
            acc  = compute_accuracy(model, text_feat, loader, device)
            hook.remove()
            del ds_sae
            flush()
            print(f"    → {acc:.2f}%")
            results["conditions"].append({
                "name": label,
                "accuracy": round(acc, 4),
                "sae_path": sae_path,
                "sae_layer": layer,
                "run_id": run_id,
            })
        except Exception as e:
            print(f"    [ERROR] {e}")

    del model
    flush()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Zero-shot accuracy: LoRA vs LoRA+ImageNet SAE vs LoRA+dataset SAE"
    )
    p.add_argument(
        "--datasets", nargs="+",
        default=list(DATASET_CFG.keys()),
        choices=list(DATASET_CFG.keys()),
        help="Datasets to evaluate (default: all)",
    )
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument(
        "--save_json", type=str, default=None,
        help="Path to save JSON results (e.g. out/lora_sae_accuracy.json)",
    )
    args = p.parse_args()

    print(f"\n{'═'*70}")
    print(f"  LoRA SAE Accuracy Evaluation")
    print(f"  Device  : {DEVICE}")
    print(f"  Backbone: {BACKBONE}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Base SAE: {BASE_SAE_PATH}")
    print(f"{'═'*70}")

    all_results = []
    for ds_name in args.datasets:
        res = evaluate_dataset(
            dataset_name=ds_name,
            cfg_dict=DATASET_CFG[ds_name],
            base_sae_path=BASE_SAE_PATH,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=DEVICE,
        )
        all_results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}\n")

    for res in all_results:
        print(f"  {res['dataset'].upper()}")
        print(f"  {'─'*60}")
        if not res["conditions"]:
            print("    (no results)")
        for cond in res["conditions"]:
            layer_tag = (f"  [layer={cond['sae_layer']}]"
                         if cond["sae_layer"] is not None else "")
            print(f"    {cond['name']:<55s}  {cond['accuracy']:6.2f}%{layer_tag}")
        print()

    if args.save_json:
        out_path = args.save_json
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()

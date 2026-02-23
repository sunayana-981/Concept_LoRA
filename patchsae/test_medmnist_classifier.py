#!/usr/bin/env python3
"""
Evaluate the three trained SAE classifiers on the PathMNIST test split.

Loads:
  - the 7,180 test images from pathmnist.npz
  - the LoRA-adapted CLIP ViT
  - each SAE + its trained linear probe

Reports per-class and overall accuracy for each classifier.

Usage (from patchsae root):
    python test_medmnist_classifier.py
    python test_medmnist_classifier.py --batch_size 128
"""

import argparse
import json
import os
import sys
import glob
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
try:
    from tasks.utils import load_sae, load_hooked_vit
    from tasks.train_sae_lora_clip import load_lora_weights
    from src.sae_training.config import Config
except ImportError:
    print("[FATAL] Could not import project modules.")
    print("Run this script from the patchsae root directory.")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

NPZ_PATH = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
DATASET_ROOT = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
BASE_SAE_PATH = "data/sae_weight/base/out.pt"
MEDMNIST_SAE_DIR = "out/checkpoints/medmnist"
LORA_CHECKPOINT_PATH = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
CLASSNAMES_PATH = "configs/classnames/medmnist_classnames.json"
CLASSIFIER_DIR = "out"
BACKBONE = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# MedMNIST canonical class ordering (matches classnames json / npz labels)
MEDMNIST_CLASSES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def build_npz_to_imagefolder_mapping(imagefolder_root):
    """
    Build a mapping from npz label indices to ImageFolder label indices.

    npz uses MedMNIST canonical order; ImageFolder uses alphabetical folder order.
    We normalise names (lowercase, underscores→spaces) to match them.
    """
    ds = datasets.ImageFolder(root=imagefolder_root)
    # ImageFolder class name → ImageFolder index
    # folder names use underscores; npz/classnames use spaces
    if_name_to_idx = {}
    for name, idx in ds.class_to_idx.items():
        normalised = name.replace("_", " ").lower()
        if_name_to_idx[normalised] = idx

    mapping = {}
    for npz_idx, npz_name in enumerate(MEDMNIST_CLASSES):
        key = npz_name.lower()
        if key in if_name_to_idx:
            mapping[npz_idx] = if_name_to_idx[key]
        else:
            print(f"[WARN] Could not map npz class {npz_idx} ('{npz_name}') "
                  f"to any ImageFolder class. Available: {list(if_name_to_idx.keys())}")
            mapping[npz_idx] = npz_idx  # fallback

    return mapping, ds.classes


class PathMNISTTestDataset(Dataset):
    """Wraps the npz test split, resizes 28×28→224×224, applies CLIP normalisation,
    and remaps labels from npz ordering to ImageFolder ordering."""

    def __init__(self, npz_path, label_mapping):
        data = np.load(npz_path)
        self.images = data["test_images"]   # (N, 28, 28, 3) uint8
        self.labels = data["test_labels"].flatten()  # (N,)
        self.label_mapping = label_mapping
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # NO Normalize — the CLIPProcessor in the eval loop handles it.
            # Using Normalize here would cause double-normalisation via the
            # lossy ToPILImage round-trip (clipping negative values).
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])  # PIL RGB
        img = self.transform(img)
        label = self.label_mapping[int(self.labels[idx])]
        return img, label


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def discover_medmnist_saes(base_dir):
    """Find final SAE checkpoints, keeping only the latest per layer."""
    pattern = os.path.join(base_dir, "*/final*/*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return []
    best_per_layer = {}
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        cfg_data = ckpt.get("cfg", ckpt.get("config"))
        if hasattr(cfg_data, "__dict__"):
            layer = cfg_data.block_layer
        elif isinstance(cfg_data, dict):
            layer = cfg_data.get("block_layer", "?")
        else:
            layer = "?"
        best_per_layer[layer] = path
        del ckpt
        gc.collect()
    return list(best_per_layer.values())


def load_classifier(path, device):
    """Load a saved linear probe checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model = LinearProbe(ckpt["input_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE classifiers on MedMNIST test set")
    parser.add_argument("--npz", type=str, default=NPZ_PATH)
    parser.add_argument("--dataset_root", type=str, default=DATASET_ROOT)
    parser.add_argument("--classifier_dir", type=str, default=CLASSIFIER_DIR)
    parser.add_argument("--lora_checkpoint", type=str, default=LORA_CHECKPOINT_PATH)
    parser.add_argument("--backbone", type=str, default=BACKBONE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    print("=" * 60)
    print("MEDMNIST SAE CLASSIFIER — TEST EVALUATION")
    print("=" * 60)

    # ── 1. Find classifiers ──────────────────────────────────────────────────
    clf_pattern = os.path.join(args.classifier_dir, "medmnist_classifier_*.pt")
    clf_paths = sorted(glob.glob(clf_pattern))
    if not clf_paths:
        print(f"[FATAL] No classifiers found matching {clf_pattern}")
        sys.exit(1)
    print(f"\nFound {len(clf_paths)} classifier(s):")
    for p in clf_paths:
        print(f"  • {os.path.basename(p)}")

    # ── 2. Build label mapping (npz → ImageFolder) ──────────────────────────
    print(f"\nBuilding label mapping (npz → ImageFolder)...")
    label_mapping, if_classes = build_npz_to_imagefolder_mapping(args.dataset_root)
    print(f"  npz→IF mapping: {label_mapping}")

    # ── 3. Load test dataset ─────────────────────────────────────────────────
    print(f"\nLoading test set from {args.npz}...")
    test_dataset = PathMNISTTestDataset(args.npz, label_mapping)
    print(f"  Test images: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ── 4. Load SAEs (need them for feature extraction) ──────────────────────
    print(f"\nLoading SAEs...")
    sae_registry = {}  # label → (sae, cfg)

    # Base SAE
    if os.path.exists(BASE_SAE_PATH):
        sae, cfg = load_sae(BASE_SAE_PATH, args.device)
        layer = cfg.block_layer if hasattr(cfg, "block_layer") else -2
        label = f"base_layer{layer}"
        sae_registry[label] = (sae, cfg)
        print(f"  [OK] {label}: d_sae={cfg.d_sae}, layer={layer}")

    # MedMNIST SAEs
    for path in discover_medmnist_saes(MEDMNIST_SAE_DIR):
        sae, cfg = load_sae(path, args.device)
        layer = cfg.block_layer if hasattr(cfg, "block_layer") else "?"
        label = f"medmnist_layer{layer}"
        sae_registry[label] = (sae, cfg)
        print(f"  [OK] {label}: d_sae={cfg.d_sae}, layer={layer}")

    # ── 5. Load LoRA ViT ─────────────────────────────────────────────────────
    print(f"\nLoading ViT ({args.backbone}) + LoRA weights...")
    # Use any SAE config to initialise the ViT
    any_cfg = next(iter(sae_registry.values()))[1]
    vit = load_hooked_vit(any_cfg, "base", args.backbone, args.device)
    load_lora_weights(vit, args.lora_checkpoint, args.device)
    print("  LoRA ViT loaded.")

    # ── 6. Evaluate each classifier ──────────────────────────────────────────
    num_classes = len(if_classes)
    to_pil = transforms.ToPILImage()
    results = {}

    for clf_path in clf_paths:
        # Determine which SAE this classifier uses
        basename = os.path.basename(clf_path)  # medmnist_classifier_base_layer-2.pt
        sae_label = basename.replace("medmnist_classifier_", "").replace(".pt", "")

        if sae_label not in sae_registry:
            print(f"\n[SKIP] No SAE loaded for classifier '{sae_label}'")
            continue

        sae, cfg = sae_registry[sae_label]
        clf_model, ckpt_info = load_classifier(clf_path, args.device)
        layer = cfg.block_layer if hasattr(cfg, "block_layer") else -2
        module = cfg.module_name if hasattr(cfg, "module_name") else "resid"
        hook_locations = [(layer, module)]

        print(f"\n{'─' * 60}")
        print(f"TESTING: {sae_label}  (layer={layer}, d_sae={cfg.d_sae})")
        print(f"  Val acc (from training): {ckpt_info.get('best_val_acc', '?'):.2f}%")
        print(f"{'─' * 60}")

        all_preds = []
        all_labels = []

        for images, labels in tqdm(test_loader, desc=f"  {sae_label}"):
            # Get LoRA CLIP activations
            inputs = vit.processor(
                images=[to_pil(img) for img in images],
                text="",
                return_tensors="pt",
                padding=True,
            ).to(args.device)
            _, cache = vit.run_with_cache(hook_locations, **inputs)
            activations = cache[(layer, module)]
            cls_acts = activations[:, 0, :]  # CLS token

            # SAE features
            _, sae_cache = sae.run_with_cache(cls_acts)
            feat_acts = sae_cache["hook_hidden_post"]

            # Classifier prediction
            logits = clf_model(feat_acts)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Overall accuracy
        overall_acc = 100.0 * (all_preds == all_labels).mean()

        # Per-class accuracy
        per_class_acc = {}
        for cls_idx in range(num_classes):
            mask = all_labels == cls_idx
            if mask.sum() > 0:
                acc = 100.0 * (all_preds[mask] == cls_idx).mean()
                per_class_acc[if_classes[cls_idx]] = (acc, int(mask.sum()))

        print(f"\n  Overall Test Accuracy: {overall_acc:.2f}%")
        print(f"  Per-class breakdown:")
        for cls_name, (acc, count) in sorted(per_class_acc.items()):
            print(f"    {cls_name:<45s} {acc:6.2f}%  (n={count})")

        results[sae_label] = {
            "test_acc": overall_acc,
            "val_acc": ckpt_info.get("best_val_acc", 0),
            "per_class": per_class_acc,
            "layer": layer,
        }

    # ── 7. Summary table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'SAE':<30s} {'Layer':>6s} {'Val Acc':>10s} {'Test Acc':>10s}")
    print(f"{'─' * 58}")
    for label, info in sorted(results.items()):
        print(f"{label:<30s} {info['layer']:>6} "
              f"{info['val_acc']:>9.2f}% {info['test_acc']:>9.2f}%")
    print(f"{'─' * 58}")
    if results:
        best_label = max(results, key=lambda k: results[k]["test_acc"])
        print(f"\nBest on test: {best_label} ({results[best_label]['test_acc']:.2f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

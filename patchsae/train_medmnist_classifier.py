#!/usr/bin/env python3
"""
Train a separate linear classifier for EACH pre-trained SAE on MedMNIST.

All SAEs (base + medmnist) receive LoRA fine-tuned CLIP activations.

Produces one classifier per SAE so you can compare them:
  - base SAE        (layer -2, ImageNet)       → out/medmnist_classifier_base_layer-2.pt
  - medmnist SAE    (layer -3, LoRA fine-tuned) → out/medmnist_classifier_medmnist_layer-3.pt
  - medmnist SAE    (layer -1, LoRA fine-tuned) → out/medmnist_classifier_medmnist_layer-1.pt

Memory-efficient: features are written to numpy memmap files on disk.

Usage (from patchsae root):
    python train_medmnist_classifier.py --include_base
    python train_medmnist_classifier.py                   # medmnist SAEs only
    python train_medmnist_classifier.py --reuse_features  # skip re-extraction
"""

import argparse
import json
import os
import sys
import glob
import gc

# ── Silence tokenizer fork warnings BEFORE any HuggingFace imports ─────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
try:
    from tasks.utils import load_sae, load_hooked_vit
    from tasks.train_sae_lora_clip import load_lora_weights
    from src.sae_training.config import Config
except ImportError:
    print("[FATAL] Could not import project modules.")
    print("Run this script from the patchsae root directory:")
    print("  cd /home/sunayana/Documents/Concept_LoRA/patchsae")
    print("  python train_medmnist_classifier.py")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

BASE_SAE_PATH = "data/sae_weight/base/out.pt"
MEDMNIST_SAE_DIR = "out/checkpoints/medmnist"
LORA_CHECKPOINT_PATH = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
DATASET_ROOT = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
CLASSNAMES_PATH = "configs/classnames/medmnist_classnames.json"
FEATURE_CACHE_DIR = "out/medmnist_features"

BACKBONE = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
NUM_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def discover_medmnist_saes(base_dir):
    """Find final SAE checkpoints, keeping only the latest per layer."""
    pattern = os.path.join(base_dir, "*/final*/*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[WARN] No SAE checkpoints found matching {pattern}")
        return []

    # Deduplicate: keep only the latest checkpoint per block_layer
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


def load_selected_saes(base_sae_path, medmnist_sae_paths, device, include_base=False):
    """Load SAEs. Returns list of (sae, cfg, label) tuples."""
    saes = []

    if include_base and os.path.exists(base_sae_path):
        sae, cfg = load_sae(base_sae_path, device)
        layer = cfg.block_layer if hasattr(cfg, "block_layer") else -2
        label = f"base_layer{layer}"
        print(f"  [OK] Base SAE: layer={layer}, d_sae={cfg.d_sae}")
        saes.append((sae, cfg, label))
    elif include_base:
        print(f"  [SKIP] Base SAE not found: {base_sae_path}")

    for path in medmnist_sae_paths:
        sae, cfg = load_sae(path, device)
        layer = cfg.block_layer if hasattr(cfg, "block_layer") else "?"
        label = f"medmnist_layer{layer}"
        print(f"  [OK] MedMNIST SAE: layer={layer}, d_sae={cfg.d_sae}")
        print(f"        path: {path}")
        saes.append((sae, cfg, label))

    return saes


def get_transform():
    """Minimal transform for images going through CLIPProcessor.

    Only Resize + ToTensor (no Normalize).  The CLIPProcessor in the
    extraction loop handles normalisation.  This avoids the
    double-normalisation / PIL clipping bug.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# MEMORY-EFFICIENT FEATURE EXTRACTION  (one file per SAE)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features_for_sae_group(vit, saes_subset, dataloader, n_samples,
                                    cache_dir, split_name, device,
                                    labels_arr=None):
    """
    Extract features for a group of SAEs that share the same ViT model.
    Writes each SAE's features to its own memmap file.

    Args:
        vit: HookedVisionTransformer (base or LoRA-adapted)
        saes_subset: list of (sae, cfg, label) that use this ViT
        dataloader: image DataLoader
        n_samples: total samples
        cache_dir: output directory
        split_name: "train" or "val"
        device: torch device
        labels_arr: pre-allocated labels array (reused across groups)

    Returns:
        paths: dict label -> (feat_path, d_sae)
        labels_arr: numpy array of labels
    """
    # Gather hook locations needed for this group
    hook_locations = []
    for sae, cfg, label in saes_subset:
        layer = cfg.block_layer if hasattr(cfg, "block_layer") else -2
        module = cfg.module_name if hasattr(cfg, "module_name") else "resid"
        loc = (layer, module)
        if loc not in hook_locations:
            hook_locations.append(loc)

    # Create one memmap per SAE
    mmaps = {}
    paths = {}
    for sae, cfg, label in saes_subset:
        d_sae = cfg.d_sae
        feat_path = os.path.join(cache_dir, f"{split_name}_{label}.npy")
        mm = np.memmap(feat_path, dtype="float16", mode="w+",
                       shape=(n_samples, d_sae))
        mmaps[label] = mm
        paths[label] = (feat_path, d_sae)

    if labels_arr is None:
        labels_arr = np.empty(n_samples, dtype=np.int64)
    save_labels = labels_arr

    to_pil = transforms.ToPILImage()
    idx = 0
    group_labels = [l for _, _, l in saes_subset]

    for images, labs in tqdm(dataloader, desc=f"Extracting [{split_name}] ({', '.join(group_labels)})"):
        bs = images.size(0)

        inputs = vit.processor(
            images=[to_pil(img) for img in images],
            text="",
            return_tensors="pt",
            padding=True,
        ).to(device)

        output, cache = vit.run_with_cache(hook_locations, **inputs)

        for sae, cfg, label in saes_subset:
            layer = cfg.block_layer if hasattr(cfg, "block_layer") else -2
            module = cfg.module_name if hasattr(cfg, "module_name") else "resid"

            activations = cache[(layer, module)]  # [B, seq_len, d_in]
            cls_acts = activations[:, 0, :]       # [B, d_in]

            _, sae_cache = sae.run_with_cache(cls_acts)
            feat_acts = sae_cache["hook_hidden_post"]  # [B, d_sae]
            mmaps[label][idx:idx + bs] = feat_acts.cpu().half().numpy()

        save_labels[idx:idx + bs] = labs.numpy()
        idx += bs

        if idx % (50 * bs) < bs:
            torch.cuda.empty_cache()

    for mm in mmaps.values():
        mm.flush()

    for label, (fp, d) in paths.items():
        sz = os.path.getsize(fp) / 1e9
        print(f"    {label}: {fp} ({sz:.2f} GB)")

    return paths, save_labels


# ═════════════════════════════════════════════════════════════════════════════
# MEMMAP DATASET  (streams from disk during training)
# ═════════════════════════════════════════════════════════════════════════════

class MemmapDataset(Dataset):
    """Read features from a numpy memmap file on the fly."""

    def __init__(self, feat_path, labels, shape):
        self.data = np.memmap(feat_path, dtype="float16", mode="r", shape=shape)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx].copy()).float()
        y = int(self.labels[idx])
        return x, y


# ═════════════════════════════════════════════════════════════════════════════
# LINEAR CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def train_classifier(train_loader, val_loader, input_dim, num_classes,
                     num_epochs, lr, weight_decay, device):
    """Train and evaluate a linear probe with mini-batch streaming."""
    model = LinearProbe(input_dim, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for feats, labs in train_loader:
            feats, labs = feats.to(device), labs.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            correct += (logits.argmax(dim=-1) == labs).sum().item()
            total += feats.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = total_loss / total

        # ── Validate ──
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for feats, labs in val_loader:
                feats, labs = feats.to(device), labs.to(device)
                logits = model(feats)
                val_loss_sum += criterion(logits, labs).item() * feats.size(0)
                val_correct += (logits.argmax(dim=-1) == labs).sum().item()
                val_total += feats.size(0)

        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss_sum / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"Best: {best_val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train SAE-based classifier on MedMNIST")
    parser.add_argument("--base_sae", type=str, default=BASE_SAE_PATH)
    parser.add_argument("--medmnist_sae_dir", type=str, default=MEDMNIST_SAE_DIR)
    parser.add_argument("--dataset_root", type=str, default=DATASET_ROOT)
    parser.add_argument("--backbone", type=str, default=BACKBONE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save_dir", type=str, default="out",
                        help="Directory to save classifier checkpoints")
    parser.add_argument("--include_base", action="store_true",
                        help="Also include the base ImageNet SAE")
    parser.add_argument("--feature_cache", type=str, default=FEATURE_CACHE_DIR,
                        help="Directory to cache extracted features on disk")
    parser.add_argument("--reuse_features", action="store_true",
                        help="Reuse previously extracted features from --feature_cache")
    parser.add_argument("--lora_checkpoint", type=str, default=LORA_CHECKPOINT_PATH,
                        help="Path to LoRA checkpoint for CLIP")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("MEDMNIST SAE CLASSIFIER TRAINING  (one per SAE)")
    print("=" * 60)

    # ── 1. Load class names ──────────────────────────────────────────────────
    with open(CLASSNAMES_PATH) as f:
        classnames = json.load(f)
    num_classes = len(classnames)
    print(f"\nClasses ({num_classes}): {classnames}")

    # ── 2. Load SAEs ─────────────────────────────────────────────────────────
    print(f"\nLoading SAEs (include_base={args.include_base})...")
    medmnist_sae_paths = discover_medmnist_saes(args.medmnist_sae_dir)
    saes = load_selected_saes(args.base_sae, medmnist_sae_paths, args.device,
                              include_base=args.include_base)

    if not saes:
        print("[FATAL] No SAEs loaded. Exiting.")
        sys.exit(1)

    sae_info_list = [(label, cfg.d_sae, cfg.block_layer) for _, cfg, label in saes]
    print(f"\nSAEs to evaluate ({len(saes)}):")
    for label, d_sae, layer in sae_info_list:
        print(f"  • {label}: d_sae={d_sae}, layer={layer}")

    # ── Paths for cached features ────────────────────────────────────────────
    os.makedirs(args.feature_cache, exist_ok=True)
    meta_path = os.path.join(args.feature_cache, "meta.json")

    # ── Check if we can reuse cached features ────────────────────────────────
    can_reuse = False
    if args.reuse_features and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        cached_labels = set(meta.get("sae_labels", []))
        needed_labels = set(label for _, _, label in saes)
        if needed_labels.issubset(cached_labels):
            can_reuse = True
            n_train = meta["n_train"]
            n_val = meta["n_val"]
            train_feat_paths = meta["train_feat_paths"]
            val_feat_paths = meta["val_feat_paths"]
            train_label_path = meta["train_label_path"]
            val_label_path = meta["val_label_path"]
            print(f"\n[REUSE] Loading cached features from {args.feature_cache}")
        else:
            missing = needed_labels - cached_labels
            print(f"\n[WARN] Cached features missing SAEs: {missing}. Re-extracting.")

    if not can_reuse:
        # ── 3. Load ViT + apply LoRA weights ─────────────────────────────────
        print(f"\nLoading ViT ({args.backbone})...")
        ref_cfg = saes[0][1]
        vit = load_hooked_vit(ref_cfg, "base", args.backbone, args.device)
        print("  Base ViT loaded.")

        print(f"  Applying LoRA weights from: {args.lora_checkpoint}")
        load_lora_weights(vit, args.lora_checkpoint, args.device)
        print("  LoRA weights merged into ViT.")

        # ── 4. Load dataset ──────────────────────────────────────────────────
        print(f"\nLoading dataset from {args.dataset_root}...")
        transform = get_transform()
        full_dataset = datasets.ImageFolder(root=args.dataset_root, transform=transform)
        print(f"  Total images: {len(full_dataset)}")
        print(f"  Classes: {full_dataset.classes}")

        n_val = int(len(full_dataset) * VAL_SPLIT)
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"  Train: {n_train}, Val: {n_val}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

        # ── 5. Extract features to disk (one file per SAE) ──────────────────
        print(f"\nExtracting SAE features (LoRA ViT) → {args.feature_cache}/")

        print(f"\n  [train]")
        train_feat_paths, train_labels_arr = extract_features_for_sae_group(
            vit, saes, train_loader, n_train,
            args.feature_cache, "train", args.device,
        )
        train_label_path = os.path.join(args.feature_cache, "train_labels.npy")
        np.save(train_label_path, train_labels_arr)

        print(f"\n  [val]")
        val_feat_paths, val_labels_arr = extract_features_for_sae_group(
            vit, saes, val_loader, n_val,
            args.feature_cache, "val", args.device,
        )
        val_label_path = os.path.join(args.feature_cache, "val_labels.npy")
        np.save(val_label_path, val_labels_arr)

        # Save metadata for --reuse_features
        with open(meta_path, "w") as f:
            json.dump({
                "sae_labels": [label for _, _, label in saes],
                "n_train": n_train,
                "n_val": n_val,
                "train_feat_paths": train_feat_paths,
                "val_feat_paths": val_feat_paths,
                "train_label_path": train_label_path,
                "val_label_path": val_label_path,
            }, f, indent=2)

        # Free ViT + SAE GPU memory before training
        del vit, train_loader, val_loader
        for sae_obj, _, _ in saes:
            sae_obj.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        print("\n  ViT and SAEs freed from GPU.")

    # ── 6. Train one classifier per SAE ──────────────────────────────────────
    train_labels = np.load(train_label_path)
    val_labels = np.load(val_label_path)

    results = {}
    os.makedirs(args.save_dir, exist_ok=True)

    for sae_obj, cfg, label in saes:
        d_sae = cfg.d_sae
        feat_path_train, _ = train_feat_paths[label]
        feat_path_val, _ = val_feat_paths[label]

        print(f"\n{'─' * 60}")
        print(f"TRAINING CLASSIFIER:  {label}  (d_sae={d_sae}, layer={cfg.block_layer})")
        print(f"{'─' * 60}")

        # Build memmap-backed datasets
        train_ds = MemmapDataset(feat_path_train, train_labels, (n_train, d_sae))
        val_ds = MemmapDataset(feat_path_val, val_labels, (n_val, d_sae))

        clf_train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                                      num_workers=0, pin_memory=True)
        clf_val_loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                                    num_workers=0, pin_memory=True)

        model, best_acc = train_classifier(
            clf_train_loader, clf_val_loader,
            input_dim=d_sae,
            num_classes=num_classes,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=WEIGHT_DECAY,
            device=args.device,
        )

        # Save classifier
        save_path = os.path.join(args.save_dir, f"medmnist_classifier_{label}.pt")
        save_dict = {
            "model_state_dict": model.state_dict(),
            "input_dim": d_sae,
            "num_classes": num_classes,
            "classnames": classnames,
            "best_val_acc": best_acc,
            "sae_label": label,
            "sae_layer": cfg.block_layer,
        }
        torch.save(save_dict, save_path)
        print(f"  → Saved: {save_path}  (best val acc: {best_acc:.2f}%)")

        results[label] = best_acc
        del model
        torch.cuda.empty_cache()

    # ── 7. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'SAE':<30s} {'Layer':>6s} {'Val Acc':>10s}")
    print(f"{'─' * 48}")
    for sae_obj, cfg, label in saes:
        acc = results[label]
        print(f"{label:<30s} {cfg.block_layer:>6} {acc:>9.2f}%")
    print(f"{'─' * 48}")
    best_label = max(results, key=results.get)
    print(f"\nBest: {best_label} ({results[best_label]:.2f}%)")
    print(f"Classifiers saved in: {args.save_dir}/")
    print(f"Feature cache: {args.feature_cache}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Diagnose why fine-tuned models show low accuracy.
Checks: label mapping, class name alignment, prediction distribution.
"""

import json, os, sys, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
from collections import Counter

try:
    from tasks.utils import load_sae, load_hooked_vit
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] {e}"); sys.exit(1)

try:
    import clip as openai_clip
except ImportError:
    openai_clip = None

TRAIN_ROOT  = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
NPZ_PATH    = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
CLASSNAMES_PATH = "configs/classnames/medmnist_classnames.json"
BACKBONE    = "openai/clip-vit-base-patch16"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

def get_transform():
    """Minimal transform for images going through CLIPProcessor.
    No Normalize — processor handles it. Avoids double-normalisation bug."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

class NpzTestDataset(Dataset):
    def __init__(self, npz_path, transform, label_mapping=None):
        data = np.load(npz_path)
        self.images = data['test_images']
        self.labels = data['test_labels'].flatten()
        self.label_mapping = label_mapping
        self.transform = transform
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = self.transform(Image.fromarray(self.images[idx]))
        raw_label = int(self.labels[idx])
        mapped_label = self.label_mapping[raw_label] if self.label_mapping else raw_label
        return img, mapped_label, raw_label


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Label alignment check")
    print("=" * 70)

    # 1. Check classnames
    with open(CLASSNAMES_PATH) as f:
        classnames = json.load(f)
    print(f"\nClassnames from JSON ({len(classnames)}):")
    for i, c in enumerate(classnames):
        print(f"  {i}: '{c}'")

    # 2. Check ImageFolder ordering
    print(f"\nImageFolder class_to_idx from {TRAIN_ROOT}:")
    ds = datasets.ImageFolder(root=TRAIN_ROOT)
    for name, idx in sorted(ds.class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: '{name}'")

    # 3. Check npz label distribution
    data = np.load(NPZ_PATH)
    test_labels = data['test_labels'].flatten()
    train_labels = data['train_labels'].flatten()
    print(f"\nNPZ test label distribution:")
    for label, count in sorted(Counter(test_labels).items()):
        print(f"  {label} ({MEDMNIST_CLASSES[label]}): {count}")

    # 4. Check the mapping
    if_name_to_idx = {n.replace('_', ' ').lower(): i for n, i in ds.class_to_idx.items()}
    print(f"\nLabel mapping (npz_idx → imagefolder_idx):")
    mapping = {}
    for npz_idx, npz_name in enumerate(MEDMNIST_CLASSES):
        key = npz_name.lower()
        if key in if_name_to_idx:
            mapped = if_name_to_idx[key]
            mapping[npz_idx] = mapped
            match = "✓" if npz_idx == mapped else "⚠ REMAPPED"
            print(f"  npz {npz_idx} ('{npz_name}') → IF {mapped} ({match})")
        else:
            print(f"  npz {npz_idx} ('{npz_name}') → NOT FOUND ✗")
            mapping[npz_idx] = npz_idx

    # 5. Check if classnames[i] matches ImageFolder index i
    print(f"\nClassnames vs ImageFolder alignment:")
    for i, c in enumerate(classnames):
        if_idx = if_name_to_idx.get(c.lower(), "NOT FOUND")
        match = "✓" if if_idx == i else f"✗ (IF idx={if_idx})"
        print(f"  classnames[{i}] = '{c}' → {match}")

    # 6. Quick prediction check with base CLIP (no SAE)
    print(f"\n{'='*70}")
    print("Quick prediction test with base CLIP (first 256 test images)")
    print(f"{'='*70}")

    # Load a dummy SAE just to get ref_cfg
    from tasks.utils import load_sae
    sae_paths = []
    import glob
    for p in sorted(glob.glob("out/checkpoints/medmnist/*/final*/*.pt")):
        sae_paths.append(p); break
    if not sae_paths:
        sae_paths = ["data/sae_weight/base/out.pt"]
    sae, ref_cfg = load_sae(sae_paths[0], DEVICE)
    del sae

    vit = load_hooked_vit(ref_cfg, "base", BACKBONE, DEVICE)
    vit.eval()

    # Text features
    mean_f = 0
    with torch.no_grad():
        for tmpl in openai_imagenet_template:
            ids = [vit.processor(text=tmpl(c), return_tensors="pt", padding=False,
                                 truncation=True).input_ids[0] for c in classnames]
            padded = pad_sequence(ids, batch_first=True).to(DEVICE)
            f = vit.model.get_text_features(padded)
            f = f / f.norm(dim=-1, keepdim=True)
            mean_f += f
        mean_f /= len(openai_imagenet_template)
        text_features = mean_f / mean_f.norm(dim=-1, keepdim=True)

    # Test with MAPPED labels
    transform = get_transform()
    test_ds = NpzTestDataset(NPZ_PATH, transform, label_mapping=mapping)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    all_preds = []
    all_mapped = []
    all_raw = []
    to_pil = transforms.ToPILImage()

    with torch.no_grad():
        for batch_idx, (images, mapped_labels, raw_labels) in enumerate(loader):
            if batch_idx >= 4:  # first 256 images
                break
            inputs = vit.processor(images=[to_pil(img) for img in images],
                                   text="", return_tensors="pt", padding=True).to(DEVICE)
            out = vit(return_type="output", **inputs)
            logits = vit.model.logit_scale.exp() * out.image_embeds @ text_features.t()
            preds = logits.argmax(dim=-1).cpu()
            all_preds.extend(preds.tolist())
            all_mapped.extend(mapped_labels.tolist())
            all_raw.extend(raw_labels.tolist())

    all_preds = np.array(all_preds)
    all_mapped = np.array(all_mapped)
    all_raw = np.array(all_raw)

    acc_mapped = (all_preds == all_mapped).mean() * 100
    acc_raw = (all_preds == all_raw).mean() * 100

    print(f"\n  Accuracy with MAPPED labels: {acc_mapped:.2f}%")
    print(f"  Accuracy with RAW npz labels: {acc_raw:.2f}%")

    print(f"\n  Prediction distribution:")
    for pred_idx, count in sorted(Counter(all_preds).items()):
        print(f"    pred={pred_idx} ({classnames[pred_idx]}): {count}")

    print(f"\n  Ground truth (mapped) distribution:")
    for gt_idx, count in sorted(Counter(all_mapped).items()):
        print(f"    gt={gt_idx} ({classnames[gt_idx]}): {count}")

    print(f"\n  Ground truth (raw npz) distribution:")
    for gt_idx, count in sorted(Counter(all_raw).items()):
        print(f"    gt={gt_idx} ({MEDMNIST_CLASSES[gt_idx]}): {count}")

    # Confusion: show which classes are confused
    print(f"\n  Per-class accuracy (mapped labels):")
    for cls in range(len(classnames)):
        mask = all_mapped == cls
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == cls).mean() * 100
        most_common_pred = Counter(all_preds[mask]).most_common(1)[0]
        print(f"    {classnames[cls]:<45s}: {cls_acc:5.1f}%  "
              f"(most predicted: {classnames[most_common_pred[0]]} [{most_common_pred[1]}/{mask.sum()}])")

    print(f"\n{'='*70}")
    print("If 'Accuracy with RAW npz labels' >> 'Accuracy with MAPPED labels',")
    print("then the label mapping is the problem.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
One-time setup: create imagefolder-compatible symlink trees for
OxfordPets and FGVC Aircraft (which store images flat + split files).

Run once before SAE training:
  python3 setup_imagefolder_datasets.py
"""

import json
import os
from pathlib import Path


DATA_ROOT = Path("/home/sunayana/Documents/Concept_LoRA/data")


def make_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


# =============================================================================
# OxfordPets
# =============================================================================
def setup_oxford_pets():
    split_file = DATA_ROOT / "OxfordPets" / "split_zhou_OxfordPets.json"
    images_dir = DATA_ROOT / "OxfordPets" / "images"
    out_dir = DATA_ROOT / "oxford_pets_imagefolder" / "train"

    with open(split_file) as f:
        splits = json.load(f)

    # Combine train + val for SAE training (more data = better)
    items = splits["train"] + splits.get("val", [])

    count = 0
    for filename, class_id, classname in items:
        src = images_dir / filename
        dst = out_dir / classname / filename
        if src.exists():
            make_symlink(src.resolve(), dst)
            count += 1

    print(f"[OxfordPets] {count} symlinks → {out_dir}")
    classes = sorted(set(x[2] for x in items))
    print(f"[OxfordPets] {len(classes)} classes")


# =============================================================================
# FGVC Aircraft (variant-level, 100 classes)
# =============================================================================
def setup_fgvc():
    train_file = DATA_ROOT / "fgvc_aircraft" / "images_variant_train.txt"
    val_file = DATA_ROOT / "fgvc_aircraft" / "images_variant_val.txt"
    images_dir = DATA_ROOT / "fgvc_aircraft" / "images"
    out_dir = DATA_ROOT / "fgvc_imagefolder" / "train"

    items = []
    for split_file in [train_file, val_file]:
        if not split_file.exists():
            continue
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "<image_id> <variant name possibly with spaces>"
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    img_id, variant = parts
                    items.append((img_id.strip(), variant.strip()))

    count = 0
    for img_id, variant in items:
        # Class dir name: replace '/' with '-' for filesystem safety
        classname = variant.replace("/", "-")
        src = images_dir / f"{img_id}.jpg"
        dst = out_dir / classname / f"{img_id}.jpg"
        if src.exists():
            make_symlink(src.resolve(), dst)
            count += 1

    classes = sorted(set(v.replace("/", "-") for _, v in items))
    print(f"[FGVC]       {count} symlinks → {out_dir}")
    print(f"[FGVC]       {len(classes)} classes (variants)")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Setting up imagefolder datasets...\n")
    setup_oxford_pets()
    setup_fgvc()
    print("\nDone. You can now run the SAE training script.")

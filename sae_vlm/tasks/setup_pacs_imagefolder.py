#!/usr/bin/env python3
"""
Set up PACS leave-one-out ImageFolder directories for SAE training.

Creates four merged directories under OUTPUT_ROOT, one per held-out domain.
Each directory is an ImageFolder whose class subdirs contain symlinks to the
images from the three source domains.

After running this script, train_sae.py can use dataset names:
    pacs_no_photo, pacs_no_art, pacs_no_cartoon, pacs_no_sketch

Usage:
    python tasks/setup_pacs_imagefolder.py --pacs_root /path/to/PACS

    # Custom output location:
    python tasks/setup_pacs_imagefolder.py \
        --pacs_root /path/to/PACS \
        --output_root /home/sunayana/Documents/Concept_LoRA/data/pacs_imagefolder

Expected PACS structure:
    {pacs_root}/
      art_painting/
        dog/  elephant/  giraffe/  guitar/  horse/  house/  person/
      cartoon/   ...
      photo/     ...
      sketch/    ...
"""

import argparse
import os
from pathlib import Path

PACS_DOMAINS  = ["art_painting", "cartoon", "photo", "sketch"]
PACS_CLASSES  = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
DEFAULT_OUT   = Path("/home/sunayana/Documents/Concept_LoRA/data/pacs_imagefolder")

# Maps split name → held-out domain
SPLITS = {
    "no_photo":       "photo",
    "no_art_painting":"art_painting",
    "no_cartoon":     "cartoon",
    "no_sketch":      "sketch",
}


def check_pacs_root(pacs_root: Path):
    missing = [d for d in PACS_DOMAINS if not (pacs_root / d).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing PACS domain directories under {pacs_root}: {missing}\n"
            f"Expected: art_painting/, cartoon/, photo/, sketch/"
        )
    print(f"[OK] PACS root verified: {pacs_root}")
    for dom in PACS_DOMAINS:
        imgs = list((pacs_root / dom).rglob("*.jpg")) + list((pacs_root / dom).rglob("*.png"))
        print(f"     {dom}: {len(imgs):5d} images")


def setup_split(pacs_root: Path, output_root: Path, split_name: str, held_out: str):
    source_doms = [d for d in PACS_DOMAINS if d != held_out]
    split_dir   = output_root / split_name
    print(f"\n[INFO] Creating split '{split_name}' (sources: {source_doms}) → {split_dir}")

    for cls in PACS_CLASSES:
        cls_dir = split_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for dom in source_doms:
            src_cls_dir = pacs_root / dom / cls
            if not src_cls_dir.is_dir():
                print(f"  [WARN] Missing: {src_cls_dir} — skipping")
                continue

            imgs = sorted(
                [f for f in src_cls_dir.iterdir()
                 if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
            )
            for img_path in imgs:
                # Hard link: {domain}__{original_filename} to avoid collisions.
                # Hard links are on the same inode — zero extra disk, followed by all tools.
                link_name = cls_dir / f"{dom}__{img_path.name}"
                if link_name.exists():
                    continue
                os.link(img_path.resolve(), link_name)

        n_links = len(list(cls_dir.iterdir()))
        print(f"  {cls:10s}: {n_links} images")


def main():
    parser = argparse.ArgumentParser(description="Set up PACS leave-one-out ImageFolder dirs")
    parser.add_argument("--pacs_root", required=True,
                        help="PACS root directory (contains art_painting/, cartoon/, ...)")
    parser.add_argument("--output_root", default=str(DEFAULT_OUT),
                        help="Where to write the merged ImageFolder splits")
    args = parser.parse_args()

    pacs_root   = Path(args.pacs_root)
    output_root = Path(args.output_root)

    check_pacs_root(pacs_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split_name, held_out in SPLITS.items():
        setup_split(pacs_root, output_root, split_name, held_out)

    print(f"\n[DONE] ImageFolder splits written to: {output_root}")
    print("Update configs/datasets/registry.yaml local_path entries if you used a custom output_root.")
    print("Then run SAE training with:")
    for split_name in SPLITS:
        print(f"  python train_sae.py --model clipood_vit_b16 --dataset pacs_{split_name} ...")


if __name__ == "__main__":
    main()

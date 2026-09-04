#!/usr/bin/env python3
"""
Convert KITTI object detection parquet data → ImageFolder classification.

Each image is assigned the class of its most-frequently annotated object
(excluding DontCare and Misc). Images with no valid annotation are skipped.

Output structure:
  data/kitti_imagefolder/
    train/
      Car/   Pedestrian/   Cyclist/  ...
    test/
      Car/   Pedestrian/   Cyclist/  ...
"""
import argparse
import glob
import io
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SKIP_CLASSES = {"DontCare", "Misc"}

KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]


def dominant_class(labels):
    """Return the most frequent non-skip object class, or None."""
    if not hasattr(labels, "__len__") or len(labels) == 0:
        return None
    counts = Counter()
    for obj in labels:
        t = obj.get("type", "") if isinstance(obj, dict) else ""
        if t and t not in SKIP_CLASSES:
            counts[t] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def convert_split(parquet_glob: str, out_dir: str):
    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {parquet_glob}")

    os.makedirs(out_dir, exist_ok=True)
    saved = Counter()
    skipped = 0

    for fpath in tqdm(files, desc=f"Shards → {os.path.basename(out_dir)}"):
        df = pd.read_parquet(fpath)
        for i, row in df.iterrows():
            cls = dominant_class(row["label"])
            if cls is None:
                skipped += 1
                continue

            cls_dir = os.path.join(out_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)

            img_data = row["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            elif isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data.astype(np.uint8)).convert("RGB")
            else:
                skipped += 1
                continue

            fname = f"{os.path.basename(fpath).split('.')[0]}_{i:06d}.jpg"
            img.save(os.path.join(cls_dir, fname), "JPEG", quality=95)
            saved[cls] += 1

    print(f"  Saved {sum(saved.values())} images, skipped {skipped}")
    for cls in sorted(saved):
        print(f"    {cls}: {saved[cls]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", default="/home/sunayana/Documents/Concept_LoRA/data/kitti/data")
    parser.add_argument("--out_root", default="/home/sunayana/Documents/Concept_LoRA/data/kitti_imagefolder")
    args = parser.parse_args()

    print("Converting KITTI → ImageFolder...")
    print(f"  Source: {args.kitti_root}")
    print(f"  Target: {args.out_root}")

    convert_split(
        os.path.join(args.kitti_root, "train-*.parquet"),
        os.path.join(args.out_root, "train"),
    )
    # KITTI test annotations are withheld — reuse train as the test/val reference.
    # Just symlink the train directory as test so dataset loaders find it.
    test_dir = os.path.join(args.out_root, "test")
    if not os.path.exists(test_dir):
        os.symlink(os.path.join(args.out_root, "train"), test_dir)
        print(f"  test/ → symlink to train/ (KITTI test has no public labels)")
    print(f"\nDone. ImageFolder at: {args.out_root}")


if __name__ == "__main__":
    main()

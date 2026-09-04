#!/usr/bin/env python3
"""
Convert Cityscapes segmentation parquet data → ImageFolder scene classification.

Scene label = the dominant semantic class (by pixel count) from segmentation_19,
ignoring background pixels (value < 0).

Cityscapes 19-class mapping:
  0=road  1=sidewalk  2=building  3=wall  4=fence  5=pole
  6=traffic_light  7=traffic_sign  8=vegetation  9=terrain
  10=sky  11=person  12=rider  13=car  14=truck  15=bus
  16=train  17=motorcycle  18=bicycle

Output structure:
  data/cityscapes_imagefolder/
    train/
      road/  building/  car/  ...
    val/
      ...
"""
import argparse
import glob
import io
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


def dominant_scene_class(seg19):
    """Return the name of the most-common non-background semantic class."""
    if seg19 is None:
        return None
    # seg19 is a (H,) array of rows, each row is a (W,) float array
    if isinstance(seg19, np.ndarray) and seg19.dtype == object:
        rows = [r for r in seg19 if isinstance(r, np.ndarray)]
        if not rows:
            return None
        seg = np.stack(rows)           # (H, W)
    else:
        seg = np.asarray(seg19)

    flat = seg.flatten()
    flat = flat[flat >= 0].astype(int)  # drop background
    if len(flat) == 0:
        return None
    cls_idx = np.bincount(flat, minlength=len(CLASS_NAMES)).argmax()
    if cls_idx >= len(CLASS_NAMES):
        return None
    return CLASS_NAMES[cls_idx]


def decode_image(img_data):
    if isinstance(img_data, dict) and "bytes" in img_data:
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    if isinstance(img_data, np.ndarray):
        if img_data.dtype == object:
            # Cityscapes format: (C,) of (H,) of (W,) float32 arrays → (C,H,W)
            try:
                chw = np.stack([np.stack(list(ch)) for ch in img_data])
                hwc = np.transpose(chw, (1, 2, 0))
                hwc = (hwc * 255).clip(0, 255).astype(np.uint8)
                return Image.fromarray(hwc).convert("RGB")
            except Exception:
                return None
        return Image.fromarray(img_data.astype(np.uint8)).convert("RGB")
    return None


def convert_split(parquet_glob: str, out_dir: str):
    files = sorted(glob.glob(parquet_glob))
    if not files:
        print(f"  No files for {parquet_glob} — skipping")
        return

    os.makedirs(out_dir, exist_ok=True)
    saved = Counter()
    skipped = 0

    for fpath in tqdm(files, desc=f"Shards → {os.path.basename(out_dir)}"):
        df = pd.read_parquet(fpath)
        for i, row in df.iterrows():
            cls = dominant_scene_class(row.get("segmentation_19"))
            if cls is None:
                skipped += 1
                continue

            cls_dir = os.path.join(out_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)

            img = decode_image(row["image"])
            if img is None:
                skipped += 1
                continue

            fname = f"{Path(fpath).stem}_{i:06d}.jpg"
            img.save(os.path.join(cls_dir, fname), "JPEG", quality=95)
            saved[cls] += 1

    print(f"  Saved {sum(saved.values())} images, skipped {skipped}")
    for cls in sorted(saved):
        print(f"    {cls}: {saved[cls]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root",
                        default="/home/sunayana/Documents/Concept_LoRA/data/cityscapes/data")
    parser.add_argument("--out_root",
                        default="/home/sunayana/Documents/Concept_LoRA/data/cityscapes_imagefolder")
    args = parser.parse_args()

    print("Converting Cityscapes → ImageFolder...")
    print(f"  Source: {args.cityscapes_root}")
    print(f"  Target: {args.out_root}")

    convert_split(
        os.path.join(args.cityscapes_root, "train-*.parquet"),
        os.path.join(args.out_root, "train"),
    )
    convert_split(
        os.path.join(args.cityscapes_root, "validation-*.parquet"),
        os.path.join(args.out_root, "val"),
    )
    print(f"\nDone. ImageFolder at: {args.out_root}")


if __name__ == "__main__":
    main()

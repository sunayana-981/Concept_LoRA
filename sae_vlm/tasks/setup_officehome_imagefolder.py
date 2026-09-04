#!/usr/bin/env python3
"""
Build a merged OfficeHome ImageFolder by symlinking all 4 domains.

OfficeHome has Art/ Clipart/ Product/ Real World/ each with 65 class subdirs.
This creates:
  data/officehome_imagefolder/
    Alarm_Clock/
      Art_0001.jpg  (symlinks or copies)
      Clipart_0001.jpg
      ...
    Backpack/
      ...

Usage:
  python tasks/setup_officehome_imagefolder.py [--copy]
"""
import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm

DOMAINS = ["Art", "Clipart", "Product", "Real World"]
SRC = Path("/home/sunayana/Documents/Concept_LoRA/data/OfficeHomeDataset_10072016")
DST = Path("/home/sunayana/Documents/Concept_LoRA/data/officehome_imagefolder")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--dst", default=str(DST))
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    total = 0
    for domain in DOMAINS:
        domain_dir = src / domain
        if not domain_dir.is_dir():
            print(f"WARNING: domain dir missing: {domain_dir}")
            continue
        classes = sorted(c for c in os.listdir(domain_dir)
                         if os.path.isdir(domain_dir / c))
        prefix = domain.replace(" ", "_")
        for cls in tqdm(classes, desc=domain):
            cls_dst = dst / cls
            cls_dst.mkdir(exist_ok=True)
            for fname in os.listdir(domain_dir / cls):
                if fname.startswith("."):
                    continue
                src_file = domain_dir / cls / fname
                if not src_file.is_file():
                    continue
                stem, ext = os.path.splitext(fname)
                dst_file = cls_dst / f"{prefix}_{stem}{ext}"
                if dst_file.exists():
                    continue
                if args.copy:
                    shutil.copy2(src_file, dst_file)
                else:
                    dst_file.symlink_to(src_file.resolve())
                total += 1

    classes_found = sorted(os.listdir(dst))
    print(f"\nDone. {total} files linked/copied into {len(classes_found)} classes.")
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()

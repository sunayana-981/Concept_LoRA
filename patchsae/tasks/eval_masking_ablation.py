#!/usr/bin/env python3
"""Evaluate a controlled masking-fraction SAE sweep on target and ImageNet.

The input manifest is produced by ``run_masking_ablation.sh``. Every checkpoint
is evaluated with the target dataset's LoRA model on (i) the target test split
and (ii) one fixed, seeded ImageNet subset. Results are written in tidy form.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from tasks.eval_matrix import compute_sae_quality, compute_zeroshot_acc
from tasks.rebuttal_common import build_dataset_splits, get_sae, get_vit, load_registry


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--imagenet_dir", required=True)
    p.add_argument("--dataset_registry", default="configs/rebuttal_datasets.json")
    p.add_argument("--out", default="out/masking_ablation/raw_results.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backbone", default="openai/clip-vit-base-patch16")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_images", type=int, default=10000)
    p.add_argument("--imagenet_max_images", type=int, default=5000)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=2026,
                   help="Fixed evaluation-subset seed; independent of training seed.")
    p.add_argument("--l0_threshold", type=float, default=1e-5)
    p.add_argument("--label_entropy_topn", type=int, default=1000)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    with open(args.manifest) as f:
        runs = json.load(f)
    registry = load_registry(args.dataset_registry)
    registry["imagenet_subset"] = {"type": "imagefolder", "path": args.imagenet_dir}

    # Build the old-domain subset once so every fraction/seed sees the same images.
    old_cap = args.max_images
    args.max_images = args.imagenet_max_images
    imagenet = build_dataset_splits("imagenet_subset", registry["imagenet_subset"], args)
    # ImageFolder labels are usually WordNet IDs; CLIP prompts require names.
    class_file = Path(_root) / "configs/classnames/imagenet_classnames.txt"
    wnid_to_name = {}
    for line in class_file.read_text().splitlines():
        if line.strip():
            wnid, name = line.split(maxsplit=1)
            wnid_to_name[wnid] = name
    folder_classes = imagenet[1].dataset.dataset.classes if hasattr(imagenet[1], "dataset") else []
    if folder_classes and all(c in wnid_to_name for c in folder_classes):
        imagenet = ([wnid_to_name[c] for c in folder_classes], imagenet[1], imagenet[2])
    elif folder_classes:
        raise ValueError("ImageNet folders must be WordNet IDs covered by " + str(class_file))
    args.max_images = old_cap

    target_splits = {}
    rows = []
    for run in runs:
        required = {"dataset", "seed", "protect_frac", "checkpoint", "lora_checkpoint"}
        missing = required - run.keys()
        if missing:
            raise ValueError(f"Manifest row missing {sorted(missing)}: {run}")
        if not os.path.isfile(run["checkpoint"]):
            raise FileNotFoundError(run["checkpoint"])
        dataset = run["dataset"]
        if dataset not in target_splits:
            target_splits[dataset] = build_dataset_splits(dataset, registry[dataset], args)
        lora_paths = {dataset: run["lora_checkpoint"]}
        vit = get_vit("lora", dataset, lora_paths, args)
        sae, cfg = get_sae(run["checkpoint"], args.device)

        for domain, splits in (("target", target_splits[dataset]), ("imagenet", imagenet)):
            if args.dry_run:
                acc, n_images = float("nan"), len(splits[2])
                quality = {k: float("nan") for k in ("l0", "dead_frac", "recon_cosine", "fve")}
            else:
                acc, n_images = compute_zeroshot_acc(
                    vit, sae, cfg, splits[0], splits[2], args
                )
                quality = compute_sae_quality(vit, sae, cfg, splits[2], args)
            rows.append({
                "dataset": dataset,
                "training_seed": int(run["seed"]),
                "protect_frac": float(run["protect_frac"]),
                "domain": domain,
                "accuracy": acc,
                "n_images": n_images,
                "checkpoint": run["checkpoint"],
                **quality,
            })
            print(f"{dataset} seed={run['seed']} protect={run['protect_frac']:.2f} "
                  f"domain={domain} acc={acc:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    metadata = vars(args) | {"n_manifest_runs": len(runs), "n_rows": len(rows)}
    with open(out.with_suffix(".metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

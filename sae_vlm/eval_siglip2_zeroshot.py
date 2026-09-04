#!/usr/bin/env python3
"""Run SigLIP2 zero-shot top-1 accuracy across a requested dataset suite.

Writes a CSV with one row per dataset and status=ok/failed.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset

from src.models.registry import get_backbone
from src.data.dataset_registry import load_registry, get_classnames, get_label_key
from src.eval.classification import zero_shot_accuracy


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/home/sunayana/Documents/Concept_LoRA/data")


def load_imagenet_wnid_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def classnames_from_imagefolder_with_wnid(ds, wnid_map: Dict[str, str]) -> List[str]:
    label_feat = ds.features.get("label")
    if label_feat is None or not hasattr(label_feat, "names"):
        raise ValueError("imagefolder dataset has no ClassLabel names")
    out = []
    for wnid in label_feat.names:
        out.append(wnid_map.get(wnid, wnid))
    return out


def try_load_dataset(candidates: List[dict]):
    last_err = None
    for cand in candidates:
        kind = cand["kind"]
        try:
            if kind == "registry":
                reg = load_registry()
                key = cand["key"]
                cfg = reg[key]
                split = cand.get("split", cfg.get("split", "train"))
                local = cfg.get("local_path")
                if local and Path(local).is_dir():
                    ds = load_dataset("imagefolder", data_dir=local, split=split,
                                      trust_remote_code=cfg.get("trust_remote_code", False))
                else:
                    kwargs = {
                        "path": cfg["hf_path"],
                        "split": split,
                        "trust_remote_code": cfg.get("trust_remote_code", False),
                    }
                    if "hf_name" in cfg:
                        kwargs["name"] = cfg["hf_name"]
                    ds = load_dataset(**kwargs)
                return ds, cand

            if kind == "imagefolder":
                ds = load_dataset("imagefolder", data_dir=str(cand["path"]), split=cand.get("split", "train"))
                return ds, cand

            if kind == "hf":
                kwargs = {"path": cand["path"], "split": cand.get("split", "train")}
                if "name" in cand:
                    kwargs["name"] = cand["name"]
                if "trust_remote_code" in cand:
                    kwargs["trust_remote_code"] = cand["trust_remote_code"]
                ds = load_dataset(**kwargs)
                return ds, cand

            raise ValueError(f"Unknown candidate kind: {kind}")
        except Exception as e:
            last_err = e
    raise RuntimeError(str(last_err) if last_err is not None else "no candidates")


def resolve_classnames(ds, chosen: dict, imagenet_wnid_map: Dict[str, str]) -> Tuple[List[str], str]:
    # Explicit wnid mapping for ImageNet-style imagefolder datasets.
    if chosen.get("use_imagenet_wnid_map", False):
        return classnames_from_imagefolder_with_wnid(ds, imagenet_wnid_map), "label"

    if chosen["kind"] == "registry":
        key = chosen["key"]
        reg = load_registry()
        cls = get_classnames(key, dataset=ds, registry=reg)
        lbl = get_label_key(key, ds, registry=reg)
        return cls, lbl

    # Generic fallback.
    label_key = "label" if "label" in ds.features else next(iter(ds.features.keys()))
    feat = ds.features.get(label_key)
    if hasattr(feat, "names"):
        return list(feat.names), label_key
    raise ValueError("Could not infer classnames from dataset features")


def main():
    p = argparse.ArgumentParser(description="SigLIP2 zero-shot benchmark suite")
    p.add_argument("--model", default="siglip2_base_patch16_224")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Optional subset of display names to run (e.g. 'Caltech101 Pets').")
    p.add_argument("--out_csv", default="out/siglip2_zeroshot_suite.csv")
    p.add_argument("--out_json", default="out/siglip2_zeroshot_suite.json")
    args = p.parse_args()

    imagenet_map = load_imagenet_wnid_map(ROOT / "configs" / "classnames" / "imagenet_classnames.txt")

    # Requested by user:
    # Caltech101, Pets, ImageNet-A, SUN397, ImageNet-Sketch, Corruption,
    # Food101, DTD, UCF101, StanfordCars, Flowers102, FGVC, EuroSAT,
    # ChestMNIST, PathMNIST.
    dataset_specs = {
        "Caltech101": [
            {"kind": "registry", "key": "caltech101", "split": "train"},
        ],
        "Pets": [
            {"kind": "registry", "key": "oxford_pets", "split": "train"},
        ],
        "ImageNet-A": [
            {"kind": "imagefolder", "path": DATA_ROOT / "imagenet-a", "split": "train", "use_imagenet_wnid_map": True},
        ],
        "SUN397": [
            {"kind": "hf", "path": "sun397", "split": "train"},
        ],
        "ImageNet-Sketch": [
            {"kind": "imagefolder", "path": DATA_ROOT / "sketch", "split": "train", "use_imagenet_wnid_map": True},
            {"kind": "registry", "key": "imagenet_sketch", "split": "test"},
        ],
        "Corruption": [
            {"kind": "hf", "path": "stanford_cars", "split": "gaussian_noise"},
            {"kind": "hf", "path": "stanford_cars", "split": "contrast"},
        ],
        "Food101": [
            {"kind": "registry", "key": "food101", "split": "train"},
        ],
        "DTD": [
            {"kind": "registry", "key": "dtd", "split": "train"},
        ],
        "UCF101": [
            {"kind": "registry", "key": "ucf101", "split": "train"},
        ],
        "StanfordCars": [
            {"kind": "hf", "path": "stanford_cars", "split": "test"},
        ],
        "Flowers102": [
            {"kind": "imagefolder", "path": DATA_ROOT / "flowers102_imagefolder", "split": "test"},
            {"kind": "registry", "key": "oxford_flowers", "split": "train"},
        ],
        "FGVC": [
            {"kind": "registry", "key": "fgvc", "split": "train"},
        ],
        "EuroSAT": [
            {"kind": "registry", "key": "eurosat", "split": "train"},
        ],
        "ChestMNIST": [
            {"kind": "imagefolder", "path": DATA_ROOT / "chestmnist_imagefolder", "split": "train"},
            {"kind": "hf", "path": "medmnist", "name": "chestmnist", "split": "test"},
        ],
        "PathMNIST": [
            {"kind": "imagefolder", "path": DATA_ROOT / "pathmnist_imagefolder", "split": "train"},
            {"kind": "hf", "path": "medmnist", "name": "pathmnist", "split": "test"},
        ],
    }

    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                args.device = "cpu"
        except Exception:
            args.device = "cpu"

    print(f"Loading backbone: {args.model} on {args.device}")
    backbone = get_backbone(args.model, device=args.device).load()

    if args.datasets is not None:
        keep = set(args.datasets)
        dataset_specs = {k: v for k, v in dataset_specs.items() if k in keep}

    rows = []
    for display_name, candidates in dataset_specs.items():
        print(f"\n=== {display_name} ===")
        row = {
            "dataset": display_name,
            "status": "failed",
            "source": "",
            "split": "",
            "n_samples": "",
            "n_classes": "",
            "top1_acc": "",
            "error": "",
        }
        try:
            ds, chosen = try_load_dataset(candidates)
            if args.max_samples is not None:
                ds = ds.select(range(min(args.max_samples, len(ds))))

            classnames, label_key = resolve_classnames(ds, chosen, imagenet_map)
            print(f"Loaded {len(ds)} samples, {len(classnames)} classes, label_key={label_key}")

            result = zero_shot_accuracy(
                backbone=backbone,
                dataset=ds,
                classnames=classnames,
                label_key=label_key,
                sae=None,
                sae_mode="none",
                layer=-1,
                batch_size=args.batch_size,
                device=args.device,
                num_workers=args.num_workers,
            )

            src = chosen.get("kind")
            if src == "registry":
                src = f"registry:{chosen['key']}"
            elif src == "imagefolder":
                src = f"imagefolder:{chosen['path']}"
            else:
                src = f"hf:{chosen.get('path')}"

            row.update({
                "status": "ok",
                "source": src,
                "split": chosen.get("split", ""),
                "n_samples": len(ds),
                "n_classes": len(classnames),
                "top1_acc": f"{result['accuracy']:.4f}",
            })
            print(f"top1 = {result['accuracy']:.4f}%")
        except Exception as e:
            row["error"] = str(e).replace("\n", " ")
            print(f"FAILED: {row['error']}")
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))

    print(f"\nWrote:\n  {out_csv}\n  {out_json}")


if __name__ == "__main__":
    main()

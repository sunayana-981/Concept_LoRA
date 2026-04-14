#!/usr/bin/env python3
"""
Run monosemanticity scoring for all LoRA SAE checkpoints.

This script reuses the stable LoRA loading + dataset pipeline from
`eval_lora_sae_accuracy.py`, then computes monosemanticity metrics for each SAE:

  - dataset-wide average monosemanticity
  - top-10 neuron average
  - per-class top-10 average

By default, it evaluates all discovered final SAEs under:
  out/checkpoints/{dataset}/*/final*/*.pt
for datasets in eval_lora_sae_accuracy.DATASET_CFG.

Usage (run from patchsae/):
  python full_pipeline.py
  python full_pipeline.py --datasets eurosat caltech101
  python full_pipeline.py --include_base --layer -2
  python full_pipeline.py --max_images 5000 --save_dir out/monosem_all
"""

import argparse
import json
import math
import os
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from eval_lora_sae_accuracy import (
    BASE_SAE_PATH,
    CKPT_ROOT,
    DATASET_CFG,
    DEVICE,
    build_lora_clip,
    discover_final_saes,
    flush,
    load_test_dataset,
)
from integrate_mono import (
    dataset_monosemanticity,
    per_class_monosemanticity,
    topk_neurons_overall,
)
from tasks.utils import load_sae

DATA_ROOT = "/home/sunayana/Documents/Concept_LoRA/data"
LORA_ROOT = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"

# Extend the base eval config with additional datasets.
EXTENDED_DATASET_CFG = dict(DATASET_CFG)
EXTENDED_DATASET_CFG.update(
    {
        "dtd": {
            "data_dir": f"{DATA_ROOT}/DTD",
            "lora_path": f"{LORA_ROOT}/dtd/16shots/seed42/lora_weights.pt",
            "loader_type": "split_json",
            "split_json": f"{DATA_ROOT}/DTD/split_zhou_DescribableTextures.json",
            "image_root": f"{DATA_ROOT}/DTD/images",
            "ckpt_dirs": ["dtd"],
        },
        "ucf101": {
            "data_dir": f"{DATA_ROOT}/UCF101",
            "lora_path": f"{LORA_ROOT}/ucf101/16shots/seed1/lora_weights.pt",
            "loader_type": "split_json",
            "split_json": f"{DATA_ROOT}/UCF101/split_zhou_UCF101.json",
            "image_root": f"{DATA_ROOT}/UCF101/UCF-101-midframes",
            "ckpt_dirs": ["ucf101"],
        },
        "fgvc": {
            "data_dir": f"{DATA_ROOT}/fgvc_aircraft",
            "lora_path": f"{LORA_ROOT}/fgvc/16shots/seed1/lora_weights.pt",
            "loader_type": "fgvc_txt",
            "variants_file": f"{DATA_ROOT}/fgvc_aircraft/variants.txt",
            "split_file": f"{DATA_ROOT}/fgvc_aircraft/images_variant_test.txt",
            "image_root": f"{DATA_ROOT}/fgvc_aircraft/images",
            "ckpt_dirs": ["fgvc"],
        },
    }
)


def _safe_tag(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _cfg_block_layer(cfg) -> int | None:
    """Read block_layer from either object-style or dict-style configs."""
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get("block_layer")
    return getattr(cfg, "block_layer", None)


def _dataset_ckpt_dirs(dataset_name: str, cfg_dict: dict) -> List[str]:
    """
    Return checkpoint subdirectories to scan for a dataset.

    For historical runs, some datasets may have checkpoints under alternate names.
    """
    if "ckpt_dirs" in cfg_dict:
        return list(cfg_dict["ckpt_dirs"])

    if dataset_name == "medmnist":
        return ["medmnist", "masked_finetune", "masked_finetune_lora"]

    return [dataset_name]


def _all_configured_ckpt_dirs() -> List[str]:
    dirs = set()
    for dataset_name, cfg in EXTENDED_DATASET_CFG.items():
        for d in _dataset_ckpt_dirs(dataset_name, cfg):
            dirs.add(d)
    return sorted(dirs)


class SplitJsonImageDataset(torch.utils.data.Dataset):
    """
    Dataset backed by split_zhou_*.json files.
    Expected row format: [relative_path, label, classname].
    """

    def __init__(self, split_json: str, image_root: str, transform, split: str = "test"):
        with open(split_json, "r") as f:
            payload = json.load(f)
        rows = payload.get(split, [])
        if not rows:
            raise ValueError(f"No '{split}' entries in {split_json}")

        self.image_root = image_root
        self.transform = transform
        self.samples = []
        idx_to_name = {}

        for rel_path, label, classname in rows:
            lbl = int(label)
            idx_to_name[lbl] = classname
            self.samples.append((rel_path, lbl))

        self.class_names = [idx_to_name[i] for i in sorted(idx_to_name.keys())]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        impath = os.path.join(self.image_root, rel_path)
        img = Image.open(impath).convert("RGB")
        return self.transform(img), label


class FGVCTestDataset(torch.utils.data.Dataset):
    """
    FGVC-Aircraft test split from images_variant_test.txt.
    """

    def __init__(self, variants_file: str, split_file: str, image_root: str, transform):
        with open(variants_file, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
        cname_to_idx = {name: i for i, name in enumerate(class_names)}

        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ")
                image_id = parts[0]
                cname = " ".join(parts[1:])
                label = cname_to_idx[cname]
                impath = os.path.join(image_root, f"{image_id}.jpg")
                self.samples.append((impath, label))

        self.transform = transform
        self.class_names = class_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        impath, label = self.samples[idx]
        img = Image.open(impath).convert("RGB")
        return self.transform(img), label


def load_eval_dataset(dataset_name: str, cfg_dict: dict, preprocess):
    loader_type = cfg_dict.get("loader_type", "default")

    if loader_type == "split_json":
        ds = SplitJsonImageDataset(
            split_json=cfg_dict["split_json"],
            image_root=cfg_dict["image_root"],
            transform=preprocess,
            split="test",
        )
        return ds, ds.class_names

    if loader_type == "fgvc_txt":
        ds = FGVCTestDataset(
            variants_file=cfg_dict["variants_file"],
            split_file=cfg_dict["split_file"],
            image_root=cfg_dict["image_root"],
            transform=preprocess,
        )
        return ds, ds.class_names

    # Reuse original loaders for eurosat/caltech101/medmnist.
    return load_test_dataset(cfg_dict, preprocess)


@torch.no_grad()
def extract_embeddings_and_sae_acts(
    model,
    sae,
    sae_cfg,
    loader: DataLoader,
    device: str,
):
    """
    Extract OpenAI CLIP image embeddings and SAE hidden activations.

    Returns:
      - image_embeddings: [N, d_clip]
      - sae_activations: [N, d_sae] (CLS token)
      - labels: [N]
    """
    all_emb = []
    all_act = []
    all_lab = []

    captured = {}
    num_blocks = len(model.visual.transformer.resblocks)
    layer = _cfg_block_layer(sae_cfg)
    if layer is None:
        raise ValueError("SAE config is missing block_layer metadata.")
    if layer < 0:
        layer = num_blocks + layer
    block = model.visual.transformer.resblocks[layer]

    def hook_fn(module, inp, output):
        # OpenAI CLIP shape: [seq_len, batch, d_model] -> [batch, seq, d_model]
        captured["act"] = output.transpose(0, 1).float()

    handle = block.register_forward_hook(hook_fn)

    for imgs, labels in tqdm(loader, desc="    extract", leave=False):
        captured.clear()
        imgs = imgs.to(device)

        # CLIP image embeddings
        img_feat = model.encode_image(imgs).float()

        # Cached activations at SAE layer
        vit_features = captured["act"]

        # SAE activations (hook_hidden_post)
        _, sae_cache = sae.run_with_cache(vit_features)
        sae_acts = sae_cache["hook_hidden_post"]
        if sae_acts.dim() == 3:
            sae_acts = sae_acts[:, 0, :]  # CLS token

        all_emb.append(img_feat.cpu())
        all_act.append(sae_acts.cpu())
        all_lab.append(labels.cpu())

    handle.remove()

    return (
        torch.cat(all_emb, dim=0),
        torch.cat(all_act, dim=0),
        torch.cat(all_lab, dim=0),
    )


def analyze_monosemanticity(
    image_embeddings: torch.Tensor,
    neuron_activations: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    topk: int = 10,
):
    """Compute monosemanticity summary and table rows."""
    ms_scores, overall_avg = dataset_monosemanticity(image_embeddings, neuron_activations)

    k = min(topk, ms_scores.numel())
    topk_idx, topk_vals = topk_neurons_overall(ms_scores, k=k)
    top10_avg = topk_vals.mean().item() if k > 0 else float("nan")

    top10_rows = []
    for rank, (nid, score) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()), start=1):
        top10_rows.append(
            {
                "rank": rank,
                "neuron_id": int(nid),
                "monosemantic_score": float(score),
            }
        )

    per_class = per_class_monosemanticity(
        image_embeddings,
        neuron_activations,
        labels,
        num_classes,
        k=topk,
    )

    per_class_rows = []
    class_avgs = []
    for class_id, neuron_scores in per_class.items():
        if not neuron_scores:
            continue
        class_avg = float(sum(score for _, score in neuron_scores[:topk]) / len(neuron_scores[:topk]))
        class_avgs.append(class_avg)
        for rank, (nid, score) in enumerate(neuron_scores[:topk], start=1):
            per_class_rows.append(
                {
                    "class_id": int(class_id),
                    "rank": rank,
                    "neuron_id": int(nid),
                    "monosemantic_score": float(score),
                    "class_avg": class_avg,
                }
            )

    class_avg = float(sum(class_avgs) / len(class_avgs)) if class_avgs else float("nan")

    return {
        "overall_avg": float(overall_avg),
        "top10_avg": float(top10_avg),
        "class_avg": class_avg,
        "top10_rows": top10_rows,
        "per_class_rows": per_class_rows,
        "n_valid_classes": len(class_avgs),
    }


def evaluate_dataset(
    dataset_name: str,
    cfg_dict: dict,
    batch_size: int,
    num_workers: int,
    include_base: bool,
    layer_filter: int | None,
    max_images: int | None,
    save_dir: str,
    device: str,
):
    """Evaluate all SAEs for one dataset and return summary rows."""
    print(f"\n{'═' * 72}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'═' * 72}")

    lora_path = cfg_dict["lora_path"]
    if not os.path.isfile(lora_path):
        print(f"  [SKIP] LoRA weights not found: {lora_path}")
        return []

    print(f"\n[1] Loading LoRA CLIP\n    {lora_path}")
    model, preprocess = build_lora_clip(lora_path, device)
    model.eval()

    print(f"\n[2] Loading dataset...")
    ds, class_names = load_eval_dataset(dataset_name, cfg_dict, preprocess)

    if max_images is not None and len(ds) > max_images:
        print(f"  Limiting images: {max_images}/{len(ds)}")
        ds = Subset(ds, range(max_images))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    print(f"  {len(ds)} images | {len(class_names)} classes")

    print(f"\n[3] Discovering SAEs...")
    sae_entries = []

    if include_base and os.path.isfile(BASE_SAE_PATH):
        base_sae, base_cfg = load_sae(BASE_SAE_PATH, device)
        base_layer = _cfg_block_layer(base_cfg)
        del base_sae
        flush()
        if base_layer is None:
            print(f"  [WARN] Base SAE has no block_layer metadata: {BASE_SAE_PATH}")
        elif layer_filter is None or base_layer == layer_filter:
            sae_entries.append(
                {
                    "run_id": "imagenet_base",
                    "layer": base_layer,
                    "path": BASE_SAE_PATH,
                    "source": "base",
                }
            )

    seen_paths = set()
    ckpt_dirs = _dataset_ckpt_dirs(dataset_name, cfg_dict)
    print(f"  Scanning checkpoint dirs: {ckpt_dirs}")
    for ckpt_subdir in ckpt_dirs:
        ds_sae_dir = os.path.join(CKPT_ROOT, ckpt_subdir)
        ds_saes = discover_final_saes(ds_sae_dir)
        if ds_saes:
            print(f"  Found {len(ds_saes)} SAE(s) in {ds_sae_dir}")
        for run_id, layer, path in ds_saes:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if layer is None:
                print(f"  [SKIP] SAE has no block_layer metadata: {path}")
                continue
            if layer_filter is not None and layer != layer_filter:
                continue
            sae_entries.append(
                {
                    "run_id": run_id,
                    "layer": layer,
                    "path": path,
                    "source": "dataset",
                }
            )

    if not sae_entries:
        print("  [SKIP] No SAE checkpoints to evaluate.")
        del model
        flush()
        return []

    print(f"  Found {len(sae_entries)} SAE(s):")
    for e in sae_entries:
        print(f"    source={e['source']:<8s} run={e['run_id']:<16s} layer={e['layer']:<3} {e['path']}")

    summary_rows = []

    for i, entry in enumerate(sae_entries, start=1):
        tag = f"{dataset_name}__{entry['source']}__{entry['run_id']}__layer{entry['layer']}"
        safe_tag = _safe_tag(tag)

        print(f"\n[4.{i}] SAE={entry['run_id']} layer={entry['layer']} ({entry['source']})")

        try:
            sae, sae_cfg = load_sae(entry["path"], device)
            sae.eval()
        except Exception as e:
            print(f"    [ERROR] SAE load failed: {e}")
            continue

        try:
            emb, acts, labels = extract_embeddings_and_sae_acts(
                model, sae, sae_cfg, loader, device
            )
            metrics = analyze_monosemanticity(
                image_embeddings=emb,
                neuron_activations=acts,
                labels=labels,
                num_classes=len(class_names),
                topk=10,
            )

            print(
                f"    overall={metrics['overall_avg']:.4f} "
                f"top10={metrics['top10_avg']:.4f} "
                f"class_avg={metrics['class_avg']:.4f}"
            )

            top10_df = pd.DataFrame(metrics["top10_rows"])
            if not top10_df.empty:
                top10_df.insert(0, "dataset", dataset_name)
                top10_df.insert(1, "sae_run_id", entry["run_id"])
                top10_df.insert(2, "sae_layer", entry["layer"])
                top10_df.insert(3, "sae_source", entry["source"])
            top10_path = os.path.join(save_dir, f"{safe_tag}_top10.csv")
            top10_df.to_csv(top10_path, index=False)

            per_class_df = pd.DataFrame(metrics["per_class_rows"])
            if not per_class_df.empty:
                per_class_df.insert(0, "dataset", dataset_name)
                per_class_df.insert(1, "sae_run_id", entry["run_id"])
                per_class_df.insert(2, "sae_layer", entry["layer"])
                per_class_df.insert(3, "sae_source", entry["source"])
            per_class_path = os.path.join(save_dir, f"{safe_tag}_perclass.csv")
            per_class_df.to_csv(per_class_path, index=False)

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "num_images": int(emb.shape[0]),
                    "num_classes": int(len(class_names)),
                    "sae_source": entry["source"],
                    "sae_run_id": entry["run_id"],
                    "sae_layer": int(entry["layer"]),
                    "sae_path": entry["path"],
                    "overall_avg": metrics["overall_avg"],
                    "top10_avg": metrics["top10_avg"],
                    "class_avg": metrics["class_avg"],
                    "n_valid_classes": int(metrics["n_valid_classes"]),
                    "top10_csv": top10_path,
                    "per_class_csv": per_class_path,
                }
            )

            del emb, acts, labels
            flush()
        except Exception as e:
            print(f"    [ERROR] extraction/analysis failed: {e}")
        finally:
            del sae
            flush()

    del model
    flush()
    return summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="Run monosemanticity for all discovered SAEs"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(EXTENDED_DATASET_CFG.keys()),
        choices=list(EXTENDED_DATASET_CFG.keys()),
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--include_base",
        action="store_true",
        help="Deprecated compatibility flag. Base SAE is evaluated by default.",
    )
    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="Skip base SAE evaluation.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Only evaluate SAEs at this block_layer (e.g. -2)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional cap on number of images per dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out/monosemanticity_all_saes",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'═' * 72}")
    print("  MONOSEMANTICITY PIPELINE (ALL SAES)")
    print(f"  Device      : {DEVICE}")
    print(f"  Datasets    : {args.datasets}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Num workers : {args.num_workers}")
    print(f"  Include base: {not args.skip_base}")
    print(f"  Layer filter: {args.layer}")
    print(f"  Max images  : {args.max_images}")
    print(f"  Save dir    : {args.save_dir}")
    print(f"{'═' * 72}")

    configured_ckpt_dirs = set(_all_configured_ckpt_dirs())
    found_ckpt_dirs = set()
    if os.path.isdir(CKPT_ROOT):
        for name in os.listdir(CKPT_ROOT):
            full = os.path.join(CKPT_ROOT, name)
            if os.path.isdir(full):
                found_ckpt_dirs.add(name)
    unsupported_dirs = sorted(found_ckpt_dirs - configured_ckpt_dirs)
    if unsupported_dirs:
        print(
            f"  [INFO] checkpoint dirs not mapped in this LoRA pipeline: "
            f"{unsupported_dirs}"
        )
        print("         (likely maple/other training variants)")

    all_rows: List[Dict] = []

    for dataset_name in args.datasets:
        rows = evaluate_dataset(
            dataset_name=dataset_name,
            cfg_dict=EXTENDED_DATASET_CFG[dataset_name],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            include_base=not args.skip_base,
            layer_filter=args.layer,
            max_images=args.max_images,
            save_dir=args.save_dir,
            device=DEVICE,
        )
        all_rows.extend(rows)

    summary_path = os.path.join(args.save_dir, "monosemanticity_summary.csv")
    summary_json = os.path.join(args.save_dir, "monosemanticity_summary.json")

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        summary_df.to_csv(summary_path, index=False)
        with open(summary_json, "w") as f:
            json.dump(all_rows, f, indent=2)

        print(f"\n{'─' * 72}")
        print("SUMMARY")
        print(f"{'─' * 72}")
        for r in all_rows:
            print(
                f"{r['dataset']:<10s} run={r['sae_run_id']:<16s} "
                f"layer={r['sae_layer']:<3d} "
                f"overall={r['overall_avg']:.4f} "
                f"top10={r['top10_avg']:.4f} "
                f"class={r['class_avg']:.4f}"
            )
        print(f"\nSaved summary CSV : {summary_path}")
        print(f"Saved summary JSON: {summary_json}")
    else:
        print("\nNo results were produced.")


if __name__ == "__main__":
    main()

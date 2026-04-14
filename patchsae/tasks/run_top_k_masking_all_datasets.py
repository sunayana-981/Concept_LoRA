#!/usr/bin/env python3
"""
Batch runner for top-k masking classification across datasets and SAE variants.

For each dataset, this script can run:
1) base SAE
2) dataset-specific SAEs discovered under out/checkpoints/<dataset>/*/final*/*.pt

For each (dataset, SAE), it executes:
- tasks.compute_class_wise_sae_activation.main(...)
- tasks.classification_with_top_k_masking.main(...)
"""

import argparse
import csv
import glob
import json
import os
import sys
import traceback
import gc
from pathlib import Path
from typing import Dict, List

# Ensure project root is on sys.path so `tasks` and `src` imports resolve
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tasks.classification_with_top_k_masking import main as classify_topk_main
from tasks.compute_class_wise_sae_activation import main as compute_cls_sae_main
from tasks.utils import DATASET_INFO, setup_save_directory

import torch


DEFAULT_EXCLUDED_DATASETS = {"imagenet", "imagenet-sketch"}


def discover_dataset_saes(ckpt_root: str, dataset_name: str) -> List[str]:
    pattern = os.path.join(ckpt_root, dataset_name, "*/final*/*.pt")
    return sorted(glob.glob(pattern))


def select_dataset_saes(paths: List[str], mode: str) -> List[str]:
    if mode == "none" or not paths:
        return []
    if mode == "latest":
        return [max(paths, key=lambda p: (os.path.getmtime(p), p))]
    return paths


def get_cls_activation_path(
    root_dir: str, save_name: str, sae_path: str, vit_type: str, dataset_name: str
) -> str:
    out_dir = setup_save_directory(root_dir, save_name, sae_path, vit_type, dataset_name)
    return os.path.join(out_dir, "cls_sae_cnt.npy")


def get_metrics_path(
    root_dir: str,
    save_name: str,
    sae_path: str,
    vit_type: str,
    dataset_name: str,
    cls_activation_path: str,
) -> str:
    class_feature_type = cls_activation_path.split("/")[-3]
    out_dir = setup_save_directory(
        root_dir, save_name, sae_path, f"{class_feature_type}_{vit_type}", dataset_name
    )
    return os.path.join(out_dir, "metrics.csv")


def sae_tag_from_path(sae_source: str, sae_path: str) -> str:
    p = Path(sae_path).resolve()
    if sae_source == "base":
        return "base"
    run_id = p.parts[-3] if len(p.parts) >= 3 else p.stem
    return f"{sae_source}_{run_id}"


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def run_compute_with_retry(dataset_name: str, sae_path: str, save_name: str, args) -> int:
    batch_size = args.compute_batch_size
    while True:
        try:
            compute_cls_sae_main(
                sae_path=sae_path,
                vit_type=args.vit_type,
                device=args.device,
                dataset_name=dataset_name,
                root_dir=args.root_dir,
                save_name=save_name,
                backbone=args.backbone,
                batch_size=batch_size,
                model_path=args.model_path,
                config_path=args.config_path,
                threshold=args.threshold,
            )
            return batch_size
        except Exception as e:
            if is_oom_error(e) and batch_size > args.min_batch_size:
                next_bs = max(args.min_batch_size, batch_size // 2)
                if next_bs == batch_size:
                    raise
                print(
                    f"      [OOM] compute_classwise at batch_size={batch_size}; "
                    f"retrying with {next_bs}"
                )
                batch_size = next_bs
                cleanup_cuda()
                continue
            raise


def run_classify_with_retry(
    dataset_name: str, sae_path: str, cls_activation_path: str, save_name: str, args
) -> int:
    batch_size = args.classify_batch_size
    while True:
        try:
            classify_topk_main(
                sae_path=sae_path,
                vit_type=args.vit_type,
                device=args.device,
                dataset_name=dataset_name,
                root_dir=args.root_dir,
                save_name=save_name,
                backbone=args.backbone,
                batch_size=batch_size,
                model_path=args.model_path,
                config_path=args.config_path,
                cls_wise_sae_activation_path=cls_activation_path,
            )
            return batch_size
        except Exception as e:
            if is_oom_error(e) and batch_size > args.min_batch_size:
                next_bs = max(args.min_batch_size, batch_size // 2)
                if next_bs == batch_size:
                    raise
                print(
                    f"      [OOM] classification at batch_size={batch_size}; "
                    f"retrying with {next_bs}"
                )
                batch_size = next_bs
                cleanup_cuda()
                continue
            raise


def run_one_pair(
    dataset_name: str,
    sae_source: str,
    sae_path: str,
    args,
) -> Dict:
    row = {
        "dataset": dataset_name,
        "sae_source": sae_source,
        "sae_path": sae_path,
        "save_name": "",
        "status": "ok",
        "cls_activation_path": "",
        "metrics_path": "",
        "compute_batch_size": "",
        "classify_batch_size": "",
        "error": "",
    }

    if not os.path.isfile(sae_path):
        row["status"] = "skipped_missing_sae"
        row["error"] = f"SAE path not found: {sae_path}"
        return row

    run_save_name = os.path.join(args.save_name, sae_tag_from_path(sae_source, sae_path))
    row["save_name"] = run_save_name

    try:
        used_compute_bs = run_compute_with_retry(
            dataset_name, sae_path, run_save_name, args
        )
        row["compute_batch_size"] = used_compute_bs
    except Exception:
        row["status"] = "failed_compute_classwise"
        row["error"] = traceback.format_exc(limit=1).strip()
        cleanup_cuda()
        return row

    cls_activation_path = get_cls_activation_path(
        root_dir=args.root_dir,
        save_name=run_save_name,
        sae_path=sae_path,
        vit_type=args.vit_type,
        dataset_name=dataset_name,
    )
    row["cls_activation_path"] = cls_activation_path

    if not os.path.isfile(cls_activation_path):
        row["status"] = "failed_missing_classwise_file"
        row["error"] = f"Expected file not found: {cls_activation_path}"
        cleanup_cuda()
        return row

    try:
        used_cls_bs = run_classify_with_retry(
            dataset_name, sae_path, cls_activation_path, run_save_name, args
        )
        row["classify_batch_size"] = used_cls_bs
    except Exception:
        row["status"] = "failed_classification"
        row["error"] = traceback.format_exc(limit=1).strip()
        cleanup_cuda()
        return row

    row["metrics_path"] = get_metrics_path(
        root_dir=args.root_dir,
        save_name=run_save_name,
        sae_path=sae_path,
        vit_type=args.vit_type,
        dataset_name=dataset_name,
        cls_activation_path=cls_activation_path,
    )
    cleanup_cuda()
    return row


def save_summary(rows: List[Dict], summary_dir: str) -> None:
    os.makedirs(summary_dir, exist_ok=True)
    json_path = os.path.join(summary_dir, "topk_masking_batch_summary.json")
    csv_path = os.path.join(summary_dir, "topk_masking_batch_summary.csv")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "dataset",
        "sae_source",
        "sae_path",
        "save_name",
        "status",
        "cls_activation_path",
        "metrics_path",
        "compute_batch_size",
        "classify_batch_size",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSummary JSON: {json_path}")
    print(f"Summary CSV : {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run class-wise activation + top-k masking for base and dataset SAEs."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names. Default: DATASET_INFO excluding huge defaults.",
    )
    parser.add_argument(
        "--dataset_sae_mode",
        type=str,
        default="all",
        choices=["all", "latest", "none"],
        help="How many dataset SAEs to run per dataset.",
    )
    parser.add_argument("--skip_base", action="store_true", help="Skip base SAE runs.")
    parser.add_argument(
        "--base_sae_path",
        type=str,
        default="data/sae_weight/base/out.pt",
        help="Base SAE checkpoint path.",
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="out/checkpoints",
        help="Root folder for dataset SAE checkpoints.",
    )
    parser.add_argument("--root_dir", type=str, default=".", help="Project root directory.")
    parser.add_argument("--save_name", type=str, default="out/feature_data")
    parser.add_argument(
        "--summary_dir",
        type=str,
        default="out/feature_data",
        help="Where to write batch summary CSV/JSON.",
    )
    parser.add_argument("--vit_type", type=str, default="base", choices=["base", "maple"])
    parser.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument(
        "--compute_batch_size",
        type=int,
        default=64,
        help="Initial batch size for class-wise activation computation.",
    )
    parser.add_argument(
        "--classify_batch_size",
        type=int,
        default=64,
        help="Initial batch size for top-k masking classification.",
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=4,
        help="Minimum batch size during OOM auto-retry.",
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument(
        "--include_heavy_datasets",
        action="store_true",
        help="Include heavy defaults such as imagenet/imagenet-sketch when --datasets is not set.",
    )
    args = parser.parse_args()

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(DATASET_INFO.keys())
        if not args.include_heavy_datasets:
            datasets = [d for d in datasets if d not in DEFAULT_EXCLUDED_DATASETS]

    print("\n" + "=" * 72)
    print("Batch Top-k Masking Runner")
    print(f"Datasets         : {datasets}")
    print(f"Vit type         : {args.vit_type}")
    print(f"Include base SAE : {not args.skip_base}")
    print(f"Dataset SAE mode : {args.dataset_sae_mode}")
    print(f"Checkpoint root  : {args.ckpt_root}")
    print(f"Compute batch    : {args.compute_batch_size}")
    print(f"Classify batch   : {args.classify_batch_size}")
    print(f"Min batch (OOM)  : {args.min_batch_size}")
    print("=" * 72)

    rows: List[Dict] = []

    for dataset_name in datasets:
        if dataset_name not in DATASET_INFO:
            msg = f"Dataset not in DATASET_INFO: {dataset_name}"
            print(f"\n[SKIP] {msg}")
            rows.append(
                {
                    "dataset": dataset_name,
                    "sae_source": "n/a",
                    "sae_path": "",
                    "status": "skipped_unknown_dataset",
                    "cls_activation_path": "",
                    "metrics_path": "",
                    "error": msg,
                }
            )
            continue

        print(f"\n[{dataset_name}]")
        run_items = []

        if not args.skip_base:
            run_items.append(("base", args.base_sae_path))

        ds_sae_paths = select_dataset_saes(
            discover_dataset_saes(args.ckpt_root, dataset_name),
            args.dataset_sae_mode,
        )
        for p in ds_sae_paths:
            run_items.append(("dataset", p))

        if not run_items:
            print("  No SAE runs to execute.")
            rows.append(
                {
                    "dataset": dataset_name,
                    "sae_source": "n/a",
                    "sae_path": "",
                    "status": "skipped_no_sae_selected",
                    "cls_activation_path": "",
                    "metrics_path": "",
                    "error": "",
                }
            )
            continue

        print(f"  Scheduled runs: {len(run_items)}")
        for sae_source, sae_path in run_items:
            print(f"    - {sae_source:7s} {sae_path}")
            row = run_one_pair(dataset_name, sae_source, sae_path, args)
            rows.append(row)

            if row["status"] != "ok":
                print(f"      [FAIL] {row['status']}: {row['error']}")
                if args.stop_on_error:
                    save_summary(rows, args.summary_dir)
                    raise RuntimeError(
                        f"Stopping on first error for {dataset_name} with SAE {sae_path}"
                    )
            else:
                print(f"      [OK] metrics: {row['metrics_path']}")

    save_summary(rows, args.summary_dir)

    total = len(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    failed = total - ok
    print(f"\nDone. total={total}, ok={ok}, non_ok={failed}")


if __name__ == "__main__":
    main()

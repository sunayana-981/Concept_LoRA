#!/usr/bin/env python3
"""
Generate ablation table:
  Row 1 — Mahalanobis distance (ImageNet → dataset) using ZS CLIP features
  Row 2 — LoRA-FT accuracy minus Cross-SAE accuracy
  Row 3 — AdaptedSAE accuracy minus Cross-SAE accuracy

For CLIP+LoRA+OneSAE across all datasets.

Usage:
    cd /home/sunayana/Documents/Concept_LoRA/sae_vlm
    python generate_metrics_table.py [--compute_mahal] [--out_dir out/metrics_table]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Accuracy data ─────────────────────────────────────────────────────────────

STATS_JSON = os.path.join(_ROOT, "out", "all_combinations_sae_stats.json")

# Column names in the JSON exactly as stored
COL_MAP = {
    "caltech101": "Caltech101",
    "dtd":        "DTD",
    "fgvc":       "FGVC",
    "ucf101":     "UCF101",
    "eurosat":    "EuroSAT",
    "medmnist":   "MedMNIST",
}

DATASETS_ORDERED = ["caltech101", "dtd", "fgvc", "ucf101", "eurosat", "medmnist"]


def load_accuracy_table(stats_path: str) -> dict:
    """Load accuracy rows from all_combinations_sae_stats.json.

    Returns dict: method_name -> {dataset: accuracy, ...}
    """
    with open(stats_path) as f:
        data = json.load(f)
    rows = data["accuracy_table"]["rows"]
    result = {}
    for row in rows:
        method = row["Method"]
        result[method] = {ds: row.get(COL_MAP[ds]) for ds in DATASETS_ORDERED}
    return result


# ── Mahalanobis distance ──────────────────────────────────────────────────────

def compute_or_load_mahalanobis(
    datasets: list,
    out_dir: str,
    model: str = "clip_vit_b16",
    reference: str = "imagenet",
    max_ref: int = 3000,
    max_tgt: int = 1000,
    device: str = "cuda",
    force_recompute: bool = False,
) -> dict:
    """Return {dataset: mean_mahalanobis_distance}.

    Loads from cache if available; computes otherwise.
    """
    cache_path = os.path.join(out_dir, "mahalanobis_cache.json")

    if not force_recompute and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        missing = [d for d in datasets if d not in cached]
        if not missing:
            print(f"[Mahalanobis] Loaded from cache: {cache_path}")
            return {d: cached[d] for d in datasets}
        print(f"[Mahalanobis] Cache missing {missing}; recomputing all.")

    print(f"[Mahalanobis] Computing distances ({model}, ref={reference})...")

    if device == "cuda" and not torch.cuda.is_available():
        print("  [WARN] CUDA not available, using CPU.")
        device = "cpu"

    from src.models.registry import get_backbone
    from src.data.dataset_registry import load_registry
    from compute_mahalanobis import (
        collect_features, fit_gaussian,
        mahalanobis_distances, mean_to_reference_distance,
    )

    registry = load_registry()
    backbone = get_backbone(model, device=device).load()

    ref_feats = collect_features(
        backbone=backbone, dataset_name=reference, registry=registry,
        layer=-1, batch_size=64, max_samples=max_ref, seed=42,
    )
    ref_mu, ref_cov_inv = fit_gaussian(ref_feats, eps=1e-5)

    results = {}
    for ds in datasets:
        if ds not in registry:
            print(f"  [skip] {ds} not in registry")
            continue
        try:
            tgt_feats = collect_features(
                backbone=backbone, dataset_name=ds, registry=registry,
                layer=-1, batch_size=64, max_samples=max_tgt, seed=42,
            )
            dist = mean_to_reference_distance(tgt_feats, ref_mu, ref_cov_inv)
            results[ds] = round(float(dist), 4)
            print(f"  {ds:<18} {dist:.4f}")
        except Exception as e:
            print(f"  [skip] {ds}: {e}")
            results[ds] = None

    os.makedirs(out_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved cache → {cache_path}")
    return results


# ── Table formatting ──────────────────────────────────────────────────────────

def fmt(v, digits=2):
    if v is None:
        return "N/A"
    return f"{v:+.{digits}f}"


def fmt_plain(v, digits=2):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def print_table(mahal: dict, acc: dict, datasets: list):
    col_w = 13
    ds_labels = {
        "caltech101": "Caltech101",
        "dtd":        "DTD",
        "fgvc":       "FGVC",
        "ucf101":     "UCF101",
        "eurosat":    "EuroSAT",
        "medmnist":   "MedMNIST",
    }

    lora_ft   = acc.get("LoRA FT CLIP", {})
    cross_sae = acc.get("LoRA + Base SAE", {})
    ad_sae    = acc.get("LoRA + LoRA SAE", {})

    header = "Metric".ljust(38) + "".join(ds_labels[d].center(col_w) for d in datasets)
    sep    = "─" * len(header)

    print(f"\n{'═'*len(header)}")
    print("  CLIP+LoRA+OneSAE Ablation Table")
    print(f"{'═'*len(header)}\n")
    print(header)
    print(sep)

    # Row 1: Mahalanobis distance
    row1 = "Mahalanobis distance (↑ harder shift)"
    vals1 = "".join(fmt_plain(mahal.get(d), 3).center(col_w) for d in datasets)
    print(f"{row1:<38}{vals1}")

    # Row 2: LoRA-FT − Cross-SAE
    row2 = "LoRA-FT − Cross-SAE acc (pp)"
    vals2 = "".join(
        fmt((lora_ft.get(d) or 0) - (cross_sae.get(d) or 0)).center(col_w)
        for d in datasets
    )
    print(f"{row2:<38}{vals2}")

    # Row 3: AdSAE − Cross-SAE
    row3 = "AdaptedSAE − Cross-SAE acc (pp)"
    vals3 = "".join(
        fmt((ad_sae.get(d) or 0) - (cross_sae.get(d) or 0)).center(col_w)
        for d in datasets
    )
    print(f"{row3:<38}{vals3}")

    print(sep)
    print("\nNotes:")
    print("  Cross-SAE = LoRA-adapted CLIP + base ImageNet SAE (no domain-specific retraining)")
    print("  AdaptedSAE = LoRA-adapted CLIP + SAE retrained on target domain")
    print("  pp = percentage points")


def print_latex_table(mahal: dict, acc: dict, datasets: list):
    """Print a LaTeX tabular block."""
    lora_ft   = acc.get("LoRA FT CLIP", {})
    cross_sae = acc.get("LoRA + Base SAE", {})
    ad_sae    = acc.get("LoRA + LoRA SAE", {})

    ds_labels = {
        "caltech101": "Caltech101",
        "dtd":        "DTD",
        "fgvc":       "FGVC",
        "ucf101":     "UCF101",
        "eurosat":    "EuroSAT",
        "medmnist":   "MedMNIST",
    }

    cols = "l" + "c" * len(datasets)
    header_cols = " & ".join(ds_labels[d] for d in datasets)
    row1_vals = " & ".join(fmt_plain(mahal.get(d), 3) for d in datasets)
    row2_vals = " & ".join(
        fmt((lora_ft.get(d) or 0) - (cross_sae.get(d) or 0)) for d in datasets
    )
    row3_vals = " & ".join(
        fmt((ad_sae.get(d) or 0) - (cross_sae.get(d) or 0)) for d in datasets
    )

    latex = f"""\\begin{{tabular}}{{{cols}}}
\\toprule
 & {header_cols} \\\\
\\midrule
Mahalanobis distance & {row1_vals} \\\\
LoRA-FT $-$ Cross-SAE (pp) & {row2_vals} \\\\
AdaptedSAE $-$ Cross-SAE (pp) & {row3_vals} \\\\
\\bottomrule
\\end{{tabular}}"""
    print("\n" + latex)


def save_json(mahal: dict, acc: dict, datasets: list, path: str):
    lora_ft   = acc.get("LoRA FT CLIP", {})
    cross_sae = acc.get("LoRA + Base SAE", {})
    ad_sae    = acc.get("LoRA + LoRA SAE", {})

    table = {}
    for d in datasets:
        ft  = lora_ft.get(d)
        cs  = cross_sae.get(d)
        ads = ad_sae.get(d)
        table[d] = {
            "mahalanobis_distance":        mahal.get(d),
            "lora_ft_acc":                 ft,
            "cross_sae_acc":               cs,
            "adapted_sae_acc":             ads,
            "lora_ft_minus_cross_sae":     round((ft or 0) - (cs or 0), 4) if (ft and cs) else None,
            "adapted_sae_minus_cross_sae": round((ads or 0) - (cs or 0), 4) if (ads and cs) else None,
        }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved JSON → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP+LoRA+OneSAE ablation table"
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS_ORDERED,
                        help="Datasets to include (default: all 6)")
    parser.add_argument("--compute_mahal", action="store_true",
                        help="Compute Mahalanobis distances (requires GPU + data)")
    parser.add_argument("--model", default="clip_vit_b16",
                        help="Backbone for Mahalanobis computation")
    parser.add_argument("--max_ref",  type=int, default=3000)
    parser.add_argument("--max_tgt",  type=int, default=1000)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--out_dir",  default="out/metrics_table")
    parser.add_argument("--latex",    action="store_true",
                        help="Also print LaTeX tabular block")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Recompute Mahalanobis even if cache exists")
    args = parser.parse_args()

    # Load accuracy data
    if not os.path.exists(STATS_JSON):
        print(f"[ERROR] Stats JSON not found: {STATS_JSON}")
        sys.exit(1)
    acc = load_accuracy_table(STATS_JSON)
    print(f"Loaded accuracy table: {list(acc.keys())}")

    # Mahalanobis distances
    if args.compute_mahal:
        mahal = compute_or_load_mahalanobis(
            datasets=args.datasets,
            out_dir=args.out_dir,
            model=args.model,
            max_ref=args.max_ref,
            max_tgt=args.max_tgt,
            device=args.device,
            force_recompute=args.force_recompute,
        )
    else:
        cache = os.path.join(args.out_dir, "mahalanobis_cache.json")
        if os.path.exists(cache):
            with open(cache) as f:
                mahal = json.load(f)
            print(f"Loaded Mahalanobis from cache: {cache}")
        else:
            print("[INFO] No Mahalanobis cache found. Use --compute_mahal to compute.")
            mahal = {d: None for d in args.datasets}

    # Print tables
    print_table(mahal, acc, args.datasets)
    if args.latex:
        print_latex_table(mahal, acc, args.datasets)

    # Save JSON
    save_json(
        mahal, acc, args.datasets,
        os.path.join(args.out_dir, "ablation_table.json"),
    )


if __name__ == "__main__":
    main()

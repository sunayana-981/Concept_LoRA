#!/usr/bin/env python3
"""Strictly validate and aggregate the SAE initialization ablation.

The command exits nonzero if any pre-registered cell is absent, duplicated,
skipped, uses the wrong fixed evaluation/probe seed, or points at a different
checkpoint than the training manifest. Only SAE *training* seeds are averaged;
evaluation/probe resampling is held fixed and never counted as replication.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        default="out/rebuttal/sae_initialization_ablation_manifest.json",
    )
    p.add_argument("--results", required=True)
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["zeroshot_acc", "l0", "dead_frac", "recon_cosine", "fve"],
    )
    p.add_argument("--vit_type", default="lora")
    p.add_argument(
        "--require_probe_seed",
        action="store_true",
        help="Also require every result row to use the manifest's fixed probe seed.",
    )
    p.add_argument(
        "--out",
        default="out/rebuttal/sae_initialization_ablation_summary.json",
    )
    return p.parse_args()


def parse_optional_int(value):
    if value in (None, "", "None", "null", "nan", "NaN"):
        return None
    return int(float(value))


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def cell_key(record):
    return (
        record["dataset"],
        record.get("sae_condition", record.get("condition")),
        parse_optional_int(record.get("training_seed")),
    )


def mean_ci95(values):
    n = len(values)
    mean = statistics.fmean(values)
    if n == 1:
        return {"n_training_seeds": 1, "mean": mean, "std": None, "ci95": None}
    std = statistics.stdev(values)
    df = n - 1
    critical = T_CRITICAL_95.get(df, 1.96)
    half_width = critical * std / math.sqrt(n)
    return {
        "n_training_seeds": n,
        "mean": mean,
        "std": std,
        "ci95": [mean - half_width, mean + half_width],
    }


def main():
    args = parse_args()
    with Path(args.manifest).open() as handle:
        manifest = json.load(handle)
    with Path(args.results).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("[FAIL] results CSV is empty")
    condition_column = (
        "sae_condition" if "sae_condition" in rows[0] else "condition"
    )
    required_columns = {
        "dataset",
        condition_column,
        "training_seed",
        "evaluation_seed",
        *args.metrics,
    }
    if args.require_probe_seed:
        required_columns.add("probe_seed")
    missing_columns = sorted(required_columns - set(rows[0]))
    if missing_columns:
        raise SystemExit(f"[FAIL] results missing columns: {missing_columns}")

    if "vit_type" in rows[0]:
        rows = [row for row in rows if row["vit_type"] == args.vit_type]

    fixed = manifest["fixed_factors"]
    expected_eval_seed = int(fixed["evaluation_seed"])
    expected_probe_seed = int(fixed["probe_seed"])
    expected = {cell_key(cell): cell for cell in manifest["expected_cells"]}

    completed_runs = {
        cell_key(run): run
        for run in manifest.get("runs", [])
        if str(run.get("status", "")).startswith("completed")
    }
    missing_training = [
        key for key in expected
        if key[1] != "gsae" and key not in completed_runs
    ]
    if missing_training:
        raise SystemExit(
            f"[FAIL] manifest lacks completed training cells: {missing_training}"
        )
    provenance_errors = []
    if len(str(fixed.get("gsae_sha256", ""))) != 64:
        provenance_errors.append("missing/invalid G-SAE SHA-256")
    for key, run in completed_runs.items():
        if len(str(run.get("checkpoint_sha256", ""))) != 64:
            provenance_errors.append(f"{key}: missing/invalid final checkpoint SHA-256")
        if len(str(run.get("adapted_model_checkpoint_sha256", ""))) != 64:
            provenance_errors.append(f"{key}: missing/invalid adapted-model SHA-256")
        identity = run.get("target_data_identity", {})
        if not identity.get("inventory_sha256") and not identity.get("recipe_identifier"):
            provenance_errors.append(f"{key}: no target split hash or recipe identifier")
    if provenance_errors:
        raise SystemExit(
            "[FAIL] incomplete provenance:\n  " + "\n  ".join(provenance_errors)
        )

    by_key = {}
    for row in rows:
        key = (
            row["dataset"],
            row[condition_column],
            parse_optional_int(row["training_seed"]),
        )
        by_key.setdefault(key, []).append(row)

    errors = []
    validated = []
    for key in expected:
        matched = by_key.get(key, [])
        if len(matched) != 1:
            errors.append(f"{key}: expected exactly one result row, found {len(matched)}")
            continue
        row = matched[0]
        if "skipped" in row and parse_bool(row["skipped"]):
            errors.append(f"{key}: result is marked skipped")
        if parse_optional_int(row["evaluation_seed"]) != expected_eval_seed:
            errors.append(
                f"{key}: evaluation_seed={row['evaluation_seed']} "
                f"!= fixed {expected_eval_seed}"
            )
        if args.require_probe_seed and (
            parse_optional_int(row["probe_seed"]) != expected_probe_seed
        ):
            errors.append(
                f"{key}: probe_seed={row['probe_seed']} != fixed {expected_probe_seed}"
            )
        run = completed_runs.get(key)
        if run and row.get("sae_path") and (
            Path(row["sae_path"]).resolve()
            != Path(run["checkpoint_path"]).resolve()
        ):
            errors.append(
                f"{key}: evaluated {row['sae_path']}, "
                f"manifest trained {run['checkpoint_path']}"
            )
        for metric in args.metrics:
            try:
                value = float(row[metric])
            except (TypeError, ValueError):
                errors.append(f"{key}: metric {metric} is not numeric ({row[metric]!r})")
                continue
            if not math.isfinite(value):
                errors.append(f"{key}: metric {metric} is non-finite ({value})")
        validated.append(row)

    unexpected = sorted(set(by_key) - set(expected))
    if unexpected:
        errors.append(f"unexpected result cells: {unexpected}")
    if errors:
        raise SystemExit("[FAIL] incomplete/invalid ablation:\n  " + "\n  ".join(errors))

    summaries = []
    datasets = [cell["dataset"] for cell in manifest["expected_cells"]]
    conditions = [cell["condition"] for cell in manifest["expected_cells"]]
    for dataset in dict.fromkeys(datasets):
        for condition in dict.fromkeys(conditions):
            group = [
                row for row in validated
                if row["dataset"] == dataset and row[condition_column] == condition
            ]
            if not group:
                continue
            summary = {"dataset": dataset, "condition": condition}
            for metric in args.metrics:
                summary[metric] = mean_ci95([float(row[metric]) for row in group])
            summaries.append(summary)

    output = {
        "manifest": str(Path(args.manifest).resolve()),
        "results": str(Path(args.results).resolve()),
        "replicate_definition": "SAE training seed only",
        "fixed_evaluation_seed": expected_eval_seed,
        "fixed_probe_seed": expected_probe_seed,
        "n_expected_cells": len(expected),
        "summaries": summaries,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"[PASS] all {len(expected)} pre-registered cells complete and unique")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

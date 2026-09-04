#!/usr/bin/env python3
"""
Converts out/rebuttal/sae_registry.json (a flat audit log of
every SAE trained for the rebuttal — one record per (dataset, condition),
written by run_rebuttal_sae_training.sh and run_fullft_sae_training.sh) into
the nested {dataset: {condition: checkpoint_path}} shape that
tasks/eval_matrix.py's and tasks/eval_steering.py's --sae_paths expects.

Kept as a separate conversion step rather than writing the registry directly
in the nested shape so the registry stays a straightforward append/dedup log
(one record per training run, easy to audit: which layer, how many tokens,
when) independent of how any particular eval script wants to consume it.

Usage:
    python tasks/registry_to_sae_paths.py \
        --registry out/rebuttal/sae_registry.json \
        --out configs/rebuttal_sae_paths.json
"""

import argparse
import json
import sys


def normalized_condition(record):
    """Apply the controlled three-arm definitions to legacy registry rows.

    Historical ``ftsae`` rows were produced by train_sae_lora_clip.py, which
    randomly initialized the SAE. They are scratch baselines, not G-SAE
    warm-starts. Only a row with explicit checkpoint initialization is allowed
    to retain the ``ftsae`` label.
    """
    condition = record["condition"]
    if condition != "ftsae":
        return condition
    initialization = record.get("sae_initialization")
    if initialization == "checkpoint":
        return "ftsae"
    return "scratchsae"


def build_sae_paths(records, training_seed=None):
    sae_paths = {}
    sources = {}
    for r in records:
        condition = normalized_condition(r)
        record_seed = r.get("training_seed", r.get("seed"))

        if training_seed is not None:
            # Frozen G-SAE controls have no SAE-training seed. Learned arms must
            # match the requested seed exactly.
            if condition != "gsae" and record_seed != training_seed:
                continue
        elif record_seed is not None:
            # A single seeded record is unambiguous, but multiple seeds for the
            # same condition must never be collapsed silently.
            pass

        key = (r["dataset"], condition)
        if key in sources:
            previous = sources[key]
            previous_seed = previous.get("training_seed", previous.get("seed"))
            if previous_seed != record_seed or previous["checkpoint_path"] != r["checkpoint_path"]:
                raise ValueError(
                    "multiple registry rows resolve to "
                    f"{r['dataset']}/{condition}; pass --training_seed "
                    f"(seeds: {previous_seed!r}, {record_seed!r})"
                )
        sources[key] = r
        sae_paths.setdefault(r["dataset"], {})[condition] = r["checkpoint_path"]
    return sae_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=str, default="out/rebuttal/sae_registry.json")
    p.add_argument("--out", type=str, default="configs/rebuttal_sae_paths.json")
    p.add_argument(
        "--training_seed",
        type=int,
        default=None,
        help="Select one SAE-training seed when the registry has seed replicates.",
    )
    args = p.parse_args()

    with open(args.registry) as f:
        records = json.load(f)

    legacy = [
        r for r in records
        if r.get("condition") == "ftsae"
        and r.get("sae_initialization") != "checkpoint"
    ]
    for record in legacy:
        print(
            "[MIGRATE] legacy random-init "
            f"{record['dataset']}/ftsae -> scratchsae",
            file=sys.stderr,
        )

    try:
        sae_paths = build_sae_paths(records, training_seed=args.training_seed)
    except ValueError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc
    with open(args.out, "w") as f:
        json.dump(sae_paths, f, indent=2)

    seed_label = (
        f" for training seed {args.training_seed}"
        if args.training_seed is not None else ""
    )
    print(f"Wrote {args.out} from {len(records)} registry record(s){seed_label}:")
    for dataset, conditions in sae_paths.items():
        for condition, path in conditions.items():
            print(f"  {dataset:16s} {condition:10s} {path}")


if __name__ == "__main__":
    main()

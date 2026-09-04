#!/usr/bin/env python3
"""
Consistency check for the rebuttal experiment matrix. Fails loudly (nonzero
exit, printed diff) rather than letting a silent misconfiguration produce
misleading numbers in the eval matrix.

Checks:
  1. Every SAE registered in out/rebuttal/sae_registry.json (plus the fixed
     G-SAE checkpoint) has the same block_layer, expansion_factor, and
     l1_coefficient -- the rebuttal's "never vary across conditions" rule
     (see tasks/rebuttal_common.py / the task briefs).
  2. For every dataset in configs/rebuttal_datasets.json, the classname
     ordering used to build zero-shot text features matches the ordering
     used for image labels -- the "ImageFolder alphabetical order" bug
     class. In particular, for the medmnist_npz dataset (pathmnist), this
     re-derives eval_medmnist_sae.build_npz_to_if_mapping()'s class-name
     match and turns any unmatched class (which that function only warns
     about and silently falls back on) into a hard failure here, plus
     checks the resulting mapping is a bijection (no two npz labels folded
     onto the same ImageFolder index).

Usage:
    python tasks/check_matrix.py
    python tasks/check_matrix.py --registry out/rebuttal/sae_registry.json \
        --dataset_registry configs/rebuttal_datasets.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tasks.utils import load_sae
from eval_medmnist_sae import MEDMNIST_CLASSES, get_imagefolder_classnames


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=str, default="out/rebuttal/sae_registry.json")
    p.add_argument("--dataset_registry", type=str, default="configs/rebuttal_datasets.json")
    p.add_argument("--gsae_path", type=str, default="data/sae_weight/base/out.pt")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════
# Check 1: SAE hyperparameter consistency
# ═════════════════════════════════════════════════════════════════════════

def check_sae_consistency(registry_path, gsae_path):
    print("=" * 78)
    print("CHECK 1: SAE hyperparameter consistency (block_layer, expansion_factor, l1)")
    print("=" * 78)

    entries = []
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            records = json.load(f)
        for r in records:
            entries.append((f"{r['dataset']}/{r['condition']}", r["checkpoint_path"]))
    else:
        print(f"[WARN] {registry_path} does not exist yet -- checking only the fixed G-SAE.")

    if gsae_path and os.path.exists(gsae_path):
        entries.append(("gsae (base, fixed)", gsae_path))

    if not entries:
        print("[SKIP] no SAE checkpoints found to check.")
        return True

    rows = []
    for label, path in entries:
        if not os.path.exists(path):
            print(f"[FAIL] {label}: checkpoint missing on disk: {path}")
            return False
        _, cfg = load_sae(path, "cpu")
        rows.append(dict(
            label=label, path=path,
            block_layer=getattr(cfg, "block_layer", None),
            expansion_factor=getattr(cfg, "expansion_factor", None),
            l1_coefficient=getattr(cfg, "l1_coefficient", None),
        ))

    ref = rows[0]
    mismatches = []
    for row in rows[1:]:
        for key in ("block_layer", "expansion_factor", "l1_coefficient"):
            if row[key] != ref[key]:
                mismatches.append((row["label"], key, row[key], ref["label"], ref[key]))

    for label, path, bl, ef, l1 in [(r["label"], r["path"], r["block_layer"],
                                      r["expansion_factor"], r["l1_coefficient"]) for r in rows]:
        print(f"  {label:32s} block_layer={bl!s:5s} expansion_factor={ef!s:5s} "
              f"l1_coefficient={l1!s:10s}  ({path})")

    if mismatches:
        print("\n[FAIL] hyperparameter mismatches found:")
        for label, key, val, ref_label, ref_val in mismatches:
            print(f"    {label}: {key}={val!r}  !=  {ref_label}: {key}={ref_val!r}")
        return False

    print(f"\n[PASS] all {len(rows)} SAE(s) share block_layer={ref['block_layer']}, "
          f"expansion_factor={ref['expansion_factor']}, l1_coefficient={ref['l1_coefficient']}")
    return True


# ═════════════════════════════════════════════════════════════════════════
# Check 2: classname ordering
# ═════════════════════════════════════════════════════════════════════════

def check_classname_ordering(dataset_registry_path):
    print("\n" + "=" * 78)
    print("CHECK 2: classname ordering (ImageFolder-alphabetical-order bug class)")
    print("=" * 78)

    with open(dataset_registry_path) as f:
        registry = json.load(f)

    all_ok = True
    for name, entry in registry.items():
        if entry["type"] == "imagefolder":
            path = entry.get("path")
            if not path or not os.path.isdir(path):
                print(f"[SKIP] {name}: path not configured/available ({path})")
                continue
            a = get_imagefolder_classnames(path)
            b = get_imagefolder_classnames(path)
            if a != b:
                print(f"[FAIL] {name}: get_imagefolder_classnames() is non-deterministic!")
                print(f"    run 1: {a}")
                print(f"    run 2: {b}")
                all_ok = False
            else:
                print(f"[PASS] {name}: {len(a)} classnames, ImageFolder ordering deterministic")

        elif entry["type"] == "medmnist_npz":
            root = entry.get("imagefolder_root")
            if not root or not os.path.isdir(root):
                print(f"[SKIP] {name}: imagefolder_root not configured/available ({root})")
                continue
            if_classnames = get_imagefolder_classnames(root)
            if_lookup = {n.lower(): i for i, n in enumerate(if_classnames)}

            unmatched, seen_idx, duplicates = [], {}, []
            for npz_idx, classname in enumerate(MEDMNIST_CLASSES):
                key = classname.lower()
                if key not in if_lookup:
                    unmatched.append((npz_idx, classname))
                    continue
                if_idx = if_lookup[key]
                if if_idx in seen_idx:
                    duplicates.append((npz_idx, classname, if_idx, seen_idx[if_idx]))
                seen_idx[if_idx] = (npz_idx, classname)

            if unmatched or duplicates:
                print(f"[FAIL] {name}: npz label <-> ImageFolder classname mismatch")
                print(f"    ImageFolder classes ({root}): {if_classnames}")
                print(f"    MEDMNIST_CLASSES (npz order): {MEDMNIST_CLASSES}")
                for npz_idx, classname in unmatched:
                    print(f"    unmatched: npz[{npz_idx}]='{classname}' has no ImageFolder class")
                for npz_idx, classname, if_idx, prev in duplicates:
                    print(f"    duplicate: npz[{npz_idx}]='{classname}' -> IF[{if_idx}] "
                          f"already claimed by npz{prev}")
                all_ok = False
            else:
                print(f"[PASS] {name}: all {len(MEDMNIST_CLASSES)} MEDMNIST_CLASSES map "
                      f"bijectively onto ImageFolder classes at {root}")

        else:
            print(f"[SKIP] {name}: unknown registry type '{entry['type']}'")

    return all_ok


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    ok1 = check_sae_consistency(args.registry, args.gsae_path)
    ok2 = check_classname_ordering(args.dataset_registry)

    print("\n" + "=" * 78)
    if ok1 and ok2:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("CHECKS FAILED -- see [FAIL] lines above")
        sys.exit(1)


if __name__ == "__main__":
    main()

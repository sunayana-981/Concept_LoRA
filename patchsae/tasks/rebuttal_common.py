#!/usr/bin/env python3
"""
Shared dataset / model / SAE plumbing for the rebuttal eval scripts
(tasks/eval_matrix.py, tasks/eval_steering.py).

Factored out of eval_matrix.py so every rebuttal driver sweeps the same
{dataset} x {vit_type} x {sae_condition} matrix through identical loading
code instead of re-implementing it.
"""

import gc
import json
import os
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets as tv_datasets

from tasks.utils import load_sae, load_hooked_vit
from tasks.train_sae_lora_clip import load_lora_weights
from eval_medmnist_sae import (
    NpzSplitDataset,
    get_imagefolder_classnames,
    build_npz_to_if_mapping,
    _hf_collate_fn,
    get_processor_transform,
)

VIT_TYPES = ["base", "lora"]
SAE_CONDITIONS = [
    "none",
    "gsae",
    "ftsae",
    "scratchsae",
    "fullftsae",
    "masked",
    "gsae_gated",
    "masked_gated",
]
EXPECTED_BLOCK_LAYER = -2
EXPECTED_D_IN = 768

DEFAULT_DATASET_REGISTRY = os.path.join(_project_root, "configs", "rebuttal_datasets.json")
DEFAULT_GSAE_PATH = os.path.join(_project_root, "data", "sae_weight", "base", "out.pt")


def add_common_args(p, out_dir_default):
    """Adds the CLI flags shared by every rebuttal driver. Callers add their
    own extra flags (e.g. --k_values, --sae_conditions choices) on top."""
    p.add_argument("--lora_checkpoints", type=str, default=None,
                    help="JSON mapping {dataset: lora_checkpoint_path}. "
                         "Overrides the registry's per-dataset default.")
    p.add_argument("--sae_paths", type=str, default=None,
                    help="JSON mapping {dataset: {condition: sae_checkpoint_path}}.")
    p.add_argument("--dataset_registry", type=str, default=DEFAULT_DATASET_REGISTRY)
    p.add_argument("--imagenet_subset_dir", type=str, default=None,
                    help="ImageFolder root for the imagenet_subset dataset.")
    p.add_argument("--gsae_path", type=str, default=DEFAULT_GSAE_PATH)
    p.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch16")
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_images", type=int, default=10000,
                    help="Cap on total images per dataset before the train/eval split.")
    p.add_argument("--val_split", type=float, default=0.15,
                    help="Fraction of images held out as the eval split "
                         "(the rest is the train split used for activation profiles).")
    p.add_argument("--out_dir", type=str, default=out_dir_default)
    p.add_argument("--cache_dir", type=str, default="out/rebuttal/cache")
    p.add_argument("--reuse_cache", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════
# Dataset registry + loading
# ═════════════════════════════════════════════════════════════════════════

def load_registry(path):
    with open(path) as f:
        return json.load(f)


def _load_wnid_name_list(map_path):
    """Parses 'configs/classnames/*_classnames.txt' lines of the form
    'n01440764 tench' -> returns (ordered_names, wnid_to_name). Line order is
    the canonical 0..999 ImageNet class index order (verified against
    torchvision's standard ordering)."""
    ordered_names = []
    wnid_to_name = {}
    with open(map_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid, name = line.split(" ", 1)
            ordered_names.append(name)
            wnid_to_name[wnid] = name
    return ordered_names, wnid_to_name


def _mapped_classnames(imagefolder_path, classname_map_path, mode):
    """Builds a classnames list ordered to match torchvision ImageFolder's own
    class_to_idx (whatever sort order that ends up being -- notably numeric
    folder names like '0'..'999' sort LEXICOGRAPHICALLY, not numerically, so
    this must go through class_to_idx explicitly rather than assuming folder
    order == numeric order).

    mode='wnid': folder name is itself a WordNet synset id (e.g. 'n01440764'),
      looked up directly in the wnid->name dict.
    mode='numeric_index': folder name is a decimal string ImageNet class index
      (e.g. '0', '17', '999'), used to index into the canonical ordered name
      list read from the same file (line order == class index order).
    """
    ds = tv_datasets.ImageFolder(root=imagefolder_path)
    idx_to_folder = {i: n for n, i in ds.class_to_idx.items()}
    ordered_names, wnid_to_name = _load_wnid_name_list(classname_map_path)

    classnames = []
    for i in range(len(idx_to_folder)):
        folder = idx_to_folder[i]
        if mode == "wnid":
            if folder not in wnid_to_name:
                raise KeyError(f"wnid '{folder}' not found in {classname_map_path}")
            classnames.append(wnid_to_name[folder])
        elif mode == "numeric_index":
            classnames.append(ordered_names[int(folder)])
        else:
            raise ValueError(f"Unknown classname_map_mode: {mode}")
    return classnames


def build_dataset_splits(name, entry, args):
    """Returns (classnames, train_dataset, eval_dataset). Datasets yield (PIL, int)."""
    seed_gen = torch.Generator().manual_seed(args.seed)

    if entry["type"] == "medmnist_npz":
        npz_path = entry["npz_path"]
        imagefolder_root = entry["imagefolder_root"]
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"npz not found: {npz_path}")
        if not os.path.isdir(imagefolder_root):
            raise FileNotFoundError(f"imagefolder_root not found: {imagefolder_root}")
        classnames = get_imagefolder_classnames(imagefolder_root)
        mapping = build_npz_to_if_mapping(imagefolder_root)
        transform = get_processor_transform()
        train_ds = NpzSplitDataset(npz_path, mapping, transform, split="train", return_pil=True)
        eval_ds = NpzSplitDataset(npz_path, mapping, transform, split="test", return_pil=True)
        return classnames, train_ds, eval_ds

    if entry["type"] == "imagefolder":
        path = entry.get("path")
        if name == "imagenet_subset":
            path = args.imagenet_subset_dir or path
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(f"ImageFolder root not found for '{name}': {path}")

        classname_map_path = entry.get("classname_map")
        if classname_map_path:
            classnames = _mapped_classnames(
                path, classname_map_path, entry.get("classname_map_mode", "wnid")
            )
        else:
            classnames = get_imagefolder_classnames(path)
        full_ds = tv_datasets.ImageFolder(root=path)

        if len(full_ds) > args.max_images:
            idx = torch.randperm(len(full_ds), generator=seed_gen)[:args.max_images].tolist()
            full_ds = Subset(full_ds, idx)

        n_eval = max(1, int(len(full_ds) * args.val_split))
        n_train = len(full_ds) - n_eval
        train_ds, eval_ds = random_split(full_ds, [n_train, n_eval], generator=seed_gen)
        return classnames, train_ds, eval_ds

    raise ValueError(f"Unknown dataset registry type: {entry['type']}")


# ═════════════════════════════════════════════════════════════════════════
# Model / SAE loading (cached within a single process run)
# ═════════════════════════════════════════════════════════════════════════

_vit_cache = {}
_sae_cache = {}


def clear_model_caches():
    """Release cached CUDA models/SAEs between independent matrix cells."""
    _vit_cache.clear()
    _sae_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_vit(vit_type, dataset_name, lora_checkpoints, args):
    """Returns a HookedVisionTransformer, or None if the required LoRA checkpoint
    is missing (caller should skip the cell)."""
    cache_key = "base" if vit_type == "base" else f"lora:{dataset_name}"
    if cache_key in _vit_cache:
        return _vit_cache[cache_key]

    vit = load_hooked_vit(None, "base", args.backbone, args.device)
    vit.eval()

    if vit_type == "lora":
        ckpt = lora_checkpoints.get(dataset_name)
        if not ckpt or not os.path.exists(ckpt):
            return None
        load_lora_weights(vit, ckpt, args.device)
        vit.eval()

    _vit_cache[cache_key] = vit
    return vit


def resolve_sae_path(sae_condition, dataset_name, sae_paths, args):
    if sae_condition == "none":
        return None
    if sae_condition == "gsae":
        return sae_paths.get(dataset_name, {}).get("gsae", args.gsae_path)
    return sae_paths.get(dataset_name, {}).get(sae_condition)


def get_sae(sae_path, device):
    if sae_path in _sae_cache:
        return _sae_cache[sae_path]
    sae, cfg = load_sae(sae_path, device)
    if getattr(cfg, "block_layer", None) != EXPECTED_BLOCK_LAYER:
        print(f"  [WARN] SAE {sae_path} has block_layer={getattr(cfg, 'block_layer', '?')}, "
              f"expected {EXPECTED_BLOCK_LAYER}")
    if getattr(cfg, "d_in", None) != EXPECTED_D_IN:
        print(f"  [WARN] SAE {sae_path} has d_in={getattr(cfg, 'd_in', '?')}, "
              f"expected {EXPECTED_D_IN}")
    _sae_cache[sae_path] = (sae, cfg)
    return sae, cfg


def validate_sae_condition_provenance(sae_condition, cfg):
    """Prevent legacy random-init checkpoints from being reported as FT-SAE."""
    if sae_condition != "ftsae":
        return
    metadata = getattr(cfg, "experiment_metadata", None)
    if isinstance(metadata, dict):
        initialization = metadata.get("sae_initialization")
    else:
        initialization = getattr(metadata, "sae_initialization", None)
    if initialization != "checkpoint":
        raise ValueError(
            "ftsae requires explicit G-SAE checkpoint initialization metadata; "
            "this checkpoint is legacy/unknown and must be labeled scratchsae"
        )


def is_valid_cell(vit_type, sae_condition):
    if sae_condition in (
        "ftsae",
        "scratchsae",
        "fullftsae",
        "masked",
        "masked_gated",
    ) and vit_type != "lora":
        return False
    return True


def cell_cache_path(cache_dir, dataset, vit_type, sae_condition, suffix="json"):
    return os.path.join(cache_dir, f"{dataset}__{vit_type}__{sae_condition}.{suffix}")


def load_common_registry_args(args):
    """Validates --datasets against the registry and resolves lora_checkpoints /
    sae_paths overrides. Returns (registry, lora_checkpoints, sae_paths)."""
    registry = load_registry(args.dataset_registry)
    for name in args.datasets:
        if name not in registry:
            raise SystemExit(f"[FATAL] Dataset '{name}' not found in registry {args.dataset_registry}")

    lora_checkpoints = {name: registry[name].get("lora_checkpoint") for name in args.datasets}
    if args.lora_checkpoints:
        with open(args.lora_checkpoints) as f:
            lora_checkpoints.update(json.load(f))

    sae_paths = {}
    if args.sae_paths:
        with open(args.sae_paths) as f:
            sae_paths = json.load(f)

    return registry, lora_checkpoints, sae_paths

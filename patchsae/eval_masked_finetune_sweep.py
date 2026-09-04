#!/usr/bin/env python3
"""
eval_masked_finetune_sweep.py — accuracy + SAE metrics for masked-finetuned
SAEs across the full 11-dataset roster and a range of protection fractions.

For each dataset, computes (reusing the exact primitives from
eval_masked_sae.py, which validated this protocol on MedMNIST):

  Shared (computed once per dataset, independent of protection fraction):
    - lora_no_sae, base_no_sae            accuracy on the dataset's own held-out split
    - lora_base_sae, base_base_sae        accuracy with the original ImageNet G-SAE
    - imagenet_* counterparts of the above, on a fixed 5k-image ImageNet subset
    - base_sae reconstruction (own dataset + ImageNet)

  Per condition (e.g. masked_p80, masked_p90 — one masked-SAE checkpoint each):
    - lora_masked_sae, base_masked_sae accuracy (own dataset + ImageNet)
    - masked_sae reconstruction (own dataset + ImageNet)
    - feature stats (L0, dead_fraction) for protected/free/all units

Checkpoints are resolved from out/rebuttal/sae_registry.json by
(dataset, condition) — see run_masked_finetune_protection_sweep.sh, which
registers "masked_p80"/"masked_p90" there.

NOTE ON TEST SPLITS: unlike MedMNIST (which has a real held-out .npz test
split), the other 10 datasets only have a single HF "train" split available
in this repo (tasks/utils.py DATASET_INFO). This script carves out a
deterministic 20% held-out slice (seed 42) per dataset for accuracy/eval —
it is NOT the same as a curated official test split, so treat these
accuracies as approximate.

Usage:
    cd /home/sunayana/Documents/Concept_LoRA/patchsae
    python eval_masked_finetune_sweep.py
    python eval_masked_finetune_sweep.py --datasets dtd eurosat --conditions masked_p80
    python eval_masked_finetune_sweep.py --imagenet_n_samples 200  # smoke test
"""

import argparse
import gc
import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_masked_sae import (
    BACKBONE, BASE_SAE_PATH, DEVICE, flush,
    get_processor_transform, get_transform, load_sae, load_hooked_vit,
    load_masked_sae, build_lora_openai_clip, OpenAISAEHook, make_sae_hook_hf,
    compute_accuracy_openai, compute_accuracy_hf,
    get_text_features_openai, get_text_features_hf,
    analyze_sae_features, measure_reconstruction_quality,
)
from tasks.utils import DATASET_INFO, get_classnames

# ── Config ──────────────────────────────────────────────────────────────────
LORA_WEIGHTS_ROOT = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
REGISTRY_PATH = "out/rebuttal/sae_registry.json"
BATCH_SIZE = 32

LORA_PATHS = {
    "caltech101":  f"{LORA_WEIGHTS_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
    "cityscapes":  f"{LORA_WEIGHTS_ROOT}/cityscapes/16shots/seed1/lora_weights.pt",
    "cub2002011":  f"{LORA_WEIGHTS_ROOT}/cub2002011/16shots/seed1/lora_weights.pt",
    "dtd":         f"{LORA_WEIGHTS_ROOT}/dtd/16shots/seed42/lora_weights.pt",
    "eurosat":     f"{LORA_WEIGHTS_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
    "fgvc":        f"{LORA_WEIGHTS_ROOT}/fgvc/16shots/seed1/lora_weights.pt",
    "kitti":       f"{LORA_WEIGHTS_ROOT}/kitti/16shots/seed1/lora_weights.pt",
    "pathmnist":   f"{LORA_WEIGHTS_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
    "officehome":  f"{LORA_WEIGHTS_ROOT}/officehome/16shots/seed1/lora_weights.pt",
    "pets":        f"{LORA_WEIGHTS_ROOT}/oxford_pets/16shots/seed1/lora_weights.pt",
    "ucf101":      f"{LORA_WEIGHTS_ROOT}/ucf101/16shots/seed1/lora_weights.pt",
}

TRAIN_DATASET_KEY = {
    "caltech101": "caltech101", "cityscapes": "cityscapes", "cub2002011": "cub2002011",
    "dtd": "dtd", "eurosat": "eurosat", "fgvc": "fgvc", "kitti": "kitti",
    "pathmnist": "medmnist", "officehome": "officehome", "pets": "oxford_pets",
    "ucf101": "ucf101",
}

ALL_DATASETS = list(LORA_PATHS.keys())
DEFAULT_CONDITIONS = ["masked_p80", "masked_p90"]


# ── Generic (non-ImageNet) HF-dataset wrapper ───────────────────────────────
class SubsetImageDataset(Dataset):
    """Index subset of an HF imagefolder-style dataset ("image"/"label" keys)."""

    def __init__(self, hf_dataset, indices, transform=None):
        self.dataset = hf_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.dataset[self.indices[i]]
        img = item["image"]
        if not hasattr(img, "convert"):
            from PIL import Image
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


def _collate(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def resolve_checkpoint(registry, dataset, condition):
    matches = [r for r in registry if r["dataset"] == dataset and r["condition"] == condition]
    if not matches:
        return None
    return sorted(matches, key=lambda r: r["registered_at"])[-1]["checkpoint_path"]


def make_split(n, seed=42, test_frac=0.2, max_test=1000):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = min(max_test, int(n * test_frac))
    return sorted(perm[:n_test].tolist())


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, torch.Tensor)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    return obj


def build_imagenet_loaders(n_samples, batch_size):
    print("  Loading ImageNet subset...")
    imagenet_ds = hf_load_dataset(
        "evanarlian/imagenet_1k_resized_256", split="train", trust_remote_code=True
    )
    imagenet_classnames = []
    with open("configs/classnames/imagenet_classnames.txt") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                imagenet_classnames.append(parts[1])

    indices = sorted(
        np.random.RandomState(42).choice(len(imagenet_ds), n_samples, replace=False).tolist()
    )
    ds_hf = SubsetImageDataset(imagenet_ds, indices, transform=get_processor_transform())
    ds_oai = SubsetImageDataset(imagenet_ds, indices, transform=get_transform())
    loader_hf = DataLoader(ds_hf, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=_collate)
    loader_oai = DataLoader(ds_oai, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader_hf, loader_oai, imagenet_classnames, imagenet_ds


def eval_one_dataset(dataset, conditions, registry, base_sae, base_cfg, vit_base,
                      imgnet_loader_hf, imgnet_loader_oai, imagenet_classnames,
                      batch_size, feature_analysis_batches, args):
    result = {"dataset": dataset, "conditions": {}}

    train_key = TRAIN_DATASET_KEY[dataset]
    print(f"\n{'=' * 70}\n  DATASET: {dataset}  (tasks/utils key: {train_key})\n{'=' * 70}")

    hf_dataset = hf_load_dataset(**DATASET_INFO[train_key])
    if isinstance(hf_dataset, dict):
        hf_dataset = hf_dataset["train"]
    classnames = get_classnames(train_key, hf_dataset)
    test_idx = make_split(len(hf_dataset), test_frac=0.2, max_test=args.max_own_test)
    print(f"  {len(hf_dataset)} total, {len(classnames)} classes, "
          f"{len(test_idx)} held out for eval")

    test_ds_hf = SubsetImageDataset(hf_dataset, test_idx, transform=get_processor_transform())
    test_ds_oai = SubsetImageDataset(hf_dataset, test_idx, transform=get_transform())
    test_loader_hf = DataLoader(test_ds_hf, batch_size=batch_size, shuffle=False,
                                 num_workers=4, collate_fn=_collate)
    test_loader_oai = DataLoader(test_ds_oai, batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)

    lora_path = LORA_PATHS[dataset]
    lora_model, _ = build_lora_openai_clip(lora_path, DEVICE)
    lora_model.eval()

    # ── shared, condition-independent metrics ──────────────────────────────
    shared = {}

    lora_text_feat = get_text_features_openai(lora_model, classnames, DEVICE)
    base_text_feat = get_text_features_hf(vit_base, DEVICE, classnames, "base")
    lora_text_feat_imgnet = get_text_features_openai(lora_model, imagenet_classnames, DEVICE)
    base_text_feat_imgnet = get_text_features_hf(vit_base, DEVICE, imagenet_classnames, "base")

    print("  [shared] lora_no_sae / base_no_sae (own dataset)")
    shared["lora_no_sae"] = compute_accuracy_openai(lora_model, lora_text_feat, test_loader_oai, DEVICE)
    shared["base_no_sae"] = compute_accuracy_hf(vit_base, base_text_feat, test_loader_hf, "base", DEVICE)
    print(f"    lora_no_sae={shared['lora_no_sae']:.2f}%  base_no_sae={shared['base_no_sae']:.2f}%")

    print("  [shared] + original ImageNet SAE (own dataset)")
    hook = OpenAISAEHook(lora_model, base_sae, base_cfg, DEVICE).register()
    shared["lora_base_sae"] = compute_accuracy_openai(lora_model, lora_text_feat, test_loader_oai, DEVICE)
    hook.remove()
    hooks = make_sae_hook_hf(base_sae, base_cfg, "base")
    shared["base_base_sae"] = compute_accuracy_hf(vit_base, base_text_feat, test_loader_hf, "base", DEVICE, hooks)
    print(f"    lora_base_sae={shared['lora_base_sae']:.2f}%  base_base_sae={shared['base_base_sae']:.2f}%")

    if not args.skip_imagenet:
        print("  [shared] no-SAE / original-SAE on ImageNet")
        shared["imagenet_lora_no_sae"] = compute_accuracy_openai(
            lora_model, lora_text_feat_imgnet, imgnet_loader_oai, DEVICE)
        shared["imagenet_base_no_sae"] = compute_accuracy_hf(
            vit_base, base_text_feat_imgnet, imgnet_loader_hf, "base", DEVICE)
        hook = OpenAISAEHook(lora_model, base_sae, base_cfg, DEVICE).register()
        shared["imagenet_lora_base_sae"] = compute_accuracy_openai(
            lora_model, lora_text_feat_imgnet, imgnet_loader_oai, DEVICE)
        hook.remove()
        hooks = make_sae_hook_hf(base_sae, base_cfg, "base")
        shared["imagenet_base_base_sae"] = compute_accuracy_hf(
            vit_base, base_text_feat_imgnet, imgnet_loader_hf, "base", DEVICE, hooks)
        print(f"    imagenet_lora_no_sae={shared['imagenet_lora_no_sae']:.2f}%  "
              f"imagenet_base_no_sae={shared['imagenet_base_no_sae']:.2f}%  "
              f"imagenet_lora_base_sae={shared['imagenet_lora_base_sae']:.2f}%  "
              f"imagenet_base_base_sae={shared['imagenet_base_base_sae']:.2f}%")

    if not args.skip_reconstruction:
        r = measure_reconstruction_quality(base_sae, test_loader_hf, n_batches=30,
                                            device=DEVICE, use_loader=True, vit=vit_base, cfg=base_cfg)
        shared["recon_base_sae_owndata"] = r
        if not args.skip_imagenet:
            r = measure_reconstruction_quality(base_sae, imgnet_loader_hf, n_batches=30,
                                                device=DEVICE, use_loader=True, vit=vit_base, cfg=base_cfg)
            shared["recon_base_sae_imagenet"] = r

    result["shared"] = shared

    # ── per-condition (masked SAE) metrics ──────────────────────────────────
    for condition in conditions:
        ckpt = resolve_checkpoint(registry, dataset, condition)
        if ckpt is None or not os.path.exists(ckpt):
            print(f"  [SKIP] {dataset}/{condition}: no checkpoint registered/found")
            result["conditions"][condition] = {"error": "checkpoint not found"}
            continue

        print(f"\n  -- condition: {condition} -> {ckpt}")
        masked_sae, masked_cfg, protected_mask = load_masked_sae(ckpt, DEVICE)
        cond_result = {"checkpoint": ckpt}
        if protected_mask is not None:
            cond_result["protect_frac"] = float(protected_mask.float().mean().item())

        acc = {}
        hook = OpenAISAEHook(lora_model, masked_sae, masked_cfg, DEVICE).register()
        acc["lora_masked_sae"] = compute_accuracy_openai(lora_model, lora_text_feat, test_loader_oai, DEVICE)
        hook.remove()
        hooks = make_sae_hook_hf(masked_sae, masked_cfg, "base")
        acc["base_masked_sae"] = compute_accuracy_hf(vit_base, base_text_feat, test_loader_hf, "base", DEVICE, hooks)
        print(f"    lora_masked_sae={acc['lora_masked_sae']:.2f}%  base_masked_sae={acc['base_masked_sae']:.2f}%")

        if not args.skip_imagenet:
            hook = OpenAISAEHook(lora_model, masked_sae, masked_cfg, DEVICE).register()
            acc["imagenet_lora_masked_sae"] = compute_accuracy_openai(
                lora_model, lora_text_feat_imgnet, imgnet_loader_oai, DEVICE)
            hook.remove()
            hooks = make_sae_hook_hf(masked_sae, masked_cfg, "base")
            acc["imagenet_base_masked_sae"] = compute_accuracy_hf(
                vit_base, base_text_feat_imgnet, imgnet_loader_hf, "base", DEVICE, hooks)
            print(f"    imagenet_lora_masked_sae={acc['imagenet_lora_masked_sae']:.2f}%  "
                  f"imagenet_base_masked_sae={acc['imagenet_base_masked_sae']:.2f}%")
        cond_result["accuracy"] = acc

        if not args.skip_reconstruction:
            recon = {}
            recon["own_data"] = measure_reconstruction_quality(
                masked_sae, test_loader_hf, n_batches=30, device=DEVICE,
                use_loader=True, vit=vit_base, cfg=masked_cfg)
            if not args.skip_imagenet:
                recon["imagenet"] = measure_reconstruction_quality(
                    masked_sae, imgnet_loader_hf, n_batches=30, device=DEVICE,
                    use_loader=True, vit=vit_base, cfg=masked_cfg)
            cond_result["reconstruction"] = recon

        if protected_mask is not None:
            feat = analyze_sae_features(
                masked_sae, test_loader_hf, protected_mask,
                n_batches=min(feature_analysis_batches, len(test_loader_hf)),
                device=DEVICE, use_loader=True, vit=vit_base, cfg=masked_cfg, vit_type="base",
            )
            cond_result["feature_analysis"] = feat

        result["conditions"][condition] = cond_result
        del masked_sae
        flush()

    del lora_model
    flush()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--imagenet_n_samples", type=int, default=5000)
    p.add_argument("--max_own_test", type=int, default=1000,
                   help="cap on per-dataset held-out eval set size")
    p.add_argument("--feature_analysis_batches", type=int, default=50)
    p.add_argument("--skip_imagenet", action="store_true")
    p.add_argument("--skip_reconstruction", action="store_true")
    p.add_argument("--save_path", type=str, default="out/eval_masked_finetune_sweep/results.json")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    registry = load_registry()

    print(f"Device: {DEVICE}")
    print(f"Datasets: {args.datasets}")
    print(f"Conditions: {args.conditions}")

    base_sae, base_cfg = load_sae(BASE_SAE_PATH, DEVICE)
    vit_base = load_hooked_vit(base_cfg, "base", BACKBONE, DEVICE)
    vit_base.eval()

    if not args.skip_imagenet:
        imgnet_loader_hf, imgnet_loader_oai, imagenet_classnames, imagenet_ds = \
            build_imagenet_loaders(args.imagenet_n_samples, args.batch_size)
    else:
        imgnet_loader_hf = imgnet_loader_oai = imagenet_classnames = imagenet_ds = None

    all_results = []
    for dataset in args.datasets:
        try:
            res = eval_one_dataset(
                dataset, args.conditions, registry, base_sae, base_cfg, vit_base,
                imgnet_loader_hf, imgnet_loader_oai, imagenet_classnames,
                args.batch_size, args.feature_analysis_batches, args,
            )
        except Exception as e:
            print(f"  [ERROR] {dataset}: {e}")
            res = {"dataset": dataset, "error": str(e)}
        all_results.append(res)

        with open(args.save_path, "w") as f:
            json.dump(to_serializable(all_results), f, indent=2)
        print(f"  [saved incremental] {args.save_path}")

    print(f"\nFinal results written to {args.save_path}")


if __name__ == "__main__":
    main()

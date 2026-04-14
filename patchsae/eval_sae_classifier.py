#!/usr/bin/env python3
"""
SAE as Classifier — Linear probe on SAE sparse latents

For each dataset (eurosat, caltech101, medmnist) compares classification
accuracy when using different feature representations:

  1. Raw LoRA CLIP features (512-d final projection)  — zero-shot upper bound
  2. ImageNet SAE latents (49152-d sparse)             — base SAE applied to LoRA acts
  3. Dataset-specific SAE latents (49152-d sparse)     — per-dataset SAE

A linear probe (logistic regression via SGDClassifier) is trained on the
train split and evaluated on the test split for comparisons 2 and 3.

Usage (from patchsae/ directory):
    python eval_sae_classifier.py
    python eval_sae_classifier.py --datasets eurosat caltech101
    python eval_sae_classifier.py --pool mean --save_json out/sae_clf.json
    python eval_sae_classifier.py --train_frac 0.8 --seed 42
"""

import argparse
import gc
import glob
import json
import math
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from tqdm import tqdm

try:
    from tasks.utils import load_sae, get_sae_latents
except ImportError as e:
    print(f"[FATAL] {e}\nRun from the patchsae/ directory.")
    sys.exit(1)

try:
    import clip as openai_clip
except ImportError:
    print("[FATAL] OpenAI CLIP not found.\n"
          "  Install: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKBONE      = "ViT-B/16"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SAE_PATH = "data/sae_weight/base/out.pt"
CKPT_ROOT     = "out/checkpoints"
LORA_ROOT     = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
DATA_ROOT     = "/home/sunayana/Documents/Concept_LoRA/data"

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

DATASET_CFG = {
    "eurosat": {
        "data_dir":  f"{DATA_ROOT}/eurosat/2750",
        "lora_path": f"{LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
        "use_npz":   False,
        "exclude_classes": set(),
    },
    "caltech101": {
        "data_dir":  f"{DATA_ROOT}/caltech-101",
        "lora_path": f"{LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
        "use_npz":   False,
        "exclude_classes": {"BACKGROUND_Google"},
    },
    "medmnist": {
        "data_dir":  f"{DATA_ROOT}/pathmnist_imagefolder",
        "npz_path":  f"{DATA_ROOT}/pathmnist.npz",
        "lora_path": f"{LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
        "use_npz":   True,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ── LoRA weight merging ───────────────────────────────────────────────────────

def _lora_AB(ld, proj):
    if proj in ld and isinstance(ld[proj], dict):
        try:
            return ld[proj]["w_lora_A"], ld[proj]["w_lora_B"]
        except (KeyError, TypeError):
            pass
    try:
        return ld[f"{proj}.w_lora_A"], ld[f"{proj}.w_lora_B"]
    except KeyError:
        return None, None


def build_lora_clip(lora_path: str, device: str):
    """Load OpenAI CLIP ViT-B/16 and merge LoRA weights in-place."""
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    model.eval()

    lora_state = torch.load(lora_path, map_location=device)
    if "weights" not in lora_state:
        print("  [WARN] No 'weights' key — using base CLIP.")
        return model, preprocess

    layers_dict = lora_state["weights"]
    meta        = lora_state["metadata"]
    scale       = meta["alpha"] / math.sqrt(meta["r"])
    print(f"  LoRA: rank={meta['r']}, alpha={meta['alpha']}, scale={scale:.6f}")

    with torch.no_grad():
        for i in range(12):
            ld = layers_dict.get(f"layer_{i}")
            if ld is None:
                continue
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _lora_AB(ld, proj)
                if A is None:
                    continue
                w[off:off + d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)
        for i in range(12, 24):
            ld = layers_dict.get(f"layer_{i}")
            if ld is None:
                continue
            w = model.visual.transformer.resblocks[i - 12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _lora_AB(ld, proj)
                if A is None:
                    continue
                w[off:off + d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)

    print("  LoRA weights merged.")
    return model, preprocess


# ── Dataset loading ───────────────────────────────────────────────────────────

def _find_imagefolder_root(root: str) -> str:
    subdirs = [d for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories in {root}")
    for sd in subdirs[:3]:
        if any(f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
               for f in os.listdir(os.path.join(root, sd))):
            return root
    for sd in sorted(subdirs):
        inner = [d for d in os.listdir(os.path.join(root, sd))
                 if os.path.isdir(os.path.join(root, sd, d))]
        if inner:
            try:
                return _find_imagefolder_root(os.path.join(root, sd))
            except (FileNotFoundError, PermissionError):
                continue
    raise FileNotFoundError(f"No image class directories found under {root}")


class FilteredImageFolder(Dataset):
    """ImageFolder with optional class exclusion and contiguous label remapping."""

    def __init__(self, root: str, transform, exclude_classes=None):
        img_root   = _find_imagefolder_root(root)
        full_ds    = datasets.ImageFolder(root=img_root, transform=transform)
        exclude    = exclude_classes or set()

        keep_idx       = [i for i, (_, lbl) in enumerate(full_ds.samples)
                          if full_ds.classes[lbl] not in exclude]
        kept_cls       = sorted({full_ds.classes[full_ds.targets[i]] for i in keep_idx})
        old_to_new     = {full_ds.class_to_idx[c]: new for new, c in enumerate(kept_cls)}

        self._ds        = full_ds
        self._idx       = keep_idx
        self._lmap      = old_to_new
        self.class_names = kept_cls

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, i):
        img, lbl = self._ds[self._idx[i]]
        return img, self._lmap[lbl]


def stratified_split(dataset, train_frac: float, seed: int):
    """Return (train_subset, test_subset) with stratified class balance."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)

    by_class = defaultdict(list)
    for i in range(len(dataset)):
        _, lbl = dataset[i]
        by_class[lbl].append(i)

    train_idx, test_idx = [], []
    for cls_indices in by_class.values():
        arr = np.array(cls_indices)
        rng.shuffle(arr)
        n_train = max(1, int(len(arr) * train_frac))
        train_idx.extend(arr[:n_train].tolist())
        test_idx.extend(arr[n_train:].tolist())

    return Subset(dataset, train_idx), Subset(dataset, test_idx)


class MedMNISTNpzDataset(Dataset):
    """PathMNIST split from .npz with label remapping to ImageFolder ordering."""

    def __init__(self, npz_path: str, imagefolder_root: str,
                 preprocess, split: str):
        data = np.load(npz_path)
        imgs_key = f"{split}_images"
        lbls_key = f"{split}_labels"
        if imgs_key not in data:
            raise KeyError(f"'{imgs_key}' not in {npz_path}. Keys: {list(data.keys())}")
        self.imgs      = data[imgs_key]
        self.labels    = data[lbls_key].flatten().astype(int)
        self.preprocess = preprocess

        # Remap npz indices → ImageFolder (alphabetical) indices
        if_root = _find_imagefolder_root(imagefolder_root)
        if_ds   = datasets.ImageFolder(root=if_root)
        if_map  = {n.replace("_", " ").lower(): i for n, i in if_ds.class_to_idx.items()}
        self.label_map = {
            npz_i: if_map.get(cls.lower(), npz_i)
            for npz_i, cls in enumerate(MEDMNIST_CLASSES)
        }
        # Class names in ImageFolder order (used for probing)
        self.class_names = [n.replace("_", " ") for n in if_ds.classes]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx]).convert("RGB")
        return self.preprocess(img), self.label_map[int(self.labels[idx])]


def load_splits(cfg_dict: dict, preprocess, train_frac: float, seed: int):
    """
    Return (train_ds, test_ds, class_names).
    MedMNIST uses npz train/test; others use a stratified random split.
    """
    if cfg_dict.get("use_npz", False):
        train_ds = MedMNISTNpzDataset(
            cfg_dict["npz_path"], cfg_dict["data_dir"], preprocess, split="train"
        )
        test_ds = MedMNISTNpzDataset(
            cfg_dict["npz_path"], cfg_dict["data_dir"], preprocess, split="test"
        )
        class_names = train_ds.class_names
    else:
        full_ds = FilteredImageFolder(
            cfg_dict["data_dir"], preprocess, cfg_dict.get("exclude_classes")
        )
        class_names = full_ds.class_names
        train_ds, test_ds = stratified_split(full_ds, train_frac, seed)

    return train_ds, test_ds, class_names


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_clip_features(model, loader: DataLoader, device: str) -> tuple:
    """Extract final 512-d CLIP image embeddings (L2-normalized). Returns (features, labels)."""
    all_feats, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="    CLIP features", leave=False):
        f = l2_normalize(model.encode_image(imgs.to(device)).float())
        all_feats.append(f.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


@torch.no_grad()
def extract_sae_latents(model, sae, sae_cfg, loader: DataLoader,
                        device: str, pool: str = "cls") -> tuple:
    """
    Capture intermediate activations at the SAE's target layer, encode
    through the SAE encoder, and pool to a (N, d_sae) array.

    OpenAI CLIP resblock output: [seq, batch, d_model] — transposed before SAE.

    pool: "cls"  → use CLS token (index 0) only
          "mean" → average all tokens
    """
    all_latents, all_labels = [], []
    captured = {}

    num_blocks = len(model.visual.transformer.resblocks)
    layer = sae_cfg.block_layer
    if layer < 0:
        layer = num_blocks + layer

    block = model.visual.transformer.resblocks[layer]

    def hook_fn(module, input, output):
        # [seq, batch, d_model] → [batch, seq, d_model]
        captured["act"] = output.transpose(0, 1).float()

    handle = block.register_forward_hook(hook_fn)

    for imgs, labels in tqdm(loader, desc=f"    SAE latents (layer={sae_cfg.block_layer})",
                             leave=False):
        captured.clear()
        _ = model.encode_image(imgs.to(device))          # triggers hook
        act     = captured["act"]                         # [B, seq, d_model]
        latents = get_sae_latents(sae, act)               # [B, seq, d_sae]

        if pool == "cls":
            pooled = latents[:, 0, :]
        else:
            pooled = latents.mean(dim=1)

        all_latents.append(pooled.cpu().half().numpy())
        all_labels.append(labels.numpy())

    handle.remove()
    return (np.concatenate(all_latents).astype(np.float32),
            np.concatenate(all_labels))


# ── Linear probe ─────────────────────────────────────────────────────────────

def train_linear_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    max_iter: int = 1000,
) -> tuple:
    """
    Fit logistic regression (SGDClassifier) on (X_train, y_train).
    Uses MaxAbsScaler — sparse-friendly, no centering.
    Returns (test_acc%, train_acc%).
    """
    scaler  = MaxAbsScaler()
    X_tr    = scaler.fit_transform(X_train)
    X_te    = scaler.transform(X_test)

    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=max_iter,
        tol=1e-3,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_train)
    return clf.score(X_te, y_test) * 100, clf.score(X_tr, y_train) * 100


# ── SAE discovery ─────────────────────────────────────────────────────────────

def discover_final_saes(sae_dir: str) -> list:
    """Find final-checkpoint .pt files, skip those with NaN W_enc."""
    pattern = os.path.join(sae_dir, "*/final*/*.pt")
    results = []
    for p in sorted(glob.glob(pattern)):
        try:
            ckpt  = torch.load(p, map_location="cpu")
            cfg   = ckpt.get("cfg", ckpt.get("config"))
            layer = (getattr(cfg, "block_layer", None)
                     if not isinstance(cfg, dict)
                     else cfg.get("block_layer", None))
            sd = ckpt.get("state_dict", ckpt)
            w  = sd.get("W_enc") if isinstance(sd, dict) else None
            if w is not None and torch.isnan(w).any():
                print(f"  [SKIP] {p} — NaN in W_enc")
                del ckpt; gc.collect()
                continue
            run_id = p.split(os.sep)[-3]
            del ckpt; gc.collect()
            results.append((run_id, layer, p))
        except Exception as e:
            print(f"  [SKIP] {p} — {e}")
    return results


# ── Per-dataset evaluation ────────────────────────────────────────────────────

def evaluate_dataset(
    dataset_name: str,
    cfg_dict: dict,
    base_sae_path: str,
    batch_size: int,
    num_workers: int,
    device: str,
    pool: str,
    train_frac: float,
    seed: int,
    probe_iters: int,
) -> dict:
    print(f"\n{'═'*70}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'═'*70}")

    results = {"dataset": dataset_name, "pool": pool, "conditions": []}

    # ── Load LoRA CLIP ─────────────────────────────────────────────────────────
    lora_path = cfg_dict["lora_path"]
    if not os.path.isfile(lora_path):
        print(f"  [SKIP] LoRA weights not found: {lora_path}")
        return results

    print(f"\n[1] Loading LoRA CLIP\n    {lora_path}")
    model, preprocess = build_lora_clip(lora_path, device)
    model.eval()

    # ── Load data splits ───────────────────────────────────────────────────────
    print(f"\n[2] Loading {dataset_name} data "
          f"({'npz splits' if cfg_dict.get('use_npz') else f'{train_frac:.0%} train split'})...")
    try:
        train_ds, test_ds, class_names = load_splits(cfg_dict, preprocess, train_frac, seed)
    except Exception as e:
        print(f"  [ERROR] {e}")
        del model; flush()
        return results

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device == "cuda"))

    n_train = len(train_ds)
    n_test  = len(test_ds)
    print(f"  {n_train} train / {n_test} test | {len(class_names)} classes")
    print(f"  Classes (first 5): {class_names[:5]}")

    # ── Condition A: Raw LoRA CLIP features (linear probe) ────────────────────
    print("\n[A] Raw LoRA CLIP features (512-d) — linear probe")
    try:
        X_tr_clip, y_tr = extract_clip_features(model, train_loader, device)
        X_te_clip, y_te = extract_clip_features(model, test_loader, device)
        acc_test, acc_train = train_linear_probe(X_tr_clip, y_tr, X_te_clip, y_te, probe_iters)
        print(f"    train={acc_train:.2f}%  test={acc_test:.2f}%")
        results["conditions"].append({
            "name": "LoRA CLIP features (linear probe)",
            "train_acc": round(acc_train, 4),
            "test_acc": round(acc_test, 4),
            "feature_dim": X_tr_clip.shape[1],
            "sae_path": None,
            "sae_layer": None,
            "run_id": None,
        })
    except Exception as e:
        print(f"    [ERROR] {e}")

    # Helper: extract latents + probe for one SAE
    def eval_sae(label, sae_path, run_id):
        print(f"\n  [{label}]")
        try:
            sae, sae_cfg = load_sae(sae_path, device)
            sae.eval()

            X_tr, _   = extract_sae_latents(model, sae, sae_cfg, train_loader, device, pool)
            X_te, _   = extract_sae_latents(model, sae, sae_cfg, test_loader,  device, pool)
            del sae; flush()

            acc_test, acc_train = train_linear_probe(X_tr, y_tr, X_te, y_te, probe_iters)
            print(f"    train={acc_train:.2f}%  test={acc_test:.2f}%  "
                  f"(d={X_tr.shape[1]}, sparsity≈{(X_tr == 0).mean():.1%})")
            results["conditions"].append({
                "name": label,
                "train_acc": round(acc_train, 4),
                "test_acc": round(acc_test, 4),
                "feature_dim": X_tr.shape[1],
                "sparsity": round(float((X_tr == 0).mean()), 4),
                "sae_path": sae_path,
                "sae_layer": sae_cfg.block_layer,
                "run_id": run_id,
            })
        except Exception as e:
            print(f"    [ERROR] {e}")

    # ── Condition B: ImageNet SAE latents ─────────────────────────────────────
    print(f"\n[B] ImageNet (base) SAE latents — linear probe\n    {base_sae_path}")
    if not os.path.isfile(base_sae_path):
        print(f"  [SKIP] {base_sae_path}")
    else:
        eval_sae("LoRA + ImageNet SAE latents", base_sae_path, "imagenet_base")

    # ── Conditions C+: Dataset-specific SAE latents ───────────────────────────
    sae_dir = os.path.join(CKPT_ROOT, dataset_name)
    print(f"\n[C] Dataset-specific SAEs in {sae_dir}/")
    dataset_saes = discover_final_saes(sae_dir)
    if not dataset_saes:
        print("  None found.")
    else:
        print(f"  Found {len(dataset_saes)} SAE(s):")
        for run_id, layer, p in dataset_saes:
            print(f"    run={run_id}  layer={layer}")

    for run_id, layer, sae_path in dataset_saes:
        eval_sae(
            f"LoRA + {dataset_name} SAE latents (run={run_id}, layer={layer})",
            sae_path, run_id,
        )

    del model; flush()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SAE as classifier: linear probe on SAE sparse latents"
    )
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASET_CFG.keys()),
                   choices=list(DATASET_CFG.keys()))
    p.add_argument("--pool",        default="cls", choices=["cls", "mean"],
                   help="Token pooling: 'cls' (CLS token) or 'mean' (all tokens)")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--train_frac",  type=float, default=0.8,
                   help="Train fraction for ImageFolder datasets (default: 0.8)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--probe_iters", type=int, default=1000,
                   help="Max SGDClassifier iterations")
    p.add_argument("--save_json",   type=str, default=None)
    args = p.parse_args()

    print(f"\n{'═'*70}")
    print(f"  SAE Classifier Evaluation (linear probe on SAE latents)")
    print(f"  Device   : {DEVICE}")
    print(f"  Backbone : {BACKBONE}")
    print(f"  Pool     : {args.pool}")
    print(f"  Datasets : {args.datasets}")
    print(f"  Base SAE : {BASE_SAE_PATH}")
    print(f"{'═'*70}")

    all_results = []
    for ds_name in args.datasets:
        res = evaluate_dataset(
            dataset_name=ds_name,
            cfg_dict=DATASET_CFG[ds_name],
            base_sae_path=BASE_SAE_PATH,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=DEVICE,
            pool=args.pool,
            train_frac=args.train_frac,
            seed=args.seed,
            probe_iters=args.probe_iters,
        )
        all_results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'═'*70}")
    print("  SUMMARY  (test accuracy %)")
    print(f"{'═'*70}\n")

    for res in all_results:
        print(f"  {res['dataset'].upper()}  [pool={res['pool']}]")
        print(f"  {'─'*60}")
        if not res["conditions"]:
            print("    (no results)")
        for cond in res["conditions"]:
            layer_tag = (f"  [layer={cond['sae_layer']}]"
                         if cond.get("sae_layer") is not None else "")
            dim_tag   = f"  d={cond['feature_dim']}"
            print(f"    {cond['name']:<58s}  {cond['test_acc']:6.2f}%"
                  f"{layer_tag}{dim_tag}")
        print()

    if args.save_json:
        out_path = args.save_json
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()

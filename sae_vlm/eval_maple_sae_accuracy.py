#!/usr/bin/env python3
"""
MaPLe CLIP Accuracy Evaluation: plain MaPLe vs MaPLe + ImageNet SAE vs MaPLe + dataset SAE

For each dataset (eurosat, caltech101, medmnist):
  1. MaPLe model alone          — zero-shot top-1 accuracy, no SAE
  2. MaPLe model + ImageNet SAE — base SAE (data/sae_weight/base/out.pt) via hook
  3. MaPLe model + dataset SAE  — each final checkpoint in out/checkpoints/{dataset}_maple/

By default the script auto-discovers available MaPLe checkpoints per dataset
("models you can find") and evaluates each discovered model.

Usage (run from patchsae/ directory):
    python eval_maple_sae_accuracy.py
    python eval_maple_sae_accuracy.py --datasets eurosat caltech101
    python eval_maple_sae_accuracy.py --batch_size 32 --save_json out/maple_sae_acc.json
    python eval_maple_sae_accuracy.py --maple_models /path/to/model.pth.tar-5
"""

import argparse
import gc
import glob
import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# ── Project imports (must run from patchsae/) ─────────────────────────────────
try:
    from src.sae_training.loaders import load_hooked_vit, load_sae
    from src.sae_training.hooked_vit import Hook
except ImportError as e:
    print(f"[FATAL] {e}\nRun from the patchsae/ directory.")
    sys.exit(1)

# ── Paths / constants ─────────────────────────────────────────────────────────
BACKBONE = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_SAE_PATH = "data/sae_weight/base/out.pt"
CKPT_ROOT = "out/checkpoints"
MAPLE_ROOT = "/home/sunayana/Documents/Concept_LoRA/maple_weights"
MAPLE_CONFIG_PATH = (
    "/home/sunayana/Documents/Concept_LoRA/patchsae/configs/models/maple/"
    "vit_b16_c2_ep5_batch4_2ctx.yaml"
)
MEDMNIST_MAPLE_MODEL_PATH = "/home/sunayana/Documents/Concept_LoRA/maple_weights/model.pth.tar-5"
DATA_ROOT = "/home/sunayana/Documents/Concept_LoRA/data"

# MedMNIST / PathMNIST tissue types
MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

DATASET_CFG = {
    "eurosat": {
        "data_dir": f"{DATA_ROOT}/eurosat/2750",
        "class_names": None,
        "use_npz": False,
        "split": "all",
    },
    "caltech101": {
        "data_dir": f"{DATA_ROOT}/caltech-101",
        "class_names": None,
        "use_npz": False,
        "split": "all",
        "exclude_classes": {"BACKGROUND_Google"},
    },
    "medmnist": {
        "data_dir": f"{DATA_ROOT}/pathmnist_imagefolder",
        "npz_path": f"{DATA_ROOT}/pathmnist.npz",
        "class_names": MEDMNIST_CLASSES,
        "use_npz": True,
        "split": "test",
        "maple_models": [MEDMNIST_MAPLE_MODEL_PATH],
    },
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def get_processor_transform():
    # Keep images as PIL objects; CLIPProcessor handles normalization.
    return transforms.Resize((224, 224))


def hf_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def _model_tag(path: str, root: str) -> str:
    try:
        rel = os.path.relpath(path, root)
        if not rel.startswith(".."):
            return rel
    except ValueError:
        pass
    return path


# ── Dataset loading ───────────────────────────────────────────────────────────

def _find_imagefolder_root(root: str) -> str:
    """
    Walk down directory tree to find the level where class sub-directories
    each contain image files (handles nested layouts like 101_ObjectCategories/).
    """
    subdirs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No sub-directories in {root}")

    for sd in subdirs[:3]:
        sd_path = os.path.join(root, sd)
        if any(
            f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
            for f in os.listdir(sd_path)
        ):
            return root

    for sd in sorted(subdirs):
        sd_path = os.path.join(root, sd)
        inner = [d for d in os.listdir(sd_path)
                 if os.path.isdir(os.path.join(sd_path, d))]
        if inner:
            try:
                return _find_imagefolder_root(sd_path)
            except (FileNotFoundError, PermissionError):
                continue

    raise FileNotFoundError(f"No image class directories found under {root}")


class FilteredImageFolder(Dataset):
    """ImageFolder with class exclusion and contiguous label remapping."""

    def __init__(self, root: str, transform=None, exclude_classes=None):
        img_root = _find_imagefolder_root(root)
        full_ds = datasets.ImageFolder(root=img_root, transform=transform)
        exclude = exclude_classes or set()

        keep_indices = [
            i for i, (_, lbl) in enumerate(full_ds.samples)
            if full_ds.classes[lbl] not in exclude
        ]
        kept_cls_names = sorted({full_ds.classes[full_ds.targets[i]] for i in keep_indices})
        old_to_new = {full_ds.class_to_idx[c]: new_i for new_i, c in enumerate(kept_cls_names)}

        self._dataset = full_ds
        self._indices = keep_indices
        self._label_map = old_to_new
        self.class_names = kept_cls_names

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        img, old_lbl = self._dataset[self._indices[idx]]
        return img, self._label_map[old_lbl]


class MedMNISTNpzDataset(Dataset):
    """
    PathMNIST test split from .npz with label remapping to match the
    ImageFolder ordering used during SAE training.
    """

    def __init__(self, npz_path: str, imagefolder_root: str,
                 transform=None, split: str = "test"):
        data = np.load(npz_path)
        imgs_key = f"{split}_images"
        lbls_key = f"{split}_labels"
        if imgs_key not in data:
            raise KeyError(
                f"'{imgs_key}' not found in {npz_path}. "
                f"Available keys: {list(data.keys())}"
            )
        self.imgs = data[imgs_key]
        self.labels = data[lbls_key].flatten().astype(int)
        self.transform = transform

        if_root = _find_imagefolder_root(imagefolder_root)
        if_ds = datasets.ImageFolder(root=if_root)
        if_map = {n.replace("_", " ").lower(): i for n, i in if_ds.class_to_idx.items()}

        self.label_map = {}
        for npz_idx, cls_name in enumerate(MEDMNIST_CLASSES):
            self.label_map[npz_idx] = if_map.get(cls_name.lower(), npz_idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label_map[int(self.labels[idx])]


def load_test_dataset(cfg_dict: dict, transform) -> tuple:
    """
    Returns (dataset, class_names).
    Uses NpzDataset for medmnist, FilteredImageFolder for others.
    """
    if cfg_dict.get("use_npz", False):
        ds = MedMNISTNpzDataset(
            npz_path=cfg_dict["npz_path"],
            imagefolder_root=cfg_dict["data_dir"],
            transform=transform,
            split=cfg_dict.get("split", "test"),
        )
        # Match ImageFolder ordering since NPZ labels are remapped to it.
        if_root = _find_imagefolder_root(cfg_dict["data_dir"])
        if_ds = datasets.ImageFolder(root=if_root)
        class_names = [n.replace("_", " ") for n in if_ds.classes]
    else:
        ds = FilteredImageFolder(
            root=cfg_dict["data_dir"],
            transform=transform,
            exclude_classes=cfg_dict.get("exclude_classes"),
        )
        class_names = cfg_dict.get("class_names") or ds.class_names

    return ds, class_names


# ── MaPLe model discovery ────────────────────────────────────────────────────

def discover_maple_models(
    dataset_name: str,
    maple_root: str,
    dataset_model_hints: list[str] | None = None,
    explicit_models: list[str] | None = None,
) -> list[str]:
    """
    Discover MaPLe checkpoints for a dataset.

    Priority:
      1) Explicit paths from --maple_models
      2) Dataset-specific checkpoints under maple_root/<dataset>/...
      3) Generic fallback checkpoints that exist locally
    """
    found = []
    seen_real = set()

    def _maybe_add(path: str):
        rp = os.path.realpath(path)
        if os.path.isfile(path) and rp not in seen_real:
            seen_real.add(rp)
            found.append(path)

    if explicit_models:
        for p in explicit_models:
            matches = sorted(glob.glob(p, recursive=True))
            if matches:
                for m in matches:
                    _maybe_add(m)
            else:
                _maybe_add(p)

    if dataset_model_hints:
        for p in dataset_model_hints:
            matches = sorted(glob.glob(p, recursive=True))
            if matches:
                for m in matches:
                    _maybe_add(m)
            else:
                _maybe_add(p)

    dataset_patterns = [
        os.path.join(maple_root, dataset_name, "**", "model.pth.tar*"),
        os.path.join(maple_root, f"{dataset_name}*", "**", "model.pth.tar*"),
    ]
    for pat in dataset_patterns:
        for p in sorted(glob.glob(pat, recursive=True)):
            _maybe_add(p)

    fallback_patterns = [
        os.path.join(maple_root, "model.pth.tar*"),
        "/home/sunayana/Documents/model.pth.tar*",
        os.path.join(SCRIPT_DIR, "data/clip/maple/imagenet/model.pth.tar*"),
    ]
    for pat in fallback_patterns:
        for p in sorted(glob.glob(pat, recursive=True)):
            _maybe_add(p)

    return found


# ── SAE hooks for MaPLe ──────────────────────────────────────────────────────

def make_maple_sae_hook(sae, sae_cfg):
    """Create SAE reconstruction hook for MaPLe residual activations."""
    logged = [False]
    block_layer = getattr(sae_cfg, "block_layer", -1)
    module_name = getattr(sae_cfg, "module_name", "resid")

    def hook_fn(act):
        orig_dtype = act.dtype

        # MaPLe hooked activations are [seq, batch, d_model].
        act_bsd = act.transpose(0, 1).float()
        sae_out, _, _ = sae(act_bsd)

        if not logged[0]:
            logged[0] = True
            mse = (act_bsd - sae_out).pow(2).mean().item()
            cos = F.cosine_similarity(
                act_bsd.reshape(-1, act_bsd.shape[-1]),
                sae_out.reshape(-1, sae_out.shape[-1]),
                dim=-1,
            ).mean().item()
            print(f"    [HOOK] MSE={mse:.6f}, cos={cos:.6f}")

        return sae_out.transpose(0, 1).to(orig_dtype)

    return [
        Hook(
            block_layer,
            module_name,
            hook_fn,
            return_module_output=False,
            is_custom=True,
        )
    ]


# ── Accuracy computation ──────────────────────────────────────────────────────

@torch.no_grad()
def get_text_features_maple(vit) -> torch.Tensor:
    tf = vit.model.get_text_features().float()
    return l2_normalize(tf)


@torch.no_grad()
def compute_accuracy_maple(
    vit,
    text_features: torch.Tensor,
    loader: DataLoader,
    device: str,
    hooks=None,
) -> float:
    """Zero-shot top-1 accuracy using MaPLe CLIP image encoder."""
    logit_scale = vit.model.logit_scale.exp()
    tf = l2_normalize(text_features.float())

    correct = total = 0
    for images, labels in tqdm(loader, desc="    eval", leave=False):
        inputs = vit.processor(
            images=images,
            text="",
            return_tensors="pt",
            padding=True,
        ).to(device)

        if hooks:
            out = vit.run_with_hooks(hooks, return_type="output", **inputs)
        else:
            out = vit(return_type="output", **inputs)

        img_feat = l2_normalize(out.float())
        logits = logit_scale * (img_feat @ tf.t())
        preds = logits.argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total * 100.0


# ── SAE checkpoint discovery ──────────────────────────────────────────────────

def discover_final_saes(sae_dir: str) -> list:
    """
    Find final-checkpoint .pt files under sae_dir/*/final*/*.pt.
    Returns list of (run_id, block_layer, path), skipping NaN checkpoints.
    """
    pattern = os.path.join(sae_dir, "*/final*/*.pt")
    paths = sorted(glob.glob(pattern))

    results = []
    for p in paths:
        try:
            ckpt = torch.load(p, map_location="cpu")
            cfg = ckpt.get("cfg", ckpt.get("config"))
            layer = (
                getattr(cfg, "block_layer", None)
                if not isinstance(cfg, dict)
                else cfg.get("block_layer", None)
            )

            sd = ckpt.get("state_dict", ckpt)
            w = sd.get("W_enc") if isinstance(sd, dict) else None
            if w is not None and torch.isnan(w).any():
                print(f"  [SKIP] {p} — NaN in W_enc")
                del ckpt
                gc.collect()
                continue

            run_id = p.split(os.sep)[-3]
            del ckpt
            gc.collect()
            results.append((run_id, layer, p))
        except Exception as e:
            print(f"  [SKIP] {p} — {e}")

    return results


# ── Per-dataset evaluation ────────────────────────────────────────────────────

def evaluate_dataset(
    dataset_name: str,
    cfg_dict: dict,
    base_sae_path: str,
    maple_root: str,
    maple_config_path: str,
    explicit_maple_models: list[str] | None,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict:
    """Run all three accuracy conditions for one dataset across discovered MaPLe models."""
    print(f"\n{'═' * 70}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'═' * 70}")

    results = {"dataset": dataset_name, "conditions": []}

    # ── 1) Load dataset once ──────────────────────────────────────────────────
    print(f"\n[1] Loading {dataset_name} dataset...")
    transform = get_processor_transform()
    try:
        ds, class_names = load_test_dataset(cfg_dict, transform)
    except Exception as e:
        print(f"  [ERROR] Could not load dataset: {e}")
        return results

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=hf_collate_fn,
        pin_memory=(device == "cuda"),
    )
    print(f"  {len(ds)} images | {len(class_names)} classes")
    print(f"  Classes (first 5): {class_names[:5]}")

    # ── 2) Discover MaPLe model checkpoints ──────────────────────────────────
    print(f"\n[2] Discovering MaPLe models for {dataset_name}...")
    maple_models = discover_maple_models(
        dataset_name=dataset_name,
        maple_root=maple_root,
        dataset_model_hints=cfg_dict.get("maple_models"),
        explicit_models=explicit_maple_models,
    )
    if not maple_models:
        print("  [SKIP] No MaPLe model checkpoints found.")
        return results

    for p in maple_models:
        print(f"    {p}")

    # ── 3) Discover dataset-specific MaPLe SAEs ──────────────────────────────
    sae_dir = os.path.join(CKPT_ROOT, f"{dataset_name}_maple")
    print(f"\n[3] Discovering dataset-specific MaPLe SAEs in {sae_dir}/")
    dataset_saes = discover_final_saes(sae_dir)
    if not dataset_saes:
        print("  None found.")
    else:
        print(f"  Found {len(dataset_saes)} SAE(s):")
        for run_id, layer, path in dataset_saes:
            print(f"    run={run_id}  layer={layer}  {path}")

    # ── 4) Load base SAE once (if available) ─────────────────────────────────
    base_sae = base_cfg = None
    if os.path.isfile(base_sae_path):
        try:
            base_sae, base_cfg = load_sae(base_sae_path, device)
            base_sae.eval()
            print(f"\n[4] Base SAE loaded: {base_sae_path} (layer={base_cfg.block_layer})")
        except Exception as e:
            print(f"\n[4] [WARN] Failed to load base SAE: {e}")
            base_sae = base_cfg = None
    else:
        print(f"\n[4] [WARN] Base SAE not found: {base_sae_path}")

    # Reference cfg needed by load_hooked_vit().
    ref_cfg = base_cfg
    if ref_cfg is None and dataset_saes:
        try:
            _, ref_cfg = load_sae(dataset_saes[0][2], device)
            print("  Using first dataset SAE cfg as MaPLe loader reference config.")
        except Exception as e:
            print(f"  [ERROR] Could not load reference cfg from dataset SAE: {e}")
            return results

    if ref_cfg is None:
        print("  [ERROR] No reference SAE cfg available to construct MaPLe model.")
        return results

    # ── 5) Evaluate each discovered MaPLe model ──────────────────────────────
    print("\n[5] Evaluating discovered MaPLe model(s)...")
    for model_path in maple_models:
        model_tag = _model_tag(model_path, maple_root)
        print(f"\n  MODEL: {model_tag}")
        print(f"    checkpoint: {model_path}")
        print(f"    config:     {maple_config_path}")

        try:
            vit = load_hooked_vit(
                ref_cfg,
                "maple",
                BACKBONE,
                device,
                model_path=model_path,
                config_path=maple_config_path,
                classnames=class_names,
            )
            vit.eval()
        except Exception as e:
            print(f"    [ERROR] Failed to load MaPLe model: {e}")
            flush()
            continue

        # Build MaPLe text features once per model.
        try:
            text_feat = get_text_features_maple(vit)
            print(f"    text_features: {text_feat.shape}")
        except Exception as e:
            print(f"    [ERROR] Could not build text features: {e}")
            del vit
            flush()
            continue

        # A) MaPLe only
        print("\n    [A] MaPLe model (no SAE)")
        try:
            acc_maple = compute_accuracy_maple(vit, text_feat, loader, device)
            print(f"      → {acc_maple:.2f}%")
            results["conditions"].append({
                "name": "MaPLe (no SAE)",
                "accuracy": round(acc_maple, 4),
                "sae_path": None,
                "sae_layer": None,
                "run_id": None,
                "maple_model_path": model_path,
                "maple_model_tag": model_tag,
            })
        except Exception as e:
            print(f"      [ERROR] {e}")

        # B) MaPLe + ImageNet SAE
        print(f"\n    [B] MaPLe + ImageNet (base) SAE\n      {base_sae_path}")
        if base_sae is None or base_cfg is None:
            print("      [SKIP] Base SAE unavailable.")
        else:
            try:
                hooks = make_maple_sae_hook(base_sae, base_cfg)
                acc_base = compute_accuracy_maple(vit, text_feat, loader, device, hooks=hooks)
                print(f"      → {acc_base:.2f}%")
                results["conditions"].append({
                    "name": "MaPLe + ImageNet SAE",
                    "accuracy": round(acc_base, 4),
                    "sae_path": base_sae_path,
                    "sae_layer": base_cfg.block_layer,
                    "run_id": "imagenet_base",
                    "maple_model_path": model_path,
                    "maple_model_tag": model_tag,
                })
            except Exception as e:
                print(f"      [ERROR] {e}")

        # C+) MaPLe + dataset-specific SAEs
        for run_id, layer, sae_path in dataset_saes:
            label = f"MaPLe + {dataset_name} SAE (run={run_id}, layer={layer})"
            print(f"\n      {label}")
            try:
                ds_sae, ds_cfg = load_sae(sae_path, device)
                ds_sae.eval()
                hooks = make_maple_sae_hook(ds_sae, ds_cfg)
                acc = compute_accuracy_maple(vit, text_feat, loader, device, hooks=hooks)
                print(f"      → {acc:.2f}%")
                results["conditions"].append({
                    "name": label,
                    "accuracy": round(acc, 4),
                    "sae_path": sae_path,
                    "sae_layer": layer,
                    "run_id": run_id,
                    "maple_model_path": model_path,
                    "maple_model_tag": model_tag,
                })
                del ds_sae
                flush()
            except Exception as e:
                print(f"      [ERROR] {e}")

        del vit
        flush()

    if base_sae is not None:
        del base_sae
    flush()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Zero-shot accuracy: MaPLe vs MaPLe+ImageNet SAE vs MaPLe+dataset SAE"
    )
    p.add_argument(
        "--datasets", nargs="+",
        default=list(DATASET_CFG.keys()),
        choices=list(DATASET_CFG.keys()),
        help="Datasets to evaluate (default: all)",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--base_sae_path", type=str, default=BASE_SAE_PATH)
    p.add_argument("--maple_root", type=str, default=MAPLE_ROOT)
    p.add_argument("--maple_config", type=str, default=MAPLE_CONFIG_PATH)
    p.add_argument(
        "--maple_models",
        nargs="*",
        default=None,
        help=(
            "Optional MaPLe checkpoint path(s) or glob(s). If omitted, auto-discover "
            "dataset-specific and fallback models."
        ),
    )
    p.add_argument(
        "--save_json", type=str, default=None,
        help="Path to save JSON results (e.g. out/maple_sae_accuracy.json)",
    )
    args = p.parse_args()

    print(f"\n{'═' * 70}")
    print("  MaPLe SAE Accuracy Evaluation")
    print(f"  Device      : {DEVICE}")
    print(f"  Backbone    : {BACKBONE}")
    print(f"  Datasets    : {args.datasets}")
    print(f"  Base SAE    : {args.base_sae_path}")
    print(f"  MaPLe root  : {args.maple_root}")
    print(f"  MaPLe config: {args.maple_config}")
    if args.maple_models:
        print(f"  MaPLe models: {args.maple_models}")
    else:
        print("  MaPLe models: auto-discover")
    print(f"{'═' * 70}")

    all_results = []
    for ds_name in args.datasets:
        res = evaluate_dataset(
            dataset_name=ds_name,
            cfg_dict=DATASET_CFG[ds_name],
            base_sae_path=args.base_sae_path,
            maple_root=args.maple_root,
            maple_config_path=args.maple_config,
            explicit_maple_models=args.maple_models,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=DEVICE,
        )
        all_results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'═' * 70}")
    print("  SUMMARY")
    print(f"{'═' * 70}\n")

    for res in all_results:
        print(f"  {res['dataset'].upper()}")
        print(f"  {'─' * 60}")
        if not res["conditions"]:
            print("    (no results)")

        for cond in res["conditions"]:
            layer_tag = (
                f"  [layer={cond['sae_layer']}]"
                if cond["sae_layer"] is not None
                else ""
            )
            model_tag = cond.get("maple_model_tag", "unknown_model")
            row_name = f"{cond['name']} | model={model_tag}"
            print(f"    {row_name:<80s}  {cond['accuracy']:6.2f}%{layer_tag}")
        print()

    if args.save_json:
        out_path = args.save_json
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()

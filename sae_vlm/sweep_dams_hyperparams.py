#!/usr/bin/env python3
"""
Full DAMS hyperparameter sweep across all discoverable SAE checkpoints + datasets.

Goal
----
Maximise separation between Base SAE and Adapted SAE by sweeping DAMS v3 mixing
hyperparameters:
    DAMS = EC^rho × (alpha × CSS_norm + beta × FSS + gamma × DAS)
    CSS_norm = css_raw / (css_raw + kappa)
    FSS      = class-wise max specificity with entropy sharpening power s
    DAS      = class-balanced kernel target alignment CKA(A_pool, label kernel)
    SUS      = chance-normalised held-out ridge-readout balanced accuracy

What this script does
---------------------
1. Auto-discovers all OpenAI SAE checkpoints under out/checkpoints/**/final_sparse_autoencoder_openai/*.pt
2. Maps each checkpoint to a canonical dataset and matching LoRA weights (when available)
3. Adds Base SAE for every dataset that has adapted checkpoints
4. Extracts raw sub-metrics once per SAE/dataset run
5. Computes SUS to estimate whether each frozen SAE is actually useful on
   each dataset
6. Analytically sweeps alpha/beta/gamma/kappa/s/rho to find the largest
   base-vs-adapted gap

Outputs
-------
- out/dams_sweep_full.csv
- out/dams_sweep_top50.csv
- out/dams_raw_metrics.json
- out/dams_run_manifest.json
"""

import argparse
import itertools
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.dams import (  # noqa: E402
    _compute_pooled_activations,
    compute_domain_alignment_score,
    compute_kernel_alignment,
    compute_mmd_score,
    compute_sae_utility_score,
    extract_fss_components,
    fss_from_components,
)
from src.sae_training.loaders import load_sae  # noqa: E402

try:  # noqa: E402
    import clip as openai_clip
except ImportError:  # noqa: E402
    import CLIP_LoRA.clip as openai_clip


# ----------------------------- Defaults --------------------------------------

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = ROOT / "out" / "checkpoints"
DEFAULT_BASE_SAE = ROOT / "out" / "sae_weight" / "base" / "out.pt"
DEFAULT_DATA_ROOT = Path("/home/sunayana/Documents/Concept_LoRA/data")
DEFAULT_LORA_ROOT = Path("/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16")

BACKBONE = "ViT-B/16"
MEDMNIST_CLASSES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]

# checkpoint top-folder -> canonical dataset
CKPT_DATASET_MAP = {
    "eurosat": "eurosat",
    "eurosat_maple": "eurosat",
    "caltech101": "caltech101",
    "caltech101_maple": "caltech101",
    "medmnist": "medmnist",
    "medmnist_maple": "medmnist",
    "masked_finetune": "medmnist",
    "masked_finetune_lora": "medmnist",
    "masked_finetune_maple": "medmnist",
    "9g0pkku9": "medmnist",
    "owdr2cw0": "medmnist",
    "dtd": "dtd",
    "cub2002011": "cub2002011",
    "ucf101": "ucf101",
}

DATASET_CFG = {
    "eurosat": {
        "data_dir": "eurosat/2750",
        "exclude_classes": None,
        "is_medmnist": False,
    },
    "caltech101": {
        "data_dir": "caltech-101",
        "exclude_classes": {"BACKGROUND_Google"},
        "is_medmnist": False,
    },
    "medmnist": {
        "data_dir": "pathmnist_imagefolder",
        "npz": "pathmnist_imagefolder/pathmnist.npz",
        "exclude_classes": None,
        "is_medmnist": True,
    },
    "cub2002011": {
        "data_dir": "cub2002011/test",
        "exclude_classes": None,
        "is_medmnist": False,
    },
    "dtd": {
        "data_dir": "dtd/images",
        "exclude_classes": None,
        "is_medmnist": False,
    },
    "ucf101": {
        "data_dir": "UCF101/UCF-101-midframes",
        "exclude_classes": None,
        "is_medmnist": False,
    },
}


@dataclass
class RunSpec:
    kind: str
    name: str
    dataset: str
    sae_path: Path
    lora_path: Optional[Path]
    source_group: str
    run_id: str


class MedMNISTDataset(Dataset):
    def __init__(self, data_root: Path, preprocess):
        npz_path = data_root / "pathmnist_imagefolder" / "pathmnist.npz"
        if not npz_path.exists():
            npz_path = data_root / "pathmnist.npz"
        imagefolder_root = data_root / "pathmnist_imagefolder"
        self.preprocess = preprocess
        self.imagefolder = None

        imagefolder = datasets.ImageFolder(root=find_imagefolder_root(imagefolder_root))
        mapped = {name.replace("_", " ").lower(): idx for name, idx in imagefolder.class_to_idx.items()}
        self.label_map = {i: mapped.get(name.lower(), i) for i, name in enumerate(MEDMNIST_CLASSES)}
        self.num_classes = len(MEDMNIST_CLASSES)

        try:
            data = np.load(npz_path)
            self.images = data["test_images"]
            self.labels = data["test_labels"].flatten().astype(int)
        except Exception as exc:
            print(f"[WARN] Falling back to ImageFolder for MedMNIST because npz load failed: {exc}")
            self.images = None
            self.labels = None
            self.imagefolder = datasets.ImageFolder(
                root=find_imagefolder_root(imagefolder_root),
                transform=preprocess,
            )

    def __len__(self) -> int:
        if self.imagefolder is not None:
            return len(self.imagefolder)
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.imagefolder is not None:
            return self.imagefolder[idx]

        image = Image.fromarray(self.images[idx])
        label = self.label_map[int(self.labels[idx])]
        return self.preprocess(image), label


class FilteredImageFolder(Dataset):
    def __init__(self, root: Path, preprocess, exclude_classes: Optional[set] = None):
        image_root = find_imagefolder_root(root)
        full = datasets.ImageFolder(root=image_root, transform=preprocess)

        exclude = exclude_classes or set()
        kept = [
            i
            for i, (_, lbl) in enumerate(full.samples)
            if full.classes[lbl] not in exclude
        ]

        self.dataset = full
        self.indices = kept

        kept_class_names = sorted({full.classes[full.targets[i]] for i in kept})
        old_to_new = {full.class_to_idx[c]: ni for ni, c in enumerate(kept_class_names)}
        self.label_map = old_to_new
        self.num_classes = len(kept_class_names)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.dataset[self.indices[idx]]
        return image, self.label_map[label]


class ActivationCapture:
    def __init__(self):
        self.act = None
        self.handle = None

    def register(self, block):
        def hook(_mod, _inp, out):
            self.act = out.detach().float().transpose(0, 1)

        self.handle = block.register_forward_hook(hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def alpha_grid(start: float, stop: float, step: float) -> List[float]:
    # inclusive stop with stable rounding
    vals = []
    x = start
    while x <= stop + 1e-12:
        vals.append(round(x, 6))
        x += step
    return vals


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)


def parse_layer_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"_(-?\d+)_resid", path.name)
    return int(m.group(1)) if m else None


def find_imagefolder_root(root: Path) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for cur, dirs, _files in os.walk(root):
        if not dirs:
            continue
        sample = Path(cur) / dirs[0]
        if not sample.exists():
            continue
        exts = {
            Path(f).suffix.lower()
            for f in os.listdir(sample)
            if (sample / f).is_file()
        }
        if exts & {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            return Path(cur)

    return root


def dataset_available(dataset_name: str, data_root: Path) -> bool:
    cfg = DATASET_CFG[dataset_name]
    data_dir = data_root / cfg["data_dir"]
    if not data_dir.exists():
        return False
    if cfg["is_medmnist"]:
        npz_path = data_root / cfg["npz"]
        return npz_path.exists()
    return True


def make_dataset(dataset_name: str, preprocess, data_root: Path):
    cfg = DATASET_CFG[dataset_name]
    if cfg["is_medmnist"]:
        return MedMNISTDataset(data_root, preprocess)

    return FilteredImageFolder(
        root=data_root / cfg["data_dir"],
        preprocess=preprocess,
        exclude_classes=cfg["exclude_classes"],
    )


def resolve_lora_path(dataset_name: str, lora_root: Path) -> Optional[Path]:
    ds_root = lora_root / dataset_name / "16shots"
    if not ds_root.exists():
        return None

    preferred = ds_root / "seed1" / "lora_weights.pt"
    if preferred.exists():
        return preferred

    candidates = sorted(ds_root.glob("seed*/lora_weights.pt"))
    return candidates[0] if candidates else None


def build_clip_model(
    device: str,
    lora_path: Optional[Path],
    backbone: str = BACKBONE,
):
    model, preprocess = openai_clip.load(backbone, device=device)

    if lora_path is None:
        model.eval()
        return model, preprocess

    state = torch.load(lora_path, map_location=device, weights_only=False)
    if "weights" not in state:
        model.load_state_dict(state, strict=False)
        model.eval()
        return model, preprocess

    layers = state["weights"]
    meta = state["metadata"]
    scale = meta["alpha"] / math.sqrt(meta["r"])

    with torch.no_grad():
        for i in range(12):
            layer_dict = layers.get(f"layer_{i+12}", {})
            if not layer_dict:
                continue

            block = model.visual.transformer.resblocks[i]
            w = block.attn.in_proj_weight.data
            d = w.shape[1]

            for proj, off in (("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)):
                try:
                    if isinstance(layer_dict.get(proj), dict):
                        A = layer_dict[proj]["w_lora_A"]
                        B = layer_dict[proj]["w_lora_B"]
                    else:
                        A = layer_dict[f"{proj}.w_lora_A"]
                        B = layer_dict[f"{proj}.w_lora_B"]
                    delta = (scale * B.float().to(device) @ A.float().to(device)).to(w.dtype)
                    w[off : off + d] += delta
                except Exception:
                    continue

    model.eval()
    return model, preprocess


def discover_runs(
    checkpoint_root: Path,
    base_sae_path: Path,
    data_root: Path,
    lora_root: Path,
    dataset_filter: Optional[Sequence[str]] = None,
) -> Tuple[List[RunSpec], Dict[str, List[str]]]:
    skipped: Dict[str, List[str]] = {
        "non_openai": [],
        "unknown_checkpoint_group": [],
        "missing_dataset_data": [],
    }

    adapted: List[RunSpec] = []

    ckpts = sorted(checkpoint_root.glob("**/final_sparse_autoencoder_openai/*.pt"))

    if not ckpts:
        raise FileNotFoundError(
            f"No OpenAI SAE checkpoints found under {checkpoint_root}/**/final_sparse_autoencoder_openai/*.pt"
        )

    dataset_filter_set = set(dataset_filter) if dataset_filter else None

    for ckpt in ckpts:
        rel = ckpt.relative_to(checkpoint_root)
        top = rel.parts[0]

        dataset = CKPT_DATASET_MAP.get(top)
        if dataset is None:
            skipped["unknown_checkpoint_group"].append(str(ckpt))
            continue

        if dataset_filter_set and dataset not in dataset_filter_set:
            continue

        if not dataset_available(dataset, data_root):
            skipped["missing_dataset_data"].append(f"{ckpt} -> {dataset}")
            continue

        run_id = rel.parts[1] if len(rel.parts) > 3 else top
        if "maple" in top:
            kind = "Maple SAE"
        elif "masked" in top:
            kind = "Masked SAE"
        else:
            kind = "Adapted SAE"

        lora_path = resolve_lora_path(dataset, lora_root)
        name = f"{kind} [{top}/{run_id}]"

        adapted.append(
            RunSpec(
                kind=kind,
                name=name,
                dataset=dataset,
                sae_path=ckpt,
                lora_path=lora_path,
                source_group=top,
                run_id=run_id,
            )
        )

    adapted.sort(key=lambda r: (r.dataset, r.source_group, r.run_id, str(r.sae_path)))

    # Add one base SAE per dataset that has adapted SAEs.
    datasets_with_adapted = sorted({r.dataset for r in adapted})
    all_runs: List[RunSpec] = []

    if not base_sae_path.exists():
        raise FileNotFoundError(f"Base SAE checkpoint not found: {base_sae_path}")

    for ds in datasets_with_adapted:
        all_runs.append(
            RunSpec(
                kind="Base SAE",
                name="Base SAE",
                dataset=ds,
                sae_path=base_sae_path,
                lora_path=None,
                source_group="base",
                run_id="base",
            )
        )

    all_runs.extend(adapted)
    return all_runs, skipped


def build_loader_for_run(
    run: RunSpec,
    device: str,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_samples: Optional[int],
    subset_seed: int,
    subset_index_cache: Dict[str, List[int]],
    clip_cache: Dict[Optional[str], Tuple[torch.nn.Module, object]],
):
    lora_key = str(run.lora_path) if run.lora_path else None
    if lora_key not in clip_cache:
        clip_cache[lora_key] = build_clip_model(device=device, lora_path=run.lora_path)

    model, preprocess = clip_cache[lora_key]
    dataset = make_dataset(run.dataset, preprocess, data_root)
    num_classes = dataset.num_classes

    if max_samples is not None and max_samples > 0 and len(dataset) > max_samples:
        if run.dataset not in subset_index_cache:
            ds_seed = subset_seed + sum(ord(c) for c in run.dataset)
            rng = np.random.default_rng(ds_seed)
            idx = rng.choice(len(dataset), size=max_samples, replace=False)
            subset_index_cache[run.dataset] = sorted(int(i) for i in idx.tolist())
        dataset = Subset(dataset, subset_index_cache[run.dataset])
        dataset.num_classes = num_classes

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return model, dataset, loader


def compute_objective(gaps: List[float], mode: str) -> float:
    gaps = [float(g) for g in gaps if np.isfinite(g)]
    if not gaps:
        return float("-inf")

    if mode == "total_gap":
        return float(np.sum(gaps))
    if mode == "mean_gap":
        return float(np.mean(gaps))
    if mode == "min_gap":
        return float(np.min(gaps))
    if mode == "balanced":
        # Makes the gap evident across all datasets, not just one outlier.
        return float(np.mean(gaps) + np.min(gaps))

    raise ValueError(f"Unknown objective mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Full DAMS hyperparameter sweep")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--base-sae", type=Path, default=DEFAULT_BASE_SAE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--lora-root", type=Path, default=DEFAULT_LORA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out")

    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASET_CFG.keys()), default=None)
    parser.add_argument("--dry-run-discovery", action="store_true")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--ec-subsample", type=int, default=2000)
    parser.add_argument("--sae-batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=0)

    parser.add_argument("--alpha-start", type=float, default=0.0)
    parser.add_argument("--alpha-stop", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.025)
    parser.add_argument(
        "--gammas",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5",
        help=(
            "Comma-separated DAS weights. For each gamma, alpha/beta split the "
            "remaining weight: alpha=(1-gamma)*alpha_frac, beta=(1-gamma)*(1-alpha_frac)."
        ),
    )
    parser.add_argument(
        "--coverage-powers",
        type=str,
        default="0.0,0.25,0.5,1.0",
        help="Comma-separated rho values for the EC reliability gate EC^rho. Use 0 to disable the gate.",
    )
    parser.add_argument("--das-subsample", type=int, default=2000)
    parser.add_argument("--no-utility-score", action="store_true")
    parser.add_argument("--utility-top-features", type=int, default=4096)
    parser.add_argument("--utility-splits", type=int, default=3)
    parser.add_argument("--utility-ridge", type=float, default=1.0)
    parser.add_argument(
        "--kappas",
        type=str,
        default="0.01,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0,1.5,2.0",
    )
    parser.add_argument("--sharpenings", type=str, default="1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0")
    parser.add_argument(
        "--objective",
        choices=["balanced", "total_gap", "mean_gap", "min_gap"],
        default="balanced",
    )

    args = parser.parse_args()

    print("=" * 90)
    print("Discovering SAE runs")
    print("=" * 90)
    runs, skipped = discover_runs(
        checkpoint_root=args.checkpoint_root,
        base_sae_path=args.base_sae,
        data_root=args.data_root,
        lora_root=args.lora_root,
        dataset_filter=args.datasets,
    )

    if not runs:
        raise RuntimeError("No runs discovered after filtering.")

    datasets_seen = sorted({r.dataset for r in runs})
    for ds in datasets_seen:
        rs = [r for r in runs if r.dataset == ds]
        base_count = sum(1 for r in rs if r.kind == "Base SAE")
        adapted_count = len(rs) - base_count
        print(f"{ds:<12}  base={base_count} adapted={adapted_count}")

    print(f"Total discovered runs: {len(runs)}")
    if skipped["unknown_checkpoint_group"]:
        print(f"Skipped unknown groups: {len(skipped['unknown_checkpoint_group'])}")
    if skipped["missing_dataset_data"]:
        print(f"Skipped missing dataset data: {len(skipped['missing_dataset_data'])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "dams_run_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(
            {
                "runs": [
                    {
                        "kind": r.kind,
                        "name": r.name,
                        "dataset": r.dataset,
                        "sae_path": str(r.sae_path),
                        "lora_path": str(r.lora_path) if r.lora_path else None,
                        "source_group": r.source_group,
                        "run_id": r.run_id,
                    }
                    for r in runs
                ],
                "skipped": skipped,
            },
            f,
            indent=2,
        )
    print(f"Saved manifest: {manifest_path}")

    if args.dry_run_discovery:
        print("Dry-run discovery requested; exiting before feature extraction.")
        return

    # ----------------------- Step 1: Extract sub-metrics ---------------------
    print("\n" + "=" * 90)
    print("Step 1/3: Extracting EC/CSS/FSS components")
    print("=" * 90)

    raw_data = []
    clip_cache: Dict[Optional[str], Tuple[torch.nn.Module, object]] = {}
    subset_index_cache: Dict[str, List[int]] = {}

    for idx, run in enumerate(runs, start=1):
        print("\n" + "-" * 90)
        print(f"[{idx}/{len(runs)}] {run.name} | dataset={run.dataset} | kind={run.kind}")

        sae, cfg = load_sae(str(run.sae_path), args.device)
        sae.eval().to(args.device)

        model, dataset, loader = build_loader_for_run(
            run=run,
            device=args.device,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            max_samples=args.max_samples,
            subset_seed=args.subset_seed,
            subset_index_cache=subset_index_cache,
            clip_cache=clip_cache,
        )

        n_layers = len(model.visual.transformer.resblocks)
        block_layer = int(cfg.block_layer)
        layer_index = block_layer if block_layer >= 0 else n_layers + block_layer

        cap = ActivationCapture()
        cap.register(model.visual.transformer.resblocks[layer_index])

        feat_batches = []
        labels: List[int] = []

        with torch.no_grad():
            for images, labs in tqdm(loader, desc="extract", leave=False):
                model.encode_image(images.to(args.device))
                feat_batches.append(cap.act.cpu())
                labels.extend(labs.tolist())

        cap.remove()
        features = torch.cat(feat_batches, dim=0)
        num_classes = dataset.num_classes

        print(f"features={tuple(features.shape)} classes={num_classes} layer={block_layer}")

        acts = _compute_pooled_activations(
            sae,
            features,
            device=args.device,
            batch_size=args.sae_batch_size,
        )
        ec, _, _ = compute_kernel_alignment(
            sae,
            features,
            device=args.device,
            batch_size=args.sae_batch_size,
            subsample=args.ec_subsample,
            precomputed_acts=acts,
        )
        css_raw, _, _ = compute_mmd_score(
            sae,
            features,
            labels,
            num_classes,
            device=args.device,
            batch_size=args.sae_batch_size,
            precomputed_acts=acts,
        )
        p_c_given_f, h_norm, support_mask = extract_fss_components(acts, labels, num_classes)
        das = compute_domain_alignment_score(
            acts,
            labels,
            num_classes,
            subsample=args.das_subsample,
        )
        if args.no_utility_score:
            utility, utility_balanced_acc, utility_chance = 0.0, 0.0, 0.0
        else:
            utility, utility_balanced_acc, utility_chance = compute_sae_utility_score(
                acts,
                labels,
                num_classes,
                n_splits=args.utility_splits,
                ridge=args.utility_ridge,
                top_features=args.utility_top_features,
            )

        raw_data.append(
            {
                "name": run.name,
                "kind": run.kind,
                "dataset": run.dataset,
                "source_group": run.source_group,
                "run_id": run.run_id,
                "sae_path": str(run.sae_path),
                "lora_path": str(run.lora_path) if run.lora_path else None,
                "layer": block_layer,
                "layer_from_filename": parse_layer_from_filename(run.sae_path),
                "ec": float(np.nan_to_num(ec, nan=0.0, posinf=0.0, neginf=0.0)),
                "css_raw": float(np.nan_to_num(css_raw, nan=0.0, posinf=0.0, neginf=0.0)),
                "das": float(np.nan_to_num(das, nan=0.0, posinf=0.0, neginf=0.0)),
                "utility": float(np.nan_to_num(utility, nan=0.0, posinf=0.0, neginf=0.0)),
                "utility_balanced_acc": float(np.nan_to_num(utility_balanced_acc, nan=0.0, posinf=0.0, neginf=0.0)),
                "utility_chance": float(np.nan_to_num(utility_chance, nan=0.0, posinf=0.0, neginf=0.0)),
                "nc": int(num_classes),
                "p_c_given_f": p_c_given_f,
                "h_norm": h_norm,
                "support_mask": support_mask,
            }
        )
        print(
            f"EC={ec:.4f} CSS_raw={css_raw:.6f} DAS={das:.4f} "
            f"SUS={utility:.4f} bal_acc={utility_balanced_acc:.4f}"
        )

        del sae, acts, features, feat_batches
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"\nExtracted raw metrics for {len(raw_data)} runs.")

    # ----------------------- Step 2: Hyperparameter sweep --------------------
    print("\n" + "=" * 90)
    print("Step 2/3: Analytical sweep")
    print("=" * 90)

    alphas = alpha_grid(args.alpha_start, args.alpha_stop, args.alpha_step)
    kappas = parse_float_list(args.kappas)
    sharpenings = parse_float_list(args.sharpenings)
    gammas = parse_float_list(args.gammas)
    coverage_powers = parse_float_list(args.coverage_powers)

    # FSS cache for all (run, sharpening)
    fss_table: Dict[Tuple[int, float], float] = {}
    for i, row in enumerate(raw_data):
        for s in sharpenings:
            fss = fss_from_components(
                row["p_c_given_f"],
                row["h_norm"],
                row["support_mask"],
                row["nc"],
                entropy_sharpening=s,
            )
            fss_table[(i, s)] = float(np.nan_to_num(fss, nan=0.0, posinf=0.0, neginf=0.0))

    datasets = sorted({r["dataset"] for r in raw_data})
    base_idx = {
        r["dataset"]: i
        for i, r in enumerate(raw_data)
        if r["kind"] == "Base SAE"
    }
    adapted_idx: Dict[str, List[int]] = {
        ds: [
            i
            for i, r in enumerate(raw_data)
            if r["dataset"] == ds and r["kind"] != "Base SAE"
        ]
        for ds in datasets
    }

    total = len(alphas) * len(kappas) * len(sharpenings) * len(gammas) * len(coverage_powers)
    print(
        f"Sweeping {total} configs | alpha_fracs={len(alphas)} gammas={len(gammas)} "
        f"kappas={len(kappas)} sharpenings={len(sharpenings)} coverage_powers={len(coverage_powers)}"
    )

    sweep_rows = []
    best_score = -float("inf")
    best_params = None

    for alpha_frac, gamma, kappa, s, coverage_power in itertools.product(
        alphas, gammas, kappas, sharpenings, coverage_powers
    ):
        if gamma < 0.0 or gamma > 1.0:
            continue
        alpha = (1.0 - gamma) * alpha_frac
        beta = (1.0 - gamma) * (1.0 - alpha_frac)

        dams_scores = {}
        for i, row in enumerate(raw_data):
            css_norm = row["css_raw"] / (row["css_raw"] + kappa)
            fss = fss_table[(i, s)]
            coverage_gate = row["ec"] ** coverage_power if coverage_power > 0 else 1.0
            dams = coverage_gate * (alpha * css_norm + beta * fss + gamma * row["das"])
            dams_scores[i] = float(np.nan_to_num(dams, nan=0.0, posinf=0.0, neginf=0.0))

        per_ds_gap = {}
        gaps = []
        for ds in datasets:
            if ds not in base_idx or not adapted_idx.get(ds):
                continue
            base_score = dams_scores[base_idx[ds]]
            best_adapted = max(dams_scores[j] for j in adapted_idx[ds])
            gap = best_adapted - base_score
            per_ds_gap[ds] = gap
            gaps.append(gap)

        score = compute_objective(gaps, args.objective)
        if not np.isfinite(score):
            continue
        total_gap = float(np.sum(gaps)) if gaps else float("nan")
        mean_gap = float(np.mean(gaps)) if gaps else float("nan")
        min_gap = float(np.min(gaps)) if gaps else float("nan")

        row = {
            "alpha_frac": alpha_frac,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "kappa": kappa,
            "sharpening": s,
            "coverage_power": coverage_power,
            "objective": score,
            "total_gap": total_gap,
            "mean_gap": mean_gap,
            "min_gap": min_gap,
        }
        for ds, gap in per_ds_gap.items():
            row[f"gap_{ds}"] = gap

        for i, r in enumerate(raw_data):
            key = f"dams_{sanitize(r['kind'])}_{sanitize(r['dataset'])}_{sanitize(r['source_group'])}_{sanitize(r['run_id'])}_{i}"
            row[key] = dams_scores[i]

        sweep_rows.append(row)

        if score > best_score:
            best_score = score
            best_params = {
                "alpha_frac": alpha_frac,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "kappa": kappa,
                "sharpening": s,
                "coverage_power": coverage_power,
                "objective": score,
                "total_gap": total_gap,
                "mean_gap": mean_gap,
                "min_gap": min_gap,
            }

    if best_params is None:
        raise RuntimeError("Sweep produced no valid best_params.")

    # ----------------------- Step 3: Reporting -------------------------------
    print("\n" + "=" * 90)
    print("Step 3/3: Reporting")
    print("=" * 90)

    df = pd.DataFrame(sweep_rows)
    df_sorted = df.sort_values("objective", ascending=False)

    gap_cols = ["total_gap", "mean_gap", "min_gap"] + [
        c for c in df_sorted.columns if c.startswith("gap_")
    ]
    top_cols = [
        "alpha_frac",
        "alpha",
        "beta",
        "gamma",
        "kappa",
        "sharpening",
        "coverage_power",
        "objective",
    ] + gap_cols

    print("Top 20 configurations:")
    print(df_sorted[top_cols].head(20).to_string(index=False))

    print("\nBest params:")
    print(json.dumps(best_params, indent=2))

    print("\nPer-dataset winners at best params:")
    alpha = best_params["alpha"]
    beta = best_params["beta"]
    gamma = best_params["gamma"]
    kappa = best_params["kappa"]
    s = best_params["sharpening"]
    coverage_power = best_params["coverage_power"]

    best_dams_scores = {}
    for i, row in enumerate(raw_data):
        css_norm = row["css_raw"] / (row["css_raw"] + kappa)
        fss = fss_table[(i, s)]
        coverage_gate = row["ec"] ** coverage_power if coverage_power > 0 else 1.0
        best_dams_scores[i] = coverage_gate * (alpha * css_norm + beta * fss + gamma * row["das"])

    print(f"{'Dataset':<12} {'Base':>10} {'Best Adapted':>14} {'Gap':>10} {'Winner':>20}")
    print("-" * 80)
    for ds in datasets:
        if ds not in base_idx:
            continue

        b_idx = base_idx[ds]
        base_score = best_dams_scores[b_idx]

        if adapted_idx.get(ds):
            j = max(adapted_idx[ds], key=lambda i: best_dams_scores[i])
            adapted_score = best_dams_scores[j]
            gap = adapted_score - base_score
            winner = raw_data[j]["name"]
            print(f"{ds:<12} {base_score:>10.4f} {adapted_score:>14.4f} {gap:>10.4f} {winner:>20}")
        else:
            print(f"{ds:<12} {base_score:>10.4f} {'(none)':>14} {'nan':>10} {'-':>20}")

    print("\nSAE Utility Score (SUS): frozen readout usefulness")
    print(f"{'Dataset':<12} {'Base SUS':>10} {'Best SUS':>10} {'Delta':>10} {'Base bAcc':>10} {'Winner':>20}")
    print("-" * 86)
    utility_rows = []
    for ds in datasets:
        if ds not in base_idx:
            continue
        b_idx = base_idx[ds]
        base_utility = raw_data[b_idx].get("utility", 0.0)
        base_bacc = raw_data[b_idx].get("utility_balanced_acc", 0.0)

        if adapted_idx.get(ds):
            j = max(adapted_idx[ds], key=lambda i: raw_data[i].get("utility", 0.0))
            adapted_utility = raw_data[j].get("utility", 0.0)
            delta = adapted_utility - base_utility
            winner = raw_data[j]["name"]
        else:
            adapted_utility = float("nan")
            delta = float("nan")
            winner = "-"

        print(
            f"{ds:<12} {base_utility:>10.4f} {adapted_utility:>10.4f} "
            f"{delta:>10.4f} {base_bacc:>10.4f} {winner:>20}"
        )
        utility_rows.append(
            {
                "dataset": ds,
                "base_utility": base_utility,
                "base_balanced_acc": base_bacc,
                "base_chance": raw_data[b_idx].get("utility_chance", 0.0),
                "best_adapted_utility": adapted_utility,
                "utility_delta": delta,
                "best_adapted_name": winner,
            }
        )

    full_csv = args.output_dir / "dams_sweep_full.csv"
    top50_csv = args.output_dir / "dams_sweep_top50.csv"
    raw_json = args.output_dir / "dams_raw_metrics.json"
    utility_csv = args.output_dir / "sae_utility_summary.csv"

    df_sorted.to_csv(full_csv, index=False)
    df_sorted.head(50).to_csv(top50_csv, index=False)

    raw_export = []
    for row in raw_data:
        r = {
            k: v
            for k, v in row.items()
            if k not in {"p_c_given_f", "h_norm", "support_mask"}
        }
        raw_export.append(r)

    with raw_json.open("w") as f:
        json.dump(raw_export, f, indent=2)

    pd.DataFrame(utility_rows).to_csv(utility_csv, index=False)

    print("\nSaved:")
    print(f"- {full_csv}")
    print(f"- {top50_csv}")
    print(f"- {raw_json}")
    print(f"- {utility_csv}")
    print(f"- {manifest_path}")


if __name__ == "__main__":
    main()

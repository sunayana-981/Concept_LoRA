#!/usr/bin/env python3
"""
Compute Mahalanobis distance between ImageNet and benchmark datasets.

The script:
1. Extracts one embedding per image from a chosen backbone/layer.
2. Fits a Gaussian (mean + covariance) on a reference dataset (default: ImageNet).
3. Computes Mahalanobis distances for each benchmark dataset to that reference.

Example:
    python compute_mahalanobis.py \
        --model clip_vit_b16 \
        --reference imagenet \
        --targets caltech101 dtd eurosat \
        --max_reference_samples 5000 \
        --max_target_samples 2000 \
        --output_dir results/mahalanobis
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.dataset_registry import load_registry
from src.models.registry import get_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Mahalanobis distance from ImageNet to benchmark datasets."
    )
    parser.add_argument("--model", default="clip_vit_b16",
                        help="Model config name from configs/models (e.g. clip_vit_b16)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Backbone layer index for embedding extraction")

    parser.add_argument("--reference", default="imagenet",
                        help="Reference dataset name from configs/datasets/registry.yaml")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Target benchmark datasets. Default: all datasets except reference.")
    parser.add_argument("--registry", default=None,
                        help="Optional custom dataset registry YAML path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sub-sampling")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction")
    parser.add_argument("--max_reference_samples", type=int, default=5000,
                        help="Max samples used to fit reference Gaussian")
    parser.add_argument("--max_target_samples", type=int, default=2000,
                        help="Max samples used per target dataset")
    parser.add_argument("--streaming_hf", action="store_true",
                        help="Use streaming=True for HuggingFace datasets (ignored for local imagefolder paths)")

    parser.add_argument("--cov_eps", type=float, default=1e-5,
                        help="Diagonal regularization added to covariance")
    parser.add_argument("--output_dir", default="results/mahalanobis",
                        help="Directory for JSON/CSV outputs")

    return parser.parse_args()


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def to_pil_image(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return ensure_rgb(x)

    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
            arr = arr.permute(1, 2, 0)
        if arr.dtype != torch.uint8:
            arr = arr.clamp(0, 255).to(torch.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        arr = arr.numpy()
        return ensure_rgb(Image.fromarray(arr))

    if hasattr(x, "ndim") and hasattr(x, "shape") and hasattr(x, "astype"):
        arr = x
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if str(arr.dtype) != "uint8":
            arr = arr.clip(0, 255).astype("uint8")
        return ensure_rgb(Image.fromarray(arr))

    if isinstance(x, dict) and "path" in x and x["path"] is not None:
        return ensure_rgb(Image.open(x["path"]))

    raise TypeError(f"Unsupported image format: {type(x)}")


def find_image_key(dataset) -> str:
    if not hasattr(dataset, "features") or dataset.features is None:
        raise ValueError("Dataset has no static features; cannot infer image key.")

    if "image" in dataset.features:
        return "image"
    for key, feat in dataset.features.items():
        if feat.__class__.__name__.lower() == "image":
            return key
    raise ValueError(f"Could not find image column. Features: {list(dataset.features.keys())}")


def find_image_key_from_item(item: dict) -> str:
    if "image" in item:
        return "image"
    for k, v in item.items():
        if isinstance(v, Image.Image):
            return k
        if isinstance(v, dict) and ("path" in v or "bytes" in v):
            return k
    raise ValueError(f"Could not infer image key from item keys: {list(item.keys())}")


def maybe_subsample(dataset, max_samples: Optional[int], seed: int):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def load_dataset_entry(
    dataset_name: str,
    registry: dict,
    streaming_hf: bool,
):
    cfg = registry[dataset_name]
    effective_split = cfg.get("split", "train")
    local = cfg.get("local_path")

    if local and os.path.isdir(local):
        ds = load_dataset(
            "imagefolder",
            data_dir=local,
            split=effective_split,
            trust_remote_code=cfg.get("trust_remote_code", False),
        )
        return ds, False

    kwargs = {
        "path": cfg["hf_path"],
        "split": effective_split,
        "trust_remote_code": cfg.get("trust_remote_code", False),
        "streaming": streaming_hf,
    }
    if "hf_name" in cfg:
        kwargs["name"] = cfg["hf_name"]
    ds = load_dataset(**kwargs)
    if isinstance(ds, dict):
        ds = ds[effective_split]
    return ds, bool(streaming_hf)


def collect_features(
    backbone,
    dataset_name: str,
    registry: dict,
    layer: int,
    batch_size: int,
    max_samples: Optional[int],
    seed: int,
    streaming_hf: bool = False,
) -> torch.Tensor:
    ds, is_streaming = load_dataset_entry(
        dataset_name=dataset_name,
        registry=registry,
        streaming_hf=streaming_hf,
    )

    feats: List[torch.Tensor] = []

    if not is_streaming:
        ds = maybe_subsample(ds, max_samples=max_samples, seed=seed)
        image_key = find_image_key(ds)

        n = len(ds)
        pbar = tqdm(range(0, n, batch_size), desc=f"[{dataset_name}] extract", leave=False)
        for start in pbar:
            end = min(start + batch_size, n)
            batch = ds[start:end]
            images = [to_pil_image(x) for x in batch[image_key]]
            with torch.no_grad():
                z = backbone.get_activations(images, layer=layer, use_cls=True)
            feats.append(z.detach().cpu().float())
        return torch.cat(feats, dim=0)

    image_key = None
    collected = 0
    images_buf: List[Image.Image] = []
    it = iter(ds)
    pbar = tqdm(
        total=max_samples,
        desc=f"[{dataset_name}] stream_extract",
        leave=False,
    )

    while True:
        if max_samples is not None and collected >= max_samples:
            break
        try:
            item = next(it)
        except StopIteration:
            break

        if image_key is None:
            try:
                image_key = find_image_key(ds)
            except Exception:
                image_key = find_image_key_from_item(item)

        images_buf.append(to_pil_image(item[image_key]))
        collected += 1
        pbar.update(1)

        if len(images_buf) == batch_size:
            with torch.no_grad():
                z = backbone.get_activations(images_buf, layer=layer, use_cls=True)
            feats.append(z.detach().cpu().float())
            images_buf = []

    if images_buf:
        with torch.no_grad():
            z = backbone.get_activations(images_buf, layer=layer, use_cls=True)
        feats.append(z.detach().cpu().float())

    pbar.close()
    if not feats:
        raise RuntimeError(f"No features collected for dataset '{dataset_name}'")
    return torch.cat(feats, dim=0)


def fit_gaussian(x: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.double()
    mu = x.mean(dim=0)
    xc = x - mu
    denom = max(int(x.shape[0]) - 1, 1)
    cov = (xc.T @ xc) / float(denom)
    cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype)

    try:
        chol = torch.linalg.cholesky(cov)
        cov_inv = torch.cholesky_inverse(chol)
    except RuntimeError:
        cov_inv = torch.linalg.pinv(cov)

    return mu, cov_inv


def mahalanobis_distances(x: torch.Tensor, mu: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
    d = x.double() - mu
    md2 = torch.einsum("nd,dd,nd->n", d, cov_inv, d)
    return torch.sqrt(torch.clamp(md2, min=0.0)).float()


def summarize_distances(values: torch.Tensor) -> Dict[str, float]:
    q = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype))
    return {
        "num_samples": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "q25": float(q[0].item()),
        "median": float(q[1].item()),
        "q75": float(q[2].item()),
        "max": float(values.max().item()),
    }


def mean_to_reference_distance(x: torch.Tensor, mu_ref: torch.Tensor, cov_inv_ref: torch.Tensor) -> float:
    mu_x = x.double().mean(dim=0, keepdim=True)
    return float(mahalanobis_distances(mu_x, mu_ref, cov_inv_ref).item())


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable, falling back to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)

    registry = load_registry(args.registry)
    if args.reference not in registry:
        raise ValueError(f"Reference dataset '{args.reference}' not found in registry.")

    target_names = args.targets if args.targets is not None else [
        name for name in registry.keys() if name != args.reference
    ]

    backbone = get_backbone(args.model, device=args.device).load()

    print(f"Model: {args.model}")
    print(f"Reference dataset: {args.reference}")
    print(f"Targets: {target_names}")
    print(f"Layer: {args.layer}")
    print("")

    ref_feats = collect_features(
        backbone=backbone,
        dataset_name=args.reference,
        registry=registry,
        layer=args.layer,
        batch_size=args.batch_size,
        max_samples=args.max_reference_samples,
        seed=args.seed,
        streaming_hf=args.streaming_hf,
    )
    print(f"Reference features: {tuple(ref_feats.shape)}")

    ref_mu, ref_cov_inv = fit_gaussian(ref_feats, eps=args.cov_eps)
    ref_self_md = mahalanobis_distances(ref_feats, ref_mu, ref_cov_inv)
    ref_self_summary = summarize_distances(ref_self_md)
    print(
        f"Reference self-distance: mean={ref_self_summary['mean']:.4f}, "
        f"std={ref_self_summary['std']:.4f}"
    )

    rows = []
    for target in target_names:
        if target not in registry:
            print(f"[skip] {target}: not found in registry")
            continue
        try:
            tgt_feats = collect_features(
                backbone=backbone,
                dataset_name=target,
                registry=registry,
                layer=args.layer,
                batch_size=args.batch_size,
                max_samples=args.max_target_samples,
                seed=args.seed,
                streaming_hf=args.streaming_hf,
            )
        except Exception as exc:
            print(f"[skip] {target}: failed to load/extract ({exc})")
            continue

        md = mahalanobis_distances(tgt_feats, ref_mu, ref_cov_inv)
        summary = summarize_distances(md)
        summary["dataset"] = target
        summary["distance_of_target_mean"] = mean_to_reference_distance(
            tgt_feats, ref_mu, ref_cov_inv
        )
        rows.append(summary)

    rows.sort(key=lambda x: x["mean"], reverse=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mahalanobis_summary.json"
    csv_path = output_dir / "mahalanobis_summary.csv"

    payload = {
        "model": args.model,
        "device": args.device,
        "layer": args.layer,
        "reference_dataset": args.reference,
        "reference_num_samples": int(ref_feats.shape[0]),
        "cov_eps": args.cov_eps,
        "reference_self_summary": ref_self_summary,
        "results": rows,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "dataset",
        "num_samples",
        "mean",
        "std",
        "min",
        "q25",
        "median",
        "q75",
        "max",
        "distance_of_target_mean",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print("")
    print("Mahalanobis summary (higher means farther from reference):")
    for row in rows:
        print(
            f"  {row['dataset']:<18} mean={row['mean']:.4f} "
            f"std={row['std']:.4f} mean_to_ref={row['distance_of_target_mean']:.4f}"
        )

    print("")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote CSV : {csv_path}")


if __name__ == "__main__":
    main()

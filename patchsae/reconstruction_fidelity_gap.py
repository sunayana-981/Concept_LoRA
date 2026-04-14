#!/usr/bin/env python3
"""
1A. Reconstruction Fidelity Gap Analysis

Applies both SAEs (ImageNet-SAE and Target-SAE) to the same target-domain activations
and compares reconstruction quality.

Metrics reported per dataset:
  - MSE / L2 loss        (ImageNet-SAE ↑, Target-SAE ↓)
  - SSIM                 (ImageNet-SAE ↓, Target-SAE ↑)
  - Pearson correlation  (ImageNet-SAE low, Target-SAE high)

For a fair comparison both SAEs are applied to activations extracted at the
TARGET SAE's trained layer.  The base ImageNet-SAE is also evaluated at its own
trained layer to establish its in-distribution baseline.

Usage (run from patchsae/):
    python reconstruction_fidelity_gap.py --dataset eurosat
    python reconstruction_fidelity_gap.py --dataset caltech101 medmnist
    python reconstruction_fidelity_gap.py --all
    python reconstruction_fidelity_gap.py --all --max_batches 20 --batch_size 64
    python reconstruction_fidelity_gap.py --all --save_json out/recon_fidelity_gap.json
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import CLIPModel, CLIPProcessor

from src.sae_training.config import Config
from src.sae_training.hooked_vit import HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.utils import get_model_activations, process_model_inputs

# Inlined from tasks/utils.py to avoid the yacs / MaPLe import chain
DATASET_INFO = {
    "imagenet": {
        "path": "evanarlian/imagenet_1k_resized_256",
        "split": "train",
        "trust_remote_code": True,
    },
    "caltech101": {
        "path": "HuggingFaceM4/Caltech-101",
        "split": "train",
        "name": "with_background_category",
        "trust_remote_code": True,
    },
    "eurosat": {
        "path": "imagefolder",
        "data_dir": str(PROJECT_ROOT / "../data/eurosat/2750"),
        "split": "train",
        "trust_remote_code": True,
    },
    "medmnist": {
        "path": "imagefolder",
        "data_dir": str(PROJECT_ROOT / "../data/pathmnist_imagefolder"),
        "split": "train",
        "trust_remote_code": True,
    },
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_SAE_PATH = str(PROJECT_ROOT / "data/sae_weight/base/out.pt")
CKPT_ROOT     = str(PROJECT_ROOT / "out/checkpoints")

DATASETS = ["eurosat", "caltech101", "medmnist"]


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_target_sae_paths(dataset: str) -> List[str]:
    """Return all final-checkpoint SAE paths for a dataset (non-maple only)."""
    pattern = os.path.join(CKPT_ROOT, dataset, "*", "final_*", "*.pt")
    return sorted(glob.glob(pattern))


# ---------------------------------------------------------------------------
# Model / SAE loading
# ---------------------------------------------------------------------------

def load_sae_safe(path: str, device: str) -> Tuple[SparseAutoencoder, object]:
    """Load an SAE checkpoint regardless of cfg key name (cfg vs config)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    raw_cfg = ckpt.get("cfg", ckpt.get("config"))
    if raw_cfg is None:
        raise ValueError(f"No cfg/config key in {path}")

    cfg = Config(raw_cfg) if isinstance(raw_cfg, dict) else Config(raw_cfg.__dict__)
    sae = SparseAutoencoder(cfg, device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(device)
    return sae, cfg


def load_hooked_vit(model_name: str, device: str) -> HookedVisionTransformer:
    """Load a base CLIP model wrapped in HookedVisionTransformer."""
    model     = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return HookedVisionTransformer(model, processor, device=device)


# ---------------------------------------------------------------------------
# Metric accumulators  (online / streaming, no giant tensors in RAM)
# ---------------------------------------------------------------------------

class MetricAccumulator:
    """Running sums for MSE, normalized-L2, SSIM, Pearson r."""

    def __init__(self):
        self.mse_sum   = 0.0
        self.nl2_sum   = 0.0
        self.ssim_sum  = 0.0
        self.pearson_sum = 0.0
        self.count     = 0      # number of *samples* (not tokens)

    # ------------------------------------------------------------------
    # The three metric helpers all work on a flat [N, D] pair.
    # We reduce per-sample then sum — safe to add across batches.
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten2d(t: torch.Tensor) -> torch.Tensor:
        """[N, ...] → [N, D]"""
        return t.reshape(t.shape[0], -1)

    def update(self, orig: torch.Tensor, recon: torch.Tensor):
        """
        orig, recon: CPU tensors, shape [B, d_in] or [B, seq, d_in].
        All per-sample metrics are computed on the flattened view [B, D].
        """
        N  = orig.shape[0]
        ox = self._flatten2d(orig.float())
        rx = self._flatten2d(recon.float())

        # MSE: mean over all elements
        self.mse_sum  += F.mse_loss(rx, ox).item() * N

        # Normalised L2: ||recon - orig|| / ||orig||  per sample
        diff_norm = (rx - ox).norm(dim=-1)
        orig_norm = ox.norm(dim=-1).clamp(min=1e-8)
        self.nl2_sum += (diff_norm / orig_norm).sum().item()

        # SSIM (per-sample, treating D-dim vector as 1-D signal)
        eps = 1e-8
        data_range = ox.amax(dim=-1, keepdim=True) - ox.amin(dim=-1, keepdim=True)
        data_range = data_range.clamp(min=eps)
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        mu_x  = ox.mean(dim=-1, keepdim=True)
        mu_y  = rx.mean(dim=-1, keepdim=True)
        x_c, y_c = ox - mu_x, rx - mu_y
        sigma_x  = (x_c ** 2).mean(dim=-1, keepdim=True)
        sigma_y  = (y_c ** 2).mean(dim=-1, keepdim=True)
        sigma_xy = (x_c * y_c).mean(dim=-1, keepdim=True)

        num  = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den  = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
        self.ssim_sum += ((num / den.clamp(min=eps)).mean(dim=-1)).sum().item()

        # Pearson r per sample
        xc = ox - ox.mean(dim=-1, keepdim=True)
        yc = rx - rx.mean(dim=-1, keepdim=True)
        denom = (xc.norm(dim=-1) * yc.norm(dim=-1)).clamp(min=eps)
        self.pearson_sum += ((xc * yc).sum(dim=-1) / denom).sum().item()

        self.count += N

    def result(self) -> Dict[str, float]:
        n = max(self.count, 1)
        return {
            "mse":           self.mse_sum   / n,
            "normalized_l2": self.nl2_sum   / n,
            "ssim":          self.ssim_sum  / n,
            "pearson_r":     self.pearson_sum / n,
        }


# ---------------------------------------------------------------------------
# SAE reconstruction helper  (memory-bounded, token-chunk loop)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sae_reconstruct_chunks(
    sae: SparseAutoencoder,
    acts: torch.Tensor,       # [B, d_in] or [B, seq, d_in]  — on CPU
    device: str,
    token_chunk: int = 2048,  # max tokens per GPU forward
) -> torch.Tensor:
    """
    Run `sae` on `acts` in GPU chunks of at most `token_chunk` tokens.
    Returns reconstructed tensor of the same shape, on CPU.
    """
    shape = acts.shape                      # e.g. (32, 197, 768) or (32, 768)
    flat  = acts.reshape(-1, shape[-1])     # [N_tokens, d_in]
    N_tok = flat.shape[0]

    out_parts = []
    for start in range(0, N_tok, token_chunk):
        chunk_gpu = flat[start:start + token_chunk].to(device)
        recon_gpu, _, _ = sae(chunk_gpu)
        out_parts.append(recon_gpu.cpu())

    recon_flat = torch.cat(out_parts, dim=0)        # [N_tokens, d_in]
    return recon_flat.reshape(shape)                 # restore original shape


# ---------------------------------------------------------------------------
# Streaming extract-and-evaluate  (one dataset pass for both SAEs)
# ---------------------------------------------------------------------------

@torch.no_grad()
def stream_and_evaluate(
    vit: HookedVisionTransformer,
    sae_a: SparseAutoencoder,
    sae_b: SparseAutoencoder,
    dataset: str,
    block_layer: int,
    module_name: str,
    class_token: bool,
    device: str,
    batch_size: int = 32,
    max_batches: Optional[int] = None,
    token_chunk: int = 2048,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """
    Single pass over the dataset.

    For each image-batch:
      1. Extract CLIP activations at (block_layer, module_name).
      2. Run sae_a and sae_b on them (in token-chunks to stay in VRAM budget).
      3. Accumulate per-sample metrics on CPU.

    Returns (metrics_a, metrics_b, n_samples).
    """
    from datasets import load_dataset

    ds_info = DATASET_INFO.get(dataset)
    if ds_info is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    hf_ds = load_dataset(**ds_info)
    if isinstance(hf_ds, dict):
        hf_ds = hf_ds["train"]

    acc_a, acc_b = MetricAccumulator(), MetricAccumulator()
    batch_imgs   = []
    n_batches    = 0

    for item in tqdm(hf_ds, desc=f"  layer {block_layer}", leave=False):
        batch_imgs.append(item["image"])

        if len(batch_imgs) == batch_size:
            inputs = vit.processor(
                images=batch_imgs, text="", return_tensors="pt", padding=True
            ).to(device)
            acts_gpu = get_model_activations(
                vit, inputs, block_layer, module_name, class_token
            )
            acts_cpu = acts_gpu.cpu()
            del acts_gpu

            recon_a = _sae_reconstruct_chunks(sae_a, acts_cpu, device, token_chunk)
            recon_b = _sae_reconstruct_chunks(sae_b, acts_cpu, device, token_chunk)

            acc_a.update(acts_cpu, recon_a)
            acc_b.update(acts_cpu, recon_b)

            batch_imgs = []
            n_batches += 1

            if max_batches and n_batches >= max_batches:
                break

    # Remaining partial batch
    if batch_imgs and not (max_batches and n_batches >= max_batches):
        inputs = vit.processor(
            images=batch_imgs, text="", return_tensors="pt", padding=True
        ).to(device)
        acts_gpu = get_model_activations(
            vit, inputs, block_layer, module_name, class_token
        )
        acts_cpu = acts_gpu.cpu()
        del acts_gpu

        recon_a = _sae_reconstruct_chunks(sae_a, acts_cpu, device, token_chunk)
        recon_b = _sae_reconstruct_chunks(sae_b, acts_cpu, device, token_chunk)

        acc_a.update(acts_cpu, recon_a)
        acc_b.update(acts_cpu, recon_b)

    return acc_a.result(), acc_b.result(), acc_a.count


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def run_dataset(
    dataset: str,
    device: str,
    batch_size: int = 64,
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Full reconstruction fidelity gap analysis for one dataset.
    Returns a dict with results for base SAE and each target SAE.
    """
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*60}")

    # ── 1. Load base ImageNet SAE ──────────────────────────────────────────
    print("\n[1/3] Loading base ImageNet SAE …")
    base_sae, base_cfg = load_sae_safe(BASE_SAE_PATH, device)
    base_layer  = int(base_cfg.block_layer)
    base_module = str(base_cfg.module_name)
    base_class_token = bool(base_cfg.class_token) if base_cfg.class_token is not None else False
    print(f"      layer={base_layer}, module={base_module}, "
          f"class_token={base_class_token}, d_sae={base_sae.d_sae}")

    # ── 2. Find target SAE checkpoints ────────────────────────────────────
    target_paths = find_target_sae_paths(dataset)
    if not target_paths:
        print(f"  [WARN] No target SAE checkpoints found for '{dataset}'. Skipping.")
        return {}

    print(f"\n[2/3] Found {len(target_paths)} target SAE checkpoint(s).")
    for p in target_paths:
        print(f"      {p}")

    # ── 3. Load CLIP ViT ───────────────────────────────────────────────────
    vit_name = getattr(base_cfg, "model_name", "openai/clip-vit-base-patch16")
    print(f"\n[3/3] Loading HookedViT ({vit_name}) …")
    vit = load_hooked_vit(vit_name, device)

    results = {"dataset": dataset, "base_sae_path": BASE_SAE_PATH, "layers": {}}

    # ── For each target SAE at its trained layer ───────────────────────────
    processed_layers = set()
    for t_path in target_paths:
        t_sae, t_cfg = load_sae_safe(t_path, device)
        t_layer       = int(t_cfg.block_layer)
        t_module      = str(t_cfg.module_name)
        t_class_token = bool(t_cfg.class_token) if t_cfg.class_token is not None else False

        if t_layer in processed_layers:
            del t_sae
            continue  # Only one (final) checkpoint per layer
        processed_layers.add(t_layer)

        print(f"\n--- Layer {t_layer} | target SAE d_sae={t_sae.d_sae} ---")
        print(f"  Streaming dataset → extract activations + evaluate both SAEs …")

        b_metrics, t_metrics, n_samples = stream_and_evaluate(
            vit=vit,
            sae_a=base_sae,
            sae_b=t_sae,
            dataset=dataset,
            block_layer=t_layer,
            module_name=t_module,
            class_token=t_class_token,
            device=device,
            batch_size=batch_size,
            max_batches=max_batches,
        )
        print(f"  Done — {n_samples} samples processed.")

        results["layers"][t_layer] = {
            "target_sae_path": t_path,
            "n_samples":       n_samples,
            "imagenet_sae":    b_metrics,
            "target_sae":      t_metrics,
        }
        del t_sae

    # ── Also evaluate base SAE at its own layer (in-distribution baseline) ─
    if base_layer not in processed_layers:
        print(f"\n--- Layer {base_layer} | Base SAE in-distribution baseline ---")
        print(f"  Streaming dataset → extract + evaluate ImageNet-SAE at its trained layer …")

        # Use a dummy second SAE that is the same as the base (cheapest option)
        b_own, _, n_samples = stream_and_evaluate(
            vit=vit,
            sae_a=base_sae,
            sae_b=base_sae,   # same model — only the first result matters
            dataset=dataset,
            block_layer=base_layer,
            module_name=base_module,
            class_token=base_class_token,
            device=device,
            batch_size=batch_size,
            max_batches=max_batches,
        )
        print(f"  Done — {n_samples} samples processed.")
        results["layers"][base_layer] = {
            "note": "base SAE trained layer (no matching target SAE)",
            "n_samples": n_samples,
            "imagenet_sae": b_own,
            "target_sae": None,
        }

    # Clean up GPU memory
    del vit, base_sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def print_table(results: Dict):
    """Print a formatted comparison table for one dataset's results."""
    dataset = results.get("dataset", "?")
    print(f"\n{'='*70}")
    print(f"  RECONSTRUCTION FIDELITY GAP — {dataset.upper()}")
    print(f"{'='*70}")

    metric_names = {
        "mse":           "MSE",
        "normalized_l2": "Norm. L2",
        "ssim":          "SSIM",
        "pearson_r":     "Pearson r",
    }

    for layer, data in sorted(results.get("layers", {}).items()):
        n = data.get("n_samples", "?")
        note = data.get("note", "")
        print(f"\n  Layer {layer}  (n={n})  {note}")
        print(f"  {'Metric':<18} {'ImageNet-SAE':>15} {'Target-SAE':>15} {'Gap':>12}")
        print(f"  {'-'*60}")

        b_met = data.get("imagenet_sae", {})
        t_met = data.get("target_sae", {})

        for key, label in metric_names.items():
            b_val = b_met.get(key)
            t_val = t_met.get(key) if t_met else None

            b_str = f"{b_val:.6f}" if b_val is not None else "     n/a"
            t_str = f"{t_val:.6f}" if t_val is not None else "     n/a"

            if b_val is not None and t_val is not None:
                gap   = b_val - t_val
                g_str = f"{gap:+.6f}"
                # For MSE / L2: positive gap = ImageNet-SAE worse (expected)
                # For SSIM / Pearson: negative gap = ImageNet-SAE worse (expected)
                if key in ("mse", "normalized_l2"):
                    tag = "✓ (expected)" if gap > 0 else "✗"
                else:
                    tag = "✓ (expected)" if gap < 0 else "✗"
            else:
                g_str = "       n/a"
                tag   = ""

            print(f"  {label:<18} {b_str:>15} {t_str:>15} {g_str:>12}  {tag}")

    print()


def print_summary_table(all_results: List[Dict]):
    """Print one-row-per-dataset summary across all datasets."""
    print(f"\n{'='*90}")
    print("  SUMMARY — mean metrics over all evaluated layers")
    print(f"{'='*90}")
    print(f"  {'Dataset':<14} {'Metric':<18} {'ImageNet-SAE':>15} {'Target-SAE':>15} {'Gap':>12}")
    print(f"  {'-'*74}")

    for res in all_results:
        ds = res.get("dataset", "?")
        layers_with_target = {
            l: d for l, d in res.get("layers", {}).items()
            if d.get("target_sae") is not None
        }
        if not layers_with_target:
            continue

        for metric in ("mse", "ssim", "pearson_r"):
            b_vals = [d["imagenet_sae"][metric] for d in layers_with_target.values()
                      if d.get("imagenet_sae") and metric in d["imagenet_sae"]]
            t_vals = [d["target_sae"][metric] for d in layers_with_target.values()
                      if d.get("target_sae") and metric in d["target_sae"]]

            if not b_vals or not t_vals:
                continue

            b_mean = float(np.mean(b_vals))
            t_mean = float(np.mean(t_vals))
            gap    = b_mean - t_mean

            label = {"mse": "MSE", "ssim": "SSIM", "pearson_r": "Pearson r"}[metric]
            print(f"  {ds:<14} {label:<18} {b_mean:>15.6f} {t_mean:>15.6f} {gap:>+12.6f}")

        print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAE Reconstruction Fidelity Gap (1A)")
    parser.add_argument("--dataset",    nargs="+", choices=DATASETS,
                        help="Dataset(s) to evaluate")
    parser.add_argument("--all",        action="store_true",
                        help="Evaluate all datasets")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Images per batch when extracting activations (default: 64)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Cap number of batches (useful for quick sanity checks)")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", default=None,
                        help="Optional path to save results as JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = DATASETS if args.all else (args.dataset or [])
    if not datasets:
        print("Specify --dataset <name> or --all")
        sys.exit(1)

    print(f"Device: {args.device}")
    print(f"Datasets: {datasets}")
    print(f"Batch size: {args.batch_size}")
    if args.max_batches:
        print(f"Max batches per dataset: {args.max_batches}")

    all_results = []
    for ds in datasets:
        res = run_dataset(
            ds,
            device=args.device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
        if res:
            all_results.append(res)
            print_table(res)

    if len(all_results) > 1:
        print_summary_table(all_results)

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    main()

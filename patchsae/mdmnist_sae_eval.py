#!/usr/bin/env python3
"""
Evaluate SAE feature quality on MedMNIST via zero-shot CLIP classification
with top-k SAE feature masking — NO classifier training required.

For each SAE this script:
  1. Computes per-class SAE activation profiles (which features fire per class)
  2. Runs CLIP zero-shot classification WITHOUT SAE intervention  (baseline)
  3. Clamps the top-k class features ON  → does accuracy increase?
  4. Clamps the top-k class features OFF → does accuracy drop?

If the SAE has learned meaningful class-specific features, clamping ON should
help and clamping OFF should hurt.  This is the same protocol as classify.py
but applied per-SAE for comparison.

All SAEs receive LoRA fine-tuned CLIP activations.

Usage (from patchsae root):
    python eval_medmnist_sae.py --include_base
    python eval_medmnist_sae.py                    # medmnist SAEs only
    python eval_medmnist_sae.py --reuse_activations # reuse cached cls_sae_cnt
"""

import argparse
import json
import os
import sys
import glob
import gc
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
try:
    from tasks.utils import (
        load_sae, load_hooked_vit, process_batch,
        load_and_organize_dataset, SAE_DIM,
    )
    from tasks.train_sae_lora_clip import load_lora_weights
    from src.sae_training.config import Config
    from src.sae_training.hooked_vit import Hook
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] Could not import project modules: {e}")
    print("Run from patchsae root:  python eval_medmnist_sae.py")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

BASE_SAE_PATH = "data/sae_weight/base/out.pt"
MEDMNIST_SAE_DIR = "out/checkpoints/medmnist"
LORA_CHECKPOINT_PATH = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
DATASET_ROOT = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
CLASSNAMES_PATH = "configs/classnames/medmnist_classnames.json"
EVAL_CACHE_DIR = "out/medmnist_eval_cache"

BACKBONE = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
VAL_SPLIT = 0.15
SEED = 42

# Top-k values to test (same as classify.py)
TOPK_LIST = [1, 2, 5, 10, 50, 100, 500, 1000, 2000]

# SAE bias from classify.py (used when reconstructing activations)
SAE_BIAS = -0.105131256516992


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def discover_medmnist_saes(base_dir):
    """Find final SAE checkpoints, keeping only the latest per layer."""
    pattern = os.path.join(base_dir, "*/final*/*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[WARN] No SAE checkpoints found matching {pattern}")
        return []
    best_per_layer = {}
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        cfg_data = ckpt.get("cfg", ckpt.get("config"))
        if hasattr(cfg_data, "__dict__"):
            layer = cfg_data.block_layer
        elif isinstance(cfg_data, dict):
            layer = cfg_data.get("block_layer", "?")
        else:
            layer = "?"
        best_per_layer[layer] = path
        del ckpt
        gc.collect()
    return list(best_per_layer.values())


def load_selected_saes(base_sae_path, medmnist_sae_paths, device, include_base=False):
    """Load SAEs. Returns list of (sae, cfg, label) tuples."""
    saes = []
    if include_base and os.path.exists(base_sae_path):
        sae, cfg = load_sae(base_sae_path, device)
        layer = cfg.block_layer if hasattr(cfg, "block_layer") else -2
        label = f"base_layer{layer}"
        print(f"  [OK] Base SAE: layer={layer}, d_sae={cfg.d_sae}")
        saes.append((sae, cfg, label))
    elif include_base:
        print(f"  [SKIP] Base SAE not found: {base_sae_path}")

    for path in medmnist_sae_paths:
        sae, cfg = load_sae(path, device)
        layer = cfg.block_layer if hasattr(cfg, "block_layer") else "?"
        label = f"medmnist_layer{layer}"
        print(f"  [OK] MedMNIST SAE: layer={layer}, d_sae={cfg.d_sae}")
        print(f"        path: {path}")
        saes.append((sae, cfg, label))
    return saes


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# TEXT FEATURES  (from classify.py)
# ═════════════════════════════════════════════════════════════════════════════

def calculate_text_features(vit, device, classnames):
    """
    Compute mean CLIP text features across 80 OpenAI ImageNet templates
    for each class.  Returns [num_classes, d_model] normalized tensor.
    """
    mean_text_features = 0

    for template_fn in openai_imagenet_template:
        prompts = [template_fn(c) for c in classnames]
        prompt_ids = [
            vit.processor(
                text=p, return_tensors="pt", padding=False, truncation=True
            ).input_ids[0]
            for p in prompts
        ]
        padded = pad_sequence(prompt_ids, batch_first=True).to(device)

        with torch.no_grad():
            text_features = vit.model.get_text_features(padded)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features += text_features

    mean_text_features = mean_text_features / len(openai_imagenet_template)
    mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
    return mean_text_features


# ═════════════════════════════════════════════════════════════════════════════
# PER-CLASS SAE ACTIVATION PROFILES
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_cls_sae_activations(vit, sae, cfg, dataloader, num_classes, device):
    """
    Compute per-class mean SAE feature activations.

    Returns:
        cls_sae_cnt: np.ndarray [num_classes, d_sae]
            Mean activation of each SAE feature for each class.
    """
    d_sae = cfg.d_sae
    layer = cfg.block_layer if hasattr(cfg, "block_layer") else -2
    module = cfg.module_name if hasattr(cfg, "module_name") else "resid"

    cls_sae_sum = np.zeros((num_classes, d_sae), dtype=np.float64)
    cls_count = np.zeros(num_classes, dtype=np.int64)

    to_pil = transforms.ToPILImage()

    for images, labels in tqdm(dataloader, desc="Computing cls_sae_cnt"):
        bs = images.size(0)

        inputs = vit.processor(
            images=[to_pil(img) for img in images],
            text="",
            return_tensors="pt",
            padding=True,
        ).to(device)

        _, cache = vit.run_with_cache([(layer, module)], **inputs)
        activations = cache[(layer, module)]  # [B, seq_len, d_in]
        cls_acts = activations[:, 0, :]       # [B, d_in]

        _, sae_cache = sae.run_with_cache(cls_acts)
        feat_acts = sae_cache["hook_hidden_post"]  # [B, d_sae]
        feat_np = feat_acts.cpu().float().numpy()

        for i in range(bs):
            c = labels[i].item()
            cls_sae_sum[c] += feat_np[i]
            cls_count[c] += 1

        del cache, activations, cls_acts, sae_cache, feat_acts
        if bs > 0 and (cls_count.sum() % (50 * BATCH_SIZE)) < bs:
            flush_gpu()

    # Mean activation per class
    for c in range(num_classes):
        if cls_count[c] > 0:
            cls_sae_sum[c] /= cls_count[c]

    return cls_sae_sum


# ═════════════════════════════════════════════════════════════════════════════
# SAE HOOKS FOR CLAMPING  (from classify.py)
# ═════════════════════════════════════════════════════════════════════════════

def create_sae_hooks(cfg, cls_features, sae, device, hook_type="on"):
    """
    Create hooks that clamp specific SAE features ON or OFF.

    hook_type="on":  set selected features to 1, rest to 0
    hook_type="off": set selected features to 0, rest to 1
    """
    d_sae = cfg.d_sae
    clamp_feat_dim = torch.ones(d_sae).bool()

    if hook_type == "on":
        clamp_value = torch.zeros(d_sae, device=device)
        for f in cls_features:
            clamp_value[f] = 1.0
    else:  # off
        clamp_value = torch.ones(d_sae, device=device)
        for f in cls_features:
            clamp_value[f] = 0.0

    def hook_fn(activations):
        activations[:, :, :] = (
            sae.forward_clamp(
                activations[:, :, :],
                clamp_feat_dim=clamp_feat_dim,
                clamp_value=clamp_value,
            )[0]
            - SAE_BIAS
        )
        return (activations,)

    return [
        Hook(
            cfg.block_layer,
            cfg.module_name,
            hook_fn,
            return_module_output=False,
            is_custom=False,
        )
    ]


# ═════════════════════════════════════════════════════════════════════════════
# ZERO-SHOT CLASSIFICATION WITH TOP-K MASKING
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_predictions(vit, inputs, text_features, hooks=None):
    """Run ViT with optional hooks and return predicted class indices."""
    if hooks:
        vit_out = vit.run_with_hooks(hooks, return_type="output", **inputs)
    else:
        vit_out = vit(return_type="output", **inputs)

    image_features = vit_out.image_embeds
    logit_scale = vit.model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.t()
    preds = logits.argmax(dim=-1)
    return preds.cpu().numpy().tolist()


def evaluate_topk_masking(vit, sae, cfg, cls_sae_cnt, text_features,
                          dataloader, num_classes, classnames, device,
                          topk_list=TOPK_LIST):
    """
    Full top-k masking evaluation for one SAE.

    For each class c and each k in topk_list:
      - Find the k SAE features with highest mean activation for class c
      - Clamp those features ON  → measure accuracy on class c images
      - Clamp those features OFF → measure accuracy on class c images
    Also measures baseline (no SAE intervention).

    Returns:
        results_df: DataFrame with columns = class indices,
                    rows = "no_sae", "on_1", "off_1", "on_5", "off_5", ...
    """
    # Organize data by class
    # Build a mapping: class_idx -> list of (image_tensor, label) pairs
    class_images = defaultdict(list)
    to_pil = transforms.ToPILImage()

    print("  Organizing images by class...")
    for images, labels in dataloader:
        for i in range(images.size(0)):
            c = labels[i].item()
            class_images[c].append(images[i])

    for c in range(num_classes):
        print(f"    Class {c} ({classnames[c]}): {len(class_images[c])} images")

    # Extend topk_list with d_sae (all features) like classify.py
    d_sae = cfg.d_sae
    full_topk = [k for k in topk_list if k < d_sae] + [d_sae]

    metrics_dict = {}

    for cls_idx in range(num_classes):
        print(f"\n  Evaluating class {cls_idx}/{num_classes-1}: {classnames[cls_idx]} "
              f"({len(class_images[cls_idx])} imgs)")

        cls_imgs = class_images[cls_idx]
        if not cls_imgs:
            metrics_dict[cls_idx] = {}
            continue

        preds_dict = defaultdict(list)

        # Process in mini-batches
        n = len(cls_imgs)
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch_imgs = cls_imgs[start:end]

            inputs = vit.processor(
                images=[to_pil(img) for img in batch_imgs],
                text="",
                return_tensors="pt",
                padding=True,
            ).to(device)

            # Baseline: no SAE
            preds_dict["no_sae"].extend(
                get_predictions(vit, inputs, text_features)
            )
            flush_gpu()

            # Top-k features for this class (sorted by activation, descending)
            sorted_feats = cls_sae_cnt[cls_idx].argsort()[::-1]

            for topk in full_topk:
                cls_features = sorted_feats[:topk].tolist()

                # ON: clamp top features to active
                hooks_on = create_sae_hooks(cfg, cls_features, sae, device, "on")
                preds_dict[f"on_{topk}"].extend(
                    get_predictions(vit, inputs, text_features, hooks_on)
                )
                flush_gpu()

                # OFF: clamp top features to inactive
                hooks_off = create_sae_hooks(cfg, cls_features, sae, device, "off")
                preds_dict[f"off_{topk}"].extend(
                    get_predictions(vit, inputs, text_features, hooks_off)
                )
                flush_gpu()

        # Compute per-class accuracy
        metrics_dict[cls_idx] = {}
        for key, preds in preds_dict.items():
            acc = sum(1 for p in preds if p == cls_idx) / len(preds) * 100
            metrics_dict[cls_idx][key] = acc

    # Build DataFrame: rows=conditions, cols=classes
    results_df = pd.DataFrame(metrics_dict)
    return results_df


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAE feature quality via zero-shot CLIP classification "
                    "with top-k feature masking (no training required)")
    parser.add_argument("--base_sae", type=str, default=BASE_SAE_PATH)
    parser.add_argument("--medmnist_sae_dir", type=str, default=MEDMNIST_SAE_DIR)
    parser.add_argument("--dataset_root", type=str, default=DATASET_ROOT)
    parser.add_argument("--backbone", type=str, default=BACKBONE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save_dir", type=str, default="out/medmnist_eval",
                        help="Directory to save evaluation results")
    parser.add_argument("--include_base", action="store_true",
                        help="Also include the base ImageNet SAE")
    parser.add_argument("--eval_cache", type=str, default=EVAL_CACHE_DIR,
                        help="Directory to cache cls_sae_cnt arrays")
    parser.add_argument("--reuse_activations", action="store_true",
                        help="Reuse cached per-class SAE activation profiles")
    parser.add_argument("--lora_checkpoint", type=str, default=LORA_CHECKPOINT_PATH,
                        help="Path to LoRA checkpoint for CLIP")
    parser.add_argument("--topk", type=int, nargs="+", default=TOPK_LIST,
                        help="Top-k values to test (e.g. --topk 1 5 10 100)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=" * 70)
    print("MEDMNIST SAE EVALUATION  (zero-shot CLIP + top-k masking)")
    print("=" * 70)

    # ── 1. Load class names ──────────────────────────────────────────────────
    with open(CLASSNAMES_PATH) as f:
        classnames = json.load(f)
    num_classes = len(classnames)
    print(f"\nClasses ({num_classes}): {classnames}")

    # ── 2. Load SAEs ─────────────────────────────────────────────────────────
    print(f"\nLoading SAEs (include_base={args.include_base})...")
    medmnist_sae_paths = discover_medmnist_saes(args.medmnist_sae_dir)
    saes = load_selected_saes(args.base_sae, medmnist_sae_paths, args.device,
                              include_base=args.include_base)
    if not saes:
        print("[FATAL] No SAEs loaded. Exiting.")
        sys.exit(1)

    print(f"\nSAEs to evaluate ({len(saes)}):")
    for _, cfg, label in saes:
        print(f"  • {label}: d_sae={cfg.d_sae}, layer={cfg.block_layer}")

    # ── 3. Load ViT + LoRA ───────────────────────────────────────────────────
    print(f"\nLoading ViT ({args.backbone})...")
    ref_cfg = saes[0][1]
    vit = load_hooked_vit(ref_cfg, "base", args.backbone, args.device)
    print("  Base ViT loaded.")

    print(f"  Applying LoRA weights from: {args.lora_checkpoint}")
    load_lora_weights(vit, args.lora_checkpoint, args.device)
    print("  LoRA weights merged into ViT.")

    # ── 4. Compute text features ─────────────────────────────────────────────
    print(f"\nComputing CLIP text features ({len(openai_imagenet_template)} templates)...")
    text_features = calculate_text_features(vit, args.device, classnames)
    print(f"  Text features shape: {text_features.shape}")

    # ── 5. Load dataset ──────────────────────────────────────────────────────
    print(f"\nLoading dataset from {args.dataset_root}...")
    transform = get_transform()
    full_dataset = datasets.ImageFolder(root=args.dataset_root, transform=transform)
    print(f"  Total images: {len(full_dataset)}")

    n_val = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train: {n_train} (for activation profiles), Val: {n_val} (for evaluation)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ── 6. Evaluate each SAE ─────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.eval_cache, exist_ok=True)

    summary_results = {}

    for sae, cfg, label in saes:
        print(f"\n{'═' * 70}")
        print(f"EVALUATING:  {label}  (d_sae={cfg.d_sae}, layer={cfg.block_layer})")
        print(f"{'═' * 70}")

        # ── 6a. Compute or load per-class SAE activation profiles ────────────
        cnt_path = os.path.join(args.eval_cache, f"cls_sae_cnt_{label}.npy")

        if args.reuse_activations and os.path.exists(cnt_path):
            print(f"  [REUSE] Loading cached activation profiles: {cnt_path}")
            cls_sae_cnt = np.load(cnt_path)
        else:
            print(f"\n  Computing per-class SAE activation profiles (train set)...")
            cls_sae_cnt = compute_cls_sae_activations(
                vit, sae, cfg, train_loader, num_classes, args.device,
            )
            np.save(cnt_path, cls_sae_cnt)
            print(f"  Saved: {cnt_path}")

        # Quick stats
        active_per_class = (cls_sae_cnt > 0).sum(axis=1)
        print(f"  Active features per class: "
              f"min={active_per_class.min()}, max={active_per_class.max()}, "
              f"mean={active_per_class.mean():.0f}")

        # ── 6b. Run top-k masking classification on val set ──────────────────
        print(f"\n  Running top-k masking evaluation on val set ({n_val} images)...")
        results_df = evaluate_topk_masking(
            vit, sae, cfg, cls_sae_cnt, text_features,
            val_loader, num_classes, classnames, args.device,
            topk_list=args.topk,
        )

        # Save per-SAE results
        csv_path = os.path.join(args.save_dir, f"metrics_{label}.csv")
        results_df.to_csv(csv_path)
        print(f"\n  Saved: {csv_path}")

        # ── 6c. Print summary for this SAE ───────────────────────────────────
        mean_accs = results_df.mean(axis=1)

        print(f"\n  {'Condition':<20s} {'Mean Acc':>10s}")
        print(f"  {'─' * 32}")

        baseline = mean_accs.get("no_sae", 0)
        print(f"  {'no_sae (baseline)':<20s} {baseline:>9.2f}%")

        for k in args.topk:
            on_key = f"on_{k}"
            off_key = f"off_{k}"
            on_acc = mean_accs.get(on_key, 0)
            off_acc = mean_accs.get(off_key, 0)
            on_delta = on_acc - baseline
            off_delta = off_acc - baseline
            print(f"  {'top-' + str(k) + ' ON':<20s} {on_acc:>9.2f}%  ({on_delta:+.2f})")
            print(f"  {'top-' + str(k) + ' OFF':<20s} {off_acc:>9.2f}%  ({off_delta:+.2f})")

        summary_results[label] = {
            "baseline": baseline,
            "best_on": max(mean_accs.get(f"on_{k}", 0) for k in args.topk),
            "worst_off": min(mean_accs.get(f"off_{k}", 0) for k in args.topk),
            "layer": cfg.block_layer,
            "d_sae": cfg.d_sae,
        }

        flush_gpu()

    # ── 7. Cross-SAE Summary ─────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("CROSS-SAE COMPARISON")
    print(f"{'═' * 70}")
    print(f"{'SAE':<28s} {'Layer':>6s} {'Baseline':>10s} {'Best ON':>10s} "
          f"{'Worst OFF':>11s} {'ON Δ':>8s} {'OFF Δ':>8s}")
    print(f"{'─' * 83}")

    for label, r in summary_results.items():
        on_delta = r["best_on"] - r["baseline"]
        off_delta = r["worst_off"] - r["baseline"]
        print(f"{label:<28s} {r['layer']:>6} {r['baseline']:>9.2f}% "
              f"{r['best_on']:>9.2f}% {r['worst_off']:>10.2f}% "
              f"{on_delta:>+7.2f} {off_delta:>+7.2f}")

    print(f"{'─' * 83}")
    print(f"\nResults saved in: {args.save_dir}/")
    print(f"Activation cache: {args.eval_cache}/")

    # Save summary JSON
    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"Summary: {summary_path}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
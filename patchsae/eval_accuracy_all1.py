#!/usr/bin/env python3
"""
Cross-model SAE Transfer Analysis
==================================

Computes SAE reconstruction metrics for all 4 combinations:

  1. LoRA SAE  ↔ LoRA activations   (matched)
  2. MaPLe SAE ↔ MaPLe activations  (matched)
  3. LoRA SAE  → MaPLe activations   (cross)
  4. MaPLe SAE → LoRA activations    (cross)

Metrics: MSE, Explained Variance, Cosine Similarity, L0, Dead Features %

Usage:
    python cross_sae_transfer.py
    python cross_sae_transfer.py --num_batches 20   # quick run
"""

import argparse, os, sys, glob, gc, math
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import json

try:
    from tasks.utils import load_sae, load_hooked_vit
    from src.sae_training.hooked_vit import Hook
except ImportError as e:
    print(f"[FATAL] {e}\nRun from patchsae root."); sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────
LORA_SAE_DIR     = "out/checkpoints/medmnist"
MAPLE_SAE_PATH   = "/home/sunayana/Documents/Concept_LoRA/patchsae/out/checkpoints/medmnist_maple/qcx3o72n/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt"
LORA_CHECKPOINT  = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
TRAIN_ROOT       = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
NPZ_PATH         = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
BACKBONE         = "openai/clip-vit-base-patch16"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = 64
SAE_BIAS         = -0.105131256516992

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

# ── Helpers ───────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def get_processor_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def get_imagefolder_classnames(root):
    ds = datasets.ImageFolder(root=root)
    idx_to_name = {i: n.replace('_', ' ') for n, i in ds.class_to_idx.items()}
    return [idx_to_name[i] for i in range(len(idx_to_name))]

def build_npz_to_if_mapping(root):
    ds = datasets.ImageFolder(root=root)
    if_map = {n.replace('_', ' ').lower(): i for n, i in ds.class_to_idx.items()}
    return {i: if_map.get(c.lower(), i) for i, c in enumerate(MEDMNIST_CLASSES)}

class NpzTestDataset(Dataset):
    def __init__(self, npz_path, mapping, transform):
        data = np.load(npz_path)
        self.imgs = data['test_images']
        self.labels = data['test_labels'].flatten()
        self.mapping = mapping
        self.transform = transform
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.imgs[i])), self.mapping[int(self.labels[i])]

def discover_layer_sae(base_dir, target_layer):
    paths = sorted(glob.glob(os.path.join(base_dir, "*/final*/*.pt")))
    for p in paths:
        ckpt = torch.load(p, map_location="cpu")
        cfg_d = ckpt.get("cfg", ckpt.get("config"))
        layer = getattr(cfg_d, "block_layer",
                        cfg_d.get("block_layer", "?") if isinstance(cfg_d, dict) else "?")
        del ckpt; gc.collect()
        if layer == target_layer:
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════
# LoRA merge into HF CLIP (matching training scaling)
# ══════════════════════════════════════════════════════════════════════════

def _extract_lora_AB(ld, proj):
    if proj in ld and isinstance(ld[proj], dict):
        try: return ld[proj]["w_lora_A"], ld[proj]["w_lora_B"]
        except: pass
    try: return ld[f"{proj}.w_lora_A"], ld[f"{proj}.w_lora_B"]
    except: return None, None

def merge_lora_into_hf_clip(vit, lora_path, device):
    lora_state = torch.load(lora_path, map_location=device)
    if "weights" not in lora_state:
        return vit
    layers_dict = lora_state["weights"]
    meta = lora_state["metadata"]
    scale = meta["alpha"] / math.sqrt(meta["r"])  # matches training
    print(f"    LoRA: rank={meta['r']}, alpha={meta['alpha']}, scale={scale:.6f}")

    hf_model = vit.model
    if hasattr(hf_model, 'model'):
        hf_model = hf_model.model

    merged = 0
    with torch.no_grad():
        for layer_key, ld in layers_dict.items():
            layer_idx = int(layer_key.split("_")[1])
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                A, B = _extract_lora_AB(ld, proj_name)
                if A is None: continue
                delta = scale * (B.float().to(device) @ A.float().to(device))
                if layer_idx < 12:
                    try:
                        w = hf_model.text_model.encoder.layers[layer_idx].self_attn.__getattr__(proj_name).weight
                        w.data += delta.to(w.dtype); merged += 1
                    except: pass
                else:
                    try:
                        w = hf_model.vision_model.encoder.layers[layer_idx-12].self_attn.__getattr__(proj_name).weight
                        w.data += delta.to(w.dtype); merged += 1
                    except: pass
    print(f"    Merged {merged} projections")
    return vit


# ══════════════════════════════════════════════════════════════════════════
# Activation collection via hooks
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_activations_base(vit, loader, block_layer, module_name, device, max_batches=None):
    """Collect activations from base/lora HF CLIP (standard hook, batch-seq-dim)."""
    all_acts = []
    all_labels = []
    to_pil = transforms.ToPILImage()
    captured = {}

    def capture_fn(act):
        captured["act"] = act[:, :, :].detach().float().cpu()
        return (act,)

    hook = Hook(block_layer, module_name, capture_fn,
                return_module_output=False, is_custom=False)

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="  collect", leave=False)):
        if max_batches and batch_idx >= max_batches:
            break
        inputs = vit.processor(images=[to_pil(img) for img in images],
                               text="", return_tensors="pt", padding=True).to(device)
        vit.run_with_hooks([hook], return_type="output", **inputs)
        all_acts.append(captured["act"])
        all_labels.append(labels)

    return torch.cat(all_acts, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def collect_activations_maple(vit, loader, block_layer, module_name, device, max_batches=None):
    """Collect activations from MaPLe HF CLIP (custom hook, seq-batch-dim)."""
    all_acts = []
    all_labels = []
    to_pil = transforms.ToPILImage()
    captured = {}

    def capture_fn(act):
        # maple custom hook: act is [seq, batch, dim]
        captured["act"] = act.transpose(0, 1).detach().float().cpu()  # → [batch, seq, dim]
        return act  # pass through unchanged

    hook = Hook(block_layer, module_name, capture_fn,
                return_module_output=False, is_custom=True)

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="  collect", leave=False)):
        if max_batches and batch_idx >= max_batches:
            break
        inputs = vit.processor(images=[to_pil(img) for img in images],
                               text="", return_tensors="pt", padding=True).to(device)
        vit.run_with_hooks([hook], return_type="output", **inputs)
        all_acts.append(captured["act"])
        all_labels.append(labels)

    return torch.cat(all_acts, dim=0), torch.cat(all_labels, dim=0)


# ══════════════════════════════════════════════════════════════════════════
# SAE metric computation
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_sae_metrics(sae, activations, device, sae_bias=SAE_BIAS, batch_size=4):
    """
    Run activations through SAE. Return reconstruction metrics.
    activations: [N, seq_len, d_model] on CPU
    """
    n_samples = activations.shape[0]
    d = activations.shape[-1]

    total_mse = 0.0
    total_cos = 0.0
    total_var = 0.0
    total_l0 = 0.0
    total_l1 = 0.0
    total_tokens = 0
    feature_fired = None

    for start in tqdm(range(0, n_samples, batch_size), desc="    metrics", leave=False):
        end = min(start + batch_size, n_samples)
        act = activations[start:end].to(device)  # [B, seq, d]
        B, S, D = act.shape
        n_tok = B * S

        sae_output = sae(act)
        recon = sae_output[0] - sae_bias

        # MSE
        mse = (act - recon).pow(2).mean(dim=-1)
        total_mse += mse.sum().item()

        # Variance
        total_var += act.var(dim=-1).sum().item()

        # Cosine sim
        cos = torch.nn.functional.cosine_similarity(
            act.reshape(-1, D), recon.reshape(-1, D), dim=-1)
        total_cos += cos.sum().item()

        total_tokens += n_tok

        # Sparsity (if hidden activations available)
        if len(sae_output) >= 2:
            h = sae_output[1].reshape(-1, sae_output[1].shape[-1])
            active = (h.abs() > 1e-8)
            total_l0 += active.float().sum(dim=-1).sum().item()
            total_l1 += h.abs().sum(dim=-1).mean().item() * n_tok

            if feature_fired is None:
                feature_fired = torch.zeros(h.shape[-1], dtype=torch.bool, device=device)
            feature_fired |= active.any(dim=0)

        del act, recon, sae_output

    m = {}
    m["mse"] = total_mse / total_tokens
    m["act_variance"] = total_var / total_tokens
    m["relative_mse"] = m["mse"] / max(m["act_variance"], 1e-10)
    m["explained_variance"] = 1.0 - m["relative_mse"]
    m["cosine_similarity"] = total_cos / total_tokens
    m["total_tokens"] = total_tokens

    if feature_fired is not None:
        n_feat = feature_fired.shape[0]
        n_dead = (~feature_fired).sum().item()
        m["l0"] = total_l0 / total_tokens
        m["l1"] = total_l1 / total_tokens
        m["n_features"] = n_feat
        m["dead_features"] = n_dead
        m["dead_pct"] = n_dead / n_feat * 100
        m["alive_pct"] = 100.0 - m["dead_pct"]

    return m


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_checkpoint", default=LORA_CHECKPOINT)
    p.add_argument("--maple_sae_path", default=MAPLE_SAE_PATH)
    p.add_argument("--model_path", default="/home/sunayana/Documents/model.pth.tar-5")
    p.add_argument("--config_path",
                   default="/home/sunayana/Documents/Concept_LoRA/patchsae/configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml")
    p.add_argument("--target_layer", type=int, default=-2)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num_batches", type=int, default=None,
                   help="Limit batches for quick run (None=all)")
    p.add_argument("--save_dir", default="out/sae_transfer")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    classnames = get_imagefolder_classnames(TRAIN_ROOT)
    label_mapping = build_npz_to_if_mapping(TRAIN_ROOT)

    print(f"{'═'*70}")
    print(f"  Cross-Model SAE Transfer Analysis (layer {args.target_layer})")
    print(f"{'═'*70}")

    # ── Load SAEs ─────────────────────────────────────────────────
    print(f"\n[1/5] Loading SAEs...")

    lora_sae_path = discover_layer_sae(LORA_SAE_DIR, args.target_layer)
    assert lora_sae_path, f"No LoRA SAE at layer {args.target_layer}"
    lora_sae, lora_cfg = load_sae(lora_sae_path, DEVICE)
    print(f"  LoRA SAE:  layer={lora_cfg.block_layer}, module={lora_cfg.module_name}")
    print(f"             {lora_sae_path}")

    maple_sae, maple_cfg = load_sae(args.maple_sae_path, DEVICE)
    print(f"  MaPLe SAE: layer={maple_cfg.block_layer}, module={maple_cfg.module_name}")
    print(f"             {args.maple_sae_path}")

    # Print SAE sizes
    for name, sae in [("LoRA", lora_sae), ("MaPLe", maple_sae)]:
        n_params = sum(p.numel() for p in sae.parameters())
        print(f"  {name} SAE: {n_params:,} params")

    # ── Shared test loader ────────────────────────────────────────
    transform = get_processor_transform()
    test_ds = NpzTestDataset(NPZ_PATH, label_mapping, transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    n_samples = len(test_ds) if args.num_batches is None else min(args.num_batches * args.batch_size, len(test_ds))
    print(f"  Test samples: ~{n_samples}")

    # ── Collect LoRA activations ──────────────────────────────────
    print(f"\n[2/5] Collecting LoRA activations...")
    vit_lora = load_hooked_vit(lora_cfg, "base", BACKBONE, DEVICE)
    merge_lora_into_hf_clip(vit_lora, args.lora_checkpoint, DEVICE)
    vit_lora.eval()

    lora_acts, lora_labels = collect_activations_base(
        vit_lora, test_loader, lora_cfg.block_layer, lora_cfg.module_name,
        DEVICE, max_batches=args.num_batches)
    print(f"  shape={lora_acts.shape}, mean={lora_acts.mean():.6f}, std={lora_acts.std():.6f}")
    del vit_lora; flush()

    # ── Collect MaPLe activations ─────────────────────────────────
    print(f"\n[3/5] Collecting MaPLe activations...")
    vit_maple = load_hooked_vit(maple_cfg, "maple", BACKBONE, DEVICE,
                                model_path=args.model_path,
                                config_path=args.config_path,
                                classnames=classnames)
    vit_maple.eval()

    maple_acts, maple_labels = collect_activations_maple(
        vit_maple, test_loader, maple_cfg.block_layer, maple_cfg.module_name,
        DEVICE, max_batches=args.num_batches)
    print(f"  shape={maple_acts.shape}, mean={maple_acts.mean():.6f}, std={maple_acts.std():.6f}")
    del vit_maple; flush()

    # ── Activation distribution comparison ────────────────────────
    print(f"\n[3.5] Activation distribution comparison:")
    print(f"  {'':20s} {'LoRA':>12s} {'MaPLe':>12s}")
    print(f"  {'mean':20s} {lora_acts.mean().item():>12.6f} {maple_acts.mean().item():>12.6f}")
    print(f"  {'std':20s} {lora_acts.std().item():>12.6f} {maple_acts.std().item():>12.6f}")
    print(f"  {'min':20s} {lora_acts.min().item():>12.4f} {maple_acts.min().item():>12.4f}")
    print(f"  {'max':20s} {lora_acts.max().item():>12.4f} {maple_acts.max().item():>12.4f}")
    print(f"  {'norm (mean)':20s} {lora_acts.norm(dim=-1).mean().item():>12.4f} {maple_acts.norm(dim=-1).mean().item():>12.4f}")

    # Direct activation similarity (CLS token only — seq lengths may differ
    # because MaPLe adds context tokens)
    min_n = min(lora_acts.shape[0], maple_acts.shape[0])
    lora_cls = lora_acts[:min_n, 0, :]   # [N, d_model] — CLS token
    maple_cls = maple_acts[:min_n, 0, :]  # [N, d_model] — CLS token
    direct_cos = torch.nn.functional.cosine_similarity(
        lora_cls, maple_cls, dim=-1).mean().item()
    print(f"  {'act cos(LoRA,MaPLe)':20s} {direct_cos:>12.6f}  (CLS token)")
    print(f"  {'LoRA seq_len':20s} {lora_acts.shape[1]:>12d}")
    print(f"  {'MaPLe seq_len':20s} {maple_acts.shape[1]:>12d}")

    # ── Compute all 4 SAE metric combinations ────────────────────
    print(f"\n[4/5] Computing SAE reconstruction metrics...")

    configs = [
        ("LoRA SAE ↔ LoRA acts",   lora_sae,  lora_acts,  "(matched)"),
        ("MaPLe SAE ↔ MaPLe acts", maple_sae, maple_acts, "(matched)"),
        ("LoRA SAE → MaPLe acts",  lora_sae,  maple_acts, "(cross)"),
        ("MaPLe SAE → LoRA acts",  maple_sae, lora_acts,  "(cross)"),
    ]

    all_metrics = {}
    for label, sae, acts, tag in configs:
        print(f"\n  {label} {tag}")
        m = compute_sae_metrics(sae, acts, DEVICE)
        all_metrics[label] = m
        flush()

    del lora_acts, maple_acts; flush()

    # ── Results table ─────────────────────────────────────────────
    print(f"\n\n[5/5] RESULTS")
    print(f"{'═'*90}")
    print(f"  Cross-Model SAE Transfer Analysis — Layer {args.target_layer}")
    print(f"{'═'*90}")

    metric_display = [
        ("mse",                "MSE ↓",         "{:.6f}"),
        ("explained_variance", "Expl Var ↑",    "{:.4f}"),
        ("cosine_similarity",  "Cos Sim ↑",     "{:.4f}"),
        ("l0",                 "L0 sparsity",   "{:.1f}"),
        ("dead_pct",           "Dead feat %",   "{:.1f}"),
        ("alive_pct",          "Alive feat %",  "{:.1f}"),
        ("n_features",         "# features",    "{:.0f}"),
    ]

    labels = [l for l, _, _, _ in configs]
    tags   = [t for _, _, _, t in configs]

    # Header
    print(f"\n  {'Metric':<16s}", end="")
    for l, t in zip(labels, tags):
        short = l.split("↔")[0].split("→")[0].strip() + " " + t
        print(f"  {short:>20s}", end="")
    print()
    print(f"  {'─'*16}" + f"  {'─'*20}" * 4)

    for key, name, fmt in metric_display:
        row = f"  {name:<16s}"
        for label in labels:
            val = all_metrics[label].get(key)
            if val is not None:
                row += f"  {fmt.format(val):>20s}"
            else:
                row += f"  {'N/A':>20s}"
        print(row)

    # ── Key insights ──────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  KEY COMPARISONS")
    print(f"{'═'*90}")

    lora_matched  = all_metrics[labels[0]]
    maple_matched = all_metrics[labels[1]]
    lora_on_maple = all_metrics[labels[2]]
    maple_on_lora = all_metrics[labels[3]]

    print(f"\n  1. Matched reconstruction (how well each SAE fits its own model):")
    print(f"     {'':20s} {'LoRA':>10s} {'MaPLe':>10s} {'Winner':>10s}")
    for key, name in [("explained_variance", "Expl Var ↑"),
                       ("cosine_similarity", "Cos Sim ↑"),
                       ("mse", "MSE ↓")]:
        lv = lora_matched[key]
        mv = maple_matched[key]
        up = "↑" in name
        w = "MaPLe" if (mv > lv if up else mv < lv) else "LoRA"
        print(f"     {name:20s} {lv:10.4f} {mv:10.4f} {w:>10s}")

    print(f"\n  2. Cross-model transfer (does one SAE generalize to the other?):")
    print(f"     {'':30s} {'Expl Var':>10s} {'Cos Sim':>10s}")
    print(f"     {'LoRA SAE → MaPLe acts':30s} "
          f"{lora_on_maple['explained_variance']:10.4f} "
          f"{lora_on_maple['cosine_similarity']:10.4f}")
    print(f"     {'MaPLe SAE → LoRA acts':30s} "
          f"{maple_on_lora['explained_variance']:10.4f} "
          f"{maple_on_lora['cosine_similarity']:10.4f}")

    # Transfer asymmetry
    ev_lora_cross = lora_on_maple["explained_variance"]
    ev_maple_cross = maple_on_lora["explained_variance"]
    ev_lora_match = lora_matched["explained_variance"]
    ev_maple_match = maple_matched["explained_variance"]

    lora_retention = ev_lora_cross / max(ev_lora_match, 1e-10) * 100
    maple_retention = ev_maple_cross / max(ev_maple_match, 1e-10) * 100

    print(f"\n  3. Transfer retention (% of matched quality preserved in cross):")
    print(f"     LoRA SAE  retains {lora_retention:.1f}% when applied to MaPLe")
    print(f"     MaPLe SAE retains {maple_retention:.1f}% when applied to LoRA")

    if lora_retention > maple_retention + 5:
        print(f"     → LoRA representations subsume MaPLe's (LoRA SAE transfers better)")
    elif maple_retention > lora_retention + 5:
        print(f"     → MaPLe representations subsume LoRA's (MaPLe SAE transfers better)")
    elif lora_retention < 50 and maple_retention < 50:
        print(f"     → Both drop heavily — models learn GENUINELY DIFFERENT representations")
    else:
        print(f"     → Moderate overlap — partially shared representation structure")

    print(f"\n{'═'*90}")

    # ── Save ──────────────────────────────────────────────────────
    save_data = {
        "layer": args.target_layer,
        "activation_cos_similarity_lora_vs_maple": direct_cos,
    }
    for label in labels:
        save_key = label.replace(" ", "_").replace("↔", "on").replace("→", "on")
        save_data[save_key] = all_metrics[label]

    save_path = os.path.join(args.save_dir, "cross_sae_transfer.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
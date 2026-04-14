#!/usr/bin/env python3
"""
Compute test-set SAE metrics: L0, Recon MSE, Dead%, and Label Entropy.

Label Entropy: for each SAE feature that fires at least once, compute
Shannon entropy of its label distribution (which classes activate it).
Low entropy = class-selective / monosemantic. Reported as the mean
across all active features (weighted by activation frequency).

Usage:
    cd /home/sunayana/Documents/Concept_LoRA/patchsae
    python compute_sae_eval_metrics.py
"""

import math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import clip as openai_clip
except ImportError:
    sys.exit("[FATAL] pip install git+https://github.com/openai/CLIP.git")

sys.path.insert(0, os.path.dirname(__file__))
from tasks.utils import load_sae

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE    = "ViT-B/16"
BATCH_SIZE  = 64
DATA_ROOT   = "/home/sunayana/Documents/Concept_LoRA/data"
LORA_ROOT   = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"

RUNS = [
    {
        "name":      "Base SAE (ImageNet)",
        "dataset":   "caltech101",
        "sae_path":  "data/sae_weight/base/out.pt",
        "lora_path": None,
    },
    {
        "name":      "Base SAE (ImageNet)",
        "dataset":   "eurosat",
        "sae_path":  "data/sae_weight/base/out.pt",
        "lora_path": None,
    },
    {
        "name":      "Base SAE (ImageNet)",
        "dataset":   "medmnist",
        "sae_path":  "data/sae_weight/base/out.pt",
        "lora_path": None,
    },
    {
        "name":      "LoRA SAE",
        "dataset":   "caltech101",
        "sae_path":  "out/checkpoints/caltech101/ted4zuln/final_sparse_autoencoder_openai/clip-vit-base-patch16_-3_resid_49152.pt",
        "lora_path": f"{LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
    },
    {
        "name":      "LoRA SAE",
        "dataset":   "eurosat",
        "sae_path":  "out/checkpoints/eurosat/bk3rbkcx/final_sparse_autoencoder_openai/clip-vit-base-patch16_-3_resid_49152.pt",
        "lora_path": f"{LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
    },
    {
        "name":      "LoRA SAE",
        "dataset":   "medmnist",
        "sae_path":  "out/checkpoints/medmnist/d2ygd3bb/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt",
        "lora_path": f"{LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
    },
]

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]


# ── LoRA loading (mirrors eval_all_saes_comprehensive.py) ─────────────────────

def build_lora_clip(lora_path, device):
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    state = torch.load(lora_path, map_location=device, weights_only=False)
    if "weights" not in state:
        model.load_state_dict(state)
        return model, preprocess
    layers, meta = state["weights"], state["metadata"]
    scale = meta["alpha"] / math.sqrt(meta["r"])
    with torch.no_grad():
        for i in range(12):
            ld = layers.get(f"layer_{i+12}", {})
            if not ld:
                continue
            blk = model.visual.transformer.resblocks[i]
            w   = blk.attn.in_proj_weight.data
            d   = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2*d)]:
                if proj in ld and isinstance(ld[proj], dict):
                    try:
                        A, B = ld[proj]["w_lora_A"], ld[proj]["w_lora_B"]
                    except Exception:
                        continue
                else:
                    try:
                        A = ld[f"{proj}.w_lora_A"]
                        B = ld[f"{proj}.w_lora_B"]
                    except Exception:
                        continue
                delta = scale * (B.float().to(device) @ A.float().to(device))
                w[off:off+d] += delta.to(w.dtype)
    return model, preprocess


def build_base_clip(device):
    return openai_clip.load(BACKBONE, device=device)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _find_imagefolder_root(path):
    for root, dirs, files in os.walk(path):
        if dirs:
            sample = os.path.join(root, dirs[0])
            exts = {os.path.splitext(f)[1].lower()
                    for f in os.listdir(sample)
                    if os.path.isfile(os.path.join(sample, f))}
            if exts & {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                return root
    return path


class FilteredImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform, exclude_classes=None):
        img_root = _find_imagefolder_root(root)
        full_ds  = datasets.ImageFolder(root=img_root, transform=transform)
        exclude  = exclude_classes or set()
        keep     = [i for i, (_, lbl) in enumerate(full_ds.samples)
                    if full_ds.classes[lbl] not in exclude]
        kept_cls = sorted({full_ds.classes[full_ds.targets[i]] for i in keep})
        old2new  = {full_ds.class_to_idx[c]: new_i
                    for new_i, c in enumerate(kept_cls)}
        self._ds    = full_ds
        self._idx   = keep
        self._map   = old2new
        self.classes = kept_cls
        self.num_classes = len(kept_cls)

    def __len__(self): return len(self._idx)

    def __getitem__(self, i):
        img, lbl = self._ds[self._idx[i]]
        return img, self._map[lbl]


class MedMNISTTestDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, imagefolder_root, preprocess):
        data = np.load(npz_path)
        self.imgs   = data["test_images"]
        self.labels = data["test_labels"].flatten().astype(int)
        self.preprocess = preprocess
        if_root = _find_imagefolder_root(imagefolder_root)
        if_ds   = datasets.ImageFolder(root=if_root)
        if_map  = {n.replace("_", " ").lower(): i for n, i in if_ds.class_to_idx.items()}
        self.label_map = {i: if_map.get(c.lower(), i)
                          for i, c in enumerate(MEDMNIST_CLASSES)}
        self.num_classes = len(MEDMNIST_CLASSES)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        img = self.preprocess(Image.fromarray(self.imgs[i]))
        return img, self.label_map[int(self.labels[i])]


def load_test_dataset(dataset_name, preprocess):
    if dataset_name == "caltech101":
        ds = FilteredImageFolder(
            f"{DATA_ROOT}/caltech-101",
            preprocess,
            exclude_classes={"BACKGROUND_Google"},
        )
    elif dataset_name == "eurosat":
        ds = FilteredImageFolder(f"{DATA_ROOT}/eurosat/2750", preprocess)
    elif dataset_name == "medmnist":
        ds = MedMNISTTestDataset(
            f"{DATA_ROOT}/pathmnist.npz",
            f"{DATA_ROOT}/pathmnist_imagefolder",
            preprocess,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return ds


# ── Activation capture hook ───────────────────────────────────────────────────

class ActivationCapture:
    """Captures residual output of a CLIP resblock (transposes SBD → BSD)."""

    def __init__(self):
        self.activations = None
        self._handle = None

    def register(self, block):
        def _hook(module, input, output):
            # OpenAI CLIP: [seq, batch, d_model] → [batch, seq, d_model]
            self.activations = output.detach().float().transpose(0, 1)
        self._handle = block.register_forward_hook(_hook)

    def remove(self):
        if self._handle:
            self._handle.remove()


# ── Core eval ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_sae(clip_model, sae, sae_cfg, dataset, device):
    """
    Returns dict with keys:
        l0          – mean active features per token per image
        recon_mse   – mean per-token MSE between input and SAE reconstruction
        dead_pct    – % features that never fired on this dataset
        label_entropy_mean   – mean entropy over active features (nats)
        label_entropy_norm   – entropy / log(num_classes)  ∈ [0,1]
    """
    n_total      = len(clip_model.visual.transformer.resblocks)
    layer_idx    = sae_cfg.block_layer if sae_cfg.block_layer >= 0 \
                   else n_total + sae_cfg.block_layer
    block        = clip_model.visual.transformer.resblocks[layer_idx]

    capture = ActivationCapture()
    capture.register(block)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    n_features   = sae.W_enc.shape[1]  # d_sae
    num_classes  = dataset.num_classes

    # Accumulators
    total_l0          = 0.0
    total_mse         = 0.0
    total_tokens      = 0
    total_images      = 0
    # label_counts[f, c] = times feature f fired for class-c image
    label_counts      = torch.zeros(n_features, num_classes, dtype=torch.float32)
    feature_fire_count = torch.zeros(n_features, dtype=torch.float32)

    sae.eval().to(device)
    clip_model.eval()

    for imgs, labels in tqdm(loader, desc="  batches", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device)
        B      = imgs.shape[0]

        # Forward CLIP (hook captures residual)
        clip_model.encode_image(imgs)
        acts_bsd = capture.activations  # [B, seq, d_model]
        seq      = acts_bsd.shape[1]

        # Flatten to [B*seq, d_model]
        acts_flat = acts_bsd.reshape(-1, acts_bsd.shape[-1])

        # SAE forward
        sae_out_flat, feat_flat, *_ = sae(acts_flat)

        # --- L0 ---
        # feat_flat: [B*seq, n_features]
        feat_bsd = feat_flat.reshape(B, seq, n_features)
        active   = (feat_bsd > 0).float()
        l0_per_image = active.sum(dim=-1).mean(dim=-1)  # [B]
        total_l0     += l0_per_image.sum().item()

        # --- MSE ---
        mse_per_token = (sae_out_flat - acts_flat).pow(2).mean(dim=-1)  # [B*seq]
        total_mse     += mse_per_token.sum().item()
        total_tokens  += B * seq
        total_images  += B

        # --- Label distribution per feature ---
        # active_feat_per_image: [B, n_features] — did feature f fire for image i?
        active_per_img = (feat_bsd > 0).any(dim=1).float()  # [B, n_features]
        # one-hot labels: [B, num_classes]
        oh = F.one_hot(labels, num_classes=num_classes).float()
        # label_counts: [n_features, num_classes]
        label_counts      += (active_per_img.T @ oh).cpu()        # [F, C]
        feature_fire_count += active_per_img.sum(0).cpu()         # [F]

    capture.remove()

    # --- Dead features ---
    dead_mask = feature_fire_count == 0
    dead_pct  = dead_mask.float().mean().item() * 100.0

    # --- Label entropy ---
    # Only over features that fired at least once
    active_feat_mask = ~dead_mask
    counts_active = label_counts[active_feat_mask]           # [F_active, C]
    probs = counts_active / counts_active.sum(dim=1, keepdim=True).clamp(min=1e-9)
    # Shannon entropy per feature (nats)
    entropy_per_feat = -(probs * (probs + 1e-12).log()).sum(dim=1)  # [F_active]
    # Weighted mean: weight by how often the feature fires
    weights = feature_fire_count[active_feat_mask]
    label_entropy_mean = (entropy_per_feat * weights / weights.sum()).sum().item()
    label_entropy_norm = label_entropy_mean / math.log(num_classes) if num_classes > 1 else 0.0

    # Unweighted mean (also useful)
    label_entropy_unweighted = entropy_per_feat.mean().item()
    label_entropy_unweighted_norm = label_entropy_unweighted / math.log(num_classes) \
                                    if num_classes > 1 else 0.0

    return {
        "l0":                          total_l0 / total_images,
        "recon_mse":                   total_mse / total_tokens,
        "dead_pct":                    dead_pct,
        "label_entropy_mean":          label_entropy_mean,
        "label_entropy_norm":          label_entropy_norm,
        "label_entropy_unweighted":    label_entropy_unweighted,
        "label_entropy_unweighted_norm": label_entropy_unweighted_norm,
        "n_active_features":           active_feat_mask.sum().item(),
        "num_classes":                 num_classes,
        "n_images":                    total_images,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDevice: {DEVICE}")
    print("=" * 80)

    results = []
    loaded_models = {}  # cache (lora_path or None) → (model, preprocess)

    for run in RUNS:
        key = (run["name"], run["dataset"])
        print(f"\n{'─'*60}")
        print(f"  SAE   : {run['name']}")
        print(f"  Dataset: {run['dataset']}")
        print(f"  SAE   : {run['sae_path']}")

        # Load SAE
        sae, sae_cfg = load_sae(run["sae_path"], DEVICE)
        print(f"  Layer : {sae_cfg.block_layer}")

        # Load CLIP model (cached by lora_path key)
        model_key = run["lora_path"]
        if model_key not in loaded_models:
            print(f"  Loading CLIP (lora={model_key is not None})...")
            if run["lora_path"] is None:
                model, preprocess = build_base_clip(DEVICE)
            else:
                model, preprocess = build_lora_clip(run["lora_path"], DEVICE)
            model.eval()
            loaded_models[model_key] = (model, preprocess)
        model, preprocess = loaded_models[model_key]

        # Load dataset
        ds = load_test_dataset(run["dataset"], preprocess)
        print(f"  Dataset size: {len(ds)}, classes: {ds.num_classes}")

        # Eval
        metrics = eval_sae(model, sae, sae_cfg, ds, DEVICE)

        row = {
            "sae":     run["name"],
            "dataset": run["dataset"],
            **metrics,
        }
        results.append(row)

        print(f"\n  L0            : {metrics['l0']:.2f}")
        print(f"  Recon MSE     : {metrics['recon_mse']:.6f}")
        print(f"  Dead features : {metrics['dead_pct']:.2f}%")
        print(f"  Label entropy (weighted, nats): {metrics['label_entropy_mean']:.4f}")
        print(f"  Label entropy (weighted, norm): {metrics['label_entropy_norm']:.4f}")
        print(f"  Label entropy (unweighted, nats): {metrics['label_entropy_unweighted']:.4f}")
        print(f"  Active features: {metrics['n_active_features']:,} / 49152")

        # Free SAE memory
        del sae
        torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n")
    print("=" * 100)
    print(f"{'SAE':<22} {'Dataset':<12} {'L0':>8} {'Dead%':>8} {'Recon_MSE':>12} "
          f"{'H_wt(nats)':>12} {'H_wt_norm':>11} {'H_uw_norm':>11}")
    print("=" * 100)
    for r in results:
        print(f"{r['sae']:<22} {r['dataset']:<12} "
              f"{r['l0']:>8.2f} {r['dead_pct']:>8.2f} {r['recon_mse']:>12.6f} "
              f"{r['label_entropy_mean']:>12.4f} {r['label_entropy_norm']:>11.4f} "
              f"{r['label_entropy_unweighted_norm']:>11.4f}")
    print("=" * 100)

    # Save
    import json
    out_path = "out/sae_eval_metrics.json"
    os.makedirs("out", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

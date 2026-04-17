#!/usr/bin/env python3
"""
Eval metrics for the DINOv2-base SAE checkpoint:
    out/checkpoints/dinov2-base_-1_resid_49152.pt

NOTE: The .pt file is a truncated zip archive. W_enc and b_enc have been
recovered into /tmp/dinov2_extracted/ (via 7z e). W_dec and b_dec are absent,
so reconstruction-based metrics (MSE, explained variance) are SKIPPED.

Metrics computed (encoder-only):
  - L0            : mean active features per token per image
  - Dead%         : fraction of the 49 152 features that never fire
  - Act_sparsity  : fraction of (feature, image) pairs that are active
  - Max_act       : mean per-feature peak activation
  - Label entropy : Shannon entropy of each active feature's class
                    distribution (low = monosemantic / class-selective)

Usage:
    cd /home/sunayana/Documents/Concept_LoRA/patchsae
    conda run -n patchsae python eval_dinov2_sae.py
"""

import math, os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
DATA_ROOT  = "/home/sunayana/Documents/Concept_LoRA/data"
EXTRACTED  = "/tmp/dinov2_extracted"  # output of `7z e ... dinov2-base_-1_resid_49152.pt`

D_IN   = 768
D_SAE  = 49152
LAYER  = -1   # last residual block (index 11)

DATASETS = {
    "caltech101": {
        "root":    f"{DATA_ROOT}/caltech-101/Caltech101/101_ObjectCategories",
        "exclude": {"BACKGROUND_Google"},
    },
    "medmnist": {
        "root":    f"{DATA_ROOT}/pathmnist_imagefolder",
        "exclude": set(),
    },
    "eurosat": {
        "root":    f"{DATA_ROOT}/eurosat/2750",
        "exclude": set(),
    },
}

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

# ── SAE encoder-only wrapper ──────────────────────────────────────────────────

class SAEEncoder:
    """Holds W_enc and b_enc from the recovered checkpoint files."""

    def __init__(self, extracted_dir: str, device: str):
        w = np.fromfile(os.path.join(extracted_dir, "0"), dtype=np.float32)
        b = np.fromfile(os.path.join(extracted_dir, "1"), dtype=np.float32)

        assert w.shape == (D_IN * D_SAE,), f"Unexpected W_enc size: {w.shape}"
        assert b.shape == (D_SAE,),        f"Unexpected b_enc size: {b.shape}"

        self.W_enc = torch.from_numpy(w.reshape(D_IN, D_SAE)).to(device)
        self.b_enc = torch.from_numpy(b).to(device)
        self.device = device
        print(f"SAEEncoder loaded — W_enc {self.W_enc.shape}, b_enc {self.b_enc.shape}")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., D_IN]  →  feature_acts: [..., D_SAE]"""
        x = x.to(self.device).float()
        return F.relu(x @ self.W_enc + self.b_enc)


# ── Dataset helpers ───────────────────────────────────────────────────────────

class FilteredImageFolder(torch.utils.data.Dataset):
    def __init__(self, root: str, transform, exclude_classes=None):
        full_ds = datasets.ImageFolder(root=root, transform=transform)
        exclude = exclude_classes or set()
        keep    = [i for i, (_, lbl) in enumerate(full_ds.samples)
                   if full_ds.classes[lbl] not in exclude]
        kept_cls = sorted({full_ds.classes[full_ds.targets[i]] for i in keep})
        old2new  = {full_ds.class_to_idx[c]: ni for ni, c in enumerate(kept_cls)}
        self._ds    = full_ds
        self._idx   = keep
        self._map   = old2new
        self.classes     = kept_cls
        self.num_classes = len(kept_cls)

    def __len__(self):  return len(self._idx)
    def __getitem__(self, i):
        img, lbl = self._ds[self._idx[i]]
        return img, self._map[lbl]


def load_dataset(name: str, processor):
    """Return (dataset, num_classes) using DINOv2 processor as transform."""

    def hf_transform(img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        enc = processor(images=img, return_tensors="pt")
        return enc["pixel_values"].squeeze(0)   # [3, H, W]

    info = DATASETS[name]
    ds = FilteredImageFolder(info["root"], transform=hf_transform,
                             exclude_classes=info["exclude"])
    print(f"  {name}: {len(ds)} images, {ds.num_classes} classes")
    return ds


# ── Activation capture hook ───────────────────────────────────────────────────

class LayerCapture:
    """Captures the output hidden states of a DINOv2 encoder layer.

    HuggingFace Dinov2Layer returns a tuple whose first element is the
    hidden-state tensor of shape [B, seq_len, D_IN].
    """
    def __init__(self):
        self.output = None
        self._handle = None

    def register(self, layer):
        def _hook(module, inp, out):
            # out is a tuple; first element is the hidden-state tensor
            hs = out[0] if isinstance(out, tuple) else out
            self.output = hs.detach().float()
        self._handle = layer.register_forward_hook(_hook)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


# ── Core eval ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_sae(dino_model, sae_enc: SAEEncoder, dataset, name: str):
    """
    Returns dict:
        l0                      – mean active features per token (all tokens)
        l0_cls                  – mean active features on CLS token only
        dead_pct                – % of 49 152 features that never fired
        act_sparsity            – fraction of (feature, image) pairs active
        mean_max_act            – mean of per-feature maximum activation
        label_entropy_mean      – weighted mean entropy across active features
        label_entropy_norm      – entropy / log(C)  (0 = monosemantic)
        n_active_features       – count of features that fired ≥ once
        n_images                – total images processed
    """
    num_classes = dataset.num_classes

    capture = LayerCapture()
    layer_idx = LAYER if LAYER >= 0 else 12 + LAYER   # 11 for -1
    capture.register(dino_model.encoder.layer[layer_idx])

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=(DEVICE == "cuda"))

    total_l0          = 0.0
    total_l0_cls      = 0.0
    total_tokens      = 0
    total_images      = 0
    label_counts      = torch.zeros(D_SAE, num_classes, dtype=torch.float32)
    feature_fire_cnt  = torch.zeros(D_SAE, dtype=torch.float32)
    feature_max_act   = torch.zeros(D_SAE, dtype=torch.float32)

    dino_model.eval()

    for imgs, labels in tqdm(loader, desc=f"  {name}", leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.cpu()
        B      = imgs.shape[0]

        # Forward DINOv2 (hook captures last-layer residual)
        dino_model(pixel_values=imgs)
        acts_bsd = capture.output            # [B, seq, 768]
        seq      = acts_bsd.shape[1]

        # ---------- feature activations ----------
        # All tokens (CLS + patches)
        acts_flat   = acts_bsd.reshape(-1, D_IN)          # [B*seq, 768]
        feats_flat  = sae_enc.encode(acts_flat)            # [B*seq, D_SAE]
        feats_bsd   = feats_flat.reshape(B, seq, D_SAE)

        # L0 per image (mean over tokens)
        active      = (feats_bsd > 0).float()
        l0_per_img  = active.sum(dim=-1).mean(dim=-1)      # [B]
        total_l0   += l0_per_img.sum().item()

        # L0 on CLS token only (index 0)
        active_cls        = (feats_bsd[:, 0, :] > 0).float()  # [B, D_SAE]
        total_l0_cls     += active_cls.sum(dim=-1).mean().item() * B

        total_tokens     += B * seq
        total_images     += B

        # Max activation per feature (for this batch)
        batch_max = feats_flat.max(dim=0).values.cpu()
        torch.maximum(feature_max_act, batch_max, out=feature_max_act)

        # ---------- label-entropy accumulators ----------
        # For each image, which features fired on ANY token?
        active_per_img = (feats_bsd > 0).any(dim=1).float().cpu()  # [B, D_SAE]
        oh = F.one_hot(labels, num_classes=num_classes).float()     # [B, C]
        label_counts      += active_per_img.T @ oh                 # [D_SAE, C]
        feature_fire_cnt  += active_per_img.sum(0)                 # [D_SAE]

    capture.remove()

    # ── Dead features ─────────────────────────────────────────────────────────
    dead_mask = feature_fire_cnt == 0
    dead_pct  = dead_mask.float().mean().item() * 100.0
    act_sparsity = feature_fire_cnt.sum().item() / (D_SAE * total_images)

    # ── Label entropy ─────────────────────────────────────────────────────────
    active_mask   = ~dead_mask
    n_active      = active_mask.sum().item()
    counts_active = label_counts[active_mask]                            # [F_a, C]
    probs         = counts_active / counts_active.sum(dim=1, keepdim=True).clamp(min=1e-9)
    entropy       = -(probs * (probs + 1e-12).log()).sum(dim=1)          # [F_a]
    weights       = feature_fire_cnt[active_mask]
    label_entropy_mean = (entropy * weights / weights.sum()).sum().item()
    label_entropy_norm = label_entropy_mean / math.log(num_classes) if num_classes > 1 else 0.0

    label_entropy_unw      = entropy.mean().item()
    label_entropy_unw_norm = label_entropy_unw / math.log(num_classes) if num_classes > 1 else 0.0

    # ── Feature max-act distribution ─────────────────────────────────────────
    max_acts_arr = feature_max_act.numpy()
    active_max   = max_acts_arr[~dead_mask.numpy()]

    return {
        "l0":                       total_l0 / total_images,
        "l0_cls":                   total_l0_cls / total_images,
        "dead_pct":                 dead_pct,
        "act_sparsity":             act_sparsity,
        "mean_max_act":             float(active_max.mean()) if len(active_max) else 0.0,
        "p50_max_act":              float(np.percentile(active_max, 50)) if len(active_max) else 0.0,
        "p95_max_act":              float(np.percentile(active_max, 95)) if len(active_max) else 0.0,
        "label_entropy_mean":       label_entropy_mean,
        "label_entropy_norm":       label_entropy_norm,
        "label_entropy_unw_norm":   label_entropy_unw_norm,
        "n_active_features":        int(n_active),
        "num_classes":              num_classes,
        "n_images":                 total_images,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDevice: {DEVICE}")
    print(f"SAE: facebook/dinov2-base | layer {LAYER} | d_sae={D_SAE}")
    print("NOTE: W_dec/b_dec absent — reconstruction metrics SKIPPED")
    print("=" * 80)

    # Load SAE encoder
    sae_enc = SAEEncoder(EXTRACTED, DEVICE)

    # Load DINOv2
    print("\nLoading facebook/dinov2-base...")
    processor  = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(DEVICE)
    dino_model.eval()
    print("DINOv2 loaded.")

    results = []

    for ds_name in DATASETS:
        print(f"\n{'─'*60}")
        print(f"Dataset: {ds_name}")
        try:
            dataset = load_dataset(ds_name, processor)
        except Exception as e:
            print(f"  SKIPPED — could not load: {e}")
            continue

        metrics = eval_sae(dino_model, sae_enc, dataset, ds_name)
        row = {"dataset": ds_name, **metrics}
        results.append(row)

        print(f"  L0  (all tokens) : {metrics['l0']:.2f}")
        print(f"  L0  (CLS only)   : {metrics['l0_cls']:.2f}")
        print(f"  Dead features    : {metrics['dead_pct']:.2f}%  "
              f"({D_SAE - metrics['n_active_features']:,} / {D_SAE:,})")
        print(f"  Act sparsity     : {metrics['act_sparsity']:.4f}  "
              f"(fraction of feature×image pairs active)")
        print(f"  Max-act  mean    : {metrics['mean_max_act']:.4f}")
        print(f"  Max-act  p50/p95 : {metrics['p50_max_act']:.4f} / "
              f"{metrics['p95_max_act']:.4f}")
        print(f"  Label entropy (weighted, norm)   : "
              f"{metrics['label_entropy_norm']:.4f}  "
              f"[0=monosemantic, 1=uniform]")
        print(f"  Label entropy (unweighted, norm) : "
              f"{metrics['label_entropy_unw_norm']:.4f}")
        print(f"  Active features  : {metrics['n_active_features']:,} / {D_SAE:,}")
        print(f"  Images evaluated : {metrics['n_images']:,}")

    # ── Summary table ──────────────────────────────────────────────────────────
    if results:
        print("\n")
        print("=" * 110)
        print(f"{'Dataset':<14} {'L0':>8} {'L0_CLS':>8} {'Dead%':>7} "
              f"{'ActSpar':>9} {'MeanMaxA':>10} "
              f"{'H_wt_norm':>11} {'H_uw_norm':>11} {'ActiveF':>9}")
        print("=" * 110)
        for r in results:
            print(
                f"{r['dataset']:<14} "
                f"{r['l0']:>8.2f} "
                f"{r['l0_cls']:>8.2f} "
                f"{r['dead_pct']:>7.2f} "
                f"{r['act_sparsity']:>9.5f} "
                f"{r['mean_max_act']:>10.4f} "
                f"{r['label_entropy_norm']:>11.4f} "
                f"{r['label_entropy_unw_norm']:>11.4f} "
                f"{r['n_active_features']:>9,}"
            )
        print("=" * 110)

        # Save JSON
        out_path = "out/dinov2_sae_eval_metrics.json"
        os.makedirs("out", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

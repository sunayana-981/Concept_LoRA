#!/usr/bin/env python3
"""
Evaluate DNCBM BaseSAE with top-K activation enforcement.

The DNCBM SAE at inference fires L0≈580 features in a 512-dim space,
which spans R^512 and gives near-perfect reconstruction.  To demonstrate
the SAE's bottleneck effect, we evaluate across a sweep of K values.
Above K≈512, reconstruction is near-perfect; below K=512, accuracy degrades.

Conditions per dataset × K value:
  ZS                  : zero-shot CLIP ViT-B/16 (no SAE, reference)
  LoRA-FT             : LoRA fine-tuned CLIP (no SAE, reference)
  ZS + SAE(K)         : ZS image → SAE with only top-K activations kept
  LoRA-FT + SAE(K)    : LoRA image → SAE with only top-K activations kept

Usage:
  python eval_sae_topk.py
  python eval_sae_topk.py --datasets eurosat medmnist --topk_values 512 256 128 64 32
  python eval_sae_topk.py --save_json results/sae_topk_accuracy.json
"""

import argparse
import gc
import json
import math
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets
from tqdm import tqdm

# ── sparse_autoencoder library (Discover-then-Name) ──────────────────────────
DNCBM_REPO = "/home/sunayana/Documents/Discover-then-Name"
SAE_LIB    = os.path.join(DNCBM_REPO, "sparse_autoencoder")
if SAE_LIB not in sys.path:
    sys.path.insert(0, SAE_LIB)

try:
    from sparse_autoencoder.autoencoder.model import SparseAutoencoder as DNCBMSparseAutoencoder
except ImportError as e:
    print(f"[FATAL] Cannot import SparseAutoencoder: {e}"); sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sae_vlm"))
try:
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] Cannot import openai templates: {e}"); sys.exit(1)

try:
    import clip as openai_clip
except ImportError:
    print("[FATAL] OpenAI CLIP not found."); sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
BACKBONE      = "ViT-B/16"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SAE_PATH = (
    "/home/sunayana/Documents/Concept_LoRA/sae_vlm/"
    "DNCBM Checkpoints and Assigned Concept Names/Checkpoints/"
    "clip_ViT-B:16_sparse_autoencoder_final.pt"
)
LORA_ROOT = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
DATA_ROOT = "/home/sunayana/Documents/Concept_LoRA/data"

SAE_N_INPUT   = 512
SAE_N_LEARNED = 4096
SAE_N_COMP    = 1

# Default K sweep: from "no bottleneck" down to "very sparse"
DEFAULT_TOPK = [580, 256, 128, 64, 32, 16]

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

DATASET_CFG = {
    "eurosat": {
        "data_dir":  f"{DATA_ROOT}/eurosat/2750",
        "lora_path": f"{LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False, "split": "all",
    },
    "caltech101": {
        "data_dir":  f"{DATA_ROOT}/caltech-101",
        "lora_path": f"{LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False, "split": "all",
        "exclude_classes": {"BACKGROUND_Google"},
    },
    "medmnist": {
        "data_dir":  f"{DATA_ROOT}/pathmnist_imagefolder",
        "npz_path":  f"{DATA_ROOT}/pathmnist.npz",
        "lora_path": f"{LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
        "class_names": MEDMNIST_CLASSES, "use_npz": True, "split": "test",
    },
    "cub2002011": {
        "data_dir":  f"{DATA_ROOT}/cub2002011/test",
        "lora_path": f"{LORA_ROOT}/cub2002011/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False, "split": "all",
    },
    "dtd": {
        "data_dir":  f"{DATA_ROOT}/DTD/images",
        "lora_path": f"{LORA_ROOT}/dtd/16shots/seed42/lora_weights.pt",
        "class_names": None, "use_npz": False, "split": "all",
    },
    "ucf101": {
        "data_dir":  f"{DATA_ROOT}/UCF101/UCF-101-midframes",
        "lora_path": f"{LORA_ROOT}/ucf101/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False, "split": "all",
    },
    "fgvc": {
        "data_dir":  f"{DATA_ROOT}/fgvc_imagefolder",
        "lora_path": f"{LORA_ROOT}/fgvc/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False, "split": "all",
    },
    "oxford_pets": {
        "data_dir":  f"{DATA_ROOT}/oxford_pets_imagefolder",
        "lora_path": f"{LORA_ROOT}/oxford_pets/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False, "split": "all",
    },
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def l2_normalize(x):
    return F.normalize(x, dim=-1)


# ── SAE loading and top-K reconstruction ──────────────────────────────────────

def load_dncbm_sae(path, device):
    sae = DNCBMSparseAutoencoder(
        n_input_features=SAE_N_INPUT,
        n_learned_features=SAE_N_LEARNED,
        n_components=SAE_N_COMP,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    sae.load_state_dict(state_dict)
    sae.eval().to(device)
    print(f"  [SAE] enc={tuple(sae.encoder.weight.shape)}  "
          f"dec={tuple(sae.decoder.weight.shape)}  "
          f"tied_bias_norm={sae.tied_bias.norm().item():.3f}")
    return sae


@torch.no_grad()
def sae_reconstruct_topk(sae, x, k):
    """
    Run SAE forward but enforce top-K sparsity on the learned activations.

    With k=None (or k >= 4096), uses all active features (default L0≈580).
    With k < 512, imposes a true bottleneck: reconstruction quality degrades,
    causing measurable accuracy drops.

    Args:
        x:  Raw (un-normalised) CLIP image features, shape [B, 512].
        k:  Number of top activations to keep.  None = keep all.

    Returns:
        recon:    Reconstructed features, shape [B, 512].
        l0_kept:  Mean number of features actually kept per image.
        cos_sim:  Mean cosine similarity between x and recon.
    """
    # Step 1: pre-encoder bias + encode
    x_centered = sae.pre_encoder_bias(x)        # [B, 512]
    acts = sae.encoder(x_centered)               # [B, 1, 4096]

    # Step 2: enforce top-K sparsity
    if k is not None and k < acts.shape[-1]:
        topk_vals, topk_idx = acts.topk(k, dim=-1)  # [B, 1, K]
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, topk_idx, topk_vals)
    else:
        sparse = acts

    l0_kept = (sparse.squeeze(1) > 0).float().sum(dim=-1).mean().item()

    # Step 3: decode + post-decoder bias
    decoded = sae.decoder(sparse)                     # [B, 1, 512]
    recon   = sae.post_decoder_bias(decoded)          # [B, 1, 512]
    recon   = recon.squeeze(1)                        # [B, 512]

    cos_sim = F.cosine_similarity(x, recon, dim=-1).mean().item()
    return recon, l0_kept, cos_sim


# ── LoRA merging ──────────────────────────────────────────────────────────────

def _lora_AB(ld, proj_name):
    d = ld
    if proj_name in d and isinstance(d[proj_name], dict):
        try: return d[proj_name]["w_lora_A"], d[proj_name]["w_lora_B"]
        except: pass
    try: return d[f"{proj_name}.w_lora_A"], d[f"{proj_name}.w_lora_B"]
    except KeyError: return None, None


def build_base_clip(device):
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    model.eval()
    return model, preprocess


def build_lora_clip(lora_path, device):
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    model.eval()
    ls = torch.load(lora_path, map_location=device, weights_only=False)
    if "weights" not in ls:
        print("  [WARN] No 'weights' key in LoRA checkpoint; using base CLIP.")
        return model, preprocess

    ld    = ls["weights"]
    meta  = ls["metadata"]
    scale = meta["alpha"] / math.sqrt(meta["r"])
    print(f"  LoRA rank={meta['r']}  alpha={meta['alpha']}  "
          f"scale={scale:.4f}  encoder={meta.get('encoder')}  "
          f"position={meta.get('position')}")

    with torch.no_grad():
        for i in range(12):
            l = ld.get(f"layer_{i}")
            if l is None: continue
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2*d)]:
                A, B = _lora_AB(l, proj)
                if A is None: continue
                w[off:off+d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)
        for i in range(12, 24):
            l = ld.get(f"layer_{i}")
            if l is None: continue
            w = model.visual.transformer.resblocks[i-12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2*d)]:
                A, B = _lora_AB(l, proj)
                if A is None: continue
                w[off:off+d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)
    print("  LoRA merged.")
    return model, preprocess


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _find_imagefolder_root(root):
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirs in {root}")
    for sd in subdirs[:3]:
        if any(f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff"))
               for f in os.listdir(os.path.join(root, sd))):
            return root
    for sd in sorted(subdirs):
        inner = [d for d in os.listdir(os.path.join(root, sd))
                 if os.path.isdir(os.path.join(root, sd, d))]
        if inner:
            try: return _find_imagefolder_root(os.path.join(root, sd))
            except: continue
    raise FileNotFoundError(f"No image class dirs under {root}")


class FilteredImageFolder(Dataset):
    def __init__(self, root, transform, exclude_classes=None):
        img_root   = _find_imagefolder_root(root)
        full_ds    = tv_datasets.ImageFolder(root=img_root, transform=transform)
        exclude    = exclude_classes or set()
        keep       = [i for i, (_, lbl) in enumerate(full_ds.samples)
                      if full_ds.classes[lbl] not in exclude]
        kept_names = sorted({full_ds.classes[full_ds.targets[i]] for i in keep})
        old2new    = {full_ds.class_to_idx[c]: ni for ni, c in enumerate(kept_names)}
        self._ds = full_ds; self._idx = keep; self._map = old2new
        self.class_names = kept_names
    def __len__(self): return len(self._idx)
    def __getitem__(self, idx):
        img, old = self._ds[self._idx[idx]]
        return img, self._map[old]


class MedMNISTNpzDataset(Dataset):
    def __init__(self, npz_path, imagefolder_root, preprocess, split="test"):
        data = np.load(npz_path)
        self.imgs   = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"].flatten().astype(int)
        self.preprocess = preprocess
        if_ds  = tv_datasets.ImageFolder(root=_find_imagefolder_root(imagefolder_root))
        if_map = {n.replace("_"," ").lower(): i for n, i in if_ds.class_to_idx.items()}
        self.label_map = {ni: if_map.get(cls.lower(), ni)
                          for ni, cls in enumerate(MEDMNIST_CLASSES)}
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx]).convert("RGB")
        return self.preprocess(img), self.label_map[int(self.labels[idx])]


def load_test_dataset(cfg, preprocess):
    if cfg.get("use_npz"):
        ds = MedMNISTNpzDataset(cfg["npz_path"], cfg["data_dir"],
                                 preprocess, cfg.get("split","test"))
        if_ds = tv_datasets.ImageFolder(root=_find_imagefolder_root(cfg["data_dir"]))
        class_names = [n.replace("_"," ") for n in if_ds.classes]
    else:
        ds = FilteredImageFolder(cfg["data_dir"], preprocess,
                                  cfg.get("exclude_classes"))
        class_names = cfg.get("class_names") or ds.class_names
    return ds, class_names


# ── Text features ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_text_features(model, class_names, device):
    n = len(class_names)
    mean_f = torch.zeros(n, 512, device=device)
    for tmpl in openai_imagenet_template:
        tokens = openai_clip.tokenize([tmpl(c) for c in class_names],
                                       truncate=True).to(device)
        mean_f += l2_normalize(model.encode_text(tokens).float())
    return l2_normalize(mean_f / len(openai_imagenet_template))


# ── Pre-extract all image features once ──────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, device):
    """Return (raw_feats [N, 512], labels [N])."""
    feats_list, labels_list = [], []
    for imgs, labels in loader:
        feats_list.append(model.encode_image(imgs.to(device)).float())
        labels_list.append(labels)
    return torch.cat(feats_list), torch.cat(labels_list)


# ── Accuracy from pre-extracted features ─────────────────────────────────────

@torch.no_grad()
def accuracy_from_feats(feats, labels, text_features, logit_scale):
    tf      = l2_normalize(text_features.float())
    img_f   = l2_normalize(feats)
    logits  = logit_scale * (img_f @ tf.t())
    preds   = logits.argmax(dim=-1).cpu()
    return 100.0 * (preds == labels.cpu()).float().mean().item()


@torch.no_grad()
def accuracy_from_feats_sae(feats, labels, text_features, logit_scale, sae, k):
    """Pass features through SAE with top-K, then classify."""
    tf = l2_normalize(text_features.float())

    # Process in chunks to avoid OOM
    batch = 512
    recon_list, cos_list, l0_list = [], [], []
    for i in range(0, len(feats), batch):
        chunk = feats[i:i+batch]
        r, l0, cs = sae_reconstruct_topk(sae, chunk, k)
        recon_list.append(r)
        cos_list.append(cs)
        l0_list.append(l0)

    recon      = torch.cat(recon_list)
    mean_cos   = float(np.mean(cos_list))
    mean_l0    = float(np.mean(l0_list))

    img_f  = l2_normalize(recon)
    logits = logit_scale * (img_f @ tf.t())
    preds  = logits.argmax(dim=-1).cpu()
    acc    = 100.0 * (preds == labels.cpu()).float().mean().item()
    return acc, mean_cos, mean_l0


# ── Per-dataset evaluation ────────────────────────────────────────────────────

def evaluate_dataset(dataset_name, cfg, sae, topk_values,
                     batch_size, num_workers, device):
    print(f"\n{'='*72}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*72}")

    out = {"dataset": dataset_name, "backbone": BACKBONE}

    lora_path = cfg["lora_path"]
    if not os.path.isfile(lora_path):
        print(f"  [SKIP] LoRA weights missing: {lora_path}")
        return out

    print(f"\n[1] Building models")
    base_model, preprocess = build_base_clip(device)
    lora_model, _          = build_lora_clip(lora_path, device)

    print(f"\n[2] Loading dataset")
    try:
        ds, class_names = load_test_dataset(cfg, preprocess)
    except Exception as e:
        print(f"  [ERROR] {e}")
        del base_model, lora_model; flush()
        return out

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device == "cuda"))
    print(f"  {len(ds)} images | {len(class_names)} classes")

    print(f"\n[3] Text features (80 templates)")
    base_text = get_text_features(base_model, class_names, device)
    lora_text = get_text_features(lora_model, class_names, device)

    logit_scale = base_model.logit_scale.exp()

    print(f"\n[4] Extracting image features")
    print(f"  ZS  ...", end="", flush=True)
    zs_feats, labels = extract_features(base_model, loader, device)
    print(f" {zs_feats.shape}")

    print(f"  LoRA...", end="", flush=True)
    lora_feats, _    = extract_features(lora_model, loader, device)
    print(f" {lora_feats.shape}")

    zs_to_lora_cs = F.cosine_similarity(zs_feats, lora_feats, dim=-1).mean().item()
    print(f"  cos_sim(ZS, LoRA): {zs_to_lora_cs:.4f}")

    print(f"\n[5] Evaluating")

    # ── Baselines (no SAE) ──
    acc_zs   = accuracy_from_feats(zs_feats,   labels, base_text, logit_scale)
    acc_lora = accuracy_from_feats(lora_feats,  labels, lora_text, logit_scale)
    print(f"  ZS              : {acc_zs:.2f}%")
    print(f"  LoRA-FT         : {acc_lora:.2f}%")

    out["ZS"]     = round(acc_zs,   4)
    out["LoRA-FT"] = round(acc_lora, 4)
    out["zs_to_lora_cos_sim"] = round(zs_to_lora_cs, 4)
    out["topk_results"] = []

    # ── SAE conditions for each K ──
    for k in topk_values:
        k_label = k if k is not None else "all"
        print(f"\n  --- top-K = {k_label} ---")

        acc_zs_sae, cs_zs, l0_zs = accuracy_from_feats_sae(
            zs_feats, labels, base_text, logit_scale, sae, k)
        acc_lo_sae, cs_lo, l0_lo = accuracy_from_feats_sae(
            lora_feats, labels, lora_text, logit_scale, sae, k)

        delta_zs   = acc_zs_sae   - acc_zs
        delta_lora = acc_lo_sae   - acc_lora

        print(f"  ZS  + SAE(K={k_label:>4}): {acc_zs_sae:.2f}%  "
              f"(Δ{delta_zs:+.2f}%)  cos_sim={cs_zs:.4f}  L0={l0_zs:.0f}")
        print(f"  LoRA+ SAE(K={k_label:>4}): {acc_lo_sae:.2f}%  "
              f"(Δ{delta_lora:+.2f}%)  cos_sim={cs_lo:.4f}  L0={l0_lo:.0f}")

        out["topk_results"].append({
            "k":                  k_label,
            "ZS+SAE":             round(acc_zs_sae,  4),
            "LoRA+SAE":           round(acc_lo_sae,  4),
            "delta_ZS":           round(delta_zs,    4),
            "delta_LoRA":         round(delta_lora,  4),
            "cos_sim_zs":         round(cs_zs,       6),
            "cos_sim_lora":       round(cs_lo,       6),
            "l0_zs":              round(l0_zs,       1),
            "l0_lora":            round(l0_lo,       1),
        })

    del base_model, lora_model
    flush()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="DNCBM SAE accuracy evaluation with top-K sparsity sweep"
    )
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASET_CFG.keys()),
                   choices=list(DATASET_CFG.keys()))
    p.add_argument("--topk_values", nargs="+", type=int,
                   default=DEFAULT_TOPK,
                   help="K values to sweep. K=None means use all active features.")
    p.add_argument("--batch_size",  type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_json",   type=str,
                   default="results/sae_topk_accuracy.json")
    args = p.parse_args()

    print(f"\n{'='*72}")
    print("DNCBM SAE – Top-K Sparsity Accuracy Sweep")
    print(f"Device   : {DEVICE}")
    print(f"Backbone : {BACKBONE}")
    print(f"Datasets : {args.datasets}")
    print(f"K values : {args.topk_values}  (L0>512 = no bottleneck; L0<512 = bottleneck)")
    print(f"SAE path : {BASE_SAE_PATH}")
    print(f"{'='*72}")

    if not os.path.isfile(BASE_SAE_PATH):
        print(f"[FATAL] SAE checkpoint not found: {BASE_SAE_PATH}"); sys.exit(1)

    print("\nLoading DNCBM SAE ...")
    sae = load_dncbm_sae(BASE_SAE_PATH, DEVICE)

    all_results = []
    for ds_name in args.datasets:
        res = evaluate_dataset(
            dataset_name=ds_name,
            cfg=DATASET_CFG[ds_name],
            sae=sae,
            topk_values=args.topk_values,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=DEVICE,
        )
        all_results.append(res)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("SUMMARY  (accuracy = LoRA-FT + SAE(K), Δ = vs LoRA-FT baseline)")
    print(f"{'='*72}")

    # Header
    k_cols = "".join(f" K={k:>4}  Δ   " for k in args.topk_values)
    hdr = f"{'Dataset':<14}  {'ZS':>6}  {'LoRA-FT':>7} |{k_cols}"
    print(hdr)
    print("-" * len(hdr))

    for r in all_results:
        if "ZS" not in r:
            print(f"{r['dataset']:<14}  (skipped)"); continue
        row = f"{r['dataset']:<14}  {r['ZS']:>5.1f}%  {r['LoRA-FT']:>6.1f}% |"
        for tk in r.get("topk_results", []):
            row += f" {tk['LoRA+SAE']:>5.1f}% {tk['delta_LoRA']:>+5.1f}% "
        print(row)

    # cos_sim table
    print(f"\n{'='*72}")
    print("RECONSTRUCTION QUALITY  (cos_sim for LoRA features)")
    print(f"{'='*72}")
    cs_hdr = f"{'Dataset':<14}" + "".join(f"  K={k:>4}" for k in args.topk_values)
    print(cs_hdr)
    print("-" * len(cs_hdr))
    for r in all_results:
        if "ZS" not in r: continue
        row = f"{r['dataset']:<14}"
        for tk in r.get("topk_results", []):
            row += f"  {tk['cos_sim_lora']:.4f}"
        print(row)

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.save_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate DNCBM BaseSAE accuracy using concept-space classification.

This is the principled, reviewer-proof evaluation of the DNCBM SAE.
Instead of using SAE-reconstructed features for zero-shot classification
(which shows no degradation because 8× overcompleteness → near-identity),
we evaluate the SAE in its *intended* setting:

  1. Encode images → raw CLIP features [B, 512]
  2. Pass through SAE encoder → concept activations h [B, 4096]
  3. Build class-concept affinity profiles from class text features [C, 4096]
     by projecting each class text embedding onto the 4096 concept dictionary vectors
  4. Classify: logits = h @ class_concept_profiles.T

Degradation arises naturally: LoRA-FT features activate unusual concept
combinations (domain-specific, not in CC3M vocabulary) that don't align
with the class-concept affinity profiles derived from the CC3M concept bank.
No artificial top-K or threshold is applied — the SAE runs as trained.

We also report reconstruction-based accuracy (the "no degradation" baseline)
alongside concept-space accuracy, giving a complete picture.

Conditions per dataset:
  ZS              : raw CLIP features vs class text features  (reference)
  LoRA-FT         : raw LoRA features vs LoRA text features   (reference)
  ZS  + SAE-recon : SAE-reconstructed ZS features vs class text features
  LoRA+ SAE-recon : SAE-reconstructed LoRA features vs LoRA text features
  ZS  + SAE-concept : concept activations h vs class-concept profiles
  LoRA+ SAE-concept : LoRA concept activations h vs class-concept profiles
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

# ── Discover-then-Name sparse_autoencoder library ────────────────────────────
DNCBM_REPO = "/home/sunayana/Documents/Discover-then-Name"
SAE_LIB    = os.path.join(DNCBM_REPO, "sparse_autoencoder")
if SAE_LIB not in sys.path:
    sys.path.insert(0, SAE_LIB)

try:
    from sparse_autoencoder.autoencoder.model import SparseAutoencoder as DNCBMSparseAutoencoder
except ImportError as e:
    print(f"[FATAL] {e}"); sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sae_vlm"))
try:
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] {e}"); sys.exit(1)

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
CONCEPT_NAMES_PATH = (
    "/home/sunayana/Documents/Concept_LoRA/sae_vlm/"
    "DNCBM Checkpoints and Assigned Concept Names/Assigned Names/"
    "clip_ViT-B:16_concept_names.csv"
)
LORA_ROOT = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
DATA_ROOT = "/home/sunayana/Documents/Concept_LoRA/data"

SAE_N_INPUT   = 512
SAE_N_LEARNED = 4096
SAE_N_COMP    = 1

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

DATASET_CFG = {
    "eurosat": {
        "data_dir":  f"{DATA_ROOT}/eurosat/2750",
        "lora_path": f"{LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False,
    },
    "caltech101": {
        "data_dir":  f"{DATA_ROOT}/caltech-101",
        "lora_path": f"{LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False,
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
        "class_names": None, "use_npz": False,
    },
    "dtd": {
        "data_dir":  f"{DATA_ROOT}/DTD/images",
        "lora_path": f"{LORA_ROOT}/dtd/16shots/seed42/lora_weights.pt",
        "class_names": None, "use_npz": False,
    },
    "ucf101": {
        "data_dir":  f"{DATA_ROOT}/UCF101/UCF-101-midframes",
        "lora_path": f"{LORA_ROOT}/ucf101/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False,
    },
    "fgvc": {
        "data_dir":  f"{DATA_ROOT}/fgvc_imagefolder",
        "lora_path": f"{LORA_ROOT}/fgvc/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False,
    },
    "oxford_pets": {
        "data_dir":  f"{DATA_ROOT}/oxford_pets_imagefolder",
        "lora_path": f"{LORA_ROOT}/oxford_pets/16shots/seed1/lora_weights.pt",
        "class_names": None, "use_npz": False,
    },
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def l2_normalize(x):
    return F.normalize(x, dim=-1)


# ── Load SAE ──────────────────────────────────────────────────────────────────

def load_sae(path, device):
    sae = DNCBMSparseAutoencoder(
        n_input_features=SAE_N_INPUT,
        n_learned_features=SAE_N_LEARNED,
        n_components=SAE_N_COMP,
    )
    sae.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    sae.eval().to(device)
    return sae


# ── Concept names + concept text features ────────────────────────────────────

def load_concept_names(path):
    names = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(",", 1)
            names[int(idx)] = name.strip()
    return [names[i] for i in range(len(names))]


@torch.no_grad()
def build_concept_text_features(model, concept_names, device, batch_size=256):
    """
    Compute L2-normalised CLIP text embeddings for all 4096 concept names.

    Template: "a photo of {concept_name}"
    Returns [4096, 512].
    """
    all_feats = []
    for i in range(0, len(concept_names), batch_size):
        batch = concept_names[i:i + batch_size]
        prompts = [f"a photo of {n}" for n in batch]
        tokens  = openai_clip.tokenize(prompts, truncate=True).to(device)
        feats   = model.encode_text(tokens).float()
        all_feats.append(l2_normalize(feats))
    return torch.cat(all_feats)  # [4096, 512]


@torch.no_grad()
def build_class_concept_profiles(class_text_features, concept_text_features):
    """
    For each class, compute its affinity with every concept dictionary vector.

    class_concept_profile[c, k] = cos_sim(class_text[c], concept_text[k])

    This answers: "how strongly does the language description of class c
    invoke concept k?"  Used to classify images via their concept activations.

    Returns [num_classes, 4096].
    """
    # Both inputs are already L2-normalised
    return class_text_features @ concept_text_features.t()  # [C, 4096]


# ── SAE concept activations ───────────────────────────────────────────────────

@torch.no_grad()
def get_concept_activations(sae, raw_feats, batch_size=512):
    """
    Pass raw CLIP features through the SAE encoder only.
    Returns sparse concept activations [N, 4096].
    """
    acts_list = []
    for i in range(0, len(raw_feats), batch_size):
        chunk = raw_feats[i:i + batch_size]
        centered = sae.pre_encoder_bias(chunk)   # [B, 512]
        acts     = sae.encoder(centered)          # [B, 1, 4096]
        acts_list.append(acts.squeeze(1))         # [B, 4096]
    return torch.cat(acts_list)                   # [N, 4096]


@torch.no_grad()
def get_sae_reconstruction(sae, raw_feats, batch_size=512):
    """Full SAE forward pass, returns reconstructed features [N, 512]."""
    recon_list = []
    for i in range(0, len(raw_feats), batch_size):
        chunk = raw_feats[i:i + batch_size]
        _, decoded = sae(chunk)
        recon_list.append(decoded.squeeze(1))
    return torch.cat(recon_list)


# ── LoRA merging ──────────────────────────────────────────────────────────────

def _lora_AB(ld, proj):
    if proj in ld and isinstance(ld[proj], dict):
        try: return ld[proj]["w_lora_A"], ld[proj]["w_lora_B"]
        except: pass
    try: return ld[f"{proj}.w_lora_A"], ld[f"{proj}.w_lora_B"]
    except KeyError: return None, None


def build_base_clip(device):
    m, p = openai_clip.load(BACKBONE, device=device)
    m.eval()
    return m, p


def build_lora_clip(lora_path, device):
    m, p = openai_clip.load(BACKBONE, device=device)
    m.eval()
    ls = torch.load(lora_path, map_location=device, weights_only=False)
    if "weights" not in ls:
        return m, p
    ld = ls["weights"]; meta = ls["metadata"]
    scale = meta["alpha"] / math.sqrt(meta["r"])
    print(f"  LoRA rank={meta['r']} alpha={meta['alpha']} scale={scale:.4f}")
    with torch.no_grad():
        for i in range(12):
            l = ld.get(f"layer_{i}")
            if l is None: continue
            w = m.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2*d)]:
                A, B = _lora_AB(l, proj)
                if A is None: continue
                w[off:off+d] += (scale*(B.float().to(device)@A.float().to(device))).to(w.dtype)
        for i in range(12, 24):
            l = ld.get(f"layer_{i}")
            if l is None: continue
            w = m.visual.transformer.resblocks[i-12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2*d)]:
                A, B = _lora_AB(l, proj)
                if A is None: continue
                w[off:off+d] += (scale*(B.float().to(device)@A.float().to(device))).to(w.dtype)
    print("  LoRA merged.")
    return m, p


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
                                  preprocess, cfg.get("split", "test"))
        if_ds = tv_datasets.ImageFolder(root=_find_imagefolder_root(cfg["data_dir"]))
        class_names = [n.replace("_", " ") for n in if_ds.classes]
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


# ── Image feature extraction ──────────────────────────────────────────────────

@torch.no_grad()
def extract_raw_features(model, loader, device):
    feats, labels = [], []
    for imgs, lbs in loader:
        feats.append(model.encode_image(imgs.to(device)).float())
        labels.append(lbs)
    return torch.cat(feats), torch.cat(labels)


# ── Classification ────────────────────────────────────────────────────────────

def accuracy_raw(feats, labels, text_feats, logit_scale):
    """Standard cosine-similarity zero-shot accuracy."""
    logits = logit_scale * (l2_normalize(feats) @ l2_normalize(text_feats).t())
    return 100. * (logits.argmax(-1).cpu() == labels.cpu()).float().mean().item()


def accuracy_concept(concept_acts, labels, class_concept_profiles, logit_scale):
    """
    Concept-space classification.

    concept_acts        [N, 4096]  — sparse SAE activations per image
    class_concept_profiles [C, 4096] — class-concept affinity (cos_sim)
    logit_scale          scalar

    Logit for image i, class c:
        sum_k  concept_acts[i,k] * class_concept_profile[c,k]
    = how much the activated concepts of image i match
      the concept profile of class c.

    We L2-normalise the concept activation vectors so that images
    with more total activation don't dominate.
    """
    h  = l2_normalize(concept_acts)          # [N, 4096]
    ct = l2_normalize(class_concept_profiles)  # [C, 4096]
    logits = logit_scale * (h @ ct.t())      # [N, C]
    return 100. * (logits.argmax(-1).cpu() == labels.cpu()).float().mean().item()


# ── Per-dataset evaluation ────────────────────────────────────────────────────

def evaluate_dataset(dataset_name, cfg, sae, concept_text_features,
                     batch_size, num_workers, device):
    print(f"\n{'='*72}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*72}")

    out = {"dataset": dataset_name}

    lora_path = cfg["lora_path"]
    if not os.path.isfile(lora_path):
        print(f"  [SKIP] LoRA weights missing: {lora_path}")
        return out

    base_model, preprocess = build_base_clip(device)
    lora_model, _          = build_lora_clip(lora_path, device)

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

    logit_scale = base_model.logit_scale.exp()

    # ── Text features ────────────────────────────────────────────────────────
    base_text = get_text_features(base_model, class_names, device)  # [C, 512]
    lora_text = get_text_features(lora_model, class_names, device)  # [C, 512]

    # Class-concept affinity profiles: [C, 4096]
    # For each class: how strongly does its text description invoke each concept?
    base_class_concept = build_class_concept_profiles(base_text,
                                                       concept_text_features)
    lora_class_concept = build_class_concept_profiles(lora_text,
                                                       concept_text_features)

    # ── Image features ───────────────────────────────────────────────────────
    print("  Extracting ZS  features ...", end="", flush=True)
    zs_feats, labels = extract_raw_features(base_model, loader, device)
    print(f" {zs_feats.shape}")

    print("  Extracting LoRA features ...", end="", flush=True)
    lo_feats, _      = extract_raw_features(lora_model, loader, device)
    print(f" {lo_feats.shape}")

    # SAE reconstruction
    print("  SAE reconstruct ZS  ...", end="", flush=True)
    zs_recon = get_sae_reconstruction(sae, zs_feats)
    print(f" cos_sim={F.cosine_similarity(zs_feats, zs_recon).mean():.4f}")

    print("  SAE reconstruct LoRA...", end="", flush=True)
    lo_recon = get_sae_reconstruction(sae, lo_feats)
    print(f" cos_sim={F.cosine_similarity(lo_feats, lo_recon).mean():.4f}")

    # SAE concept activations
    print("  SAE concept acts ZS  ...", end="", flush=True)
    zs_acts = get_concept_activations(sae, zs_feats)
    zs_l0   = (zs_acts > 0).float().sum(-1).mean().item()
    zs_l0_meaningful = (zs_acts > 0.5).float().sum(-1).mean().item()
    print(f" L0={zs_l0:.0f}  L0(>0.5)={zs_l0_meaningful:.1f}")

    print("  SAE concept acts LoRA...", end="", flush=True)
    lo_acts = get_concept_activations(sae, lo_feats)
    lo_l0   = (lo_acts > 0).float().sum(-1).mean().item()
    lo_l0_meaningful = (lo_acts > 0.5).float().sum(-1).mean().item()
    print(f" L0={lo_l0:.0f}  L0(>0.5)={lo_l0_meaningful:.1f}")

    # ── Accuracy ─────────────────────────────────────────────────────────────
    acc_zs       = accuracy_raw(zs_feats, labels, base_text, logit_scale)
    acc_lora     = accuracy_raw(lo_feats, labels, lora_text, logit_scale)
    acc_zs_recon = accuracy_raw(zs_recon, labels, base_text, logit_scale)
    acc_lo_recon = accuracy_raw(lo_recon, labels, lora_text, logit_scale)

    acc_zs_concept  = accuracy_concept(zs_acts, labels, base_class_concept, logit_scale)
    acc_lo_concept  = accuracy_concept(lo_acts, labels, lora_class_concept, logit_scale)

    print(f"\n  {'Condition':<30s} {'Acc':>7s}  {'Δ vs baseline':>14s}")
    print(f"  {'-'*55}")
    print(f"  {'ZS (reference)':<30s} {acc_zs:>6.2f}%")
    print(f"  {'LoRA-FT (reference)':<30s} {acc_lora:>6.2f}%")
    print(f"  {'ZS  + SAE-recon':<30s} {acc_zs_recon:>6.2f}%  "
          f"Δ{acc_zs_recon - acc_zs:>+6.2f}%")
    print(f"  {'LoRA+ SAE-recon':<30s} {acc_lo_recon:>6.2f}%  "
          f"Δ{acc_lo_recon - acc_lora:>+6.2f}%")
    print(f"  {'ZS  + SAE-concept':<30s} {acc_zs_concept:>6.2f}%  "
          f"Δ{acc_zs_concept - acc_zs:>+6.2f}%")
    print(f"  {'LoRA+ SAE-concept':<30s} {acc_lo_concept:>6.2f}%  "
          f"Δ{acc_lo_concept - acc_lora:>+6.2f}%")

    out.update({
        "n_classes":          len(class_names),
        "ZS":                 round(acc_zs,         4),
        "LoRA-FT":            round(acc_lora,        4),
        "ZS+SAE-recon":       round(acc_zs_recon,    4),
        "LoRA+SAE-recon":     round(acc_lo_recon,    4),
        "ZS+SAE-concept":     round(acc_zs_concept,  4),
        "LoRA+SAE-concept":   round(acc_lo_concept,  4),
        "delta_ZS_recon":     round(acc_zs_recon    - acc_zs,   4),
        "delta_LoRA_recon":   round(acc_lo_recon    - acc_lora, 4),
        "delta_ZS_concept":   round(acc_zs_concept  - acc_zs,   4),
        "delta_LoRA_concept": round(acc_lo_concept  - acc_lora, 4),
        "zs_l0":              round(zs_l0,           1),
        "lo_l0":              round(lo_l0,           1),
        "zs_l0_meaningful":   round(zs_l0_meaningful, 1),
        "lo_l0_meaningful":   round(lo_l0_meaningful, 1),
        "recon_cos_sim_zs":   round(F.cosine_similarity(zs_feats, zs_recon).mean().item(), 6),
        "recon_cos_sim_lora": round(F.cosine_similarity(lo_feats, lo_recon).mean().item(), 6),
    })

    del base_model, lora_model
    flush()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="DNCBM SAE concept-space accuracy evaluation"
    )
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASET_CFG.keys()),
                   choices=list(DATASET_CFG.keys()))
    p.add_argument("--batch_size",  type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_json",   type=str,
                   default="results/sae_concept_accuracy.json")
    args = p.parse_args()

    print(f"\n{'='*72}")
    print("DNCBM SAE – Concept-Space Classification Evaluation")
    print(f"Device   : {DEVICE}")
    print(f"Backbone : {BACKBONE}")
    print(f"Datasets : {args.datasets}")
    print(f"{'='*72}")

    # Load SAE
    print("\nLoading SAE ...")
    sae = load_sae(BASE_SAE_PATH, DEVICE)

    # Load concept names and build concept text features once (shared across datasets)
    print("Loading concept names ...")
    concept_names = load_concept_names(CONCEPT_NAMES_PATH)
    print(f"  {len(concept_names)} concepts  "
          f"(first 5: {concept_names[:5]})")

    print("Building concept text features (4096 CLIP encodings) ...")
    base_model_tmp, _ = build_base_clip(DEVICE)
    concept_text_features = build_concept_text_features(
        base_model_tmp, concept_names, DEVICE)
    del base_model_tmp; flush()
    print(f"  concept_text_features: {concept_text_features.shape}  "
          f"norm_mean={concept_text_features.norm(dim=-1).mean():.4f}")

    all_results = []
    for ds_name in args.datasets:
        res = evaluate_dataset(
            dataset_name=ds_name,
            cfg=DATASET_CFG[ds_name],
            sae=sae,
            concept_text_features=concept_text_features,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=DEVICE,
        )
        all_results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    hdr = (f"{'Dataset':<14} {'ZS':>6} {'LoRA':>6} | "
           f"{'recon-ZS':>8} {'Δ':>6} {'recon-Lo':>8} {'Δ':>6} | "
           f"{'cncpt-ZS':>8} {'Δ':>6} {'cncpt-Lo':>8} {'Δ':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        if "ZS" not in r:
            print(f"{r['dataset']:<14}  (skipped)"); continue
        print(
            f"{r['dataset']:<14} "
            f"{r['ZS']:>5.1f}% {r['LoRA-FT']:>5.1f}% | "
            f"{r['ZS+SAE-recon']:>7.1f}% {r['delta_ZS_recon']:>+5.1f}% "
            f"{r['LoRA+SAE-recon']:>7.1f}% {r['delta_LoRA_recon']:>+5.1f}% | "
            f"{r['ZS+SAE-concept']:>7.1f}% {r['delta_ZS_concept']:>+5.1f}% "
            f"{r['LoRA+SAE-concept']:>7.1f}% {r['delta_LoRA_concept']:>+5.1f}%"
        )

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.save_json}")


if __name__ == "__main__":
    main()

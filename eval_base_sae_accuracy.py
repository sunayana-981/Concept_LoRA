#!/usr/bin/env python3
"""
Evaluate four conditions using the DNCBM BaseSAE (sparse_autoencoder_pytorch format).

Conditions per dataset:
  - ZS                  : zero-shot CLIP ViT-B/16
  - LoRA-FT             : LoRA fine-tuned CLIP
  - ZS + BaseSAE-1      : ZS image features → DNCBM SAE reconstruction → classify
  - LoRA-FT + BaseSAE-1 : LoRA image features → DNCBM SAE reconstruction → classify

Architecture details (from Discover-then-Name repo):
  - SAE is trained on raw (un-normalised) encode_image() output (512-dim for ViT-B/16)
  - Checkpoint IS the state_dict of SparseAutoencoder(n_input_features=512,
      n_learned_features=4096, n_components=1)
  - Forward: input [B, 512] → output (learned_acts [B,1,4096], decoded [B,1,512])
  - decoded.squeeze(1) is used as the reconstructed feature for classification

Usage:
  python eval_base_sae_accuracy.py
  python eval_base_sae_accuracy.py --datasets eurosat cub2002011
  python eval_base_sae_accuracy.py --save_json results/base_sae_accuracy.json
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

# ── sparse_autoencoder library (Discover-then-Name repo) ─────────────────────
DNCBM_REPO = "/home/sunayana/Documents/Discover-then-Name"
SAE_LIB    = os.path.join(DNCBM_REPO, "sparse_autoencoder")
if SAE_LIB not in sys.path:
    sys.path.insert(0, SAE_LIB)

try:
    from sparse_autoencoder.autoencoder.model import SparseAutoencoder as DNCBMSparseAutoencoder
except ImportError as e:
    print(f"[FATAL] Cannot import SparseAutoencoder from Discover-then-Name:\n  {e}")
    print(f"  Expected path: {SAE_LIB}")
    sys.exit(1)

# ── 80-template list from sae_vlm ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sae_vlm"))
try:
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] Cannot import openai_imagenet_template:\n  {e}")
    sys.exit(1)

try:
    import clip as openai_clip
except ImportError:
    print("[FATAL] OpenAI CLIP not found.\n"
          "  pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
BACKBONE = "ViT-B/16"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

BASE_SAE_PATH = (
    "/home/sunayana/Documents/Concept_LoRA/sae_vlm/"
    "DNCBM Checkpoints and Assigned Concept Names/Checkpoints/"
    "clip_ViT-B:16_sparse_autoencoder_final.pt"
)
LORA_ROOT = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
DATA_ROOT = "/home/sunayana/Documents/Concept_LoRA/data"

# DNCBM SAE dims (clip_ViT-B16_out: input=512, expansion_factor=8 → 4096)
SAE_N_INPUT    = 512
SAE_N_LEARNED  = 4096
SAE_N_COMP     = 1

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

DATASET_CFG = {
    "eurosat": {
        "data_dir":    f"{DATA_ROOT}/eurosat/2750",
        "lora_path":   f"{LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz":     False,
        "split":       "all",
    },
    "caltech101": {
        "data_dir":    f"{DATA_ROOT}/caltech-101",
        "lora_path":   f"{LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz":     False,
        "split":       "all",
        "exclude_classes": {"BACKGROUND_Google"},
    },
    "medmnist": {
        "data_dir":    f"{DATA_ROOT}/pathmnist_imagefolder",
        "npz_path":    f"{DATA_ROOT}/pathmnist.npz",
        "lora_path":   f"{LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
        "class_names": MEDMNIST_CLASSES,
        "use_npz":     True,
        "split":       "test",
    },
    "cub2002011": {
        "data_dir":    f"{DATA_ROOT}/cub2002011/test",
        "lora_path":   f"{LORA_ROOT}/cub2002011/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz":     False,
        "split":       "all",
    },
    "dtd": {
        "data_dir":    f"{DATA_ROOT}/DTD/images",
        "lora_path":   f"{LORA_ROOT}/dtd/16shots/seed42/lora_weights.pt",
        "class_names": None,
        "use_npz":     False,
        "split":       "all",
    },
    "ucf101": {
        "data_dir":    f"{DATA_ROOT}/UCF101/UCF-101-midframes",
        "lora_path":   f"{LORA_ROOT}/ucf101/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz":     False,
        "split":       "all",
    },
    "fgvc": {
        "data_dir":    f"{DATA_ROOT}/fgvc_imagefolder",
        "lora_path":   f"{LORA_ROOT}/fgvc/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz":     False,
        "split":       "all",
    },
    "oxford_pets": {
        "data_dir":    f"{DATA_ROOT}/oxford_pets_imagefolder",
        "lora_path":   f"{LORA_ROOT}/oxford_pets/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz":     False,
        "split":       "all",
    },
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


# ── SAE loading ───────────────────────────────────────────────────────────────

def load_dncbm_sae(path: str, device: str) -> DNCBMSparseAutoencoder:
    """
    Load the DNCBM SparseAutoencoder using the proper class from
    Discover-then-Name/sparse_autoencoder.

    The checkpoint IS the state_dict (not nested under a 'state_dict' key).
    Matches get_sae_ckpt() in dncbm/utils.py:
        state_dict = torch.load(ckpt_path)
        autoencoder.load_state_dict(state_dict)
    """
    sae = DNCBMSparseAutoencoder(
        n_input_features=SAE_N_INPUT,
        n_learned_features=SAE_N_LEARNED,
        n_components=SAE_N_COMP,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    sae.load_state_dict(state_dict)
    sae.eval().to(device)

    enc_shape = tuple(sae.encoder.weight.shape)
    dec_shape = tuple(sae.decoder.weight.shape)
    print(f"  [DNCBM SAE] enc={enc_shape}  dec={dec_shape}  "
          f"n_components={SAE_N_COMP}")
    return sae


@torch.no_grad()
def sae_reconstruct(sae: DNCBMSparseAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """
    Pass raw encode_image() features through the DNCBM SAE and return the
    decoded reconstruction.

    Matches save_concept_strengths.py usage:
        concepts, reconstructions = autoencoder(features)   # features: [B, D]
        reconstructions = reconstructions.squeeze()         # [B, D]

    Args:
        x: Raw (un-normalised) CLIP image features, shape [B, 512].

    Returns:
        decoded: Reconstructed features, shape [B, 512].
    """
    _, decoded = sae(x)        # decoded: [B, 1, 512]
    return decoded.squeeze(1)  # [B, 512]


# ── LoRA merging (OpenAI CLIP) ────────────────────────────────────────────────

def _lora_AB(layer_dict: dict, proj_name: str):
    d = layer_dict
    if proj_name in d and isinstance(d[proj_name], dict):
        try:
            return d[proj_name]["w_lora_A"], d[proj_name]["w_lora_B"]
        except (KeyError, TypeError):
            pass
    try:
        return d[f"{proj_name}.w_lora_A"], d[f"{proj_name}.w_lora_B"]
    except KeyError:
        return None, None


def build_base_clip(device: str):
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    model.eval()
    return model, preprocess


def build_lora_clip(lora_path: str, device: str):
    """
    Load CLIP ViT-B/16 and merge LoRA delta weights in-place.

    Scaling: alpha / sqrt(r), matching the training code in
    CLIP_LoRA/loralib/layers.py line 42:
        self.scaling = self.lora_alpha / math.sqrt(self.r)
    """
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    model.eval()

    lora_state  = torch.load(lora_path, map_location=device, weights_only=False)
    if "weights" not in lora_state:
        print("  [WARN] No 'weights' key in LoRA checkpoint; using base CLIP.")
        return model, preprocess

    layers_dict = lora_state["weights"]
    meta        = lora_state["metadata"]
    scale       = meta["alpha"] / math.sqrt(meta["r"])
    print(f"  LoRA: rank={meta['r']}  alpha={meta['alpha']}  "
          f"scale={scale:.6f}  encoder={meta.get('encoder')}  "
          f"position={meta.get('position')}")

    with torch.no_grad():
        for i in range(12):                # text encoder layers 0–11
            ld = layers_dict.get(f"layer_{i}")
            if ld is None:
                continue
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _lora_AB(ld, proj)
                if A is None:
                    continue
                w[off:off + d] += (
                    scale * (B.float().to(device) @ A.float().to(device))
                ).to(w.dtype)

        for i in range(12, 24):            # vision encoder layers 12–23
            ld = layers_dict.get(f"layer_{i}")
            if ld is None:
                continue
            w = model.visual.transformer.resblocks[i - 12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _lora_AB(ld, proj)
                if A is None:
                    continue
                w[off:off + d] += (
                    scale * (B.float().to(device) @ A.float().to(device))
                ).to(w.dtype)

    print("  LoRA weights merged.")
    return model, preprocess


# ── Dataset loading ───────────────────────────────────────────────────────────

def _find_imagefolder_root(root: str) -> str:
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        raise FileNotFoundError(f"No sub-directories in {root}")
    for sd in subdirs[:3]:
        sd_path = os.path.join(root, sd)
        if any(f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
               for f in os.listdir(sd_path)):
            return root
    for sd in sorted(subdirs):
        sd_path = os.path.join(root, sd)
        inner   = [d for d in os.listdir(sd_path)
                   if os.path.isdir(os.path.join(sd_path, d))]
        if inner:
            try:
                return _find_imagefolder_root(sd_path)
            except (FileNotFoundError, PermissionError):
                continue
    raise FileNotFoundError(f"No image class directories found under {root}")


class FilteredImageFolder(Dataset):
    def __init__(self, root: str, transform, exclude_classes=None):
        img_root    = _find_imagefolder_root(root)
        full_ds     = tv_datasets.ImageFolder(root=img_root, transform=transform)
        exclude     = exclude_classes or set()
        keep        = [i for i, (_, lbl) in enumerate(full_ds.samples)
                       if full_ds.classes[lbl] not in exclude]
        kept_names  = sorted({full_ds.classes[full_ds.targets[i]] for i in keep})
        old2new     = {full_ds.class_to_idx[c]: ni for ni, c in enumerate(kept_names)}
        self._ds        = full_ds
        self._idx       = keep
        self._map       = old2new
        self.class_names = kept_names

    def __len__(self):  return len(self._idx)
    def __getitem__(self, idx):
        img, old_lbl = self._ds[self._idx[idx]]
        return img, self._map[old_lbl]


class MedMNISTNpzDataset(Dataset):
    def __init__(self, npz_path: str, imagefolder_root: str, preprocess, split="test"):
        data      = np.load(npz_path)
        imgs_key  = f"{split}_images"
        lbls_key  = f"{split}_labels"
        if imgs_key not in data:
            raise KeyError(f"'{imgs_key}' not in {npz_path}. Keys: {list(data.keys())}")
        self.imgs       = data[imgs_key]
        self.labels     = data[lbls_key].flatten().astype(int)
        self.preprocess = preprocess
        if_root  = _find_imagefolder_root(imagefolder_root)
        if_ds    = tv_datasets.ImageFolder(root=if_root)
        if_map   = {n.replace("_", " ").lower(): i for n, i in if_ds.class_to_idx.items()}
        self.label_map = {ni: if_map.get(cls.lower(), ni)
                          for ni, cls in enumerate(MEDMNIST_CLASSES)}

    def __len__(self):  return len(self.labels)
    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx]).convert("RGB")
        return self.preprocess(img), self.label_map[int(self.labels[idx])]


def load_test_dataset(cfg: dict, preprocess):
    if cfg.get("use_npz"):
        ds = MedMNISTNpzDataset(
            npz_path=cfg["npz_path"],
            imagefolder_root=cfg["data_dir"],
            preprocess=preprocess,
            split=cfg.get("split", "test"),
        )
        if_root     = _find_imagefolder_root(cfg["data_dir"])
        if_ds       = tv_datasets.ImageFolder(root=if_root)
        class_names = [n.replace("_", " ") for n in if_ds.classes]
    else:
        ds = FilteredImageFolder(
            root=cfg["data_dir"],
            transform=preprocess,
            exclude_classes=cfg.get("exclude_classes"),
        )
        class_names = cfg.get("class_names") or ds.class_names
    return ds, class_names


# ── Text features ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_text_features(model, class_names: list, device: str) -> torch.Tensor:
    """Average 80 OpenAI templates → L2-normalised text feature matrix [n, 512]."""
    n      = len(class_names)
    mean_f = torch.zeros(n, 512, device=device)
    for tmpl in openai_imagenet_template:
        prompts = [tmpl(c) for c in class_names]
        tokens  = openai_clip.tokenize(prompts, truncate=True).to(device)
        feats   = model.encode_text(tokens).float()
        mean_f += l2_normalize(feats)
    mean_f /= len(openai_imagenet_template)
    return l2_normalize(mean_f)


# ── Accuracy computation ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_accuracy(
    model,
    text_features: torch.Tensor,
    loader: DataLoader,
    device: str,
    sae: DNCBMSparseAutoencoder | None = None,
) -> tuple[float, dict]:
    """
    Zero-shot top-1 accuracy.

    If sae is not None:
      1. Get raw (un-normalised) encode_image() features  [B, 512]
      2. Pass through DNCBM SAE → decoded.squeeze(1)      [B, 512]
      3. L2-normalise decoded output for cosine similarity

    Returns (accuracy_pct, debug_stats).
    """
    logit_scale = model.logit_scale.exp()
    tf          = l2_normalize(text_features.float())

    correct = total = 0
    cos_sims = []
    l0_counts = []

    for imgs, labels in tqdm(loader, desc="    eval", leave=False):
        imgs    = imgs.to(device)
        raw_feat = model.encode_image(imgs).float()   # [B, 512] raw (un-normalised)

        if sae is not None:
            # Reconstruct via DNCBM SAE (input: raw [B,512], output: [B,1,512])
            learned_acts, decoded = sae(raw_feat)
            recon = decoded.squeeze(1)                # [B, 512]

            # Debug: track reconstruction quality and sparsity
            cos_sims.append(
                F.cosine_similarity(raw_feat, recon, dim=-1).mean().item()
            )
            l0_counts.append(
                (learned_acts.squeeze(1) > 0).float().sum(dim=-1).mean().item()
            )
            img_feat = l2_normalize(recon)
        else:
            img_feat = l2_normalize(raw_feat)

        logits = logit_scale * (img_feat @ tf.t())
        preds  = logits.argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    acc   = 100.0 * correct / total
    stats = {}
    if cos_sims:
        stats["recon_cos_sim"] = round(float(np.mean(cos_sims)), 6)
        stats["recon_l0"]      = round(float(np.mean(l0_counts)), 1)
    return acc, stats


# ── Per-dataset evaluation ────────────────────────────────────────────────────

def evaluate_dataset(
    dataset_name: str,
    cfg: dict,
    sae: DNCBMSparseAutoencoder,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict:
    print(f"\n{'='*72}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*72}")

    out = {"dataset": dataset_name, "backbone": BACKBONE}

    lora_path = cfg["lora_path"]
    if not os.path.isfile(lora_path):
        print(f"  [SKIP] LoRA weights missing: {lora_path}")
        return out

    print(f"\n[1] Loading base CLIP ({BACKBONE})")
    base_model, preprocess = build_base_clip(device)

    print(f"\n[2] Loading LoRA CLIP\n    {lora_path}")
    lora_model, _ = build_lora_clip(lora_path, device)

    print(f"\n[3] Loading dataset: {dataset_name}")
    try:
        ds, class_names = load_test_dataset(cfg, preprocess)
    except Exception as e:
        print(f"  [ERROR] {e}")
        del base_model, lora_model; flush()
        return out

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    print(f"  {len(ds)} images | {len(class_names)} classes")

    print("\n[4] Building text features (80 templates)")
    base_text = get_text_features(base_model, class_names, device)
    lora_text = get_text_features(lora_model, class_names, device)

    print("\n[5] Evaluating four conditions")

    print("    ZS ...")
    acc_zs, _ = compute_accuracy(base_model, base_text, loader, device, sae=None)
    print(f"    ZS                   : {acc_zs:.2f}%")

    print("    LoRA-FT ...")
    acc_lora, _ = compute_accuracy(lora_model, lora_text, loader, device, sae=None)
    print(f"    LoRA-FT              : {acc_lora:.2f}%")

    print("    ZS + BaseSAE-1 ...")
    acc_zs_sae, stats_zs = compute_accuracy(base_model, base_text, loader, device, sae=sae)
    print(f"    ZS + BaseSAE-1       : {acc_zs_sae:.2f}%  "
          f"[cos_sim={stats_zs.get('recon_cos_sim','?')}  "
          f"L0={stats_zs.get('recon_l0','?')}/4096]")

    print("    LoRA-FT + BaseSAE-1 ...")
    acc_lora_sae, stats_lora = compute_accuracy(lora_model, lora_text, loader, device, sae=sae)
    print(f"    LoRA-FT + BaseSAE-1  : {acc_lora_sae:.2f}%  "
          f"[cos_sim={stats_lora.get('recon_cos_sim','?')}  "
          f"L0={stats_lora.get('recon_l0','?')}/4096]")

    out["ZS"]                    = round(acc_zs,       4)
    out["LoRA-FT"]               = round(acc_lora,     4)
    out["ZS + BaseSAE-1"]        = round(acc_zs_sae,   4)
    out["LoRA-FT + BaseSAE-1"]   = round(acc_lora_sae, 4)
    out["sae_recon_cos_sim_zs"]   = stats_zs.get("recon_cos_sim")
    out["sae_recon_cos_sim_lora"] = stats_lora.get("recon_cos_sim")
    out["sae_l0"]                 = stats_zs.get("recon_l0")

    del base_model, lora_model
    flush()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Evaluate ZS / LoRA-FT / ZS+BaseSAE-1 / LoRA-FT+BaseSAE-1"
    )
    p.add_argument(
        "--datasets", nargs="+",
        default=list(DATASET_CFG.keys()),
        choices=list(DATASET_CFG.keys()),
    )
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--save_json", type=str,
        default="results/base_sae_accuracy.json",
    )
    args = p.parse_args()

    print(f"\n{'='*72}")
    print("DNCBM BaseSAE Accuracy Evaluation")
    print(f"Device   : {DEVICE}")
    print(f"Backbone : {BACKBONE}")
    print(f"Datasets : {args.datasets}")
    print(f"SAE path : {BASE_SAE_PATH}")
    print(f"SAE dims : n_input={SAE_N_INPUT}  n_learned={SAE_N_LEARNED}  n_components={SAE_N_COMP}")
    print(f"{'='*72}")

    if not os.path.isfile(BASE_SAE_PATH):
        print(f"[FATAL] SAE checkpoint not found:\n  {BASE_SAE_PATH}")
        sys.exit(1)

    print("\nLoading DNCBM BaseSAE ...")
    sae = load_dncbm_sae(BASE_SAE_PATH, DEVICE)

    all_results = []
    for ds_name in args.datasets:
        res = evaluate_dataset(
            dataset_name=ds_name,
            cfg=DATASET_CFG[ds_name],
            sae=sae,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=DEVICE,
        )
        all_results.append(res)

    print(f"\n\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    hdr = f"{'Dataset':<14s} {'ZS':>7s} {'LoRA-FT':>9s} {'ZS+SAE':>9s} {'LoRA+SAE':>10s}  {'cos_sim_zs':>10s}  {'cos_sim_lora':>12s}  {'L0':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        if "ZS" not in r:
            print(f"{r['dataset']:<14s}  (skipped)")
            continue
        print(
            f"{r['dataset']:<14s} "
            f"{r['ZS']:>6.2f}% "
            f"{r['LoRA-FT']:>8.2f}% "
            f"{r['ZS + BaseSAE-1']:>8.2f}% "
            f"{r['LoRA-FT + BaseSAE-1']:>9.2f}%  "
            f"{str(r.get('sae_recon_cos_sim_zs','?')):>10s}  "
            f"{str(r.get('sae_recon_cos_sim_lora','?')):>12s}  "
            f"{str(r.get('sae_l0','?')):>6s}"
        )

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.save_json}")


if __name__ == "__main__":
    main()

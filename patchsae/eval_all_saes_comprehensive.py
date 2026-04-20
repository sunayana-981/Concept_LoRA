#!/usr/bin/env python3
"""
eval_all_saes_comprehensive.py — Comprehensive SAE evaluation across all checkpoints.

Metrics (per SAE × backbone):
  1. Normalized MSE  nMSE = E[||x-x̂||²] / E[||x||²]  — global, CLS, spatial
  2. L0              mean ± std + percentiles           — CLS vs spatial
  3. Dead latent fraction
  4. Cosine similarity of reconstructions               — global, CLS, spatial
  5. Zero-shot accuracy retention (with SAE vs without)

SAE labels:
  - Architecture : standard | topk | jumprelu  (from checkpoint filename)
  - Method       : base | lora | maple | masked_finetune | masked_finetune_lora |
                   masked_finetune_maple  (from directory structure)
  - Dataset      : imagenet | medmnist | eurosat | caltech101 | dtd | ucf101 |
                   cub2002011 | dora  (from directory structure)
  - Layer        : from cfg.block_layer

Evaluation is run per-dataset: each SAE is evaluated against its matching
LoRA checkpoint and eval dataset (in-distribution evaluation).
Additionally, base_domain uses base HF CLIP on the same dataset.

Usage (run from patchsae/):
    python eval_all_saes_comprehensive.py
    python eval_all_saes_comprehensive.py --out out/eval_all_saes.json
    python eval_all_saes_comprehensive.py --n_batches 30 --batch_size 32
    python eval_all_saes_comprehensive.py --skip_accuracy  # faster
    python eval_all_saes_comprehensive.py --methods lora maple --datasets_filter medmnist
"""

import argparse
import gc
import glob
import json
import math
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# ── SAE loading: prefer sae_vlm version (has topk + jumprelu) ────────────────
_SAELIB_ROOTS = [
    str(Path(__file__).parent.parent / "sae_vlm"),   # sibling sae_vlm directory
    str(Path(__file__).parent),                       # patchsae itself
]
for _r in _SAELIB_ROOTS:
    if _r not in sys.path:
        sys.path.insert(0, _r)

try:
    from src.sae_training.sparse_autoencoder import SparseAutoencoder
    from src.sae_training.config import Config
    _SAE_ROOT = next(
        r for r in _SAELIB_ROOTS
        if os.path.isfile(os.path.join(r, "src", "sae_training", "sparse_autoencoder.py"))
    )
    print(f"[SAE] Loaded SparseAutoencoder from: {_SAE_ROOT}")
except (ImportError, StopIteration) as e:
    print(f"[FATAL] Cannot import SparseAutoencoder: {e}")
    sys.exit(1)

try:
    from tasks.utils import load_hooked_vit
    from src.sae_training.hooked_vit import Hook
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] patchsae imports failed: {e}\nRun from patchsae/ directory.")
    sys.exit(1)

try:
    import clip as openai_clip
    HAS_OPENAI_CLIP = True
except ImportError:
    HAS_OPENAI_CLIP = False
    print("[WARN] OpenAI CLIP not found — LoRA backbone unavailable")

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_ROOT_PATCHSAE = Path(__file__).parent / "out/checkpoints"
CKPT_ROOT_SAEVLM   = Path(__file__).parent.parent / "sae_vlm/out/checkpoints"
BASE_SAE_PATH      = str(Path(__file__).parent / "data/sae_weight/base/out.pt")
BACKBONE           = "openai/clip-vit-base-patch16"
LORA_CKPT          = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
TRAIN_ROOT         = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
NPZ_PATH           = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

_DATA_ROOT  = "/home/sunayana/Documents/Concept_LoRA/data"
_LORA_ROOT  = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"

# Per-dataset config: lora_ckpt, eval dataset loader type, data path, lora seed
DATASET_CONFIG = {
    "medmnist": {
        "lora_ckpt":   f"{_LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
        "loader_type": "npz",
        "data_path":   f"{_DATA_ROOT}/pathmnist.npz",
        "train_root":  f"{_DATA_ROOT}/pathmnist_imagefolder",
    },
    "eurosat": {
        "lora_ckpt":   f"{_LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
        "loader_type": "imagefolder",
        "data_path":   f"{_DATA_ROOT}/eurosat/2750",
        "split_json":  f"{_DATA_ROOT}/eurosat/split_zhou_EuroSAT.json",
    },
    "caltech101": {
        "lora_ckpt":   f"{_LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
        "loader_type": "imagefolder",
        "data_path":   f"{_DATA_ROOT}/caltech-101/Caltech101/101_ObjectCategories",
        "split_json":  f"{_DATA_ROOT}/caltech-101/Caltech101/split_zhou_Caltech101.json",
        "exclude":     {"BACKGROUND_Google"},
    },
    "dtd": {
        "lora_ckpt":   f"{_LORA_ROOT}/dtd/16shots/seed42/lora_weights.pt",
        "loader_type": "imagefolder",
        "data_path":   f"{_DATA_ROOT}/DTD/images",
        "split_json":  f"{_DATA_ROOT}/dtd/split_zhou_DescribableTextures.json",
    },
    "ucf101": {
        "lora_ckpt":   f"{_LORA_ROOT}/ucf101/16shots/seed1/lora_weights.pt",
        "loader_type": "imagefolder",
        "data_path":   f"{_DATA_ROOT}/UCF101/UCF-101-midframes",
        "split_json":  f"{_DATA_ROOT}/UCF101/split_zhou_UCF101.json",
    },
    "cub2002011": {
        "lora_ckpt":   f"{_LORA_ROOT}/cub2002011/16shots/seed1/lora_weights.pt",
        "loader_type": "imagefolder",
        "data_path":   f"{_DATA_ROOT}/cub2002011/test",
    },
    "fgvc": {
        "lora_ckpt":   f"{_LORA_ROOT}/fgvc/16shots/seed1/lora_weights.pt",
        "loader_type": "imagefolder",
        "data_path":   f"{_DATA_ROOT}/fgvc_imagefolder/train",
    },
    "oxford_pets": {
        "lora_ckpt":   f"{_LORA_ROOT}/oxford_pets/16shots/seed1/lora_weights.pt",
        "loader_type": "imagefolder",
        "data_path":   f"{_DATA_ROOT}/OxfordPets",
        "split_json":  f"{_DATA_ROOT}/OxfordPets/split_zhou_OxfordPets.json",
    },
    # fallback for 'unknown' dataset SAEs — use medmnist
    "unknown": {
        "lora_ckpt":   f"{_LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
        "loader_type": "npz",
        "data_path":   f"{_DATA_ROOT}/pathmnist.npz",
        "train_root":  f"{_DATA_ROOT}/pathmnist_imagefolder",
    },
    "imagenet": {
        "lora_ckpt":   None,   # no LoRA for imagenet
        "loader_type": "npz",
        "data_path":   f"{_DATA_ROOT}/pathmnist.npz",
        "train_root":  f"{_DATA_ROOT}/pathmnist_imagefolder",
    },
}

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

PERCENTILES = [0, 10, 25, 50, 75, 90, 95, 99, 100]


# ── Utilities ──────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def l2_normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ── SAE loading & labeling ─────────────────────────────────────────────────────

def parse_sae_label(path: str):
    """Parse architecture, method, dataset from checkpoint path."""
    fname = os.path.basename(path).lower()

    # Architecture from filename
    if "_topk_" in fname:
        arch = "topk"
    elif "_jumprelu_" in fname:
        arch = "jumprelu"
    else:
        arch = "standard"

    parts = [p.lower() for p in path.replace("\\", "/").split("/")]

    # Method from directory hierarchy
    if "masked_finetune_lora" in parts:
        method = "masked_finetune_lora"
    elif "masked_finetune_maple" in parts:
        method = "masked_finetune_maple"
    elif "masked_finetune" in parts:
        method = "masked_finetune"
    elif "dora" in parts:
        method = "dora"
    elif any(d.endswith("_maple") for d in parts):
        method = "maple"
    elif "base" in parts and "sae_weight" in parts:
        method = "base"
    else:
        method = "lora"

    # Dataset from directory name
    dataset_keys = ["medmnist", "eurosat", "caltech101", "dtd", "ucf101",
                    "cub2002011", "imagenet", "fgvc", "oxford_pets"]
    dataset = "unknown"
    for d in parts:
        base_d = d.replace("_maple", "").replace("_lora", "")
        if base_d in dataset_keys:
            dataset = base_d
            break
    if dataset == "unknown" and "dora" in parts:
        # dora filenames: ..._dora_clip_<dataset>_...
        for k in dataset_keys:
            if f"_{k}_" in fname or f"_{k}-" in fname:
                dataset = k
                break

    return arch, method, dataset


def load_sae_universal(path: str, device: str):
    """Load any SAE checkpoint using sae_vlm's SparseAutoencoder (handles all types)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_raw = ckpt.get("cfg", ckpt.get("config"))
    if cfg_raw is None:
        raise ValueError(f"No config found in {path}")

    if isinstance(cfg_raw, dict):
        cfg = Config(cfg_raw)
    else:
        cfg = cfg_raw

    # Ensure required attributes have safe defaults
    for attr, default in [
        ("sae_type", "standard"), ("gated_sae", False),
        ("topk_k", 64), ("jumprelu_bandwidth", 0.001),
        ("use_ghost_grads", False), ("mse_cls_coefficient", 1.0),
        ("l1_coefficient", 0.0), ("dtype", torch.float32),
    ]:
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)

    sae = SparseAutoencoder(cfg, device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(device)
    return sae, cfg


def discover_all_saes():
    """Find all final SAE checkpoints in patchsae and sae_vlm directories."""
    results = []
    seen_run_ids = set()

    # Base SAE
    if os.path.isfile(BASE_SAE_PATH):
        results.append({
            "path": BASE_SAE_PATH, "run_id": "base",
            "arch": "standard", "method": "base", "dataset": "imagenet",
            "layer": -1,
        })
        seen_run_ids.add("base")

    roots = [r for r in [str(CKPT_ROOT_PATCHSAE), str(CKPT_ROOT_SAEVLM)]
             if Path(r).exists()]

    for root in roots:
        # Standard checkpoints: file lives inside a final_*/ subdirectory
        # DoRA checkpoints: file IS named final_*.pt directly under dora/*/
        paths = sorted(set(
            glob.glob(os.path.join(root, "**/final*/*.pt"), recursive=True) +
            glob.glob(os.path.join(root, "dora/*/final*.pt"))
        ))
        for p in paths:
            parts = p.replace("\\", "/").split("/")
            # grandparent is run_id when file is inside final_*/ dir;
            # parent is run_id when file IS the final_*.pt (DoRA style)
            if len(parts) >= 2 and parts[-2].startswith("final"):
                run_id = parts[-3] if len(parts) >= 3 else "unknown"
            else:
                run_id = parts[-2] if len(parts) >= 2 else "unknown"
            if run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)

            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                cfg_raw = ckpt.get("cfg", ckpt.get("config"))
                if cfg_raw is None:
                    continue
                layer = (getattr(cfg_raw, "block_layer", "?")
                         if not isinstance(cfg_raw, dict)
                         else cfg_raw.get("block_layer", "?"))
                del ckpt
                gc.collect()
            except Exception as e:
                print(f"  [SKIP] {p}: {e}")
                continue

            arch, method, dataset = parse_sae_label(p)
            results.append({
                "path": p, "run_id": run_id,
                "arch": arch, "method": method, "dataset": dataset,
                "layer": layer,
            })

    print(f"\n[DISCOVER] Found {len(results)} unique SAE checkpoints")
    return results


# ── Dataset ────────────────────────────────────────────────────────────────────

class NpzTestDataset(Dataset):
    def __init__(self, npz_path, mapping, transform, return_pil=False):
        data = np.load(npz_path)
        self.imgs   = data["test_images"]
        self.labels = data["test_labels"].flatten()
        self.mapping = mapping
        self.transform = transform
        self.return_pil = return_pil

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i])
        img = self.transform(img)
        return img, self.mapping[int(self.labels[i])]


def _hf_collate(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels)


def build_medmnist_loaders(batch_size, lora_preprocess=None):
    """Build HF (PIL) and optionally OAI (tensor) medmnist test loaders."""
    ds = datasets.ImageFolder(root=TRAIN_ROOT)
    if_map = {n.replace("_", " ").lower(): i for n, i in ds.class_to_idx.items()}
    mapping = {i: if_map.get(c.lower(), i) for i, c in enumerate(MEDMNIST_CLASSES)}
    classnames = [c.replace("_", " ") for c in ds.classes]

    hf_transform = transforms.Resize((224, 224))
    hf_ds = NpzTestDataset(NPZ_PATH, mapping, hf_transform, return_pil=True)
    hf_loader = DataLoader(hf_ds, batch_size=batch_size, shuffle=False,
                           num_workers=2, collate_fn=_hf_collate)

    oai_loader = None
    if lora_preprocess is not None:
        oai_ds = NpzTestDataset(NPZ_PATH, mapping, lora_preprocess)
        oai_loader = DataLoader(oai_ds, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)

    return hf_loader, oai_loader, classnames


class SplitJsonDataset(Dataset):
    """Dataset loader for zhou-split JSON files (eurosat, caltech101, dtd, ucf101)."""
    def __init__(self, split_json, img_root, transform, split="test", exclude=None):
        import json as _json
        with open(split_json) as f:
            split_data = _json.load(f)
        items = split_data.get(split, split_data.get("test", []))
        # items: list of [rel_path, label_int, classname]  or  [rel_path, classname]
        exclude = exclude or set()
        classes = sorted({it[2] if len(it) >= 3 else it[1]
                          for it in items
                          if (it[2] if len(it) >= 3 else it[1]) not in exclude})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.classes = classes
        self.samples = []
        for it in items:
            rel_path = it[0]
            classname = it[2] if len(it) >= 3 else it[1]
            if classname in exclude:
                continue
            full_path = os.path.join(img_root, rel_path)
            if os.path.isfile(full_path):
                self.samples.append((full_path, self.class_to_idx[classname]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataset_loaders(dataset_name: str, batch_size: int, lora_preprocess=None):
    """Build HF and OAI loaders for any supported dataset. Returns (hf_loader, oai_loader, classnames)."""
    cfg = DATASET_CONFIG.get(dataset_name, DATASET_CONFIG["unknown"])
    loader_type = cfg["loader_type"]
    data_path   = cfg["data_path"]

    if loader_type == "npz":
        return build_medmnist_loaders(batch_size, lora_preprocess)

    # ImageFolder or split-json based
    pil_transform  = transforms.Resize((224, 224))

    split_json = cfg.get("split_json")
    exclude    = cfg.get("exclude", set())

    if split_json and os.path.isfile(split_json):
        hf_ds = SplitJsonDataset(split_json, data_path, pil_transform,
                                 split="test", exclude=exclude)
        classnames = hf_ds.classes
    elif os.path.isdir(data_path):
        full_ds = datasets.ImageFolder(data_path, transform=pil_transform)
        if exclude:
            keep = [i for i, (_, l) in enumerate(full_ds.samples)
                    if full_ds.classes[l] not in exclude]
            from torch.utils.data import Subset  # noqa: F401 — torch always available
            hf_ds = Subset(full_ds, keep)
            classnames = [c for c in full_ds.classes if c not in exclude]
        else:
            hf_ds = full_ds
            classnames = full_ds.classes
    else:
        print(f"  [WARN] Dataset path not found: {data_path}, falling back to MedMNIST")
        return build_medmnist_loaders(batch_size, lora_preprocess)

    def _pil_collate(batch):
        imgs, labels = zip(*batch)
        # imgs may already be tensors from Resize transform; convert to PIL list for HF
        pil_imgs = []
        for img in imgs:
            if isinstance(img, torch.Tensor):
                arr = img.permute(1, 2, 0).numpy() if img.ndim == 3 else img.numpy()
                pil_imgs.append(Image.fromarray((arr * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)))
            else:
                pil_imgs.append(img)
        return pil_imgs, torch.tensor(labels)

    hf_loader = DataLoader(hf_ds, batch_size=batch_size, shuffle=False,
                           num_workers=2, collate_fn=_pil_collate)

    oai_loader = None
    if lora_preprocess is not None:
        if split_json and os.path.isfile(split_json):
            oai_ds = SplitJsonDataset(split_json, data_path, lora_preprocess,
                                      split="test", exclude=exclude)
        elif os.path.isdir(data_path):
            full_ds2 = datasets.ImageFolder(data_path, transform=lora_preprocess)
            if exclude:
                keep2 = [i for i, (_, l) in enumerate(full_ds2.samples)
                         if full_ds2.classes[l] not in exclude]
                from torch.utils.data import Subset  # noqa: F401 — torch always available
                oai_ds = Subset(full_ds2, keep2)
            else:
                oai_ds = full_ds2
        else:
            oai_ds = None

        if oai_ds is not None:
            oai_loader = DataLoader(oai_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True)

    return hf_loader, oai_loader, classnames


# ── LoRA backbone ──────────────────────────────────────────────────────────────

def build_lora_clip(lora_path, device):
    if not HAS_OPENAI_CLIP or not os.path.isfile(lora_path):
        print(f"  [WARN] LoRA backbone unavailable (path={lora_path})")
        return None, None

    model, preprocess = openai_clip.load("ViT-B/16", device=device)
    model.eval()

    lora_state = torch.load(lora_path, map_location=device, weights_only=False)
    if "weights" not in lora_state:
        return model, preprocess

    layers, meta = lora_state["weights"], lora_state["metadata"]
    scale = meta["alpha"] / math.sqrt(meta["r"])

    def _AB(ld, proj):
        if proj in ld and isinstance(ld[proj], dict):
            try:
                return ld[proj]["w_lora_A"], ld[proj]["w_lora_B"]
            except KeyError:
                pass
        try:
            return ld[f"{proj}.w_lora_A"], ld[f"{proj}.w_lora_B"]
        except KeyError:
            return None, None

    with torch.no_grad():
        for i in range(12):
            ld = layers.get(f"layer_{i}")
            if ld is None:
                continue
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _AB(ld, proj)
                if A is not None:
                    w[off:off + d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)
        for i in range(12, 24):
            ld = layers.get(f"layer_{i}")
            if ld is None:
                continue
            w = model.visual.transformer.resblocks[i - 12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _AB(ld, proj)
                if A is not None:
                    w[off:off + d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)

    print(f"  LoRA merged: rank={meta['r']}, alpha={meta['alpha']}, scale={scale:.4f}")
    return model, preprocess


# ── Activation capture ────────────────────────────────────────────────────────

@torch.no_grad()
def capture_activations_hf(vit, loader, cfg, device, n_batches=None):
    """Capture [N, S, D] activations from base/maple HF CLIP at cfg.block_layer."""
    all_acts = []
    captured = {}

    def capture_fn(act):
        captured["act"] = act.clone().float()
        return (act,)

    hooks = [Hook(cfg.block_layer, cfg.module_name, capture_fn,
                  return_module_output=False, is_custom=False)]

    for i, (images, _labels) in enumerate(tqdm(loader, desc="    capture(HF)", leave=False)):
        if n_batches is not None and i >= n_batches:
            break
        inputs = vit.processor(images=images, text="",
                               return_tensors="pt", padding=True).to(device)
        captured.clear()
        _ = vit.run_with_hooks(hooks, return_type="output", **inputs)
        if "act" in captured:
            all_acts.append(captured["act"].cpu())

    return torch.cat(all_acts, dim=0) if all_acts else None  # [N, S, D]


@torch.no_grad()
def capture_activations_openai(model, loader, cfg, device, n_batches=None):
    """Capture [N, S, D] activations from OpenAI CLIP at cfg.block_layer."""
    all_acts = []
    captured = {}

    num_blocks = len(model.visual.transformer.resblocks)
    layer = cfg.block_layer
    if layer < 0:
        layer = num_blocks + layer
    block = model.visual.transformer.resblocks[layer]

    def hook_fn(module, inp, output):
        captured["act"] = output.transpose(0, 1).float()   # [seq,B,D] → [B,seq,D]

    handle = block.register_forward_hook(hook_fn)

    for i, (imgs, _labels) in enumerate(tqdm(loader, desc="    capture(OAI)", leave=False)):
        if n_batches is not None and i >= n_batches:
            break
        captured.clear()
        _ = model.encode_image(imgs.to(device))
        if "act" in captured:
            all_acts.append(captured["act"].cpu())

    handle.remove()
    return torch.cat(all_acts, dim=0) if all_acts else None  # [N, S, D]


# ── Core metrics ──────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_reconstruction_metrics(sae, activations, device, chunk_size=64):
    """
    Compute nMSE and cosine similarity.
    nMSE = E[||x - x̂||²] / E[||x||²]  — reported globally, CLS, spatial.
    activations: [N, S, D] float32 CPU tensor.
    """
    sae.eval()
    N, S, D = activations.shape

    nmse_buf, cos_buf = [], []
    nmse_cls_buf, cos_cls_buf = [], []
    nmse_spa_buf, cos_spa_buf = [], []

    for start in range(0, N, chunk_size):
        act = activations[start:start + chunk_size].float().to(device)
        try:
            sae_out, _fa, _ = sae(act)
        except Exception:
            continue

        sae_out = sae_out.float()

        # ── global ──
        mse = (act - sae_out).pow(2).mean(dim=-1)   # [B, S]
        var = act.pow(2).mean(dim=-1)                 # [B, S]
        nmse_buf.append((mse.mean() / var.mean().clamp(1e-8)).item())
        cos_buf.append(
            F.cosine_similarity(act.reshape(-1, D), sae_out.reshape(-1, D), dim=-1).mean().item()
        )

        # ── CLS (token 0) ──
        mse_c = (act[:, 0] - sae_out[:, 0]).pow(2).mean(dim=-1)
        var_c = act[:, 0].pow(2).mean(dim=-1)
        nmse_cls_buf.append((mse_c.mean() / var_c.mean().clamp(1e-8)).item())
        cos_cls_buf.append(F.cosine_similarity(act[:, 0], sae_out[:, 0], dim=-1).mean().item())

        # ── spatial (tokens 1:) ──
        mse_s = (act[:, 1:] - sae_out[:, 1:]).pow(2).mean(dim=-1)
        var_s = act[:, 1:].pow(2).mean(dim=-1)
        nmse_spa_buf.append((mse_s.mean() / var_s.mean().clamp(1e-8)).item())
        cos_spa_buf.append(
            F.cosine_similarity(act[:, 1:].reshape(-1, D), sae_out[:, 1:].reshape(-1, D), dim=-1).mean().item()
        )

    if not nmse_buf:
        return {}

    return {
        "nmse":             float(np.mean(nmse_buf)),
        "nmse_cls":         float(np.mean(nmse_cls_buf)),
        "nmse_spatial":     float(np.mean(nmse_spa_buf)),
        "cos_sim":          float(np.mean(cos_buf)),
        "cos_sim_cls":      float(np.mean(cos_cls_buf)),
        "cos_sim_spatial":  float(np.mean(cos_spa_buf)),
    }


@torch.no_grad()
def compute_l0_and_dead(sae, activations, device, chunk_size=64, dead_thresh=1e-5):
    """
    Compute L0 distribution (CLS vs spatial) and dead latent fraction.
    activations: [N, S, D] float32 CPU tensor.
    """
    sae.eval()
    N, S, D = activations.shape
    d_sae = sae.d_sae

    l0_cls_all     = []   # per-sample: L0 of CLS token
    l0_spatial_all = []   # per-sample: mean L0 over spatial tokens
    fire_count = torch.zeros(d_sae, device=device)
    total_tokens = 0

    for start in range(0, N, chunk_size):
        act = activations[start:start + chunk_size].float().to(device)
        try:
            _out, feature_acts, _ = sae(act)
        except Exception:
            continue

        feature_acts = feature_acts.float()              # [B, S, d_sae]
        active = (feature_acts > 0)                      # [B, S, d_sae] bool

        l0_per_token = active.float().sum(dim=-1)        # [B, S]
        l0_cls_all.extend(l0_per_token[:, 0].cpu().tolist())
        # mean L0 over spatial tokens per sample
        l0_spatial_all.extend(l0_per_token[:, 1:].mean(dim=-1).cpu().tolist())

        fire_count  += active.float().sum(dim=(0, 1))    # [d_sae]
        total_tokens += act.shape[0] * act.shape[1]

    if not l0_cls_all:
        return {}

    fire_rate     = fire_count / max(total_tokens, 1)
    dead_fraction = (fire_rate < dead_thresh).float().mean().item()

    l0_cls_arr     = np.array(l0_cls_all)
    l0_spatial_arr = np.array(l0_spatial_all)

    return {
        "l0_cls_mean":     float(l0_cls_arr.mean()),
        "l0_cls_std":      float(l0_cls_arr.std()),
        "l0_spatial_mean": float(l0_spatial_arr.mean()),
        "l0_spatial_std":  float(l0_spatial_arr.std()),
        "l0_cls_pct":     {str(p): float(np.percentile(l0_cls_arr, p))
                            for p in PERCENTILES},
        "l0_spatial_pct": {str(p): float(np.percentile(l0_spatial_arr, p))
                            for p in PERCENTILES},
        "dead_fraction":  dead_fraction,
        "n_dead":         int((fire_rate < dead_thresh).sum().item()),
        "n_latents":      d_sae,
    }


# ── Zero-shot accuracy ────────────────────────────────────────────────────────

@torch.no_grad()
def zero_shot_accuracy_hf(vit, classnames, loader, device, sae=None, cfg=None):
    """Zero-shot accuracy using HF CLIP, optionally with SAE hook."""
    from torch.nn.utils.rnn import pad_sequence

    logit_scale = vit.model.logit_scale.exp()

    mean_f = 0
    for tmpl in openai_imagenet_template:
        ids = [vit.processor(text=tmpl(c), return_tensors="pt",
                             padding=False, truncation=True).input_ids[0]
               for c in classnames]
        padded = pad_sequence(ids, batch_first=True).to(device)
        f = l2_normalize(vit.model.get_text_features(padded).float())
        mean_f = mean_f + f if not isinstance(mean_f, int) else f
    text_feat = l2_normalize(mean_f / len(openai_imagenet_template))

    hooks = []
    if sae is not None and cfg is not None:
        logged = [False]

        def sae_fn(act):
            orig_dtype = act.dtype
            sae_out, _fa, _ = sae(act.clone().float())
            if not logged[0]:
                logged[0] = True
            act[:, :, :] = sae_out.to(orig_dtype)
            return (act,)

        hooks = [Hook(cfg.block_layer, cfg.module_name, sae_fn,
                      return_module_output=False, is_custom=False)]

    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="      acc(HF)", leave=False):
        inputs = vit.processor(images=images, text="",
                               return_tensors="pt", padding=True).to(device)
        if hooks:
            out = vit.run_with_hooks(hooks, return_type="output", **inputs)
        else:
            out = vit(return_type="output", **inputs)
        img_feat = l2_normalize(out.image_embeds.float())
        preds = (logit_scale * (img_feat @ text_feat.t())).argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total * 100.0


@torch.no_grad()
def zero_shot_accuracy_openai(model, classnames, loader, device, sae=None, cfg=None):
    """Zero-shot accuracy using OpenAI CLIP, optionally with SAE hook."""
    logit_scale = model.logit_scale.exp()

    mean_f = torch.zeros(len(classnames), 512, device=device)
    for tmpl in openai_imagenet_template:
        tokens = openai_clip.tokenize([tmpl(c) for c in classnames], truncate=True).to(device)
        mean_f += l2_normalize(model.encode_text(tokens).float())
    text_feat = l2_normalize(mean_f / len(openai_imagenet_template))

    hook_handle = None
    if sae is not None and cfg is not None:
        num_blocks = len(model.visual.transformer.resblocks)
        layer = cfg.block_layer
        if layer < 0:
            layer = num_blocks + layer
        block = model.visual.transformer.resblocks[layer]

        def hook_fn(module, inp, output):
            orig_dtype = output.dtype
            act_bsd = output.transpose(0, 1).float()
            sae_out, _fa, _ = sae(act_bsd)
            return sae_out.transpose(0, 1).to(orig_dtype)

        hook_handle = block.register_forward_hook(hook_fn)

    correct, total = 0, 0
    for imgs, labels in tqdm(loader, desc="      acc(OAI)", leave=False):
        img_feat = l2_normalize(model.encode_image(imgs.to(device)).float())
        preds = (logit_scale * (img_feat @ text_feat.t())).argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    if hook_handle is not None:
        hook_handle.remove()

    return correct / total * 100.0


# ── Per-SAE evaluation ────────────────────────────────────────────────────────

def evaluate_one_sae(sae_info, vit_base, lora_cache, args):
    """Evaluate a single SAE against its matching dataset's backbone and eval loader.

    lora_cache: dict mapping lora_ckpt_path → (lora_model, lora_preprocess) or None
    """
    path    = sae_info["path"]
    run_id  = sae_info["run_id"]
    dataset = sae_info["dataset"]
    label   = (f"{sae_info['method']}/{dataset}/{sae_info['arch']}/"
               f"layer{sae_info['layer']}/{run_id}")
    print(f"\n{'─'*70}")
    print(f"  SAE: {label}")
    print(f"  Path: {path}")

    result = {**sae_info, "label": label}

    # Resolve per-dataset config
    ds_cfg     = DATASET_CONFIG.get(dataset, DATASET_CONFIG["unknown"])
    lora_ckpt  = ds_cfg["lora_ckpt"]
    print(f"  Dataset config: {dataset}  lora={lora_ckpt}")

    # Get (or build) LoRA model for this dataset
    lora_model, lora_preprocess = None, None
    if lora_ckpt and os.path.isfile(lora_ckpt):
        if lora_ckpt not in lora_cache:
            print(f"  [LOAD] Building LoRA model for {dataset}...")
            lora_cache[lora_ckpt] = build_lora_clip(lora_ckpt, DEVICE)
        lora_model, lora_preprocess = lora_cache[lora_ckpt]

    # Build per-dataset loaders
    hf_loader, oai_loader, classnames = build_dataset_loaders(
        dataset, args.batch_size, lora_preprocess)
    print(f"  Loaders: HF={len(hf_loader)}b  OAI={len(oai_loader) if oai_loader else 'N/A'}b"
          f"  classes={len(classnames)}")

    # Load SAE
    try:
        sae, cfg = load_sae_universal(path, DEVICE)
    except Exception as e:
        print(f"  [ERROR] Load failed: {e}")
        result["error"] = str(e)
        return result

    sae_type = getattr(cfg, "sae_type", "standard")
    print(f"  d_in={sae.d_in}, d_sae={sae.d_sae}, layer={cfg.block_layer}, "
          f"module={getattr(cfg,'module_name','?')}, type={sae_type}")

    # ── Backbone A: Base HF CLIP ──────────────────────────────────
    print(f"\n  [BASE-DOMAIN] Capturing via HF base CLIP on {dataset}...")
    base_acts = None
    try:
        base_acts = capture_activations_hf(
            vit_base, hf_loader, cfg, DEVICE, n_batches=args.n_batches)
    except Exception as e:
        print(f"    [ERROR] capture failed: {e}")

    base_metrics = {}
    if base_acts is not None:
        print(f"    acts: {base_acts.shape}")
        try:
            base_metrics.update(compute_reconstruction_metrics(sae, base_acts, DEVICE))
            _print_recon(base_metrics, prefix="    ")
        except Exception as e:
            print(f"    [WARN] recon metrics failed: {e}")
        try:
            base_metrics.update(compute_l0_and_dead(sae, base_acts, DEVICE))
            _print_l0(base_metrics, prefix="    ")
        except Exception as e:
            print(f"    [WARN] L0/dead metrics failed: {e}")
        del base_acts
        flush()

    result["base_domain"] = base_metrics

    # ── Backbone B: LoRA OpenAI CLIP (in-distribution for this dataset) ──
    lora_metrics = {}
    if lora_model is not None and oai_loader is not None:
        print(f"\n  [LORA-DOMAIN] Capturing via LoRA-{dataset} OpenAI CLIP...")
        lora_acts = None
        try:
            lora_acts = capture_activations_openai(
                lora_model, oai_loader, cfg, DEVICE, n_batches=args.n_batches)
        except Exception as e:
            print(f"    [ERROR] capture failed: {e}")

        if lora_acts is not None:
            print(f"    acts: {lora_acts.shape}")
            try:
                lora_metrics.update(compute_reconstruction_metrics(sae, lora_acts, DEVICE))
                _print_recon(lora_metrics, prefix="    ")
            except Exception as e:
                print(f"    [WARN] recon metrics failed: {e}")
            try:
                lora_metrics.update(compute_l0_and_dead(sae, lora_acts, DEVICE))
                _print_l0(lora_metrics, prefix="    ")
            except Exception as e:
                print(f"    [WARN] L0/dead metrics failed: {e}")
            del lora_acts
            flush()

    result["lora_domain"] = lora_metrics

    # ── Zero-shot accuracy ────────────────────────────────────────
    if not args.skip_accuracy:
        print(f"\n  [ACCURACY] Zero-shot on {dataset} test...")

        # Base backbone
        try:
            acc_no  = zero_shot_accuracy_hf(vit_base, classnames, hf_loader, DEVICE)
            acc_yes = zero_shot_accuracy_hf(vit_base, classnames, hf_loader, DEVICE,
                                            sae=sae, cfg=cfg)
            result["acc_base_no_sae"]   = round(acc_no, 3)
            result["acc_base_with_sae"] = round(acc_yes, 3)
            result["acc_base_delta"]    = round(acc_yes - acc_no, 3)
            print(f"    Base  — no-SAE: {acc_no:.2f}%  "
                  f"with-SAE: {acc_yes:.2f}%  Δ={acc_yes-acc_no:+.2f}%")
        except Exception as e:
            print(f"    [WARN] Base accuracy failed: {e}")

        # LoRA backbone (in-distribution)
        if lora_model is not None and oai_loader is not None:
            try:
                acc_no  = zero_shot_accuracy_openai(lora_model, classnames, oai_loader, DEVICE)
                acc_yes = zero_shot_accuracy_openai(lora_model, classnames, oai_loader, DEVICE,
                                                    sae=sae, cfg=cfg)
                result["acc_lora_no_sae"]   = round(acc_no, 3)
                result["acc_lora_with_sae"] = round(acc_yes, 3)
                result["acc_lora_delta"]    = round(acc_yes - acc_no, 3)
                print(f"    LoRA  — no-SAE: {acc_no:.2f}%  "
                      f"with-SAE: {acc_yes:.2f}%  Δ={acc_yes-acc_no:+.2f}%")
            except Exception as e:
                print(f"    [WARN] LoRA accuracy failed: {e}")

    del sae
    flush()
    return result


def _print_recon(m, prefix=""):
    if "nmse" in m:
        print(f"{prefix}nMSE(global/CLS/spatial): "
              f"{m['nmse']:.4f} / {m.get('nmse_cls','?'):.4f} / {m.get('nmse_spatial','?'):.4f}")
    if "cos_sim" in m:
        print(f"{prefix}CosSim(global/CLS/spatial): "
              f"{m['cos_sim']:.4f} / {m.get('cos_sim_cls','?'):.4f} / {m.get('cos_sim_spatial','?'):.4f}")


def _print_l0(m, prefix=""):
    if "l0_cls_mean" in m:
        print(f"{prefix}L0 CLS:     mean={m['l0_cls_mean']:.1f} ± {m['l0_cls_std']:.1f}  "
              f"p50={m['l0_cls_pct'].get('50','?')}  p95={m['l0_cls_pct'].get('95','?')}")
        print(f"{prefix}L0 Spatial: mean={m['l0_spatial_mean']:.1f} ± {m['l0_spatial_std']:.1f}  "
              f"p50={m['l0_spatial_pct'].get('50','?')}  p95={m['l0_spatial_pct'].get('95','?')}")
    if "dead_fraction" in m:
        print(f"{prefix}Dead: {m['dead_fraction']*100:.1f}%  ({m.get('n_dead','?')}/{m.get('n_latents','?')})")


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(all_results):
    print(f"\n{'═'*110}")
    print(f"  COMPREHENSIVE SAE EVALUATION — SUMMARY")
    print(f"  Domains: B=base-CLIP, L=LoRA-CLIP  (both on MedMNIST test)")
    print(f"{'═'*110}")

    HDR = (f"  {'Label':<58s}  "
           f"{'nMSE-B':>7s}  {'nMSE-L':>7s}  "
           f"{'Cos-B':>6s}  {'Cos-L':>6s}  "
           f"{'L0cls-B':>7s}  {'L0spa-B':>7s}  "
           f"{'Dead-B':>6s}  "
           f"{'AccB(Δ)':>10s}  {'AccL(Δ)':>10s}")
    print(HDR)
    SEP = (f"  {'─'*58}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  "
           f"{'─'*7}  {'─'*7}  {'─'*6}  {'─'*10}  {'─'*10}")
    print(SEP)

    by_method = {}
    for r in all_results:
        if "error" in r:
            continue
        by_method.setdefault(r.get("method", "?"), []).append(r)

    METHOD_ORDER = ["base", "lora", "maple", "masked_finetune",
                    "masked_finetune_lora", "masked_finetune_maple", "dora"]

    for method in METHOD_ORDER + [m for m in by_method if m not in METHOD_ORDER]:
        if method not in by_method:
            continue
        print(f"\n  ── {method.upper()} ──")
        for r in sorted(by_method[method],
                        key=lambda x: (x.get("dataset",""), x.get("arch",""), str(x.get("layer","")))):
            label = r.get("label","?")[:58]
            bd    = r.get("base_domain", {})
            ld    = r.get("lora_domain", {})

            def _f(d, k, fmt=".4f"):
                v = d.get(k)
                return f"{v:{fmt}}" if v is not None else "   N/A"

            acc_b = (f"{r['acc_base_with_sae']:.1f}({r['acc_base_delta']:+.1f})"
                     if "acc_base_with_sae" in r else "       N/A")
            acc_l = (f"{r['acc_lora_with_sae']:.1f}({r['acc_lora_delta']:+.1f})"
                     if "acc_lora_with_sae" in r else "       N/A")

            dead_str = (f"{bd['dead_fraction']*100:.1f}%"
                        if bd.get("dead_fraction") is not None else "  N/A")

            print(f"  {label:<58s}  "
                  f"{_f(bd,'nmse'):>7s}  {_f(ld,'nmse'):>7s}  "
                  f"{_f(bd,'cos_sim'):>6s}  {_f(ld,'cos_sim'):>6s}  "
                  f"{_f(bd,'l0_cls_mean','.1f'):>7s}  {_f(bd,'l0_spatial_mean','.1f'):>7s}  "
                  f"{dead_str:>6s}  "
                  f"{acc_b:>10s}  {acc_l:>10s}")

    print(f"\n{'═'*110}")

    # L0 distribution table
    print(f"\n{'═'*90}")
    print(f"  L0 DISTRIBUTION — Base Domain  (p=percentile)")
    print(f"{'═'*90}")
    hdr2 = f"  {'Label':<50s}  {'Token':>6s}  {'p10':>5s}  {'p25':>5s}  {'p50':>5s}  {'p75':>5s}  {'p90':>5s}  {'p95':>5s}  {'p99':>5s}"
    print(hdr2)
    print(f"  {'─'*50}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
    for r in all_results:
        if "error" in r:
            continue
        bd = r.get("base_domain", {})
        if not bd.get("l0_cls_pct"):
            continue
        label = r.get("label","?")[:50]
        for tok, key in [("CLS", "l0_cls_pct"), ("spatial", "l0_spatial_pct")]:
            pct = bd.get(key, {})
            row = f"  {label if tok=='CLS' else '':50s}  {tok:>6s}"
            for p in [10, 25, 50, 75, 90, 95, 99]:
                v = pct.get(str(p), pct.get(p))
                row += f"  {v:5.1f}" if v is not None else "    N/A"
            print(row)
    print(f"{'═'*90}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Comprehensive SAE evaluation across all checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", default="out/eval_all_saes_comprehensive.json")
    p.add_argument("--n_batches", type=int, default=50,
                   help="Batches for activation capture / dead-latent counting")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--skip_accuracy", action="store_true",
                   help="Skip zero-shot accuracy (faster)")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Filter by method (e.g. lora maple)")
    p.add_argument("--datasets_filter", nargs="+", default=None,
                   help="Filter by training dataset (e.g. medmnist eurosat)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  COMPREHENSIVE SAE EVALUATION")
    print(f"  Device    : {DEVICE}")
    print(f"  n_batches : {args.n_batches}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  skip_acc  : {args.skip_accuracy}")
    print(f"{'═'*70}")

    # ── Discover SAEs ──────────────────────────────────────────────
    all_saes = discover_all_saes()
    if args.methods:
        all_saes = [s for s in all_saes if s["method"] in args.methods]
    if args.datasets_filter:
        all_saes = [s for s in all_saes if s["dataset"] in args.datasets_filter]
    print(f"\n[EVAL] Running on {len(all_saes)} SAE(s)")

    # ── Base HF CLIP (shared across all SAEs) ─────────────────────
    print(f"\n[LOAD] Base HF CLIP...")
    dummy_cfg = None
    for s in all_saes:
        try:
            _, dummy_cfg = load_sae_universal(s["path"], "cpu")
            break
        except Exception:
            continue
    if dummy_cfg is None:
        print("[FATAL] Could not load any SAE for dummy_cfg")
        sys.exit(1)

    vit_base = load_hooked_vit(dummy_cfg, "base", BACKBONE, DEVICE)
    vit_base.eval()

    # ── LoRA models are built lazily per-dataset and cached ────────
    lora_cache = {}   # lora_ckpt_path → (model, preprocess)

    # ── Evaluate each SAE against its matching dataset ────────────
    all_results = []
    for i, sae_info in enumerate(all_saes):
        print(f"\n[{i+1}/{len(all_saes)}]")
        result = evaluate_one_sae(sae_info, vit_base, lora_cache, args)
        all_results.append(result)

        # Save after each SAE (incremental checkpoint)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Final summary ──────────────────────────────────────────────
    print_summary(all_results)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[DONE] Results saved → {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
eval_unified_metrics.py — accuracy + SAE stats for every available SAE.

For each dataset × model-type combination the following conditions are evaluated:

  CLIP-based (base + LoRA):
    1. Zero-shot base CLIP                    → accuracy
    2. Base CLIP + ImageNet SAE               → accuracy + SAE stats
    3. LoRA model                             → accuracy
    4. LoRA model + LoRA/dataset SAE (final)  → accuracy + SAE stats  [per run]

  MaPLe (if weights found):
    5. MaPLe model                            → accuracy
    6. MaPLe model + MaPLe SAE (final)        → accuracy + SAE stats  [per run]

SAE stats: L0 (mean active features/image), label_entropy_norm, dead_pct, recon_mse.

Usage:
    cd /home/sunayana/Documents/Concept_LoRA/patchsae
    python eval_unified_metrics.py
    python eval_unified_metrics.py --datasets eurosat caltech101 medmnist
    python eval_unified_metrics.py --save_json out/unified_metrics.json
"""

import argparse
import gc
import glob
import json
import math
import os
import re
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from tasks.utils import load_sae, load_hooked_vit
    from src.sae_training.hooked_vit import Hook
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    sys.exit(f"[FATAL] {e}\nRun from the patchsae/ directory.")

try:
    import clip as openai_clip
except ImportError:
    sys.exit("[FATAL] pip install git+https://github.com/openai/CLIP.git")

# ── Paths ──────────────────────────────────────────────────────────────────────
BACKBONE        = "ViT-B/16"
HF_BACKBONE     = "openai/clip-vit-base-patch16"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 64
NUM_WORKERS     = 4

CKPT_ROOT       = "out/checkpoints"
BASE_SAE_PATH   = "data/sae_weight/base/out.pt"
LORA_ROOT       = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
MAPLE_ROOT      = "/home/sunayana/Documents/Concept_LoRA/maple_weights"
MAPLE_CONFIG    = (
    "/home/sunayana/Documents/Concept_LoRA/patchsae/configs/models/maple/"
    "vit_b16_c2_ep5_batch4_2ctx.yaml"
)
DATA_ROOT       = "/home/sunayana/Documents/Concept_LoRA/data"

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

# ── Dataset configuration ──────────────────────────────────────────────────────
# Each entry: data_dir, (optional) npz_path, lora_path, exclude_classes,
#             maple_ckpt_dir (subdir under CKPT_ROOT for MaPLe SAEs),
#             maple_model_paths (list of MaPLe weight files to try)

DATASET_CFG = {
    "eurosat": {
        "data_dir":         f"{DATA_ROOT}/eurosat/2750",
        "use_npz":          False,
        "exclude_classes":  set(),
        "lora_path":        f"{LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
        "lora_sae_dir":     f"{CKPT_ROOT}/eurosat",
        "maple_sae_dir":    f"{CKPT_ROOT}/eurosat_maple",
        "maple_model_paths": sorted(glob.glob(
            f"{MAPLE_ROOT}/eurosat/**/model.pth.tar*", recursive=True
        )),
    },
    "caltech101": {
        "data_dir":         f"{DATA_ROOT}/caltech-101",
        "use_npz":          False,
        "exclude_classes":  {"BACKGROUND_Google"},
        "lora_path":        f"{LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
        "lora_sae_dir":     f"{CKPT_ROOT}/caltech101",
        "maple_sae_dir":    f"{CKPT_ROOT}/caltech101_maple",
        "maple_model_paths": sorted(glob.glob(
            f"{MAPLE_ROOT}/caltech101/**/model.pth.tar*", recursive=True
        )),
    },
    "medmnist": {
        "data_dir":         f"{DATA_ROOT}/pathmnist_imagefolder",
        "npz_path":         f"{DATA_ROOT}/pathmnist.npz",
        "use_npz":          True,
        "exclude_classes":  set(),
        "lora_path":        f"{LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
        "lora_sae_dir":     f"{CKPT_ROOT}/medmnist",
        "maple_sae_dir":    f"{CKPT_ROOT}/medmnist_maple",
        "maple_model_paths": [
            f"{MAPLE_ROOT}/model.pth.tar-5",
        ],
    },
    "dtd": {
        "data_dir":         f"{DATA_ROOT}/DTD/images",
        "use_npz":          False,
        "exclude_classes":  set(),
        "lora_path":        f"{LORA_ROOT}/dtd/16shots/seed42/lora_weights.pt",
        "lora_sae_dir":     f"{CKPT_ROOT}/dtd",
        "maple_sae_dir":    None,
        "maple_model_paths": [],
    },
    "cub2002011": {
        "data_dir":         f"{DATA_ROOT}/cub2002011/test",
        "use_npz":          False,
        "exclude_classes":  set(),
        "lora_path":        f"{LORA_ROOT}/cub2002011/16shots/seed1/lora_weights.pt",
        "lora_sae_dir":     f"{CKPT_ROOT}/cub2002011",
        "maple_sae_dir":    None,
        "maple_model_paths": [],
    },
}

# ── Utilities ──────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _find_imagefolder_root(path: str) -> str:
    """Walk into path until subdirs contain image files (handles nested layouts)."""
    for root, dirs, _ in os.walk(path):
        if dirs:
            sample = os.path.join(root, dirs[0])
            if os.path.isdir(sample):
                exts = {
                    os.path.splitext(f)[1].lower()
                    for f in os.listdir(sample)
                    if os.path.isfile(os.path.join(sample, f))
                }
                if exts & {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                    return root
    return path


# ── Dataset classes ────────────────────────────────────────────────────────────

class FilteredImageFolder(Dataset):
    """ImageFolder with class exclusion and contiguous label remapping."""

    def __init__(self, root: str, transform=None, exclude_classes=None):
        img_root = _find_imagefolder_root(root)
        full_ds  = datasets.ImageFolder(root=img_root, transform=transform)
        exclude  = exclude_classes or set()

        keep    = [i for i, (_, l) in enumerate(full_ds.samples)
                   if full_ds.classes[l] not in exclude]
        kept    = sorted({full_ds.classes[full_ds.targets[i]] for i in keep})
        o2n     = {full_ds.class_to_idx[c]: ni for ni, c in enumerate(kept)}

        self._ds        = full_ds
        self._idx       = keep
        self._map       = o2n
        self.class_names = [n.replace("_", " ") for n in kept]
        self.num_classes = len(kept)

    def __len__(self):  return len(self._idx)

    def __getitem__(self, i):
        img, lbl = self._ds[self._idx[i]]
        return img, self._map[lbl]


class MedMNISTDataset(Dataset):
    """PathMNIST test split from .npz with label remapping."""

    def __init__(self, npz_path: str, imagefolder_root: str,
                 transform=None, split: str = "test"):
        data        = np.load(npz_path)
        self.imgs   = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"].flatten().astype(int)
        self.transform = transform

        if_root = _find_imagefolder_root(imagefolder_root)
        if_ds   = datasets.ImageFolder(root=if_root)
        if_map  = {n.replace("_", " ").lower(): i for n, i in if_ds.class_to_idx.items()}

        self.label_map   = {i: if_map.get(c.lower(), i)
                            for i, c in enumerate(MEDMNIST_CLASSES)}
        self.class_names = [n.replace("_", " ") for n in if_ds.classes]
        self.num_classes = len(MEDMNIST_CLASSES)

    def __len__(self):  return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label_map[int(self.labels[i])]


def load_dataset_clip(dataset_name: str, cfg: dict, preprocess):
    """Returns (dataset, class_names) using CLIP preprocess transform (tensors)."""
    if cfg["use_npz"]:
        ds = MedMNISTDataset(
            cfg["npz_path"], cfg["data_dir"],
            transform=preprocess, split="test",
        )
        return ds, ds.class_names
    else:
        ds = FilteredImageFolder(
            cfg["data_dir"], transform=preprocess,
            exclude_classes=cfg.get("exclude_classes"),
        )
        return ds, ds.class_names


def load_dataset_maple(dataset_name: str, cfg: dict):
    """Returns (dataset, class_names) using resize-only transform (PIL images for MaPLe)."""
    resize = transforms.Resize((224, 224))
    if cfg["use_npz"]:
        ds = MedMNISTDataset(
            cfg["npz_path"], cfg["data_dir"],
            transform=resize, split="test",
        )
        return ds, ds.class_names
    else:
        ds = FilteredImageFolder(
            cfg["data_dir"], transform=resize,
            exclude_classes=cfg.get("exclude_classes"),
        )
        return ds, ds.class_names


def hf_collate(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


# ── Model loading ──────────────────────────────────────────────────────────────

def build_base_clip(device):
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    model.eval()
    return model, preprocess


def build_lora_clip(lora_path: str, device):
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    state = torch.load(lora_path, map_location=device, weights_only=False)
    if "weights" not in state:
        model.load_state_dict(state)
        model.eval()
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
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
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
                w[off:off + d] += delta.to(w.dtype)
    model.eval()
    return model, preprocess


def build_maple_vit(maple_path: str, class_names: list, ref_cfg, device):
    """Load MaPLe model as HookedVisionTransformer."""
    from tasks.utils import load_hooked_vit
    vit = load_hooked_vit(
        ref_cfg, "maple", HF_BACKBONE, device,
        model_path=maple_path,
        config_path=MAPLE_CONFIG,
        classnames=class_names,
    )
    vit.eval()
    return vit


# ── Text features ──────────────────────────────────────────────────────────────

@torch.no_grad()
def get_text_features_clip(model, class_names: list, device) -> torch.Tensor:
    n      = len(class_names)
    mean_f = torch.zeros(n, 512, device=device)
    for tmpl in openai_imagenet_template:
        tokens = openai_clip.tokenize(
            [tmpl(c) for c in class_names], truncate=True
        ).to(device)
        feats  = model.encode_text(tokens).float()
        mean_f += l2_norm(feats)
    mean_f /= len(openai_imagenet_template)
    return l2_norm(mean_f)


@torch.no_grad()
def get_text_features_maple(vit) -> torch.Tensor:
    tf = vit.model.get_text_features().float()
    return l2_norm(tf)


# ── Accuracy (CLIP / LoRA) ─────────────────────────────────────────────────────

class _SAEHookCLIP:
    """Inserts SAE reconstruction into an OpenAI CLIP resblock output."""
    def __init__(self, model, sae, sae_cfg, device):
        n     = len(model.visual.transformer.resblocks)
        layer = sae_cfg.block_layer if sae_cfg.block_layer >= 0 else n + sae_cfg.block_layer
        self.block  = model.visual.transformer.resblocks[layer]
        self.sae    = sae.to(device)
        self.handle = None

    def _fn(self, module, inp, output):
        dtype   = output.dtype
        act_bsd = output.transpose(0, 1).float()      # [B, seq, d]
        result  = self.sae(act_bsd)
        sae_out = result[0] if isinstance(result, (tuple, list)) else result
        return sae_out.transpose(0, 1).to(dtype)       # back to [seq, B, d]

    def register(self):
        self.handle = self.block.register_forward_hook(self._fn)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()


@torch.no_grad()
def compute_accuracy_clip(model, text_feat: torch.Tensor,
                          loader: DataLoader, device: str) -> float:
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        vis    = l2_norm(model.encode_image(imgs).float())
        logits = 100.0 * vis @ text_feat.T
        correct += (logits.argmax(-1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_accuracy_clip_sae(model, sae, sae_cfg,
                               text_feat: torch.Tensor,
                               loader: DataLoader, device: str) -> float:
    hook = _SAEHookCLIP(model, sae, sae_cfg, device).register()
    acc  = compute_accuracy_clip(model, text_feat, loader, device)
    hook.remove()
    return acc


# ── Accuracy (MaPLe) ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_accuracy_maple(vit, text_feat: torch.Tensor,
                           loader: DataLoader, device: str,
                           hooks=None) -> float:
    logit_scale = vit.model.logit_scale.exp()
    tf = l2_norm(text_feat.float())
    correct = total = 0
    for images, labels in loader:
        inputs = vit.processor(
            images=images, text="", return_tensors="pt", padding=True
        ).to(device)
        if hooks:
            out = vit.run_with_hooks(hooks, return_type="output", **inputs)
        else:
            out = vit(return_type="output", **inputs)
        img_feat = l2_norm(out.float())
        logits   = logit_scale * (img_feat @ tf.t())
        correct += (logits.argmax(-1).cpu() == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def make_maple_sae_hook(sae, sae_cfg):
    """Create SAE reconstruction Hook for MaPLe (is_custom=True path)."""
    block_layer = getattr(sae_cfg, "block_layer", -1)
    module_name = getattr(sae_cfg, "module_name", "resid")

    def hook_fn(act):
        orig_dtype = act.dtype
        act_bsd    = act.transpose(0, 1).float()  # [seq,B,d] → [B,seq,d]
        result     = sae(act_bsd)
        sae_out    = result[0] if isinstance(result, (tuple, list)) else result
        return sae_out.transpose(0, 1).to(orig_dtype)

    return [Hook(block_layer, module_name, hook_fn,
                 return_module_output=False, is_custom=True)]


# ── SAE stats (L0, label entropy, dead%, recon_mse) ───────────────────────────
# Definitions:
#   L0            – mean number of active (>0) features per image
#                   (averaged across sequence positions, then images)
#   label_entropy – Shannon entropy H(p) of each feature's label distribution
#                   (nats), weighted by feature firing frequency;
#                   normalised by log(num_classes) so ∈ [0,1]
#                   Low → class-selective / monosemantic
#   dead_pct      – % of features that never fired on this dataset
#   recon_mse     – mean squared error between SAE reconstruction and input,
#                   per token, across the dataset


class _ActivationCaptureCLIP:
    """Read-only hook that captures the output of an OpenAI CLIP resblock."""
    def __init__(self):
        self.acts   = None
        self.handle = None

    def register(self, block):
        def _fn(module, inp, output):
            # output: [seq, B, d] → store as [B, seq, d]
            self.acts = output.detach().float().transpose(0, 1)
        self.handle = block.register_forward_hook(_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()


@torch.no_grad()
def compute_sae_stats_clip(model, sae, sae_cfg,
                            dataset: Dataset, device: str) -> dict:
    """
    Compute L0, label_entropy_norm, dead_pct, recon_mse for an OpenAI CLIP model.
    """
    n_total   = len(model.visual.transformer.resblocks)
    layer_idx = (sae_cfg.block_layer if sae_cfg.block_layer >= 0
                 else n_total + sae_cfg.block_layer)
    block     = model.visual.transformer.resblocks[layer_idx]

    capture = _ActivationCaptureCLIP()
    capture.register(block)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    n_features  = sae.W_enc.shape[1]
    num_classes = dataset.num_classes

    total_l0      = 0.0
    total_mse     = 0.0
    total_tokens  = 0
    total_images  = 0
    label_counts  = torch.zeros(n_features, num_classes)
    fire_count    = torch.zeros(n_features)

    sae.eval().to(device)
    model.eval()

    for imgs, labels in tqdm(loader, desc="  SAE stats", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        model.encode_image(imgs)                   # triggers capture hook
        acts_bsd = capture.acts                    # [B, seq, d_model]
        B, seq, _ = acts_bsd.shape

        acts_flat = acts_bsd.reshape(-1, acts_bsd.shape[-1])
        sae_out_flat, feat_flat, *_ = sae(acts_flat)

        feat_bsd = feat_flat.reshape(B, seq, n_features)
        active   = (feat_bsd > 0).float()

        # L0: mean active features per image (avg over seq, then over batch)
        total_l0   += active.sum(-1).mean(-1).sum().item()
        total_images += B

        # Recon MSE per token
        total_mse    += (sae_out_flat - acts_flat).pow(2).mean(-1).sum().item()
        total_tokens += B * seq

        # Label distribution per feature: did feature f fire for image i?
        active_per_img = active.any(dim=1).float()          # [B, F]
        oh = F.one_hot(labels, num_classes=num_classes).float()
        label_counts += (active_per_img.T @ oh).cpu()
        fire_count   += active_per_img.sum(0).cpu()

    capture.remove()
    return _aggregate_sae_stats(
        total_l0, total_mse, total_tokens, total_images,
        label_counts, fire_count, num_classes,
    )


@torch.no_grad()
def compute_sae_stats_maple(vit, sae, sae_cfg,
                             dataset: Dataset, device: str) -> dict:
    """
    Compute L0, label_entropy_norm, dead_pct, recon_mse for a MaPLe model.
    Uses run_with_cache to capture activations non-destructively.
    """
    block_layer = getattr(sae_cfg, "block_layer", -1)
    n_features  = sae.W_enc.shape[1]
    num_classes = dataset.num_classes

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=hf_collate,
        pin_memory=(device == "cuda"),
    )

    total_l0      = 0.0
    total_mse     = 0.0
    total_tokens  = 0
    total_images  = 0
    label_counts  = torch.zeros(n_features, num_classes)
    fire_count    = torch.zeros(n_features)

    sae.eval().to(device)

    for images, labels in tqdm(loader, desc="  SAE stats (MaPLe)", leave=False):
        inputs = vit.processor(
            images=images, text="", return_tensors="pt", padding=True
        ).to(device)
        labels = labels.to(device)
        B      = labels.size(0)

        # Capture activations without modifying the forward pass
        _, cache = vit.run_with_cache(
            [(block_layer, "resid")],
            return_type="output",
            **inputs,
        )
        acts_sbd = cache[(block_layer, "resid")]   # [seq, B, d_model]
        acts_bsd = acts_sbd.transpose(0, 1).float()  # [B, seq, d_model]
        seq = acts_bsd.shape[1]

        acts_flat = acts_bsd.reshape(-1, acts_bsd.shape[-1])
        sae_out_flat, feat_flat, *_ = sae(acts_flat)

        feat_bsd = feat_flat.reshape(B, seq, n_features)
        active   = (feat_bsd > 0).float()

        total_l0    += active.sum(-1).mean(-1).sum().item()
        total_images += B

        total_mse    += (sae_out_flat - acts_flat).pow(2).mean(-1).sum().item()
        total_tokens += B * seq

        active_per_img = active.any(dim=1).float()
        oh = F.one_hot(labels, num_classes=num_classes).float()
        label_counts += (active_per_img.T @ oh).cpu()
        fire_count   += active_per_img.sum(0).cpu()

    return _aggregate_sae_stats(
        total_l0, total_mse, total_tokens, total_images,
        label_counts, fire_count, num_classes,
    )


def _aggregate_sae_stats(total_l0, total_mse, total_tokens, total_images,
                          label_counts, fire_count, num_classes) -> dict:
    dead_mask = fire_count == 0
    dead_pct  = dead_mask.float().mean().item() * 100.0

    active_mask   = ~dead_mask
    counts_active = label_counts[active_mask]
    probs = counts_active / counts_active.sum(1, keepdim=True).clamp(min=1e-9)
    entropy_per   = -(probs * (probs + 1e-12).log()).sum(1)
    weights       = fire_count[active_mask]
    label_entropy_norm = (
        (entropy_per * weights / weights.sum()).sum().item()
        / math.log(num_classes)
        if num_classes > 1 else 0.0
    )

    return {
        "l0":                 total_l0 / total_images,
        "recon_mse":          total_mse / total_tokens,
        "dead_pct":           dead_pct,
        "label_entropy_norm": label_entropy_norm,
        "n_active_features":  int(active_mask.sum().item()),
    }


# ── SAE discovery ──────────────────────────────────────────────────────────────

def discover_final_saes(sae_dir: str) -> list[tuple]:
    """Return [(run_id, block_layer, path)] for final checkpoints only."""
    if not sae_dir or not os.path.isdir(sae_dir):
        return []
    results = []
    for p in sorted(glob.glob(os.path.join(sae_dir, "*/final*/*.pt"))):
        parts = p.split(os.sep)
        run_id = parts[-3]
        m = re.search(r"_(-?\d+)_resid", parts[-1])
        layer = int(m.group(1)) if m else None
        try:
            ckpt = torch.load(p, map_location="cpu")
            sd   = ckpt.get("state_dict", ckpt)
            w    = sd.get("W_enc") if isinstance(sd, dict) else None
            if w is not None and torch.isnan(w).any():
                print(f"  [SKIP] {p} — NaN in W_enc")
                continue
        except Exception as e:
            print(f"  [SKIP] {p} — {e}")
            continue
        results.append((run_id, layer, p))
    return results


# ── Per-dataset evaluation ─────────────────────────────────────────────────────

def evaluate_dataset(dataset_name: str, cfg: dict) -> dict:
    print(f"\n{'═'*72}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'═'*72}")

    results = {"dataset": dataset_name, "conditions": []}

    # ── 1) Load base CLIP + dataset ───────────────────────────────────────────
    print("\n[1] Loading base CLIP...")
    try:
        base_model, preprocess = build_base_clip(DEVICE)
    except Exception as e:
        print(f"  [ERROR] {e}"); return results

    try:
        ds_clip, class_names = load_dataset_clip(dataset_name, cfg, preprocess)
    except Exception as e:
        print(f"  [ERROR] dataset load: {e}"); del base_model; flush(); return results

    loader_clip = DataLoader(
        ds_clip, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    print(f"  {len(ds_clip)} images | {len(class_names)} classes")

    # Text features for CLIP
    text_feat_clip = get_text_features_clip(base_model, class_names, DEVICE)

    # ── 2) Zero-shot base CLIP ────────────────────────────────────────────────
    print("\n[2] Zero-shot base CLIP (no SAE)...")
    acc_zs = compute_accuracy_clip(base_model, text_feat_clip, loader_clip, DEVICE)
    print(f"  → {acc_zs:.2f}%")
    results["conditions"].append({
        "name": "zero_shot_base_clip",
        "model": "base_clip",
        "sae": None,
        "accuracy": round(acc_zs, 4),
        "sae_stats": None,
    })

    # ── 3) Base CLIP + ImageNet SAE ───────────────────────────────────────────
    if os.path.isfile(BASE_SAE_PATH):
        print(f"\n[3] Base CLIP + ImageNet SAE ({BASE_SAE_PATH})...")
        try:
            base_sae, base_sae_cfg = load_sae(BASE_SAE_PATH, DEVICE)
            base_sae.eval()
            acc_base_sae = compute_accuracy_clip_sae(
                base_model, base_sae, base_sae_cfg, text_feat_clip, loader_clip, DEVICE
            )
            print(f"  accuracy → {acc_base_sae:.2f}%")
            stats = compute_sae_stats_clip(
                base_model, base_sae, base_sae_cfg, ds_clip, DEVICE
            )
            print(f"  L0={stats['l0']:.1f}  entropy_norm={stats['label_entropy_norm']:.4f}"
                  f"  dead%={stats['dead_pct']:.1f}")
            results["conditions"].append({
                "name": "base_clip_plus_imagenet_sae",
                "model": "base_clip",
                "sae": BASE_SAE_PATH,
                "sae_layer": base_sae_cfg.block_layer,
                "accuracy": round(acc_base_sae, 4),
                "sae_stats": stats,
            })
            del base_sae; flush()
        except Exception as e:
            print(f"  [ERROR] {e}")

    # ── 4) LoRA model ─────────────────────────────────────────────────────────
    lora_model = lora_preprocess = None
    lora_path = cfg.get("lora_path")
    text_feat_lora = None

    if lora_path and os.path.isfile(lora_path):
        print(f"\n[4] LoRA model ({lora_path})...")
        try:
            lora_model, lora_preprocess = build_lora_clip(lora_path, DEVICE)
            ds_lora, _  = load_dataset_clip(dataset_name, cfg, lora_preprocess)
            loader_lora = DataLoader(
                ds_lora, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
            )
            text_feat_lora = get_text_features_clip(lora_model, class_names, DEVICE)
            acc_lora = compute_accuracy_clip(lora_model, text_feat_lora, loader_lora, DEVICE)
            print(f"  accuracy → {acc_lora:.2f}%")
            results["conditions"].append({
                "name": "lora_model",
                "model": "lora",
                "sae": None,
                "accuracy": round(acc_lora, 4),
                "sae_stats": None,
            })
        except Exception as e:
            print(f"  [ERROR] LoRA load: {e}")
            lora_model = None
    else:
        print(f"\n[4] LoRA weights not found: {lora_path} — skipping.")

    # ── 5) LoRA model + LoRA SAEs ─────────────────────────────────────────────
    lora_sae_dir = cfg.get("lora_sae_dir")
    lora_saes    = discover_final_saes(lora_sae_dir)
    if lora_model is None:
        print(f"\n[5] Skipping LoRA SAEs (no LoRA model).")
    elif not lora_saes:
        print(f"\n[5] No final LoRA SAE checkpoints found in {lora_sae_dir}.")
    else:
        print(f"\n[5] LoRA model + LoRA SAEs ({len(lora_saes)} found):")
        for run_id, layer, sae_path in lora_saes:
            label = f"lora_plus_sae_{run_id}_L{layer}"
            print(f"  ↳ {run_id} layer={layer}")
            try:
                sae, sae_cfg = load_sae(sae_path, DEVICE)
                sae.eval()
                acc = compute_accuracy_clip_sae(
                    lora_model, sae, sae_cfg, text_feat_lora, loader_lora, DEVICE
                )
                print(f"    accuracy → {acc:.2f}%")
                stats = compute_sae_stats_clip(lora_model, sae, sae_cfg, ds_lora, DEVICE)
                print(f"    L0={stats['l0']:.1f}  entropy_norm={stats['label_entropy_norm']:.4f}"
                      f"  dead%={stats['dead_pct']:.1f}")
                results["conditions"].append({
                    "name": label,
                    "model": "lora",
                    "sae": sae_path,
                    "sae_run": run_id,
                    "sae_layer": layer,
                    "accuracy": round(acc, 4),
                    "sae_stats": stats,
                })
                del sae; flush()
            except Exception as e:
                print(f"    [ERROR] {e}")

    del base_model
    if lora_model is not None:
        del lora_model
    flush()

    # ── 6) MaPLe model + MaPLe SAEs ──────────────────────────────────────────
    maple_paths  = [p for p in cfg.get("maple_model_paths", []) if os.path.isfile(p)]
    maple_sae_dir = cfg.get("maple_sae_dir")
    maple_saes   = discover_final_saes(maple_sae_dir)

    if not maple_paths:
        print(f"\n[6/7] No MaPLe model weights found — skipping MaPLe evaluation.")
    else:
        print(f"\n[6] MaPLe model(s) found: {len(maple_paths)}")

        # Load dataset for MaPLe (PIL images)
        try:
            ds_maple, _ = load_dataset_maple(dataset_name, cfg)
            loader_maple = DataLoader(
                ds_maple, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, collate_fn=hf_collate,
                pin_memory=(DEVICE == "cuda"),
            )
        except Exception as e:
            print(f"  [ERROR] MaPLe dataset load: {e}")
            maple_paths = []

        # We need a reference SAE cfg to build MaPLe model
        ref_cfg = None
        if os.path.isfile(BASE_SAE_PATH):
            try:
                _, ref_cfg = load_sae(BASE_SAE_PATH, DEVICE)
            except Exception:
                pass
        if ref_cfg is None and maple_saes:
            try:
                _, ref_cfg = load_sae(maple_saes[0][2], DEVICE)
            except Exception:
                pass

        for maple_path in maple_paths:
            tag = os.path.relpath(maple_path, MAPLE_ROOT)
            print(f"\n  MaPLe model: {tag}")
            if ref_cfg is None:
                print("  [ERROR] No reference SAE cfg to build MaPLe model — skipping.")
                continue
            try:
                vit = build_maple_vit(maple_path, class_names, ref_cfg, DEVICE)
            except Exception as e:
                print(f"  [ERROR] build MaPLe: {e}"); flush(); continue

            try:
                text_feat_maple = get_text_features_maple(vit)
            except Exception as e:
                print(f"  [ERROR] text features: {e}"); del vit; flush(); continue

            # 6a) MaPLe only
            print("\n  [6a] MaPLe (no SAE)...")
            try:
                acc_maple = compute_accuracy_maple(vit, text_feat_maple, loader_maple, DEVICE)
                print(f"    accuracy → {acc_maple:.2f}%")
                results["conditions"].append({
                    "name": f"maple_model_{tag}",
                    "model": "maple",
                    "sae": None,
                    "accuracy": round(acc_maple, 4),
                    "sae_stats": None,
                    "maple_model": maple_path,
                })
            except Exception as e:
                print(f"    [ERROR] {e}")

            # 6b) MaPLe + ImageNet base SAE
            if os.path.isfile(BASE_SAE_PATH):
                print("\n  [6b] MaPLe + ImageNet SAE...")
                try:
                    b_sae, b_cfg = load_sae(BASE_SAE_PATH, DEVICE)
                    b_sae.eval()
                    hooks = make_maple_sae_hook(b_sae, b_cfg)
                    acc = compute_accuracy_maple(vit, text_feat_maple, loader_maple, DEVICE, hooks)
                    print(f"    accuracy → {acc:.2f}%")
                    stats = compute_sae_stats_maple(vit, b_sae, b_cfg, ds_maple, DEVICE)
                    print(f"    L0={stats['l0']:.1f}  entropy_norm={stats['label_entropy_norm']:.4f}"
                          f"  dead%={stats['dead_pct']:.1f}")
                    results["conditions"].append({
                        "name": f"maple_plus_imagenet_sae_{tag}",
                        "model": "maple",
                        "sae": BASE_SAE_PATH,
                        "sae_layer": b_cfg.block_layer,
                        "accuracy": round(acc, 4),
                        "sae_stats": stats,
                        "maple_model": maple_path,
                    })
                    del b_sae; flush()
                except Exception as e:
                    print(f"    [ERROR] {e}")

            # 6c) MaPLe + dataset-specific MaPLe SAEs
            if not maple_saes:
                print(f"\n  [6c] No MaPLe SAEs found in {maple_sae_dir}.")
            else:
                print(f"\n  [6c] MaPLe + dataset SAEs ({len(maple_saes)} found):")
                for run_id, layer, sae_path in maple_saes:
                    label = f"maple_plus_sae_{run_id}_L{layer}_{tag}"
                    print(f"    ↳ {run_id} layer={layer}")
                    try:
                        sae, sae_cfg = load_sae(sae_path, DEVICE)
                        sae.eval()
                        hooks = make_maple_sae_hook(sae, sae_cfg)
                        acc = compute_accuracy_maple(
                            vit, text_feat_maple, loader_maple, DEVICE, hooks
                        )
                        print(f"      accuracy → {acc:.2f}%")
                        stats = compute_sae_stats_maple(vit, sae, sae_cfg, ds_maple, DEVICE)
                        print(f"      L0={stats['l0']:.1f}"
                              f"  entropy_norm={stats['label_entropy_norm']:.4f}"
                              f"  dead%={stats['dead_pct']:.1f}")
                        results["conditions"].append({
                            "name": label,
                            "model": "maple",
                            "sae": sae_path,
                            "sae_run": run_id,
                            "sae_layer": layer,
                            "accuracy": round(acc, 4),
                            "sae_stats": stats,
                            "maple_model": maple_path,
                        })
                        del sae; flush()
                    except Exception as e:
                        print(f"      [ERROR] {e}")

            del vit; flush()

    return results


# ── Pretty-print summary table ─────────────────────────────────────────────────

def print_summary(all_results: list):
    print(f"\n\n{'═'*110}")
    print("  SUMMARY")
    print(f"{'═'*110}")
    hdr = (f"  {'Condition':<52} {'Acc%':>6}  "
           f"{'L0':>7}  {'EntNorm':>7}  {'Dead%':>6}  {'MSE':>9}")
    print(hdr)
    print(f"  {'─'*106}")
    for res in all_results:
        print(f"\n  [{res['dataset'].upper()}]")
        for c in res["conditions"]:
            s = c.get("sae_stats")
            acc_s  = f"{c['accuracy']:6.2f}%"
            l0_s   = f"{s['l0']:7.1f}"  if s else f"{'—':>7}"
            ent_s  = f"{s['label_entropy_norm']:7.4f}" if s else f"{'—':>7}"
            dead_s = f"{s['dead_pct']:6.1f}"            if s else f"{'—':>6}"
            mse_s  = f"{s['recon_mse']:9.6f}"           if s else f"{'—':>9}"
            print(f"  {c['name']:<52} {acc_s}  {l0_s}  {ent_s}  {dead_s}  {mse_s}")
    print(f"\n{'═'*110}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Unified accuracy + SAE-stats evaluation for all available SAEs."
    )
    ap.add_argument(
        "--datasets", nargs="+",
        default=list(DATASET_CFG.keys()),
        choices=list(DATASET_CFG.keys()),
        help="Datasets to evaluate (default: all).",
    )
    ap.add_argument(
        "--save_json", default="out/unified_metrics.json",
        help="Output JSON path.",
    )
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    # update module-level batch/worker counts so all helper fns pick them up
    import eval_unified_metrics as _self
    _self.BATCH_SIZE  = args.batch_size
    _self.NUM_WORKERS = args.num_workers

    print(f"\n  Device   : {DEVICE}")
    print(f"  Datasets : {args.datasets}")
    print(f"  Save to  : {args.save_json}")

    all_results = []
    for ds_name in args.datasets:
        cfg = DATASET_CFG.get(ds_name)
        if cfg is None:
            print(f"[WARN] No config for {ds_name} — skipping.")
            continue
        res = evaluate_dataset(ds_name, cfg)
        all_results.append(res)

    print_summary(all_results)

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {args.save_json}")


if __name__ == "__main__":
    main()

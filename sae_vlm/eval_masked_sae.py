#!/usr/bin/env python3
"""
Evaluate a masked-finetuned SAE on both ImageNet and MedMNIST concepts.

Evaluates concept preservation (ImageNet) and concept adaptation (MedMNIST)
by comparing the masked-finetuned SAE against the original base ImageNet SAE.

Metrics:
  1. Zero-shot accuracy (with SAE reconstruction hook in the vision encoder)
     - On MedMNIST test set: does the finetuned SAE help domain accuracy?
     - On ImageNet val subset: does it preserve general concepts?
  2. SAE reconstruction quality (MSE, cosine sim)
  3. Feature analysis: protected vs free unit activity, sparsity, L0
  4. Linear probe on SAE latents (logistic regression on sparse codes)

Usage:
    python eval_masked_sae.py \
        --masked_sae_path out/checkpoints/masked_finetune/.../final_*.pt \
        --base_sae_path data/sae_weight/base/out.pt \
        --lora_checkpoint /path/to/lora_weights.pt

    # Quick run (skip ImageNet, skip linear probe):
    python eval_masked_sae.py \
        --masked_sae_path out/checkpoints/masked_finetune/.../final_*.pt \
        --skip_imagenet --skip_linear_probe
"""




import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from src.sae_training.loaders import load_sae, load_hooked_vit
    from src.sae_training.hooked_vit import Hook
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] {e}\nRun from patchsae root.")
    sys.exit(1)

try:
    import clip as openai_clip
    HAS_OPENAI_CLIP = True
except ImportError:
    HAS_OPENAI_CLIP = False
import os

#set tokenize parallelism to false to avoid deadlocks on some systems
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

# ══════════════════════════════════════════════════════════════════════════
# Config defaults
# ══════════════════════════════════════════════════════════════════════════
BASE_SAE_PATH = "data/sae_weight/base/out.pt"
LORA_CHECKPOINT = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
TRAIN_ROOT = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
NPZ_PATH = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
IMAGENET_VAL_PATH = "evanarlian/imagenet_1k_resized_256"
BACKBONE = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def l2_normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def get_processor_transform():
    """Minimal transform: resize, keep as PIL for HF CLIPProcessor."""
    return transforms.Resize((224, 224))


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def get_imagefolder_classnames(root):
    ds = datasets.ImageFolder(root=root)
    idx_to_name = {i: n.replace("_", " ") for n, i in ds.class_to_idx.items()}
    return [idx_to_name[i] for i in range(len(idx_to_name))]


def build_npz_to_if_mapping(root):
    ds = datasets.ImageFolder(root=root)
    if_map = {n.replace("_", " ").lower(): i for n, i in ds.class_to_idx.items()}
    mapping = {}
    for npz_idx, classname in enumerate(MEDMNIST_CLASSES):
        key = classname.lower()
        if key in if_map:
            mapping[npz_idx] = if_map[key]
        else:
            mapping[npz_idx] = npz_idx
    return mapping


# ══════════════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════════════

class NpzSplitDataset(Dataset):
    """MedMNIST dataset from .npz file (train/val/test split)."""
    def __init__(self, npz_path, mapping, transform, split="test", return_pil=False):
        data = np.load(npz_path)
        key = {"train": "train", "val": "val", "test": "test"}[split]
        self.imgs = data[f"{key}_images"]
        self.labels = data[f"{key}_labels"].flatten()
        self.mapping = mapping
        self.transform = transform
        self.return_pil = return_pil

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i])
        img = self.transform(img)
        return img, self.mapping[int(self.labels[i])]


class ImageNetSubsetDataset(Dataset):
    """Random subset of ImageNet for quick evaluation."""
    def __init__(self, hf_dataset, n_samples=5000, transform=None, seed=42):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(hf_dataset), min(n_samples, len(hf_dataset)), replace=False)
        self.indices = sorted(indices.tolist())
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.dataset[self.indices[i]]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


def _hf_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)


# ══════════════════════════════════════════════════════════════════════════
# SAE loading (supports masked SAE checkpoints with protected_mask)
# ══════════════════════════════════════════════════════════════════════════

def load_masked_sae(path, device):
    """Load a masked-finetuned SAE checkpoint.

    Returns:
        sae: SparseAutoencoder module
        cfg: Config object
        protected_mask: bool tensor [d_sae] or None
    """
    from src.sae_training.config import Config
    from src.sae_training.sparse_autoencoder import SparseAutoencoder

    ckpt = torch.load(path, map_location="cpu")
    cfg = Config(ckpt.get("cfg", ckpt.get("config")))
    sae = SparseAutoencoder(cfg, device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(device)

    protected_mask = ckpt.get("protected_mask", None)
    if protected_mask is not None:
        protected_mask = protected_mask.to(device).bool()
        n_prot = protected_mask.sum().item()
        print(f"  Protected mask loaded: {n_prot}/{sae.d_sae} units protected "
              f"({100 * n_prot / sae.d_sae:.1f}%)")
    else:
        print("  No protected_mask in checkpoint.")

    return sae, cfg, protected_mask


# ══════════════════════════════════════════════════════════════════════════
# LoRA CLIP (OpenAI)
# ══════════════════════════════════════════════════════════════════════════

def _extract_lora_AB(ld, proj):
    if proj in ld and isinstance(ld[proj], dict):
        try:
            return ld[proj]["w_lora_A"], ld[proj]["w_lora_B"]
        except KeyError:
            pass
    try:
        return ld[f"{proj}.w_lora_A"], ld[f"{proj}.w_lora_B"]
    except KeyError:
        return None, None


def build_lora_openai_clip(lora_path, device):
    """Load OpenAI CLIP and merge LoRA weights."""
    if not HAS_OPENAI_CLIP:
        print("[FATAL] pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    model, preprocess = openai_clip.load("ViT-B/16", device=device)
    lora_state = torch.load(lora_path, map_location=device)

    if "weights" not in lora_state:
        model.load_state_dict(lora_state)
        return model, preprocess

    layers, meta = lora_state["weights"], lora_state["metadata"]
    scale = meta["alpha"] / math.sqrt(meta["r"])

    with torch.no_grad():
        for i in range(12):
            key = f"layer_{i}"
            if key not in layers:
                continue
            ld = layers[key]
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _extract_lora_AB(ld, proj)
                if A is None:
                    continue
                w[off : off + d] += (
                    scale * (B.float().to(device) @ A.float().to(device))
                ).to(w.dtype)

        for i in range(12, 24):
            key = f"layer_{i}"
            if key not in layers:
                continue
            ld = layers[key]
            w = model.visual.transformer.resblocks[i - 12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _extract_lora_AB(ld, proj)
                if A is None:
                    continue
                w[off : off + d] += (
                    scale * (B.float().to(device) @ A.float().to(device))
                ).to(w.dtype)

    return model, preprocess


# ══════════════════════════════════════════════════════════════════════════
# SAE application helpers
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def apply_sae(sae, act, sae_bias=0.0, alpha=1.0):
    """Run activations through SAE and return result + diagnostics."""
    sae_out = sae(act)[0] - sae_bias

    if alpha == 1.0:
        result = sae_out
    elif alpha == 0.0:
        result = act
    else:
        result = act + alpha * (sae_out - act)

    mse = (act - sae_out).pow(2).mean().item()
    cos = F.cosine_similarity(
        act.reshape(-1, act.shape[-1]),
        sae_out.reshape(-1, sae_out.shape[-1]),
        dim=-1,
    ).mean().item()
    delta_norm = (sae_out - act).norm(dim=-1).mean().item()
    act_norm = act.norm(dim=-1).mean().item()

    return result, {
        "mse": mse,
        "cos_sim": cos,
        "delta_norm": delta_norm,
        "act_norm": act_norm,
        "relative_delta": delta_norm / max(act_norm, 1e-8),
    }


def get_sae_latents(sae, activations):
    """Extract sparse latent codes from SAE encoder."""
    out = sae(activations)
    if isinstance(out, tuple) and len(out) >= 2:
        candidate = out[1]
        if candidate.shape[-1] >= activations.shape[-1]:
            return candidate

    if hasattr(sae, "W_enc") and hasattr(sae, "b_enc"):
        x = activations
        if hasattr(sae, "b_dec"):
            x = x - sae.b_dec
        z = x @ sae.W_enc + sae.b_enc
        return torch.nn.functional.relu(z)

    raise RuntimeError("Cannot extract SAE latents.")


# ══════════════════════════════════════════════════════════════════════════
# OpenAI CLIP hooks
# ══════════════════════════════════════════════════════════════════════════

class OpenAISAEHook:
    """Hook SAE reconstruction into OpenAI CLIP resblock."""

    def __init__(self, model, sae, cfg, device, sae_bias=0.0, alpha=1.0):
        self.sae = sae
        self.device = device
        self.sae_bias = sae_bias
        self.alpha = alpha
        self.handle = None
        self._logged = False

        num_blocks = len(model.visual.transformer.resblocks)
        layer = cfg.block_layer
        if layer < 0:
            layer = num_blocks + layer
        self.block = model.visual.transformer.resblocks[layer]
        self.layer_idx = layer

    def _hook_fn(self, module, input, output):
        act = output
        orig_dtype = act.dtype
        act_bsd = act.transpose(0, 1).float()
        result_bsd, diag = apply_sae(self.sae, act_bsd, self.sae_bias, self.alpha)
        if not self._logged:
            self._logged = True
            print(f"    [HOOK] MSE={diag['mse']:.6f}, cos={diag['cos_sim']:.6f}")
        return result_bsd.transpose(0, 1).to(orig_dtype)

    def register(self):
        self.handle = self.block.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ══════════════════════════════════════════════════════════════════════════
# HF CLIP hooks
# ══════════════════════════════════════════════════════════════════════════

def make_sae_hook_hf(sae, cfg, vit_type, sae_bias=0.0, alpha=1.0):
    """Create SAE reconstruction hook for HuggingFace CLIP."""
    logged = [False]

    if vit_type == "maple":
        def fn(act):
            orig_dtype = act.dtype
            a = act.transpose(0, 1).float()
            result, diag = apply_sae(sae, a, sae_bias, alpha)
            if not logged[0]:
                logged[0] = True
                print(f"    [HOOK] MSE={diag['mse']:.6f}, cos={diag['cos_sim']:.6f}")
            return result.transpose(0, 1).to(orig_dtype)

        return [Hook(cfg.block_layer, cfg.module_name, fn,
                     return_module_output=False, is_custom=True)]
    else:
        def fn(act):
            orig = act[:, :, :].clone().float()
            result, diag = apply_sae(sae, orig, sae_bias, alpha)
            act[:, :, :] = result.to(act.dtype)
            if not logged[0]:
                logged[0] = True
                print(f"    [HOOK] MSE={diag['mse']:.6f}, cos={diag['cos_sim']:.6f}")
            return (act,)

        return [Hook(cfg.block_layer, cfg.module_name, fn,
                     return_module_output=False, is_custom=False)]


# ══════════════════════════════════════════════════════════════════════════
# Accuracy computation
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_accuracy_openai(model, text_features, loader, device):
    logit_scale = model.logit_scale.exp()
    tf = l2_normalize(text_features.float())
    correct, total = 0, 0
    for imgs, labels in tqdm(loader, desc="      eval", leave=False):
        img_feat = model.encode_image(imgs.to(device))
        img_feat = l2_normalize(img_feat.float())
        preds = (logit_scale * (img_feat @ tf.t())).argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100


@torch.no_grad()
def get_text_features_openai(model, classnames, device):
    mean_f = torch.zeros(len(classnames), 512, device=device)
    for tmpl in openai_imagenet_template:
        prompts = [tmpl(c) for c in classnames]
        tokens = openai_clip.tokenize(prompts).to(device)
        f = model.encode_text(tokens).float()
        f = l2_normalize(f)
        mean_f += f
    mean_f /= len(openai_imagenet_template)
    return l2_normalize(mean_f)


@torch.no_grad()
def compute_accuracy_hf(vit, text_features, loader, vit_type, device, hooks=None):
    correct, total = 0, 0
    logit_scale = vit.model.logit_scale.exp()
    tf = l2_normalize(text_features.float())
    for images, labels in tqdm(loader, desc="      eval", leave=False):
        inputs = vit.processor(
            images=images, text="", return_tensors="pt", padding=True
        ).to(device)
        if hooks:
            out = vit.run_with_hooks(hooks, return_type="output", **inputs)
        else:
            out = vit(return_type="output", **inputs)
        img_feat = out if vit_type == "maple" else out.image_embeds
        img_feat = l2_normalize(img_feat.float())
        preds = (logit_scale * (img_feat @ tf.t())).argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100


@torch.no_grad()
def get_text_features_hf(vit, device, classnames, vit_type):
    from torch.nn.utils.rnn import pad_sequence

    if vit_type == "maple":
        tf = vit.model.get_text_features()
        return l2_normalize(tf)
    mean_f = 0
    for tmpl in openai_imagenet_template:
        ids = [
            vit.processor(text=tmpl(c), return_tensors="pt", padding=False, truncation=True).input_ids[0]
            for c in classnames
        ]
        padded = pad_sequence(ids, batch_first=True).to(device)
        f = vit.model.get_text_features(padded)
        f = l2_normalize(f)
        mean_f += f
    mean_f /= len(openai_imagenet_template)
    return l2_normalize(mean_f)


# ══════════════════════════════════════════════════════════════════════════
# Feature analysis (protected vs free)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_sae_features(sae, activation_store_or_loader, protected_mask,
                         n_batches=50, device="cuda", use_loader=False,
                         vit=None, cfg=None, vit_type="base",
                         openai_model=None):
    """Analyze feature activity for protected vs free units.

    Args:
        openai_model: If provided, use OpenAI CLIP model for activation capture
                      instead of HF vit. The loader should yield (tensor_imgs, labels).

    Returns dict with per-group stats:
      - mean_activity, max_activity, l0, dead_fraction, top_features
    """
    sae.eval()

    d_sae = sae.d_sae
    acc_activity = torch.zeros(d_sae, device=device, dtype=torch.float32)
    acc_fire_count = torch.zeros(d_sae, device=device, dtype=torch.float32)
    total_tokens = 0
    n_done = 0

    if use_loader and openai_model is not None:
        # OpenAI CLIP path: hook into resblocks to capture activations
        captured = {}
        num_blocks = len(openai_model.visual.transformer.resblocks)
        layer = cfg.block_layer
        if layer < 0:
            layer = num_blocks + layer
        block = openai_model.visual.transformer.resblocks[layer]

        def hook_fn(module, input, output):
            captured["act"] = output.transpose(0, 1).float()  # (S,B,D) -> (B,S,D)

        handle = block.register_forward_hook(hook_fn)

        for imgs, labels in activation_store_or_loader:
            if n_done >= n_batches:
                break
            captured.clear()
            _ = openai_model.encode_image(imgs.to(device))

            act = captured["act"]
            _, feature_acts, _ = sae.forward_standard(act)
            flat_acts = feature_acts.reshape(-1, d_sae)
            acc_activity += flat_acts.abs().mean(dim=0)
            acc_fire_count += (flat_acts > 0).float().sum(dim=0)
            total_tokens += flat_acts.shape[0]
            n_done += 1

        handle.remove()

    elif use_loader:
        # HF CLIP path: use hooked_vit
        captured = {}

        def capture_fn(act):
            captured["act"] = act[:, :, :].clone().float()
            return (act,)

        hooks = [Hook(cfg.block_layer, cfg.module_name, capture_fn,
                      return_module_output=False, is_custom=False)]

        for images, labels in activation_store_or_loader:
            if n_done >= n_batches:
                break
            inputs = vit.processor(
                images=images, text="", return_tensors="pt", padding=True
            ).to(device)
            captured.clear()
            _ = vit.run_with_hooks(hooks, return_type="output", **inputs)

            act = captured["act"]
            _, feature_acts, _ = sae.forward_standard(act)
            flat_acts = feature_acts.reshape(-1, d_sae)
            acc_activity += flat_acts.abs().mean(dim=0)
            acc_fire_count += (flat_acts > 0).float().sum(dim=0)
            total_tokens += flat_acts.shape[0]
            n_done += 1
    else:
        # Direct activation store
        for _ in range(n_batches):
            x = activation_store_or_loader.get_batch_activations()
            _, feature_acts, _ = sae.forward_standard(x)
            flat_acts = feature_acts.reshape(-1, d_sae)
            acc_activity += flat_acts.abs().mean(dim=0)
            acc_fire_count += (flat_acts > 0).float().sum(dim=0)
            total_tokens += flat_acts.shape[0]
            n_done += 1

    mean_activity = acc_activity / max(n_done, 1)
    fire_freq = acc_fire_count / max(total_tokens, 1)

    if protected_mask is not None:
        prot = protected_mask.bool()
        free = ~prot
    else:
        prot = torch.zeros(d_sae, device=device, dtype=torch.bool)
        free = torch.ones(d_sae, device=device, dtype=torch.bool)

    results = {}
    for name, mask in [("protected", prot), ("free", free), ("all", torch.ones(d_sae, device=device, dtype=torch.bool))]:
        n = mask.sum().item()
        if n == 0:
            results[name] = {"n": 0}
            continue

        act_subset = mean_activity[mask]
        freq_subset = fire_freq[mask]

        results[name] = {
            "n": int(n),
            "mean_activity": act_subset.mean().item(),
            "max_activity": act_subset.max().item(),
            "median_activity": act_subset.median().item(),
            "l0": freq_subset.mean().item() * n,  # avg number of active features
            "fire_rate_mean": freq_subset.mean().item(),
            "dead_fraction": (freq_subset < 1e-6).float().mean().item(),
            "top5_indices": torch.topk(act_subset, min(5, n)).indices.tolist(),
            "top5_activity": torch.topk(act_subset, min(5, n)).values.tolist(),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════
# Reconstruction quality
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_reconstruction_quality(sae, loader_or_store, n_batches=30,
                                   device="cuda", use_loader=False,
                                   vit=None, cfg=None, openai_model=None):
    """Measure SAE reconstruction MSE, cosine sim over batches."""
    sae.eval()
    mse_total, cos_total, count = 0.0, 0.0, 0

    if use_loader and openai_model is not None:
        # OpenAI CLIP path
        captured = {}
        num_blocks = len(openai_model.visual.transformer.resblocks)
        layer = cfg.block_layer
        if layer < 0:
            layer = num_blocks + layer
        block = openai_model.visual.transformer.resblocks[layer]

        def hook_fn(module, input, output):
            captured["act"] = output.transpose(0, 1).float()

        handle = block.register_forward_hook(hook_fn)
        n_done = 0
        for imgs, labels in loader_or_store:
            if n_done >= n_batches:
                break
            captured.clear()
            _ = openai_model.encode_image(imgs.to(device))
            x = captured["act"]
            x_hat = sae(x)[0]
            flat_x = x.reshape(-1, x.shape[-1])
            flat_xh = x_hat.reshape(-1, x_hat.shape[-1])
            mse_total += (flat_x - flat_xh).pow(2).mean().item()
            cos_total += F.cosine_similarity(flat_x, flat_xh, dim=-1).mean().item()
            count += 1
            n_done += 1
        handle.remove()

    elif use_loader:
        # HF CLIP path
        captured = {}

        def capture_fn(act):
            captured["act"] = act[:, :, :].clone().float()
            return (act,)

        hooks = [Hook(cfg.block_layer, cfg.module_name, capture_fn,
                      return_module_output=False, is_custom=False)]
        n_done = 0
        for images, labels in loader_or_store:
            if n_done >= n_batches:
                break
            inputs = vit.processor(
                images=images, text="", return_tensors="pt", padding=True
            ).to(device)
            captured.clear()
            _ = vit.run_with_hooks(hooks, return_type="output", **inputs)
            x = captured["act"]
            x_hat = sae(x)[0]
            flat_x = x.reshape(-1, x.shape[-1])
            flat_xh = x_hat.reshape(-1, x_hat.shape[-1])
            mse_total += (flat_x - flat_xh).pow(2).mean().item()
            cos_total += F.cosine_similarity(flat_x, flat_xh, dim=-1).mean().item()
            count += 1
            n_done += 1
    else:
        for _ in range(n_batches):
            x = loader_or_store.get_batch_activations()
            x_hat = sae(x)[0]
            flat_x = x.reshape(-1, x.shape[-1])
            flat_xh = x_hat.reshape(-1, x_hat.shape[-1])
            mse_total += (flat_x - flat_xh).pow(2).mean().item()
            cos_total += F.cosine_similarity(flat_x, flat_xh, dim=-1).mean().item()
            count += 1

    return {
        "mse": mse_total / max(count, 1),
        "cos_sim": cos_total / max(count, 1),
        "n_batches": count,
    }


# ══════════════════════════════════════════════════════════════════════════
# Latent extraction & linear probe
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_sae_latents_openai(model, sae, cfg, loader, device, pool="cls"):
    all_latents, all_labels = [], []
    captured = {}

    num_blocks = len(model.visual.transformer.resblocks)
    layer = cfg.block_layer
    if layer < 0:
        layer = num_blocks + layer
    block = model.visual.transformer.resblocks[layer]

    def hook_fn(module, input, output):
        captured["act"] = output.transpose(0, 1).float()

    handle = block.register_forward_hook(hook_fn)

    for imgs, labels in tqdm(loader, desc="      extract latents", leave=False):
        captured.clear()
        _ = model.encode_image(imgs.to(device))
        act = captured["act"]
        latents = get_sae_latents(sae, act)
        if pool == "cls":
            pooled = latents[:, 0, :]
        else:
            pooled = latents.mean(dim=1)
        all_latents.append(pooled.cpu().half().numpy())
        all_labels.append(labels.numpy())

    handle.remove()
    return np.concatenate(all_latents).astype(np.float32), np.concatenate(all_labels)


@torch.no_grad()
def extract_sae_latents_hf(vit, sae, cfg, loader, vit_type, device, pool="cls"):
    all_latents, all_labels = [], []
    captured = {}

    def capture_fn(act):
        captured["act"] = act[:, :, :].clone().float()
        return (act,)

    hooks = [Hook(cfg.block_layer, cfg.module_name, capture_fn,
                  return_module_output=False, is_custom=False)]

    for images, labels in tqdm(loader, desc="      extract latents", leave=False):
        inputs = vit.processor(
            images=images, text="", return_tensors="pt", padding=True
        ).to(device)
        captured.clear()
        _ = vit.run_with_hooks(hooks, return_type="output", **inputs)
        act = captured["act"]
        latents = get_sae_latents(sae, act)
        if pool == "cls":
            pooled = latents[:, 0, :]
        else:
            pooled = latents.mean(dim=1)
        all_latents.append(pooled.cpu().half().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_latents).astype(np.float32), np.concatenate(all_labels)


def train_linear_probe(train_latents, train_labels, test_latents, test_labels,
                       max_iter=1000):
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.linear_model import SGDClassifier

    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(train_latents)
    X_test = scaler.transform(test_latents)

    clf = SGDClassifier(
        loss="log_loss", alpha=1e-4, max_iter=max_iter,
        tol=1e-3, random_state=42, n_jobs=1,
    )
    clf.fit(X_train, train_labels)

    train_acc = clf.score(X_train, train_labels) * 100
    test_acc = clf.score(X_test, test_labels) * 100
    return test_acc, train_acc


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Evaluate masked-finetuned SAE on ImageNet and MedMNIST concepts"
    )
    p.add_argument("--masked_sae_path", type=str, required=True,
                   help="Path to masked-finetuned SAE checkpoint (.pt)")
    p.add_argument("--base_sae_path", type=str, default=BASE_SAE_PATH,
                   help="Path to base ImageNet SAE for comparison")
    p.add_argument("--lora_checkpoint", type=str, default=LORA_CHECKPOINT)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--save_dir", type=str, default="out/eval_masked_sae")
    p.add_argument("--sae_alpha", type=float, default=1.0)
    p.add_argument("--sae_bias", type=float, default=0.0)

    # What to evaluate
    p.add_argument("--skip_imagenet", action="store_true",
                   help="Skip ImageNet evaluation (faster)")
    p.add_argument("--skip_linear_probe", action="store_true",
                   help="Skip linear probe evaluation")
    p.add_argument("--skip_reconstruction", action="store_true",
                   help="Skip reconstruction quality measurement")
    p.add_argument("--imagenet_n_samples", type=int, default=5000,
                   help="Number of ImageNet samples for eval (default: 5000)")
    p.add_argument("--feature_analysis_batches", type=int, default=50,
                   help="Number of batches for feature analysis")

    p.add_argument("--maple_model_path", type=str, default=None, help="Path to MaPLe model checkpoint (pth.tar)")
    p.add_argument("--maple_config_path", type=str, default=None, help="Path to MaPLe config YAML")

    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print("  MASKED SAE EVALUATION")
    print("=" * 70)
    print(f"  Device:           {DEVICE}")
    print(f"  Masked SAE:       {args.masked_sae_path}")
    print(f"  Base SAE:         {args.base_sae_path}")
    print(f"  LoRA checkpoint:  {args.lora_checkpoint}")
    if args.maple_model_path and args.maple_config_path:
        print(f"  MaPLe model:      {args.maple_model_path}")
        print(f"  MaPLe config:     {args.maple_config_path}")
    print(f"  SAE alpha:        {args.sae_alpha}")
    print()

    all_results = {}

    # ── Load SAEs ─────────────────────────────────────────────────
    print("[1] Loading SAEs...")
    masked_sae, masked_cfg, protected_mask = load_masked_sae(args.masked_sae_path, DEVICE)
    print(f"    Masked SAE: d_in={masked_sae.d_in}, d_sae={masked_sae.d_sae}, "
          f"block_layer={masked_cfg.block_layer}")

    base_sae, base_cfg = load_sae(args.base_sae_path, DEVICE)
    print(f"    Base SAE:   d_in={base_sae.d_in}, d_sae={base_sae.d_sae}, "
          f"block_layer={base_cfg.block_layer}")

    # ── MedMNIST setup ────────────────────────────────────────────
    print("\n[2] Setting up MedMNIST data...")
    medmnist_classnames = get_imagefolder_classnames(TRAIN_ROOT)
    label_mapping = build_npz_to_if_mapping(TRAIN_ROOT)

    hf_transform = get_processor_transform()
    medmnist_test_ds = NpzSplitDataset(NPZ_PATH, label_mapping, hf_transform,
                                        split="test", return_pil=True)
    medmnist_test_loader = DataLoader(medmnist_test_ds, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4,
                                      collate_fn=_hf_collate_fn)
    print(f"    MedMNIST test: {len(medmnist_test_ds)} samples, {len(medmnist_classnames)} classes")

    # ── Build LoRA model (OpenAI CLIP) ────────────────────────────
    print("\n[3] Loading LoRA CLIP model...")
    lora_model, lora_preprocess = build_lora_openai_clip(args.lora_checkpoint, DEVICE)
    lora_model.eval()

    medmnist_test_ds_oai = NpzSplitDataset(NPZ_PATH, label_mapping, lora_preprocess,
                                            split="test", return_pil=False)
    medmnist_test_loader_oai = DataLoader(medmnist_test_ds_oai, batch_size=args.batch_size,
                                           shuffle=False, num_workers=4, pin_memory=True)

    # ── Build base HF model ──────────────────────────────────────
    print("[4] Loading base HF CLIP model...")
    ref_cfg = masked_cfg
    vit_base = load_hooked_vit(ref_cfg, "base", BACKBONE, DEVICE)
    vit_base.eval()

    # ── Build MaPLe model (if requested) ─────────────────────────────
    if args.maple_model_path and args.maple_config_path:
        print("\n[5] Loading MaPLe-adapted CLIP model...")
        maple_classnames = medmnist_classnames  # For MedMNIST eval
        maple_vit = load_hooked_vit(
            masked_cfg, "maple", BACKBONE, DEVICE,
            model_path=args.maple_model_path,
            config_path=args.maple_config_path,
            classnames=maple_classnames,
        )
        maple_vit.eval()

        # MedMNIST eval on MaPLe
        print(f"\n{'═' * 70}")
        print("  SECTION A-MAPLE: MEDMNIST ZERO-SHOT ACCURACY (MaPLe)")
        print(f"{'═' * 70}")
        maple_medmnist_results = {}
        maple_text_feat = get_text_features_hf(maple_vit, DEVICE, maple_classnames, "maple")
        # A1) MaPLe model — no SAE
        print("\n  [A1-M] MaPLe CLIP — no SAE")
        acc = compute_accuracy_hf(maple_vit, maple_text_feat, medmnist_test_loader, "maple", DEVICE)
        maple_medmnist_results["maple_no_sae"] = acc
        print(f"       Accuracy: {acc:.2f}%")
        # A2) MaPLe model + base SAE
        print("\n  [A2-M] MaPLe CLIP + Base SAE")
        hooks = make_sae_hook_hf(base_sae, base_cfg, "maple", sae_bias=args.sae_bias, alpha=args.sae_alpha)
        acc = compute_accuracy_hf(maple_vit, maple_text_feat, medmnist_test_loader, "maple", DEVICE, hooks)
        maple_medmnist_results["maple_base_sae"] = acc
        print(f"       Accuracy: {acc:.2f}%  (Δ {acc - maple_medmnist_results['maple_no_sae']:+.2f})")
        # A3) MaPLe model + masked SAE
        print("\n  [A3-M] MaPLe CLIP + Masked-Finetuned SAE")
        hooks = make_sae_hook_hf(masked_sae, masked_cfg, "maple", sae_bias=args.sae_bias, alpha=args.sae_alpha)
        acc = compute_accuracy_hf(maple_vit, maple_text_feat, medmnist_test_loader, "maple", DEVICE, hooks)
        maple_medmnist_results["maple_masked_sae"] = acc
        print(f"       Accuracy: {acc:.2f}%  (Δ {acc - maple_medmnist_results['maple_no_sae']:+.2f})")
        all_results["medmnist_accuracy_maple"] = maple_medmnist_results

        # MedMNIST summary table (MaPLe)
        print(f"\n  {'─' * 50}")
        print(f"  {'MedMNIST Accuracy Summary (MaPLe)':^50}")
        print(f"  {'─' * 50}")
        print(f"  {'Configuration':<35s}  {'Acc':>7s}")
        print(f"  {'─' * 35}  {'─' * 7}")
        for k, v in maple_medmnist_results.items():
            print(f"  {k:<35s}  {v:>6.2f}%")
        print(f"  {'─' * 50}")

        # ImageNet eval on MaPLe
        if not args.skip_imagenet:
            print(f"\n{'═' * 70}")
            print("  SECTION B-MAPLE: IMAGENET ZERO-SHOT ACCURACY (MaPLe)")
            print(f"{'═' * 70}")
            print("  Loading ImageNet subset for MaPLe...")
            from datasets import load_dataset as hf_load_dataset
            imagenet_ds = hf_load_dataset("evanarlian/imagenet_1k_resized_256", split="train", trust_remote_code=True)
            imagenet_classnames = []
            with open("configs/classnames/imagenet_classnames.txt") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        imagenet_classnames.append(parts[1])
            imgnet_subset_pil = ImageNetSubsetDataset(
                imagenet_ds, n_samples=args.imagenet_n_samples,
                transform=get_processor_transform(), seed=42,
            )
            imgnet_loader_hf = DataLoader(imgnet_subset_pil, batch_size=args.batch_size,
                                           shuffle=False, num_workers=4,
                                           collate_fn=_hf_collate_fn)
            maple_imagenet_results = {}
            maple_text_feat_imgnet = get_text_features_hf(maple_vit, DEVICE, imagenet_classnames, "maple")
            # B1) MaPLe model — no SAE
            print("\n  [B1-M] MaPLe CLIP — no SAE")
            acc = compute_accuracy_hf(maple_vit, maple_text_feat_imgnet, imgnet_loader_hf, "maple", DEVICE)
            maple_imagenet_results["maple_no_sae"] = acc
            print(f"       Accuracy: {acc:.2f}%")
            # B2) MaPLe model + base SAE
            print("\n  [B2-M] MaPLe CLIP + Base SAE")
            hooks = make_sae_hook_hf(base_sae, base_cfg, "maple", sae_bias=args.sae_bias, alpha=args.sae_alpha)
            acc = compute_accuracy_hf(maple_vit, maple_text_feat_imgnet, imgnet_loader_hf, "maple", DEVICE, hooks)
            maple_imagenet_results["maple_base_sae"] = acc
            print(f"       Accuracy: {acc:.2f}%  (Δ {acc - maple_imagenet_results['maple_no_sae']:+.2f})")
            # B3) MaPLe model + masked SAE
            print("\n  [B3-M] MaPLe CLIP + Masked-Finetuned SAE")
            hooks = make_sae_hook_hf(masked_sae, masked_cfg, "maple", sae_bias=args.sae_bias, alpha=args.sae_alpha)
            acc = compute_accuracy_hf(maple_vit, maple_text_feat_imgnet, imgnet_loader_hf, "maple", DEVICE, hooks)
            maple_imagenet_results["maple_masked_sae"] = acc
            print(f"       Accuracy: {acc:.2f}%  (Δ {acc - maple_imagenet_results['maple_no_sae']:+.2f})")
            all_results["imagenet_accuracy_maple"] = maple_imagenet_results

            # ImageNet summary table (MaPLe)
            print(f"\n  {'─' * 50}")
            print(f"  {'ImageNet Accuracy Summary (MaPLe)':^50}")
            print(f"  {'─' * 50}")
            print(f"  {'Configuration':<35s}  {'Acc':>7s}")
            print(f"  {'─' * 35}  {'─' * 7}")
            for k, v in maple_imagenet_results.items():
                print(f"  {k:<35s}  {v:>6.2f}%")
            print(f"  {'─' * 50}")

    # ══════════════════════════════════════════════════════════════
    # SECTION A: MedMNIST Zero-shot Accuracy
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  SECTION A: MEDMNIST ZERO-SHOT ACCURACY")
    print(f"{'═' * 70}")

    medmnist_results = {}

    # A1) LoRA model — no SAE
    print("\n  [A1] LoRA CLIP — no SAE")
    lora_text_feat = get_text_features_openai(lora_model, medmnist_classnames, DEVICE)
    acc = compute_accuracy_openai(lora_model, lora_text_feat, medmnist_test_loader_oai, DEVICE)
    medmnist_results["lora_no_sae"] = acc
    print(f"       Accuracy: {acc:.2f}%")

    # A2) LoRA model + base SAE
    print("\n  [A2] LoRA CLIP + Base SAE")
    hook = OpenAISAEHook(lora_model, base_sae, base_cfg, DEVICE,
                         sae_bias=args.sae_bias, alpha=args.sae_alpha).register()
    acc = compute_accuracy_openai(lora_model, lora_text_feat, medmnist_test_loader_oai, DEVICE)
    hook.remove()
    medmnist_results["lora_base_sae"] = acc
    print(f"       Accuracy: {acc:.2f}%  (Δ {acc - medmnist_results['lora_no_sae']:+.2f})")

    # A3) LoRA model + masked SAE
    print("\n  [A3] LoRA CLIP + Masked-Finetuned SAE")
    hook = OpenAISAEHook(lora_model, masked_sae, masked_cfg, DEVICE,
                         sae_bias=args.sae_bias, alpha=args.sae_alpha).register()
    acc = compute_accuracy_openai(lora_model, lora_text_feat, medmnist_test_loader_oai, DEVICE)
    hook.remove()
    medmnist_results["lora_masked_sae"] = acc
    print(f"       Accuracy: {acc:.2f}%  (Δ {acc - medmnist_results['lora_no_sae']:+.2f})")

    # A4) Base HF model — no SAE
    print("\n  [A4] Base CLIP (HF) — no SAE")
    base_text_feat = get_text_features_hf(vit_base, DEVICE, medmnist_classnames, "base")
    acc = compute_accuracy_hf(vit_base, base_text_feat, medmnist_test_loader, "base", DEVICE)
    medmnist_results["base_no_sae"] = acc
    print(f"       Accuracy: {acc:.2f}%")

    # A5) Base HF model + base SAE
    print("\n  [A5] Base CLIP (HF) + Base SAE")
    hooks = make_sae_hook_hf(base_sae, base_cfg, "base",
                             sae_bias=args.sae_bias, alpha=args.sae_alpha)
    acc = compute_accuracy_hf(vit_base, base_text_feat, medmnist_test_loader, "base", DEVICE, hooks)
    medmnist_results["base_base_sae"] = acc
    print(f"       Accuracy: {acc:.2f}%  (Δ {acc - medmnist_results['base_no_sae']:+.2f})")

    # A6) Base HF model + masked SAE
    print("\n  [A6] Base CLIP (HF) + Masked-Finetuned SAE")
    hooks = make_sae_hook_hf(masked_sae, masked_cfg, "base",
                             sae_bias=args.sae_bias, alpha=args.sae_alpha)
    acc = compute_accuracy_hf(vit_base, base_text_feat, medmnist_test_loader, "base", DEVICE, hooks)
    medmnist_results["base_masked_sae"] = acc
    print(f"       Accuracy: {acc:.2f}%  (Δ {acc - medmnist_results['base_no_sae']:+.2f})")

    all_results["medmnist_accuracy"] = medmnist_results

    # MedMNIST summary table
    print(f"\n  {'─' * 50}")
    print(f"  {'MedMNIST Accuracy Summary':^50}")
    print(f"  {'─' * 50}")
    print(f"  {'Configuration':<35s}  {'Acc':>7s}")
    print(f"  {'─' * 35}  {'─' * 7}")
    for k, v in medmnist_results.items():
        print(f"  {k:<35s}  {v:>6.2f}%")
    print(f"  {'─' * 50}")

    # ══════════════════════════════════════════════════════════════
    # SECTION B: ImageNet Zero-shot Accuracy (concept preservation)
    # ══════════════════════════════════════════════════════════════
    if not args.skip_imagenet:
        print(f"\n{'═' * 70}")
        print("  SECTION B: IMAGENET ZERO-SHOT ACCURACY (concept preservation)")
        print(f"{'═' * 70}")

        print("  Loading ImageNet subset...")
        from datasets import load_dataset as hf_load_dataset
        imagenet_ds = hf_load_dataset("evanarlian/imagenet_1k_resized_256", split="train",
                                       trust_remote_code=True)

        # Read imagenet classnames
        imagenet_classnames = []
        with open("configs/classnames/imagenet_classnames.txt") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    imagenet_classnames.append(parts[1])
        print(f"  ImageNet: {len(imagenet_classnames)} classes")

        # PIL subset for HF
        imgnet_subset_pil = ImageNetSubsetDataset(
            imagenet_ds, n_samples=args.imagenet_n_samples,
            transform=get_processor_transform(), seed=42,
        )
        imgnet_loader_hf = DataLoader(imgnet_subset_pil, batch_size=args.batch_size,
                                       shuffle=False, num_workers=4,
                                       collate_fn=_hf_collate_fn)

        # Tensor subset for OpenAI
        imgnet_subset_oai = ImageNetSubsetDataset(
            imagenet_ds, n_samples=args.imagenet_n_samples,
            transform=get_transform(), seed=42,
        )
        imgnet_loader_oai = DataLoader(imgnet_subset_oai, batch_size=args.batch_size,
                                        shuffle=False, num_workers=4, pin_memory=True)

        imagenet_results = {}

        # B1) Base HF — no SAE
        print("\n  [B1] Base CLIP (HF) — no SAE")
        imgnet_text_feat_hf = get_text_features_hf(vit_base, DEVICE, imagenet_classnames, "base")
        acc = compute_accuracy_hf(vit_base, imgnet_text_feat_hf, imgnet_loader_hf, "base", DEVICE)
        imagenet_results["base_no_sae"] = acc
        print(f"       Accuracy: {acc:.2f}%")

        # B2) Base HF + base SAE
        print("\n  [B2] Base CLIP (HF) + Base SAE")
        hooks = make_sae_hook_hf(base_sae, base_cfg, "base",
                                 sae_bias=args.sae_bias, alpha=args.sae_alpha)
        acc = compute_accuracy_hf(vit_base, imgnet_text_feat_hf, imgnet_loader_hf, "base", DEVICE, hooks)
        imagenet_results["base_base_sae"] = acc
        print(f"       Accuracy: {acc:.2f}%  (Δ {acc - imagenet_results['base_no_sae']:+.2f})")

        # B3) Base HF + masked SAE
        print("\n  [B3] Base CLIP (HF) + Masked-Finetuned SAE")
        hooks = make_sae_hook_hf(masked_sae, masked_cfg, "base",
                                 sae_bias=args.sae_bias, alpha=args.sae_alpha)
        acc = compute_accuracy_hf(vit_base, imgnet_text_feat_hf, imgnet_loader_hf, "base", DEVICE, hooks)
        imagenet_results["base_masked_sae"] = acc
        print(f"       Accuracy: {acc:.2f}%  (Δ {acc - imagenet_results['base_no_sae']:+.2f})")

        # B4) LoRA OpenAI — no SAE
        print("\n  [B4] LoRA CLIP — no SAE")
        imgnet_text_feat_lora = get_text_features_openai(lora_model, imagenet_classnames, DEVICE)
        acc = compute_accuracy_openai(lora_model, imgnet_text_feat_lora, imgnet_loader_oai, DEVICE)
        imagenet_results["lora_no_sae"] = acc
        print(f"       Accuracy: {acc:.2f}%")

        # B5) LoRA + base SAE
        print("\n  [B5] LoRA CLIP + Base SAE")
        hook = OpenAISAEHook(lora_model, base_sae, base_cfg, DEVICE,
                             sae_bias=args.sae_bias, alpha=args.sae_alpha).register()
        acc = compute_accuracy_openai(lora_model, imgnet_text_feat_lora, imgnet_loader_oai, DEVICE)
        hook.remove()
        imagenet_results["lora_base_sae"] = acc
        print(f"       Accuracy: {acc:.2f}%  (Δ {acc - imagenet_results['lora_no_sae']:+.2f})")

        # B6) LoRA + masked SAE
        print("\n  [B6] LoRA CLIP + Masked-Finetuned SAE")
        hook = OpenAISAEHook(lora_model, masked_sae, masked_cfg, DEVICE,
                             sae_bias=args.sae_bias, alpha=args.sae_alpha).register()
        acc = compute_accuracy_openai(lora_model, imgnet_text_feat_lora, imgnet_loader_oai, DEVICE)
        hook.remove()
        imagenet_results["lora_masked_sae"] = acc
        print(f"       Accuracy: {acc:.2f}%  (Δ {acc - imagenet_results['lora_no_sae']:+.2f})")

        all_results["imagenet_accuracy"] = imagenet_results

        # ImageNet summary table
        print(f"\n  {'─' * 50}")
        print(f"  {'ImageNet Accuracy Summary':^50}")
        print(f"  {'─' * 50}")
        print(f"  {'Configuration':<35s}  {'Acc':>7s}")
        print(f"  {'─' * 35}  {'─' * 7}")
        for k, v in imagenet_results.items():
            print(f"  {k:<35s}  {v:>6.2f}%")
        print(f"  {'─' * 50}")

        del imagenet_ds, imgnet_subset_pil, imgnet_subset_oai
        flush()

    # ══════════════════════════════════════════════════════════════
    # SECTION C: Feature Analysis (protected vs free)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  SECTION C: FEATURE ANALYSIS (Protected vs Free units)")
    print(f"{'═' * 70}")

    # Analyze on MedMNIST (via LoRA model)
    print("\n  [C1] Masked SAE feature activity on MedMNIST (LoRA model)")
    medmnist_feat_lora = analyze_sae_features(
        masked_sae, medmnist_test_loader_oai, protected_mask,
        n_batches=args.feature_analysis_batches, device=DEVICE,
        use_loader=True, cfg=masked_cfg, openai_model=lora_model,
    )

    print("\n  [C2] Masked SAE feature activity on MedMNIST (Base HF model)")
    medmnist_feat = analyze_sae_features(
        masked_sae, medmnist_test_loader, protected_mask,
        n_batches=min(args.feature_analysis_batches, len(medmnist_test_loader)),
        device=DEVICE, use_loader=True, vit=vit_base, cfg=masked_cfg, vit_type="base",
    )

    print("\n  [C3] Base SAE feature activity on MedMNIST (Base HF model)")
    base_feat = analyze_sae_features(
        base_sae, medmnist_test_loader, protected_mask,
        n_batches=min(args.feature_analysis_batches, len(medmnist_test_loader)),
        device=DEVICE, use_loader=True, vit=vit_base, cfg=base_cfg, vit_type="base",
    )

    feature_analysis = {
        "masked_sae_on_medmnist_lora": medmnist_feat_lora,
        "masked_sae_on_medmnist": medmnist_feat,
        "base_sae_on_medmnist": base_feat,
    }
    all_results["feature_analysis"] = feature_analysis

    print(f"\n  {'─' * 65}")
    print(f"  {'Feature Analysis Summary':^65}")
    print(f"  {'─' * 65}")
    print(f"  {'Group':<12s} {'SAE':<12s} {'N':>6s} {'MeanAct':>9s} {'FireRate':>9s} {'Dead%':>7s} {'L0':>7s}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 6} {'─' * 9} {'─' * 9} {'─' * 7} {'─' * 7}")
    for sae_name, feat_data in [("masked(lora)", medmnist_feat_lora), ("masked(base)", medmnist_feat), ("base", base_feat)]:
        for group in ["protected", "free", "all"]:
            d = feat_data.get(group, {})
            if d.get("n", 0) == 0:
                continue
            print(f"  {group:<12s} {sae_name:<12s} {d['n']:>6d} "
                  f"{d['mean_activity']:>9.6f} {d['fire_rate_mean']:>9.6f} "
                  f"{d['dead_fraction'] * 100:>6.1f}% {d['l0']:>7.1f}")
    print(f"  {'─' * 65}")

    # ══════════════════════════════════════════════════════════════
    # SECTION D: Reconstruction Quality
    # ══════════════════════════════════════════════════════════════
    if not args.skip_reconstruction:
        print(f"\n{'═' * 70}")
        print("  SECTION D: RECONSTRUCTION QUALITY")
        print(f"{'═' * 70}")

        recon_results = {}

        print("\n  [D1] Masked SAE reconstruction on MedMNIST")
        r = measure_reconstruction_quality(
            masked_sae, medmnist_test_loader, n_batches=30, device=DEVICE,
            use_loader=True, vit=vit_base, cfg=masked_cfg,
        )
        recon_results["masked_sae_medmnist"] = r
        print(f"       MSE={r['mse']:.6f}, cos_sim={r['cos_sim']:.6f}")

        print("\n  [D2] Base SAE reconstruction on MedMNIST")
        r = measure_reconstruction_quality(
            base_sae, medmnist_test_loader, n_batches=30, device=DEVICE,
            use_loader=True, vit=vit_base, cfg=base_cfg,
        )
        recon_results["base_sae_medmnist"] = r
        print(f"       MSE={r['mse']:.6f}, cos_sim={r['cos_sim']:.6f}")

        if not args.skip_imagenet:
            print("\n  [D3] Masked SAE reconstruction on ImageNet")
            r = measure_reconstruction_quality(
                masked_sae, imgnet_loader_hf, n_batches=30, device=DEVICE,
                use_loader=True, vit=vit_base, cfg=masked_cfg,
            )
            recon_results["masked_sae_imagenet"] = r
            print(f"       MSE={r['mse']:.6f}, cos_sim={r['cos_sim']:.6f}")

            print("\n  [D4] Base SAE reconstruction on ImageNet")
            r = measure_reconstruction_quality(
                base_sae, imgnet_loader_hf, n_batches=30, device=DEVICE,
                use_loader=True, vit=vit_base, cfg=base_cfg,
            )
            recon_results["base_sae_imagenet"] = r
            print(f"       MSE={r['mse']:.6f}, cos_sim={r['cos_sim']:.6f}")

        all_results["reconstruction"] = recon_results

        print(f"\n  {'─' * 55}")
        print(f"  {'Reconstruction Summary':^55}")
        print(f"  {'─' * 55}")
        print(f"  {'Config':<30s}  {'MSE':>10s}  {'Cos Sim':>10s}")
        print(f"  {'─' * 30}  {'─' * 10}  {'─' * 10}")
        for k, v in recon_results.items():
            print(f"  {k:<30s}  {v['mse']:>10.6f}  {v['cos_sim']:>10.6f}")
        print(f"  {'─' * 55}")

    # ══════════════════════════════════════════════════════════════
    # SECTION E: Linear Probe on SAE Latents
    # ══════════════════════════════════════════════════════════════
    if not args.skip_linear_probe:
        print(f"\n{'═' * 70}")
        print("  SECTION E: LINEAR PROBE ON SAE LATENTS (MedMNIST)")
        print(f"{'═' * 70}")

        probe_results = {}

        # Train/test loaders for MedMNIST
        medmnist_train_ds_hf = NpzSplitDataset(NPZ_PATH, label_mapping, hf_transform,
                                                split="train", return_pil=True)
        medmnist_train_loader_hf = DataLoader(medmnist_train_ds_hf, batch_size=args.batch_size,
                                               shuffle=False, num_workers=4,
                                               collate_fn=_hf_collate_fn)

        medmnist_train_ds_oai = NpzSplitDataset(NPZ_PATH, label_mapping, lora_preprocess,
                                                 split="train", return_pil=False)
        medmnist_train_loader_oai = DataLoader(medmnist_train_ds_oai, batch_size=args.batch_size,
                                                shuffle=False, num_workers=4, pin_memory=True)

        # E1) Base SAE → base model
        print("\n  [E1] Base SAE latents → Base CLIP")
        train_lat, train_lab = extract_sae_latents_hf(
            vit_base, base_sae, base_cfg, medmnist_train_loader_hf, "base", DEVICE)
        test_lat, test_lab = extract_sae_latents_hf(
            vit_base, base_sae, base_cfg, medmnist_test_loader, "base", DEVICE)
        test_acc, train_acc = train_linear_probe(train_lat, train_lab, test_lat, test_lab)
        probe_results["base_sae_base_model"] = {"test": test_acc, "train": train_acc}
        print(f"       Train: {train_acc:.2f}%  Test: {test_acc:.2f}%")

        # E2) Masked SAE → base model
        print("\n  [E2] Masked SAE latents → Base CLIP")
        train_lat, train_lab = extract_sae_latents_hf(
            vit_base, masked_sae, masked_cfg, medmnist_train_loader_hf, "base", DEVICE)
        test_lat, test_lab = extract_sae_latents_hf(
            vit_base, masked_sae, masked_cfg, medmnist_test_loader, "base", DEVICE)
        test_acc, train_acc = train_linear_probe(train_lat, train_lab, test_lat, test_lab)
        probe_results["masked_sae_base_model"] = {"test": test_acc, "train": train_acc}
        print(f"       Train: {train_acc:.2f}%  Test: {test_acc:.2f}%")

        # E3) Base SAE → LoRA model
        print("\n  [E3] Base SAE latents → LoRA CLIP")
        train_lat, train_lab = extract_sae_latents_openai(
            lora_model, base_sae, base_cfg, medmnist_train_loader_oai, DEVICE)
        test_lat, test_lab = extract_sae_latents_openai(
            lora_model, base_sae, base_cfg, medmnist_test_loader_oai, DEVICE)
        test_acc, train_acc = train_linear_probe(train_lat, train_lab, test_lat, test_lab)
        probe_results["base_sae_lora_model"] = {"test": test_acc, "train": train_acc}
        print(f"       Train: {train_acc:.2f}%  Test: {test_acc:.2f}%")

        # E4) Masked SAE → LoRA model
        print("\n  [E4] Masked SAE latents → LoRA CLIP")
        train_lat, train_lab = extract_sae_latents_openai(
            lora_model, masked_sae, masked_cfg, medmnist_train_loader_oai, DEVICE)
        test_lat, test_lab = extract_sae_latents_openai(
            lora_model, masked_sae, masked_cfg, medmnist_test_loader_oai, DEVICE)
        test_acc, train_acc = train_linear_probe(train_lat, train_lab, test_lat, test_lab)
        probe_results["masked_sae_lora_model"] = {"test": test_acc, "train": train_acc}
        print(f"       Train: {train_acc:.2f}%  Test: {test_acc:.2f}%")

        all_results["linear_probe"] = probe_results

        print(f"\n  {'─' * 55}")
        print(f"  {'Linear Probe Summary (MedMNIST)':^55}")
        print(f"  {'─' * 55}")
        print(f"  {'Configuration':<30s}  {'Train':>8s}  {'Test':>8s}")
        print(f"  {'─' * 30}  {'─' * 8}  {'─' * 8}")
        for k, v in probe_results.items():
            print(f"  {k:<30s}  {v['train']:>7.2f}%  {v['test']:>7.2f}%")
        print(f"  {'─' * 55}")

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  FINAL SUMMARY: Masked SAE vs Base SAE")
    print(f"{'═' * 70}")

    med = all_results.get("medmnist_accuracy", {})
    img = all_results.get("imagenet_accuracy", {})

    print(f"\n  {'Metric':<40s}  {'Base SAE':>10s}  {'Masked SAE':>11s}  {'Delta':>8s}")
    print(f"  {'─' * 40}  {'─' * 10}  {'─' * 11}  {'─' * 8}")

    # MedMNIST on LoRA
    if "lora_base_sae" in med and "lora_masked_sae" in med:
        b, m = med["lora_base_sae"], med["lora_masked_sae"]
        print(f"  {'MedMNIST acc (LoRA model)':<40s}  {b:>9.2f}%  {m:>10.2f}%  {m - b:>+7.2f}%")

    # MedMNIST on base
    if "base_base_sae" in med and "base_masked_sae" in med:
        b, m = med["base_base_sae"], med["base_masked_sae"]
        print(f"  {'MedMNIST acc (Base model)':<40s}  {b:>9.2f}%  {m:>10.2f}%  {m - b:>+7.2f}%")

    # ImageNet on base
    if "base_base_sae" in img and "base_masked_sae" in img:
        b, m = img["base_base_sae"], img["base_masked_sae"]
        print(f"  {'ImageNet acc (Base model)':<40s}  {b:>9.2f}%  {m:>10.2f}%  {m - b:>+7.2f}%")

    # ImageNet on LoRA
    if "lora_base_sae" in img and "lora_masked_sae" in img:
        b, m = img["lora_base_sae"], img["lora_masked_sae"]
        print(f"  {'ImageNet acc (LoRA model)':<40s}  {b:>9.2f}%  {m:>10.2f}%  {m - b:>+7.2f}%")

    # Reconstruction
    recon = all_results.get("reconstruction", {})
    if "base_sae_medmnist" in recon and "masked_sae_medmnist" in recon:
        b, m = recon["base_sae_medmnist"], recon["masked_sae_medmnist"]
        print(f"  {'MedMNIST recon MSE':<40s}  {b['mse']:>10.6f}  {m['mse']:>11.6f}  {m['mse'] - b['mse']:>+8.6f}")
        print(f"  {'MedMNIST recon cos_sim':<40s}  {b['cos_sim']:>10.6f}  {m['cos_sim']:>11.6f}  {m['cos_sim'] - b['cos_sim']:>+8.6f}")

    # Linear probe
    probe = all_results.get("linear_probe", {})
    if "base_sae_lora_model" in probe and "masked_sae_lora_model" in probe:
        b, m = probe["base_sae_lora_model"]["test"], probe["masked_sae_lora_model"]["test"]
        print(f"  {'Linear probe test (LoRA)':<40s}  {b:>9.2f}%  {m:>10.2f}%  {m - b:>+7.2f}%")

    print(f"{'═' * 70}")

    # Save results
    save_path = os.path.join(args.save_dir, "eval_masked_sae_results.json")

    # Convert numpy/torch types to plain Python for JSON
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, torch.Tensor)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        return obj

    with open(save_path, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to: {save_path}")

    # Cleanup
    del lora_model, vit_base
    flush()


if __name__ == "__main__":
    main()
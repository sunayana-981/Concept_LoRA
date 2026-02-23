#!/usr/bin/env python3
"""
Evaluate SAE zero-shot classification on MedMNIST test set.

Supports three ViT backends:
  --vit_type base   → regular CLIP (default, already benchmarked)
  --vit_type lora   → LoRA fine-tuned CLIP
  --vit_type maple  → MaPLe fine-tuned CLIP

Same top-k masking protocol as classify.py for each.

Usage:
    python eval_medmnist_sae.py --vit_type base  --include_base
    python eval_medmnist_sae.py --vit_type lora  --include_base
    python eval_medmnist_sae.py --vit_type maple --include_base --model_path ... --config_path ...
    python eval_medmnist_sae.py --vit_type lora  --reuse_activations
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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

try:
    from tasks.utils import load_sae, load_hooked_vit, SAE_DIM
    from src.sae_training.hooked_vit import Hook
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] {e}\nRun from patchsae root.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

BASE_SAE_PATH    = "data/sae_weight/base/out.pt"
MEDMNIST_SAE_DIR = "out/checkpoints/medmnist"
LORA_CHECKPOINT  = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
MAPLE_MODEL_PATH = None   # set via --model_path
MAPLE_CONFIG_PATH = None  # set via --config_path
TRAIN_ROOT = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
NPZ_PATH   = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
CLASSNAMES_PATH = "configs/classnames/medmnist_classnames.json"

# MedMNIST canonical class ordering (matches npz label indices)
MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

BACKBONE   = "openai/clip-vit-base-patch16"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
SEED       = 42

TOPK_LIST  = [1, 2, 5, 10, 50, 100, 500, 1000, 2000]
SAE_BIAS   = -0.105131256516992


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_transform():
    """Full CLIP preprocessing (resize + normalize).  Use for tensors fed directly to model."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_processor_transform():
    """Minimal transform for images that will go through CLIPProcessor.

    Only resizes + converts to tensor in [0,1].  Do NOT normalize here —
    the CLIPProcessor handles normalisation.  This avoids the
    double-normalisation / PIL-round-trip clipping bug.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def get_imagefolder_classnames(root):
    """Return class names in ImageFolder alphabetical order."""
    ds = datasets.ImageFolder(root=root)
    idx_to_name = {i: n.replace('_', ' ') for n, i in ds.class_to_idx.items()}
    return [idx_to_name[i] for i in range(len(idx_to_name))]


def build_npz_to_imagefolder_mapping(imagefolder_root):
    """Map npz label indices -> ImageFolder label indices (alphabetical folder order)."""
    ds = datasets.ImageFolder(root=imagefolder_root)
    if_name_to_idx = {}
    for name, idx in ds.class_to_idx.items():
        if_name_to_idx[name.replace('_', ' ').lower()] = idx
    mapping = {}
    for npz_idx, npz_name in enumerate(MEDMNIST_CLASSES):
        key = npz_name.lower()
        if key in if_name_to_idx:
            mapping[npz_idx] = if_name_to_idx[key]
        else:
            print(f"[WARN] Cannot map npz class {npz_idx} ('{npz_name}')")
            mapping[npz_idx] = npz_idx
    return mapping


class NpzTestDataset(Dataset):
    """Load test images from pathmnist.npz with label remapping."""
    def __init__(self, npz_path, label_mapping, transform):
        data = np.load(npz_path)
        self.images = data['test_images']          # (N, 28, 28, 3)
        self.labels = data['test_labels'].flatten() # (N,)
        self.label_mapping = label_mapping
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        img = self.transform(img)
        label = self.label_mapping[int(self.labels[idx])]
        return img, label


def discover_medmnist_saes(base_dir):
    pattern = os.path.join(base_dir, "*/final*/*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[WARN] No checkpoints matching {pattern}")
        return []
    best = {}
    for p in paths:
        ckpt = torch.load(p, map_location="cpu")
        cfg_d = ckpt.get("cfg", ckpt.get("config"))
        layer = getattr(cfg_d, "block_layer",
                        cfg_d.get("block_layer", "?") if isinstance(cfg_d, dict) else "?")
        best[layer] = p
        del ckpt; gc.collect()
    return list(best.values())

def load_selected_saes(base_path, med_paths, device, include_base=False):
    saes = []
    if include_base and os.path.exists(base_path):
        sae, cfg = load_sae(base_path, device)
        layer = cfg.block_layer if hasattr(cfg, "block_layer") else -2
        print(f"  [OK] Base SAE: layer={layer}, d_sae={cfg.d_sae}")
        saes.append((sae, cfg, f"base_layer{layer}"))
    elif include_base:
        print(f"  [SKIP] Base SAE not found: {base_path}")
    for p in med_paths:
        sae, cfg = load_sae(p, device)
        layer = cfg.block_layer if hasattr(cfg, "block_layer") else "?"
        print(f"  [OK] MedMNIST SAE: layer={layer}, d_sae={cfg.d_sae}")
        print(f"        path: {p}")
        saes.append((sae, cfg, f"medmnist_layer{layer}"))
    return saes


# ═══════════════════════════════════════════════════════════════════════════
# LORA MERGE  (matching merge_lora_into_clip.py logic)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_lora_AB(lora_dict, proj_name):
    """
    Extract LoRA A and B matrices for a given projection,
    handling both nested and flat dict structures.
    Returns (A, B) or (None, None) if not found.
    """
    # Nested: lora_dict[proj_name]["w_lora_A"]
    if proj_name in lora_dict and isinstance(lora_dict[proj_name], dict):
        try:
            return lora_dict[proj_name]["w_lora_A"], lora_dict[proj_name]["w_lora_B"]
        except KeyError:
            pass

    # Flat: lora_dict["proj_name.w_lora_A"]
    try:
        return lora_dict[f"{proj_name}.w_lora_A"], lora_dict[f"{proj_name}.w_lora_B"]
    except KeyError:
        return None, None


def _apply_lora_to_combined_qkv(block, lora_dict, scale, d_model, device):
    """
    Apply LoRA to a block that uses a combined in_proj_weight for Q, K, V
    (OpenAI CLIP style: transformer.resblocks[i].attn.in_proj_weight).
    """
    w = block.attn.in_proj_weight.data

    for proj_name, start in [("q_proj", 0), ("k_proj", d_model), ("v_proj", 2 * d_model)]:
        A, B = _extract_lora_AB(lora_dict, proj_name)
        if A is None:
            continue
        A, B = A.to(device), B.to(device)
        delta_w = scale * (B @ A)
        w[start:start + d_model, :] += delta_w


def _apply_lora_to_separate_qkv(self_attn, lora_dict, scale, device):
    """
    Apply LoRA to a block that uses separate q_proj, k_proj, v_proj layers
    (HuggingFace CLIP style: encoder.layers[i].self_attn.{q,k,v}_proj.weight).
    """
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        A, B = _extract_lora_AB(lora_dict, proj_name)
        if A is None:
            continue
        A, B = A.to(device), B.to(device)
        delta_w = scale * (B @ A)
        proj_layer = getattr(self_attn, proj_name, None)
        if proj_layer is not None:
            proj_layer.weight.data += delta_w


def apply_lora_to_hooked_vit(vit, lora_path, device):
    """
    Load LoRA weights and merge them into the hooked ViT model.

    Supports TWO checkpoint formats:

    1. Raw LoRA weights (from training):
       - Has keys "weights" and "metadata"
       - Merges delta_w = scale * (B @ A) into the base model

    2. Pre-merged state dict (from merge_lora_into_clip.py):
       - A plain state_dict produced by torch.save(model.state_dict(), ...)
       - Loaded directly into the underlying model

    For format 1, handles both:
      - OpenAI CLIP (combined in_proj_weight)
      - HuggingFace CLIP (separate q_proj/k_proj/v_proj)
    """
    print(f"  Loading LoRA state from: {lora_path}")
    lora_state = torch.load(lora_path, map_location=device)

    # ── Detect format ──────────────────────────────────────────────
    is_raw_lora = "weights" in lora_state and "metadata" in lora_state

    if not is_raw_lora:
        # Format 2: pre-merged full state dict → load directly
        print("  Detected pre-merged state dict, loading into model...")
        model = vit.model

        # The hooked_vit wraps HF CLIPModel, but the merged checkpoint
        # may come from OpenAI clip.load(). Try loading as-is first;
        # if that fails, try wrapping with a prefix or partial load.
        is_hf = hasattr(model, "text_model")

        if is_hf:
            # HuggingFace CLIPModel — check if keys already match
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(lora_state.keys())

            if model_keys & ckpt_keys:
                # Keys overlap → direct load (strict=False to tolerate minor mismatches)
                info = model.load_state_dict(lora_state, strict=False)
                n_loaded = len(model_keys) - len(info.missing_keys)
                print(f"  Loaded {n_loaded}/{len(model_keys)} params "
                      f"(missing={len(info.missing_keys)}, "
                      f"unexpected={len(info.unexpected_keys)})")
            else:
                # Keys don't overlap — likely OpenAI-format state dict
                # into HF model; need key remapping
                print("  Key mismatch detected — attempting OpenAI→HF remap...")
                _load_openai_state_into_hf(model, lora_state, device)
        else:
            # OpenAI-style model — direct load
            info = model.load_state_dict(lora_state, strict=False)
            print(f"  Loaded (missing={len(info.missing_keys)}, "
                  f"unexpected={len(info.unexpected_keys)})")

        print("  Pre-merged weights loaded.")
        return

    # ── Format 1: raw LoRA weights → merge ─────────────────────────
    layers = lora_state["weights"]
    meta = lora_state["metadata"]

    r = meta["r"]
    alpha = meta["alpha"]
    scale = alpha / r
    print(f"  LoRA params: rank={r}, alpha={alpha}, scale={scale}")
    print(f"  Total LoRA layer entries: {len(layers)}")

    model = vit.model  # underlying CLIP model

    with torch.no_grad():
        is_hf = hasattr(model, "text_model")

        if is_hf:
            text_blocks = model.text_model.encoder.layers
            vision_blocks = model.vision_model.encoder.layers
        else:
            text_blocks = model.transformer.resblocks
            vision_blocks = model.visual.transformer.resblocks

        num_text = len(text_blocks)
        num_vision = len(vision_blocks)

        # ── Text Encoder (layers 0 .. num_text-1) ─────────────────
        print(f"  Merging text encoder ({num_text} layers)...")
        for i in range(num_text):
            layer_key = f"layer_{i}"
            if layer_key not in layers:
                continue
            lora_dict = layers[layer_key]
            block = text_blocks[i]

            if is_hf:
                _apply_lora_to_separate_qkv(block.self_attn, lora_dict, scale, device)
            else:
                d_model = block.attn.in_proj_weight.shape[1]
                _apply_lora_to_combined_qkv(block, lora_dict, scale, d_model, device)
            print(f"    ✓ Text layer {i}")

        # ── Vision Encoder (layers num_text .. num_text+num_vision-1) ──
        print(f"  Merging vision encoder ({num_vision} layers)...")
        for i in range(num_text, num_text + num_vision):
            layer_key = f"layer_{i}"
            if layer_key not in layers:
                continue
            lora_dict = layers[layer_key]
            vision_idx = i - num_text
            block = vision_blocks[vision_idx]

            if is_hf:
                _apply_lora_to_separate_qkv(block.self_attn, lora_dict, scale, device)
            else:
                d_model = block.attn.in_proj_weight.shape[1]
                _apply_lora_to_combined_qkv(block, lora_dict, scale, d_model, device)
            print(f"    ✓ Vision layer {vision_idx}")

    print("  LoRA merge complete.")


def _load_openai_state_into_hf(hf_model, openai_sd, device):
    """
    Best-effort load of an OpenAI CLIP state dict into a HuggingFace
    CLIPModel. Handles the common case where the merge script saved
    weights in OpenAI format but the hooked_vit uses HF architecture.

    Strategy: manually map the vision encoder attention weights
    (in_proj_weight → q_proj/k_proj/v_proj) and load remaining keys
    by suffix matching.
    """
    hf_sd = hf_model.state_dict()
    new_sd = {}
    matched = 0

    # Build a suffix → hf_key lookup for non-attention params
    hf_suffix_map = {}
    for k in hf_sd:
        # Use last 3 dot-separated components as suffix key
        parts = k.split(".")
        suffix = ".".join(parts[-3:]) if len(parts) >= 3 else k
        hf_suffix_map[suffix] = k

    for oai_key, oai_val in openai_sd.items():
        # Try direct match first
        if oai_key in hf_sd:
            new_sd[oai_key] = oai_val
            matched += 1
            continue

        # Try suffix match
        parts = oai_key.split(".")
        suffix = ".".join(parts[-3:]) if len(parts) >= 3 else oai_key
        if suffix in hf_suffix_map:
            hf_key = hf_suffix_map[suffix]
            if hf_sd[hf_key].shape == oai_val.shape:
                new_sd[hf_key] = oai_val
                matched += 1
                continue

        # Handle combined in_proj_weight → split q/k/v
        if "in_proj_weight" in oai_key:
            d = oai_val.shape[0] // 3
            # Try to find q_proj in the same layer path
            base = oai_key.replace("attn.in_proj_weight", "")
            for proj, start in [("q_proj.weight", 0), ("k_proj.weight", d), ("v_proj.weight", 2*d)]:
                for hf_key in hf_sd:
                    if proj in hf_key and hf_sd[hf_key].shape == (d, oai_val.shape[1]):
                        new_sd[hf_key] = oai_val[start:start+d]
                        matched += 1
                        break

        if "in_proj_bias" in oai_key:
            d = oai_val.shape[0] // 3
            for proj, start in [("q_proj.bias", 0), ("k_proj.bias", d), ("v_proj.bias", 2*d)]:
                for hf_key in hf_sd:
                    if proj in hf_key and hf_sd[hf_key].shape == (d,):
                        new_sd[hf_key] = oai_val[start:start+d]
                        matched += 1
                        break

    info = hf_model.load_state_dict(new_sd, strict=False)
    print(f"  OpenAI→HF remap: {matched} tensors mapped, "
          f"missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")


# ═══════════════════════════════════════════════════════════════════════════
# VIT LOADING  (base / lora / maple)
# ═══════════════════════════════════════════════════════════════════════════

def load_vit_for_type(vit_type, ref_cfg, backbone, device,
                      lora_path=None, model_path=None, config_path=None,
                      classnames=None):
    """
    Load the hooked ViT for the given type.
      base  → vanilla CLIP
      lora  → CLIP + LoRA weight merge
      maple → MaPLe prompted CLIP (needs model_path + config_path)
    """
    if vit_type == "maple":
        if not model_path or not config_path:
            print("[FATAL] --model_path and --config_path required for maple.")
            sys.exit(1)
        vit = load_hooked_vit(ref_cfg, "maple", backbone, device,
                              model_path=model_path, config_path=config_path,
                              classnames=classnames)
        print(f"  MaPLe ViT loaded (model={model_path})")

    elif vit_type == "lora":
        vit = load_hooked_vit(ref_cfg, "base", backbone, device)
        print("  Base ViT loaded, applying LoRA weights...")
        apply_lora_to_hooked_vit(vit, lora_path, device)

    else:  # base
        vit = load_hooked_vit(ref_cfg, "base", backbone, device)
        print("  Base CLIP ViT loaded.")

    vit.eval()
    return vit


# ═══════════════════════════════════════════════════════════════════════════
# TEXT FEATURES  (differs for base/lora vs maple)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def calculate_text_features(vit, device, classnames, vit_type):
    """
    base/lora → mean over 80 OpenAI ImageNet prompt templates
    maple     → vit.model.get_text_features() (prompts baked into model)
    """
    if vit_type == "maple":
        text_features = vit.model.get_text_features()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    # base / lora
    mean_feats = 0
    for template_fn in openai_imagenet_template:
        prompts = [template_fn(c) for c in classnames]
        ids = [vit.processor(text=p, return_tensors="pt", padding=False,
                             truncation=True).input_ids[0] for p in prompts]
        padded = pad_sequence(ids, batch_first=True).to(device)
        feats = vit.model.get_text_features(padded)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        mean_feats += feats
    mean_feats = mean_feats / len(openai_imagenet_template)
    return mean_feats / mean_feats.norm(dim=-1, keepdim=True)


# ═══════════════════════════════════════════════════════════════════════════
# PER-CLASS ACTIVATION PROFILES  (train set)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_cls_sae_activations(vit, sae, cfg, loader, num_classes, device):
    d_sae = cfg.d_sae
    layer = cfg.block_layer if hasattr(cfg, "block_layer") else -2
    module = cfg.module_name if hasattr(cfg, "module_name") else "resid"

    cls_sum = np.zeros((num_classes, d_sae), dtype=np.float64)
    cls_n   = np.zeros(num_classes, dtype=np.int64)
    to_pil  = transforms.ToPILImage()

    for images, labels in tqdm(loader, desc="  cls_sae_cnt (train)"):
        inputs = vit.processor(images=[to_pil(img) for img in images],
                               text="", return_tensors="pt", padding=True).to(device)
        _, cache = vit.run_with_cache([(layer, module)], **inputs)
        cls_acts = cache[(layer, module)][:, 0, :]
        _, sc = sae.run_with_cache(cls_acts)
        feat = sc["hook_hidden_post"].cpu().float().numpy()
        for i in range(images.size(0)):
            c = labels[i].item()
            cls_sum[c] += feat[i]
            cls_n[c] += 1
        del cache, sc; flush()

    for c in range(num_classes):
        if cls_n[c] > 0:
            cls_sum[c] /= cls_n[c]
    return cls_sum


# ═══════════════════════════════════════════════════════════════════════════
# SAE HOOKS  (different for base/lora vs maple — from classify.py)
# ═══════════════════════════════════════════════════════════════════════════

def create_sae_hooks(cfg, cls_features, sae, device, vit_type, hook_type="on"):
    """
    Create hooks that clamp SAE features ON or OFF.
    MaPLe needs transposed activations and is_custom=True.
    """
    d_sae = cfg.d_sae
    clamp_feat_dim = torch.ones(d_sae).bool()
    if hook_type == "on":
        clamp_value = torch.zeros(d_sae, device=device)
        for f in cls_features:
            clamp_value[f] = 1.0
    else:
        clamp_value = torch.ones(d_sae, device=device)
        for f in cls_features:
            clamp_value[f] = 0.0

    if vit_type == "maple":
        def hook_fn(activations):
            # maple passes [seq_len, B, d_model] — transpose to [B, seq_len, d_model]
            orig_dtype = activations.dtype
            act = activations.transpose(0, 1).float()
            processed = (
                sae.forward_clamp(act[:, :, :],
                                  clamp_feat_dim=clamp_feat_dim,
                                  clamp_value=clamp_value)[0]
                - SAE_BIAS
            )
            return processed.transpose(0, 1).to(orig_dtype)

        return [Hook(cfg.block_layer, cfg.module_name, hook_fn,
                     return_module_output=False, is_custom=True)]
    else:
        def hook_fn(activations):
            activations[:, :, :] = (
                sae.forward_clamp(activations[:, :, :],
                                  clamp_feat_dim=clamp_feat_dim,
                                  clamp_value=clamp_value)[0]
                - SAE_BIAS
            )
            return (activations,)

        return [Hook(cfg.block_layer, cfg.module_name, hook_fn,
                     return_module_output=False, is_custom=False)]


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION  (output format differs for base/lora vs maple)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_predictions(vit, inputs, text_features, vit_type, hooks=None):
    """
    base/lora → vit_out.image_embeds
    maple     → vit_out directly (already image features)
    """
    if hooks:
        vit_out = vit.run_with_hooks(hooks, return_type="output", **inputs)
    else:
        vit_out = vit(return_type="output", **inputs)

    if vit_type == "base" or vit_type == "lora":
        image_features = vit_out.image_embeds
    else:  # maple
        image_features = vit_out

    logit_scale = vit.model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.t()
    return logits.argmax(dim=-1).cpu().numpy().tolist()


# ═══════════════════════════════════════════════════════════════════════════
# TOP-K MASKING EVALUATION  (test set)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_sae(vit, sae, cfg, cls_sae_cnt, text_features,
                 test_loader, num_classes, classnames, device,
                 vit_type, topk_list):
    class_images = defaultdict(list)
    to_pil = transforms.ToPILImage()
    for images, labels in test_loader:
        for i in range(images.size(0)):
            class_images[labels[i].item()].append(images[i])

    d_sae = cfg.d_sae
    full_topk = [k for k in topk_list if k < d_sae] + [d_sae]

    metrics = {}
    for cls_idx in range(num_classes):
        imgs = class_images[cls_idx]
        if not imgs:
            metrics[cls_idx] = {}
            continue

        preds = defaultdict(list)
        sorted_feats = cls_sae_cnt[cls_idx].argsort()[::-1]

        for start in range(0, len(imgs), BATCH_SIZE):
            batch = imgs[start:start + BATCH_SIZE]
            inputs = vit.processor(images=[to_pil(img) for img in batch],
                                   text="", return_tensors="pt", padding=True).to(device)

            preds["no_sae"].extend(
                get_predictions(vit, inputs, text_features, vit_type))

            for k in full_topk:
                feats = sorted_feats[:k].tolist()
                hooks_on = create_sae_hooks(cfg, feats, sae, device, vit_type, "on")
                preds[f"on_{k}"].extend(
                    get_predictions(vit, inputs, text_features, vit_type, hooks_on))

                hooks_off = create_sae_hooks(cfg, feats, sae, device, vit_type, "off")
                preds[f"off_{k}"].extend(
                    get_predictions(vit, inputs, text_features, vit_type, hooks_off))
            flush()

        metrics[cls_idx] = {key: sum(p == cls_idx for p in ps) / len(ps) * 100
                            for key, ps in preds.items()}

        bl = metrics[cls_idx].get("no_sae", 0)
        print(f"    {classnames[cls_idx]:<45s} baseline={bl:5.1f}%  "
              f"on@10={metrics[cls_idx].get('on_10', 0):5.1f}%  "
              f"off@10={metrics[cls_idx].get('off_10', 0):5.1f}%")

    return pd.DataFrame(metrics)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Eval SAEs on MedMNIST test set (base / lora / maple)")
    parser.add_argument("--vit_type", type=str, default="base",
                        choices=["base", "lora", "maple"],
                        help="ViT backbone: base (vanilla CLIP), lora, or maple")
    parser.add_argument("--base_sae", type=str, default=BASE_SAE_PATH)
    parser.add_argument("--medmnist_sae_dir", type=str, default=MEDMNIST_SAE_DIR)
    parser.add_argument("--train_root", type=str, default=TRAIN_ROOT)
    parser.add_argument("--npz", type=str, default=NPZ_PATH)
    parser.add_argument("--backbone", type=str, default=BACKBONE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output dir (default: out/medmnist_eval_{vit_type})")
    parser.add_argument("--include_base", action="store_true",
                        help="Include the base ImageNet SAE")
    parser.add_argument("--reuse_activations", action="store_true",
                        help="Reuse cached cls_sae_cnt .npy files")
    parser.add_argument("--topk", type=int, nargs="+", default=TOPK_LIST)
    # LoRA args
    parser.add_argument("--lora_checkpoint", type=str, default=LORA_CHECKPOINT,
                        help="LoRA weights path (for --vit_type lora)")
    # MaPLe args
    parser.add_argument("--model_path", type=str, default=MAPLE_MODEL_PATH,
                        help="MaPLe model checkpoint (for --vit_type maple)")
    parser.add_argument("--config_path", type=str, default=MAPLE_CONFIG_PATH,
                        help="MaPLe config path (for --vit_type maple)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Default save dir includes vit_type so runs don't overwrite each other
    if args.save_dir is None:
        args.save_dir = f"out/medmnist_eval_{args.vit_type}"
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print(f"MEDMNIST SAE EVAL  (vit_type={args.vit_type}, top-k masking, test set)")
    print("=" * 70)

    # 1. Class names — use ImageFolder alphabetical order so that
    #    text_features[i] matches label i from NpzTestDataset.
    classnames = get_imagefolder_classnames(args.train_root)
    num_classes = len(classnames)
    print(f"\nClasses ({num_classes}, ImageFolder order): {classnames}")

    # 2. Load SAEs
    print(f"\nLoading SAEs...")
    med_paths = discover_medmnist_saes(args.medmnist_sae_dir)
    saes = load_selected_saes(args.base_sae, med_paths, args.device,
                              include_base=args.include_base)
    if not saes:
        print("[FATAL] No SAEs loaded."); sys.exit(1)

    # 3. Load ViT
    print(f"\nLoading ViT ({args.vit_type})...")
    ref_cfg = saes[0][1]
    vit = load_vit_for_type(
        args.vit_type, ref_cfg, args.backbone, args.device,
        lora_path=args.lora_checkpoint,
        model_path=args.model_path,
        config_path=args.config_path,
        classnames=classnames,
    )

    # 4. Text features
    print(f"\nComputing text features (vit_type={args.vit_type})...")
    text_features = calculate_text_features(vit, args.device, classnames, args.vit_type)
    print(f"  Shape: {text_features.shape}")

    # 5. Datasets
    #    Use get_processor_transform() (Resize + ToTensor only, NO Normalize)
    #    because images are converted to PIL and re-processed by CLIPProcessor
    #    in the eval loop.  Using get_transform() would double-normalise and
    #    corrupt images via the lossy ToPILImage round-trip.
    transform = get_processor_transform()
    train_dataset = datasets.ImageFolder(args.train_root, transform=transform)
    print(f"\nTrain: {len(train_dataset)} (activation profiles)")

    label_mapping = build_npz_to_imagefolder_mapping(args.train_root)
    test_dataset = NpzTestDataset(args.npz, label_mapping, transform)
    print(f"Test:  {len(test_dataset)} (from npz, labels remapped)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # 6. Evaluate each SAE
    all_summaries = {}

    for sae, cfg, label in saes:
        print(f"\n{'═' * 70}")
        print(f"SAE: {label}  (d_sae={cfg.d_sae}, layer={cfg.block_layer})")
        print(f"{'═' * 70}")

        # 6a. Per-class activation profiles (cached per vit_type + sae)
        cnt_path = os.path.join(args.save_dir, f"cls_sae_cnt_{label}.npy")
        if args.reuse_activations and os.path.exists(cnt_path):
            print(f"  [REUSE] {cnt_path}")
            cls_sae_cnt = np.load(cnt_path)
        else:
            cls_sae_cnt = compute_cls_sae_activations(
                vit, sae, cfg, train_loader, num_classes, args.device)
            np.save(cnt_path, cls_sae_cnt)
            print(f"  Saved: {cnt_path}")

        # 6b. Top-k masking on test set
        print(f"\n  Test set evaluation:")
        df = evaluate_sae(vit, sae, cfg, cls_sae_cnt, text_features,
                          test_loader, num_classes, classnames, args.device,
                          args.vit_type, args.topk)

        csv_path = os.path.join(args.save_dir, f"metrics_{label}.csv")
        df.to_csv(csv_path)
        print(f"\n  Saved: {csv_path}")

        # Summary
        mean_accs = df.mean(axis=1)
        baseline = mean_accs.get("no_sae", 0)
        print(f"\n  {'Condition':<20s} {'Mean Acc':>10s} {'Δ':>8s}")
        print(f"  {'─' * 40}")
        print(f"  {'baseline':<20s} {baseline:>9.2f}%")
        for k in args.topk:
            on  = mean_accs.get(f"on_{k}", 0)
            off = mean_accs.get(f"off_{k}", 0)
            print(f"  {'top-'+str(k)+' ON':<20s} {on:>9.2f}% {on-baseline:>+7.2f}")
            print(f"  {'top-'+str(k)+' OFF':<20s} {off:>9.2f}% {off-baseline:>+7.2f}")

        all_summaries[label] = {
            "baseline": baseline,
            "best_on": max(mean_accs.get(f"on_{k}", 0) for k in args.topk),
            "worst_off": min(mean_accs.get(f"off_{k}", 0) for k in args.topk),
            "layer": cfg.block_layer, "d_sae": cfg.d_sae,
        }
        flush()

    # 7. Cross-SAE comparison
    print(f"\n{'═' * 70}")
    print(f"COMPARISON  (vit_type={args.vit_type})")
    print(f"{'═' * 70}")
    print(f"{'SAE':<25s} {'Layer':>6s} {'Baseline':>10s} {'Best ON':>10s} "
          f"{'Worst OFF':>11s} {'ON Δ':>8s} {'OFF Δ':>8s}")
    print(f"{'─' * 82}")
    for label, r in all_summaries.items():
        print(f"{label:<25s} {r['layer']:>6} {r['baseline']:>9.2f}% "
              f"{r['best_on']:>9.2f}% {r['worst_off']:>10.2f}% "
              f"{r['best_on']-r['baseline']:>+7.2f} "
              f"{r['worst_off']-r['baseline']:>+7.2f}")

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"vit_type": args.vit_type, **all_summaries}, f, indent=2)
    print(f"\nSaved: {summary_path}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
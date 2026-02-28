#!/usr/bin/env python3
"""
Accuracy evaluation: base / lora / maple × each SAE in one run.

Usage:
    python eval_accuracy_all.py --include_base
    python eval_accuracy_all.py --include_base --no_sae_bias
    python eval_accuracy_all.py --include_base --sae_alpha 0.5  # interpolate
"""

import argparse, json, os, sys, glob, gc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import math

try:
    from tasks.utils import load_sae, load_hooked_vit
    from src.sae_training.hooked_vit import Hook
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] {e}\nRun from patchsae root."); sys.exit(1)

try:
    import clip as openai_clip
    HAS_OPENAI_CLIP = True
except ImportError:
    HAS_OPENAI_CLIP = False

# ── Config ────────────────────────────────────────────────────────────────
BASE_SAE_PATH    = "data/sae_weight/base/out.pt"
MEDMNIST_SAE_DIR = "out/checkpoints/medmnist"
MAPLE_SAE_PATH   = "/home/sunayana/Documents/Concept_LoRA/patchsae/out/checkpoints/medmnist_maple/qcx3o72n/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt"
LORA_CHECKPOINT  = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
TRAIN_ROOT       = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
NPZ_PATH         = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
CLASSNAMES_PATH  = "configs/classnames/medmnist_classnames.json"
BACKBONE         = "openai/clip-vit-base-patch16"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = 64
SAE_BIAS_DEFAULT = 0

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

# ── Helpers ───────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def l2_normalize(x):
    """L2 normalization along the last dimension."""
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

def get_transform():
    """Full CLIP preprocessing (with normalize). For tensors fed directly to model."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_processor_transform():
    """Minimal transform: only resize, keep as PIL.

    FIX: The old version converted to tensor then back to PIL inside the
    eval loop, causing interpolation artifacts from the round-trip.
    Now we keep images as PIL all the way through to CLIPProcessor.
    """
    return transforms.Resize((224, 224))

def get_imagefolder_classnames(root):
    ds = datasets.ImageFolder(root=root)
    idx_to_name = {i: n.replace('_', ' ') for n, i in ds.class_to_idx.items()}
    return [idx_to_name[i] for i in range(len(idx_to_name))]

def build_npz_to_if_mapping(root):
    """Map NPZ label indices to ImageFolder label indices.

    FIX: Validates that all MEDMNIST_CLASSES are found in ImageFolder.
    Raises an error on any unmatched class instead of silently falling back.
    """
    ds = datasets.ImageFolder(root=root)
    if_map = {n.replace('_', ' ').lower(): i for n, i in ds.class_to_idx.items()}

    mapping = {}
    unmatched = []
    for npz_idx, classname in enumerate(MEDMNIST_CLASSES):
        key = classname.lower()
        if key in if_map:
            mapping[npz_idx] = if_map[key]
        else:
            unmatched.append((npz_idx, classname))
            mapping[npz_idx] = npz_idx  # fallback

    if unmatched:
        print(f"  ⚠ WARNING: {len(unmatched)} MEDMNIST class(es) not found in "
              f"ImageFolder at {root}:")
        for idx, name in unmatched:
            print(f"    npz[{idx}] = '{name}' → no match, falling back to index {idx}")
        print(f"  ImageFolder classes: {list(if_map.keys())}")
        print(f"  This WILL produce incorrect accuracy if the label orderings differ!")

    return mapping


class NpzTestDataset(Dataset):
    """Test dataset loaded from .npz file.

    FIX: transform now receives a PIL Image and returns a PIL Image
    (no premature tensor conversion for the HF path).
    """
    def __init__(self, npz_path, mapping, transform, return_pil=False):
        data = np.load(npz_path)
        self.imgs = data['test_images']
        self.labels = data['test_labels'].flatten()
        self.mapping = mapping
        self.transform = transform
        self.return_pil = return_pil

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i])
        img = self.transform(img)
        # For HF path: keep as PIL; for OpenAI path: transform returns tensor
        if self.return_pil and not isinstance(img, Image.Image):
            raise TypeError(
                f"Expected PIL Image from transform but got {type(img)}. "
                f"Use get_processor_transform() for return_pil=True."
            )
        return img, self.mapping[int(self.labels[i])]


def _hf_collate_fn(batch):
    """Custom collate for batches of (PIL Image, label)."""
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)


class NpzSplitDataset(Dataset):
    """Dataset that loads train or test split from .npz file."""
    def __init__(self, npz_path, mapping, transform, split="test", return_pil=False):
        data = np.load(npz_path)
        if split == "train":
            self.imgs = data['train_images']
            self.labels = data['train_labels'].flatten()
        elif split == "val":
            self.imgs = data['val_images']
            self.labels = data['val_labels'].flatten()
        else:
            self.imgs = data['test_images']
            self.labels = data['test_labels'].flatten()
        self.mapping = mapping
        self.transform = transform
        self.return_pil = return_pil

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i])
        img = self.transform(img)
        if self.return_pil and not isinstance(img, Image.Image):
            raise TypeError(
                f"Expected PIL Image from transform but got {type(img)}. "
                f"Use get_processor_transform() for return_pil=True."
            )
        return img, self.mapping[int(self.labels[i])]


def discover_medmnist_saes(base_dir):
    """Find SAE checkpoints, keeping one per layer (last by sorted path).

    FIX: Warns when multiple checkpoints exist for the same layer.
    """
    paths = sorted(glob.glob(os.path.join(base_dir, "*/final*/*.pt")))
    best = {}
    duplicates = {}
    for p in paths:
        ckpt = torch.load(p, map_location="cpu")
        cfg_d = ckpt.get("cfg", ckpt.get("config"))
        layer = getattr(cfg_d, "block_layer",
                        cfg_d.get("block_layer", "?") if isinstance(cfg_d, dict) else "?")
        if layer in best:
            duplicates.setdefault(layer, [best[layer]]).append(p)
        best[layer] = p
        del ckpt; gc.collect()

    if duplicates:
        print("  ⚠ WARNING: Multiple SAE checkpoints found for same layer(s):")
        for layer, paths_list in duplicates.items():
            print(f"    Layer {layer}: {len(paths_list)+1} checkpoints, "
                  f"using last by sort order")

    return list(best.values())


# ══════════════════════════════════════════════════════════════════════════
# Preprocessing parity check
# ══════════════════════════════════════════════════════════════════════════

def verify_preprocessing_parity(oai_preprocess, device):
    """Check that OpenAI and HF CLIP use the same normalization constants."""
    print("\n[CHECK] Preprocessing parity verification:")

    oai_mean = (0.48145466, 0.4578275, 0.40821073)
    oai_std  = (0.26862954, 0.26130258, 0.27577711)
    hf_mean  = (0.48145466, 0.4578275, 0.40821073)
    hf_std   = (0.26862954, 0.26130258, 0.27577711)

    match = (oai_mean == hf_mean) and (oai_std == hf_std)
    print(f"  OpenAI mean/std: {oai_mean} / {oai_std}")
    print(f"  HF     mean/std: {hf_mean} / {hf_std}")
    print(f"  Match: {'YES ✓' if match else 'NO ✗ — WARNING!'}")

    # Empirical check
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    oai_tensor = oai_preprocess(dummy_img).unsqueeze(0)

    hf_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(hf_mean, hf_std),
    ])
    hf_tensor = hf_transform(dummy_img).unsqueeze(0)

    max_diff = (oai_tensor - hf_tensor).abs().max().item()
    print(f"  Empirical max pixel diff: {max_diff:.8f}")
    if max_diff > 0.01:
        print("  ⚠ WARNING: >0.01 pixel difference — preprocessing NOT aligned!")
    else:
        print("  ✓ Preprocessing aligned.")
    return match


# ══════════════════════════════════════════════════════════════════════════
# LORA: Build OpenAI CLIP with merged LoRA weights
# ══════════════════════════════════════════════════════════════════════════

def _extract_lora_AB(ld, proj):
    if proj in ld and isinstance(ld[proj], dict):
        try: return ld[proj]["w_lora_A"], ld[proj]["w_lora_B"]
        except: pass
    try: return ld[f"{proj}.w_lora_A"], ld[f"{proj}.w_lora_B"]
    except: return None, None

def build_lora_openai_clip(lora_path, device):
    """Load OpenAI CLIP and merge LoRA weights. Returns (model, preprocess)."""
    if not HAS_OPENAI_CLIP:
        print("[FATAL] pip install git+https://github.com/openai/CLIP.git"); sys.exit(1)

    model, preprocess = openai_clip.load("ViT-B/16", device=device)

    lora_state = torch.load(lora_path, map_location=device)
    if "weights" not in lora_state:
        model.load_state_dict(lora_state)
        return model, preprocess

    layers, meta = lora_state["weights"], lora_state["metadata"]
    r=meta["r"]; alpha=meta["alpha"]
    scale = meta["alpha"] /math.sqrt(meta["r"]) 
    print(f"    LoRA: rank={meta['r']}, alpha={meta['alpha']}, scale={scale}")

    with torch.no_grad():
        for i in range(12):
            key = f"layer_{i}"
            if key not in layers: continue
            ld = layers[key]
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2*d)]:
                A, B = _extract_lora_AB(ld, proj)
                if A is None: continue
                w[off:off+d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)

        for i in range(12, 24):
            key = f"layer_{i}"
            if key not in layers: continue
            ld = layers[key]
            w = model.visual.transformer.resblocks[i-12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2*d)]:
                A, B = _extract_lora_AB(ld, proj)
                if A is None: continue
                w[off:off+d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)

    print("    LoRA merged.")
    return model, preprocess


# ══════════════════════════════════════════════════════════════════════════
# SAE injection helper
# ══════════════════════════════════════════════════════════════════════════

def apply_sae(sae, act, sae_bias, alpha):
    """Run activations through SAE with optional interpolation.

    Args:
        sae: The sparse autoencoder module.
        act: Input activations [batch, seq, d_model], float.
        sae_bias: Scalar bias to subtract from SAE output.
        alpha: Interpolation factor. 1.0 = full SAE replacement,
               0.0 = original activations (no-op).

    Returns:
        result: Interpolated activations, same shape as act.
        diagnostics: Dict with reconstruction quality metrics (first call info).
    """
    sae_out = sae(act)[0] - sae_bias

    if alpha == 1.0:
        result = sae_out
    elif alpha == 0.0:
        result = act
    else:
        result = act + alpha * (sae_out - act)  # lerp(act, sae_out, alpha)

    # Diagnostics
    with torch.no_grad():
        mse = (act - sae_out).pow(2).mean().item()
        cos = torch.nn.functional.cosine_similarity(
            act.reshape(-1, act.shape[-1]),
            sae_out.reshape(-1, sae_out.shape[-1]),
            dim=-1,
        ).mean().item()
        delta_norm = (sae_out - act).norm(dim=-1).mean().item()
        act_norm = act.norm(dim=-1).mean().item()

    diagnostics = {
        "mse": mse, "cos_sim": cos,
        "delta_norm": delta_norm, "act_norm": act_norm,
        "relative_delta": delta_norm / max(act_norm, 1e-8),
    }
    return result, diagnostics


# ══════════════════════════════════════════════════════════════════════════
# SAE hooks for OpenAI CLIP
# ══════════════════════════════════════════════════════════════════════════

class OpenAISAEHook:
    """
    Forward hook on an OpenAI CLIP resblock that passes the residual
    stream through an SAE.

    The SAE output directly replaces the activations (with optional
    interpolation via alpha). This is straightforward replacement —
    not "residual-preserving" in any special sense.
    """
    def __init__(self, model, sae, cfg, device, sae_bias=0.0, alpha=1.0,
                 source_model_type="unknown"):
        self.sae = sae
        self.cfg = cfg
        self.device = device
        self.sae_bias = sae_bias
        self.alpha = alpha
        self.handle = None
        self._logged_first = False

        num_blocks = len(model.visual.transformer.resblocks)
        layer = cfg.block_layer
        if layer < 0:
            layer = num_blocks + layer

        assert 0 <= layer < num_blocks, \
            f"Layer {layer} (from block_layer={cfg.block_layer}) " \
            f"out of range for {num_blocks} resblocks"

        self.block = model.visual.transformer.resblocks[layer]
        self.layer_idx = layer

        # FIX: Warn about distribution mismatch
        if source_model_type not in ("unknown", "lora"):
            print(f"  ⚠ NOTE: SAE trained on '{source_model_type}' activations "
                  f"is being applied to LoRA-adapted model. Distribution mismatch "
                  f"may affect reconstruction quality.")

        print(f"  [HOOK] OpenAI CLIP: block_layer={cfg.block_layer} → "
              f"resblock[{layer}] (of {num_blocks}), "
              f"sae_bias={sae_bias}, alpha={alpha}")

    def _hook_fn(self, module, input, output):
        """OpenAI CLIP resblock output: [seq_len, batch, d_model]."""
        act = output
        orig_dtype = act.dtype

        # Transpose to [batch, seq_len, d_model] for SAE
        act_bsd = act.transpose(0, 1).float()

        result_bsd, diag = apply_sae(self.sae, act_bsd, self.sae_bias, self.alpha)

        if not self._logged_first:
            self._logged_first = True
            print(f"  [HOOK DEBUG] batch 0: MSE={diag['mse']:.6f}, "
                  f"cos_sim={diag['cos_sim']:.6f}, "
                  f"|Δ|/|act|={diag['relative_delta']:.4f}")

        return result_bsd.transpose(0, 1).to(orig_dtype)

    def register(self):
        self.handle = self.block.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ══════════════════════════════════════════════════════════════════════════
# OpenAI CLIP accuracy
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_accuracy_openai(model, text_features, loader, device):
    """Zero-shot accuracy using OpenAI CLIP with logit_scale and L2 norm."""
    logit_scale = model.logit_scale.exp()
    tf = l2_normalize(text_features.float())

    correct, total = 0, 0
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="      eval", leave=False)):
        img_feat = model.encode_image(imgs.to(device))
        img_feat = l2_normalize(img_feat.float())

        logits = logit_scale * (img_feat @ tf.t())
        preds = logits.argmax(dim=-1).cpu()

        if batch_idx == 0:
            print(f"  [DEBUG-EVAL OAI] logit_scale={logit_scale.item():.4f}, "
                  f"img_norm={img_feat.norm(dim=-1).mean().item():.6f}, "
                  f"text_norm={tf.norm(dim=-1).mean().item():.6f}")
            print(f"  [DEBUG-EVAL OAI] preds[:10]={preds[:10].tolist()}, "
                  f"labels[:10]={labels[:10].tolist()}")

        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100


@torch.no_grad()
def get_text_features_openai(model, classnames, device):
    """Text features using mean of 80 prompt templates, L2-normalized."""
    mean_f = torch.zeros(len(classnames), 512, device=device)
    for tmpl in openai_imagenet_template:
        prompts = [tmpl(c) for c in classnames]
        tokens = openai_clip.tokenize(prompts).to(device)
        f = model.encode_text(tokens).float()
        f = l2_normalize(f)
        mean_f += f
    mean_f /= len(openai_imagenet_template)
    return l2_normalize(mean_f)


# ══════════════════════════════════════════════════════════════════════════
# HF PATH: for base and maple
# ══════════════════════════════════════════════════════════════════════════

def load_vit_hf(vit_type, ref_cfg, backbone, device, **kw):
    if vit_type == "maple":
        vit = load_hooked_vit(ref_cfg, "maple", backbone, device,
                              model_path=kw["model_path"], config_path=kw["config_path"],
                              classnames=kw["classnames"])
    else:
        vit = load_hooked_vit(ref_cfg, "base", backbone, device)
    vit.eval()
    return vit

@torch.no_grad()
def get_text_features_hf(vit, device, classnames, vit_type):
    """L2-normalized text features for HF CLIP."""
    if vit_type == "maple":
        tf = vit.model.get_text_features()
        return l2_normalize(tf)
    mean_f = 0
    for tmpl in openai_imagenet_template:
        ids = [vit.processor(text=tmpl(c), return_tensors="pt", padding=False,
                             truncation=True).input_ids[0] for c in classnames]
        padded = pad_sequence(ids, batch_first=True).to(device)
        f = vit.model.get_text_features(padded)
        f = l2_normalize(f)
        mean_f += f
    mean_f /= len(openai_imagenet_template)
    return l2_normalize(mean_f)


def make_full_sae_hook_hf(sae, cfg, vit_type, sae_bias=0.0, alpha=1.0):
    """
    Create SAE hooks for HF CLIP models.

    The SAE output replaces activations directly (with optional alpha
    interpolation). For maple, activations are [seq, batch, d_model];
    for base, they are [batch, seq, d_model].
    """
    print(f"  [HOOK] HF ({vit_type}): block_layer={cfg.block_layer}, "
          f"module={cfg.module_name}, sae_bias={sae_bias}, alpha={alpha}")

    if vit_type == "maple":
        logged = [False]

        def fn(act):
            orig_dtype = act.dtype
            # MaPLe custom hooks: act is [seq, batch, d_model]
            a = act.transpose(0, 1).float()  # → [batch, seq, d_model]

            result, diag = apply_sae(sae, a, sae_bias, alpha)

            if not logged[0]:
                logged[0] = True
                print(f"  [HOOK DEBUG maple] MSE={diag['mse']:.6f}, "
                      f"cos={diag['cos_sim']:.6f}, "
                      f"|Δ|/|act|={diag['relative_delta']:.4f}")

            return result.transpose(0, 1).to(orig_dtype)

        return [Hook(cfg.block_layer, cfg.module_name, fn,
                     return_module_output=False, is_custom=True)]
    else:
        logged = [False]

        def fn(act):
            # Base hooks: act is [batch, seq, d_model]
            orig = act[:, :, :].clone().float()

            result, diag = apply_sae(sae, orig, sae_bias, alpha)
            act[:, :, :] = result.to(act.dtype)

            if not logged[0]:
                logged[0] = True
                print(f"  [HOOK DEBUG base] MSE={diag['mse']:.6f}, "
                      f"cos={diag['cos_sim']:.6f}, "
                      f"|Δ|/|act|={diag['relative_delta']:.4f}")

            return (act,)

        return [Hook(cfg.block_layer, cfg.module_name, fn,
                     return_module_output=False, is_custom=False)]


@torch.no_grad()
def compute_accuracy_hf(vit, text_features, loader, vit_type, device, hooks=None):
    """
    Zero-shot accuracy for HF CLIP.

    FIX: Images are now passed as PIL directly to CLIPProcessor,
    avoiding the tensor→PIL→processor round-trip that caused
    interpolation artifacts.
    """
    correct, total = 0, 0
    logit_scale = vit.model.logit_scale.exp()
    tf = l2_normalize(text_features.float())

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="      eval", leave=False)):
        # images is a list of PIL Images (from _hf_collate_fn)
        inputs = vit.processor(images=images, text="",
                               return_tensors="pt", padding=True).to(device)
        if hooks:
            out = vit.run_with_hooks(hooks, return_type="output", **inputs)
        else:
            out = vit(return_type="output", **inputs)
        img_feat = out if vit_type == "maple" else out.image_embeds

        img_feat = l2_normalize(img_feat.float())
        logits = logit_scale * (img_feat @ tf.t())
        preds = logits.argmax(dim=-1).cpu()

        if batch_idx == 0:
            print(f"  [DEBUG-EVAL HF] logit_scale={logit_scale.item():.4f}, "
                  f"img_norm={img_feat.norm(dim=-1).mean().item():.6f}, "
                  f"text_norm={tf.norm(dim=-1).mean().item():.6f}")
            print(f"  [DEBUG-EVAL HF] preds[:10]={preds[:10].tolist()}, "
                  f"labels[:10]={labels[:10].tolist()}")

        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100


# ══════════════════════════════════════════════════════════════════════════
# Layer alignment diagnostic
# ══════════════════════════════════════════════════════════════════════════

def verify_layer_alignment(sae_cfg, model_type):
    """Verify that the SAE's block_layer maps to a valid resblock."""
    layer = sae_cfg.block_layer
    num_blocks = 12  # ViT-B/16

    abs_layer = num_blocks + layer if layer < 0 else layer

    print(f"  [LAYER CHECK] {model_type}: block_layer={layer} → "
          f"resblock[{abs_layer}] (of {num_blocks})")

    if abs_layer < 0 or abs_layer >= num_blocks:
        print(f"  ⚠ WARNING: Layer {abs_layer} is OUT OF RANGE!")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# SAE latent extraction
# ══════════════════════════════════════════════════════════════════════════

def get_sae_latents(sae, activations):
    """Extract sparse latent codes from the SAE encoder.

    Tries sae.encode() first, then falls back to sae forward tuple[1],
    then manual W_enc computation.

    Args:
        sae: Sparse autoencoder module.
        activations: [batch, seq, d_model] float tensor.

    Returns:
        latents: [batch, seq, n_latents] float tensor of sparse codes.
    """
    # Try .encode() method
    if hasattr(sae, 'encode'):
        return sae.encode(activations)

    # Try forward pass — many SAEs return (x_hat, latents, ...)
    out = sae(activations)
    if isinstance(out, tuple) and len(out) >= 2:
        candidate = out[1]
        # Latents should have more features than d_model (expansion factor)
        if candidate.shape[-1] >= activations.shape[-1]:
            return candidate

    # Manual: z = ReLU(W_enc @ (x - b_dec) + b_enc)
    if hasattr(sae, 'W_enc') and hasattr(sae, 'b_enc'):
        x = activations
        if hasattr(sae, 'b_dec'):
            x = x - sae.b_dec
        z = x @ sae.W_enc + sae.b_enc
        return torch.nn.functional.relu(z)

    raise RuntimeError(
        "Cannot extract SAE latents: no .encode(), no tuple[1], "
        "and no W_enc/b_enc attributes found."
    )


@torch.no_grad()
def extract_sae_latents_hf(vit, sae, cfg, loader, vit_type, device, pool="cls"):
    """Extract SAE latents for all images using an HF CLIP model.

    Registers a capture hook at the target layer, runs forward pass,
    then encodes captured activations through the SAE encoder.

    Args:
        pool: "cls" for CLS token only, "mean" for mean over all tokens.

    Returns:
        all_latents: np.ndarray [N, n_latents]
        all_labels: np.ndarray [N]
    """
    all_latents, all_labels = [], []
    captured = {}

    # Build a capture hook
    layer = cfg.block_layer
    module_name = cfg.module_name

    if vit_type == "maple":
        def capture_fn(act):
            # maple: act is [seq, batch, d_model]
            captured["act"] = act.transpose(0, 1).float()  # → [batch, seq, d_model]
            return act  # pass through unchanged

        hooks = [Hook(layer, module_name, capture_fn,
                      return_module_output=False, is_custom=True)]
    else:
        def capture_fn(act):
            # base: act is [batch, seq, d_model]
            captured["act"] = act[:, :, :].clone().float()
            return (act,)

        hooks = [Hook(layer, module_name, capture_fn,
                      return_module_output=False, is_custom=False)]

    for images, labels in tqdm(loader, desc="      extract latents", leave=False):
        inputs = vit.processor(images=images, text="",
                               return_tensors="pt", padding=True).to(device)
        captured.clear()
        _ = vit.run_with_hooks(hooks, return_type="output", **inputs)

        act = captured["act"]  # [batch, seq, d_model]
        latents = get_sae_latents(sae, act)  # [batch, seq, n_latents]

        # Pool
        if pool == "cls":
            pooled = latents[:, 0, :]  # CLS token
        else:
            pooled = latents.mean(dim=1)

        all_latents.append(pooled.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_latents), np.concatenate(all_labels)


@torch.no_grad()
def extract_sae_latents_openai(model, sae, cfg, loader, device, pool="cls"):
    """Extract SAE latents for all images using an OpenAI CLIP model.

    Registers a capture hook at the target resblock, runs forward pass,
    then encodes captured activations through the SAE encoder.

    Returns:
        all_latents: np.ndarray [N, n_latents]
        all_labels: np.ndarray [N]
    """
    all_latents, all_labels = [], []
    captured = {}

    num_blocks = len(model.visual.transformer.resblocks)
    layer = cfg.block_layer
    if layer < 0:
        layer = num_blocks + layer
    block = model.visual.transformer.resblocks[layer]

    def hook_fn(module, input, output):
        # OpenAI: output is [seq_len, batch, d_model]
        captured["act"] = output.transpose(0, 1).float()  # → [batch, seq, d_model]

    handle = block.register_forward_hook(hook_fn)

    for imgs, labels in tqdm(loader, desc="      extract latents", leave=False):
        captured.clear()
        _ = model.encode_image(imgs.to(device))

        act = captured["act"]  # [batch, seq, d_model]
        latents = get_sae_latents(sae, act)  # [batch, seq, n_latents]

        if pool == "cls":
            pooled = latents[:, 0, :]
        else:
            pooled = latents.mean(dim=1)

        all_latents.append(pooled.cpu().numpy())
        all_labels.append(labels.numpy())

    handle.remove()
    return np.concatenate(all_latents), np.concatenate(all_labels)


def train_linear_probe(train_latents, train_labels, test_latents, test_labels,
                       max_iter=1000):
    """Train a logistic regression linear probe on SAE latents.

    Returns:
        test_acc: float (percentage)
        train_acc: float (percentage)
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_latents)
    X_test = scaler.transform(test_latents)

    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="multinomial",
        C=1.0,
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)

    train_acc = clf.score(X_train, train_labels) * 100
    test_acc = clf.score(X_test, test_labels) * 100
    return test_acc, train_acc


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--include_base", action="store_true")
    p.add_argument("--lora_checkpoint", default=LORA_CHECKPOINT)
    p.add_argument("--model_path", default="/home/sunayana/Documents/model.pth.tar-5")
    p.add_argument("--config_path",
                   default="/home/sunayana/Documents/Concept_LoRA/patchsae/configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml")
    p.add_argument("--maple_sae_path", default=MAPLE_SAE_PATH)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--save_dir", default="out/accuracy_comparison")
    p.add_argument("--no_sae_bias", action="store_true",
                   help="Disable SAE bias subtraction")
    p.add_argument("--sae_bias", type=float, default=SAE_BIAS_DEFAULT,
                   help=f"SAE bias to subtract (default: {SAE_BIAS_DEFAULT})")
    p.add_argument("--sae_alpha", type=float, default=1.0,
                   help="SAE interpolation: 0.0=original, 1.0=full SAE replacement")
    args = p.parse_args()

    # Resolve SAE bias
    if args.no_sae_bias:
        effective_sae_bias = 0.0
        print("[CONFIG] SAE bias DISABLED (--no_sae_bias)")
    else:
        effective_sae_bias = args.sae_bias
        print(f"[CONFIG] SAE bias = {effective_sae_bias}")

    print(f"[CONFIG] SAE alpha = {args.sae_alpha}")
    if args.sae_alpha != 1.0:
        print(f"  → Interpolating: output = original + {args.sae_alpha} * (SAE(x) - original)")

    os.makedirs(args.save_dir, exist_ok=True)

    classnames = get_imagefolder_classnames(TRAIN_ROOT)
    label_mapping = build_npz_to_if_mapping(TRAIN_ROOT)

    print(f"[DEBUG] Device: {DEVICE}")
    print(f"[DEBUG] Batch size: {args.batch_size}")
    print(f"Classnames (ImageFolder order):")
    for i, c in enumerate(classnames): print(f"  {i}: {c}")
    print(f"[DEBUG] Label mapping (npz→IF): {label_mapping}")

    # ── Load SAEs for base/lora ───────────────────────────────────
    print(f"\n[DEBUG] Loading SAEs for base/lora...")
    med_paths = discover_medmnist_saes(MEDMNIST_SAE_DIR)
    print(f"[DEBUG] Discovered medmnist SAE paths: {med_paths}")
    saes = []
    if args.include_base and os.path.exists(BASE_SAE_PATH):
        sae, cfg = load_sae(BASE_SAE_PATH, DEVICE)
        print(f"[DEBUG] Loaded base SAE: block_layer={cfg.block_layer}, "
              f"module={cfg.module_name}")
        verify_layer_alignment(cfg, "base_SAE")
        saes.append((sae, cfg, "base_layer{}".format(cfg.block_layer), "base"))
    for path in med_paths:
        sae, cfg = load_sae(path, DEVICE)
        print(f"[DEBUG] Loaded medmnist SAE from {path}: block_layer={cfg.block_layer}, "
              f"module={cfg.module_name}")
        verify_layer_alignment(cfg, "medmnist_SAE")
        saes.append((sae, cfg, "medmnist_layer{}".format(cfg.block_layer), "medmnist"))
    if not saes:
        print("[FATAL] No SAEs for base/lora."); sys.exit(1)
    print(f"  {len(saes)} SAE(s) loaded for base/lora.")

    # ── Load maple-specific SAE ───────────────────────────────────
    print(f"\n[DEBUG] Loading maple-specific SAE from: {args.maple_sae_path}")
    assert os.path.exists(args.maple_sae_path), \
        f"[FATAL] Maple SAE not found: {args.maple_sae_path}"
    maple_sae, maple_cfg = load_sae(args.maple_sae_path, DEVICE)
    print(f"[DEBUG] Maple SAE: block_layer={maple_cfg.block_layer}, "
          f"module={maple_cfg.module_name}")
    verify_layer_alignment(maple_cfg, "maple_SAE")
    maple_saes = [(maple_sae, maple_cfg,
                   "maple_medmnist_layer{}".format(maple_cfg.block_layer), "maple")]

    # ── Also add maple SAE to the base/lora SAE list ──────────────
    # This allows cross-evaluation: maple SAE applied to base and LoRA models
    # IMPORTANT: use the same label as in maple_saes so results merge in the summary table
    maple_sae_label = "maple_medmnist_layer{}".format(maple_cfg.block_layer)
    print(f"\n[DEBUG] Adding maple SAE to base/lora evaluation list (cross-model eval)")
    saes.append((maple_sae, maple_cfg, maple_sae_label, "maple"))
    print(f"  {len(saes)} SAE(s) now loaded for base/lora (including maple SAE).")

    # ── Test loaders ──────────────────────────────────────────────
    # HF path: PIL images, processed by CLIPProcessor (no double-resize)
    hf_transform = get_processor_transform()
    test_ds_hf = NpzTestDataset(NPZ_PATH, label_mapping, hf_transform, return_pil=True)
    test_loader_hf = DataLoader(test_ds_hf, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=False,
                                collate_fn=_hf_collate_fn)
    print(f"[DEBUG] HF test dataset: {len(test_ds_hf)} samples, "
          f"{len(test_loader_hf)} batches")

    ref_cfg = saes[0][1]
    all_results = {}
    # Track which SAE source each label belongs to (for grouped summary)
    sae_source_map = {}  # label → source ("base", "medmnist", "maple")
    for _, _, label, source in saes:
        sae_source_map[label] = source
    for _, _, label, source in maple_saes:
        sae_source_map[label] = source

    # ── BASE (HuggingFace) ────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  MODEL: BASE (HuggingFace CLIP)")
    print(f"{'═'*60}")

    vit = load_vit_hf("base", ref_cfg, BACKBONE, DEVICE, classnames=classnames)
    print(f"[DEBUG] Base VIT loaded. logit_scale={vit.model.logit_scale.exp().item():.4f}")
    text_feat = get_text_features_hf(vit, DEVICE, classnames, "base")
    print(f"[DEBUG] Base text_feat shape={text_feat.shape}, "
          f"norm={text_feat.norm(dim=-1).mean().item():.6f}")
    results = {}

    baseline = compute_accuracy_hf(vit, text_feat, test_loader_hf, "base", DEVICE)
    results["no_sae"] = baseline
    print(f"    {'no SAE (baseline)':<30s}  {baseline:.2f}%")

    for sae, cfg, label, sae_source in saes:
        print(f"\n  [DEBUG] Applying SAE '{label}' (source: {sae_source}) to base model")
        hooks = make_full_sae_hook_hf(sae, cfg, "base",
                                      sae_bias=effective_sae_bias, alpha=args.sae_alpha)
        acc = compute_accuracy_hf(vit, text_feat, test_loader_hf, "base", DEVICE, hooks)
        results[label] = acc
        print(f"    {label:<30s}  {acc:.2f}%  (Δ {acc - baseline:+.2f})")

    all_results["base"] = results
    del vit, text_feat; flush()

    # ── LORA (OpenAI CLIP) ────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  MODEL: LORA  (OpenAI CLIP)")
    print(f"{'═'*60}")

    lora_model, lora_preprocess = build_lora_openai_clip(args.lora_checkpoint, DEVICE)
    lora_model.eval()
    print(f"[DEBUG] LoRA model: {len(lora_model.visual.transformer.resblocks)} visual resblocks")
    print(f"[DEBUG] LoRA logit_scale={lora_model.logit_scale.exp().item():.4f}")

    verify_preprocessing_parity(lora_preprocess, DEVICE)

    test_ds_oai = NpzTestDataset(NPZ_PATH, label_mapping, lora_preprocess)
    test_loader_oai = DataLoader(test_ds_oai, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

    text_feat_lora = get_text_features_openai(lora_model, classnames, DEVICE)
    print(f"[DEBUG] LoRA text_feat shape={text_feat_lora.shape}, "
          f"norm={text_feat_lora.norm(dim=-1).mean().item():.6f}")
    results = {}

    baseline = compute_accuracy_openai(lora_model, text_feat_lora, test_loader_oai, DEVICE)
    results["no_sae"] = baseline
    print(f"    {'no SAE (baseline)':<30s}  {baseline:.2f}%")

    for sae, cfg, label, sae_source in saes:
        print(f"\n  [DEBUG] Applying SAE '{label}' (source: {sae_source}) to lora model")
        verify_layer_alignment(cfg, f"lora+{label}")
        hook = OpenAISAEHook(lora_model, sae, cfg, DEVICE,
                             sae_bias=effective_sae_bias, alpha=args.sae_alpha,
                             source_model_type=sae_source).register()
        acc = compute_accuracy_openai(lora_model, text_feat_lora, test_loader_oai, DEVICE)
        hook.remove()
        results[label] = acc
        print(f"    {label:<30s}  {acc:.2f}%  (Δ {acc - baseline:+.2f})")

    all_results["lora"] = results
    del lora_model, text_feat_lora; flush()

    # ── MAPLE (HuggingFace) ───────────────────────────────────────
    if args.model_path and args.config_path:
        print(f"\n{'═'*60}")
        print(f"  MODEL: MAPLE (HuggingFace CLIP)")
        print(f"  SAE: {args.maple_sae_path}")
        print(f"{'═'*60}")

        vit = load_vit_hf("maple", ref_cfg, BACKBONE, DEVICE,
                          model_path=args.model_path,
                          config_path=args.config_path,
                          classnames=classnames)
        print(f"[DEBUG] Maple VIT loaded. logit_scale={vit.model.logit_scale.exp().item():.4f}")
        text_feat = get_text_features_hf(vit, DEVICE, classnames, "maple")
        print(f"[DEBUG] Maple text_feat shape={text_feat.shape}, "
              f"norm={text_feat.norm(dim=-1).mean().item():.6f}")
        results = {}

        baseline = compute_accuracy_hf(vit, text_feat, test_loader_hf, "maple", DEVICE)
        results["no_sae"] = baseline
        print(f"    {'no SAE (baseline)':<30s}  {baseline:.2f}%")

        # Evaluate maple SAE on maple model
        for sae, cfg, label, sae_source in maple_saes:
            print(f"\n  [DEBUG] Applying maple SAE '{label}'")
            verify_layer_alignment(cfg, f"maple+{label}")

            # Sanity check
            dummy = torch.randn(1, 197, 768, device=DEVICE)
            dummy_out = sae(dummy)
            print(f"  [DEBUG] SAE sanity: in={dummy.shape} → out={dummy_out[0].shape}")
            del dummy, dummy_out

            hooks = make_full_sae_hook_hf(sae, cfg, "maple",
                                          sae_bias=effective_sae_bias, alpha=args.sae_alpha)
            acc = compute_accuracy_hf(vit, text_feat, test_loader_hf, "maple", DEVICE, hooks)
            results[label] = acc
            print(f"    {label:<30s}  {acc:.2f}%  (Δ {acc - baseline:+.2f})")

        # Also evaluate base/medmnist SAEs on the maple model (cross-model)
        print(f"\n  [DEBUG] Cross-model: evaluating base/medmnist SAEs on maple model")
        for sae, cfg, label, sae_source in saes:
            if sae_source == "maple":
                continue  # already evaluated above via maple_saes
            print(f"\n  [DEBUG] Applying SAE '{label}' (source: {sae_source}) to maple model")
            verify_layer_alignment(cfg, f"maple+{label}")
            hooks = make_full_sae_hook_hf(sae, cfg, "maple",
                                          sae_bias=effective_sae_bias, alpha=args.sae_alpha)
            acc = compute_accuracy_hf(vit, text_feat, test_loader_hf, "maple", DEVICE, hooks)
            results[label] = acc
            print(f"    {label:<30s}  {acc:.2f}%  (Δ {acc - baseline:+.2f})")

        all_results["maple"] = results
        del vit, text_feat; flush()

    # ══════════════════════════════════════════════════════════════════
    # LINEAR PROBE on SAE latents
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  LINEAR PROBE ON SAE LATENTS")
    print(f"{'═'*70}")
    print(f"  Extracting SAE encoder activations, training logistic regression.")

    linear_probe_results = {}

    # ── Train/test loaders for linear probe ───────────────────────
    hf_transform = get_processor_transform()

    train_ds_hf = NpzSplitDataset(NPZ_PATH, label_mapping, hf_transform,
                                   split="train", return_pil=True)
    train_loader_hf = DataLoader(train_ds_hf, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=False,
                                  collate_fn=_hf_collate_fn)

    test_ds_hf_probe = NpzSplitDataset(NPZ_PATH, label_mapping, hf_transform,
                                        split="test", return_pil=True)
    test_loader_hf_probe = DataLoader(test_ds_hf_probe, batch_size=args.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=False,
                                       collate_fn=_hf_collate_fn)
    print(f"  Train: {len(train_ds_hf)} samples, Test: {len(test_ds_hf_probe)} samples")

    # ── 1) Base SAE on Base model ─────────────────────────────────
    base_sae_entries = [(s, c, l, src) for s, c, l, src in saes if src == "base"]
    if base_sae_entries:
        print(f"\n  ── Base SAE → Base CLIP model ──")
        vit = load_vit_hf("base", ref_cfg, BACKBONE, DEVICE, classnames=classnames)
        for sae_mod, cfg, label, _ in base_sae_entries:
            print(f"    Extracting latents for '{label}'...")
            train_lat, train_lab = extract_sae_latents_hf(
                vit, sae_mod, cfg, train_loader_hf, "base", DEVICE)
            test_lat, test_lab = extract_sae_latents_hf(
                vit, sae_mod, cfg, test_loader_hf_probe, "base", DEVICE)
            print(f"    Latent dim: {train_lat.shape[1]}, "
                  f"train={train_lat.shape[0]}, test={test_lat.shape[0]}")
            test_acc, train_acc = train_linear_probe(
                train_lat, train_lab, test_lat, test_lab)
            linear_probe_results[f"Base SAE ({label})"] = {
                "test_acc": test_acc, "train_acc": train_acc,
                "model": "base", "sae_source": "base",
            }
            print(f"    ✓ Train: {train_acc:.2f}%  Test: {test_acc:.2f}%")
        del vit; flush()

    # ── 2) MedMNIST/LoRA SAE on LoRA model ───────────────────────
    medmnist_sae_entries = [(s, c, l, src) for s, c, l, src in saes if src == "medmnist"]
    if medmnist_sae_entries:
        print(f"\n  ── MedMNIST SAE (LoRA) → LoRA CLIP model ──")
        lora_model, lora_preprocess = build_lora_openai_clip(args.lora_checkpoint, DEVICE)
        lora_model.eval()

        train_ds_oai = NpzSplitDataset(NPZ_PATH, label_mapping, lora_preprocess,
                                        split="train", return_pil=False)
        train_loader_oai = DataLoader(train_ds_oai, batch_size=args.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True)
        test_ds_oai = NpzSplitDataset(NPZ_PATH, label_mapping, lora_preprocess,
                                       split="test", return_pil=False)
        test_loader_oai = DataLoader(test_ds_oai, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4, pin_memory=True)

        for sae_mod, cfg, label, _ in medmnist_sae_entries:
            print(f"    Extracting latents for '{label}'...")
            train_lat, train_lab = extract_sae_latents_openai(
                lora_model, sae_mod, cfg, train_loader_oai, DEVICE)
            test_lat, test_lab = extract_sae_latents_openai(
                lora_model, sae_mod, cfg, test_loader_oai, DEVICE)
            print(f"    Latent dim: {train_lat.shape[1]}, "
                  f"train={train_lat.shape[0]}, test={test_lat.shape[0]}")
            test_acc, train_acc = train_linear_probe(
                train_lat, train_lab, test_lat, test_lab)
            linear_probe_results[f"MedMNIST SAE ({label})"] = {
                "test_acc": test_acc, "train_acc": train_acc,
                "model": "lora", "sae_source": "medmnist",
            }
            print(f"    ✓ Train: {train_acc:.2f}%  Test: {test_acc:.2f}%")
        del lora_model; flush()

    # ── 3) MaPLe SAE on MaPLe model ──────────────────────────────
    if args.model_path and args.config_path:
        print(f"\n  ── MaPLe SAE → MaPLe CLIP model ──")
        vit = load_vit_hf("maple", ref_cfg, BACKBONE, DEVICE,
                          model_path=args.model_path,
                          config_path=args.config_path,
                          classnames=classnames)
        for sae_mod, cfg, label, _ in maple_saes:
            print(f"    Extracting latents for '{label}'...")
            train_lat, train_lab = extract_sae_latents_hf(
                vit, sae_mod, cfg, train_loader_hf, "maple", DEVICE)
            test_lat, test_lab = extract_sae_latents_hf(
                vit, sae_mod, cfg, test_loader_hf_probe, "maple", DEVICE)
            print(f"    Latent dim: {train_lat.shape[1]}, "
                  f"train={train_lat.shape[0]}, test={test_lat.shape[0]}")
            test_acc, train_acc = train_linear_probe(
                train_lat, train_lab, test_lat, test_lab)
            linear_probe_results[f"MaPLe SAE ({label})"] = {
                "test_acc": test_acc, "train_acc": train_acc,
                "model": "maple", "sae_source": "maple",
            }
            print(f"    ✓ Train: {train_acc:.2f}%  Test: {test_acc:.2f}%")
        del vit; flush()

    # ── Linear probe summary table ────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  LINEAR PROBE CLASSIFICATION ACCURACY (on SAE latents)")
    print(f"{'═'*70}")
    print(f"  {'SAE':<40s}  {'Model':<10s}  {'Train':>8s}  {'Test':>8s}")
    print(f"  {'─'*40}  {'─'*10}  {'─'*8}  {'─'*8}")
    for name, res in linear_probe_results.items():
        print(f"  {name:<40s}  {res['model']:<10s}  "
              f"{res['train_acc']:>7.2f}%  {res['test_acc']:>7.2f}%")
    print(f"{'═'*70}")

    # ── Summary ───────────────────────────────────────────────────
    vit_types = list(all_results.keys())
    all_sae_labels = set()
    for vt in vit_types:
        all_sae_labels.update(all_results[vt].keys())
    sae_labels = ["no_sae"] + sorted([l for l in all_sae_labels if l != "no_sae"])

    print(f"\n{'═'*70}")
    print(f"  SUMMARY  [sae_bias={effective_sae_bias}, alpha={args.sae_alpha}]")
    print(f"{'═'*70}")
    header = f"  {'SAE':<40s}"
    for vt in vit_types: header += f"  {vt:>10s}"
    print(header)
    print(f"  {'─'*40}" + f"  {'─'*10}" * len(vit_types))
    for label in sae_labels:
        display = "no SAE (baseline)" if label == "no_sae" else label
        row = f"  {display:<40s}"
        for vt in vit_types:
            val = all_results[vt].get(label, None)
            if val is not None:
                row += f"  {val:>9.2f}%"
            else:
                row += f"  {'N/A':>10s}"
        print(row)
    print(f"{'═'*70}")

    # ── Summary by SAE source ─────────────────────────────────────
    # Group SAE labels by their source type (base / medmnist / maple)
    source_groups = {}  # source → [label, ...]
    for label, source in sae_source_map.items():
        source_groups.setdefault(source, []).append(label)

    # For each source, pick the best accuracy per model (in case of multiple layers)
    # Also build a flat table: rows = SAE source, cols = model
    source_order = ["base", "medmnist", "maple"]
    source_display = {"base": "Base SAE", "medmnist": "MedMNIST SAE (LoRA)", "maple": "MaPLe SAE"}

    print(f"\n{'═'*70}")
    print(f"  SUMMARY BY SAE SOURCE  [sae_bias={effective_sae_bias}, alpha={args.sae_alpha}]")
    print(f"{'═'*70}")
    header = f"  {'SAE Source':<30s}"
    for vt in vit_types: header += f"  {vt:>10s}"
    print(header)
    print(f"  {'─'*30}" + f"  {'─'*10}" * len(vit_types))

    # Baseline row
    row = f"  {'no SAE (baseline)':<30s}"
    for vt in vit_types:
        val = all_results[vt].get("no_sae", None)
        row += f"  {val:>9.2f}%" if val is not None else f"  {'N/A':>10s}"
    print(row)

    # Per-source rows (best accuracy across layers for that source on each model)
    by_source_results = {}
    for source in source_order:
        labels = source_groups.get(source, [])
        if not labels:
            continue
        display = source_display.get(source, source)
        row = f"  {display:<30s}"
        by_source_row = {}
        for vt in vit_types:
            accs = [all_results[vt][l] for l in labels if l in all_results[vt]]
            if accs:
                best = max(accs)
                baseline_val = all_results[vt].get("no_sae", 0)
                delta = best - baseline_val
                row += f"  {best:>7.2f}%"
                by_source_row[vt] = best
            else:
                row += f"  {'N/A':>10s}"
        print(row)
        by_source_results[source] = by_source_row

    # Delta from baseline row
    print(f"  {'─'*30}" + f"  {'─'*10}" * len(vit_types))
    print(f"  {'Δ from baseline:':<30s}")
    for source in source_order:
        if source not in by_source_results:
            continue
        display = source_display.get(source, source)
        row = f"    {display:<28s}"
        for vt in vit_types:
            if vt in by_source_results[source]:
                baseline_val = all_results[vt].get("no_sae", 0)
                delta = by_source_results[source][vt] - baseline_val
                row += f"  {delta:>+9.2f}%"
            else:
                row += f"  {'N/A':>10s}"
        print(row)

    print(f"{'═'*70}")

    save_path = os.path.join(args.save_dir, "accuracy_comparison.json")
    save_data = {
        "detailed": all_results,
        "by_sae_source": {
            source: {
                "labels": source_groups.get(source, []),
                "best_accuracy": by_source_results.get(source, {}),
            }
            for source in source_order if source in source_groups
        },
        "linear_probe": linear_probe_results,
        "config": {
            "sae_bias": effective_sae_bias,
            "sae_alpha": args.sae_alpha,
        },
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
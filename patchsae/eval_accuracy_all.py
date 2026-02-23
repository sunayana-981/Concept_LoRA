#!/usr/bin/env python3
"""
Accuracy evaluation: base / lora / maple × each SAE in one run.

Key design:
  - base & maple: use HookedVisionTransformer (HuggingFace CLIP) as before
  - lora: use OpenAI CLIP directly (since LoRA was trained on it)
    with manual PyTorch hooks for SAE intervention

Usage:
    python eval_accuracy_all.py --include_base
"""

import argparse, json, os, sys, glob, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
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
    print(f"[FATAL] {e}\nRun from patchsae root."); sys.exit(1)

try:
    import clip as openai_clip
    HAS_OPENAI_CLIP = True
except ImportError:
    HAS_OPENAI_CLIP = False

# ── Config ────────────────────────────────────────────────────────────────
BASE_SAE_PATH    = "data/sae_weight/base/out.pt"
MEDMNIST_SAE_DIR = "out/checkpoints/medmnist"
LORA_CHECKPOINT  = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
TRAIN_ROOT       = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
NPZ_PATH         = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
CLASSNAMES_PATH  = "configs/classnames/medmnist_classnames.json"
BACKBONE         = "openai/clip-vit-base-patch16"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = 64
SAE_BIAS         = -0.105131256516992

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

# ── Helpers ───────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def get_transform():
    """Full CLIP preprocessing (with normalize). For tensors fed directly to model."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_processor_transform():
    """Minimal transform for images going through CLIPProcessor.

    Only Resize + ToTensor (no Normalize).  The CLIPProcessor will
    normalise. This avoids the double-normalisation / PIL clipping bug.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def get_imagefolder_classnames(root):
    ds = datasets.ImageFolder(root=root)
    idx_to_name = {i: n.replace('_', ' ') for n, i in ds.class_to_idx.items()}
    return [idx_to_name[i] for i in range(len(idx_to_name))]

def build_npz_to_if_mapping(root):
    ds = datasets.ImageFolder(root=root)
    if_map = {n.replace('_', ' ').lower(): i for n, i in ds.class_to_idx.items()}
    return {i: if_map.get(c.lower(), i) for i, c in enumerate(MEDMNIST_CLASSES)}

class NpzTestDataset(Dataset):
    def __init__(self, npz_path, mapping, transform):
        data = np.load(npz_path)
        self.imgs = data['test_images']
        self.labels = data['test_labels'].flatten()
        self.mapping = mapping
        self.transform = transform
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.imgs[i])), self.mapping[int(self.labels[i])]

def discover_medmnist_saes(base_dir):
    paths = sorted(glob.glob(os.path.join(base_dir, "*/final*/*.pt")))
    best = {}
    for p in paths:
        ckpt = torch.load(p, map_location="cpu")
        cfg_d = ckpt.get("cfg", ckpt.get("config"))
        layer = getattr(cfg_d, "block_layer",
                        cfg_d.get("block_layer", "?") if isinstance(cfg_d, dict) else "?")
        best[layer] = p
        del ckpt; gc.collect()
    return list(best.values())


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
        # Already merged state dict
        model.load_state_dict(lora_state)
        return model, preprocess

    layers, meta = lora_state["weights"], lora_state["metadata"]
    scale = meta["alpha"] / meta["r"]
    print(f"    LoRA: rank={meta['r']}, alpha={meta['alpha']}, scale={scale}")

    with torch.no_grad():
        for i in range(12):
            key = f"layer_{i}"
            if key not in layers: continue
            ld = layers[key]
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj",0),("k_proj",d),("v_proj",2*d)]:
                A, B = _extract_lora_AB(ld, proj)
                if A is None: continue
                w[off:off+d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)

        for i in range(12, 24):
            key = f"layer_{i}"
            if key not in layers: continue
            ld = layers[key]
            w = model.visual.transformer.resblocks[i-12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj",0),("k_proj",d),("v_proj",2*d)]:
                A, B = _extract_lora_AB(ld, proj)
                if A is None: continue
                w[off:off+d] += (scale * (B.float().to(device) @ A.float().to(device))).to(w.dtype)

    print("    LoRA merged.")
    return model, preprocess


# ══════════════════════════════════════════════════════════════════════════
# LORA: SAE hooks for OpenAI CLIP (manual PyTorch hooks)
# ══════════════════════════════════════════════════════════════════════════

class OpenAISAEHook:
    """
    Register a forward hook on an OpenAI CLIP resblock that passes
    the residual stream through an SAE.

    OpenAI CLIP resblocks:
      visual.transformer.resblocks[i]
    The residual stream after resblock i is the output of the block.

    block_layer in the SAE config uses negative indexing:
      -1 = last block (11), -2 = block 10, -3 = block 9, etc.
    """
    def __init__(self, model, sae, cfg, device):
        self.sae = sae
        self.cfg = cfg
        self.device = device
        self.handle = None

        num_blocks = len(model.visual.transformer.resblocks)
        layer = cfg.block_layer
        if layer < 0:
            layer = num_blocks + layer
        self.block = model.visual.transformer.resblocks[layer]
        self.layer_idx = layer

    def _hook_fn(self, module, input, output):
        """
        OpenAI CLIP resblock output shape: [seq_len, batch, d_model]
        SAE expects: [batch, seq_len, d_model]
        """
        # output is the residual stream tensor
        act = output  # [seq_len, batch, d_model]
        orig_dtype = act.dtype

        # Transpose to [batch, seq_len, d_model] for SAE
        act_bsd = act.transpose(0, 1).float()

        # Run through SAE
        reconstructed = self.sae(act_bsd)[0] - SAE_BIAS

        # Transpose back to [seq_len, batch, d_model]
        result = reconstructed.transpose(0, 1).to(orig_dtype)
        return result

    def register(self):
        self.handle = self.block.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ══════════════════════════════════════════════════════════════════════════
# LORA: Accuracy computation using OpenAI CLIP
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_accuracy_openai(model, text_features, loader, device):
    correct, total = 0, 0
    for imgs, labels in tqdm(loader, desc="      eval", leave=False):
        img_feat = model.encode_image(imgs.to(device))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims = img_feat.float() @ text_features.float().t()
        preds = sims.argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100


@torch.no_grad()
def get_text_features_openai(model, classnames, device):
    """Text features using mean of 80 prompt templates (same as base/lora HF path)."""
    mean_f = torch.zeros(len(classnames), 512, device=device)
    for tmpl in openai_imagenet_template:
        prompts = [tmpl(c) for c in classnames]
        tokens = openai_clip.tokenize(prompts).to(device)
        f = model.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
        mean_f += f
    mean_f /= len(openai_imagenet_template)
    return mean_f / mean_f.norm(dim=-1, keepdim=True)


# ══════════════════════════════════════════════════════════════════════════
# HF PATH: for base and maple (unchanged)
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
    if vit_type == "maple":
        tf = vit.model.get_text_features()
        return tf / tf.norm(dim=-1, keepdim=True)
    mean_f = 0
    for tmpl in openai_imagenet_template:
        ids = [vit.processor(text=tmpl(c), return_tensors="pt", padding=False,
                             truncation=True).input_ids[0] for c in classnames]
        padded = pad_sequence(ids, batch_first=True).to(device)
        f = vit.model.get_text_features(padded)
        f = f / f.norm(dim=-1, keepdim=True)
        mean_f += f
    mean_f /= len(openai_imagenet_template)
    return mean_f / mean_f.norm(dim=-1, keepdim=True)

def make_full_sae_hook_hf(sae, cfg, vit_type):
    if vit_type == "maple":
        def fn(act):
            orig_dtype = act.dtype
            a = act.transpose(0, 1).float()
            a = sae(a)[0] - SAE_BIAS
            return a.transpose(0, 1).to(orig_dtype)
        return [Hook(cfg.block_layer, cfg.module_name, fn,
                     return_module_output=False, is_custom=True)]
    else:
        def fn(act):
            act[:, :, :] = sae(act[:, :, :])[0] - SAE_BIAS
            return (act,)
        return [Hook(cfg.block_layer, cfg.module_name, fn,
                     return_module_output=False, is_custom=False)]

@torch.no_grad()
def compute_accuracy_hf(vit, text_features, loader, vit_type, device, hooks=None):
    correct, total = 0, 0
    to_pil = transforms.ToPILImage()
    for images, labels in tqdm(loader, desc="      eval", leave=False):
        inputs = vit.processor(images=[to_pil(img) for img in images],
                               text="", return_tensors="pt", padding=True).to(device)
        if hooks:
            out = vit.run_with_hooks(hooks, return_type="output", **inputs)
        else:
            out = vit(return_type="output", **inputs)
        img_feat = out if vit_type == "maple" else out.image_embeds
        logits = vit.model.logit_scale.exp() * img_feat @ text_features.t()
        preds = logits.argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100


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
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--save_dir", default="out/accuracy_comparison")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    classnames = get_imagefolder_classnames(TRAIN_ROOT)
    label_mapping = build_npz_to_if_mapping(TRAIN_ROOT)

    print(f"Classnames (ImageFolder order):")
    for i, c in enumerate(classnames): print(f"  {i}: {c}")

    # Load SAEs
    print(f"\nLoading SAEs...")
    med_paths = discover_medmnist_saes(MEDMNIST_SAE_DIR)
    saes = []
    if args.include_base and os.path.exists(BASE_SAE_PATH):
        sae, cfg = load_sae(BASE_SAE_PATH, DEVICE)
        saes.append((sae, cfg, f"base_layer{cfg.block_layer}"))
    for path in med_paths:
        sae, cfg = load_sae(path, DEVICE)
        saes.append((sae, cfg, f"medmnist_layer{cfg.block_layer}"))
    if not saes:
        print("[FATAL] No SAEs."); sys.exit(1)
    print(f"  {len(saes)} SAE(s) loaded.")

    # Test loaders — two versions:
    #   OpenAI CLIP uses its own preprocess (from clip.load)
    #   HF uses processor_transform (Resize+ToTensor only, NO Normalize)
    #   because the eval loop converts tensors to PIL and feeds them to
    #   CLIPProcessor which handles normalisation.  Using get_transform()
    #   would double-normalise via the lossy ToPILImage round-trip.
    hf_transform = get_processor_transform()
    test_ds_hf = NpzTestDataset(NPZ_PATH, label_mapping, hf_transform)
    test_loader_hf = DataLoader(test_ds_hf, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

    ref_cfg = saes[0][1]
    all_results = {}

    # ── BASE (HuggingFace) ────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  MODEL: BASE")
    print(f"{'═'*60}")

    vit = load_vit_hf("base", ref_cfg, BACKBONE, DEVICE, classnames=classnames)
    text_feat = get_text_features_hf(vit, DEVICE, classnames, "base")
    results = {}

    baseline = compute_accuracy_hf(vit, text_feat, test_loader_hf, "base", DEVICE)
    results["no_sae"] = baseline
    print(f"    {'no SAE (baseline)':<30s}  {baseline:.2f}%")

    for sae, cfg, label in saes:
        hooks = make_full_sae_hook_hf(sae, cfg, "base")
        acc = compute_accuracy_hf(vit, text_feat, test_loader_hf, "base", DEVICE, hooks)
        results[label] = acc
        print(f"    {label:<30s}  {acc:.2f}%  (Δ {acc - baseline:+.2f})")

    all_results["base"] = results
    del vit, text_feat; flush()

    # ── LORA (OpenAI CLIP directly) ───────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  MODEL: LORA  (OpenAI CLIP)")
    print(f"{'═'*60}")

    lora_model, lora_preprocess = build_lora_openai_clip(args.lora_checkpoint, DEVICE)
    lora_model.eval()

    # Test dataset with OpenAI CLIP's own preprocessing
    test_ds_oai = NpzTestDataset(NPZ_PATH, label_mapping, lora_preprocess)
    test_loader_oai = DataLoader(test_ds_oai, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

    text_feat_lora = get_text_features_openai(lora_model, classnames, DEVICE)
    results = {}

    baseline = compute_accuracy_openai(lora_model, text_feat_lora, test_loader_oai, DEVICE)
    results["no_sae"] = baseline
    print(f"    {'no SAE (baseline)':<30s}  {baseline:.2f}%")

    for sae, cfg, label in saes:
        hook = OpenAISAEHook(lora_model, sae, cfg, DEVICE).register()
        acc = compute_accuracy_openai(lora_model, text_feat_lora, test_loader_oai, DEVICE)
        hook.remove()
        results[label] = acc
        print(f"    {label:<30s}  {acc:.2f}%  (Δ {acc - baseline:+.2f})")

    all_results["lora"] = results
    del lora_model, text_feat_lora; flush()

    # ── MAPLE (HuggingFace) ───────────────────────────────────────
    if args.model_path and args.config_path:
        print(f"\n{'═'*60}")
        print(f"  MODEL: MAPLE")
        print(f"{'═'*60}")

        vit = load_vit_hf("maple", ref_cfg, BACKBONE, DEVICE,
                          model_path=args.model_path,
                          config_path=args.config_path,
                          classnames=classnames)
        text_feat = get_text_features_hf(vit, DEVICE, classnames, "maple")
        results = {}

        baseline = compute_accuracy_hf(vit, text_feat, test_loader_hf, "maple", DEVICE)
        results["no_sae"] = baseline
        print(f"    {'no SAE (baseline)':<30s}  {baseline:.2f}%")

        for sae, cfg, label in saes:
            hooks = make_full_sae_hook_hf(sae, cfg, "maple")
            acc = compute_accuracy_hf(vit, text_feat, test_loader_hf, "maple", DEVICE, hooks)
            results[label] = acc
            print(f"    {label:<30s}  {acc:.2f}%  (Δ {acc - baseline:+.2f})")

        all_results["maple"] = results
        del vit, text_feat; flush()

    # ── Summary ───────────────────────────────────────────────────
    vit_types = list(all_results.keys())
    sae_labels = ["no_sae"] + [l for _, _, l in saes]

    print(f"\n{'═'*70}")
    print(f"  SUMMARY: Accuracy (%)")
    print(f"{'═'*70}")
    header = f"  {'SAE':<30s}"
    for vt in vit_types: header += f"  {vt:>10s}"
    print(header)
    print(f"  {'─'*30}" + f"  {'─'*10}" * len(vit_types))
    for label in sae_labels:
        display = "no SAE (baseline)" if label == "no_sae" else label
        row = f"  {display:<30s}"
        for vt in vit_types:
            row += f"  {all_results[vt].get(label, float('nan')):>9.2f}%"
        print(row)
    print(f"{'═'*70}")

    save_path = os.path.join(args.save_dir, "accuracy_comparison.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
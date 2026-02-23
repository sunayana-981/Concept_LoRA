#!/usr/bin/env python3
"""
Diagnostic script for MaPLe accuracy issues.
Run from patchsae root:  python diagnose_maple.py
"""

import os, sys, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

try:
    from tasks.utils import load_sae, load_hooked_vit, SAE_DIM
    from src.sae_training.hooked_vit import Hook
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] {e}\nRun from patchsae root."); sys.exit(1)

# ── Config (match your eval script) ──────────────────────────────────────
TRAIN_ROOT   = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
NPZ_PATH     = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
BACKBONE     = "openai/clip-vit-base-patch16"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH   = "/home/sunayana/Documents/model.pth.tar-5"
CONFIG_PATH  = "/home/sunayana/Documents/Concept_LoRA/patchsae/configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml"
BASE_SAE_PATH = "data/sae_weight/base/out.pt"

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

# ── Helpers ───────────────────────────────────────────────────────────────

def sep(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")

def get_transform():
    """Minimal transform for images going through CLIPProcessor.

    Only Resize + ToTensor (no Normalize).  The CLIPProcessor handles
    normalisation.  This avoids the double-normalisation / PIL clipping bug.
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


# ══════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKS
# ══════════════════════════════════════════════════════════════════════════

def main():
    classnames = get_imagefolder_classnames(TRAIN_ROOT)
    label_mapping = build_npz_to_if_mapping(TRAIN_ROOT)

    # ── CHECK 1: Class names and label mapping ─────────────────────
    sep("CHECK 1: Classnames & Label Mapping")
    print(f"  ImageFolder classnames ({len(classnames)}):")
    for i, c in enumerate(classnames):
        print(f"    {i}: '{c}'")

    print(f"\n  MEDMNIST_CLASSES ({len(MEDMNIST_CLASSES)}):")
    for i, c in enumerate(MEDMNIST_CLASSES):
        print(f"    {i}: '{c}'")

    print(f"\n  NPZ→IF label mapping:")
    for npz_idx, if_idx in sorted(label_mapping.items()):
        mapped_name = classnames[if_idx] if if_idx < len(classnames) else "OUT_OF_RANGE"
        orig_name = MEDMNIST_CLASSES[npz_idx]
        match = "✓" if orig_name.lower() == mapped_name.lower() else "✗ MISMATCH"
        print(f"    NPZ {npz_idx} ('{orig_name}') → IF {if_idx} ('{mapped_name}')  {match}")

    # Check label distribution in test set
    data = np.load(NPZ_PATH)
    test_labels = data['test_labels'].flatten()
    print(f"\n  Test set label distribution (raw NPZ):")
    for lbl in sorted(set(test_labels)):
        count = (test_labels == lbl).sum()
        print(f"    label {lbl}: {count} samples ({count/len(test_labels)*100:.1f}%)")

    # ── CHECK 2: MaPLe checkpoint ──────────────────────────────────
    sep("CHECK 2: MaPLe Checkpoint")
    print(f"  Path: {MODEL_PATH}")
    print(f"  Exists: {os.path.exists(MODEL_PATH)}")
    if not os.path.exists(MODEL_PATH):
        print("  [FATAL] Checkpoint not found! Stopping.")
        return

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    print(f"  Type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"  Top-level keys: {list(ckpt.keys())}")
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            print(f"  state_dict has {len(sd)} keys")
            print(f"  First 20 keys:")
            for i, k in enumerate(list(sd.keys())[:20]):
                v = sd[k]
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}, "
                      f"mean={v.float().mean():.6f}, std={v.float().std():.6f}")

            # Look for prompt-related keys
            prompt_keys = [k for k in sd.keys() if any(
                w in k.lower() for w in ["prompt", "ctx", "compound", "proj"]
            )]
            print(f"\n  Prompt-related keys ({len(prompt_keys)}):")
            for k in prompt_keys:
                v = sd[k]
                print(f"    {k}: shape={v.shape}, mean={v.float().mean():.6f}, "
                      f"std={v.float().std():.6f}, all_zero={torch.all(v==0).item()}")
        if "epoch" in ckpt:
            print(f"  Epoch: {ckpt['epoch']}")
    del ckpt; gc.collect()

    # ── CHECK 3: Config file ──────────────────────────────────────
    sep("CHECK 3: MaPLe Config")
    print(f"  Path: {CONFIG_PATH}")
    print(f"  Exists: {os.path.exists(CONFIG_PATH)}")
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            content = f.read()
        print(f"  Contents:\n{'─'*40}")
        print(content)
        print(f"{'─'*40}")

    # ── CHECK 4: Load SAE to get ref_cfg ──────────────────────────
    sep("CHECK 4: Load ref SAE for config")
    if os.path.exists(BASE_SAE_PATH):
        sae, ref_cfg = load_sae(BASE_SAE_PATH, DEVICE)
        print(f"  SAE loaded. ref_cfg type: {type(ref_cfg)}")
        print(f"  ref_cfg attrs: {vars(ref_cfg) if hasattr(ref_cfg, '__dict__') else ref_cfg}")
    else:
        print("  [WARN] Base SAE not found, trying medmnist SAEs...")
        import glob
        med_paths = sorted(glob.glob("out/checkpoints/medmnist/*/final*/*.pt"))
        if med_paths:
            sae, ref_cfg = load_sae(med_paths[0], DEVICE)
            print(f"  SAE loaded from {med_paths[0]}")
        else:
            print("  [FATAL] No SAE found for ref_cfg"); return

    # ── CHECK 5: Load MaPLe model ─────────────────────────────────
    sep("CHECK 5: Load MaPLe HookedVisionTransformer")
    try:
        vit = load_hooked_vit(
            ref_cfg, "maple", BACKBONE, DEVICE,
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            classnames=classnames,
        )
        vit.eval()
        print("  ✓ MaPLe model loaded successfully")
    except Exception as e:
        print(f"  [FATAL] Failed to load MaPLe: {e}")
        import traceback; traceback.print_exc()
        return

    # ── CHECK 5a: Inspect model structure ─────────────────────────
    sep("CHECK 5a: Model Structure")
    print(f"  vit type: {type(vit)}")
    print(f"  vit.model type: {type(vit.model)}")

    # Check for classnames stored on model
    for attr in ["classnames", "class_names", "classes", "num_classes",
                 "prompt_learner", "text_encoder", "image_encoder",
                 "compound_prompts_text", "compound_prompts_vision"]:
        if hasattr(vit.model, attr):
            val = getattr(vit.model, attr)
            if isinstance(val, (list, tuple)):
                print(f"  vit.model.{attr}: {val}")
            elif isinstance(val, torch.nn.Module):
                print(f"  vit.model.{attr}: {type(val).__name__}")
                # Print submodule params
                for n, p in val.named_parameters():
                    print(f"    .{n}: shape={p.shape}, requires_grad={p.requires_grad}, "
                          f"mean={p.float().mean():.6f}, std={p.float().std():.6f}")
            elif isinstance(val, torch.Tensor):
                print(f"  vit.model.{attr}: shape={val.shape}, "
                      f"mean={val.float().mean():.6f}")
            else:
                print(f"  vit.model.{attr}: {val}")

    # Check what methods are available
    custom_methods = [m for m in dir(vit.model) if not m.startswith('_')
                      and callable(getattr(vit.model, m, None))
                      and m not in dir(torch.nn.Module)]
    print(f"\n  Custom methods on vit.model: {custom_methods[:30]}")

    # ── CHECK 6: Text features ────────────────────────────────────
    sep("CHECK 6: Text Features")

    # MaPLe path: get_text_features()
    print("  Calling vit.model.get_text_features()...")
    try:
        tf_maple = vit.model.get_text_features()
        print(f"    Shape: {tf_maple.shape}")
        print(f"    Dtype: {tf_maple.dtype}")
        print(f"    Norms: {tf_maple.norm(dim=-1)}")
        print(f"    Mean per row: {tf_maple.mean(dim=-1)}")
        print(f"    Any NaN: {torch.any(torch.isnan(tf_maple)).item()}")
        print(f"    Any Inf: {torch.any(torch.isinf(tf_maple)).item()}")

        tf_maple_normed = tf_maple / tf_maple.norm(dim=-1, keepdim=True)
        # Cosine similarity matrix between text features
        sim = tf_maple_normed @ tf_maple_normed.t()
        print(f"\n    Text feature self-similarity matrix:")
        print(f"    (should be ~1.0 on diagonal, <1.0 off-diagonal)")
        for i in range(sim.shape[0]):
            row_str = "    " + " ".join(f"{sim[i,j]:.3f}" for j in range(sim.shape[1]))
            print(row_str)

        # Check if all text features are identical (common failure mode)
        all_same = all(
            torch.allclose(tf_maple_normed[0], tf_maple_normed[i], atol=1e-3)
            for i in range(1, tf_maple_normed.shape[0])
        )
        if all_same:
            print("    ⚠ WARNING: All text features are nearly identical!")
            print("      This means the prompt learner is not differentiating classes.")
        else:
            print("    ✓ Text features are distinct across classes.")

    except Exception as e:
        print(f"    [ERROR] get_text_features() failed: {e}")
        import traceback; traceback.print_exc()

    # Also try the template-based approach for comparison
    print("\n  Computing template-based text features (80 templates)...")
    try:
        mean_f = 0
        for tmpl in openai_imagenet_template:
            ids = [vit.processor(text=tmpl(c), return_tensors="pt", padding=False,
                                 truncation=True).input_ids[0] for c in classnames]
            padded = pad_sequence(ids, batch_first=True).to(DEVICE)
            f = vit.model.get_text_features(padded)
            f = f / f.norm(dim=-1, keepdim=True)
            mean_f += f
        mean_f /= len(openai_imagenet_template)
        tf_template = mean_f / mean_f.norm(dim=-1, keepdim=True)

        print(f"    Shape: {tf_template.shape}")

        sim_t = tf_template @ tf_template.t()
        print(f"\n    Template text feature self-similarity matrix:")
        for i in range(sim_t.shape[0]):
            row_str = "    " + " ".join(f"{sim_t[i,j]:.3f}" for j in range(sim_t.shape[1]))
            print(row_str)

        # Cross-similarity between maple and template text features
        cross = tf_maple_normed @ tf_template.t()
        print(f"\n    Cross-similarity (maple_row × template_col):")
        for i in range(cross.shape[0]):
            row_str = "    " + " ".join(f"{cross[i,j]:.3f}" for j in range(cross.shape[1]))
            print(row_str)

    except Exception as e:
        print(f"    [ERROR] Template text features failed: {e}")
        import traceback; traceback.print_exc()

    # ── CHECK 7: Image features ───────────────────────────────────
    sep("CHECK 7: Image Forward Pass")

    transform = get_transform()
    test_ds = NpzTestDataset(NPZ_PATH, label_mapping, transform)
    loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    to_pil = transforms.ToPILImage()

    images, labels = next(iter(loader))
    print(f"  Batch: images={images.shape}, labels={labels}")

    inputs = vit.processor(
        images=[to_pil(img) for img in images],
        text="", return_tensors="pt", padding=True
    ).to(DEVICE)
    print(f"  Processor outputs: {list(inputs.keys())}")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

    # Forward pass
    print("\n  Running forward pass...")
    with torch.no_grad():
        try:
            out = vit(return_type="output", **inputs)
            print(f"  Output type: {type(out)}")

            if hasattr(out, '__dict__'):
                for k, v in vars(out).items():
                    if isinstance(v, torch.Tensor):
                        print(f"    .{k}: shape={v.shape}, mean={v.float().mean():.6f}")
            elif isinstance(out, torch.Tensor):
                print(f"    Tensor: shape={out.shape}, mean={out.float().mean():.6f}")
            elif isinstance(out, tuple):
                for i, v in enumerate(out):
                    if isinstance(v, torch.Tensor):
                        print(f"    [{i}]: shape={v.shape}, mean={v.float().mean():.6f}")

            # What your eval script does:
            img_feat = out  # for maple
            if isinstance(img_feat, torch.Tensor):
                print(f"\n  img_feat (raw from model): shape={img_feat.shape}")
                img_feat_normed = img_feat / img_feat.norm(dim=-1, keepdim=True)
                print(f"  img_feat norms: {img_feat.norm(dim=-1)}")
            else:
                # Try image_embeds
                img_feat = getattr(out, 'image_embeds', None)
                if img_feat is not None:
                    print(f"\n  out.image_embeds: shape={img_feat.shape}")
                    img_feat_normed = img_feat / img_feat.norm(dim=-1, keepdim=True)
                else:
                    print("  ⚠ Cannot extract image features from output!")
                    print(f"    Output dir: {[x for x in dir(out) if not x.startswith('_')]}")
                    return

        except Exception as e:
            print(f"  [ERROR] Forward pass failed: {e}")
            import traceback; traceback.print_exc()
            return

    # ── CHECK 8: Logit computation ────────────────────────────────
    sep("CHECK 8: Logits & Predictions")
    with torch.no_grad():
        logit_scale = vit.model.logit_scale.exp()
        print(f"  logit_scale: {logit_scale.item():.4f}")

        logits = logit_scale * img_feat_normed.float() @ tf_maple_normed.float().t()
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits (first 4 samples):")
        for i in range(min(4, logits.shape[0])):
            row = " ".join(f"{logits[i,j]:.3f}" for j in range(logits.shape[1]))
            print(f"    sample {i} (true={labels[i].item()}): [{row}]")

        preds = logits.argmax(dim=-1).cpu()
        print(f"\n  Predictions: {preds.tolist()}")
        print(f"  True labels: {labels.tolist()}")
        correct = (preds == labels).sum().item()
        print(f"  Correct: {correct}/{len(labels)} = {correct/len(labels)*100:.1f}%")

        # Check if predictions are always the same class
        pred_dist = torch.bincount(preds, minlength=len(classnames))
        print(f"\n  Prediction distribution (this batch):")
        for i in range(len(classnames)):
            print(f"    class {i} ('{classnames[i]}'): {pred_dist[i].item()}")

    # ── CHECK 9: Run with hooks (sanity) ──────────────────────────
    sep("CHECK 9: run_with_hooks sanity check")
    print("  Running with empty hooks list to verify run_with_hooks works...")
    with torch.no_grad():
        try:
            out_hooks = vit.run_with_hooks([], return_type="output", **inputs)
            if isinstance(out_hooks, torch.Tensor) and isinstance(out, torch.Tensor):
                diff = (out_hooks - out).abs().max().item()
                print(f"  Diff (hooked vs unhooked): {diff:.6e}")
                if diff < 1e-4:
                    print("  ✓ run_with_hooks consistent with forward pass")
                else:
                    print("  ⚠ WARNING: run_with_hooks gives different results!")
            else:
                print(f"  out_hooks type: {type(out_hooks)}")
        except Exception as e:
            print(f"  [ERROR] run_with_hooks failed: {e}")
            import traceback; traceback.print_exc()

    # ── CHECK 10: Compare with base model ─────────────────────────
    sep("CHECK 10: Base model comparison")
    print("  Loading base model for comparison...")
    try:
        vit_base = load_hooked_vit(ref_cfg, "base", BACKBONE, DEVICE)
        vit_base.eval()

        # Base text features (template-based)
        mean_f = 0
        for tmpl in openai_imagenet_template:
            ids = [vit_base.processor(text=tmpl(c), return_tensors="pt", padding=False,
                                      truncation=True).input_ids[0] for c in classnames]
            padded = pad_sequence(ids, batch_first=True).to(DEVICE)
            f = vit_base.model.get_text_features(padded)
            f = f / f.norm(dim=-1, keepdim=True)
            mean_f += f
        mean_f /= len(openai_imagenet_template)
        tf_base = mean_f / mean_f.norm(dim=-1, keepdim=True)

        # Base image features
        inputs_base = vit_base.processor(
            images=[to_pil(img) for img in images],
            text="", return_tensors="pt", padding=True
        ).to(DEVICE)
        out_base = vit_base(return_type="output", **inputs_base)
        img_feat_base = out_base.image_embeds
        img_feat_base = img_feat_base / img_feat_base.norm(dim=-1, keepdim=True)

        logits_base = vit_base.model.logit_scale.exp() * img_feat_base.float() @ tf_base.float().t()
        preds_base = logits_base.argmax(dim=-1).cpu()
        print(f"  Base predictions: {preds_base.tolist()}")
        print(f"  Base correct: {(preds_base == labels).sum().item()}/{len(labels)}")

        # Compare features
        if isinstance(out, torch.Tensor):
            maple_img = img_feat_normed
        else:
            maple_img = img_feat / img_feat.norm(dim=-1, keepdim=True)

        cosine = (maple_img.float() * img_feat_base.float()).sum(dim=-1)
        print(f"\n  Cosine similarity (maple vs base image features):")
        print(f"    mean={cosine.mean():.4f}, min={cosine.min():.4f}, max={cosine.max():.4f}")
        if cosine.mean() > 0.99:
            print("  ⚠ MaPLe image features ≈ base features (visual prompts may not be active)")

        del vit_base
    except Exception as e:
        print(f"  [ERROR] Base comparison failed: {e}")
        import traceback; traceback.print_exc()

    # ── SUMMARY ───────────────────────────────────────────────────
    sep("DIAGNOSTIC COMPLETE")
    print("  Review the output above for:")
    print("    1. Label mapping mismatches (CHECK 1)")
    print("    2. Zero/random prompt weights in checkpoint (CHECK 2)")
    print("    3. Identical text features across classes (CHECK 6)")
    print("    4. Predictions always same class (CHECK 8)")
    print("    5. MaPLe features ≈ base features (CHECK 10)")


if __name__ == "__main__":
    main()
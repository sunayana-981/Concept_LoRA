#!/usr/bin/env python3
"""Inspect the hooked_vit to understand what we need to monkey-patch."""
import os, sys, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch, glob

try:
    from tasks.utils import load_sae, load_hooked_vit
except ImportError as e:
    print(f"[FATAL] {e}"); sys.exit(1)

BACKBONE = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Need a ref_cfg
paths = sorted(glob.glob("out/checkpoints/medmnist/*/final*/*.pt"))
if not paths:
    paths = ["data/sae_weight/base/out.pt"]
sae, ref_cfg = load_sae(paths[0], DEVICE)
del sae; gc.collect()

vit = load_hooked_vit(ref_cfg, "base", BACKBONE, DEVICE)

print(f"type(vit): {type(vit)}")
print(f"type(vit.model): {type(vit.model)}")
print(f"\nvit attributes:")
for attr in sorted(dir(vit)):
    if not attr.startswith('_'):
        obj = getattr(vit, attr, None)
        if callable(obj):
            print(f"  {attr}() — method")
        else:
            t = type(obj).__name__
            print(f"  {attr} — {t}")

print(f"\nvit.model type: {type(vit.model).__name__}")
print(f"Has text_model: {hasattr(vit.model, 'text_model')}")
print(f"Has vision_model: {hasattr(vit.model, 'vision_model')}")
print(f"Has transformer: {hasattr(vit.model, 'transformer')}")
print(f"Has visual: {hasattr(vit.model, 'visual')}")

# Check what run_with_cache and run_with_hooks expect
print(f"\nrun_with_cache signature:")
import inspect
try:
    sig = inspect.signature(vit.run_with_cache)
    print(f"  {sig}")
except: print("  (could not inspect)")

print(f"\nrun_with_hooks signature:")
try:
    sig = inspect.signature(vit.run_with_hooks)
    print(f"  {sig}")
except: print("  (could not inspect)")

# Check processor
print(f"\ntype(vit.processor): {type(vit.processor)}")

# Check what __call__ returns
print(f"\nTesting vit forward...")
from PIL import Image
import numpy as np
dummy_img = Image.fromarray(np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8))
inputs = vit.processor(images=[dummy_img], text="a photo", return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    out = vit(return_type="output", **inputs)
print(f"type(out): {type(out)}")
print(f"out attributes: {[a for a in dir(out) if not a.startswith('_')]}")
if hasattr(out, 'image_embeds'):
    print(f"out.image_embeds shape: {out.image_embeds.shape}")
if hasattr(out, 'text_embeds'):
    print(f"out.text_embeds shape: {out.text_embeds.shape}")